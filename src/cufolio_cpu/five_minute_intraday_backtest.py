"""Causal five-minute portfolio forecast audit from one-minute stock bars.

This research-only backtest generates a fresh, long-only portfolio every five
minutes of one completed session, then records the exact subsequent five-minute
portfolio return.  It deliberately uses only non-overlapping five-minute labels
from timestamps strictly before each decision.  Therefore, one-minute data
improves the number of available five-minute observations without pretending
that overlapping labels are independent samples.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, time as clock_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .optimize import forecast_mean_variance_weights

NEW_YORK = ZoneInfo("America/New_York")
SESSION_OPEN = clock_time(9, 30)
SESSION_CLOSE = clock_time(16, 0)
FORECAST_INTERVAL_MINUTES = 5
FORECAST_HORIZON_MINUTES = 5
# The paper target loader rejects non-positive weights.  This removes solver
# dust before an audit portfolio is recorded or consumed elsewhere.
MIN_EMITTABLE_TARGET_WEIGHT = 1e-6


@dataclass(frozen=True)
class FiveMinuteForecastBacktestResult:
    """Every causal decision, its holdings, and a reconciliation summary."""

    ledger: pd.DataFrame
    holdings: pd.DataFrame
    summary: dict[str, object]


def _regular_session_minute_closes(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "symbol", "close"}
    if missing := required.difference(bars.columns):
        raise ValueError(f"intraday bars are missing required columns: {sorted(missing)}")
    clean = bars.loc[:, ["timestamp", "symbol", "close"]].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")
    clean["symbol"] = clean["symbol"].astype(str).str.upper().str.strip()
    clean["close"] = pd.to_numeric(clean["close"], errors="coerce")
    clean = clean.dropna(subset=["timestamp", "close"])
    clean = clean[(clean["symbol"] != "") & (clean["close"] > 0)]
    clean = clean.drop_duplicates(["timestamp", "symbol"], keep="last")
    local = clean["timestamp"].dt.tz_convert(NEW_YORK)
    regular = (local.dt.time >= SESSION_OPEN) & (local.dt.time <= SESSION_CLOSE)
    clean = clean.loc[regular].copy()
    if clean.empty:
        raise ValueError("no positive one-minute closes fall in the US regular session")
    clean["session_date"] = local.loc[regular].dt.tz_localize(None).dt.date
    return clean.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _timestamp(session_day: date, at: clock_time) -> pd.Timestamp:
    return pd.Timestamp.combine(session_day, at).tz_localize(NEW_YORK).tz_convert("UTC")


def _decision_grid(session_day: date, *, interval_minutes: int) -> pd.DatetimeIndex:
    start = _timestamp(session_day, SESSION_OPEN)
    final_decision = _timestamp(session_day, SESSION_CLOSE) - timedelta(minutes=interval_minutes)
    return pd.date_range(start, final_decision, freq=f"{interval_minutes}min")


def _non_overlapping_label_panel(
    closes: pd.DataFrame, session_days: list[date], *, interval_minutes: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Return non-overlapping intraday log-return labels and their ends."""
    rows: list[pd.Series] = []
    ends: dict[pd.Timestamp, pd.Timestamp] = {}
    for session_day in session_days:
        for decision_at in _decision_grid(session_day, interval_minutes=interval_minutes):
            target_end = decision_at + timedelta(minutes=interval_minutes)
            start_prices = closes.reindex([decision_at]).iloc[0]
            end_prices = closes.reindex([target_end]).iloc[0]
            row = np.log(end_prices / start_prices)
            row.name = decision_at
            rows.append(row)
            ends[decision_at] = target_end
    if not rows:
        raise ValueError("no complete regular-session intraday label windows are available")
    return pd.DataFrame(rows).sort_index(), pd.Series(ends, name="target_end")


def _sanitize_weights(weights: pd.Series, *, max_weight: float) -> pd.Series:
    """Remove numerical dust while preserving long-only full investment."""
    clean = weights.clip(lower=0.0, upper=max_weight).copy()
    clean.loc[clean < MIN_EMITTABLE_TARGET_WEIGHT] = 0.0
    residual = float(1 - clean.sum())
    while abs(residual) > 1e-12:
        if residual > 0:
            capacity = max_weight - clean
            recipients = capacity[capacity > 1e-12]
            if recipients.empty:
                raise RuntimeError("cannot restore capped portfolio weight feasibility")
            addition = min(residual / len(recipients), float(recipients.min()))
            clean.loc[recipients.index] += addition
        else:
            donors = clean[clean > 1e-12]
            if donors.empty:
                raise RuntimeError("cannot restore non-negative portfolio weight feasibility")
            reduction = min((-residual) / len(donors), float(donors.min()))
            clean.loc[donors.index] -= reduction
        residual = float(1 - clean.sum())
    return clean.loc[clean > 0].copy()


def _pooled_time_of_day_expected_returns(
    training: pd.DataFrame,
    *,
    decision_at: pd.Timestamp,
    seasonality_prior_observations: int,
) -> pd.Series:
    """Forecast each asset by a shrunk same-clock mean over pooled history.

    The global mean uses all earlier non-overlapping five-minute windows.  The
    same-clock mean captures a possible intraday pattern but is shrunk toward
    that global value, avoiding a claim that a small count of prior days is a
    precise 09:30 (or similar) forecast.
    """
    global_mean = training.mean(skipna=True)
    local_clock = decision_at.tz_convert(NEW_YORK).time()
    same_clock_mask = training.index.tz_convert(NEW_YORK).time == local_clock
    same_clock = training.loc[same_clock_mask]
    same_clock_mean = same_clock.mean(skipna=True)
    same_clock_count = same_clock.notna().sum(axis=0)
    shrinkage = same_clock_count / (same_clock_count + seasonality_prior_observations)
    return global_mean + shrinkage * (same_clock_mean - global_mean)


def _select_portfolio(
    labels: pd.DataFrame,
    target_ends: pd.Series,
    closes: pd.DataFrame,
    *,
    decision_at: pd.Timestamp,
    top_n: int,
    max_weight: float,
    risk_aversion: float,
    lookback_observations: int,
    min_training_sessions: int,
    min_covariance_scenarios: int,
    seasonality_prior_observations: int,
) -> tuple[pd.Series | None, pd.Series | None, dict[str, object]]:
    """Select one causal five-minute portfolio and disclose all quality gates."""
    training = labels.loc[target_ends.reindex(labels.index) < decision_at].tail(lookback_observations)
    training_ends = target_ends.reindex(training.index)
    current_prices = closes.reindex([decision_at]).iloc[0].dropna()
    training_days = training.index.tz_convert(NEW_YORK).date
    required_candidates = int(np.ceil((1 - 1e-12) / max_weight))
    diagnostic: dict[str, object] = {
        "forecast_status": "unavailable",
        "decision_timestamp": decision_at.isoformat(),
        "training_observations": int(len(training)),
        "training_sessions": int(pd.Index(training_days).nunique()),
        "input_assets": int(len(training.columns)),
        "assets_with_decision_price": int(len(current_prices)),
        "required_candidates_for_weight_cap": required_candidates,
        "minimum_training_sessions": min_training_sessions,
        "minimum_covariance_scenarios": min_covariance_scenarios,
        "training_end": training_ends.max().isoformat() if not training_ends.empty else None,
    }
    if diagnostic["training_sessions"] < min_training_sessions:
        diagnostic["reason"] = "insufficient_prior_sessions"
        return None, None, diagnostic
    expected = _pooled_time_of_day_expected_returns(
        training,
        decision_at=decision_at,
        seasonality_prior_observations=seasonality_prior_observations,
    )
    eligible = expected.index[
        (training.notna().sum(axis=0) >= min_covariance_scenarios)
        & expected.notna()
        & expected.index.isin(current_prices.index)
    ]
    diagnostic["eligible_assets"] = int(len(eligible))
    if len(eligible) < required_candidates:
        diagnostic["reason"] = "insufficient_assets_with_complete_training_and_decision_prices"
        return None, None, diagnostic
    candidates: list[str] = []
    common_rows = pd.Series(True, index=training.index)
    for symbol in expected.reindex(eligible).sort_values(ascending=False).index:
        with_symbol = common_rows & training[symbol].notna()
        if int(with_symbol.sum()) < min_covariance_scenarios:
            continue
        candidates.append(symbol)
        common_rows = with_symbol
        if len(candidates) == top_n:
            break
    scenarios = training.loc[common_rows, candidates]
    diagnostic.update(
        {
            "candidate_count": int(len(candidates)),
            "covariance_scenarios": int(len(scenarios)),
            "candidate_weight_capacity": float(max_weight * len(candidates)),
        }
    )
    if len(candidates) < required_candidates:
        diagnostic["reason"] = "insufficient_complete_candidates_for_weight_cap"
        return None, None, diagnostic
    if len(scenarios) < min_covariance_scenarios:
        diagnostic["reason"] = "insufficient_complete_covariance_scenarios"
        return None, None, diagnostic
    allocation = forecast_mean_variance_weights(
        scenarios,
        expected.reindex(candidates),
        risk_aversion=risk_aversion,
        max_weight=max_weight,
    )
    weights = _sanitize_weights(allocation.weights, max_weight=max_weight)
    diagnostic.update(
        {
            "forecast_status": "ok",
            "reason": "ok",
            "optimizer_status": allocation.status,
            "expected_portfolio_log_return": allocation.expected_return,
            "expected_portfolio_simple_return": float(np.expm1(allocation.expected_return)),
        }
    )
    return weights, expected.reindex(weights.index), diagnostic


def _empty_holdings() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "decision_timestamp",
            "target_end",
            "symbol",
            "target_weight",
            "predicted_asset_log_return",
            "realized_asset_log_return",
            "realized_asset_simple_return",
        ]
    )


def _summary(
    ledger: pd.DataFrame,
    *,
    requested_session_date: str | None,
    evaluated_session_date: date,
    interval_minutes: int,
) -> dict[str, object]:
    forecasted = ledger[ledger["forecast_status"].eq("ok")].copy()
    realized = forecasted[forecasted["realized_status"].eq("ok")].copy()
    expected = realized["expected_portfolio_simple_return"].astype(float)
    actual = realized["actual_portfolio_simple_return"].astype(float)
    errors = actual - expected
    expected_log = realized["expected_portfolio_log_return"].astype(float)
    actual_log = realized["actual_portfolio_log_return"].astype(float)
    return {
        "research_only": True,
        "bar_interval": "1m",
        "forecast_interval_minutes": interval_minutes,
        "forecast_horizon_minutes": interval_minutes,
        "requested_session_date": requested_session_date,
        "evaluated_session_date": evaluated_session_date.isoformat(),
        "decision_windows": int(len(ledger)),
        "forecast_windows": int(len(forecasted)),
        "realized_windows": int(len(realized)),
        "unrealized_or_incomplete_windows": int(len(forecasted) - len(realized)),
        "sum_expected_simple_return": float(expected.sum()) if not expected.empty else None,
        "sum_actual_simple_return": float(actual.sum()) if not actual.empty else None,
        "sum_expected_log_return": float(expected_log.sum()) if not expected_log.empty else None,
        "sum_actual_log_return": float(actual_log.sum()) if not actual_log.empty else None,
        "compounded_expected_return": float(np.expm1(expected_log.sum())) if not expected_log.empty else None,
        "compounded_actual_return": float(np.expm1(actual_log.sum())) if not actual_log.empty else None,
        "arithmetic_sum_forecast_error": float(errors.sum()) if not errors.empty else None,
        "compounded_forecast_error": (
            float(np.expm1(actual_log.sum()) - np.expm1(expected_log.sum())) if not actual_log.empty else None
        ),
        "mean_absolute_error_bps": float(errors.abs().mean() * 10_000) if not errors.empty else None,
        "root_mean_squared_error_bps": float(np.sqrt(np.mean(errors**2)) * 10_000) if not errors.empty else None,
        "directional_accuracy": (
            float((np.sign(expected) == np.sign(actual)).mean()) if not expected.empty else None
        ),
        "reconciliation_rule": (
            "Five-minute simple returns are shown as arithmetic sums for the requested comparison; "
            "the economically correct session total compounds each realized portfolio return, equivalently sums log returns."
        ),
        "causality_rule": (
            "Every forecast uses only five-minute labels whose target endpoint is strictly before its decision timestamp; "
            "a missing realized endpoint marks that forecast incomplete rather than changing the historical weights."
        ),
    }


def _render_markdown_report(result: FiveMinuteForecastBacktestResult) -> str:
    summary = result.summary
    interval_minutes = int(summary["forecast_interval_minutes"])
    def percent(value: object) -> str:
        return "n/a" if value is None or pd.isna(value) else f"{float(value):+.4%}"

    lines = [
        f"# Causal {interval_minutes}-minute forecast audit",
        "",
        f"- Evaluated session: `{summary['evaluated_session_date']}`",
        f"- Decision windows: {summary['decision_windows']}; forecasts: {summary['forecast_windows']}; exact realized windows: {summary['realized_windows']}",
        f"- Forecast model: {summary['forecast_model']}",
        "- This is research-only: it creates no orders and does not claim that an expected return must equal a realized return.",
        "",
        "## Reconciliation",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Arithmetic sum of expected {interval_minutes}-minute returns | {percent(summary['sum_expected_simple_return'])} |",
        f"| Arithmetic sum of actual {interval_minutes}-minute returns | {percent(summary['sum_actual_simple_return'])} |",
        f"| Compounded expected return | {percent(summary['compounded_expected_return'])} |",
        f"| Compounded actual return | {percent(summary['compounded_actual_return'])} |",
        f"| Compounded forecast error | {percent(summary['compounded_forecast_error'])} |",
        f"| Mean absolute forecast error | {summary['mean_absolute_error_bps']:.2f} bps" if summary["mean_absolute_error_bps"] is not None else "| Mean absolute forecast error | n/a |",
        f"| Directional accuracy | {percent(summary['directional_accuracy'])} |",
        "",
        f"The arithmetic sums answer the requested sum comparison. Compounded returns are the correct session-level wealth comparison because the {interval_minutes}-minute windows are sequential.",
        "",
        f"## Every {interval_minutes}-minute decision",
        "",
        "| Decision (New York) | Forecast | Realized | Expected | Actual | Error | Holdings | Reason |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result.ledger.itertuples(index=False):
        decision = pd.Timestamp(row.decision_timestamp).tz_convert(NEW_YORK).strftime("%H:%M")
        expected = percent(row.expected_portfolio_simple_return)
        actual = percent(row.actual_portfolio_simple_return)
        error = (
            percent(row.actual_portfolio_simple_return - row.expected_portfolio_simple_return)
            if pd.notna(row.actual_portfolio_simple_return) and pd.notna(row.expected_portfolio_simple_return)
            else "n/a"
        )
        lines.append(
            f"| {decision} | {row.forecast_status} | {row.realized_status} | {expected} | {actual} | {error} | {row.holding_count} | {row.reason} |"
        )
    return "\n".join(lines) + "\n"


def _write_markdown_report(path: Path, result: FiveMinuteForecastBacktestResult) -> None:
    path.write_text(_render_markdown_report(result), encoding="utf-8")


def run_intraday_forecast_backtest(
    minute_bars: pd.DataFrame,
    *,
    interval_minutes: int,
    session_date: str | date | None = None,
    top_n: int = 20,
    max_weight: float = 0.10,
    risk_aversion: float = 10.0,
    lookback_observations: int = 780,
    min_training_sessions: int = 10,
    min_covariance_scenarios: int = 500,
    seasonality_prior_observations: int = 20,
) -> FiveMinuteForecastBacktestResult:
    """Audit every non-overlapping intraday forecast in one completed session.

    ``lookback_observations`` counts non-overlapping interval labels, not raw
    minute bars. This is intentionally distinct from the hourly runner's
    120-scenario / twenty-session policy.
    """
    if interval_minutes not in {5, 15}:
        raise ValueError("interval_minutes must be 5 or 15")
    if top_n < 2:
        raise ValueError("top_n must be at least two")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")
    if max_weight * top_n < 1 - 1e-12:
        raise ValueError("top_n is too small for a fully invested portfolio at the requested max_weight")
    if lookback_observations < min_covariance_scenarios:
        raise ValueError("lookback_observations must be at least min_covariance_scenarios")
    if min_training_sessions < 2:
        raise ValueError("min_training_sessions must be at least two")
    if min_covariance_scenarios < top_n:
        raise ValueError("min_covariance_scenarios must be at least top_n")
    if seasonality_prior_observations < 1:
        raise ValueError("seasonality_prior_observations must be positive")

    clean = _regular_session_minute_closes(minute_bars)
    available_days = sorted(set(clean["session_date"]))
    requested = date.fromisoformat(str(session_date)) if session_date is not None else None
    eligible_days = [day for day in available_days if requested is None or day <= requested]
    if not eligible_days:
        raise ValueError("the requested session date is not present in the supplied regular-session bars")
    evaluated_day = eligible_days[-1]
    closes = clean.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    labels, target_ends = _non_overlapping_label_panel(
        closes, available_days, interval_minutes=interval_minutes
    )

    ledger_rows: list[dict[str, object]] = []
    holdings_rows: list[dict[str, object]] = []
    for decision_at in _decision_grid(evaluated_day, interval_minutes=interval_minutes):
        target_end = decision_at + timedelta(minutes=interval_minutes)
        weights, expected_by_asset, diagnostic = _select_portfolio(
            labels,
            target_ends,
            closes,
            decision_at=decision_at,
            top_n=top_n,
            max_weight=max_weight,
            risk_aversion=risk_aversion,
            lookback_observations=lookback_observations,
            min_training_sessions=min_training_sessions,
            min_covariance_scenarios=min_covariance_scenarios,
            seasonality_prior_observations=seasonality_prior_observations,
        )
        row: dict[str, object] = {
            "decision_timestamp": decision_at,
            "target_end": target_end,
            **diagnostic,
            "holding_count": 0,
            "realized_status": "not_forecast",
            "actual_portfolio_log_return": np.nan,
            "actual_portfolio_simple_return": np.nan,
        }
        if weights is not None and expected_by_asset is not None:
            row["holding_count"] = int(len(weights))
            start_prices = closes.reindex([decision_at], columns=weights.index).iloc[0]
            end_prices = closes.reindex([target_end], columns=weights.index).iloc[0]
            realized_simple = end_prices / start_prices - 1
            if realized_simple.isna().any() or not np.isfinite(realized_simple.to_numpy(dtype=float)).all():
                row["realized_status"] = "missing_exact_realized_endpoint"
            else:
                portfolio_simple = float(weights.dot(realized_simple))
                row["realized_status"] = "ok"
                row["actual_portfolio_simple_return"] = portfolio_simple
                row["actual_portfolio_log_return"] = float(np.log1p(portfolio_simple))
            realized_log = np.log1p(realized_simple)
            for symbol, weight in weights.items():
                holdings_rows.append(
                    {
                        "decision_timestamp": decision_at,
                        "target_end": target_end,
                        "symbol": symbol,
                        "target_weight": float(weight),
                        "predicted_asset_log_return": float(expected_by_asset[symbol]),
                        "realized_asset_log_return": float(realized_log[symbol]) if pd.notna(realized_log[symbol]) else np.nan,
                        "realized_asset_simple_return": float(realized_simple[symbol]) if pd.notna(realized_simple[symbol]) else np.nan,
                    }
                )
        ledger_rows.append(row)
    ledger = pd.DataFrame(ledger_rows)
    holdings = pd.DataFrame(holdings_rows) if holdings_rows else _empty_holdings()
    summary = _summary(
        ledger,
        requested_session_date=requested.isoformat() if requested else None,
        evaluated_session_date=evaluated_day,
        interval_minutes=interval_minutes,
    )
    summary.update(
        {
            "forecast_model": (
                "pooled non-overlapping intraday log-return mean with a same-clock-time mean "
                "shrunk toward the pooled mean; covariance is estimated only from complete prior labels"
            ),
            "top_n": top_n,
            "max_weight": max_weight,
            "risk_aversion": risk_aversion,
            "lookback_observations": lookback_observations,
            "minimum_training_sessions": min_training_sessions,
            "minimum_covariance_scenarios": min_covariance_scenarios,
            "seasonality_prior_observations": seasonality_prior_observations,
        }
    )
    return FiveMinuteForecastBacktestResult(ledger=ledger, holdings=holdings, summary=summary)


def run_five_minute_forecast_backtest(
    minute_bars: pd.DataFrame,
    **kwargs: object,
) -> FiveMinuteForecastBacktestResult:
    """Backward-compatible five-minute specialization of the generic audit."""
    return run_intraday_forecast_backtest(minute_bars, interval_minutes=5, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a causal research-only five-minute intraday forecast audit.")
    parser.add_argument("--input", required=True, help="CSV or CSV.GZ with timestamp,symbol,close one-minute bars")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--session-date", help="YYYY-MM-DD; defaults to the latest complete session in the input")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-weight", type=float, default=0.10)
    parser.add_argument("--risk-aversion", type=float, default=10.0)
    parser.add_argument("--lookback-observations", type=int, default=780)
    parser.add_argument("--min-training-sessions", type=int, default=10)
    parser.add_argument("--min-covariance-scenarios", type=int, default=500)
    parser.add_argument("--seasonality-prior-observations", type=int, default=20)
    args = parser.parse_args()
    result = run_intraday_forecast_backtest(
        pd.read_csv(args.input),
        interval_minutes=FORECAST_INTERVAL_MINUTES,
        session_date=args.session_date,
        top_n=args.top_n,
        max_weight=args.max_weight,
        risk_aversion=args.risk_aversion,
        lookback_observations=args.lookback_observations,
        min_training_sessions=args.min_training_sessions,
        min_covariance_scenarios=args.min_covariance_scenarios,
        seasonality_prior_observations=args.seasonality_prior_observations,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.ledger.to_csv(output / "five_minute_forecast_ledger.csv", index=False)
    result.holdings.to_csv(output / "five_minute_forecast_holdings.csv", index=False)
    (output / "five_minute_forecast_summary.json").write_text(json.dumps(result.summary, indent=2) + "\n", encoding="utf-8")
    _write_markdown_report(output / "five_minute_forecast_report.md", result)
    print(
        f"Wrote {len(result.ledger)} five-minute decision rows and {len(result.holdings)} holdings rows "
        f"for {result.summary['evaluated_session_date']}"
    )


if __name__ == "__main__":
    main()
