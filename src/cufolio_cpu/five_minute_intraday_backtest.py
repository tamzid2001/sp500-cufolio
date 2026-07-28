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
MINUTE_BAR_WIDTH = timedelta(minutes=1)
# The paper target loader rejects non-positive weights.  This removes solver
# dust before an audit portfolio is recorded or consumed elsewhere.
MIN_EMITTABLE_TARGET_WEIGHT = 1e-6


@dataclass(frozen=True)
class FiveMinuteForecastBacktestResult:
    """Every causal decision, its holdings, and a reconciliation summary."""

    ledger: pd.DataFrame
    holdings: pd.DataFrame
    summary: dict[str, object]


@dataclass(frozen=True)
class IntradayForecastCandidate:
    """One causal 5- or 15-minute target available for an execution runner."""

    weights: pd.DataFrame
    status: dict[str, object]


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
    # Alpaca labels a minute bar with the *left* side of the interval.  The
    # regular session therefore contains 09:30 through 15:59, not a synthetic
    # 16:00 bar.  See the Market Data FAQ linked in the README.
    regular = (local.dt.time >= SESSION_OPEN) & (local.dt.time < SESSION_CLOSE)
    clean = clean.loc[regular].copy()
    if clean.empty:
        raise ValueError("no positive one-minute closes fall in the US regular session")
    clean["session_date"] = local.loc[regular].dt.tz_localize(None).dt.date
    return clean.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _timestamp(session_day: date, at: clock_time) -> pd.Timestamp:
    return pd.Timestamp.combine(session_day, at).tz_localize(NEW_YORK).tz_convert("UTC")


def _decision_grid(session_day: date, *, interval_minutes: int) -> pd.DatetimeIndex:
    # A decision at 09:30 cannot causally use the 09:30 one-minute close: that
    # close is only known just after 09:31.  Start after the first complete
    # interval, and use the prior bar's close for every decision below.
    start = _timestamp(session_day, SESSION_OPEN) + timedelta(minutes=interval_minutes)
    final_decision = _timestamp(session_day, SESSION_CLOSE) - timedelta(minutes=interval_minutes)
    return pd.date_range(start, final_decision, freq=f"{interval_minutes}min")


def _close_timestamp_known_at(boundary: pd.Timestamp) -> pd.Timestamp:
    """Return the left-labeled bar whose close is known at ``boundary``."""
    return boundary - MINUTE_BAR_WIDTH


def _as_utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _non_overlapping_label_panel(
    closes: pd.DataFrame, session_days: list[date], *, interval_minutes: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Return non-overlapping intraday log-return labels and their ends."""
    rows: list[pd.Series] = []
    ends: dict[pd.Timestamp, pd.Timestamp] = {}
    for session_day in session_days:
        for decision_at in _decision_grid(session_day, interval_minutes=interval_minutes):
            target_end = decision_at + timedelta(minutes=interval_minutes)
            start_prices = closes.reindex([_close_timestamp_known_at(decision_at)]).iloc[0]
            end_prices = closes.reindex([_close_timestamp_known_at(target_end)]).iloc[0]
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
    # A left-labelled 09:35 bar completes at 09:36, so a 09:35 decision can
    # only use the prior bar's close.  This must match the realized start
    # endpoint and the live runner's completed-minute cache.
    current_prices = closes.reindex([_close_timestamp_known_at(decision_at)]).iloc[0].dropna()
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


def generate_intraday_forecast_candidate(
    minute_bars: pd.DataFrame,
    *,
    decision_at: str | pd.Timestamp,
    interval_minutes: int = FORECAST_INTERVAL_MINUTES,
    top_n: int = 20,
    max_weight: float = 0.10,
    risk_aversion: float = 10.0,
    lookback_observations: int = 780,
    min_training_sessions: int = 10,
    min_covariance_scenarios: int = 500,
    seasonality_prior_observations: int = 20,
) -> IntradayForecastCandidate:
    """Generate the single completed-bar forecast used by the live runner.

    This is deliberately the same selection function as the audit.  Labels
    are limited to targets completed strictly before ``decision_at`` and the
    eligibility price is the preceding, fully completed one-minute close.
    The supplied frame may include later rows (as it does in a historical
    test); they cannot influence this candidate.
    """
    if interval_minutes not in {5, 15}:
        raise ValueError("interval_minutes must be 5 or 15")
    decision = _as_utc_timestamp(decision_at)
    clean = _regular_session_minute_closes(minute_bars)
    closes = clean.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    session_day = decision.tz_convert(NEW_YORK).date()
    if session_day not in set(clean["session_date"]):
        raise ValueError("decision_at session is not present in the supplied regular-session bars")
    valid_decisions = _decision_grid(session_day, interval_minutes=interval_minutes)
    if decision not in valid_decisions:
        raise ValueError(
            f"decision_at must be one of the causal {interval_minutes}-minute session boundaries"
        )
    available_days = sorted(set(clean["session_date"]))
    labels, target_ends = _non_overlapping_label_panel(
        closes, available_days, interval_minutes=interval_minutes,
    )
    weights, expected_by_asset, diagnostic = _select_portfolio(
        labels,
        target_ends,
        closes,
        decision_at=decision,
        top_n=top_n,
        max_weight=max_weight,
        risk_aversion=risk_aversion,
        lookback_observations=lookback_observations,
        min_training_sessions=min_training_sessions,
        min_covariance_scenarios=min_covariance_scenarios,
        seasonality_prior_observations=seasonality_prior_observations,
    )
    status: dict[str, object] = {
        **diagnostic,
        "weights_generated": weights is not None and expected_by_asset is not None,
        "decision_timestamp": decision.isoformat(),
        "decision_price_timestamp": _close_timestamp_known_at(decision).isoformat(),
        "target_end": (decision + timedelta(minutes=interval_minutes)).isoformat(),
        "forecast_interval_minutes": interval_minutes,
        "forecast_horizon_minutes": interval_minutes,
        "causality_rule": (
            "training target_end is strictly earlier than the decision and the decision price is "
            "the preceding completed one-minute close"
        ),
    }
    if weights is None or expected_by_asset is None:
        return IntradayForecastCandidate(
            pd.DataFrame(columns=["symbol", "target_weight", "predicted_asset_log_return"]),
            status,
        )
    candidates = pd.DataFrame(
        {
            "symbol": weights.index,
            "target_weight": weights.to_numpy(dtype=float),
            "predicted_asset_log_return": expected_by_asset.reindex(weights.index).to_numpy(dtype=float),
        }
    )
    return IntradayForecastCandidate(candidates, status)


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
            "realized_price_source",
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
    all_expected = forecasted["expected_portfolio_simple_return"].astype(float)
    all_expected_log = forecasted["expected_portfolio_log_return"].astype(float)
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
        "exact_realized_coverage": float(len(realized) / len(forecasted)) if len(forecasted) else None,
        "full_session_exact_realization_available": bool(len(realized) == len(forecasted) and len(forecasted) > 0),
        "all_forecast_sum_expected_simple_return": float(all_expected.sum()) if not all_expected.empty else None,
        "all_forecast_sum_expected_log_return": float(all_expected_log.sum()) if not all_expected_log.empty else None,
        "all_forecast_compounded_expected_return": (
            float(np.expm1(all_expected_log.sum())) if not all_expected_log.empty else None
        ),
        "matched_sum_expected_simple_return": float(expected.sum()) if not expected.empty else None,
        "matched_sum_actual_simple_return": float(actual.sum()) if not actual.empty else None,
        "matched_sum_expected_log_return": float(expected_log.sum()) if not expected_log.empty else None,
        "matched_sum_actual_log_return": float(actual_log.sum()) if not actual_log.empty else None,
        "matched_compounded_expected_return": (
            float(np.expm1(expected_log.sum())) if not expected_log.empty else None
        ),
        "matched_compounded_actual_return": (
            float(np.expm1(actual_log.sum())) if not actual_log.empty else None
        ),
        "matched_arithmetic_sum_forecast_error": float(errors.sum()) if not errors.empty else None,
        "matched_compounded_forecast_error": (
            float(np.expm1(actual_log.sum()) - np.expm1(expected_log.sum())) if not actual_log.empty else None
        ),
        # Retained for compatibility.  These legacy names now deliberately
        # refer to the same matched windows as the actual-return field.
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
            "Expected and actual returns are compared only across the same windows with exact realized endpoints. "
            "All-forecast expected return is reported separately. A compounded actual return is a full-session "
            "wealth result only when every forecast window has an exact realization."
        ),
        "causality_rule": (
            "Alpaca minute bars are left-labeled. Each decision uses the prior minute's completed close, and every "
            "forecast uses only labels whose interval endpoint is strictly before its decision timestamp. A missing "
            "realized endpoint marks that forecast incomplete rather than changing historical weights."
        ),
    }


def _recompute_realized_portfolios(ledger: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    """Rebuild realized portfolio returns from the recorded, fixed holdings."""
    refreshed = ledger.copy()
    if "realized_price_source" not in holdings.columns:
        holdings = holdings.copy()
        holdings["realized_price_source"] = np.where(
            holdings["realized_asset_simple_return"].notna(), "primary_input", "missing"
        )
    for index, row in refreshed.iterrows():
        if row["forecast_status"] != "ok":
            continue
        decision_at = _as_utc_timestamp(row["decision_timestamp"])
        selected = holdings[pd.to_datetime(holdings["decision_timestamp"], utc=True) == decision_at]
        returns = pd.to_numeric(selected["realized_asset_simple_return"], errors="coerce")
        if len(selected) != int(row["holding_count"]) or returns.isna().any() or not np.isfinite(returns).all():
            refreshed.at[index, "realized_status"] = "missing_exact_realized_endpoint"
            refreshed.at[index, "actual_portfolio_simple_return"] = np.nan
            refreshed.at[index, "actual_portfolio_log_return"] = np.nan
            refreshed.at[index, "realized_data_source"] = "incomplete"
            continue
        weights = pd.to_numeric(selected["target_weight"], errors="coerce")
        portfolio_simple = float(weights.dot(returns))
        source_names = sorted(set(selected["realized_price_source"].dropna()) - {"missing"})
        refreshed.at[index, "realized_status"] = "ok"
        refreshed.at[index, "actual_portfolio_simple_return"] = portfolio_simple
        refreshed.at[index, "actual_portfolio_log_return"] = float(np.log1p(portfolio_simple))
        refreshed.at[index, "realized_data_source"] = "+".join(source_names) if source_names else "primary_input"
    return refreshed


def reconcile_intraday_forecast_outcomes(
    result: FiveMinuteForecastBacktestResult,
    fallback_minute_bars: pd.DataFrame,
    *,
    fallback_source: str = "yfinance_1m_fallback",
) -> FiveMinuteForecastBacktestResult:
    """Fill only missing realized asset endpoints without changing a forecast.

    The forecasted symbols, weights, expected returns, and primary-input
    realizations remain fixed.  A fallback is used only when both endpoint
    closes for a previously missing holding are available from that fallback.
    That preserves the original causal decision and makes every repaired price
    source explicit in the holdings artifact.
    """
    holdings = result.holdings.copy()
    if holdings.empty or fallback_minute_bars.empty:
        refreshed_ledger = _recompute_realized_portfolios(result.ledger, holdings)
        refreshed_summary = _summary(
            refreshed_ledger,
            requested_session_date=result.summary.get("requested_session_date"),
            evaluated_session_date=date.fromisoformat(str(result.summary["evaluated_session_date"])),
            interval_minutes=int(result.summary["forecast_interval_minutes"]),
        )
        refreshed_summary.update({key: value for key, value in result.summary.items() if key not in refreshed_summary})
        refreshed_summary.update(
            {
                "outcome_fallback_source": fallback_source,
                "fallback_repaired_asset_endpoints": 0,
                "fallback_available": False,
            }
        )
        return FiveMinuteForecastBacktestResult(refreshed_ledger, holdings, refreshed_summary)

    if "realized_price_source" not in holdings.columns:
        holdings["realized_price_source"] = np.where(
            holdings["realized_asset_simple_return"].notna(), "primary_input", "missing"
        )
    try:
        fallback_clean = _regular_session_minute_closes(fallback_minute_bars)
    except ValueError:
        return reconcile_intraday_forecast_outcomes(
            result,
            pd.DataFrame(columns=["timestamp", "symbol", "close"]),
            fallback_source=fallback_source,
        )
    fallback_closes = fallback_clean.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    repaired = 0
    for index, holding in holdings[holdings["realized_asset_simple_return"].isna()].iterrows():
        decision_at = _as_utc_timestamp(holding["decision_timestamp"])
        target_end = _as_utc_timestamp(holding["target_end"])
        symbol = str(holding["symbol"])
        if symbol not in fallback_closes.columns:
            continue
        start = fallback_closes.at[_close_timestamp_known_at(decision_at), symbol] if _close_timestamp_known_at(decision_at) in fallback_closes.index else np.nan
        end = fallback_closes.at[_close_timestamp_known_at(target_end), symbol] if _close_timestamp_known_at(target_end) in fallback_closes.index else np.nan
        if pd.isna(start) or pd.isna(end) or not np.isfinite(start) or not np.isfinite(end) or start <= 0 or end <= 0:
            continue
        simple_return = float(end / start - 1)
        holdings.at[index, "realized_asset_simple_return"] = simple_return
        holdings.at[index, "realized_asset_log_return"] = float(np.log1p(simple_return))
        holdings.at[index, "realized_price_source"] = fallback_source
        repaired += 1
    refreshed_ledger = _recompute_realized_portfolios(result.ledger, holdings)
    refreshed_summary = _summary(
        refreshed_ledger,
        requested_session_date=result.summary.get("requested_session_date"),
        evaluated_session_date=date.fromisoformat(str(result.summary["evaluated_session_date"])),
        interval_minutes=int(result.summary["forecast_interval_minutes"]),
    )
    refreshed_summary.update({key: value for key, value in result.summary.items() if key not in refreshed_summary})
    refreshed_summary.update(
        {
            "outcome_fallback_source": fallback_source,
            "fallback_repaired_asset_endpoints": repaired,
            "fallback_available": True,
            "realized_price_source_counts": holdings["realized_price_source"].value_counts(dropna=False).to_dict(),
        }
    )
    return FiveMinuteForecastBacktestResult(refreshed_ledger, holdings, refreshed_summary)


def _render_markdown_report(result: FiveMinuteForecastBacktestResult) -> str:
    summary = result.summary
    interval_minutes = int(summary["forecast_interval_minutes"])
    def percent(value: object) -> str:
        return "n/a" if value is None or pd.isna(value) else f"{float(value):+.4%}"

    if summary.get("evaluation_start") and summary.get("evaluation_end"):
        evaluated_label = f"`{summary['evaluation_start']}` through `{summary['evaluation_end']}` ({summary.get('evaluation_sessions', 'n/a')} sessions with bars)"
    else:
        evaluated_label = f"`{summary['evaluated_session_date']}`"
    lines = [
        f"# Causal {interval_minutes}-minute forecast audit",
        "",
        f"- Evaluation: {evaluated_label}",
        f"- Decision windows: {summary['decision_windows']}; forecasts: {summary['forecast_windows']}; exact realized windows: {summary['realized_windows']}",
        f"- Forecast model: {summary['forecast_model']}",
        "- This is research-only: it creates no orders and does not claim that an expected return must equal a realized return.",
        "",
        "## Reconciliation",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Exact realized coverage | {percent(summary['exact_realized_coverage'])} ({summary['realized_windows']}/{summary['forecast_windows']} windows) |",
        f"| All-forecast arithmetic expected return | {percent(summary['all_forecast_sum_expected_simple_return'])} |",
        f"| All-forecast compounded expected return | {percent(summary['all_forecast_compounded_expected_return'])} |",
        f"| Matched-window arithmetic expected return | {percent(summary['matched_sum_expected_simple_return'])} |",
        f"| Matched-window arithmetic actual return | {percent(summary['matched_sum_actual_simple_return'])} |",
        f"| Matched-window compounded expected return | {percent(summary['matched_compounded_expected_return'])} |",
        f"| Matched-window compounded actual return | {percent(summary['matched_compounded_actual_return'])} |",
        f"| Matched-window compounded forecast error | {percent(summary['matched_compounded_forecast_error'])} |",
        f"| Mean absolute forecast error | {summary['mean_absolute_error_bps']:.2f} bps" if summary["mean_absolute_error_bps"] is not None else "| Mean absolute forecast error | n/a |",
        f"| Directional accuracy | {percent(summary['directional_accuracy'])} |",
        "",
        "The requested expected-versus-actual arithmetic comparison uses only the exact realized windows shown above. "
        f"The compounded matched result is a full-session wealth return only if all {summary['forecast_windows']} windows are realized; otherwise it is explicitly a partial, non-contiguous diagnostic.",
        "",
        f"## Every {interval_minutes}-minute decision",
        "",
        "| Decision (New York) | Forecast | Realized | Expected | Actual | Error | Holdings | Reason |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result.ledger.itertuples(index=False):
        decision = pd.Timestamp(row.decision_timestamp).tz_convert(NEW_YORK).strftime("%Y-%m-%d %H:%M")
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
    if summary.get("fallback_available"):
        lines.extend(
            [
                "",
                "## Outcome data repair",
                "",
                f"- Fallback source: `{summary['outcome_fallback_source']}`",
                f"- Repaired missing asset endpoint pairs: {summary['fallback_repaired_asset_endpoints']}",
                "- The repair does not alter forecasts, selected symbols, target weights, or primary-input realized prices.",
            ]
        )
    return "\n".join(lines) + "\n"


def _write_markdown_report(path: Path, result: FiveMinuteForecastBacktestResult) -> None:
    path.write_text(_render_markdown_report(result), encoding="utf-8")


def _validate_intraday_backtest_parameters(
    *,
    interval_minutes: int,
    top_n: int,
    max_weight: float,
    lookback_observations: int,
    min_training_sessions: int,
    min_covariance_scenarios: int,
    seasonality_prior_observations: int,
) -> None:
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


def run_intraday_forecast_backtest_range(
    minute_bars: pd.DataFrame,
    *,
    interval_minutes: int,
    evaluation_start: str | date,
    evaluation_end: str | date,
    top_n: int = 20,
    max_weight: float = 0.10,
    risk_aversion: float = 10.0,
    lookback_observations: int = 780,
    min_training_sessions: int = 10,
    min_covariance_scenarios: int = 500,
    seasonality_prior_observations: int = 20,
) -> FiveMinuteForecastBacktestResult:
    """Audit every causal forecast over an inclusive range of sessions.

    The minute panel and non-overlapping label matrix are built once, then
    each session's decisions consume only labels completed before that exact
    decision.  This permits a multi-month audit without rebuilding the raw
    one-minute panel for every session.
    """
    _validate_intraday_backtest_parameters(
        interval_minutes=interval_minutes,
        top_n=top_n,
        max_weight=max_weight,
        lookback_observations=lookback_observations,
        min_training_sessions=min_training_sessions,
        min_covariance_scenarios=min_covariance_scenarios,
        seasonality_prior_observations=seasonality_prior_observations,
    )
    first = date.fromisoformat(str(evaluation_start))
    last = date.fromisoformat(str(evaluation_end))
    if first > last:
        raise ValueError("evaluation_start must not be after evaluation_end")

    clean = _regular_session_minute_closes(minute_bars)
    available_days = sorted(set(clean["session_date"]))
    evaluated_days = [day for day in available_days if first <= day <= last]
    if not evaluated_days:
        raise ValueError("the requested evaluation range has no regular sessions in the supplied bars")
    closes = clean.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    labels, target_ends = _non_overlapping_label_panel(
        closes, available_days, interval_minutes=interval_minutes
    )

    ledger_rows: list[dict[str, object]] = []
    holdings_rows: list[dict[str, object]] = []
    for evaluated_day in evaluated_days:
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
                "session_date": evaluated_day.isoformat(),
                "decision_timestamp": decision_at,
                "target_end": target_end,
                "decision_price_timestamp": _close_timestamp_known_at(decision_at),
                "realized_price_timestamp": _close_timestamp_known_at(target_end),
                **diagnostic,
                "holding_count": 0,
                "realized_status": "not_forecast",
                "actual_portfolio_log_return": np.nan,
                "actual_portfolio_simple_return": np.nan,
                "realized_data_source": "not_forecast",
            }
            if weights is not None and expected_by_asset is not None:
                row["holding_count"] = int(len(weights))
                start_prices = closes.reindex([_close_timestamp_known_at(decision_at)], columns=weights.index).iloc[0]
                end_prices = closes.reindex([_close_timestamp_known_at(target_end)], columns=weights.index).iloc[0]
                realized_simple = end_prices / start_prices - 1
                if realized_simple.isna().any() or not np.isfinite(realized_simple.to_numpy(dtype=float)).all():
                    row["realized_status"] = "missing_exact_realized_endpoint"
                    row["realized_data_source"] = "incomplete"
                else:
                    portfolio_simple = float(weights.dot(realized_simple))
                    row["realized_status"] = "ok"
                    row["actual_portfolio_simple_return"] = portfolio_simple
                    row["actual_portfolio_log_return"] = float(np.log1p(portfolio_simple))
                    row["realized_data_source"] = "primary_input"
                realized_log = np.log1p(realized_simple)
                for symbol, weight in weights.items():
                    holdings_rows.append(
                        {
                            "session_date": evaluated_day.isoformat(),
                            "decision_timestamp": decision_at,
                            "target_end": target_end,
                            "symbol": symbol,
                            "target_weight": float(weight),
                            "predicted_asset_log_return": float(expected_by_asset[symbol]),
                            "realized_asset_log_return": float(realized_log[symbol]) if pd.notna(realized_log[symbol]) else np.nan,
                            "realized_asset_simple_return": float(realized_simple[symbol]) if pd.notna(realized_simple[symbol]) else np.nan,
                            "realized_price_source": "primary_input" if pd.notna(realized_simple[symbol]) else "missing",
                        }
                    )
            ledger_rows.append(row)
    ledger = pd.DataFrame(ledger_rows)
    holdings = pd.DataFrame(holdings_rows) if holdings_rows else _empty_holdings()
    summary = _summary(
        ledger,
        requested_session_date=None,
        evaluated_session_date=evaluated_days[-1],
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
            "evaluation_start": first.isoformat(),
            "evaluation_end": last.isoformat(),
            "evaluation_sessions": int(len(evaluated_days)),
        }
    )
    return FiveMinuteForecastBacktestResult(ledger=ledger, holdings=holdings, summary=summary)


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
    """Audit every non-overlapping intraday forecast in one completed session."""
    clean = _regular_session_minute_closes(minute_bars)
    available_days = sorted(set(clean["session_date"]))
    requested = date.fromisoformat(str(session_date)) if session_date is not None else None
    eligible_days = [day for day in available_days if requested is None or day <= requested]
    if not eligible_days:
        raise ValueError("the requested session date is not present in the supplied regular-session bars")
    evaluated_day = eligible_days[-1]
    result = run_intraday_forecast_backtest_range(
        minute_bars,
        interval_minutes=interval_minutes,
        evaluation_start=evaluated_day,
        evaluation_end=evaluated_day,
        top_n=top_n,
        max_weight=max_weight,
        risk_aversion=risk_aversion,
        lookback_observations=lookback_observations,
        min_training_sessions=min_training_sessions,
        min_covariance_scenarios=min_covariance_scenarios,
        seasonality_prior_observations=seasonality_prior_observations,
    )
    result.summary["requested_session_date"] = requested.isoformat() if requested else None
    return result


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
    parser.add_argument("--evaluation-start", help="inclusive YYYY-MM-DD range start; requires --evaluation-end")
    parser.add_argument("--evaluation-end", help="inclusive YYYY-MM-DD range end; requires --evaluation-start")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-weight", type=float, default=0.10)
    parser.add_argument("--risk-aversion", type=float, default=10.0)
    parser.add_argument("--lookback-observations", type=int, default=780)
    parser.add_argument("--min-training-sessions", type=int, default=10)
    parser.add_argument("--min-covariance-scenarios", type=int, default=500)
    parser.add_argument("--seasonality-prior-observations", type=int, default=20)
    parser.add_argument(
        "--fallback-input",
        help="Optional one-minute fallback bars used only to repair otherwise-missing realized endpoints",
    )
    args = parser.parse_args()
    if bool(args.evaluation_start) != bool(args.evaluation_end):
        parser.error("--evaluation-start and --evaluation-end must be supplied together")
    if args.session_date and args.evaluation_start:
        parser.error("--session-date cannot be combined with --evaluation-start/--evaluation-end")
    common = {
        "interval_minutes": FORECAST_INTERVAL_MINUTES,
        "top_n": args.top_n,
        "max_weight": args.max_weight,
        "risk_aversion": args.risk_aversion,
        "lookback_observations": args.lookback_observations,
        "min_training_sessions": args.min_training_sessions,
        "min_covariance_scenarios": args.min_covariance_scenarios,
        "seasonality_prior_observations": args.seasonality_prior_observations,
    }
    minute_bars = pd.read_csv(args.input)
    if args.evaluation_start:
        result = run_intraday_forecast_backtest_range(
            minute_bars,
            evaluation_start=args.evaluation_start,
            evaluation_end=args.evaluation_end,
            **common,
        )
    else:
        result = run_intraday_forecast_backtest(
            minute_bars,
            session_date=args.session_date,
            **common,
        )
    if args.fallback_input:
        result = reconcile_intraday_forecast_outcomes(
            result,
            pd.read_csv(args.fallback_input),
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
