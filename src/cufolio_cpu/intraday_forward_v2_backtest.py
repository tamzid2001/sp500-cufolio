"""Causal daily-signal audit for the V2 15-minute forward-return model.

Every evaluation-session close creates one V2 portfolio using only labels that
were complete at that time.  The portfolio is then valued at V2's exact
34-bar (510-minute) endpoint.  Consecutive daily signals overlap, so their
mean outcome is reported separately from a non-overlapping, compoundable
cohort series.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .intraday_forward_v2 import (
    FEATURE_COLUMNS,
    _feature_frame,
    _fit_predict,
    _select_covariance_scenarios,
    _select_diagonal_covariance_assets,
    build_forward_dataset,
)
from .optimize import (
    forecast_diagonal_mean_variance_weights,
    forecast_mean_variance_weights,
)


@dataclass(frozen=True)
class DailyForwardV2BacktestResult:
    """A daily V2 signal ledger, its holdings, and aggregate diagnostics."""

    ledger: pd.DataFrame
    holdings: pd.DataFrame
    summary: dict[str, object]


def _empty_holdings() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "session_date",
            "decision_timestamp",
            "target_timestamp",
            "symbol",
            "target_weight",
            "predicted_forward_excess_return",
            "realized_asset_simple_return",
            "realized_asset_log_return",
            "realized_status",
        ]
    )


def _realized_endpoint_frame(features: pd.DataFrame, *, horizon_bars: int) -> pd.DataFrame:
    """Attach the exact per-symbol V2 endpoint without filling missing bars."""
    result = features.loc[:, ["timestamp", "symbol", "close"]].copy()
    grouped = result.groupby("symbol", sort=False)
    result["target_close"] = grouped["close"].shift(-horizon_bars)
    result["target_timestamp"] = grouped["timestamp"].shift(-horizon_bars)
    return result.set_index(["timestamp", "symbol"])


def _select_v2_weights(
    history: pd.DataFrame,
    current: pd.DataFrame,
    *,
    top_n: int,
    max_weight: float,
) -> tuple[pd.Series | None, pd.Series | None, dict[str, object]]:
    """Run the V2 forecast and covariance allocation from completed labels."""
    if len(current) < 2 or len(history) < 500:
        return None, None, {"forecast_status": "insufficient_current_features_or_completed_labels"}

    prediction = pd.Series(
        _fit_predict(history, current),
        index=current["symbol"].astype(str),
        name="predicted_forward_excess_return",
    )
    scenarios, selected_symbols = _select_covariance_scenarios(
        history, prediction, top_n=top_n, max_weight=max_weight
    )
    if scenarios.empty:
        selected, variances, scenario_count = _select_diagonal_covariance_assets(
            history, prediction, top_n=top_n, max_weight=max_weight
        )
        if selected.empty:
            return None, None, {
                "forecast_status": "insufficient_observed_forward_return_scenarios",
                "covariance_estimator": "unavailable",
                "forward_return_scenarios": 0,
                "covariance_assets": 0,
            }
        allocation = forecast_diagonal_mean_variance_weights(
            selected, variances, risk_aversion=10.0, max_weight=max_weight
        )
        return allocation.weights, prediction, {
            "forecast_status": "ok",
            "optimizer_status": allocation.status,
            "covariance_estimator": "diagonal_unpaired_forward_returns",
            "forward_return_scenarios": scenario_count,
            "covariance_assets": int(len(selected)),
            "portfolio_predicted_forward_excess_return": float(allocation.expected_return),
        }

    selected = prediction.reindex(selected_symbols)
    allocation = forecast_mean_variance_weights(
        scenarios, selected, risk_aversion=10.0, max_weight=max_weight
    )
    return allocation.weights, prediction, {
        "forecast_status": "ok",
        "optimizer_status": allocation.status,
        "covariance_estimator": "complete_case_forward_returns",
        "forward_return_scenarios": int(len(scenarios)),
        "covariance_assets": int(len(selected)),
        "portfolio_predicted_forward_excess_return": float(allocation.expected_return),
    }


def _non_overlapping_returns(realized: pd.DataFrame) -> pd.Series:
    """Choose the earliest subsequent signal after each exact holding window."""
    selected: list[float] = []
    eligible_at: pd.Timestamp | None = None
    for row in realized.sort_values("decision_timestamp").itertuples(index=False):
        decision_at = pd.Timestamp(row.decision_timestamp)
        if eligible_at is not None and decision_at < eligible_at:
            continue
        selected.append(float(row.actual_portfolio_simple_return))
        eligible_at = pd.Timestamp(row.target_timestamp)
    return pd.Series(selected, dtype=float)


def run_daily_forward_v2_backtest(
    bars: pd.DataFrame,
    *,
    evaluation_start: str | date,
    evaluation_end: str | date,
    horizon_minutes: int = 500,
    top_n: int = 20,
    max_weight: float = 0.10,
    min_sessions: int = 20,
    lookback_calendar_days: int = 120,
) -> DailyForwardV2BacktestResult:
    """Create and value one causal V2 portfolio at each regular-session close.

    The model's live generator retrieves a rolling 120 calendar days, which is
    also the default here. The caller is responsible for supplying that warm-up
    before ``evaluation_start``; this function then enforces the more important
    causality boundary: a training label must finish strictly before the signal
    timestamp.
    """
    if top_n < 2:
        raise ValueError("top_n must be at least two")
    if not 0 < max_weight <= 1 or top_n * max_weight < 1 - 1e-12:
        raise ValueError("top_n and max_weight cannot form a fully invested long-only portfolio")
    if lookback_calendar_days < 30:
        raise ValueError("lookback_calendar_days must be at least 30")
    first = pd.Timestamp(evaluation_start).date()
    last = pd.Timestamp(evaluation_end).date()
    if first > last:
        raise ValueError("evaluation_start must not be after evaluation_end")

    dataset, horizon_bars, effective_minutes = build_forward_dataset(
        bars, interval="15m", horizon_minutes=horizon_minutes
    )
    features = _feature_frame(bars)
    endpoints = _realized_endpoint_frame(features, horizon_bars=horizon_bars)
    session_ends = (
        features.groupby("session_date", sort=True)["timestamp"].max().sort_values()
    )
    decision_sessions = [
        (pd.Timestamp(session).date(), timestamp)
        for session, timestamp in session_ends.items()
        if first <= pd.Timestamp(session).date() <= last
    ]
    if not decision_sessions:
        raise ValueError("the requested evaluation range has no regular sessions in the supplied bars")

    ledger_rows: list[dict[str, object]] = []
    holdings_rows: list[dict[str, object]] = []
    for session_day, decision_at in decision_sessions:
        current = features.loc[features["timestamp"].eq(decision_at)].dropna(subset=FEATURE_COLUMNS).copy()
        # A label whose end equals the signal time was not known until that
        # close, so it is deliberately excluded along with all later labels.
        window_start = decision_at - pd.Timedelta(days=lookback_calendar_days)
        history = dataset.loc[
            (dataset["timestamp"] >= window_start) & (dataset["target_timestamp"] < decision_at)
        ].copy()
        history_sessions = int(history["session_date"].nunique())
        row: dict[str, object] = {
            "session_date": session_day.isoformat(),
            "decision_timestamp": decision_at,
            "horizon_bars": horizon_bars,
            "effective_horizon_minutes": effective_minutes,
            "history_sessions": history_sessions,
            "history_rows": int(len(history)),
            "current_feature_assets": int(len(current)),
            "forecast_status": "not_attempted",
            "holding_count": 0,
            "realized_status": "not_forecast",
            "target_timestamp": pd.NaT,
            "actual_portfolio_simple_return": np.nan,
            "actual_portfolio_log_return": np.nan,
        }
        if history_sessions < min_sessions:
            row["forecast_status"] = "insufficient_completed_history_sessions"
            ledger_rows.append(row)
            continue

        weights, prediction, diagnostic = _select_v2_weights(
            history, current, top_n=top_n, max_weight=max_weight
        )
        row.update(diagnostic)
        if weights is None or prediction is None:
            ledger_rows.append(row)
            continue

        row["holding_count"] = int(len(weights))
        selected = current.set_index("symbol").reindex(weights.index)
        outcome_index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([decision_at] * len(weights)), weights.index],
            names=["timestamp", "symbol"],
        )
        outcomes = endpoints.reindex(outcome_index)
        outcomes.index = weights.index
        realized_simple = outcomes["target_close"] / selected["close"] - 1.0
        target_times = pd.to_datetime(outcomes["target_timestamp"], utc=True)
        is_complete = (
            selected["close"].notna().all()
            and outcomes["target_close"].notna().all()
            and (outcomes["target_close"] > 0).all()
            and target_times.notna().all()
        )
        if is_complete:
            row["target_timestamp"] = target_times.max()
            row["realized_status"] = "ok"
            actual_simple = float(weights.dot(realized_simple))
            row["actual_portfolio_simple_return"] = actual_simple
            row["actual_portfolio_log_return"] = float(np.log1p(actual_simple))
        else:
            row["realized_status"] = "missing_exact_v2_endpoint"

        for symbol, weight in weights.items():
            asset_return = realized_simple.get(symbol)
            asset_target = target_times.get(symbol)
            holdings_rows.append(
                {
                    "session_date": session_day.isoformat(),
                    "decision_timestamp": decision_at,
                    "target_timestamp": asset_target,
                    "symbol": symbol,
                    "target_weight": float(weight),
                    "predicted_forward_excess_return": float(prediction[symbol]),
                    "realized_asset_simple_return": float(asset_return) if pd.notna(asset_return) else np.nan,
                    "realized_asset_log_return": float(np.log1p(asset_return)) if pd.notna(asset_return) else np.nan,
                    "realized_status": "ok" if is_complete else "missing_exact_v2_endpoint",
                }
            )
        ledger_rows.append(row)

    ledger = pd.DataFrame(ledger_rows)
    holdings = pd.DataFrame(holdings_rows) if holdings_rows else _empty_holdings()
    realized = ledger.loc[ledger["realized_status"].eq("ok")].copy()
    returns = pd.to_numeric(realized["actual_portfolio_simple_return"], errors="coerce").dropna()
    non_overlapping = _non_overlapping_returns(realized) if not realized.empty else pd.Series(dtype=float)
    previous = pd.Series(dtype=float)
    turnovers: list[float] = []
    for _, group in holdings.groupby("decision_timestamp", sort=True):
        current_weights = group.set_index("symbol")["target_weight"].astype(float)
        all_symbols = current_weights.index.union(previous.index)
        turnovers.append(
            float(
                0.5
                * (
                    current_weights.reindex(all_symbols, fill_value=0.0)
                    - previous.reindex(all_symbols, fill_value=0.0)
                ).abs().sum()
            )
        )
        previous = current_weights

    summary: dict[str, object] = {
        "research_only": True,
        "model_version": "leakage_safe_ensemble_v2",
        "decision_cadence": "one signal at each regular-session close",
        "realization_rule": "each portfolio is held for its exact per-symbol 34-bar V2 endpoint; no endpoint is forward-filled",
        "overlap_caveat": "consecutive daily 510-minute signals overlap, so all-signal outcomes are not compounded as a single fully invested strategy",
        "evaluation_start": first.isoformat(),
        "evaluation_end": last.isoformat(),
        "evaluation_sessions": int(len(decision_sessions)),
        "daily_signals": int(ledger["forecast_status"].eq("ok").sum()),
        "realized_daily_signals": int(len(realized)),
        "exact_realized_coverage": float(len(realized) / ledger["forecast_status"].eq("ok").sum())
        if int(ledger["forecast_status"].eq("ok").sum())
        else None,
        "mean_daily_signal_return": float(returns.mean()) if not returns.empty else None,
        "median_daily_signal_return": float(returns.median()) if not returns.empty else None,
        "daily_signal_win_rate": float((returns > 0).mean()) if not returns.empty else None,
        "non_overlapping_signals": int(len(non_overlapping)),
        "non_overlapping_compounded_return": float((1.0 + non_overlapping).prod() - 1.0)
        if not non_overlapping.empty
        else None,
        "mean_target_turnover": float(np.mean(turnovers)) if turnovers else None,
        "distinct_selected_symbols": int(holdings["symbol"].nunique()) if not holdings.empty else 0,
        "top_n": top_n,
        "max_weight": max_weight,
        "requested_horizon_minutes": horizon_minutes,
        "effective_horizon_minutes": effective_minutes,
        "lookback_calendar_days": lookback_calendar_days,
        "universe_caveat": "The input uses the current S&P 500 constituent list, not point-in-time membership, so this research backtest has survivorship bias.",
        "transaction_costs_included": False,
    }
    return DailyForwardV2BacktestResult(ledger=ledger, holdings=holdings, summary=summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a causal daily-signal V2 forward-return research backtest.")
    parser.add_argument("--input", required=True, help="CSV or CSV.GZ with timestamp,symbol,close 15-minute bars")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evaluation-start", required=True, help="inclusive YYYY-MM-DD")
    parser.add_argument("--evaluation-end", required=True, help="inclusive YYYY-MM-DD")
    parser.add_argument("--horizon-minutes", type=int, default=500)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-weight", type=float, default=0.10)
    parser.add_argument("--min-sessions", type=int, default=20)
    parser.add_argument("--lookback-calendar-days", type=int, default=120)
    args = parser.parse_args()
    result = run_daily_forward_v2_backtest(
        pd.read_csv(args.input),
        evaluation_start=args.evaluation_start,
        evaluation_end=args.evaluation_end,
        horizon_minutes=args.horizon_minutes,
        top_n=args.top_n,
        max_weight=args.max_weight,
        min_sessions=args.min_sessions,
        lookback_calendar_days=args.lookback_calendar_days,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.ledger.to_csv(output / "v2_daily_signal_ledger.csv", index=False)
    result.holdings.to_csv(output / "v2_daily_signal_holdings.csv", index=False)
    (output / "v2_daily_backtest_summary.json").write_text(json.dumps(result.summary, indent=2) + "\n")
    print(
        f"Wrote {len(result.ledger)} daily V2 signal rows, {len(result.holdings)} holdings rows, "
        f"and {result.summary['realized_daily_signals']} exactly realized signals."
    )


if __name__ == "__main__":
    main()
