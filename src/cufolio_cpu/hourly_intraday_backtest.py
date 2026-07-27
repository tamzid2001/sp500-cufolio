"""Causal one-hour portfolio research backtest using exact one-minute bars.

The strategy is deliberately separate from the daily target and paper-trading
paths.  It selects a long-only mean--variance portfolio at 09:30, 10:30, ...,
14:30 New York time, estimates one-hour returns from *completed* earlier
hourly windows only, and restores that portfolio's weights every 15 minutes
until the next hourly selection.  It never crosses an overnight boundary,
never forward-fills a missing price, and never treats a missing interval as a
zero return.

This is a historical research backtest.  Its optimizer is mathematically
optimal only for the supplied trailing mean/covariance assumptions; it is not
a claim of future performance or an order-generation system.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .optimize import mean_variance_weights

NEW_YORK = ZoneInfo("America/New_York")
SESSION_OPEN = timedelta(hours=9, minutes=30)
SESSION_CLOSE = timedelta(hours=16)
HOLDING_MINUTES = 60
REBALANCE_MINUTES = 15
HOLDING_DELTA = timedelta(minutes=60)
REBALANCE_DELTA = timedelta(minutes=15)
HOURLY_SELECTION_STEPS = range(6)


@dataclass(frozen=True)
class HourlyBacktestResult:
    performance: pd.DataFrame
    selections: pd.DataFrame
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
    clean = clean.drop_duplicates(["symbol", "timestamp"], keep="last")
    local = clean["timestamp"].dt.tz_convert(NEW_YORK)
    clock = local.dt.time
    in_regular_session = (
        (clock >= pd.Timestamp("09:30").time())
        & (clock <= pd.Timestamp("16:00").time())
    )
    clean = clean.loc[in_regular_session].copy()
    clean["session_date"] = local.loc[in_regular_session].dt.tz_localize(None).dt.normalize()
    if clean.empty:
        raise ValueError("no positive one-minute closes fall in the US regular session")
    return clean.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _selection_timestamps(session_dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Return 09:30, 10:30, ..., 14:30 in UTC for every session date."""
    timestamps: list[pd.Timestamp] = []
    for session_date in session_dates:
        open_at = pd.Timestamp(session_date).tz_localize(NEW_YORK) + SESSION_OPEN
        for step in HOURLY_SELECTION_STEPS:
            start = open_at + step * HOLDING_DELTA
            if start + HOLDING_DELTA <= (
                pd.Timestamp(session_date).tz_localize(NEW_YORK) + SESSION_CLOSE
            ):
                timestamps.append(start.tz_convert("UTC"))
    return timestamps


def build_one_hour_return_panel(
    minute_bars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build exact same-session hourly log-return outcomes from minute closes.

    The resulting row at timestamp *t* is the return from the known close at
    *t* through the close exactly 60 minutes later.  Missing endpoint prices
    remain missing.  They are not substituted with stale prices or zeros.
    """
    clean = _regular_session_minute_closes(minute_bars)
    closes = clean.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    session_dates = pd.DatetimeIndex(sorted(clean["session_date"].unique()))
    starts = _selection_timestamps(session_dates)
    rows: list[pd.Series] = []
    ends: dict[pd.Timestamp, pd.Timestamp] = {}
    for start in starts:
        end = start + HOLDING_DELTA
        start_prices = closes.reindex([start]).iloc[0]
        end_prices = closes.reindex([end]).iloc[0]
        row = np.log(end_prices / start_prices)
        row.name = start
        rows.append(row)
        ends[start] = end
    if not rows:
        raise ValueError("no complete regular-session hourly windows are available")
    return pd.DataFrame(rows).sort_index(), pd.Series(ends, name="target_end"), closes


def _quarter_hour_returns(
    closes: pd.DataFrame, start: pd.Timestamp, symbols: pd.Index,
) -> pd.DataFrame | None:
    """Return the four exact 15-minute holding returns for one hourly target."""
    records: list[pd.Series] = []
    for step in range(HOLDING_MINUTES // REBALANCE_MINUTES):
        rebalance_at = start + step * REBALANCE_DELTA
        interval_end = rebalance_at + REBALANCE_DELTA
        first = closes.reindex([rebalance_at], columns=symbols).iloc[0]
        second = closes.reindex([interval_end], columns=symbols).iloc[0]
        simple = second / first - 1
        if simple.isna().any() or not np.isfinite(simple.to_numpy(dtype=float)).all():
            return None
        simple.name = rebalance_at
        records.append(simple)
    return pd.DataFrame(records)


def _target_weights(
    hourly_returns: pd.DataFrame,
    target_ends: pd.Series,
    closes: pd.DataFrame,
    *,
    decision_at: pd.Timestamp,
    top_n: int,
    lookback_scenarios: int,
    min_training_scenarios: int,
    max_weight: float,
    risk_aversion: float,
) -> tuple[pd.Series | None, dict[str, object]]:
    """Select an allocation using only labels completed strictly before a decision."""
    training = hourly_returns.loc[target_ends.reindex(hourly_returns.index) < decision_at]
    training = training.tail(lookback_scenarios)
    current_prices = closes.reindex([decision_at]).iloc[0].dropna()
    eligible = training.columns[
        (training.notna().sum(axis=0) >= min_training_scenarios)
        & training.columns.isin(current_prices.index)
    ]
    complete = training.reindex(columns=eligible).dropna(axis=0, how="any")
    diagnostic: dict[str, object] = {
        "decision_timestamp": decision_at.isoformat(),
        "training_rows_before_complete_case": int(len(training)),
        "training_rows": int(len(complete)),
        "training_end": (
            target_ends.loc[complete.index].max().isoformat() if not complete.empty else None
        ),
        "eligible_assets": int(len(eligible)),
    }
    if len(complete) < min_training_scenarios or complete.shape[1] < 2:
        diagnostic["reason"] = "insufficient_completed_one_hour_training_returns"
        return None, diagnostic
    expected = complete.mean().sort_values(ascending=False)
    candidates = expected.head(top_n)
    if len(candidates) < 2 or max_weight * len(candidates) < 1 - 1e-12:
        diagnostic["reason"] = "insufficient_candidates_for_weight_cap"
        return None, diagnostic
    scenarios = complete.reindex(columns=candidates.index).dropna(axis=0, how="any")
    if len(scenarios) < min_training_scenarios:
        diagnostic["reason"] = "insufficient_complete_candidate_scenarios"
        return None, diagnostic
    allocation = mean_variance_weights(
        scenarios, risk_aversion=risk_aversion, max_weight=max_weight,
    )
    # A conic solver may return a value a few floating-point ulps above the
    # hard bound.  Restore feasibility without changing the optimizer's
    # ranking: clip, then distribute the residual only to available capacity.
    weights = allocation.weights.clip(lower=0.0, upper=max_weight)
    residual = float(1 - weights.sum())
    while abs(residual) > 1e-12:
        if residual > 0:
            capacity = max_weight - weights
            recipients = capacity[capacity > 1e-12]
            if recipients.empty:
                raise RuntimeError("cannot restore capped portfolio weight feasibility")
            addition = min(residual / len(recipients), float(recipients.min()))
            weights.loc[recipients.index] += addition
        else:
            donors = weights[weights > 1e-12]
            if donors.empty:
                raise RuntimeError("cannot restore non-negative portfolio weight feasibility")
            reduction = min((-residual) / len(donors), float(donors.min()))
            weights.loc[donors.index] -= reduction
        residual = float(1 - weights.sum())
    diagnostic.update(
        {
            "reason": "ok",
            "optimizer_status": allocation.status,
            "expected_one_hour_log_return": allocation.expected_return,
            "candidate_count": int(len(candidates)),
            "training_end": target_ends.loc[scenarios.index].max().isoformat(),
        }
    )
    return weights, diagnostic


def run_hourly_one_hour_backtest(
    minute_bars: pd.DataFrame,
    *,
    top_n: int = 20,
    lookback_scenarios: int = 120,
    min_training_scenarios: int = 20,
    max_weight: float = 0.10,
    risk_aversion: float = 10.0,
    transaction_cost_bps: float = 5.0,
) -> HourlyBacktestResult:
    """Backtest hourly one-hour selections with 15-minute target rebalancing.

    The 09:30 portfolio is held/rebalanced through 10:30.  At 10:30 a new
    portfolio is selected using only previously completed hourly outcomes and
    is held/rebalanced through 11:30, continuing through the 14:30--15:30
    window.  No partial final-hour target is invented after 15:30.
    """
    if top_n < 2:
        raise ValueError("top_n must be at least two")
    if lookback_scenarios < min_training_scenarios:
        raise ValueError("lookback_scenarios must be at least min_training_scenarios")
    if min_training_scenarios < 5:
        raise ValueError("min_training_scenarios must be at least five")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps cannot be negative")

    hourly_returns, target_ends, closes = build_one_hour_return_panel(minute_bars)
    all_symbols = closes.columns
    holdings = pd.Series(0.0, index=all_symbols)
    cash = 1.0
    equity = 1.0
    performance_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    selected_windows = 0
    skipped_windows = 0

    for decision_at in hourly_returns.index:
        weights, diagnostic = _target_weights(
            hourly_returns,
            target_ends,
            closes,
            decision_at=decision_at,
            top_n=top_n,
            lookback_scenarios=lookback_scenarios,
            min_training_scenarios=min_training_scenarios,
            max_weight=max_weight,
            risk_aversion=risk_aversion,
        )
        if weights is None:
            skipped_windows += 1
            selection_rows.append({
                "decision_timestamp": decision_at,
                "target_end": target_ends.loc[decision_at],
                "symbol": None,
                "target_weight": None,
                **diagnostic,
            })
            # A missing causal training set is not a reason to synthesize a
            # portfolio.  Exit to cash at this rebalance instead.
            target = pd.Series(0.0, index=all_symbols)
            quarter_returns = pd.DataFrame(
                0.0,
                index=[
                    decision_at + step * REBALANCE_DELTA
                    for step in range(HOLDING_MINUTES // REBALANCE_MINUTES)
                ],
                columns=all_symbols,
            )
        else:
            quarter_returns = _quarter_hour_returns(closes, decision_at, weights.index)
            if quarter_returns is None:
                skipped_windows += 1
                selection_rows.append({
                    "decision_timestamp": decision_at,
                    "target_end": target_ends.loc[decision_at],
                    "symbol": None,
                    "target_weight": None,
                    **diagnostic,
                    "reason": "missing_exact_one_minute_execution_prices",
                })
                target = pd.Series(0.0, index=all_symbols)
                quarter_returns = pd.DataFrame(
                    0.0,
                    index=[
                        decision_at + step * REBALANCE_DELTA
                        for step in range(HOLDING_MINUTES // REBALANCE_MINUTES)
                    ],
                    columns=all_symbols,
                )
            else:
                selected_windows += 1
                target = weights.reindex(all_symbols).fillna(0.0)
                for symbol, weight in weights.items():
                    selection_rows.append({
                        "decision_timestamp": decision_at,
                        "target_end": target_ends.loc[decision_at],
                        "symbol": symbol,
                        "target_weight": float(weight),
                        **diagnostic,
                    })

        for rebalance_at, interval_returns in quarter_returns.iterrows():
            pre_trade_equity = equity
            if pre_trade_equity <= 0:
                raise RuntimeError("backtest equity is non-positive")
            pre_trade_weights = holdings / pre_trade_equity
            cash_weight = cash / pre_trade_equity
            target_cash_weight = float(1 - target.sum())
            turnover = 0.5 * (
                float((target - pre_trade_weights).abs().sum())
                + abs(target_cash_weight - cash_weight)
            )
            transaction_cost = pre_trade_equity * turnover * transaction_cost_bps / 10_000
            post_cost_equity = pre_trade_equity - transaction_cost
            holdings = target * post_cost_equity
            cash = target_cash_weight * post_cost_equity
            holdings = holdings * (1 + interval_returns.reindex(all_symbols).fillna(0.0))
            equity = float(holdings.sum() + cash)
            gross_simple_return = float(equity / post_cost_equity - 1) if post_cost_equity else 0.0
            net_simple_return = float(equity / pre_trade_equity - 1)
            performance_rows.append({
                "rebalance_timestamp": rebalance_at,
                "interval_end": rebalance_at + REBALANCE_DELTA,
                "hourly_target_start": decision_at,
                "hourly_target_end": target_ends.loc[decision_at],
                "selection_recomputed": rebalance_at == decision_at,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "gross_simple_return": gross_simple_return,
                "net_simple_return": net_simple_return,
                "equity": equity,
            })

    performance = pd.DataFrame(performance_rows)
    selections = pd.DataFrame(selection_rows)
    status: dict[str, object] = {
        "research_only": True,
        "bar_interval": "1m",
        "target_horizon_minutes": HOLDING_MINUTES,
        "selection_frequency_minutes": HOLDING_MINUTES,
        "execution_rebalance_frequency_minutes": REBALANCE_MINUTES,
        "hourly_windows_available": int(len(hourly_returns)),
        "hourly_windows_selected": selected_windows,
        "hourly_windows_skipped": skipped_windows,
        "quarter_hour_rebalances": int(len(performance)),
        "total_return": float(equity - 1),
        "ending_equity": equity,
        "total_transaction_cost": float(performance["transaction_cost"].sum()) if not performance.empty else 0.0,
        "selection_rule": "top trailing expected one-hour log returns, then long-only mean-variance allocation",
        "causality_rule": "each training target_end is strictly earlier than its decision timestamp",
        "data_quality_rule": "missing one-minute endpoint prices skip the affected window; no forward fill or zero-return substitution",
    }
    return HourlyBacktestResult(performance, selections, status)


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only hourly one-hour portfolio backtest from one-minute bars.")
    parser.add_argument("--input", required=True, help="CSV with timestamp,symbol,close one-minute bars")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--lookback-scenarios", type=int, default=120)
    parser.add_argument("--min-training-scenarios", type=int, default=20)
    parser.add_argument("--max-weight", type=float, default=0.10)
    parser.add_argument("--risk-aversion", type=float, default=10.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    args = parser.parse_args()
    result = run_hourly_one_hour_backtest(
        pd.read_csv(args.input),
        top_n=args.top_n,
        lookback_scenarios=args.lookback_scenarios,
        min_training_scenarios=args.min_training_scenarios,
        max_weight=args.max_weight,
        risk_aversion=args.risk_aversion,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.performance.to_csv(output_dir / "quarter_hour_rebalances.csv", index=False)
    result.selections.to_csv(output_dir / "hourly_one_hour_portfolios.csv", index=False)
    (output_dir / "hourly_one_hour_backtest_status.json").write_text(
        json.dumps(result.status, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(result.performance)} 15-minute rebalance rows and "
        f"{result.status['hourly_windows_selected']} selected hourly portfolios"
    )


if __name__ == "__main__":
    main()
