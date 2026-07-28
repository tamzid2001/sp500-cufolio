"""Causal one-hour portfolio research backtest using exact one-minute bars.

The strategy is deliberately separate from the daily target and paper-trading
paths.  It estimates one-hour returns from *completed* earlier hourly windows
only, and restores a selected portfolio's weights every 15 minutes during the
one-hour holding window.  It never crosses an overnight boundary, never
forward-fills a missing price, and never treats a missing interval as a zero
return.

This is a historical research backtest.  Its optimizer is mathematically
optimal only for the supplied trailing mean/covariance assumptions; it is not
a claim of future performance or an order-generation system.
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

from .optimize import mean_variance_weights

NEW_YORK = ZoneInfo("America/New_York")
SESSION_OPEN = timedelta(hours=9, minutes=30)
SESSION_CLOSE = timedelta(hours=16)
HOLDING_MINUTES = 60
REBALANCE_MINUTES = 15
HOLDING_DELTA = timedelta(minutes=60)
REBALANCE_DELTA = timedelta(minutes=15)
HOURLY_SELECTION_STEPS = range(6)
# This is the production paper cadence, not the legacy same-minute research
# cadence below.  Each tuple is (observed selection minute, one-hour target
# start), both in New York time.  The last selection is deliberately refreshed
# at 14:20 for the 14:30--15:30 close target.
PAPER_CADENCE_FORECAST_TIMES = (
    (clock_time(9, 20), clock_time(10, 30)),
    (clock_time(10, 20), clock_time(11, 30)),
    (clock_time(11, 20), clock_time(12, 30)),
    (clock_time(12, 20), clock_time(13, 30)),
    (clock_time(14, 20), clock_time(14, 30)),
)
# Paper-target files reject zero weights. Optimizer tolerance can otherwise
# leave an economically empty numerical remainder in a selected position.
MIN_EMITTABLE_TARGET_WEIGHT = 1e-6


@dataclass(frozen=True)
class HourlyBacktestResult:
    performance: pd.DataFrame
    selections: pd.DataFrame
    status: dict[str, object]


@dataclass(frozen=True)
class HourlyCandidateResult:
    """A causal, research-only allocation for one selected hourly window."""

    weights: pd.DataFrame
    status: dict[str, object]


@dataclass(frozen=True)
class HourlyPaperCadenceBacktestResult:
    """Exact historical audit of the live hourly paper-trader cadence."""

    performance: pd.DataFrame
    selections: pd.DataFrame
    forecasts: pd.DataFrame
    status: dict[str, object]


def hourly_paper_cadence_endpoint_times() -> frozenset[clock_time]:
    """Return every exact minute needed by the live-cadence audit.

    The set contains forecast decisions, all completed one-hour training-label
    endpoints, and every 15-minute execution endpoint.  It lets a long
    historical download be compacted without synthesizing any price.
    """
    endpoints = {clock_time(hour, 30) for hour in range(9, 16)}
    endpoints.update(selection for selection, _target_start in PAPER_CADENCE_FORECAST_TIMES)
    anchor = pd.Timestamp("2000-01-03", tz=NEW_YORK)
    for _selection, target_start in PAPER_CADENCE_FORECAST_TIMES:
        start = anchor.replace(hour=target_start.hour, minute=target_start.minute)
        for step in range(HOLDING_MINUTES // REBALANCE_MINUTES + 1):
            endpoints.add((start + step * REBALANCE_DELTA).time())
    return frozenset(endpoints)


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


def _exact_minute_prices(minute_bars: pd.DataFrame, at: pd.Timestamp) -> pd.Series:
    """Return the exact observed one-minute close at ``at`` for each symbol.

    This intentionally does not apply the regular-session filter: a forecast
    made at 09:20 New York time relies on the exact 09:20 pre-market close,
    while the one-hour training labels remain regular-session-only.  A missing
    decision-minute quote is excluded rather than filled from an older quote.
    """
    required = {"timestamp", "symbol", "close"}
    if missing := required.difference(minute_bars.columns):
        raise ValueError(f"intraday bars are missing required columns: {sorted(missing)}")
    clean = minute_bars.loc[:, ["timestamp", "symbol", "close"]].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")
    clean["symbol"] = clean["symbol"].astype(str).str.upper().str.strip()
    clean["close"] = pd.to_numeric(clean["close"], errors="coerce")
    exact = clean.loc[
        (clean["timestamp"] == at)
        & clean["close"].notna()
        & (clean["close"] > 0)
        & clean["symbol"].ne("")
    ]
    return exact.drop_duplicates("symbol", keep="last").set_index("symbol")["close"]


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
    *,
    decision_at: pd.Timestamp,
    current_prices: pd.Series,
    top_n: int,
    lookback_scenarios: int,
    min_training_scenarios: int,
    max_weight: float,
    risk_aversion: float,
) -> tuple[pd.Series | None, dict[str, object]]:
    """Select an allocation using only labels completed strictly before a decision."""
    training = hourly_returns.loc[target_ends.reindex(hourly_returns.index) < decision_at]
    training = training.tail(lookback_scenarios)
    required_candidates = int(np.ceil((1 - 1e-12) / max_weight))
    eligible = training.columns[
        (training.notna().sum(axis=0) >= min_training_scenarios)
        & training.columns.isin(current_prices.index)
    ]
    diagnostic: dict[str, object] = {
        "decision_timestamp": decision_at.isoformat(),
        "training_rows_before_complete_case": int(len(training)),
        "input_assets": int(len(training.columns)),
        "assets_with_decision_price": int(len(current_prices)),
        "eligible_assets": int(len(eligible)),
        "required_candidates_for_weight_cap": required_candidates,
    }
    if len(eligible) < 2:
        diagnostic.update(
            {
                "training_rows": 0,
                "training_end": None,
                "reason": "insufficient_assets_with_completed_one_hour_training_returns",
            }
        )
        return None, diagnostic
    # A current S&P 500 convenience universe includes changing constituents,
    # occasional halts, and incomplete data coverage.  Requiring every one of
    # hundreds of names to share every observation creates an empty sample.
    # Rank eligible assets on their own completed labels, then greedily retain
    # only high-ranked names that preserve a complete covariance sample.
    expected = training.reindex(columns=eligible).mean(skipna=True).sort_values(ascending=False)
    candidates: list[str] = []
    common_rows = pd.Series(True, index=training.index)
    for symbol in expected.index:
        with_symbol = common_rows & training[symbol].notna()
        if int(with_symbol.sum()) < min_training_scenarios:
            continue
        candidates.append(symbol)
        common_rows = with_symbol
        if len(candidates) == top_n:
            break
    scenarios = training.loc[common_rows, candidates]
    diagnostic.update(
        {
            "training_rows": int(len(scenarios)),
            "training_end": target_ends.loc[scenarios.index].max().isoformat() if not scenarios.empty else None,
            "candidate_count": int(len(candidates)),
            "candidate_weight_capacity": float(max_weight * len(candidates)),
        }
    )
    if len(scenarios) < min_training_scenarios or scenarios.shape[1] < 2:
        diagnostic["reason"] = "insufficient_complete_candidate_scenarios"
        return None, diagnostic
    if len(candidates) < 2 or max_weight * len(candidates) < 1 - 1e-12:
        diagnostic["reason"] = "insufficient_candidates_for_weight_cap"
        return None, diagnostic
    allocation = mean_variance_weights(
        scenarios, risk_aversion=risk_aversion, max_weight=max_weight,
    )
    # A conic solver may return a value a few floating-point ulps above the
    # hard bound.  Restore feasibility without changing the optimizer's
    # ranking: clip, then distribute the residual only to available capacity.
    weights = allocation.weights.clip(lower=0.0, upper=max_weight)
    weights.loc[weights < MIN_EMITTABLE_TARGET_WEIGHT] = 0.0
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
    weights = weights.loc[weights > 0].copy()
    diagnostic.update(
        {
            "reason": "ok",
            "optimizer_status": allocation.status,
            "expected_one_hour_log_return": allocation.expected_return,
            "training_end": target_ends.loc[scenarios.index].max().isoformat(),
        }
    )
    return weights, diagnostic


def generate_hourly_one_hour_candidate(
    minute_bars: pd.DataFrame,
    *,
    decision_at: str | pd.Timestamp,
    target_start_at: str | pd.Timestamp | None = None,
    top_n: int = 20,
    lookback_scenarios: int = 120,
    min_training_scenarios: int = 20,
    max_weight: float = 0.10,
    risk_aversion: float = 10.0,
) -> HourlyCandidateResult:
    """Create a causal candidate for a one-hour regular-session target.

    ``decision_at`` is the exact observed pricing minute.  When
    ``target_start_at`` is supplied, it may be a later hourly target; for
    example, a 09:20 New York decision can forecast the 10:30--11:30 holding
    window.  The forecast does not require the target window's prices to have
    occurred.  With no target supplied, the legacy same-timestamp target is
    retained for research-only candidate workflows.
    """
    if top_n < 2:
        raise ValueError("top_n must be at least two")
    if lookback_scenarios < min_training_scenarios:
        raise ValueError("lookback_scenarios must be at least min_training_scenarios")
    if min_training_scenarios < 5:
        raise ValueError("min_training_scenarios must be at least five")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")

    decision = pd.Timestamp(decision_at)
    if decision.tzinfo is None:
        raise ValueError("decision_at must include a UTC offset")
    decision = decision.tz_convert("UTC")
    target_start = pd.Timestamp(target_start_at) if target_start_at is not None else decision
    if target_start.tzinfo is None:
        raise ValueError("target_start_at must include a UTC offset")
    target_start = target_start.tz_convert("UTC")
    if target_start_at is not None and target_start <= decision:
        raise ValueError("target_start_at must be strictly later than decision_at")
    target_local = target_start.tz_convert(NEW_YORK)
    if (
        target_local.hour not in range(9, 15)
        or target_local.minute != 30
        or target_local.second != 0
        or target_local.microsecond != 0
        or target_start + HOLDING_DELTA
        > target_local.normalize() + SESSION_CLOSE
    ):
        raise ValueError(
            "target_start_at must be an exact 09:30, 10:30, ..., 14:30 New York timestamp "
            "with a complete same-session one-hour holding window"
        )
    hourly_returns, target_ends, _ = build_one_hour_return_panel(minute_bars)
    current_prices = _exact_minute_prices(minute_bars, decision)
    weights, diagnostic = _target_weights(
        hourly_returns,
        target_ends,
        decision_at=decision,
        current_prices=current_prices,
        top_n=top_n,
        lookback_scenarios=lookback_scenarios,
        min_training_scenarios=min_training_scenarios,
        max_weight=max_weight,
        risk_aversion=risk_aversion,
    )
    status: dict[str, object] = {
        "research_only": True,
        "bar_interval": "1m",
        "decision_timestamp": decision.isoformat(),
        "target_start": target_start.isoformat(),
        "target_end": (target_start + HOLDING_DELTA).isoformat(),
        "target_horizon_minutes": HOLDING_MINUTES,
        "execution_rebalance_frequency_minutes": REBALANCE_MINUTES,
        "weights_generated": weights is not None,
        "causality_rule": "each training target_end is strictly earlier than its decision timestamp",
        "data_quality_rule": "the decision-minute close and completed target labels must exist; no forward fill or zero-return substitution",
        **diagnostic,
    }
    if weights is None:
        return HourlyCandidateResult(
            pd.DataFrame(columns=["decision_timestamp", "target_start", "target_end", "symbol", "target_weight"]),
            status,
        )
    candidate = pd.DataFrame(
        {
            "decision_timestamp": decision,
            "target_start": target_start,
            "target_end": target_start + HOLDING_DELTA,
            "symbol": weights.index,
            "target_weight": weights.to_numpy(dtype=float),
        }
    )
    return HourlyCandidateResult(candidate, status)


def generate_to_close_candidate(
    minute_bars: pd.DataFrame,
    *,
    decision_at: str | pd.Timestamp,
    target_end_at: str | pd.Timestamp,
    top_n: int = 20,
    lookback_scenarios: int = 120,
    min_training_scenarios: int = 20,
    max_weight: float = 0.10,
    risk_aversion: float = 10.0,
) -> HourlyCandidateResult:
    """Generate a causal same-time-of-day portfolio held through ``target_end_at``.

    This is used for an explicitly requested one-time shortened session-close
    paper run.  Each historical label begins at the same New York clock minute
    as ``decision_at`` and ends at the same clock minute as ``target_end_at``;
    neither current-session target prices nor filled-in prices may enter the
    allocation.
    """
    if top_n < 2:
        raise ValueError("top_n must be at least two")
    if lookback_scenarios < min_training_scenarios:
        raise ValueError("lookback_scenarios must be at least min_training_scenarios")
    if min_training_scenarios < 5:
        raise ValueError("min_training_scenarios must be at least five")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")

    decision = pd.Timestamp(decision_at)
    target_end = pd.Timestamp(target_end_at)
    if decision.tzinfo is None or target_end.tzinfo is None:
        raise ValueError("decision_at and target_end_at must include UTC offsets")
    decision, target_end = decision.tz_convert("UTC"), target_end.tz_convert("UTC")
    if target_end <= decision:
        raise ValueError("target_end_at must be later than decision_at")
    decision_local, target_local = decision.tz_convert(NEW_YORK), target_end.tz_convert(NEW_YORK)
    if decision_local.date() != target_local.date() or target_local.time() > clock_time(16):
        raise ValueError("decision and target end must be in the same New York regular session")

    clean = _regular_session_minute_closes(minute_bars)
    closes = clean.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    rows: list[pd.Series] = []
    ends: dict[pd.Timestamp, pd.Timestamp] = {}
    for session_date in pd.DatetimeIndex(sorted(clean["session_date"].unique())):
        local_day = pd.Timestamp(session_date).date()
        start = pd.Timestamp.combine(local_day, decision_local.time()).tz_localize(NEW_YORK).tz_convert("UTC")
        end = pd.Timestamp.combine(local_day, target_local.time()).tz_localize(NEW_YORK).tz_convert("UTC")
        if end <= start:
            continue
        row = np.log(closes.reindex([end]).iloc[0] / closes.reindex([start]).iloc[0])
        row.name = start
        rows.append(row)
        ends[start] = end
    if not rows:
        raise ValueError("no same-session decision-to-close windows are available")
    returns = pd.DataFrame(rows).sort_index()
    target_ends = pd.Series(ends, name="target_end")
    current_prices = _exact_minute_prices(minute_bars, decision)
    weights, diagnostic = _target_weights(
        returns,
        target_ends,
        decision_at=decision,
        current_prices=current_prices,
        top_n=top_n,
        lookback_scenarios=lookback_scenarios,
        min_training_scenarios=min_training_scenarios,
        max_weight=max_weight,
        risk_aversion=risk_aversion,
    )
    horizon_minutes = int((target_end - decision).total_seconds() // 60)
    status: dict[str, object] = {
        "research_only": True,
        "bar_interval": "1m",
        "decision_timestamp": decision.isoformat(),
        "target_end": target_end.isoformat(),
        "target_horizon_minutes": horizon_minutes,
        "weights_generated": weights is not None,
        "causality_rule": "each same-clock-time training target_end is strictly earlier than its decision timestamp",
        "data_quality_rule": "the exact decision and target endpoint closes must exist; no forward fill or zero-return substitution",
        **diagnostic,
    }
    if weights is None:
        return HourlyCandidateResult(pd.DataFrame(columns=["decision_timestamp", "target_end", "symbol", "target_weight"]), status)
    return HourlyCandidateResult(
        pd.DataFrame(
            {
                "decision_timestamp": decision,
                "target_end": target_end,
                "symbol": weights.index,
                "target_weight": weights.to_numpy(dtype=float),
            }
        ),
        status,
    )


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
            decision_at=decision_at,
            current_prices=closes.reindex([decision_at]).iloc[0].dropna(),
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


def _session_timestamp(session_day: date, at: clock_time) -> pd.Timestamp:
    return pd.Timestamp.combine(session_day, at).tz_localize(NEW_YORK).tz_convert("UTC")


def _optional_session_date(value: str | date | None, *, name: str) -> date | None:
    if value is None:
        return None
    result = pd.Timestamp(value).date()
    if not isinstance(result, date):  # defensive: pandas currently always returns date here
        raise ValueError(f"{name} must be a calendar date")
    return result


def run_hourly_paper_cadence_backtest(
    minute_bars: pd.DataFrame,
    *,
    top_n: int = 20,
    lookback_scenarios: int = 120,
    min_training_scenarios: int = 20,
    max_weight: float = 0.10,
    risk_aversion: float = 10.0,
    transaction_cost_bps: float = 0.0,
    evaluation_start: str | date | None = None,
    evaluation_end: str | date | None = None,
) -> HourlyPaperCadenceBacktestResult:
    """Audit the exact forecast and rebalance cadence used by paper trading.

    For each session this generates forecasts at 09:20, 10:20, 11:20, 12:20,
    and 14:20 New York time for the later one-hour targets defined in
    :data:`PAPER_CADENCE_FORECAST_TIMES`.  The target is rebalanced at its
    start and then every 15 minutes.  A forecast is scored only when every
    selected asset has each exact execution endpoint; an unavailable endpoint
    is reported as unavailable, never carried forward or converted to zero.
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

    evaluation_first = _optional_session_date(evaluation_start, name="evaluation_start")
    evaluation_last = _optional_session_date(evaluation_end, name="evaluation_end")
    if evaluation_first is not None and evaluation_last is not None and evaluation_first > evaluation_last:
        raise ValueError("evaluation_start must not be after evaluation_end")

    hourly_returns, target_ends, closes = build_one_hour_return_panel(minute_bars)
    all_symbols = closes.columns
    session_days = sorted({timestamp.tz_convert(NEW_YORK).date() for timestamp in closes.index})
    selected_session_days = [
        session_day
        for session_day in session_days
        if (evaluation_first is None or session_day >= evaluation_first)
        and (evaluation_last is None or session_day <= evaluation_last)
    ]
    holdings = pd.Series(0.0, index=all_symbols)
    cash = 1.0
    equity = 1.0
    performance_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []

    for session_day in selected_session_days:
        for selection_time, target_time in PAPER_CADENCE_FORECAST_TIMES:
            decision_at = _session_timestamp(session_day, selection_time)
            target_start = _session_timestamp(session_day, target_time)
            target_end = target_start + HOLDING_DELTA
            current_prices = _exact_minute_prices(minute_bars, decision_at)
            weights, diagnostic = _target_weights(
                hourly_returns,
                target_ends,
                decision_at=decision_at,
                current_prices=current_prices,
                top_n=top_n,
                lookback_scenarios=lookback_scenarios,
                min_training_scenarios=min_training_scenarios,
                max_weight=max_weight,
                risk_aversion=risk_aversion,
            )
            base_row: dict[str, object] = {
                "session_date": session_day.isoformat(),
                "decision_timestamp": decision_at,
                "target_start": target_start,
                "target_end": target_end,
                "execution_rebalance_frequency_minutes": REBALANCE_MINUTES,
                **diagnostic,
            }
            if weights is None:
                forecast_rows.append(
                    {
                        **base_row,
                        "forecast_status": str(diagnostic.get("reason", "forecast_not_generated")),
                        "selected_symbols": 0,
                        "expected_one_hour_log_return": None,
                        "expected_one_hour_simple_return": None,
                        "realized_gross_simple_return": None,
                        "realized_net_simple_return": None,
                        "prediction_error_percentage_points": None,
                    }
                )
                # The live trader has no target in this case.  Preserve an
                # auditable cash state instead of manufacturing an outcome.
                holdings = pd.Series(0.0, index=all_symbols)
                cash = equity
                continue

            expected_log = float(diagnostic["expected_one_hour_log_return"])
            expected_simple = float(np.expm1(expected_log))
            for symbol, weight in weights.items():
                selection_rows.append(
                    {
                        **base_row,
                        "symbol": symbol,
                        "target_weight": float(weight),
                    }
                )
            quarter_returns = _quarter_hour_returns(closes, target_start, weights.index)
            if quarter_returns is None:
                forecast_rows.append(
                    {
                        **base_row,
                        "forecast_status": "missing_exact_one_minute_execution_prices",
                        "selected_symbols": int(len(weights)),
                        "expected_one_hour_log_return": expected_log,
                        "expected_one_hour_simple_return": expected_simple,
                        "realized_gross_simple_return": None,
                        "realized_net_simple_return": None,
                        "prediction_error_percentage_points": None,
                    }
                )
                # Do not assume an unobservable return.  Reset to cash before
                # the next independent target and leave this forecast out of
                # all realized-performance aggregates.
                holdings = pd.Series(0.0, index=all_symbols)
                cash = equity
                continue

            opening_equity = equity
            target = weights.reindex(all_symbols).fillna(0.0)
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
                performance_rows.append(
                    {
                        "session_date": session_day.isoformat(),
                        "decision_timestamp": decision_at,
                        "rebalance_timestamp": rebalance_at,
                        "interval_end": rebalance_at + REBALANCE_DELTA,
                        "hourly_target_start": target_start,
                        "hourly_target_end": target_end,
                        "selection_recomputed": rebalance_at == target_start,
                        "turnover": turnover,
                        "transaction_cost": transaction_cost,
                        "gross_simple_return": float((interval_returns * weights).sum()),
                        "net_simple_return": float(equity / pre_trade_equity - 1),
                        "equity": equity,
                    }
                )
            realized_gross = float((1 + quarter_returns.mul(weights, axis=1).sum(axis=1)).prod() - 1)
            realized_net = float(equity / opening_equity - 1)
            forecast_rows.append(
                {
                    **base_row,
                    "forecast_status": "selected_and_realized",
                    "selected_symbols": int(len(weights)),
                    "expected_one_hour_log_return": expected_log,
                    "expected_one_hour_simple_return": expected_simple,
                    "realized_gross_simple_return": realized_gross,
                    "realized_net_simple_return": realized_net,
                    "prediction_error_percentage_points": (realized_gross - expected_simple) * 100,
                }
            )

    performance = pd.DataFrame(performance_rows)
    selections = pd.DataFrame(selection_rows)
    forecasts = pd.DataFrame(forecast_rows)
    realized = forecasts.loc[
        forecasts.get("forecast_status", pd.Series(dtype=object)).eq("selected_and_realized")
    ].copy()
    if realized.empty:
        expected_compounded = actual_compounded = None
        mean_absolute_error_bps = directional_accuracy = None
    else:
        expected = realized["expected_one_hour_simple_return"].astype(float)
        actual = realized["realized_gross_simple_return"].astype(float)
        expected_compounded = float((1 + expected).prod() - 1)
        actual_compounded = float((1 + actual).prod() - 1)
        mean_absolute_error_bps = float((actual - expected).abs().mean() * 10_000)
        nonzero = (expected != 0) & (actual != 0)
        directional_accuracy = float((np.sign(expected.loc[nonzero]) == np.sign(actual.loc[nonzero])).mean()) if nonzero.any() else None
    status: dict[str, object] = {
        "research_only": True,
        "strategy_cadence": "09:20,10:20,11:20,12:20,14:20 New York forecasts; 15-minute target rebalances",
        "bar_interval": "1m",
        "market_data_rule": "exact IEX minute closes only; no forward fill, stale-price substitution, or zero-return substitution",
        "causality_rule": "each training target_end is strictly earlier than its forecast decision timestamp",
        "target_horizon_minutes": HOLDING_MINUTES,
        "execution_rebalance_frequency_minutes": REBALANCE_MINUTES,
        "evaluation_start": evaluation_first.isoformat() if evaluation_first else None,
        "evaluation_end": evaluation_last.isoformat() if evaluation_last else None,
        "evaluation_sessions_available": int(len(selected_session_days)),
        "forecast_windows_considered": int(len(forecasts)),
        "forecast_windows_selected": int(forecasts.get("selected_symbols", pd.Series(dtype=int)).gt(0).sum()),
        "forecast_windows_realized_exactly": int(len(realized)),
        "forecast_coverage": float(len(realized) / len(forecasts)) if len(forecasts) else 0.0,
        "quarter_hour_rebalance_rows": int(len(performance)),
        "rebalances_per_exact_realized_window": int(len(performance) / len(realized)) if len(realized) else 0,
        "expected_compounded_return_on_realized_windows": expected_compounded,
        "actual_gross_compounded_return": actual_compounded,
        "mean_absolute_error_bps": mean_absolute_error_bps,
        "directional_accuracy": directional_accuracy,
        "total_transaction_cost": float(performance["transaction_cost"].sum()) if not performance.empty else 0.0,
        "ending_equity": equity,
    }
    return HourlyPaperCadenceBacktestResult(performance, selections, forecasts, status)


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
    parser.add_argument(
        "--paper-cadence",
        action="store_true",
        help="audit the exact live paper forecast schedule instead of the legacy same-minute hourly research cadence",
    )
    parser.add_argument("--evaluation-start", help="optional inclusive New York calendar date for --paper-cadence")
    parser.add_argument("--evaluation-end", help="optional inclusive New York calendar date for --paper-cadence")
    parser.add_argument(
        "--decision-at",
        help="UTC-offset timestamp for a causal candidate-only output (for example 2026-07-28T14:30:00Z)",
    )
    parser.add_argument(
        "--target-start-at",
        help="optional later UTC-offset hourly target start for a forecast candidate",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bars = pd.read_csv(args.input)
    if args.decision_at:
        candidate = generate_hourly_one_hour_candidate(
            bars,
            decision_at=args.decision_at,
            target_start_at=args.target_start_at,
            top_n=args.top_n,
            lookback_scenarios=args.lookback_scenarios,
            min_training_scenarios=args.min_training_scenarios,
            max_weight=args.max_weight,
            risk_aversion=args.risk_aversion,
        )
        candidate.weights.to_csv(output_dir / "hourly_one_hour_candidate.csv", index=False)
        (output_dir / "hourly_one_hour_candidate_status.json").write_text(
            json.dumps(candidate.status, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(
            f"Wrote {len(candidate.weights)} candidate weights for "
            f"{candidate.status['decision_timestamp']}"
        )
        return
    if args.paper_cadence:
        result = run_hourly_paper_cadence_backtest(
            bars,
            top_n=args.top_n,
            lookback_scenarios=args.lookback_scenarios,
            min_training_scenarios=args.min_training_scenarios,
            max_weight=args.max_weight,
            risk_aversion=args.risk_aversion,
            transaction_cost_bps=args.transaction_cost_bps,
            evaluation_start=args.evaluation_start,
            evaluation_end=args.evaluation_end,
        )
        result.performance.to_csv(output_dir / "paper_cadence_quarter_hour_rebalances.csv", index=False)
        result.selections.to_csv(output_dir / "paper_cadence_hourly_portfolios.csv", index=False)
        result.forecasts.to_csv(output_dir / "paper_cadence_forecasts.csv", index=False)
        (output_dir / "paper_cadence_backtest_status.json").write_text(
            json.dumps(result.status, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(
            f"Wrote {len(result.forecasts)} live-cadence forecast rows, "
            f"{result.status['forecast_windows_realized_exactly']} exact realized windows, and "
            f"{len(result.performance)} 15-minute rebalance rows"
        )
    else:
        result = run_hourly_one_hour_backtest(
            bars,
            top_n=args.top_n,
            lookback_scenarios=args.lookback_scenarios,
            min_training_scenarios=args.min_training_scenarios,
            max_weight=args.max_weight,
            risk_aversion=args.risk_aversion,
            transaction_cost_bps=args.transaction_cost_bps,
        )
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
