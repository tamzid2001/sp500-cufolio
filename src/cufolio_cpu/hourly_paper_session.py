"""Paper-first hourly portfolio forecast session with 15-minute rebalancing.

This runner is intentionally separate from the daily paper worker.  It holds
one process for a bounded regular-session window, forecasts each one-hour
target before it begins, and submits rebalances every fifteen minutes during
that target. Paper trading is the default; live mode requires a separate,
explicit acknowledgement and separate credentials.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, time as clock_time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

from .alpaca import download_minute_bars
from .hourly_intraday_backtest import NEW_YORK, generate_hourly_one_hour_candidate
from .paper_rebalance import AlpacaTradingClient, load_target_weights, run_end_of_day_flatten, run_rebalance
from .universe import current_sp500_universe

FORECAST_LEAD = timedelta(hours=1, minutes=10)
FINAL_TARGET_REFRESH_LEAD = timedelta(minutes=10)
TARGET_FORECAST_LEADS = (
    (10, FORECAST_LEAD),
    (11, FORECAST_LEAD),
    (12, FORECAST_LEAD),
    (13, FORECAST_LEAD),
    # The final 14:30--15:30 target is refreshed at 14:20 using the newest
    # observed minute, as requested.  It replaces the old 70-minute lead for
    # this final window; there is no duplicate target or duplicate order.
    (14, FINAL_TARGET_REFRESH_LEAD),
)
MAX_START_LAG = timedelta(minutes=10)


@dataclass(frozen=True)
class SessionEvent:
    kind: Literal["select", "rebalance", "flatten"]
    due_at: pd.Timestamp
    selection_at: pd.Timestamp | None = None
    target_start: pd.Timestamp | None = None


def session_events(session_day: date) -> list[SessionEvent]:
    """Return the exact paper-session schedule for a New York session date.

    Forecasts are made at 09:20, 10:20, 11:20, 12:20, and 14:20 New York
    time using only the exact observed selection-minute price. They target
    the respectively later 10:30--11:30 through 14:30--15:30 windows. The
    final 14:20 selection produces the 14:30--15:30 target requested by the
    user. Each target is first rebalanced at its start and then at +15, +30,
    and +45 minutes, followed by a 15:30 flatten.
    """
    events: list[SessionEvent] = []
    for hour, forecast_lead in TARGET_FORECAST_LEADS:
        target_start = pd.Timestamp.combine(session_day, clock_time(hour, 30)).tz_localize(NEW_YORK).tz_convert("UTC")
        selection_at = target_start - forecast_lead
        events.append(SessionEvent("select", selection_at, selection_at, target_start))
        for minutes in (0, 15, 30, 45):
            events.append(
                SessionEvent(
                    "rebalance",
                    target_start + timedelta(minutes=minutes),
                    selection_at,
                    target_start,
                )
            )
    flatten_at = pd.Timestamp.combine(session_day, clock_time(15, 30)).tz_localize(NEW_YORK).tz_convert("UTC")
    events.append(SessionEvent("flatten", flatten_at))
    return sorted(events, key=lambda event: event.due_at)


def _utc_timestamp(value: datetime | pd.Timestamp | None = None) -> pd.Timestamp:
    timestamp = pd.Timestamp(value or datetime.now(tz=NEW_YORK))
    if timestamp.tzinfo is None:
        raise ValueError("timestamps must include a UTC offset")
    return timestamp.tz_convert("UTC")


def _history_through(
    bars: pd.DataFrame | None,
    symbols: list[str],
    *,
    start: str,
    decision_at: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch only missing minute bars, through and including decision minute."""
    end = decision_at + timedelta(minutes=1)
    if bars is None or bars.empty:
        return download_minute_bars(symbols, start, end)
    existing = bars.copy()
    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
    latest = existing["timestamp"].max()
    if latest >= decision_at:
        return existing
    incremental = download_minute_bars(symbols, latest + timedelta(minutes=1), end)
    if incremental.empty:
        return existing
    combined = pd.concat([existing, incremental], ignore_index=True)
    return combined.drop_duplicates(["symbol", "timestamp"], keep="last").sort_values(["symbol", "timestamp"])


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def run_hourly_paper_session(
    *,
    session_day: date,
    history_start: str,
    output_dir: str | Path,
    top_n: int = 20,
    lookback_scenarios: int = 120,
    min_training_scenarios: int = 20,
    max_weight: Decimal = Decimal("0.10"),
    risk_aversion: float = 10.0,
    min_order_notional: Decimal = Decimal("1"),
    min_weight_drift: Decimal = Decimal("0.0025"),
    mode: Literal["paper", "live"] = "paper",
    allow_live_trading: bool = False,
    now: Callable[[], pd.Timestamp] = _utc_timestamp,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, object]]:
    """Run the bounded selection/rebalance/flatten session.

    A delayed GitHub Actions start of more than ten minutes after a planned
    selection fails closed before submitting any stale order. ``live`` is
    deliberately unavailable unless ``allow_live_trading`` is explicit.
    """
    if mode == "live" and not allow_live_trading:
        raise ValueError("live mode requires allow_live_trading=True")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    universe = current_sp500_universe()
    symbol_column = "source_symbol" if "source_symbol" in universe.columns else "symbol"
    symbols = universe[symbol_column].dropna().astype(str).str.upper().drop_duplicates().to_list()
    universe.to_csv(output / "current_sp500_universe.csv", index=False)
    client = AlpacaTradingClient.from_environment(mode=mode)
    bars: pd.DataFrame | None = None
    targets_by_start: dict[pd.Timestamp, dict[str, Decimal]] = {}
    forecast_details: dict[pd.Timestamp, dict[str, object]] = {}
    ledger: list[dict[str, object]] = []
    # The full S&P one-minute history is the expensive operation.  Start it
    # before the first 09:20 forecast, then fetch only the new minutes at each
    # later forecast.  This keeps the 10:30 paper entry timely without looking
    # past any forecast's exact observed minute.
    events = session_events(session_day)
    first_selection = events[0].selection_at
    assert first_selection is not None
    prefetch_now = now()
    if prefetch_now < first_selection:
        bars = download_minute_bars(symbols, history_start, prefetch_now + timedelta(minutes=1))

    for event in events:
        while now() < event.due_at:
            sleep(min(30.0, float((event.due_at - now()).total_seconds())))
        observed_at = now()
        if event.kind == "select":
            assert event.selection_at is not None
            assert event.target_start is not None
            if observed_at > event.due_at + MAX_START_LAG:
                raise RuntimeError(
                    f"forecast selection at {event.selection_at.isoformat()} is too late for the planned entry: "
                    f"runner reached it at {observed_at.isoformat()}"
                )
            bars = _history_through(bars, symbols, start=history_start, decision_at=event.selection_at)
            if now() > event.due_at + MAX_START_LAG:
                raise RuntimeError(
                    f"history for {event.selection_at.isoformat()} was not available in time; "
                    "refusing a stale forecast"
                )
            candidate = generate_hourly_one_hour_candidate(
                bars,
                decision_at=event.selection_at,
                target_start_at=event.target_start,
                top_n=top_n,
                lookback_scenarios=lookback_scenarios,
                min_training_scenarios=min_training_scenarios,
                max_weight=float(max_weight),
                risk_aversion=risk_aversion,
            )
            stamp = f"{event.target_start.strftime('%H%M')}_from_{event.selection_at.strftime('%H%M')}"
            candidate_path = output / f"candidate_for_{stamp}.csv"
            candidate.weights.to_csv(candidate_path, index=False)
            _write_json(output / f"candidate_{stamp}_status.json", candidate.status)
            if not candidate.status["weights_generated"]:
                raise RuntimeError(
                    f"no exact causal forecast for {event.target_start.isoformat()} "
                    f"from {event.selection_at.isoformat()}: {candidate.status}"
                )
            targets_by_start[event.target_start] = load_target_weights(candidate_path, max_weight=max_weight)
            expected_log = float(candidate.status["expected_one_hour_log_return"])
            forecast_details[event.target_start] = {
                "selection_timestamp": event.selection_at.isoformat(),
                "target_start": event.target_start.isoformat(),
                "target_end": candidate.status["target_end"],
                "trading_mode": client.mode,
                "expected_one_hour_log_return": expected_log,
                "expected_one_hour_simple_return": float(np.expm1(expected_log)),
                "candidate_file": candidate_path.name,
            }
            ledger.append(
                {
                    "event": "forecast_selected",
                    "scheduled_at": event.due_at.isoformat(),
                    "observed_at": observed_at.isoformat(),
                    **forecast_details[event.target_start],
                }
            )
        elif event.kind == "rebalance":
            assert event.selection_at is not None
            assert event.target_start is not None
            if observed_at > event.due_at + MAX_START_LAG:
                raise RuntimeError(
                    f"portfolio rebalance for {event.target_start.isoformat()} is too late: "
                    f"runner reached it at {observed_at.isoformat()}"
                )
            if event.target_start not in targets_by_start:
                raise RuntimeError(f"missing active target for {event.due_at.isoformat()}")
            report = run_rebalance(
                client,
                targets_by_start[event.target_start],
                min_order_notional=min_order_notional,
                min_weight_drift=min_weight_drift,
                liquidate_non_target_positions=True,
                execute=True,
            )
            ledger.append(
                {
                    "event": "quarter_hour_paper_rebalance",
                    "scheduled_at": event.due_at.isoformat(),
                    "observed_at": observed_at.isoformat(),
                    **forecast_details[event.target_start],
                    "paper_rebalance": report,
                }
            )
        else:
            report = run_end_of_day_flatten(client, execute=True)
            ledger.append(
                {
                    "event": "final_paper_flatten",
                    "scheduled_at": event.due_at.isoformat(),
                    "observed_at": observed_at.isoformat(),
                    "paper_rebalance": report,
                }
            )
        _write_json(output / "hourly_paper_session_ledger.json", ledger)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded causal hourly Alpaca portfolio session.")
    parser.add_argument("--session-date", required=True, help="New York session date, YYYY-MM-DD")
    parser.add_argument("--history-start", required=True, help="UTC start for one-minute training history")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--lookback-scenarios", type=int, default=120)
    parser.add_argument("--min-training-scenarios", type=int, default=20)
    parser.add_argument("--max-weight", type=Decimal, default=Decimal("0.10"))
    parser.add_argument("--risk-aversion", type=float, default=10.0)
    parser.add_argument("--min-order-notional", type=Decimal, default=Decimal("1"))
    parser.add_argument("--min-weight-drift", type=Decimal, default=Decimal("0.0025"))
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument(
        "--allow-live-trading",
        action="store_true",
        help="required with --mode live; paper is the default",
    )
    args = parser.parse_args()
    if args.mode == "live" and not args.allow_live_trading:
        parser.error("--mode live requires --allow-live-trading and ALPACA_LIVE_* credentials")
    ledger = run_hourly_paper_session(
        session_day=date.fromisoformat(args.session_date),
        history_start=args.history_start,
        output_dir=args.output_dir,
        top_n=args.top_n,
        lookback_scenarios=args.lookback_scenarios,
        min_training_scenarios=args.min_training_scenarios,
        max_weight=args.max_weight,
        risk_aversion=args.risk_aversion,
        min_order_notional=args.min_order_notional,
        min_weight_drift=args.min_weight_drift,
        mode=args.mode,
        allow_live_trading=args.allow_live_trading,
    )
    print(f"Completed {len(ledger)} {args.mode}-session events")


if __name__ == "__main__":
    main()
