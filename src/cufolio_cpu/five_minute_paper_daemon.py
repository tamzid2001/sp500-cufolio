"""24/7 handoff supervisor for causal five-minute paper sessions."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

from .five_minute_intraday_backtest import NEW_YORK, OPENING_DECISION_PRICE_TIME
from .five_minute_paper_session import run_five_minute_paper_session

DEFAULT_HISTORY_CALENDAR_DAYS = 55


def utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(tz=NEW_YORK)).tz_convert("UTC")


def rolling_history_start(reference: pd.Timestamp, *, calendar_days: int) -> str:
    """Return enough completed sessions for the ten-session five-minute model."""
    if calendar_days < 28:
        raise ValueError("history_calendar_days must be at least 28")
    local_day = reference.tz_convert(NEW_YORK).date() - timedelta(days=calendar_days)
    return (
        pd.Timestamp.combine(local_day, OPENING_DECISION_PRICE_TIME)
        .tz_localize(NEW_YORK)
        .tz_convert("UTC")
        .isoformat()
    )


def run_five_minute_paper_daemon(
    *,
    run_seconds: int,
    state_path: str | Path,
    output_dir: str | Path,
    top_n: int,
    max_weight: str,
    history_start: str | None = None,
    history_calendar_days: int = DEFAULT_HISTORY_CALENDAR_DAYS,
    minute_cache_path: str | Path = "var/five_minute_paper_24x7_minute_history.csv.gz",
    historical_minute_cache_path: str | Path | None = "var/iex_one_minute_history.csv.gz",
) -> None:
    if run_seconds <= 0:
        raise ValueError("run_seconds must be positive")
    deadline = time.monotonic() + run_seconds
    print(
        "FIVE MINUTE PAPER DAEMON STARTED | "
        f"mode=paper slice={run_seconds}s state={state_path} minute_cache={minute_cache_path}",
        flush=True,
    )
    heartbeat: pd.Timestamp | None = None
    while time.monotonic() < deadline:
        current = utc_now()
        local = current.tz_convert(NEW_YORK)
        if local.weekday() < 5 and local.time().isoformat() < "16:01:00":
            try:
                session_history_start = history_start or rolling_history_start(
                    current, calendar_days=history_calendar_days
                )
                ledger = run_five_minute_paper_session(
                    session_day=local.date(), history_start=session_history_start, output_dir=output_dir,
                    top_n=top_n, max_weight=Decimal(max_weight), checkpoint_path=state_path,
                    minute_cache_path=minute_cache_path,
                    historical_minute_cache_path=historical_minute_cache_path,
                    stop_at=current + timedelta(seconds=max(0, deadline - time.monotonic() - 30)),
                    resume=True,
                )
                print(f"FIVE MINUTE PAPER SESSION RETURNED | ledger_events={len(ledger)}", flush=True)
            except Exception as error:
                print(f"FIVE MINUTE PAPER DAEMON RETRY | reason={error}", flush=True)
                time.sleep(min(60, max(1, deadline - time.monotonic())))
                continue
        if heartbeat is None or current - heartbeat >= timedelta(minutes=5):
            print(
                "FIVE MINUTE PAPER DAEMON IDLE | "
                f"utc={current.isoformat()} new_york={local.isoformat()} remaining={max(0, int(deadline-time.monotonic()))}s",
                flush=True,
            )
            heartbeat = current
        time.sleep(min(300, max(1, deadline - time.monotonic())))
    print("FIVE MINUTE PAPER DAEMON HANDOFF READY | slice deadline reached", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 24/7 handoff slice of the five-minute paper strategy.")
    parser.add_argument("--run-seconds", type=int, default=20700)
    parser.add_argument("--state", default="var/five_minute_paper_24x7_state.json")
    parser.add_argument("--minute-cache", default="var/five_minute_paper_24x7_minute_history.csv.gz")
    parser.add_argument("--historical-minute-cache", default="var/iex_one_minute_history.csv.gz")
    parser.add_argument("--history-start")
    parser.add_argument("--history-calendar-days", type=int, default=DEFAULT_HISTORY_CALENDAR_DAYS)
    parser.add_argument("--output-dir", default="artifacts/five_minute_paper_24x7")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-weight", default="0.10")
    args = parser.parse_args()
    run_five_minute_paper_daemon(
        run_seconds=args.run_seconds, state_path=args.state, minute_cache_path=args.minute_cache,
        historical_minute_cache_path=args.historical_minute_cache, output_dir=args.output_dir,
        top_n=args.top_n, max_weight=args.max_weight, history_start=args.history_start,
        history_calendar_days=args.history_calendar_days,
    )


if __name__ == "__main__":
    main()
