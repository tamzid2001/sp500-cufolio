"""24/7 paper-only supervisor for resumable hourly portfolio sessions.

Each GitHub Actions invocation is deliberately capped below the hosted-runner
limit.  State is written after every selection/rebalance/flatten event, then
the workflow commits it and hands off to the next invocation.  Outside New
York weekday session hours the daemon remains idle; it never submits an order.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, time as clock_time, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

from .hourly_intraday_backtest import NEW_YORK
from .hourly_paper_session import run_hourly_paper_session

DEFAULT_HISTORY_CALENDAR_DAYS = 45


def utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(tz=NEW_YORK)).tz_convert("UTC")


def rolling_history_start(reference: pd.Timestamp, *, calendar_days: int) -> str:
    """Return a compact pre-market start with enough one-hour scenarios.

    Each complete session supplies six labels, so 45 calendar days normally
    gives about 30 sessions: materially more than the 20 sessions needed for
    the 120-observation trailing fit, with room for holidays and sparse names.
    """
    if calendar_days < 35:
        raise ValueError("history_calendar_days must be at least 35")
    local_day = reference.tz_convert(NEW_YORK).date() - timedelta(days=calendar_days)
    return pd.Timestamp.combine(local_day, clock_time(9, 20)).tz_localize(NEW_YORK).tz_convert("UTC").isoformat()


def run_hourly_paper_daemon(
    *,
    run_seconds: int,
    state_path: str | Path,
    output_dir: str | Path,
    top_n: int,
    max_weight: str,
    history_start: str | None = None,
    history_calendar_days: int = DEFAULT_HISTORY_CALENDAR_DAYS,
    minute_cache_path: str | Path = "var/hourly_paper_24x7_minute_endpoints.csv.gz",
) -> None:
    """Keep an hourly paper session alive until this handoff slice expires."""

    if run_seconds <= 0:
        raise ValueError("run_seconds must be positive")
    deadline = time.monotonic() + run_seconds
    state = Path(state_path)
    output = Path(output_dir)
    while time.monotonic() < deadline:
        now = utc_now()
        new_york = now.tz_convert(NEW_YORK)
        # The session runner itself waits for 09:20 when this handoff enters
        # before the first forecast.  After 15:30 or on weekends we only idle.
        if new_york.weekday() < 5 and new_york.time().isoformat() < "15:30:00":
            try:
                session_history_start = history_start or rolling_history_start(
                    now, calendar_days=history_calendar_days,
                )
                run_hourly_paper_session(
                    session_day=new_york.date(),
                    history_start=session_history_start,
                    output_dir=output,
                    top_n=top_n,
                    max_weight=Decimal(max_weight),
                    checkpoint_path=state,
                    minute_cache_path=minute_cache_path,
                    stop_at=now + timedelta(seconds=max(0, deadline - time.monotonic() - 30)),
                    resume=True,
                )
            except Exception as error:  # preserve the durable checkpoint and retry within this slice
                print(f"HOURLY PAPER DAEMON RETRY | {error}", flush=True)
                time.sleep(min(60, max(1, deadline - time.monotonic())))
                continue
        time.sleep(min(300, max(1, deadline - time.monotonic())))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 24/7 handoff slice of the hourly paper portfolio session.")
    parser.add_argument("--run-seconds", type=int, default=20700)
    parser.add_argument("--state", default="var/hourly_paper_24x7_state.json")
    parser.add_argument("--minute-cache", default="var/hourly_paper_24x7_minute_endpoints.csv.gz")
    parser.add_argument(
        "--history-start",
        help="optional fixed UTC bootstrap start; the default is a rolling 45-calendar-day window",
    )
    parser.add_argument("--history-calendar-days", type=int, default=DEFAULT_HISTORY_CALENDAR_DAYS)
    parser.add_argument("--output-dir", default="artifacts/hourly_paper_24x7")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-weight", default="0.10")
    args = parser.parse_args()
    run_hourly_paper_daemon(
        run_seconds=args.run_seconds,
        state_path=args.state,
        minute_cache_path=args.minute_cache,
        output_dir=args.output_dir,
        top_n=args.top_n,
        max_weight=args.max_weight,
        history_start=args.history_start,
        history_calendar_days=args.history_calendar_days,
    )


if __name__ == "__main__":
    main()
