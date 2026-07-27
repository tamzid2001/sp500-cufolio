"""Persistent, durable paper-trading session worker.

Run this on an always-on host, not GitHub Actions.  Each completed action is
recorded before the worker queues its next market-time action, so a restart can
resume the session without repeating an opening buy or an end-of-day flatten.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .daily_cycle import run_daily_cycle
from .paper_rebalance import AlpacaTradingClient, run_end_of_day_flatten
from .prepare_daily_target import prepare_daily_target

NY = ZoneInfo("America/New_York")


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text()) if path.exists() else {"completed": {}}


def _save(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def _session(value: object) -> str:
    return pd.Timestamp(value).tz_convert(NY).date().isoformat()


def _run_once(client: AlpacaTradingClient, state_path: Path, targets: Path, status: Path) -> dict[str, object]:
    clock = client.get_clock()
    if not clock["is_open"]:
        return {"status": "market_closed"}
    now = pd.Timestamp(clock["timestamp"]).tz_convert(NY)
    day = _session(clock["timestamp"])
    state = _load(state_path)
    completed: dict[str, str] = state.setdefault("completed", {})  # type: ignore[assignment]
    minute = now.hour * 60 + now.minute

    def done(name: str, result: dict[str, object]) -> dict[str, object]:
        completed[name] = now.isoformat()
        state["last_result"] = result
        _save(state_path, state)
        return result

    # Flatten takes priority: no strategy position may remain after 15:55.
    if minute >= 15 * 60 + 55 and f"flatten:{day}" not in completed:
        return done(f"flatten:{day}", run_end_of_day_flatten(client, execute=True))
    # The next-session solve is complete before flattening and never submits orders.
    if minute >= 15 * 60 + 45 and f"select:{day}" not in completed:
        frame, target_status = prepare_daily_target(client)
        if frame.empty:
            return {"status": "selection_not_ready"}
        frame.to_csv(targets, index=False)
        status.write_text(json.dumps(target_status, indent=2) + "\n")
        return done(f"select:{day}", {"status": "next_session_target_prepared", **target_status})
    # At/after 09:30, buy today's prepared target then correct drift every 15m.
    # Stop drift trading at 15:30; 15:55 is the only closeout action.
    if minute >= 9 * 60 + 30 and minute < 15 * 60 + 30:
        slot = now.floor("15min").strftime("%H%M")
        name = f"rebalance:{day}:{slot}"
        if name not in completed:
            result = run_daily_cycle(client, targets_path=targets, target_status_path=status, execute=True)
            return done(name, result)
    return {"status": "waiting"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent paper-only session worker.")
    parser.add_argument("--state", default="var/paper_worker_state.json")
    parser.add_argument("--targets", default="assets/active_daily_target.csv")
    parser.add_argument("--target-status", default="assets/active_daily_target_status.json")
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--run-seconds", type=int, default=None)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    client = AlpacaTradingClient.from_environment(mode="paper")
    deadline = time.monotonic() + args.run_seconds if args.run_seconds else None
    while True:
        try:
            result = _run_once(client, Path(args.state), Path(args.targets), Path(args.target_status))
            print(json.dumps(result, default=str), flush=True)
        except Exception as error:  # retry on a future poll without losing durable state
            print(json.dumps({"status": "error", "reason": str(error)}), flush=True)
        if deadline is not None and time.monotonic() >= deadline:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
