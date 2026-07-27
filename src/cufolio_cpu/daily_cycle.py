"""Run one 15-minute rebalance or end-of-day liquidation cycle.

This module consumes the dated target prepared before the prior market close.
It intentionally does not re-optimize during the session: market orders follow
one frozen target, while the 15-minute loop only corrects material allocation
drift and manages the exit before the close.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .paper_rebalance import AlpacaTradingClient, load_target_weights, run_end_of_day_transition, run_rebalance

NEW_YORK = ZoneInfo("America/New_York")


def _session_date(value: object) -> str:
    return pd.Timestamp(value).tz_convert(NEW_YORK).date().isoformat()


def _within_close_buffer(clock: dict[str, object], close_buffer_minutes: int) -> bool:
    if not clock.get("is_open"):
        return False
    timestamp = pd.Timestamp(clock["timestamp"])
    next_close = pd.Timestamp(clock["next_close"])
    return timedelta(0) <= next_close - timestamp <= timedelta(minutes=close_buffer_minutes)


def _load_dated_targets(
    path: str | Path, status_path: str | Path, *, expected_session: str, max_weight: Decimal
) -> tuple[dict[str, Decimal], dict[str, object]]:
    status = json.loads(Path(status_path).read_text())
    target_session = status.get("target_session")
    if target_session != expected_session:
        raise ValueError(
            f"prepared target is for {target_session!r}, not the current market session {expected_session!r}"
        )
    return load_target_weights(path, max_weight=max_weight), status


def run_daily_cycle(
    client: AlpacaTradingClient,
    *,
    targets_path: str | Path,
    target_status_path: str | Path,
    close_buffer_minutes: int = 30,
    max_weight: Decimal = Decimal("0.10"),
    min_order_notional: Decimal = Decimal("1"),
    min_weight_drift: Decimal = Decimal("0.0025"),
    execute: bool = False,
) -> dict[str, object]:
    """Execute one cycle against the pre-close target for today's session."""
    if not 1 <= close_buffer_minutes <= 120:
        raise ValueError("close_buffer_minutes must be between 1 and 120")
    clock = client.get_clock()
    if not clock.get("is_open"):
        return {
            "trading_mode": client.mode,
            "trading_endpoint": client.base_url,
            "execute": execute,
            "clock_timestamp": clock.get("timestamp"),
            "market_open": False,
            "status": "market_closed",
            "orders": [],
        }
    if _within_close_buffer(clock, close_buffer_minutes):
        next_session = _session_date(clock["next_open"])
        try:
            next_targets, target_status = _load_dated_targets(
                targets_path, target_status_path, expected_session=next_session, max_weight=max_weight
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return {
                "trading_mode": client.mode,
                "trading_endpoint": client.base_url,
                "execute": execute,
                "clock_timestamp": clock.get("timestamp"),
                "market_open": True,
                "status": "no_valid_prepared_target_for_next_session",
                "reason": str(error),
                "orders": [],
            }
        result = run_end_of_day_transition(client, next_targets, execute=execute)
        result["cycle_phase"] = "end_of_day_transition"
        result["close_buffer_minutes"] = close_buffer_minutes
        result["next_target_session"] = next_session
        result["target_prepared_at"] = target_status.get("prepared_at")
        return result

    current_session = _session_date(clock["timestamp"])
    try:
        targets, target_status = _load_dated_targets(
            targets_path, target_status_path, expected_session=current_session, max_weight=max_weight
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {
            "trading_mode": client.mode,
            "trading_endpoint": client.base_url,
            "execute": execute,
            "clock_timestamp": clock.get("timestamp"),
            "market_open": True,
            "status": "no_valid_prepared_target_for_current_session",
            "reason": str(error),
            "orders": [],
        }
    result = run_rebalance(
        client,
        targets,
        min_order_notional=min_order_notional,
        min_weight_drift=min_weight_drift,
        # Any previous target is exited before a changed target can be bought.
        liquidate_non_target_positions=True,
        execute=execute,
    )
    result["cycle_phase"] = "intraday_rebalance"
    result["target_session"] = current_session
    result["target_prepared_at"] = target_status.get("prepared_at")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one dated-target intraday strategy cycle.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--target-status", required=True)
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--allow-live-trading", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--close-buffer-minutes", type=int, default=30)
    parser.add_argument("--max-weight", type=Decimal, default=Decimal("0.10"))
    parser.add_argument("--min-order-notional", type=Decimal, default=Decimal("1"))
    parser.add_argument("--min-weight-drift", type=Decimal, default=Decimal("0.0025"))
    args = parser.parse_args()
    if args.mode == "live" and not args.allow_live_trading:
        parser.error("--mode live requires --allow-live-trading and ALPACA_LIVE_* credentials")
    result = run_daily_cycle(
        AlpacaTradingClient.from_environment(mode=args.mode),
        targets_path=args.targets,
        target_status_path=args.target_status,
        close_buffer_minutes=args.close_buffer_minutes,
        max_weight=args.max_weight,
        min_order_notional=args.min_order_notional,
        min_weight_drift=args.min_weight_drift,
        execute=args.execute,
    )
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.mode.capitalize()} intraday cycle: {result['status']} ({len(result['orders'])} planned orders)")


if __name__ == "__main__":
    main()
