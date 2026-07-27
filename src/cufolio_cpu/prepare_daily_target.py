"""Prepare tomorrow's full-S&P-500 target before today's market close."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .intraday_selection import current_sp500_forward_targets
from .paper_rebalance import AlpacaTradingClient

NEW_YORK = ZoneInfo("America/New_York")


def _session_date(value: object) -> str:
    return pd.Timestamp(value).tz_convert(NEW_YORK).date().isoformat()


def prepare_daily_target(
    client: AlpacaTradingClient,
    *,
    lookback_calendar_days: int = 60,
    horizon_minutes: int = 500,
    top_n: int = 20,
    max_weight: Decimal = Decimal("0.10"),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build tomorrow's target using the existing purged 15-minute model."""
    clock = client.get_clock()
    if not clock.get("is_open"):
        return pd.DataFrame(columns=["symbol", "target_weight"]), {
            "status": "market_closed",
            "trading_mode": client.mode,
            "prepared_at": datetime.now(NEW_YORK).isoformat(),
        }
    api_key, secret_key = client.market_data_credentials()
    selection = current_sp500_forward_targets(
        api_key=api_key,
        secret_key=secret_key,
        # The downloader itself excludes an unfinished 15-minute bar.
        as_of=str(clock["timestamp"]),
        lookback_calendar_days=lookback_calendar_days,
        horizon_minutes=horizon_minutes,
        top_n=top_n,
        max_weight=float(max_weight),
    )
    status = dict(selection.status)
    status.update(
        {
            "prepared_at": datetime.now(NEW_YORK).isoformat(),
            "signal_cutoff_session": _session_date(clock["timestamp"]),
            "target_session": _session_date(clock["next_open"]),
            "trading_mode": client.mode,
        }
    )
    return selection.targets.loc[:, ["symbol", "target_weight"]], status


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tomorrow's 15-minute forward-model S&P 500 target.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--lookback-calendar-days", type=int, default=60)
    parser.add_argument("--horizon-minutes", type=int, default=500)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-weight", type=Decimal, default=Decimal("0.10"))
    args = parser.parse_args()
    targets, status = prepare_daily_target(
        AlpacaTradingClient.from_environment(mode=args.mode),
        lookback_calendar_days=args.lookback_calendar_days,
        horizon_minutes=args.horizon_minutes,
        top_n=args.top_n,
        max_weight=args.max_weight,
    )
    if targets.empty:
        print("No daily target prepared because the market is closed")
        return
    output = Path(args.output)
    status_path = Path(args.status)
    output.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output, index=False)
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(f"Prepared {len(targets)} long-only targets for {status['target_session']}")


if __name__ == "__main__":
    main()
