"""Prepare tomorrow's full-S&P-500 target before today's market close."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .daily_selection import current_sp500_daily_targets
from .paper_rebalance import AlpacaTradingClient

NEW_YORK = ZoneInfo("America/New_York")


def _session_date(value: object) -> str:
    return pd.Timestamp(value).tz_convert(NEW_YORK).date().isoformat()


def prepare_daily_target(
    client: AlpacaTradingClient,
    *,
    lookback_calendar_days: int = 180,
    lookback_sessions: int = 90,
    candidate_count: int = 50,
    top_n: int = 20,
    max_weight: Decimal = Decimal("0.10"),
    scenario_count: int = 2_000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build a target for Alpaca's next market session, without submitting orders."""
    clock = client.get_clock()
    if not clock.get("is_open"):
        return pd.DataFrame(columns=["symbol", "target_weight"]), {
            "status": "market_closed",
            "trading_mode": client.mode,
            "prepared_at": datetime.now(NEW_YORK).isoformat(),
        }
    api_key, secret_key = client.market_data_credentials()
    selection = current_sp500_daily_targets(
        api_key=api_key,
        secret_key=secret_key,
        # Excludes today's unfinished daily bar, making this a true pre-close target.
        as_of_session=str(clock["timestamp"]),
        lookback_calendar_days=lookback_calendar_days,
        lookback_sessions=lookback_sessions,
        candidate_count=candidate_count,
        top_n=top_n,
        max_weight=float(max_weight),
        scenario_count=scenario_count,
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
    parser = argparse.ArgumentParser(description="Prepare tomorrow's S&P 500 Mean-CVaR target.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--lookback-calendar-days", type=int, default=180)
    parser.add_argument("--lookback-sessions", type=int, default=90)
    parser.add_argument("--candidate-count", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-weight", type=Decimal, default=Decimal("0.10"))
    parser.add_argument("--scenario-count", type=int, default=2_000)
    args = parser.parse_args()
    targets, status = prepare_daily_target(
        AlpacaTradingClient.from_environment(mode=args.mode),
        lookback_calendar_days=args.lookback_calendar_days,
        lookback_sessions=args.lookback_sessions,
        candidate_count=args.candidate_count,
        top_n=args.top_n,
        max_weight=args.max_weight,
        scenario_count=args.scenario_count,
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
