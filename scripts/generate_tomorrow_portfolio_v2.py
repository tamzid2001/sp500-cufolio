"""Generate a research-only long-only portfolio from completed Alpaca IEX bars."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from cufolio_cpu.intraday_forward_v2 import run_forward_research
from cufolio_cpu.intraday_selection import download_fifteen_minute_bars
from cufolio_cpu.universe import current_sp500_universe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/tomorrow_portfolio_v2")
    parser.add_argument("--lookback-calendar-days", type=int, default=90)
    parser.add_argument("--horizon-minutes", type=int, default=500)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-weight", type=float, default=0.10)
    args = parser.parse_args()

    api_key = os.getenv("ALPACA_PAPER_API_KEY") or os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_PAPER_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("Alpaca paper market-data credentials are not configured")

    universe = current_sp500_universe()
    symbol_column = "source_symbol" if "source_symbol" in universe.columns else "symbol"
    symbols = universe[symbol_column].astype(str).str.upper().drop_duplicates().to_list()
    bars = download_fifteen_minute_bars(
        symbols,
        api_key=api_key,
        secret_key=secret_key,
        end=datetime.now(timezone.utc),
        lookback_calendar_days=args.lookback_calendar_days,
    )
    result = run_forward_research(
        bars,
        interval="15m",
        horizon_minutes=args.horizon_minutes,
        top_n=args.top_n,
        max_weight=args.max_weight,
        min_sessions=20,
    )
    if not result.status.get("model_run"):
        raise RuntimeError(f"model did not produce a portfolio: {result.status.get('reason')}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.portfolio.to_csv(output_dir / "tomorrow_long_only_portfolio.csv", index=False)
    result.validation.to_csv(output_dir / "walk_forward_validation.csv", index=False)
    (output_dir / "model_status.json").write_text(json.dumps(result.status, indent=2) + "\n")
    print(result.portfolio.to_string(index=False))
    print(json.dumps(result.status, indent=2))


if __name__ == "__main__":
    main()
