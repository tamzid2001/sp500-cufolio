"""Measure one read-only concurrent latest-IEX-minute refresh for the cached universe."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from cufolio_cpu.alpaca import download_latest_iex_minute_bars
from cufolio_cpu.universe import cached_alpaca_tradable_fractionable_universe


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a full-Alpaca latest-minute IEX refresh.")
    parser.add_argument("--universe-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_workers < 1:
        parser.error("--batch-size and --max-workers must be positive")

    universe = cached_alpaca_tradable_fractionable_universe(args.universe_cache)
    started = time.monotonic()
    bars = download_latest_iex_minute_bars(
        universe["symbol"].tolist(), batch_size=args.batch_size, max_workers=args.max_workers
    )
    elapsed = time.monotonic() - started
    timestamps = pd.to_datetime(bars.get("timestamp", pd.Series(dtype="object")), utc=True, errors="coerce").dropna()
    summary = {
        "research_only": True,
        "source": "Alpaca latest IEX minute-bar endpoint",
        "universe_symbols": int(len(universe)),
        "batches": int((len(universe) + args.batch_size - 1) // args.batch_size),
        "batch_size": args.batch_size,
        "max_workers": args.max_workers,
        "bars_returned": int(len(bars)),
        "symbols_returned": int(bars["symbol"].nunique()) if not bars.empty else 0,
        "elapsed_seconds": round(elapsed, 3),
        "oldest_returned_timestamp": timestamps.min().isoformat() if not timestamps.empty else None,
        "newest_returned_timestamp": timestamps.max().isoformat() if not timestamps.empty else None,
        "stale_bar_rule": "The live runner uses only a bar whose left-labelled timestamp equals its completed-minute boundary; older latest bars are discarded.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
