"""Optional Alpaca minute-bar downloader; it has no order-routing capability."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


def load_symbols(path: str | Path) -> list[str]:
    universe = pd.read_csv(path)
    if "symbol" not in universe.columns:
        raise ValueError(f"{path} must contain a 'symbol' column")
    symbols = universe["symbol"].dropna().astype(str).str.upper().str.strip()
    symbols = symbols[symbols.ne("")].drop_duplicates().to_list()
    if not symbols:
        raise ValueError("the requested universe has no symbols")
    return symbols


def _utc_datetime(value: str | datetime) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def download_minute_bars(
    symbols: list[str], start: str | datetime, end: str | datetime, *, batch_size: int = 100
) -> pd.DataFrame:
    """Download one-minute stock bars using Alpaca's market-data API only."""
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set; no request was sent")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(key, secret)
    frames: list[pd.DataFrame] = []
    for offset in range(0, len(symbols), batch_size):
        batch = symbols[offset : offset + batch_size]
        request = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Minute,
            start=_utc_datetime(start),
            end=_utc_datetime(end),
        )
        frame = client.get_stock_bars(request).df
        if frame.empty:
            continue
        frames.append(frame.reset_index().loc[:, ["timestamp", "symbol", "close"]])
    if not frames:
        return pd.DataFrame(columns=["timestamp", "symbol", "close"])
    result = pd.concat(frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    return result.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Alpaca one-minute stock bars; never submits orders.")
    parser.add_argument("--symbols", required=True, help="CSV containing a symbol column")
    parser.add_argument("--start", required=True, help="UTC start date/time")
    parser.add_argument("--end", required=True, help="UTC end date/time")
    parser.add_argument("--output", required=True, help="minute-bar CSV output path")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    bars = download_minute_bars(load_symbols(args.symbols), args.start, args.end, batch_size=args.batch_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(output, index=False)
    count = bars["symbol"].nunique() if not bars.empty else 0
    print(f"Wrote {len(bars):,} one-minute bars for {count} symbols to {output}")


if __name__ == "__main__":
    main()
