"""Credential-free Yahoo Finance downloader for short-horizon research data."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .alpaca import load_symbols


def _utc_timestamp(value: str | datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _close_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "close"])
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol not in raw.columns.get_level_values(0):
            return pd.DataFrame(columns=["timestamp", "symbol", "close"])
        frame = raw[symbol].copy()
    else:
        frame = raw.copy()
    if "Close" not in frame.columns:
        return pd.DataFrame(columns=["timestamp", "symbol", "close"])
    result = frame.loc[:, ["Close"]].rename(columns={"Close": "close"}).dropna().reset_index()
    timestamp_column = "Datetime" if "Datetime" in result.columns else "Date"
    result = result.rename(columns={timestamp_column: "timestamp"})
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result["symbol"] = symbol
    return result.loc[:, ["timestamp", "symbol", "close"]]


def download_minute_bars(
    symbols: list[str],
    start: str | datetime,
    end: str | datetime,
    *,
    batch_size: int = 25,
    retries: int = 3,
) -> pd.DataFrame:
    """Download regular-session one-minute bars without sending any orders.

    Yahoo Finance limits one-minute retrieval to a short recent window. The
    explicit 60-day guard prevents a partial response being mistaken for a
    complete historical S&P 500 backtest.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    start_at, end_at = _utc_timestamp(start), _utc_timestamp(end)
    if start_at >= end_at:
        raise ValueError("start must precede end")
    if end_at - start_at > pd.Timedelta(days=60):
        raise ValueError("Yahoo Finance one-minute retrieval is limited to the most recent 60 days")

    import yfinance as yf

    frames: list[pd.DataFrame] = []
    for offset in range(0, len(symbols), batch_size):
        batch = symbols[offset : offset + batch_size]
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                raw = yf.download(
                    tickers=batch,
                    start=start_at.to_pydatetime(),
                    end=end_at.to_pydatetime(),
                    interval="1m",
                    auto_adjust=True,
                    prepost=False,
                    group_by="ticker",
                    threads=False,
                    progress=False,
                    timeout=30,
                )
                frames.extend(_close_frame(raw, symbol) for symbol in batch)
                break
            except Exception as error:  # API/network errors are retried, but never hidden.
                last_error = error
                if attempt + 1 < retries:
                    time.sleep(2**attempt)
        else:
            raise RuntimeError(f"Yahoo Finance request failed for batch {batch}") from last_error
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "close"])
    return result.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Yahoo Finance one-minute bars; never submits orders.")
    parser.add_argument("--symbols", required=True, help="CSV containing a symbol column")
    parser.add_argument("--start", required=True, help="UTC start date/time; maximum 60 days before end")
    parser.add_argument("--end", required=True, help="UTC end date/time")
    parser.add_argument("--output", required=True, help="minute-bar CSV output path")
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()
    bars = download_minute_bars(load_symbols(args.symbols), args.start, args.end, batch_size=args.batch_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(output, index=False)
    count = bars["symbol"].nunique() if not bars.empty else 0
    print(f"Wrote {len(bars):,} one-minute bars for {count} symbols to {output}")


if __name__ == "__main__":
    main()
