"""Build a resumable exact-endpoint IEX cache for the full Alpaca universe.

The cache intentionally retains only the native one-minute closes that the
causal five-minute Cufolio audit consumes.  This keeps the cache small without
replacing a missing one-minute endpoint with a five-minute aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import date, datetime, time as clock_time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass
from alpaca.trading.requests import GetAssetsRequest


NEW_YORK = ZoneInfo("America/New_York")
UNIVERSE_FILENAME = "alpaca_tradable_fractionable_universe.csv"
BARS_FILENAME = "iex_one_minute_exact_five_minute_endpoints.csv.gz"
PARTS_FILENAME = "download_parts.jsonl"
METADATA_FILENAME = "metadata.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _endpoint_times() -> frozenset[clock_time]:
    first = datetime.combine(date(2000, 1, 1), clock_time(9, 34))
    return frozenset(
        [clock_time(9, 29)] + [(first + timedelta(minutes=5 * offset)).time() for offset in range(78)]
    )


def _data_window(evaluation_end: date) -> tuple[date, pd.Timestamp, pd.Timestamp]:
    evaluation_start = (pd.Timestamp(evaluation_end) - pd.DateOffset(months=3)).date()
    data_start = (
        pd.Timestamp(evaluation_start, tz=NEW_YORK)
        - pd.Timedelta(days=28)
        + pd.Timedelta(hours=9, minutes=29)
    ).tz_convert("UTC")
    data_end = (pd.Timestamp(evaluation_end, tz=NEW_YORK) + pd.Timedelta(days=1)).tz_convert("UTC")
    return evaluation_start, data_start, data_end


def _calendar_windows(start: pd.Timestamp, end: pd.Timestamp, days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start
    while current < end:
        nxt = min(current + pd.Timedelta(days=days), end)
        windows.append((current, nxt))
        current = nxt
    return windows


def _completed_parts(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {
            (int(item["window"]), int(item["batch"]))
            for line in handle
            if line.strip()
            for item in [json.loads(line)]
        }


def _read_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _cache_is_complete(directory: Path, evaluation_end: date) -> bool:
    metadata = _read_metadata(directory / METADATA_FILENAME)
    return (
        bool(metadata.get("complete"))
        and metadata.get("evaluation_end") == evaluation_end.isoformat()
        and (directory / UNIVERSE_FILENAME).is_file()
        and (directory / BARS_FILENAME).is_file()
    )


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _snapshot_universe(key: str, secret: str) -> pd.DataFrame:
    client = TradingClient(key, secret, paper=True)
    assets = client.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY))
    selected = sorted((asset for asset in assets if asset.tradable and asset.fractionable), key=lambda asset: asset.symbol)
    return pd.DataFrame(
        {
            "symbol": [asset.symbol for asset in selected],
            "name": [asset.name for asset in selected],
            "exchange": [str(asset.exchange) for asset in selected],
            "status": [str(asset.status) for asset in selected],
            "tradable": [asset.tradable for asset in selected],
            "fractionable": [asset.fractionable for asset in selected],
            "shortable": [asset.shortable for asset in selected],
            "easy_to_borrow": [asset.easy_to_borrow for asset in selected],
        }
    )


def build_cache(
    directory: Path,
    *,
    evaluation_end: date,
    batch_size: int = 100,
    calendar_window_days: int = 7,
    request_pause_seconds: float = 2.0,
) -> dict[str, Any]:
    """Create an exact, resumable endpoint cache using only read-only APIs."""
    if batch_size < 1 or calendar_window_days < 1 or request_pause_seconds < 0:
        raise ValueError("batch size, window days, and request pause must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    if _cache_is_complete(directory, evaluation_end):
        metadata = _read_metadata(directory / METADATA_FILENAME)
        print(f"FULL-UNIVERSE IEX CACHE HIT | symbols={metadata['universe_symbols']:,} rows={metadata['retained_rows']:,}")
        return metadata

    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be configured")
    universe_path = directory / UNIVERSE_FILENAME
    bars_path = directory / BARS_FILENAME
    parts_path = directory / PARTS_FILENAME
    metadata_path = directory / METADATA_FILENAME
    if not universe_path.exists():
        _snapshot_universe(key, secret).to_csv(universe_path, index=False)
    universe = pd.read_csv(universe_path)
    symbols = universe["symbol"].dropna().astype(str).str.upper().tolist()
    evaluation_start, data_start, data_end = _data_window(evaluation_end)
    windows = _calendar_windows(data_start, data_end, calendar_window_days)
    batches = [symbols[offset : offset + batch_size] for offset in range(0, len(symbols), batch_size)]
    expected_parts = len(windows) * len(batches)
    metadata = {
        "research_only": True,
        "complete": False,
        "universe_query": "US equity assets filtered by tradable and fractionable",
        "universe_symbols": len(symbols),
        "evaluation_start": evaluation_start.isoformat(),
        "evaluation_end": evaluation_end.isoformat(),
        "data_start": data_start.isoformat(),
        "data_end_exclusive": data_end.isoformat(),
        "market_data_feed": "IEX",
        "source_bar_interval": "1m",
        "retained_timestamps": "09:29 plus 09:34, 09:39, ..., 15:59 America/New_York",
        "causality": "Only native exact one-minute close endpoints are retained; missing endpoints are never interpolated or substituted.",
        "batch_size": batch_size,
        "calendar_window_days": calendar_window_days,
        "request_pause_seconds": request_pause_seconds,
        "expected_request_parts": expected_parts,
    }
    _write_metadata(metadata_path, metadata)
    completed = _completed_parts(parts_path)
    client = StockHistoricalDataClient(key, secret)
    endpoints = _endpoint_times()
    header = not bars_path.exists()
    started = time.monotonic()
    for window_number, (start, end) in enumerate(windows, start=1):
        for batch_number, batch in enumerate(batches, start=1):
            part = (window_number, batch_number)
            if part in completed:
                continue
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Minute,
                start=start.to_pydatetime(),
                end=end.to_pydatetime(),
                feed=DataFeed.IEX,
            )
            for attempt in range(1, 7):
                try:
                    frame = client.get_stock_bars(request).df
                    break
                except Exception as error:
                    if attempt == 6:
                        raise RuntimeError(
                            f"IEX request failed after {attempt} attempts for window={window_number}, batch={batch_number}"
                        ) from error
                    pause = min(120.0, 15.0 * attempt)
                    print(
                        f"IEX RATE/NETWORK RETRY | window={window_number}/{len(windows)} batch={batch_number}/{len(batches)} "
                        f"attempt={attempt} pause={pause:.0f}s error={type(error).__name__}",
                        flush=True,
                    )
                    time.sleep(pause)
            if frame.empty:
                retained = pd.DataFrame(columns=["timestamp", "symbol", "close"])
            else:
                retained = frame.reset_index().loc[:, ["timestamp", "symbol", "close"]].copy()
                retained["timestamp"] = pd.to_datetime(retained["timestamp"], utc=True)
                retained["symbol"] = retained["symbol"].astype(str).str.upper().str.strip()
                retained["close"] = pd.to_numeric(retained["close"], errors="coerce")
                local_times = retained["timestamp"].dt.tz_convert(NEW_YORK).dt.time
                retained = retained.loc[
                    local_times.isin(endpoints) & retained["close"].gt(0),
                    ["timestamp", "symbol", "close"],
                ].drop_duplicates(["timestamp", "symbol"], keep="last").sort_values(["symbol", "timestamp"])
            retained.to_csv(bars_path, mode="a", index=False, header=header, compression="gzip")
            header = False
            record = {
                "window": window_number,
                "batch": batch_number,
                "symbols_requested": len(batch),
                "raw_bars": int(len(frame)),
                "retained_exact_endpoints": int(len(retained)),
                "symbols_with_retained_endpoint": int(retained["symbol"].nunique()),
                "elapsed_seconds": round(time.monotonic() - started, 1),
            }
            with parts_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"IEX CACHE | window={window_number:02d}/{len(windows)} batch={batch_number:02d}/{len(batches)} "
                f"raw={len(frame):,} endpoints={len(retained):,} elapsed={record['elapsed_seconds']:.1f}s",
                flush=True,
            )
            time.sleep(request_pause_seconds)
    logs = [json.loads(line) for line in parts_path.read_text(encoding="utf-8").splitlines() if line]
    completed_now = {(int(item["window"]), int(item["batch"])) for item in logs}
    if len(completed_now) != expected_parts:
        raise RuntimeError(f"cache is incomplete: {len(completed_now)} of {expected_parts} request parts completed")
    metadata.update(
        {
            "complete": True,
            "retained_rows": int(sum(item["retained_exact_endpoints"] for item in logs)),
            "raw_rows_fetched": int(sum(item["raw_bars"] for item in logs)),
            "completed_request_parts": len(completed_now),
            "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "universe_sha256": _sha256(universe_path),
            "endpoint_data_sha256": _sha256(bars_path),
        }
    )
    _write_metadata(metadata_path, metadata)
    print(f"FULL-UNIVERSE IEX CACHE COMPLETE | symbols={len(symbols):,} rows={metadata['retained_rows']:,}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a full-Alpaca exact one-minute endpoint cache for research.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--evaluation-end", required=True, help="Last completed New York session (YYYY-MM-DD)")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--calendar-window-days", type=int, default=7)
    parser.add_argument("--request-pause-seconds", type=float, default=2.0)
    args = parser.parse_args()
    metadata = build_cache(
        Path(args.cache_dir),
        evaluation_end=date.fromisoformat(args.evaluation_end),
        batch_size=args.batch_size,
        calendar_window_days=args.calendar_window_days,
        request_pause_seconds=args.request_pause_seconds,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
