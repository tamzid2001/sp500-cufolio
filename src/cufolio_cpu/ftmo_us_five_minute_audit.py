"""Causal five-minute FTMO US simulated-asset research audit.

The FTMO US account is a simulated environment.  Its current asset list is
published by FTMO and is verified at download time.  Dukascopy is used only as
an independent historical *proxy* feed: it does not reproduce FTMO US quotes,
fills, swaps, contract rolls, volume limits, or commissions.  This module
therefore reports a price-return research audit, never an FTMO execution or
net-performance result.

Native one-minute Dukascopy ASK and BID candles are downloaded and reduced to
five-minute endpoints.  Each decision observes the minute ending immediately
before the decision; all fitting labels end strictly before that decision.
The realized executable proxy return buys at the Dukascopy ASK endpoint and
sells at the next Dukascopy BID endpoint, so the available proxy spread is not
silently ignored.
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

FTMO_US_SYMBOLS_URL = "https://ftmo.oanda.com/wp-json/ftmo/symbols"
DUKASCOPY_CANDLE_URL = (
    "https://www.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/"
    "{side}_candles_min_1.bi5"
)
DUKASCOPY_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; sp500-cufolio-research/1.0)",
    "Accept": "*/*",
}
M1_CANDLE_STRUCT = struct.Struct("!IIIIIf")
FIVE_MINUTES = pd.Timedelta(5, unit="min")
ONE_MINUTE = pd.Timedelta(1, unit="min")
TIMEFRAME_INTERVALS = {
    "M5": FIVE_MINUTES,
    "H1": pd.Timedelta(1, unit="h"),
    "H4": pd.Timedelta(4, unit="h"),
    "D1": pd.Timedelta(1, unit="D"),
}
TIMEFRAME_FREQUENCIES = {
    "M5": "5min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}
MULTI_TIMEFRAME_PARAMETERS = {
    "H1": {"lookback_windows": 720, "min_training_windows": 250},
    "H4": {"lookback_windows": 360, "min_training_windows": 100},
    "D1": {"lookback_windows": 60, "min_training_windows": 20},
}
MAPPING_COLUMNS = {
    "ftmo_symbol",
    "asset_class",
    "dukascopy_symbol",
    "price_divisor",
    "proxy_note",
}


def _progress(stage: str, **values: object) -> None:
    """Emit a compact, flushed line suitable for live GitHub Actions logs."""
    details = " | ".join(f"{key}={value}" for key, value in values.items())
    print(f"FTMO US PROXY AUDIT {stage} | {details}", flush=True)


@dataclass(frozen=True)
class DownloadResult:
    """The parsed result of one native Dukascopy BID or ASK daily file."""

    ftmo_symbol: str
    side: str
    day: date
    status: str
    rows: list[tuple[pd.Timestamp, float]]
    detail: str | None = None


@dataclass(frozen=True)
class FtmoUsAuditResult:
    """Every forecast, every selected asset, and disclosure-rich summary data."""

    ledger: pd.DataFrame
    holdings: pd.DataFrame
    summary: dict[str, Any]


def _default_mapping_path() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "ftmo_us_dukascopy_mapping.csv"


def _date(value: str | date | datetime | pd.Timestamp) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _timeframe(value: str) -> str:
    timeframe = str(value).upper().strip()
    if timeframe not in TIMEFRAME_INTERVALS:
        raise ValueError(f"unsupported timeframe {value!r}; expected one of {sorted(TIMEFRAME_INTERVALS)}")
    return timeframe


def load_ftmo_us_mapping(path: str | Path | None = None) -> pd.DataFrame:
    """Read the reviewed FTMO-US-to-Dukascopy proxy mapping and validate it."""
    mapping_path = Path(path) if path is not None else _default_mapping_path()
    mapping = pd.read_csv(mapping_path)
    missing = MAPPING_COLUMNS.difference(mapping.columns)
    if missing:
        raise ValueError(f"FTMO US mapping is missing columns: {sorted(missing)}")
    mapping = mapping.loc[:, sorted(MAPPING_COLUMNS)].copy()
    mapping["ftmo_symbol"] = mapping["ftmo_symbol"].astype(str).str.upper().str.strip()
    mapping["dukascopy_symbol"] = mapping["dukascopy_symbol"].astype(str).str.upper().str.strip()
    mapping["asset_class"] = mapping["asset_class"].astype(str).str.lower().str.strip()
    mapping["price_divisor"] = pd.to_numeric(mapping["price_divisor"], errors="raise")
    if mapping["ftmo_symbol"].duplicated().any() or mapping["dukascopy_symbol"].eq("").any():
        raise ValueError("FTMO US mapping must have unique non-empty FTMO and Dukascopy symbols")
    if (~np.isfinite(mapping["price_divisor"]) | mapping["price_divisor"].le(0)).any():
        raise ValueError("FTMO US mapping has an invalid Dukascopy price divisor")
    return mapping.sort_values("ftmo_symbol").reset_index(drop=True)


def fetch_ftmo_us_manifest(*, timeout_seconds: int = 30) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch FTMO's public US simulated-asset manifest and retain active US symbols."""
    response = requests.get(FTMO_US_SYMBOLS_URL, headers=DUKASCOPY_HEADERS, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    symbols = payload.get("data", {}).get("symbols", []) if isinstance(payload, dict) else []
    if not isinstance(symbols, list):
        raise ValueError("FTMO US symbol endpoint returned an invalid symbols payload")
    rows = [
        item
        for item in symbols
        if isinstance(item, dict)
        and str(item.get("region", "")).lower() == "us"
        and bool(item.get("active"))
        and str(item.get("code", "")).upper().endswith(".SIM")
    ]
    manifest = pd.DataFrame(rows)
    if manifest.empty or "code" not in manifest:
        raise ValueError("FTMO US symbol endpoint returned no active simulated assets")
    manifest["code"] = manifest["code"].astype(str).str.upper().str.strip()
    manifest = manifest.drop_duplicates("code").sort_values("code").reset_index(drop=True)
    metadata = {
        "source_url": FTMO_US_SYMBOLS_URL,
        "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "active_us_simulated_asset_count": int(len(manifest)),
        "asset_class_counts": {
            str(name): int(count)
            for name, count in manifest.get("assetClass", pd.Series(dtype="object")).value_counts().sort_index().items()
        },
    }
    return manifest, metadata


def select_verified_assets(
    mapping: pd.DataFrame,
    manifest: pd.DataFrame,
    requested_symbols: str = "all",
) -> pd.DataFrame:
    """Select requested mapped assets, failing closed for unknown active FTMO US assets.

    ``all`` means every currently active FTMO US asset in the official manifest.
    A newly listed symbol needs an explicit reviewed proxy mapping before an
    audit can claim it was included or excluded.
    """
    active = set(manifest["code"].astype(str).str.upper())
    mapped = set(mapping["ftmo_symbol"])
    unmapped = sorted(active.difference(mapped))
    if unmapped:
        raise ValueError(
            "active FTMO US assets have no reviewed Dukascopy proxy mapping: " + ", ".join(unmapped)
        )
    requested = str(requested_symbols).strip()
    if not requested or requested.lower() == "all":
        selected_codes = active
    else:
        selected_codes = {item.strip().upper() for item in requested.split(",") if item.strip()}
        unknown = sorted(selected_codes.difference(active))
        if unknown:
            raise ValueError("requested symbols are not active FTMO US simulated assets: " + ", ".join(unknown))
    selected = mapping[mapping["ftmo_symbol"].isin(selected_codes)].copy()
    if selected.empty:
        raise ValueError("no FTMO US symbols remain after applying the requested universe")
    asset_class = manifest.loc[:, [column for column in ("code", "assetClass", "name") if column in manifest]].copy()
    selected = selected.merge(asset_class, left_on="ftmo_symbol", right_on="code", how="left").drop(columns="code")
    return selected.sort_values("ftmo_symbol").reset_index(drop=True)


def _decompress_bi5(compressed: bytes) -> bytes:
    """Decompress one or more LZMA streams stored in a Dukascopy BI5 file."""
    if not compressed:
        return b""
    output: list[bytes] = []
    remaining = compressed
    while remaining:
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_AUTO)
        chunk = decoder.decompress(remaining)
        output.append(chunk)
        remaining = decoder.unused_data
        if not remaining:
            break
        if not decoder.eof:
            raise lzma.LZMAError("Dukascopy BI5 stream ended before its end marker")
    return b"".join(output)


def _parse_native_m1_close(
    compressed: bytes,
    *,
    source_day: date,
    price_divisor: float,
) -> list[tuple[pd.Timestamp, float]]:
    """Decode native M1 candles and retain only real, non-forward-filled closes."""
    raw = _decompress_bi5(compressed)
    if len(raw) % M1_CANDLE_STRUCT.size:
        raise ValueError("Dukascopy native M1 file has a partial candle record")
    base = pd.Timestamp(datetime.combine(source_day, datetime.min.time()), tz="UTC")
    records: list[tuple[pd.Timestamp, float]] = []
    previous: tuple[int, int, int, int, float] | None = None
    for offset, opening, closing, low, high, volume in M1_CANDLE_STRUCT.iter_unpack(raw):
        raw_values = (opening, closing, low, high, volume)
        if opening == 0 and closing == 0 and low == 0 and high == 0:
            previous = raw_values
            continue
        # Dukascopy writes identical OHLCV candles through many market-closed
        # minutes. Keeping them would invent tradable five-minute endpoints.
        if raw_values == previous:
            continue
        previous = raw_values
        # Pass the seconds unit explicitly.  With newer NumPy/Pandas versions,
        # the keyword form can still route through a deprecated generic unit.
        timestamp = base + pd.Timedelta(int(offset), unit="s")
        records.append((timestamp, float(closing / price_divisor)))
    return records


def _download_native_m1_day(
    ftmo_symbol: str,
    dukascopy_symbol: str,
    price_divisor: float,
    side: str,
    source_day: date,
    *,
    retries: int,
    timeout_seconds: int,
) -> DownloadResult:
    url = DUKASCOPY_CANDLE_URL.format(
        symbol=dukascopy_symbol,
        year=source_day.year,
        month=source_day.month - 1,
        day=source_day.day,
        side=side,
    )
    last_detail: str | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=DUKASCOPY_HEADERS, timeout=timeout_seconds)
            if response.status_code == 404:
                return DownloadResult(ftmo_symbol, side, source_day, "missing", [])
            if response.status_code == 200 and response.content:
                rows = _parse_native_m1_close(
                    response.content, source_day=source_day, price_divisor=price_divisor
                )
                return DownloadResult(ftmo_symbol, side, source_day, "ok", rows)
            last_detail = f"HTTP {response.status_code}; bytes={len(response.content)}"
        except (requests.RequestException, lzma.LZMAError, ValueError) as error:
            last_detail = f"{type(error).__name__}: {error}"
        if attempt + 1 < retries:
            time.sleep(0.35 * (attempt + 1))
    return DownloadResult(ftmo_symbol, side, source_day, "error", [], last_detail)


def download_dukascopy_ftmo_us_m1(
    assets: pd.DataFrame,
    *,
    start: str | date,
    end: str | date,
    workers: int = 8,
    retries: int = 3,
    timeout_seconds: int = 30,
    on_m1_chunk: Callable[[pd.DataFrame], None] | None = None,
    collect_m1: bool = True,
    progress_every_files: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download native M1 BID/ASK closes for an FTMO US proxy asset set.

    The date interval is inclusive.  It intentionally probes every calendar
    day because BTCUSD may trade when the FX and index proxy feeds do not.
    """
    start_day, end_day = _date(start), _date(end)
    if start_day > end_day:
        raise ValueError("download start must not be after download end")
    if not 1 <= workers <= 24:
        raise ValueError("workers must be between 1 and 24")
    if retries < 1:
        raise ValueError("retries must be positive")
    if progress_every_files < 0:
        raise ValueError("progress_every_files must be non-negative")
    days = [value.date() for value in pd.date_range(start_day, end_day, freq="D")]
    status_counts: dict[str, int] = {}
    failures: list[dict[str, str | None]] = []
    m1_bid_ask_rows = 0
    symbols_with_data: set[str] = set()
    collected_frames: list[pd.DataFrame] = []
    total_files = len(days) * len(assets) * 2
    completed_files = 0
    download_started = time.monotonic()
    if progress_every_files:
        _progress(
            "DOWNLOAD START",
            assets=len(assets),
            calendar_days=len(days),
            files=total_files,
            workers=workers,
        )
    # Process each calendar day independently. A full 49-asset three-month
    # input has millions of M1 closes, so holding all parsed Python tuples
    # until the final request would needlessly exhaust an Actions runner.
    for day_number, source_day in enumerate(days, start=1):
        day_jobs = [
            (row.ftmo_symbol, row.dukascopy_symbol, float(row.price_divisor), side, source_day)
            for row in assets.itertuples(index=False)
            for side in ("BID", "ASK")
        ]
        day_results: list[DownloadResult] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _download_native_m1_day,
                    *job,
                    retries=retries,
                    timeout_seconds=timeout_seconds,
                )
                for job in day_jobs
            ]
            for future in as_completed(futures):
                result = future.result()
                day_results.append(result)
                completed_files += 1
                status_counts[result.status] = status_counts.get(result.status, 0) + 1
                if result.status == "error":
                    failures.append(
                        {
                            "ftmo_symbol": result.ftmo_symbol,
                            "side": result.side,
                            "date": result.day.isoformat(),
                            "detail": result.detail,
                        }
                    )
                if (
                    progress_every_files
                    and len(day_results) % progress_every_files == 0
                    and len(day_results) < len(day_jobs)
                ):
                    _progress(
                        "DOWNLOAD PROGRESS",
                        day=f"{day_number}/{len(days)}",
                        date=source_day.isoformat(),
                        day_files=f"{len(day_results)}/{len(day_jobs)}",
                        total_files=f"{completed_files}/{total_files}",
                        ok=status_counts.get("ok", 0),
                        missing=status_counts.get("missing", 0),
                        errors=status_counts.get("error", 0),
                        elapsed=f"{time.monotonic() - download_started:.1f}s",
                    )
        close_rows = [
            {"timestamp": timestamp, "ftmo_symbol": result.ftmo_symbol, "side": result.side, "close": close}
            for result in day_results
            for timestamp, close in result.rows
        ]
        long = pd.DataFrame(close_rows, columns=["timestamp", "ftmo_symbol", "side", "close"])
        if long.empty:
            day_merged = pd.DataFrame(
                columns=["timestamp", "ftmo_symbol", "bid_close", "ask_close", "mid_close"]
            )
        else:
            long["timestamp"] = pd.to_datetime(long["timestamp"], utc=True)
            long = long.drop_duplicates(["timestamp", "ftmo_symbol", "side"], keep="last")
            day_merged = (
                long.pivot(index=["timestamp", "ftmo_symbol"], columns="side", values="close")
                .rename(columns={"ASK": "ask_close", "BID": "bid_close"})
                .reset_index()
            )
            for column in ("bid_close", "ask_close"):
                if column not in day_merged:
                    day_merged[column] = np.nan
            day_merged["mid_close"] = (day_merged["bid_close"] + day_merged["ask_close"]) / 2.0
            day_merged = day_merged.sort_values(["timestamp", "ftmo_symbol"]).reset_index(drop=True)
        valid = day_merged[["bid_close", "ask_close"]].notna().all(axis=1) if not day_merged.empty else pd.Series(dtype=bool)
        valid_rows = int(valid.sum())
        m1_bid_ask_rows += valid_rows
        if not day_merged.empty:
            symbols_with_data.update(day_merged.loc[valid, "ftmo_symbol"].unique())
            if on_m1_chunk is not None:
                on_m1_chunk(day_merged)
            if collect_m1:
                collected_frames.append(day_merged)
        if progress_every_files:
            _progress(
                "DOWNLOAD DAY COMPLETE",
                day=f"{day_number}/{len(days)}",
                date=source_day.isoformat(),
                total_files=f"{completed_files}/{total_files}",
                ok=status_counts.get("ok", 0),
                missing=status_counts.get("missing", 0),
                errors=status_counts.get("error", 0),
                day_exact_bid_ask_rows=valid_rows,
                total_exact_bid_ask_rows=m1_bid_ask_rows,
                elapsed=f"{time.monotonic() - download_started:.1f}s",
            )
    merged = (
        pd.concat(collected_frames, ignore_index=True).sort_values(["timestamp", "ftmo_symbol"]).reset_index(drop=True)
        if collected_frames
        else pd.DataFrame(columns=["timestamp", "ftmo_symbol", "bid_close", "ask_close", "mid_close"])
    )
    metadata = {
        "data_provider": "Dukascopy native M1 candles",
        "price_sides": ["BID", "ASK"],
        "download_start": start_day.isoformat(),
        "download_end": end_day.isoformat(),
        "assets_requested": int(len(assets)),
        "calendar_days_requested": int(len(days)),
        "files_requested": int(len(days) * len(assets) * 2),
        "file_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "failed_files": failures[:200],
        "failed_file_count": int(len(failures)),
        "m1_bid_ask_rows": m1_bid_ask_rows,
        "symbols_with_bid_ask_data": int(len(symbols_with_data)),
        "m1_rows_retained_in_memory": bool(collect_m1),
        "feed_caveat": (
            "Dukascopy is a historical proxy feed, not FTMO US. BID/ASK proxy prices do not reproduce "
            "FTMO simulated quotes, fills, rollovers, limits, commissions, swaps, or slippage."
        ),
    }
    if progress_every_files:
        _progress(
            "DOWNLOAD COMPLETE",
            files=f"{completed_files}/{total_files}",
            ok=status_counts.get("ok", 0),
            missing=status_counts.get("missing", 0),
            errors=status_counts.get("error", 0),
            exact_bid_ask_rows=m1_bid_ask_rows,
            symbols_with_data=len(symbols_with_data),
            elapsed=f"{time.monotonic() - download_started:.1f}s",
        )
    return merged, metadata


def native_m1_to_timeframe_quotes(minute_closes: pd.DataFrame, *, timeframe: str) -> pd.DataFrame:
    """Retain timeframe buckets with exact final M1 BID and ASK endpoints.

    Minute bars are left-labelled.  Thus the close of the 00:04 M1 bar is the
    first M5 decision price, timestamped 00:05.  The same rule applies to H1,
    H4, and D1: missing final M1 endpoints cannot be quietly replaced.
    """
    normalized_timeframe = _timeframe(timeframe)
    interval = TIMEFRAME_INTERVALS[normalized_timeframe]
    frequency = TIMEFRAME_FREQUENCIES[normalized_timeframe]
    required = {"timestamp", "ftmo_symbol", "bid_close", "ask_close"}
    missing = required.difference(minute_closes.columns)
    if missing:
        raise ValueError(f"minute closes are missing required columns: {sorted(missing)}")
    clean = minute_closes.loc[:, ["timestamp", "ftmo_symbol", "bid_close", "ask_close"]].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")
    clean["ftmo_symbol"] = clean["ftmo_symbol"].astype(str).str.upper().str.strip()
    clean["bid_close"] = pd.to_numeric(clean["bid_close"], errors="coerce")
    clean["ask_close"] = pd.to_numeric(clean["ask_close"], errors="coerce")
    clean = clean.dropna(subset=["timestamp", "bid_close", "ask_close"])
    clean = clean[(clean["ftmo_symbol"] != "") & (clean["bid_close"] > 0) & (clean["ask_close"] > 0)]
    clean = clean.drop_duplicates(["timestamp", "ftmo_symbol"], keep="last").sort_values(
        ["ftmo_symbol", "timestamp"]
    )
    if clean.empty:
        return pd.DataFrame(columns=["timestamp", "ftmo_symbol", "bid_close", "ask_close", "mid_close"])
    clean["decision_timestamp"] = clean["timestamp"].dt.floor(frequency) + interval
    last = clean.groupby(["ftmo_symbol", "decision_timestamp"], as_index=False).tail(1).copy()
    exact_endpoint = last["timestamp"].eq(last["decision_timestamp"] - ONE_MINUTE)
    quotes = last.loc[exact_endpoint, ["decision_timestamp", "ftmo_symbol", "bid_close", "ask_close"]].rename(
        columns={"decision_timestamp": "timestamp"}
    )
    quotes["mid_close"] = (quotes["bid_close"] + quotes["ask_close"]) / 2.0
    return quotes.sort_values(["timestamp", "ftmo_symbol"]).reset_index(drop=True)


def native_m1_to_five_minute_quotes(minute_closes: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible M5 wrapper for ``native_m1_to_timeframe_quotes``."""
    return native_m1_to_timeframe_quotes(minute_closes, timeframe="M5")


def _sanitize_weights(weights: pd.Series, *, max_weight: float) -> pd.Series:
    """Remove numerical dust while retaining a fully invested long-only portfolio."""
    clean = weights.clip(lower=0.0, upper=max_weight).copy()
    clean.loc[clean < 1e-8] = 0.0
    residual = float(1.0 - clean.sum())
    while abs(residual) > 1e-12:
        if residual > 0:
            capacity = max_weight - clean
            candidates = capacity[capacity > 1e-12]
            if candidates.empty:
                raise RuntimeError("cannot restore capped portfolio feasibility")
            addition = min(residual / len(candidates), float(candidates.min()))
            clean.loc[candidates.index] += addition
        else:
            candidates = clean[clean > 1e-12]
            if candidates.empty:
                raise RuntimeError("cannot restore non-negative portfolio feasibility")
            reduction = min(-residual / len(candidates), float(candidates.min()))
            clean.loc[candidates.index] -= reduction
        residual = float(1.0 - clean.sum())
    return clean[clean > 0]


def _previous_interval(panel: pd.DataFrame, interval: pd.Timedelta) -> pd.DataFrame:
    previous = panel.copy()
    previous.index = previous.index + interval
    return previous.reindex(panel.index)


def _select_causal_portfolio(
    expected: pd.Series,
    training_counts: pd.Series,
    current_mid: pd.Series,
    *,
    training_end: pd.Timestamp | pd.NaT,
    top_n: int,
    max_weight: float,
    min_training_windows: int,
) -> tuple[pd.Series | None, pd.Series | None, dict[str, Any]]:
    required_candidates = int(math.ceil((1.0 - 1e-12) / max_weight))
    observed_training_windows = training_counts.max() if not training_counts.empty else np.nan
    diagnostic: dict[str, Any] = {
        "forecast_status": "unavailable",
        "reason": "unknown",
        "training_windows": int(observed_training_windows) if pd.notna(observed_training_windows) else 0,
        "training_end": training_end.isoformat() if pd.notna(training_end) else None,
        "required_candidates_for_weight_cap": required_candidates,
        "minimum_training_windows": min_training_windows,
        "assets_with_decision_price": int(current_mid.notna().sum()),
    }
    eligible = expected.index[
        expected.notna()
        & current_mid.reindex(expected.index).notna()
        & training_counts.reindex(expected.index).ge(min_training_windows)
    ]
    diagnostic["eligible_assets"] = int(len(eligible))
    if len(eligible) < required_candidates:
        diagnostic["reason"] = "insufficient_assets_with_training_and_exact_decision_prices"
        return None, None, diagnostic
    selected = list(expected.reindex(eligible).sort_values(ascending=False).head(top_n).index)
    diagnostic.update({"candidate_count": int(len(selected)), "covariance_scenarios": None})
    if len(selected) < required_candidates:
        diagnostic["reason"] = "insufficient_candidates_for_weight_cap"
        return None, None, diagnostic
    # Equal allocation makes every published five-minute prediction tractable
    # across the full 49-asset, three-month universe.  It is capped by
    # construction because the candidate-count gate above ensures 1/n <= cap.
    weights = _sanitize_weights(pd.Series(1.0 / len(selected), index=selected), max_weight=max_weight)
    predicted_log_return = float(weights.dot(expected.reindex(weights.index)))
    diagnostic.update(
        {
            "forecast_status": "ok",
            "reason": "ok",
            "optimizer_status": "equal_weight_top_forecasts",
            "expected_portfolio_log_return": predicted_log_return,
            "expected_portfolio_simple_return": float(np.expm1(predicted_log_return)),
        }
    )
    return weights, expected.reindex(weights.index), diagnostic


def run_ftmo_us_timeframe_audit(
    timeframe_quotes: pd.DataFrame,
    *,
    evaluation_start: str | date,
    evaluation_end: str | date,
    timeframe: str = "M5",
    top_n: int = 10,
    max_weight: float = 0.20,
    risk_aversion: float = 10.0,
    lookback_windows: int = 720,
    min_training_windows: int = 250,
    progress_every_decisions: int = 0,
) -> FtmoUsAuditResult:
    """Run a causal, non-overlapping cross-asset forecast audit at one timeframe."""
    normalized_timeframe = _timeframe(timeframe)
    interval = TIMEFRAME_INTERVALS[normalized_timeframe]
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")
    if top_n < math.ceil((1.0 - 1e-12) / max_weight):
        raise ValueError("top_n cannot satisfy the configured maximum weight cap")
    if lookback_windows < min_training_windows or min_training_windows < 2:
        raise ValueError("lookback_windows must be at least min_training_windows, which must be >= 2")
    if progress_every_decisions < 0:
        raise ValueError("progress_every_decisions must be non-negative")
    required = {"timestamp", "ftmo_symbol", "bid_close", "ask_close", "mid_close"}
    missing = required.difference(timeframe_quotes.columns)
    if missing:
        raise ValueError(f"five-minute quotes are missing required columns: {sorted(missing)}")
    quotes = timeframe_quotes.loc[:, sorted(required)].copy()
    quotes["timestamp"] = pd.to_datetime(quotes["timestamp"], utc=True, errors="coerce")
    quotes["ftmo_symbol"] = quotes["ftmo_symbol"].astype(str).str.upper().str.strip()
    for column in ("bid_close", "ask_close", "mid_close"):
        quotes[column] = pd.to_numeric(quotes[column], errors="coerce")
    quotes = quotes.dropna(subset=["timestamp", "ftmo_symbol", "bid_close", "ask_close", "mid_close"])
    quotes = quotes[(quotes[["bid_close", "ask_close", "mid_close"]] > 0).all(axis=1)]
    quotes = quotes.drop_duplicates(["timestamp", "ftmo_symbol"], keep="last").sort_values(["timestamp", "ftmo_symbol"])
    if quotes.empty:
        raise ValueError("no valid five-minute BID/ASK proxy quotes are available")
    mid = quotes.pivot(index="timestamp", columns="ftmo_symbol", values="mid_close").sort_index()
    bid = quotes.pivot(index="timestamp", columns="ftmo_symbol", values="bid_close").reindex(mid.index)
    ask = quotes.pivot(index="timestamp", columns="ftmo_symbol", values="ask_close").reindex(mid.index)
    labels = np.log(mid / _previous_interval(mid, interval))
    # ``shift(1)`` is the causal purge: at a decision timestamp, the most
    # recent label ending at that timestamp is never part of its forecast.
    rolling_expected = labels.rolling(lookback_windows, min_periods=min_training_windows).mean().shift(1)
    rolling_counts = labels.rolling(lookback_windows, min_periods=1).count().shift(1)
    label_observed = labels.notna().any(axis=1)
    training_end = pd.Series(mid.index.where(label_observed), index=mid.index).ffill().shift(1)
    start_day, end_day = _date(evaluation_start), _date(evaluation_end)
    if start_day > end_day:
        raise ValueError("evaluation_start must not be after evaluation_end")
    start_timestamp = pd.Timestamp(start_day, tz="UTC")
    end_exclusive = pd.Timestamp(end_day + timedelta(days=1), tz="UTC")
    decisions = mid.index[(mid.index >= start_timestamp) & (mid.index < end_exclusive)]
    ledger_rows: list[dict[str, Any]] = []
    holdings_rows: list[dict[str, Any]] = []
    forecast_windows = 0
    realized_windows = 0
    audit_started = time.monotonic()
    if progress_every_decisions:
        _progress(
            "PREDICTION START",
            timeframe=normalized_timeframe,
            decisions=len(decisions),
            evaluation_start=start_day.isoformat(),
            evaluation_end=end_day.isoformat(),
        )
    for decision_number, decision in enumerate(decisions, start=1):
        current_mid = mid.loc[decision]
        weights, expected, diagnostic = _select_causal_portfolio(
            rolling_expected.loc[decision],
            rolling_counts.loc[decision],
            current_mid,
            training_end=training_end.loc[decision],
            top_n=top_n,
            max_weight=max_weight,
            min_training_windows=min_training_windows,
        )
        row: dict[str, Any] = {
            "decision_timestamp": decision.isoformat(),
            "decision_price_timestamp": (decision - ONE_MINUTE).isoformat(),
            "target_end": (decision + FIVE_MINUTES).isoformat(),
            "forecast_status": diagnostic["forecast_status"],
            "reason": diagnostic["reason"],
            "training_windows": diagnostic["training_windows"],
            "training_end": diagnostic["training_end"],
            "eligible_assets": diagnostic.get("eligible_assets"),
            "candidate_count": diagnostic.get("candidate_count"),
            "covariance_scenarios": diagnostic.get("covariance_scenarios"),
            "holding_count": int(len(weights)) if weights is not None else 0,
            "predicted_portfolio_log_return": diagnostic.get("expected_portfolio_log_return"),
            "predicted_portfolio_simple_return": diagnostic.get("expected_portfolio_simple_return"),
            "realized_status": "not_forecast",
            "actual_mid_portfolio_log_return": None,
            "actual_mid_portfolio_simple_return": None,
            "actual_executable_proxy_log_return": None,
            "actual_executable_proxy_simple_return": None,
        }
        if weights is not None and expected is not None:
            target = decision + interval
            target_mid = mid.reindex([target]).iloc[0]
            target_bid = bid.reindex([target]).iloc[0]
            entry_ask = ask.loc[decision]
            selected = weights.index
            mid_return = np.log(target_mid.reindex(selected) / current_mid.reindex(selected))
            executable_return = np.log(target_bid.reindex(selected) / entry_ask.reindex(selected))
            if mid_return.isna().any() or executable_return.isna().any():
                row["realized_status"] = "missing_exact_bid_ask_endpoint"
            else:
                actual_mid_log = float(weights.dot(mid_return))
                actual_executable_log = float(weights.dot(executable_return))
                row.update(
                    {
                        "realized_status": "ok",
                        "actual_mid_portfolio_log_return": actual_mid_log,
                        "actual_mid_portfolio_simple_return": float(np.expm1(actual_mid_log)),
                        "actual_executable_proxy_log_return": actual_executable_log,
                        "actual_executable_proxy_simple_return": float(np.expm1(actual_executable_log)),
                    }
                )
            for symbol, weight in weights.items():
                holdings_rows.append(
                    {
                        "decision_timestamp": decision.isoformat(),
                        "target_end": target.isoformat(),
                        "symbol": symbol,
                        "target_weight": float(weight),
                        "predicted_asset_log_return": float(expected.loc[symbol]),
                        "entry_mid_close": float(current_mid.loc[symbol]),
                        "entry_ask_close": float(entry_ask.loc[symbol]),
                        "target_mid_close": float(target_mid.loc[symbol]) if pd.notna(target_mid.loc[symbol]) else None,
                        "target_bid_close": float(target_bid.loc[symbol]) if pd.notna(target_bid.loc[symbol]) else None,
                        "realized_mid_asset_log_return": float(mid_return.loc[symbol]) if pd.notna(mid_return.loc[symbol]) else None,
                        "realized_executable_proxy_asset_log_return": float(executable_return.loc[symbol]) if pd.notna(executable_return.loc[symbol]) else None,
                    }
                )
        ledger_rows.append(row)
        forecast_windows += int(row["forecast_status"] == "ok")
        realized_windows += int(row["realized_status"] == "ok")
        if progress_every_decisions and (
            decision_number % progress_every_decisions == 0 or decision_number == len(decisions)
        ):
            _progress(
                "PREDICTION PROGRESS",
                decisions=f"{decision_number}/{len(decisions)}",
                forecasts=forecast_windows,
                realized=realized_windows,
                elapsed=f"{time.monotonic() - audit_started:.1f}s",
            )
    ledger = pd.DataFrame(ledger_rows)
    holdings = pd.DataFrame(holdings_rows)
    realized = ledger.loc[ledger["realized_status"].eq("ok")].copy() if not ledger.empty else pd.DataFrame()
    matched_expected = pd.to_numeric(realized.get("predicted_portfolio_log_return"), errors="coerce").dropna()
    matched_mid = pd.to_numeric(realized.get("actual_mid_portfolio_log_return"), errors="coerce").dropna()
    matched_executable = pd.to_numeric(realized.get("actual_executable_proxy_log_return"), errors="coerce").dropna()
    common = realized.dropna(subset=["predicted_portfolio_simple_return", "actual_mid_portfolio_simple_return"])
    error_bps = (
        (common["predicted_portfolio_simple_return"] - common["actual_mid_portfolio_simple_return"]) * 10_000
        if not common.empty
        else pd.Series(dtype=float)
    )
    direction = (
        np.sign(common["predicted_portfolio_simple_return"])
        == np.sign(common["actual_mid_portfolio_simple_return"])
        if not common.empty
        else pd.Series(dtype=bool)
    )
    summary: dict[str, Any] = {
        "research_only": True,
        "timeframe": normalized_timeframe,
        "portfolio_mode": "rebalanced_each_decision",
        "evaluation_start": start_day.isoformat(),
        "evaluation_end": end_day.isoformat(),
        "decision_points": int(len(decisions)),
        "five_minute_decision_points": int(len(decisions)) if normalized_timeframe == "M5" else None,
        "forecast_windows": int(ledger["forecast_status"].eq("ok").sum()) if not ledger.empty else 0,
        "realized_windows": int(len(realized)),
        "exact_bid_ask_realized_coverage": float(len(realized) / max(1, int(ledger["forecast_status"].eq("ok").sum())) if not ledger.empty else 0.0),
        "matched_compounded_expected_return": float(np.expm1(matched_expected.sum())) if not matched_expected.empty else None,
        "matched_compounded_mid_return": float(np.expm1(matched_mid.sum())) if not matched_mid.empty else None,
        "matched_compounded_executable_proxy_return": float(np.expm1(matched_executable.sum())) if not matched_executable.empty else None,
        "forecast_mae_bps_vs_mid": float(error_bps.abs().mean()) if not error_bps.empty else None,
        "forecast_rmse_bps_vs_mid": float(np.sqrt((error_bps**2).mean())) if not error_bps.empty else None,
        "forecast_directional_accuracy_vs_mid": float(direction.mean()) if not direction.empty else None,
        "distinct_selected_symbols": int(holdings["symbol"].nunique()) if not holdings.empty else 0,
        "model": f"trailing causal {normalized_timeframe} mean forecast plus capped long-only equal-weight allocation",
        "causality_rule": (
            f"each label ends at a {normalized_timeframe} decision; model training uses labels with end strictly earlier "
            "than that decision; decisions use the immediately preceding completed native M1 endpoint"
        ),
        "price_basis": "forecast target: proxy mid; executable proxy: enter ASK and exit next BID",
        "costs_included": "Dukascopy proxy BID/ASK spread only",
        "costs_excluded": "FTMO commissions, swaps, contract rolls, slippage, market impact, and all account-rule effects",
        "ftmo_execution_claim": False,
        "feed_caveat": (
            "Dukascopy is not FTMO US. Results are a proxy-feed research audit and cannot be represented as "
            "FTMO US fills, returns, or a pass/fail expectation."
        ),
    }
    return FtmoUsAuditResult(ledger=ledger, holdings=holdings, summary=summary)


def run_ftmo_us_no_rebalance_audit(
    timeframe_quotes: pd.DataFrame,
    rebalanced_result: FtmoUsAuditResult,
    *,
    evaluation_start: str | date,
    evaluation_end: str | date,
    timeframe: str,
) -> FtmoUsAuditResult:
    """Hold the first causal portfolio unchanged through the evaluation end.

    This is a buy-and-hold comparison against the same timeframe's repeatedly
    rebalanced portfolio.  It intentionally makes only one selection, using
    the first forecast that was eligible under the rebalanced audit's causal
    training rules.
    """
    normalized_timeframe = _timeframe(timeframe)
    eligible = rebalanced_result.ledger.loc[rebalanced_result.ledger["forecast_status"].eq("ok")].copy()
    if eligible.empty:
        raise ValueError(f"{normalized_timeframe} has no causal portfolio eligible for the no-rebalance audit")
    first = eligible.iloc[0]
    entry_timestamp = pd.Timestamp(first["decision_timestamp"])
    selected = rebalanced_result.holdings.loc[
        rebalanced_result.holdings["decision_timestamp"].eq(first["decision_timestamp"])
    ].copy()
    if selected.empty:
        raise ValueError("first causal portfolio has no holdings")
    weights = selected.set_index("symbol")["target_weight"].astype(float)
    quotes = timeframe_quotes.copy()
    quotes["timestamp"] = pd.to_datetime(quotes["timestamp"], utc=True, errors="coerce")
    quotes["ftmo_symbol"] = quotes["ftmo_symbol"].astype(str).str.upper().str.strip()
    for column in ("bid_close", "ask_close", "mid_close"):
        quotes[column] = pd.to_numeric(quotes[column], errors="coerce")
    quotes = quotes.dropna(subset=["timestamp", "ftmo_symbol", "bid_close", "ask_close", "mid_close"])
    quotes = quotes.drop_duplicates(["timestamp", "ftmo_symbol"], keep="last")
    mid = quotes.pivot(index="timestamp", columns="ftmo_symbol", values="mid_close").sort_index()
    bid = quotes.pivot(index="timestamp", columns="ftmo_symbol", values="bid_close").reindex(mid.index)
    ask = quotes.pivot(index="timestamp", columns="ftmo_symbol", values="ask_close").reindex(mid.index)
    end_exclusive = pd.Timestamp(_date(evaluation_end) + timedelta(days=1), tz="UTC")
    candidate_times = mid.index[(mid.index > entry_timestamp) & (mid.index <= end_exclusive)]
    exit_timestamp: pd.Timestamp | None = None
    for candidate in reversed(candidate_times):
        if mid.loc[candidate].reindex(weights.index).notna().all() and bid.loc[candidate].reindex(weights.index).notna().all():
            exit_timestamp = candidate
            break
    if exit_timestamp is None:
        raise ValueError("no common exact BID/mid endpoint is available for the no-rebalance exit")
    entry_mid = mid.loc[entry_timestamp].reindex(weights.index)
    entry_ask = ask.loc[entry_timestamp].reindex(weights.index)
    exit_mid = mid.loc[exit_timestamp].reindex(weights.index)
    exit_bid = bid.loc[exit_timestamp].reindex(weights.index)
    if entry_mid.isna().any() or entry_ask.isna().any():
        raise ValueError("first causal portfolio is missing an exact entry quote")
    expected = selected.set_index("symbol")["predicted_asset_log_return"].astype(float).reindex(weights.index)
    mid_returns = np.log(exit_mid / entry_mid)
    executable_returns = np.log(exit_bid / entry_ask)
    expected_log = float(weights.dot(expected))
    mid_log = float(weights.dot(mid_returns))
    executable_log = float(weights.dot(executable_returns))
    no_rebalance_holdings = pd.DataFrame(
        {
            "decision_timestamp": entry_timestamp.isoformat(),
            "target_end": exit_timestamp.isoformat(),
            "symbol": weights.index,
            "target_weight": weights.to_numpy(),
            "predicted_asset_log_return": expected.to_numpy(),
            "entry_mid_close": entry_mid.to_numpy(),
            "entry_ask_close": entry_ask.to_numpy(),
            "target_mid_close": exit_mid.to_numpy(),
            "target_bid_close": exit_bid.to_numpy(),
            "realized_mid_asset_log_return": mid_returns.to_numpy(),
            "realized_executable_proxy_asset_log_return": executable_returns.to_numpy(),
        }
    )
    ledger = pd.DataFrame(
        [
            {
                "decision_timestamp": entry_timestamp.isoformat(),
                "decision_price_timestamp": (entry_timestamp - ONE_MINUTE).isoformat(),
                "target_end": exit_timestamp.isoformat(),
                "forecast_status": "ok",
                "reason": "first_causal_portfolio_held_without_rebalancing",
                "training_windows": int(first["training_windows"]),
                "training_end": first["training_end"],
                "eligible_assets": int(first["eligible_assets"]),
                "candidate_count": int(first["candidate_count"]),
                "holding_count": int(len(weights)),
                "predicted_portfolio_log_return": expected_log,
                "predicted_portfolio_simple_return": float(np.expm1(expected_log)),
                "realized_status": "ok",
                "actual_mid_portfolio_log_return": mid_log,
                "actual_mid_portfolio_simple_return": float(np.expm1(mid_log)),
                "actual_executable_proxy_log_return": executable_log,
                "actual_executable_proxy_simple_return": float(np.expm1(executable_log)),
            }
        ]
    )
    summary: dict[str, Any] = {
        "research_only": True,
        "timeframe": normalized_timeframe,
        "portfolio_mode": "no_rebalance_buy_and_hold",
        "evaluation_start": _date(evaluation_start).isoformat(),
        "evaluation_end": _date(evaluation_end).isoformat(),
        "hold_entry_timestamp": entry_timestamp.isoformat(),
        "hold_exit_timestamp": exit_timestamp.isoformat(),
        "decision_points": 1,
        "five_minute_decision_points": 1 if normalized_timeframe == "M5" else None,
        "forecast_windows": 1,
        "realized_windows": 1,
        "exact_bid_ask_realized_coverage": 1.0,
        # The forecast only covers the first interval.  It selected the static
        # basket, but it is not a three-month expected return and must not be
        # compared to the full holding-period outcome below.
        "matched_compounded_expected_return": None,
        "initial_selection_expected_one_interval_return": float(np.expm1(expected_log)),
        "matched_compounded_mid_return": float(np.expm1(mid_log)),
        "matched_compounded_executable_proxy_return": float(np.expm1(executable_log)),
        "forecast_mae_bps_vs_mid": None,
        "forecast_rmse_bps_vs_mid": None,
        "forecast_directional_accuracy_vs_mid": None,
        "distinct_selected_symbols": int(len(weights)),
        "model": f"first causal {normalized_timeframe} forecast held without rebalancing",
        "causality_rule": (
            f"the first eligible {normalized_timeframe} portfolio is selected using only labels ending before its entry "
            "and is held unchanged to the last common exact endpoint at or before evaluation end"
        ),
        "price_basis": "forecast target: proxy mid; executable proxy: enter ASK and exit BID at the static hold exit",
        "costs_included": "Dukascopy proxy BID/ASK spread only",
        "costs_excluded": "FTMO commissions, swaps, contract rolls, slippage, market impact, and all account-rule effects",
        "ftmo_execution_claim": False,
        "feed_caveat": (
            "Dukascopy is not FTMO US. Results are a proxy-feed research audit and cannot be represented as "
            "FTMO US fills, returns, or a pass/fail expectation."
        ),
    }
    return FtmoUsAuditResult(ledger=ledger, holdings=no_rebalance_holdings, summary=summary)


def run_ftmo_us_five_minute_audit(
    five_minute_quotes: pd.DataFrame,
    *,
    evaluation_start: str | date,
    evaluation_end: str | date,
    top_n: int = 10,
    max_weight: float = 0.20,
    risk_aversion: float = 10.0,
    lookback_windows: int = 720,
    min_training_windows: int = 250,
    progress_every_decisions: int = 0,
) -> FtmoUsAuditResult:
    """Backward-compatible M5 wrapper for ``run_ftmo_us_timeframe_audit``."""
    return run_ftmo_us_timeframe_audit(
        five_minute_quotes,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        timeframe="M5",
        top_n=top_n,
        max_weight=max_weight,
        risk_aversion=risk_aversion,
        lookback_windows=lookback_windows,
        min_training_windows=min_training_windows,
        progress_every_decisions=progress_every_decisions,
    )


def _pct(value: object) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{float(value):+.4%}"


def _bps(value: object) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{float(value):.2f} bps"


def _render_markdown_report(summary: dict[str, Any], download: dict[str, Any]) -> str:
    portfolio_mode = str(summary.get("portfolio_mode", "unspecified")).replace("_", " ")
    return "\n".join(
        [
            f"## FTMO US proxy-feed causal {summary['timeframe']} audit ({portfolio_mode})",
            "",
            f"- FTMO US assets verified at run time: {download.get('verified_ftmo_us_assets', 'n/a')}; mapped assets requested: {download['assets_requested']:,}.",
            f"- Proxy data: {download['m1_bid_ask_rows']:,} native Dukascopy M1 BID/ASK rows across {download['symbols_with_bid_ask_data']:,} symbols.",
            f"- Evaluation: {summary['evaluation_start']} through {summary['evaluation_end']}; decisions: {summary['decision_points']:,}.",
            f"- Forecasts / exact proxy outcomes: {summary['forecast_windows']:,} / {summary['realized_windows']:,} ({_pct(summary['exact_bid_ask_realized_coverage'])}).",
            f"- Matched compounded expected / mid realized / executable-proxy realized: {_pct(summary['matched_compounded_expected_return'])} / {_pct(summary['matched_compounded_mid_return'])} / {_pct(summary['matched_compounded_executable_proxy_return'])}.",
            f"- Forecast MAE / RMSE / direction accuracy versus proxy mid: {_bps(summary['forecast_mae_bps_vs_mid'])} / {_bps(summary['forecast_rmse_bps_vs_mid'])} / {_pct(summary['forecast_directional_accuracy_vs_mid'])}.",
            f"- Costs included: {summary['costs_included']}. Excluded: {summary['costs_excluded']}.",
            "- Important: FTMO US is simulated and Dukascopy is not its execution feed. This is not an FTMO fill, net-performance, pass-probability, or trading recommendation report.",
        ]
    ) + "\n"


def write_ftmo_us_audit(
    output_dir: str | Path,
    *,
    minute_closes: pd.DataFrame | None,
    five_minute_quotes: pd.DataFrame,
    assets: pd.DataFrame,
    manifest: pd.DataFrame,
    manifest_metadata: dict[str, Any],
    download_metadata: dict[str, Any],
    result: FtmoUsAuditResult,
) -> None:
    """Write reproducible inputs, decisions, outcomes, and disclosure report."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if minute_closes is not None:
        minute_closes.to_csv(output / "ftmo_us_dukascopy_m1_bid_ask.csv.gz", index=False, compression="gzip")
    five_minute_quotes.to_csv(output / "ftmo_us_dukascopy_five_minute_quotes.csv.gz", index=False, compression="gzip")
    assets.to_csv(output / "ftmo_us_selected_proxy_assets.csv", index=False)
    manifest.to_json(output / "ftmo_us_official_manifest.json", orient="records", indent=2)
    combined_download = {**download_metadata, "verified_ftmo_us_assets": int(len(manifest)), **manifest_metadata}
    (output / "ftmo_us_download_metadata.json").write_text(json.dumps(combined_download, indent=2) + "\n", encoding="utf-8")
    result.ledger.to_csv(output / "ftmo_us_five_minute_forecast_ledger.csv", index=False)
    result.holdings.to_csv(output / "ftmo_us_five_minute_forecast_holdings.csv", index=False)
    (output / "ftmo_us_five_minute_forecast_summary.json").write_text(
        json.dumps(result.summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "ftmo_us_five_minute_audit_report.md").write_text(
        _render_markdown_report(result.summary, combined_download), encoding="utf-8"
    )


def run_download_and_audit(
    *,
    output_dir: str | Path,
    download_start: str | date,
    evaluation_start: str | date,
    evaluation_end: str | date,
    requested_symbols: str = "all",
    mapping_path: str | Path | None = None,
    workers: int = 8,
    retries: int = 3,
    timeout_seconds: int = 30,
    top_n: int = 10,
    max_weight: float = 0.20,
    risk_aversion: float = 10.0,
    lookback_windows: int = 720,
    min_training_windows: int = 250,
    progress_every_files: int = 25,
    progress_every_decisions: int = 250,
    timeframe: str = "M5",
) -> FtmoUsAuditResult:
    """Verify the FTMO US universe, download proxy quotes, and audit one timeframe."""
    normalized_timeframe = _timeframe(timeframe)
    mapping = load_ftmo_us_mapping(mapping_path)
    manifest, manifest_metadata = fetch_ftmo_us_manifest(timeout_seconds=timeout_seconds)
    assets = select_verified_assets(mapping, manifest, requested_symbols)
    _progress(
        "UNIVERSE VERIFIED",
        active_ftmo_us_assets=len(manifest),
        mapped_assets=len(assets),
        download_start=_date(download_start).isoformat(),
        evaluation_end=_date(evaluation_end).isoformat(),
    )
    five_minute_chunks: list[pd.DataFrame] = []

    def _reduce_m1_chunk(minute_chunk: pd.DataFrame) -> None:
        quotes = native_m1_to_timeframe_quotes(minute_chunk, timeframe=normalized_timeframe)
        if not quotes.empty:
            five_minute_chunks.append(quotes)

    _, download_metadata = download_dukascopy_ftmo_us_m1(
        assets,
        start=download_start,
        end=evaluation_end,
        workers=workers,
        retries=retries,
        timeout_seconds=timeout_seconds,
        on_m1_chunk=_reduce_m1_chunk,
        collect_m1=False,
        progress_every_files=progress_every_files,
    )
    if not five_minute_chunks:
        raise ValueError("Dukascopy returned no exact M1 BID/ASK endpoints for the selected FTMO US proxy assets")
    five_minute_quotes = pd.concat(five_minute_chunks, ignore_index=True).sort_values(
        ["timestamp", "ftmo_symbol"]
    ).reset_index(drop=True)
    result = run_ftmo_us_timeframe_audit(
        five_minute_quotes,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        timeframe=normalized_timeframe,
        top_n=top_n,
        max_weight=max_weight,
        risk_aversion=risk_aversion,
        lookback_windows=lookback_windows,
        min_training_windows=min_training_windows,
        progress_every_decisions=progress_every_decisions,
    )
    write_ftmo_us_audit(
        output_dir,
        minute_closes=None,
        five_minute_quotes=five_minute_quotes,
        assets=assets,
        manifest=manifest,
        manifest_metadata=manifest_metadata,
        download_metadata=download_metadata,
        result=result,
    )
    _progress(
        "ARTIFACT WRITTEN",
        forecasts=result.summary["forecast_windows"],
        realized=result.summary["realized_windows"],
        output_dir=output_dir,
    )
    return result


def run_download_and_multi_timeframe_audit(
    *,
    output_dir: str | Path,
    download_start: str | date,
    evaluation_start: str | date,
    evaluation_end: str | date,
    requested_symbols: str = "all",
    mapping_path: str | Path | None = None,
    workers: int = 8,
    retries: int = 3,
    timeout_seconds: int = 30,
    top_n: int = 10,
    max_weight: float = 0.20,
    risk_aversion: float = 10.0,
    progress_every_files: int = 25,
    progress_every_decisions: int = 250,
) -> dict[str, FtmoUsAuditResult]:
    """Download M1 proxy data once, then audit H1, H4, and D1 causally."""
    mapping = load_ftmo_us_mapping(mapping_path)
    manifest, manifest_metadata = fetch_ftmo_us_manifest(timeout_seconds=timeout_seconds)
    assets = select_verified_assets(mapping, manifest, requested_symbols)
    _progress(
        "MULTI-TIMEFRAME UNIVERSE VERIFIED",
        active_ftmo_us_assets=len(manifest),
        mapped_assets=len(assets),
        download_start=_date(download_start).isoformat(),
        evaluation_end=_date(evaluation_end).isoformat(),
        timeframes=",".join(MULTI_TIMEFRAME_PARAMETERS),
    )
    quote_chunks: dict[str, list[pd.DataFrame]] = {timeframe: [] for timeframe in MULTI_TIMEFRAME_PARAMETERS}

    def _reduce_m1_chunk(minute_chunk: pd.DataFrame) -> None:
        for timeframe, chunks in quote_chunks.items():
            quotes = native_m1_to_timeframe_quotes(minute_chunk, timeframe=timeframe)
            if not quotes.empty:
                chunks.append(quotes)

    _, download_metadata = download_dukascopy_ftmo_us_m1(
        assets,
        start=download_start,
        end=evaluation_end,
        workers=workers,
        retries=retries,
        timeout_seconds=timeout_seconds,
        on_m1_chunk=_reduce_m1_chunk,
        collect_m1=False,
        progress_every_files=progress_every_files,
    )
    results: dict[str, FtmoUsAuditResult] = {}
    root = Path(output_dir)
    for timeframe, parameters in MULTI_TIMEFRAME_PARAMETERS.items():
        if not quote_chunks[timeframe]:
            raise ValueError(f"Dukascopy returned no exact M1 endpoints for {timeframe}")
        quotes = pd.concat(quote_chunks[timeframe], ignore_index=True).sort_values(
            ["timestamp", "ftmo_symbol"]
        ).reset_index(drop=True)
        _progress("TIMEFRAME AUDIT START", timeframe=timeframe, quote_rows=len(quotes))
        result = run_ftmo_us_timeframe_audit(
            quotes,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            timeframe=timeframe,
            top_n=top_n,
            max_weight=max_weight,
            risk_aversion=risk_aversion,
            progress_every_decisions=progress_every_decisions,
            **parameters,
        )
        write_ftmo_us_audit(
            root / timeframe.lower() / "rebalanced",
            minute_closes=None,
            five_minute_quotes=quotes,
            assets=assets,
            manifest=manifest,
            manifest_metadata=manifest_metadata,
            download_metadata=download_metadata,
            result=result,
        )
        _progress(
            "TIMEFRAME AUDIT COMPLETE",
            timeframe=timeframe,
            forecasts=result.summary["forecast_windows"],
            realized=result.summary["realized_windows"],
            executable_proxy_return=_pct(result.summary["matched_compounded_executable_proxy_return"]),
        )
        no_rebalance = run_ftmo_us_no_rebalance_audit(
            quotes,
            result,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            timeframe=timeframe,
        )
        write_ftmo_us_audit(
            root / timeframe.lower() / "no_rebalance",
            minute_closes=None,
            five_minute_quotes=quotes,
            assets=assets,
            manifest=manifest,
            manifest_metadata=manifest_metadata,
            download_metadata=download_metadata,
            result=no_rebalance,
        )
        _progress(
            "NO-REBALANCE AUDIT COMPLETE",
            timeframe=timeframe,
            entry=no_rebalance.summary["hold_entry_timestamp"],
            exit=no_rebalance.summary["hold_exit_timestamp"],
            executable_proxy_return=_pct(no_rebalance.summary["matched_compounded_executable_proxy_return"]),
        )
        results[f"{timeframe}_rebalanced"] = result
        results[f"{timeframe}_no_rebalance"] = no_rebalance
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run causal FTMO US proxy-feed research audits.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="verify FTMO US assets, download Dukascopy proxy candles, and audit")
    multi = subparsers.add_parser("multi", help="download once and audit H1, H4, and D1")

    def add_shared_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--output-dir", required=True)
        command.add_argument("--download-start", required=True, help="inclusive UTC date YYYY-MM-DD for model warm-up")
        command.add_argument("--evaluation-start", required=True, help="inclusive UTC date YYYY-MM-DD")
        command.add_argument("--evaluation-end", required=True, help="inclusive UTC date YYYY-MM-DD")
        command.add_argument("--symbols", default="all", help="all or comma-separated active FTMO US .sim symbols")
        command.add_argument("--mapping", default=str(_default_mapping_path()))
        command.add_argument("--workers", type=int, default=8)
        command.add_argument("--retries", type=int, default=3)
        command.add_argument("--timeout-seconds", type=int, default=30)
        command.add_argument("--top-n", type=int, default=10)
        command.add_argument("--max-weight", type=float, default=0.20)
        command.add_argument("--risk-aversion", type=float, default=10.0)
        command.add_argument("--progress-every-files", type=int, default=25)
        command.add_argument("--progress-every-decisions", type=int, default=250)

    add_shared_arguments(run)
    run.add_argument("--timeframe", default="M5", choices=sorted(TIMEFRAME_INTERVALS))
    run.add_argument("--lookback-windows", type=int, default=720)
    run.add_argument("--min-training-windows", type=int, default=250)
    add_shared_arguments(multi)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "run":
        result = run_download_and_audit(
            output_dir=args.output_dir,
            download_start=args.download_start,
            evaluation_start=args.evaluation_start,
            evaluation_end=args.evaluation_end,
            requested_symbols=args.symbols,
            mapping_path=args.mapping,
            workers=args.workers,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            top_n=args.top_n,
            max_weight=args.max_weight,
            risk_aversion=args.risk_aversion,
            lookback_windows=args.lookback_windows,
            min_training_windows=args.min_training_windows,
            progress_every_files=args.progress_every_files,
            progress_every_decisions=args.progress_every_decisions,
            timeframe=args.timeframe,
        )
        print(
            "FTMO US PROXY AUDIT COMPLETE | "
            f"forecasts={result.summary['forecast_windows']} realized={result.summary['realized_windows']}"
        )
    if args.command == "multi":
        results = run_download_and_multi_timeframe_audit(
            output_dir=args.output_dir,
            download_start=args.download_start,
            evaluation_start=args.evaluation_start,
            evaluation_end=args.evaluation_end,
            requested_symbols=args.symbols,
            mapping_path=args.mapping,
            workers=args.workers,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            top_n=args.top_n,
            max_weight=args.max_weight,
            risk_aversion=args.risk_aversion,
            progress_every_files=args.progress_every_files,
            progress_every_decisions=args.progress_every_decisions,
        )
        print(
            "FTMO US MULTI-TIMEFRAME PROXY AUDIT COMPLETE | "
            + " | ".join(
                f"{timeframe}: forecasts={result.summary['forecast_windows']} realized={result.summary['realized_windows']}"
                for timeframe, result in results.items()
            )
        )


if __name__ == "__main__":
    main()
