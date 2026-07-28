"""Optional Alpaca minute-bar downloader; it has no order-routing capability."""

from __future__ import annotations

import argparse
import os
import ssl
import threading
from collections.abc import Collection, Mapping
from datetime import datetime, time as clock_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


MINUTE_BAR_COLUMNS = ["timestamp", "symbol", "close"]
NEW_YORK = ZoneInfo("America/New_York")


def load_symbols(path: str | Path) -> list[str]:
    universe = pd.read_csv(path)
    if "symbol" not in universe.columns:
        raise ValueError(f"{path} must contain a 'symbol' column")
    # The convenience universe retains both a yfinance-friendly `symbol`
    # (for example BRK-B) and the exchange/Alpaca `source_symbol` (BRK.B).
    # Prefer the latter whenever it exists so a full S&P request does not
    # fail on share-class tickers.
    symbol_column = "source_symbol" if "source_symbol" in universe.columns else "symbol"
    symbols = universe[symbol_column].dropna().astype(str).str.upper().str.strip()
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


def _market_data_credentials() -> tuple[str, str]:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set; no request was sent")
    return key, secret


def _empty_minute_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=MINUTE_BAR_COLUMNS)


def _minute_bar_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return _empty_minute_bars()
    result = pd.DataFrame(rows, columns=MINUTE_BAR_COLUMNS)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    result["symbol"] = result["symbol"].astype(str).str.upper().str.strip()
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result.dropna(subset=MINUTE_BAR_COLUMNS)
    result = result[(result["symbol"] != "") & (result["close"] > 0)]
    return result.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"], keep="last")


class AlpacaMinuteBarStream:
    """Keep completed one-minute IEX bars received over Alpaca's websocket.

    The historical endpoint remains necessary for the initial training window,
    but it must not be the source of a live cache refresh.  This stream uses
    the IEX feed explicitly, which is available to paper accounts that do not
    subscribe to SIP.  Its rows are only released to callers after the caller
    supplies a completed-minute boundary.
    """

    def __init__(self, symbols: list[str]) -> None:
        normalized = list(dict.fromkeys(symbol.upper().strip() for symbol in symbols if symbol.strip()))
        if not normalized:
            raise ValueError("at least one symbol is required for the Alpaca minute-bar stream")
        self._symbols = normalized
        self._lock = threading.Lock()
        self._rows: list[dict[str, object]] = []
        self._stream: Any | None = None
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._last_bar_at: pd.Timestamp | None = None

    @property
    def error(self) -> Exception | None:
        return self._error

    @property
    def connected(self) -> bool:
        """Whether Alpaca currently has an authenticated websocket transport."""
        websocket = getattr(self._stream, "_ws", None)
        return websocket is not None and not getattr(websocket, "closed", False) and self._error is None

    @property
    def available(self) -> bool:
        """Whether the stream worker is still able to receive IEX bars.

        Alpaca creates its internal websocket lazily, so ``connected`` can be
        false for a short authenticated-startup interval.  A live cache must
        wait through that interval instead of treating it as a failure and
        repeatedly falling through to another provider.
        """
        return self._error is None and self._thread is not None and self._thread.is_alive()

    @property
    def last_bar_at(self) -> pd.Timestamp | None:
        """Latest exchange minute received, if the stream has delivered one."""
        return self._last_bar_at

    def start(self) -> None:
        """Open the IEX websocket in a daemon thread and subscribe to bars."""
        if self._thread is not None:
            return
        key, secret = _market_data_credentials()
        import certifi
        from alpaca.data.enums import DataFeed
        from alpaca.data.live import StockDataStream

        # macOS framework Python often has no system roots configured.  Use
        # certifi explicitly so the data connection has the same verified TLS
        # behavior locally and on hosted GitHub runners.
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        self._stream = StockDataStream(key, secret, feed=DataFeed.IEX, websocket_params={"ssl": ssl_context})
        self._stream.subscribe_bars(self._on_bar, *self._symbols)
        self._thread = threading.Thread(target=self._run, name="alpaca-iex-minute-bars", daemon=True)
        self._thread.start()

    async def _on_bar(self, bar: Any) -> None:
        if isinstance(bar, Mapping):
            symbol = bar.get("symbol")
            timestamp = bar.get("timestamp")
            close = bar.get("close")
        else:
            symbol = getattr(bar, "symbol", None)
            timestamp = getattr(bar, "timestamp", None)
            close = getattr(bar, "close", None)
        if symbol is None or timestamp is None or close is None:
            return
        timestamp_at = pd.Timestamp(timestamp)
        if timestamp_at.tzinfo is None:
            timestamp_at = timestamp_at.tz_localize("UTC")
        else:
            timestamp_at = timestamp_at.tz_convert("UTC")
        with self._lock:
            self._rows.append({"timestamp": timestamp_at, "symbol": symbol, "close": close})
            self._last_bar_at = timestamp_at

    def _run(self) -> None:
        assert self._stream is not None
        try:
            self._stream.run()
        except Exception as error:  # The caller falls back without losing the scheduled event.
            self._error = error

    def completed_bars_through(self, completed_at: pd.Timestamp) -> pd.DataFrame:
        """Return the stream rows no later than the caller's causal boundary."""
        with self._lock:
            rows = list(self._rows)
        result = _minute_bar_frame(rows)
        if result.empty:
            return result
        return result.loc[result["timestamp"] <= pd.Timestamp(completed_at).tz_convert("UTC")].copy()

    def stop(self) -> None:
        """Close the websocket without blocking the paper-session shutdown."""
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception as error:  # A disconnect is non-fatal during shutdown.
                self._error = self._error or error
        if self._thread is not None:
            self._thread.join(timeout=5)


def download_minute_bars(
    symbols: list[str], start: str | datetime, end: str | datetime, *, batch_size: int = 100
) -> pd.DataFrame:
    """Download one-minute stock bars using Alpaca's market-data API only."""
    key, secret = _market_data_credentials()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.enums import DataFeed
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
            # The default feed can be SIP, which a paper-only account need
            # not be entitled to query in real time.  IEX is deliberately
            # selected for both the bootstrap and any narrow repair request.
            feed=DataFeed.IEX,
        )
        frame = client.get_stock_bars(request).df
        if frame.empty:
            continue
        frames.append(frame.reset_index().loc[:, ["timestamp", "symbol", "close"]])
    if not frames:
        return _empty_minute_bars()
    result = pd.concat(frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    return result.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"])


def download_minute_endpoint_bars(
    symbols: list[str],
    start: str | datetime,
    end: str | datetime,
    *,
    endpoint_times: Collection[clock_time],
    batch_size: int = 100,
) -> pd.DataFrame:
    """Download IEX minutes but retain only exact New York endpoint closes.

    A multi-week intraday audit needs a small, auditable set of timestamps
    rather than millions of non-decision rows.  Filtering each Alpaca response
    before appending it keeps the downloaded result compact while preserving
    the exact prices used for selections, labels, and 15-minute rebalances.
    This is read-only market-data access and explicitly never requests SIP.
    """
    key, secret = _market_data_credentials()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    selected_times = frozenset(endpoint_times)
    if not selected_times:
        raise ValueError("endpoint_times must contain at least one New York clock time")

    from alpaca.data.enums import DataFeed
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
            feed=DataFeed.IEX,
        )
        frame = client.get_stock_bars(request).df
        if frame.empty:
            continue
        normalized = _minute_bar_frame(frame.reset_index().loc[:, ["timestamp", "symbol", "close"]].to_dict("records"))
        if normalized.empty:
            continue
        local_times = normalized["timestamp"].dt.tz_convert(NEW_YORK).dt.time
        endpoint_rows = normalized.loc[local_times.isin(selected_times)]
        if not endpoint_rows.empty:
            frames.append(endpoint_rows)
    if not frames:
        return _empty_minute_bars()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["symbol", "timestamp"])
        .drop_duplicates(["symbol", "timestamp"], keep="last")
        .reset_index(drop=True)
    )


def download_yfinance_minute_bars(
    symbols: list[str], start: str | datetime, end: str | datetime
) -> pd.DataFrame:
    """Return a best-effort Yahoo one-minute fallback mapped to Alpaca symbols.

    This function is intentionally reserved for a short, current-session
    repair after the websocket fails.  It never replaces the Alpaca IEX
    historical bootstrap and keeps a share-class mapping (``BRK.B`` to
    ``BRK-B``) local to the fallback boundary.
    """
    start_at, end_at = pd.Timestamp(start), pd.Timestamp(end)
    if start_at.tzinfo is None:
        start_at = start_at.tz_localize("UTC")
    else:
        start_at = start_at.tz_convert("UTC")
    if end_at.tzinfo is None:
        end_at = end_at.tz_localize("UTC")
    else:
        end_at = end_at.tz_convert("UTC")
    if end_at - start_at > pd.Timedelta("8D"):
        raise ValueError("Yahoo Finance fallback is limited to an eight-day one-minute window")

    from .yfinance_data import download_minute_bars as download_yahoo_minute_bars

    yahoo_to_alpaca = {symbol.upper().replace(".", "-"): symbol.upper() for symbol in symbols}
    yahoo_bars, _ = download_yahoo_minute_bars(list(yahoo_to_alpaca), start_at, end_at)
    if yahoo_bars.empty:
        return _empty_minute_bars()
    result = yahoo_bars.copy()
    result["symbol"] = result["symbol"].astype(str).str.upper().map(yahoo_to_alpaca)
    return _minute_bar_frame(result.to_dict("records"))


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
