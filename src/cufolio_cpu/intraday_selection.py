"""Full-S&P-500 15-minute forward-target preparation with Alpaca history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .intraday_forward import ForwardModelResult, run_forward_research
from .universe import current_sp500_universe

NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class IntradayForwardSelection:
    targets: pd.DataFrame
    validation: pd.DataFrame
    status: dict[str, object]


def _last_completed_15_minute_boundary(value: str | datetime | pd.Timestamp) -> datetime:
    """Return an exclusive endpoint that cannot contain an in-progress 15m bar."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(NEW_YORK)
    else:
        timestamp = timestamp.tz_convert(NEW_YORK)
    return timestamp.floor("15min").tz_convert("UTC").to_pydatetime()


def download_fifteen_minute_bars(
    symbols: list[str],
    *,
    api_key: str,
    secret_key: str,
    end: str | datetime | pd.Timestamp,
    lookback_calendar_days: int = 60,
    batch_size: int = 50,
) -> pd.DataFrame:
    """Download completed 15-minute stock bars in bounded symbol batches."""
    if lookback_calendar_days < 30:
        raise ValueError("lookback_calendar_days must be at least 30")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    end_at = _last_completed_15_minute_boundary(end)
    start_at = (pd.Timestamp(end_at) - pd.Timedelta(days=lookback_calendar_days)).to_pydatetime()

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.enums import DataFeed
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(api_key, secret_key)
    frames: list[pd.DataFrame] = []
    for offset in range(0, len(symbols), batch_size):
        request = StockBarsRequest(
            symbol_or_symbols=symbols[offset : offset + batch_size],
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            start=start_at,
            end=end_at,
            # Paper credentials commonly include IEX but not recent SIP history.
            feed=DataFeed.IEX,
        )
        frame = client.get_stock_bars(request).df
        if not frame.empty:
            frames.append(frame.reset_index().loc[:, ["timestamp", "symbol", "close"]])
    if not frames:
        return pd.DataFrame(columns=["timestamp", "symbol", "close"])
    bars = pd.concat(frames, ignore_index=True)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    return bars.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"])


def current_sp500_forward_targets(
    *,
    api_key: str,
    secret_key: str,
    as_of: str | datetime | pd.Timestamp,
    lookback_calendar_days: int = 60,
    horizon_minutes: int = 500,
    top_n: int = 20,
    max_weight: float = 0.10,
) -> IntradayForwardSelection:
    """Run the existing purged 500-trading-minute model across the full S&P 500."""
    universe = current_sp500_universe()
    # Alpaca accepts the exchange symbols for the two class-share constituents
    # (BRK.B/BF.B), whereas the universe's ``symbol`` column is Yahoo-compatible.
    symbol_column = "source_symbol" if "source_symbol" in universe.columns else "symbol"
    symbols = universe[symbol_column].astype(str).str.upper().drop_duplicates().to_list()
    bars = download_fifteen_minute_bars(
        symbols,
        api_key=api_key,
        secret_key=secret_key,
        end=as_of,
        lookback_calendar_days=lookback_calendar_days,
    )
    result: ForwardModelResult = run_forward_research(
        bars,
        interval="15m",
        horizon_minutes=horizon_minutes,
        top_n=top_n,
        max_weight=max_weight,
        min_sessions=20,
    )
    status = dict(result.status)
    status.update(
        {
            "universe_symbols_requested": len(symbols),
            "universe_source": str(universe["universe_source"].iloc[0]),
            "symbols_with_intraday_bars": int(bars["symbol"].nunique()),
            "intraday_rows": int(len(bars)),
            "history_end_exclusive": _last_completed_15_minute_boundary(as_of).isoformat(),
        }
    )
    if not status["model_run"]:
        raise RuntimeError(f"forward model did not produce a target: {status['reason']}")
    return IntradayForwardSelection(result.portfolio, result.validation, status)
