"""Convert regular-session minute bars into causally usable daily returns."""

from __future__ import annotations

from collections.abc import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NEW_YORK = ZoneInfo("America/New_York")
REQUIRED_MINUTE_COLUMNS = {"timestamp", "symbol", "close"}


def _validate_minute_bars(minute_bars: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_MINUTE_COLUMNS.difference(minute_bars.columns)
    if missing:
        raise ValueError(f"minute bars are missing required columns: {sorted(missing)}")
    bars = minute_bars.loc[:, ["timestamp", "symbol", "close"]].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
    bars["symbol"] = bars["symbol"].astype(str).str.upper().str.strip()
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    bars = bars.dropna(subset=["timestamp", "symbol", "close"])
    bars = bars[bars["close"] > 0]
    if bars.empty:
        raise ValueError("no positive, timestamped minute bars remain after cleaning")
    return bars.sort_values(["symbol", "timestamp"]).drop_duplicates(
        ["symbol", "timestamp"], keep="last"
    )


def minute_log_returns(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """Return one-minute log returns confined to the US regular session.

    The first bar of each symbol/session is deliberately NaN: it is not an
    intraday return and must not carry an overnight return into daily P/L.
    """
    bars = _validate_minute_bars(minute_bars)
    local = bars["timestamp"].dt.tz_convert(NEW_YORK)
    clock = local.dt.time
    regular = (clock >= pd.Timestamp("09:30").time()) & (clock <= pd.Timestamp("16:00").time())
    bars = bars.loc[regular].copy()
    if bars.empty:
        raise ValueError("no bars fall within the 09:30–16:00 America/New_York session")
    bars["session_date"] = local.loc[regular].dt.tz_localize(None).dt.normalize()
    bars["log_close"] = np.log(bars["close"])
    bars["minute_log_return"] = bars.groupby(["symbol", "session_date"], sort=False)[
        "log_close"
    ].diff()
    return bars.loc[:, ["timestamp", "session_date", "symbol", "close", "minute_log_return"]]


def daily_asset_log_returns(
    minute_bars: pd.DataFrame, min_minutes_per_session: int = 300
) -> pd.DataFrame:
    """Aggregate intraday log returns, retaining only adequately covered sessions.

    A regular US session has 390 one-minute intervals. The conservative default
    permits holidays/market-data gaps only when at least 300 return observations
    are available. Missing coverage stays ``NaN``; it is never converted to 0.
    """
    if min_minutes_per_session < 1:
        raise ValueError("min_minutes_per_session must be positive")
    logs = minute_log_returns(minute_bars).dropna(subset=["minute_log_return"])
    grouped = logs.groupby(["session_date", "symbol"], sort=True)["minute_log_return"].agg(
        daily_log_return="sum", observed_minutes="count"
    )
    returns = grouped["daily_log_return"].unstack("symbol").sort_index()
    coverage = grouped["observed_minutes"].unstack("symbol").reindex_like(returns)
    return returns.where(coverage >= min_minutes_per_session)


def daily_returns_from_minute_bars(
    minute_bars: pd.DataFrame, min_minutes_per_session: int = 300
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-asset daily log and simple return matrices, indexed by session date."""
    daily_log = daily_asset_log_returns(minute_bars, min_minutes_per_session)
    return daily_log, np.expm1(daily_log)


def _as_weight_frame(
    weights: pd.Series | pd.DataFrame,
    index: Iterable[pd.Timestamp],
    columns: Iterable[str],
) -> pd.DataFrame:
    requested_index = pd.DatetimeIndex(index)
    if isinstance(weights, pd.Series):
        values = weights.reindex(list(columns)).fillna(0.0).to_numpy(dtype=float)
        return pd.DataFrame(
            np.repeat(values[None, :], len(requested_index), axis=0),
            index=requested_index,
            columns=list(columns),
        )
    else:
        frame = weights.copy()
        frame.index = pd.to_datetime(frame.index)
    frame.columns = frame.columns.astype(str)
    frame = frame.reindex(columns=list(columns))
    expanded_index = frame.index.union(requested_index).sort_values()
    return frame.reindex(expanded_index).ffill().reindex(requested_index).fillna(0.0)


def portfolio_daily_returns(
    daily_simple_returns: pd.DataFrame,
    weights: pd.Series | pd.DataFrame,
    *,
    lag_weights: bool = True,
) -> pd.DataFrame:
    """Compute portfolio returns with explicit missing-data and timing treatment.

    ``lag_weights=True`` is the safe default: weights published at date *t* are
    applied at *t+1*. If any held asset has no valid return for a date, that
    portfolio date is marked missing rather than inventing a zero return.
    """
    returns = daily_simple_returns.copy().sort_index()
    returns.index = pd.to_datetime(returns.index)
    weights_frame = _as_weight_frame(weights, returns.index, returns.columns)
    if lag_weights:
        weights_frame = weights_frame.shift(1).fillna(0.0)
    held_missing = returns.isna() & weights_frame.ne(0.0)
    simple = (returns.fillna(0.0) * weights_frame).sum(axis=1)
    simple = simple.mask(held_missing.any(axis=1))
    return pd.DataFrame(
        {
            "portfolio_simple_return": simple,
            "portfolio_log_return": np.log1p(simple),
            "gross_exposure": weights_frame.abs().sum(axis=1),
        }
    )
