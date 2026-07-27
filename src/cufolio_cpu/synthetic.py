"""Deterministic synthetic minute bars for tests and notebook execution."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_minute_bars(
    symbols: list[str] | None = None, *, sessions: int = 45, seed: int = 42
) -> pd.DataFrame:
    """Generate regular-session US-equity-style minute closes with shared factors."""
    symbols = symbols or ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "XOM", "JNJ"]
    rng = np.random.default_rng(seed)
    session_dates = pd.bdate_range("2025-01-02", periods=sessions)
    asset_bias = rng.normal(0.0, 0.000025, len(symbols))
    prices = np.linspace(75, 250, len(symbols))
    records: list[tuple[pd.Timestamp, str, float]] = []
    for date in session_dates:
        # 390 returns after the 09:30 opening close observation.  The 16:00 bar
        # is included as the final timestamp.
        clock = pd.date_range(f"{date.date()} 09:30", periods=391, freq="min", tz="America/New_York")
        factor = rng.normal(0.0, 0.00045, len(clock) - 1)
        idiosyncratic = rng.normal(0.0, 0.0007, (len(clock) - 1, len(symbols)))
        returns = factor[:, None] * 0.35 + idiosyncratic + asset_bias
        day_prices = np.vstack([prices, prices * np.exp(np.cumsum(returns, axis=0))])
        prices = day_prices[-1]
        for column, symbol in enumerate(symbols):
            records.extend((timestamp.tz_convert("UTC"), symbol, float(value)) for timestamp, value in zip(clock, day_prices[:, column]))
    return pd.DataFrame(records, columns=["timestamp", "symbol", "close"])
