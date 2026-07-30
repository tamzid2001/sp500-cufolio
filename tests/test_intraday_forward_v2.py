import numpy as np
import pandas as pd
import pytest

from cufolio_cpu.intraday_forward_v2 import build_forward_dataset, run_forward_research


def _fifteen_minute_bars_with_59_labelled_sessions() -> pd.DataFrame:
    """Create 61 sessions: the two-session label horizon leaves 59 rows."""
    rows = []
    symbols = [f"S{number:02d}" for number in range(20)]
    for day_number, session in enumerate(pd.bdate_range("2026-01-02", periods=61)):
        timestamps = pd.date_range(
            session.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30),
            periods=27,
            freq="15min",
        ).tz_convert("UTC")
        for bar_number, timestamp in enumerate(timestamps):
            sequence = day_number * len(timestamps) + bar_number
            for symbol_number, symbol in enumerate(symbols):
                intraday_shape = 0.00035 * np.sin(sequence / (5 + symbol_number % 4))
                cross_sectional_shape = 0.00003 * (symbol_number - len(symbols) / 2) * np.cos(sequence / 11)
                rows.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "close": 100 * np.exp(0.0003 * sequence + intraday_shape + cross_sectional_shape),
                    }
                )
    return pd.DataFrame(rows)


def test_v2_runs_with_59_labelled_sessions_and_purged_validation() -> None:
    bars = _fifteen_minute_bars_with_59_labelled_sessions()
    dataset, _, _ = build_forward_dataset(bars, interval="15m", horizon_minutes=500)

    assert dataset["session_date"].nunique() == 59

    result = run_forward_research(
        bars,
        interval="15m",
        horizon_minutes=500,
        top_n=10,
        max_weight=0.10,
    )

    assert result.status["model_run"] is True
    assert len(result.validation) == 5
    assert result.portfolio["target_weight"].sum() == pytest.approx(1.0)
    assert (result.portfolio["target_weight"] >= 0).all()
