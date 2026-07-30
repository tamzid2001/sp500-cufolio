import numpy as np
import pandas as pd
import pytest

from cufolio_cpu.intraday_forward_v2_backtest import run_daily_forward_v2_backtest


def _bars() -> pd.DataFrame:
    rows = []
    symbols = [f"S{number:02d}" for number in range(20)]
    for day_number, session in enumerate(pd.bdate_range("2026-01-02", periods=64)):
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


def test_daily_v2_backtest_is_causal_and_uses_exact_endpoints() -> None:
    bars = _bars()
    evaluation_day = "2026-03-27"
    baseline = run_daily_forward_v2_backtest(
        bars,
        evaluation_start=evaluation_day,
        evaluation_end=evaluation_day,
        top_n=10,
        max_weight=0.10,
    )

    assert len(baseline.ledger) == 1
    assert baseline.ledger.loc[0, "forecast_status"] == "ok"
    assert baseline.ledger.loc[0, "realized_status"] == "ok"
    assert baseline.holdings["target_weight"].sum() == pytest.approx(1.0)
    assert baseline.summary["realized_daily_signals"] == 1

    decision_at = pd.Timestamp(baseline.ledger.loc[0, "decision_timestamp"])
    changed = bars.copy()
    changed.loc[pd.to_datetime(changed["timestamp"], utc=True) > decision_at, "close"] *= 1.25
    repeated = run_daily_forward_v2_backtest(
        changed,
        evaluation_start=evaluation_day,
        evaluation_end=evaluation_day,
        top_n=10,
        max_weight=0.10,
    )

    assert repeated.holdings[["symbol", "target_weight"]].equals(
        baseline.holdings[["symbol", "target_weight"]]
    )
    assert repeated.ledger.loc[0, "actual_portfolio_simple_return"] > baseline.ledger.loc[
        0, "actual_portfolio_simple_return"
    ]
