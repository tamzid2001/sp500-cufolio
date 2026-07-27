from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from cufolio_cpu.hourly_intraday_backtest import (
    NEW_YORK,
    _quarter_hour_returns,
    build_one_hour_return_panel,
    run_hourly_one_hour_backtest,
)


def _minute_bars(sessions: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    prices = {"AAA": 100.0, "BBB": 90.0, "CCC": 110.0}
    rows: list[dict[str, object]] = []
    for day_number, session in enumerate(pd.bdate_range("2026-01-02", periods=sessions)):
        timestamps = pd.date_range(
            session + timedelta(hours=9, minutes=30),
            periods=391,
            freq="min",
            tz=NEW_YORK,
        )
        market = rng.normal(0.00001, 0.00012, len(timestamps) - 1)
        for symbol_number, symbol in enumerate(prices):
            intraday = np.r_[0.0, market + rng.normal(0.00001 * (symbol_number + 1), 0.00008, len(market))]
            values = prices[symbol] * np.exp(np.cumsum(intraday))
            prices[symbol] = float(values[-1])
            rows.extend(
                {"timestamp": timestamp.tz_convert("UTC"), "symbol": symbol, "close": float(close)}
                for timestamp, close in zip(timestamps, values, strict=True)
            )
    return pd.DataFrame(rows)


def test_one_hour_targets_are_exact_same_session_windows() -> None:
    hourly, ends, _ = build_one_hour_return_panel(_minute_bars())

    assert len(hourly) == 12 * 6
    assert ((ends - hourly.index) == timedelta(minutes=60)).all()
    starts = hourly.index.tz_convert(NEW_YORK)
    assert set(starts.hour) == {9, 10, 11, 12, 13, 14}
    assert (starts.normalize() == ends.dt.tz_convert(NEW_YORK).dt.normalize()).all()


def test_backtest_selects_hourly_and_rebalances_every_fifteen_minutes_causally() -> None:
    result = run_hourly_one_hour_backtest(
        _minute_bars(),
        top_n=3,
        lookback_scenarios=30,
        min_training_scenarios=5,
        max_weight=0.50,
        transaction_cost_bps=5.0,
    )

    selected = result.selections.dropna(subset=["symbol"])
    assert not selected.empty
    assert (pd.to_datetime(selected["training_end"], utc=True) < pd.to_datetime(selected["decision_timestamp"], utc=True)).all()
    assert np.allclose(selected.groupby("decision_timestamp")["target_weight"].sum(), 1.0)
    assert (selected["target_weight"] >= 0).all()
    assert (selected["target_weight"] <= 0.50 + 1e-12).all()

    performance = result.performance
    assert not performance.empty
    assert (pd.to_datetime(performance["interval_end"], utc=True) - pd.to_datetime(performance["rebalance_timestamp"], utc=True) == timedelta(minutes=15)).all()
    assert performance.groupby("hourly_target_start").size().eq(4).all()
    assert result.status["execution_rebalance_frequency_minutes"] == 15
    assert result.status["target_horizon_minutes"] == 60


def test_missing_execution_endpoint_is_not_forward_filled() -> None:
    bars = _minute_bars(sessions=1)
    _, _, closes = build_one_hour_return_panel(bars)
    start = pd.Timestamp("2026-01-02 14:30:00+00:00")
    missing_timestamp = start + timedelta(minutes=15)
    closes.loc[missing_timestamp, "AAA"] = np.nan

    assert _quarter_hour_returns(closes, start, pd.Index(["AAA", "BBB"])) is None
