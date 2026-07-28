from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from cufolio_cpu.fifteen_minute_intraday_backtest import run_fifteen_minute_forecast_backtest
from cufolio_cpu.five_minute_intraday_backtest import (
    NEW_YORK,
    _render_markdown_report,
    reconcile_intraday_forecast_outcomes,
    run_five_minute_forecast_backtest,
)


def _minute_bars(sessions: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(73)
    prices = {"AAA": 100.0, "BBB": 90.0, "CCC": 110.0}
    rows: list[dict[str, object]] = []
    for session in pd.bdate_range("2026-01-02", periods=sessions):
        timestamps = pd.date_range(
            session + timedelta(hours=9, minutes=30), periods=391, freq="min", tz=NEW_YORK
        )
        market = rng.normal(0.0, 0.00018, len(timestamps) - 1)
        for number, symbol in enumerate(prices):
            returns = np.r_[0.0, market + rng.normal(0.000005 * (number + 1), 0.00006, len(market))]
            values = prices[symbol] * np.exp(np.cumsum(returns))
            prices[symbol] = float(values[-1])
            rows.extend(
                {"timestamp": timestamp.tz_convert("UTC"), "symbol": symbol, "close": float(close)}
                for timestamp, close in zip(timestamps, values, strict=True)
            )
    return pd.DataFrame(rows)


def test_five_minute_audit_is_causal_and_reconciles_each_non_overlapping_window() -> None:
    bars = _minute_bars()
    result = run_five_minute_forecast_backtest(
        bars,
        session_date="2026-01-19",
        top_n=3,
        max_weight=0.50,
        lookback_observations=780,
        min_training_sessions=10,
        min_covariance_scenarios=500,
    )

    ledger = result.ledger
    assert len(ledger) == 77
    assert set(ledger["forecast_status"]) == {"ok"}
    assert set(ledger["realized_status"]) == {"ok"}
    assert (pd.to_datetime(ledger["target_end"], utc=True) - pd.to_datetime(ledger["decision_timestamp"], utc=True) == timedelta(minutes=5)).all()
    assert (
        pd.to_datetime(ledger["decision_price_timestamp"], utc=True)
        == pd.to_datetime(ledger["decision_timestamp"], utc=True) - timedelta(minutes=1)
    ).all()
    assert (
        pd.to_datetime(ledger["realized_price_timestamp"], utc=True)
        == pd.to_datetime(ledger["target_end"], utc=True) - timedelta(minutes=1)
    ).all()
    assert (pd.to_datetime(ledger["training_end"], utc=True) < pd.to_datetime(ledger["decision_timestamp"], utc=True)).all()
    assert np.allclose(result.holdings.groupby("decision_timestamp")["target_weight"].sum(), 1.0)
    assert (result.holdings["target_weight"] > 0).all()
    assert (result.holdings["target_weight"] <= 0.50 + 1e-12).all()
    assert result.summary["realized_windows"] == 77
    assert result.summary["compounded_actual_return"] == np.expm1(ledger["actual_portfolio_log_return"].sum())


def test_future_target_prices_do_not_change_the_first_five_minute_forecast() -> None:
    bars = _minute_bars()
    baseline = run_five_minute_forecast_backtest(
        bars,
        session_date="2026-01-19",
        top_n=3,
        max_weight=0.50,
        lookback_observations=780,
        min_training_sessions=10,
        min_covariance_scenarios=500,
    )
    edited = bars.copy()
    target_future = (pd.to_datetime(edited["timestamp"], utc=True) >= pd.Timestamp("2026-01-19T14:35:00Z"))
    edited.loc[target_future, "close"] *= 1.5
    changed = run_five_minute_forecast_backtest(
        edited,
        session_date="2026-01-19",
        top_n=3,
        max_weight=0.50,
        lookback_observations=780,
        min_training_sessions=10,
        min_covariance_scenarios=500,
    )

    pd.testing.assert_frame_equal(
        baseline.holdings[baseline.holdings["decision_timestamp"] == pd.Timestamp("2026-01-19T14:30:00Z")]
        .loc[:, ["symbol", "target_weight", "predicted_asset_log_return"]]
        .reset_index(drop=True),
        changed.holdings[changed.holdings["decision_timestamp"] == pd.Timestamp("2026-01-19T14:30:00Z")]
        .loc[:, ["symbol", "target_weight", "predicted_asset_log_return"]]
        .reset_index(drop=True),
    )


def test_missing_realized_endpoint_marks_a_forecast_incomplete_without_reweighting() -> None:
    bars = _minute_bars()
    baseline = run_five_minute_forecast_backtest(
        bars,
        session_date="2026-01-19",
        top_n=3,
        max_weight=0.50,
        lookback_observations=780,
        min_training_sessions=10,
        min_covariance_scenarios=500,
    )
    first_symbol = baseline.holdings.loc[
        baseline.holdings["decision_timestamp"] == baseline.ledger.iloc[0]["decision_timestamp"], "symbol"
    ].iloc[0]
    timestamps = pd.to_datetime(bars["timestamp"], utc=True)
    missing = timestamps.eq(pd.Timestamp("2026-01-19T14:39:00Z")) & bars["symbol"].eq(first_symbol)
    incomplete = run_five_minute_forecast_backtest(
        bars.loc[~missing].copy(),
        session_date="2026-01-19",
        top_n=3,
        max_weight=0.50,
        lookback_observations=780,
        min_training_sessions=10,
        min_covariance_scenarios=500,
    )

    first = incomplete.ledger.iloc[0]
    assert first["forecast_status"] == "ok"
    assert first["realized_status"] == "missing_exact_realized_endpoint"
    assert pd.isna(first["actual_portfolio_simple_return"])

    repaired = reconcile_intraday_forecast_outcomes(incomplete, bars)
    repaired_first = repaired.ledger.iloc[0]
    assert repaired_first["realized_status"] == "ok"
    assert repaired.summary["fallback_repaired_asset_endpoints"] >= 1
    assert repaired.summary["full_session_exact_realization_available"]


def test_fifteen_minute_audit_has_its_own_non_overlapping_interval_and_sample_gate() -> None:
    result = run_fifteen_minute_forecast_backtest(
        _minute_bars(),
        session_date="2026-01-19",
        top_n=3,
        max_weight=0.50,
        lookback_observations=260,
        min_training_sessions=10,
        min_covariance_scenarios=200,
    )

    assert len(result.ledger) == 25
    assert set(result.ledger["forecast_status"]) == {"ok"}
    assert result.summary["forecast_interval_minutes"] == 15
    assert result.summary["forecast_horizon_minutes"] == 15
    assert (pd.to_datetime(result.ledger["target_end"], utc=True) - pd.to_datetime(result.ledger["decision_timestamp"], utc=True) == timedelta(minutes=15)).all()


def test_report_discloses_arithmetic_and_compounded_reconciliation() -> None:
    result = run_five_minute_forecast_backtest(
        _minute_bars(),
        session_date="2026-01-19",
        top_n=3,
        max_weight=0.50,
        lookback_observations=780,
        min_training_sessions=10,
        min_covariance_scenarios=500,
    )
    text = _render_markdown_report(result)
    assert "Causal 5-minute forecast audit" in text
    assert "All-forecast arithmetic expected return" in text
    assert "Matched-window compounded actual return" in text
