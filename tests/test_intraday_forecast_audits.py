from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from cufolio_cpu.fifteen_minute_intraday_backtest import run_fifteen_minute_forecast_backtest
from cufolio_cpu.five_minute_intraday_backtest import (
    NEW_YORK,
    RealtimeIntradayForecastEngine,
    generate_intraday_forecast_candidate,
    _render_markdown_report,
    reconcile_intraday_forecast_outcomes,
    run_five_minute_forecast_backtest,
    run_intraday_forecast_backtest_range,
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


def test_five_minute_candidate_uses_the_prior_completed_decision_bar() -> None:
    bars = _minute_bars()
    decision = pd.Timestamp("2026-01-19T14:35:00Z")  # 09:35 New York
    baseline = generate_intraday_forecast_candidate(
        bars,
        decision_at=decision,
        top_n=3,
        max_weight=0.50,
        lookback_observations=780,
        min_training_sessions=10,
        min_covariance_scenarios=500,
    )
    # The 09:35-labelled bar is not complete until 09:36. Removing all of
    # those future-at-decision observations must not change the 09:35 target.
    without_incomplete_bar = bars.loc[pd.to_datetime(bars["timestamp"], utc=True).ne(decision)].copy()
    causal = generate_intraday_forecast_candidate(
        without_incomplete_bar,
        decision_at=decision,
        top_n=3,
        max_weight=0.50,
        lookback_observations=780,
        min_training_sessions=10,
        min_covariance_scenarios=500,
    )

    assert baseline.status["weights_generated"]
    assert causal.status["weights_generated"]
    assert baseline.status["decision_price_timestamp"] == "2026-01-19T14:34:00+00:00"
    pd.testing.assert_frame_equal(baseline.weights, causal.weights)


def test_realtime_engine_prepares_training_before_the_final_decision_bar_and_matches_full_rebuild() -> None:
    bars = _minute_bars()
    decision = pd.Timestamp("2026-01-19T14:35:00Z")
    # At 09:34:00 the 09:33 bar is the latest completed minute. The model's
    # training labels are already fixed, but the 09:34 close needed for the
    # final current-price eligibility screen is not available yet.
    timestamps = pd.to_datetime(bars["timestamp"], utc=True)
    before_final_bar = bars.loc[
        timestamps.lt(pd.Timestamp("2026-01-19T14:34:00Z")) | timestamps.dt.date.ne(decision.date())
    ].copy()
    final_bar = bars.loc[timestamps.eq(pd.Timestamp("2026-01-19T14:34:00Z"))].copy()
    engine = RealtimeIntradayForecastEngine(
        before_final_bar,
        top_n=3,
        max_weight=0.50,
        lookback_observations=780,
        min_training_sessions=10,
        min_covariance_scenarios=500,
    )
    engine.prepare_for_decision(decision, prepared_at="2026-01-19T14:34:00Z")
    engine.update_minute_bars(final_bar)
    prepared_candidate = engine.generate_candidate(decision)
    full_rebuild = generate_intraday_forecast_candidate(
        bars,
        decision_at=decision,
        top_n=3,
        max_weight=0.50,
        lookback_observations=780,
        min_training_sessions=10,
        min_covariance_scenarios=500,
    )

    assert prepared_candidate.status["training_prepared_at"] == "2026-01-19T14:34:00+00:00"
    assert prepared_candidate.status["decision_price_timestamp"] == "2026-01-19T14:34:00+00:00"
    pd.testing.assert_frame_equal(prepared_candidate.weights, full_rebuild.weights)
    assert prepared_candidate.status["expected_portfolio_log_return"] == full_rebuild.status[
        "expected_portfolio_log_return"
    ]


def test_closing_four_minute_engine_uses_four_minute_labels_and_ends_at_1559() -> None:
    bars = _minute_bars()
    decision = pd.Timestamp("2026-01-19T20:55:00Z")  # 15:55 New York
    timestamps = pd.to_datetime(bars["timestamp"], utc=True)
    # At 15:54 the 15:53 bar is complete.  Its 15:50 -> 15:54 training
    # return is known, while the 15:54 decision-price bar is not yet usable.
    before_final_bar = bars.loc[
        timestamps.lt(pd.Timestamp("2026-01-19T20:54:00Z")) | timestamps.dt.date.ne(decision.date())
    ].copy()
    final_bar = bars.loc[timestamps.eq(pd.Timestamp("2026-01-19T20:54:00Z"))].copy()
    options = {
        "interval_minutes": 5,
        "forecast_horizon_minutes": 4,
        "top_n": 3,
        "max_weight": 0.50,
        "lookback_observations": 780,
        "min_training_sessions": 10,
        "min_covariance_scenarios": 500,
    }
    engine = RealtimeIntradayForecastEngine(before_final_bar, **options)
    engine.prepare_for_decision(decision, prepared_at="2026-01-19T20:54:00Z")
    engine.update_minute_bars(final_bar)
    prepared = engine.generate_candidate(decision)
    full = generate_intraday_forecast_candidate(bars, decision_at=decision, **options)

    assert prepared.status["forecast_interval_minutes"] == 5
    assert prepared.status["forecast_horizon_minutes"] == 4
    assert prepared.status["target_end"] == "2026-01-19T20:59:00+00:00"
    assert prepared.status["training_prepared_at"] == "2026-01-19T20:54:00+00:00"
    pd.testing.assert_frame_equal(prepared.weights, full.weights)
    assert prepared.status["expected_portfolio_log_return"] == full.status[
        "expected_portfolio_log_return"
    ]


def test_trailing_sixteen_session_cache_preserves_the_780_label_target() -> None:
    bars = _minute_bars(sessions=20)
    final_day = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(NEW_YORK).dt.date.max()
    decision = pd.Timestamp.combine(final_day, pd.Timestamp("09:35").time()).tz_localize(NEW_YORK).tz_convert("UTC")
    local_days = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert(NEW_YORK).dt.date
    retained_days = sorted(local_days.unique())[-16:]
    compact = bars.loc[local_days.isin(retained_days)].copy()
    options = {
        "top_n": 3,
        "max_weight": 0.50,
        "lookback_observations": 780,
        "min_training_sessions": 10,
        "min_covariance_scenarios": 500,
    }
    full = RealtimeIntradayForecastEngine(bars, **options).generate_candidate(decision)
    cached = RealtimeIntradayForecastEngine(compact, **options).generate_candidate(decision)

    pd.testing.assert_frame_equal(full.weights, cached.weights)
    assert full.status["expected_portfolio_log_return"] == cached.status["expected_portfolio_log_return"]


def test_multi_session_five_minute_audit_keeps_each_session_and_causal_boundaries() -> None:
    result = run_intraday_forecast_backtest_range(
        _minute_bars(),
        interval_minutes=5,
        evaluation_start="2026-01-16",
        evaluation_end="2026-01-19",
        top_n=3,
        max_weight=0.50,
        lookback_observations=780,
        min_training_sessions=10,
        min_covariance_scenarios=500,
    )
    assert result.summary["evaluation_sessions"] == 2
    assert result.summary["evaluation_start"] == "2026-01-16"
    assert result.summary["evaluation_end"] == "2026-01-19"
    assert len(result.ledger) == 154
    assert result.ledger["session_date"].nunique() == 2
    assert set(result.ledger["forecast_status"]) == {"ok"}
    assert (
        pd.to_datetime(result.ledger["training_end"], utc=True)
        < pd.to_datetime(result.ledger["decision_timestamp"], utc=True)
    ).all()


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
