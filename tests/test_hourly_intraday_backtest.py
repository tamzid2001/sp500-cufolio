from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from cufolio_cpu.hourly_intraday_backtest import (
    NEW_YORK,
    _quarter_hour_returns,
    build_one_hour_return_panel,
    generate_hourly_one_hour_candidate,
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


def test_candidate_uses_only_completed_labels_and_needs_no_future_execution_prices() -> None:
    bars = _minute_bars()
    decision = pd.Timestamp("2026-01-16 15:30:00+00:00")  # 10:30 New York time
    observed = bars[pd.to_datetime(bars["timestamp"], utc=True) <= decision].copy()

    candidate = generate_hourly_one_hour_candidate(
        observed,
        decision_at=decision,
        top_n=3,
        lookback_scenarios=30,
        min_training_scenarios=5,
        max_weight=0.50,
    )

    assert candidate.status["weights_generated"] is True
    assert candidate.status["target_end"] == "2026-01-16T16:30:00+00:00"
    assert pd.Timestamp(candidate.status["training_end"]) < decision
    assert np.isclose(candidate.weights["target_weight"].sum(), 1.0)
    assert (candidate.weights["target_weight"] <= 0.50 + 1e-12).all()


def test_candidate_can_forecast_a_one_hour_window_seventy_minutes_before_its_start() -> None:
    bars = _minute_bars()
    selection = pd.Timestamp("2026-01-16 14:20:00+00:00")  # 09:20 New York pre-market
    target_start = pd.Timestamp("2026-01-16 15:30:00+00:00")  # 10:30 New York
    premarket_prices = pd.DataFrame(
        [
            {"timestamp": selection, "symbol": symbol, "close": close}
            for symbol, close in {"AAA": 100.0, "BBB": 90.0, "CCC": 110.0}.items()
        ]
    )
    observed = pd.concat([bars, premarket_prices], ignore_index=True)
    observed = observed[pd.to_datetime(observed["timestamp"], utc=True) <= selection].copy()

    candidate = generate_hourly_one_hour_candidate(
        observed,
        decision_at=selection,
        target_start_at=target_start,
        top_n=3,
        lookback_scenarios=30,
        min_training_scenarios=5,
        max_weight=0.50,
    )

    assert candidate.status["weights_generated"] is True
    assert candidate.status["decision_timestamp"] == selection.isoformat()
    assert candidate.status["target_start"] == target_start.isoformat()
    assert candidate.status["target_end"] == "2026-01-16T16:30:00+00:00"
    assert pd.Timestamp(candidate.status["training_end"]) < selection
    assert np.isclose(candidate.weights["target_weight"].sum(), 1.0)


def test_candidate_keeps_a_complete_top_ranked_subset_when_other_assets_are_sparse() -> None:
    bars = _minute_bars()
    # Remove one endpoint for CCC from each historical hourly target.  AAA and
    # BBB still have a complete sample, so a full-universe complete-case filter
    # would be too strict while the candidate remains estimable.
    ccc_hourly_endpoints = pd.to_datetime(bars["timestamp"], utc=True).dt.strftime("%H:%M").eq("15:30")
    sparse = bars.loc[~((bars["symbol"] == "CCC") & ccc_hourly_endpoints)].copy()
    decision = pd.Timestamp("2026-01-16 15:30:00+00:00")
    observed = sparse[pd.to_datetime(sparse["timestamp"], utc=True) <= decision]

    candidate = generate_hourly_one_hour_candidate(
        observed,
        decision_at=decision,
        top_n=3,
        lookback_scenarios=30,
        min_training_scenarios=5,
        max_weight=0.50,
    )

    assert candidate.status["weights_generated"] is True
    assert candidate.status["training_rows"] >= 5
    assert set(candidate.weights["symbol"]).issubset({"AAA", "BBB"})
