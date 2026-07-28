from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from cufolio_cpu.hourly_intraday_backtest import (
    NEW_YORK,
    _quarter_hour_returns,
    build_one_hour_return_panel,
    generate_to_close_candidate,
    generate_hourly_one_hour_candidate,
    hourly_paper_cadence_endpoint_times,
    run_hourly_paper_cadence_backtest,
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
    assert candidate.status["required_candidates_for_weight_cap"] == 2
    assert pd.Timestamp(candidate.status["training_end"]) < decision
    assert np.isclose(candidate.weights["target_weight"].sum(), 1.0)
    assert (candidate.weights["target_weight"] > 0).all()
    assert (candidate.weights["target_weight"] <= 0.50 + 1e-12).all()


def test_same_clock_time_candidate_can_target_a_shortened_session_close_window() -> None:
    bars = _minute_bars()
    decision = pd.Timestamp("2026-01-16T19:40:00Z")  # 14:40 New York
    target_end = pd.Timestamp("2026-01-16T20:30:00Z")  # 15:30 New York
    observed = bars[pd.to_datetime(bars["timestamp"], utc=True) <= decision].copy()

    candidate = generate_to_close_candidate(
        observed,
        decision_at=decision,
        target_end_at=target_end,
        top_n=3,
        lookback_scenarios=30,
        min_training_scenarios=5,
        max_weight=0.50,
    )

    assert candidate.status["weights_generated"] is True
    assert candidate.status["target_horizon_minutes"] == 50
    assert pd.Timestamp(candidate.status["training_end"]) < decision
    assert np.isclose(candidate.weights["target_weight"].sum(), 1.0)
    assert (candidate.weights["target_weight"] > 0).all()


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


def test_paper_cadence_backtest_uses_live_forecast_leads_and_exact_quarter_hour_returns() -> None:
    bars = _minute_bars()
    premarket_rows: list[dict[str, object]] = []
    for session in pd.bdate_range("2026-01-02", periods=12):
        session_open = pd.Timestamp(session, tz=NEW_YORK) + pd.Timedelta(hours=9, minutes=30)
        day_opens = bars.loc[pd.to_datetime(bars["timestamp"], utc=True).eq(session_open.tz_convert("UTC"))]
        for selection_time in ("09:20", "10:20", "11:20", "12:20", "14:20"):
            selection = pd.Timestamp(f"{session.date()} {selection_time}", tz=NEW_YORK).tz_convert("UTC")
            for row in day_opens.itertuples(index=False):
                premarket_rows.append({"timestamp": selection, "symbol": row.symbol, "close": row.close})
    result = run_hourly_paper_cadence_backtest(
        pd.concat([bars, pd.DataFrame(premarket_rows)], ignore_index=True),
        top_n=3,
        lookback_scenarios=30,
        min_training_scenarios=5,
        max_weight=0.50,
        transaction_cost_bps=0.0,
    )

    realized = result.forecasts.loc[result.forecasts["forecast_status"].eq("selected_and_realized")]
    assert not realized.empty
    decisions = pd.to_datetime(realized["decision_timestamp"], utc=True).dt.tz_convert(NEW_YORK)
    targets = pd.to_datetime(realized["target_start"], utc=True).dt.tz_convert(NEW_YORK)
    assert set(decisions.dt.strftime("%H:%M")) == {"09:20", "10:20", "11:20", "12:20", "14:20"}
    assert set(targets.dt.strftime("%H:%M")) == {"10:30", "11:30", "12:30", "13:30", "14:30"}
    assert (
        pd.to_datetime(result.selections["training_end"], utc=True)
        < pd.to_datetime(result.selections["decision_timestamp"], utc=True)
    ).all()
    assert result.performance.groupby("hourly_target_start").size().eq(4).all()
    assert result.status["forecast_windows_realized_exactly"] == len(realized)
    assert result.status["forecast_coverage"] > 0
    assert result.status["quarter_hour_rebalance_rows"] == len(result.performance)
    assert result.status["rebalances_per_exact_realized_window"] == 4


def test_paper_cadence_endpoint_set_covers_decisions_labels_and_all_rebalance_boundaries() -> None:
    endpoints = hourly_paper_cadence_endpoint_times()

    assert {pd.Timestamp.combine(pd.Timestamp("2026-01-02"), endpoint).strftime("%H:%M") for endpoint in endpoints} >= {
        "09:20", "09:30", "10:20", "10:30", "10:45", "11:00", "11:15", "11:30",
        "12:00", "12:15", "12:20", "12:30", "13:00", "13:15", "13:30", "14:00",
        "14:15", "14:20", "14:30", "14:45", "15:00", "15:15", "15:30",
    }
