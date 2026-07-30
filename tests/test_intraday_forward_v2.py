import numpy as np
import pandas as pd
import pytest

from cufolio_cpu.intraday_forward_v2 import (
    _select_diagonal_covariance_assets,
    _select_covariance_scenarios,
    build_forward_dataset,
    run_forward_research,
)


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


def test_v2_uses_a_ranked_cohort_with_complete_covariance_scenarios() -> None:
    sessions = pd.date_range("2026-01-02", periods=6, freq="B")
    rows = []
    high_ranked = [f"HIGH{number:02d}" for number in range(10)]
    viable = [f"VIABLE{number:02d}" for number in range(10)]
    for number, symbol in enumerate(high_ranked):
        rows.append(
            {
                "session_date": sessions[number % 4],
                "symbol": symbol,
                "target_excess_return": 0.01,
            }
        )
    for number, symbol in enumerate(viable):
        for session in sessions:
            rows.append(
                {
                    "session_date": session,
                    "symbol": symbol,
                    "target_excess_return": 0.001 * (number + 1),
                }
            )
    prediction = pd.Series(
        [*range(20, 10, -1), *range(10, 0, -1)],
        index=[*high_ranked, *viable],
        dtype=float,
    )

    scenarios, selected = _select_covariance_scenarios(
        pd.DataFrame(rows), prediction, top_n=20, max_weight=0.10
    )

    assert selected.tolist() == viable
    assert scenarios.shape == (6, 10)


def test_v2_uses_individual_return_histories_when_no_complete_cohort_exists() -> None:
    sessions = pd.date_range("2026-01-02", periods=6, freq="B")
    symbols = [f"SPARSE{number:02d}" for number in range(10)]
    history = pd.DataFrame(
        {
            "session_date": [session for _ in symbols for session in sessions],
            "symbol": [symbol for symbol in symbols for _ in sessions],
            "target_excess_return": [0.001 * (number + 1) for number in range(60)],
        }
    )
    prediction = pd.Series(range(10, 0, -1), index=symbols, dtype=float)

    selected, variances, minimum_observations = _select_diagonal_covariance_assets(
        history, prediction, top_n=10, max_weight=0.10
    )

    assert selected.index.tolist() == symbols
    assert variances.index.tolist() == symbols
    assert minimum_observations == 6
