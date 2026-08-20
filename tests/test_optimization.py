import numpy as np
import pandas as pd

from cufolio_cpu.backtest import walk_forward_rebalance
from cufolio_cpu.optimize import (
    efficient_frontier,
    forecast_diagonal_mean_variance_weights,
    mean_cvar_weights,
    mean_variance_weights,
)


def sample_returns() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        rng.normal(0.0004, 0.01, size=(36, 4)),
        columns=["AAPL", "MSFT", "NVDA", "JPM"],
        index=pd.bdate_range("2025-01-01", periods=36),
    )


def test_cpu_optimizers_respect_full_investment_and_max_weight() -> None:
    data = sample_returns()
    cvar = mean_cvar_weights(data, max_weight=0.35)
    mean_variance = mean_variance_weights(data, max_weight=0.35)
    for result in (cvar, mean_variance):
        assert np.isclose(result.weights.sum(), 1.0)
        assert (result.weights >= -1e-9).all()
        assert (result.weights <= 0.35 + 1e-8).all()


def test_mean_variance_drops_non_finite_return_scenarios() -> None:
    data = sample_returns()
    data.loc[data.index[0], "AAPL"] = np.inf

    result = mean_variance_weights(data, max_weight=0.35)

    assert np.isfinite(result.expected_return)
    assert np.isclose(result.weights.sum(), 1.0)


def test_mean_variance_regularizes_a_singular_complete_case_covariance() -> None:
    data = sample_returns()
    data["NVDA"] = 0.0  # finite, but its sample-covariance diagonal is zero

    result = mean_variance_weights(data, max_weight=0.35)

    assert np.isfinite(result.expected_return)
    assert np.isclose(result.weights.sum(), 1.0)
    assert (result.weights <= 0.35 + 1e-8).all()


def test_forecast_diagonal_mean_variance_weights_respects_caps() -> None:
    result = forecast_diagonal_mean_variance_weights(
        pd.Series({"AAPL": 0.03, "MSFT": 0.02, "NVDA": 0.01}),
        pd.Series({"AAPL": 0.004, "MSFT": 0.003, "NVDA": 0.002}),
        max_weight=0.50,
    )

    assert np.isclose(result.weights.sum(), 1.0)
    assert (result.weights >= 0).all()
    assert (result.weights <= 0.50 + 1e-8).all()


def test_frontier_and_causal_backtest_run() -> None:
    data = sample_returns()
    frontier = efficient_frontier(data, [1.0, 5.0], max_weight=0.35)
    performance, weights = walk_forward_rebalance(
        data, lookback_days=20, rebalance_every_days=5, max_weight=0.35
    )
    assert len(frontier) == 2
    assert len(performance) == len(data) - 20
    assert performance.index.min() > data.index[19]
    assert weights.index.equals(performance.index)
