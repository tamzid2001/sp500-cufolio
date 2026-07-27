import numpy as np
import pandas as pd

from cufolio_cpu.backtest import walk_forward_rebalance
from cufolio_cpu.optimize import efficient_frontier, mean_cvar_weights, mean_variance_weights


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
