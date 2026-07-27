"""Causal daily rebalancing backtest for the CPU notebook examples."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .optimize import mean_cvar_weights


def walk_forward_rebalance(
    daily_simple_returns: pd.DataFrame,
    *,
    lookback_days: int = 20,
    rebalance_every_days: int = 5,
    risk_aversion: float = 5.0,
    max_weight: float = 0.30,
    transaction_cost_bps: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest, choosing weights using dates strictly before each held date."""
    returns = daily_simple_returns.dropna(axis=0, how="any").sort_index()
    if len(returns) <= lookback_days:
        raise ValueError("more complete days than lookback_days are required")
    weights = pd.Series(1 / returns.shape[1], index=returns.columns, dtype=float)
    history = []
    weights_history = []
    for position in range(lookback_days, len(returns)):
        date = returns.index[position]
        turnover = 0.0
        if (position - lookback_days) % rebalance_every_days == 0:
            training = returns.iloc[position - lookback_days : position]
            result = mean_cvar_weights(
                training,
                risk_aversion=risk_aversion,
                max_weight=max_weight,
                current_weights=weights,
                turnover_limit=1.5,
            )
            new_weights = result.weights
            turnover = float((new_weights - weights).abs().sum())
            weights = new_weights
        gross = float(returns.iloc[position] @ weights)
        cost = turnover * transaction_cost_bps / 10_000
        history.append((date, gross - cost, gross, cost, turnover))
        weights_history.append((date, *weights.reindex(returns.columns).to_list()))
    performance = pd.DataFrame(
        history,
        columns=["date", "net_simple_return", "gross_simple_return", "transaction_cost", "turnover"],
    ).set_index("date")
    performance["equity"] = (1 + performance["net_simple_return"]).cumprod()
    weights_frame = pd.DataFrame(weights_history, columns=["date", *returns.columns]).set_index("date")
    return performance, weights_frame
