"""Small, transparent CPU implementations of the blueprint's core optimizers."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series
    expected_return: float
    cvar: float | None
    status: str


def _coerce_returns(returns: pd.DataFrame) -> pd.DataFrame:
    clean = returns.dropna(axis=0, how="any").astype(float)
    if len(clean) < 5 or clean.shape[1] < 2:
        raise ValueError("at least five complete observations and two assets are required")
    return clean


def mean_cvar_weights(
    scenarios: pd.DataFrame,
    *,
    risk_aversion: float = 5.0,
    confidence: float = 0.95,
    max_weight: float = 0.25,
    current_weights: pd.Series | None = None,
    turnover_limit: float | None = None,
) -> OptimizationResult:
    """Solve a long-only, fully invested Mean–CVaR linear program on CPU."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    returns = _coerce_returns(scenarios)
    n_obs, n_assets = returns.shape
    if max_weight * n_assets < 1 - 1e-12:
        raise ValueError("max_weight is too small to construct a fully invested portfolio")
    values = returns.to_numpy()
    expected = values.mean(axis=0)
    weights = cp.Variable(n_assets)
    value_at_risk = cp.Variable()
    tail_losses = cp.Variable(n_obs, nonneg=True)
    losses = -values @ weights
    cvar = value_at_risk + cp.sum(tail_losses) / ((1 - confidence) * n_obs)
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= max_weight,
        tail_losses >= losses - value_at_risk,
    ]
    if turnover_limit is not None:
        if current_weights is None:
            raise ValueError("current_weights is required when turnover_limit is set")
        prior = current_weights.reindex(returns.columns).fillna(0.0).to_numpy(dtype=float)
        constraints.append(cp.norm1(weights - prior) <= turnover_limit)
    problem = cp.Problem(cp.Maximize(expected @ weights - risk_aversion * cvar), constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weights.value is None:
        raise RuntimeError(f"Mean–CVaR solve failed with status {problem.status}")
    result_weights = pd.Series(np.asarray(weights.value).ravel(), index=returns.columns).clip(lower=0)
    result_weights /= result_weights.sum()
    realized_losses = -(values @ result_weights.to_numpy())
    var = np.quantile(realized_losses, confidence)
    realized_cvar = float(realized_losses[realized_losses >= var].mean())
    return OptimizationResult(
        weights=result_weights,
        expected_return=float(expected @ result_weights.to_numpy()),
        cvar=realized_cvar,
        status=str(problem.status),
    )


def mean_variance_weights(
    returns: pd.DataFrame, *, risk_aversion: float = 10.0, max_weight: float = 0.25
) -> OptimizationResult:
    """Solve the long-only full-investment Markowitz baseline on CPU."""
    clean = _coerce_returns(returns)
    n_assets = clean.shape[1]
    if max_weight * n_assets < 1 - 1e-12:
        raise ValueError("max_weight is too small to construct a fully invested portfolio")
    mean = clean.mean().to_numpy()
    covariance = clean.cov().to_numpy() + np.eye(n_assets) * 1e-10
    weights = cp.Variable(n_assets)
    problem = cp.Problem(
        cp.Maximize(mean @ weights - risk_aversion * cp.quad_form(weights, covariance)),
        [cp.sum(weights) == 1, weights >= 0, weights <= max_weight],
    )
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weights.value is None:
        raise RuntimeError(f"Mean–variance solve failed with status {problem.status}")
    vector = np.asarray(weights.value).ravel()
    return OptimizationResult(
        weights=pd.Series(vector, index=clean.columns),
        expected_return=float(mean @ vector),
        cvar=None,
        status=str(problem.status),
    )


def efficient_frontier(
    returns: pd.DataFrame, risk_aversions: list[float], max_weight: float = 0.25
) -> pd.DataFrame:
    """Return one Mean–CVaR solution per risk-aversion setting."""
    rows = []
    for risk_aversion in risk_aversions:
        result = mean_cvar_weights(
            returns, risk_aversion=risk_aversion, max_weight=max_weight
        )
        rows.append(
            {
                "risk_aversion": risk_aversion,
                "expected_return": result.expected_return,
                "cvar": result.cvar,
                "status": result.status,
                "weights": result.weights.to_dict(),
            }
        )
    return pd.DataFrame(rows)
