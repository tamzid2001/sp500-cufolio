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
    # An invalid market-data endpoint makes its complete return scenario
    # unusable. Never pass an infinity through to a covariance solver.
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").astype(float)
    if len(clean) < 5 or clean.shape[1] < 2:
        raise ValueError("at least five complete observations and two assets are required")
    if not np.isfinite(clean.to_numpy(dtype=float)).all():
        raise ValueError("return scenarios must be finite")
    return clean


def _regularized_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Return a finite, symmetric, positive-definite sample covariance.

    Real minute data can yield an exactly constant endpoint-return column in a
    short complete-case panel.  The sample covariance is then singular, which
    causes CVXPY's generic PSD checker to call an unstable sparse eigensolver
    on some runner builds.  A scale-aware ridge makes the objective strictly
    convex; ``psd_wrap`` below records that constructed fact without asking
    the generic checker to rediscover it numerically.
    """
    values = returns.to_numpy(dtype=float)
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    covariance = (covariance + covariance.T) / 2
    if not np.isfinite(covariance).all():
        raise ValueError("sample covariance must be finite")
    diagonal_scale = max(float(np.abs(np.diag(covariance)).max()), 1e-12)
    ridge = max(diagonal_scale * 1e-8, 1e-12)
    return covariance + np.eye(covariance.shape[0]) * ridge


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
    covariance = _regularized_covariance(clean)
    weights = cp.Variable(n_assets)
    problem = cp.Problem(
        cp.Maximize(mean @ weights - risk_aversion * cp.quad_form(weights, cp.psd_wrap(covariance))),
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


def forecast_mean_variance_weights(
    scenarios: pd.DataFrame,
    expected_returns: pd.Series,
    *,
    risk_aversion: float = 10.0,
    max_weight: float = 0.10,
) -> OptimizationResult:
    """Long-only mean–variance allocation using model forecasts as expected returns.

    ``scenarios`` supplies the historical forward-return covariance matrix;
    ``expected_returns`` is supplied by a separate causal forecast model. This
    avoids substituting historical average return for the model prediction.
    """
    clean = _coerce_returns(scenarios)
    forecast = expected_returns.reindex(clean.columns).astype(float)
    if forecast.isna().any():
        raise ValueError("expected_returns must cover every scenario column")
    n_assets = clean.shape[1]
    if max_weight * n_assets < 1 - 1e-12:
        raise ValueError("max_weight is too small to construct a fully invested portfolio")
    covariance = _regularized_covariance(clean)
    weights = cp.Variable(n_assets)
    problem = cp.Problem(
        cp.Maximize(forecast.to_numpy() @ weights - risk_aversion * cp.quad_form(weights, cp.psd_wrap(covariance))),
        [cp.sum(weights) == 1, weights >= 0, weights <= max_weight],
    )
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weights.value is None:
        raise RuntimeError(f"forecast Mean–variance solve failed with status {problem.status}")
    vector = np.asarray(weights.value).ravel()
    return OptimizationResult(
        weights=pd.Series(vector, index=clean.columns),
        expected_return=float(forecast.to_numpy() @ vector),
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
