"""Mean-variance portfolio optimizers.

All optimizers are long-only (weights >= 0) and fully invested
(weights sum to 1). Optimization uses :func:`scipy.optimize.minimize` with the
sequential least squares programming (``SLSQP``) method.

``mu`` is an annualized expected-return :class:`pandas.Series` and ``cov`` an
annualized covariance :class:`pandas.DataFrame`, as produced by
:mod:`src.models.expected_returns` and :mod:`src.models.covariance`.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def _portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """Expected portfolio return for the given weights."""
    return float(weights @ mu)


def _portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio volatility (standard deviation) for the given weights."""
    return float(np.sqrt(max(weights @ cov @ weights, 0.0)))


def _summary(
    weights: np.ndarray,
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float,
) -> Dict[str, object]:
    """Build a result dict for a set of portfolio weights.

    Args:
        weights: Optimized portfolio weights.
        mu: Annualized expected returns per asset.
        cov: Annualized covariance matrix.
        risk_free_rate: Annualized risk-free rate used for the Sharpe ratio.

    Returns:
        Dict with keys ``weights`` (Series), ``return``, ``volatility`` and
        ``sharpe``.
    """
    ret = _portfolio_return(weights, mu.to_numpy(dtype=float))
    vol = _portfolio_volatility(weights, cov.to_numpy(dtype=float))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
    return {
        "weights": pd.Series(weights, index=mu.index),
        "return": ret,
        "volatility": vol,
        "sharpe": sharpe,
    }


def _bounds_and_constraints(n_assets: int):
    """Return the long-only, fully-invested bounds and sum-to-one constraint."""
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    return bounds, constraints


def minimum_variance(mu: pd.Series, cov: pd.DataFrame) -> Dict[str, object]:
    """Long-only minimum-variance portfolio.

    Args:
        mu: Annualized expected returns per asset.
        cov: Annualized covariance matrix.

    Returns:
        Dict with keys ``weights``, ``return``, ``volatility`` and ``sharpe``.
    """
    n = len(mu)
    cov_values = cov.to_numpy(dtype=float)
    bounds, constraints = _bounds_and_constraints(n)
    x0 = np.full(n, 1.0 / n)

    result = minimize(
        lambda w: w @ cov_values @ w,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        logger.warning("minimum_variance did not converge: %s", result.message)
    return _summary(result.x, mu, cov, risk_free_rate=0.04)


def maximum_sharpe(
    mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float = 0.04
) -> Dict[str, object]:
    """Long-only maximum-Sharpe (tangency) portfolio.

    Args:
        mu: Annualized expected returns per asset.
        cov: Annualized covariance matrix.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Dict with keys ``weights``, ``return``, ``volatility`` and ``sharpe``.
    """
    n = len(mu)
    mu_values = mu.to_numpy(dtype=float)
    cov_values = cov.to_numpy(dtype=float)
    bounds, constraints = _bounds_and_constraints(n)
    x0 = np.full(n, 1.0 / n)

    def negative_sharpe(weights: np.ndarray) -> float:
        ret = weights @ mu_values
        vol = np.sqrt(max(weights @ cov_values @ weights, 1e-12))
        return -(ret - risk_free_rate) / vol

    result = minimize(
        negative_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        logger.warning("maximum_sharpe did not converge: %s", result.message)
    return _summary(result.x, mu, cov, risk_free_rate=risk_free_rate)


def naive_portfolio(mu: pd.Series, cov: pd.DataFrame) -> Dict[str, object]:
    """Equal-weight (1/N) benchmark portfolio.

    Args:
        mu: Annualized expected returns per asset.
        cov: Annualized covariance matrix.

    Returns:
        Dict with keys ``weights``, ``return``, ``volatility`` and ``sharpe``.
    """
    n = len(mu)
    weights = np.full(n, 1.0 / n)
    return _summary(weights, mu, cov, risk_free_rate=0.04)


def efficient_frontier(
    mu: pd.Series, cov: pd.DataFrame, n_points: int = 100
) -> pd.DataFrame:
    """Compute the long-only efficient frontier.

    Sweeps target returns from the minimum-variance portfolio return up to the
    maximum attainable return (the single highest-return asset), solving a
    minimum-variance problem at each target.

    Args:
        mu: Annualized expected returns per asset.
        cov: Annualized covariance matrix.
        n_points: Number of points (target returns) along the frontier.

    Returns:
        DataFrame with columns ``return``, ``volatility``, ``sharpe`` and one
        weight column per ticker.
    """
    n = len(mu)
    mu_values = mu.to_numpy(dtype=float)
    cov_values = cov.to_numpy(dtype=float)
    bounds, _ = _bounds_and_constraints(n)

    min_var = minimum_variance(mu, cov)
    return_low = float(min_var["return"])
    return_high = float(mu_values.max())
    if return_high <= return_low:
        return_high = return_low
    targets = np.linspace(return_low, return_high, n_points)

    rows = []
    x0 = np.full(n, 1.0 / n)
    for target in targets:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {
                "type": "eq",
                "fun": lambda w, t=target: w @ mu_values - t,
            },
        )
        result = minimize(
            lambda w: w @ cov_values @ w,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if not result.success:
            logger.debug(
                "efficient_frontier: no solution at target=%.6f (%s)",
                target,
                result.message,
            )
            continue
        weights = result.x
        x0 = weights  # warm-start the next optimization
        ret = _portfolio_return(weights, mu_values)
        vol = _portfolio_volatility(weights, cov_values)
        sharpe = (ret - 0.04) / vol if vol > 0 else 0.0
        row = {"return": ret, "volatility": vol, "sharpe": sharpe}
        row.update(dict(zip(mu.index, weights)))
        rows.append(row)

    columns = ["return", "volatility", "sharpe", *list(mu.index)]
    frontier = pd.DataFrame(rows, columns=columns)
    logger.debug("Computed efficient frontier with %d points", len(frontier))
    return frontier
