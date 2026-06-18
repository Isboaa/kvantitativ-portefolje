"""Portfolio performance metrics.

The core risk/return metrics take portfolio ``weights`` together with an
annualized expected-return vector ``mu`` and annualized covariance matrix
``cov``. ``mu`` and ``cov`` are assumed to already be annualized (as produced
by :mod:`src.models.expected_returns` and :mod:`src.models.covariance`), so the
``frequency`` argument only rescales when a different convention is required.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def portfolio_return(
    weights: np.ndarray, mu: pd.Series, frequency: int = 252
) -> float:
    """Expected annualized portfolio return.

    Args:
        weights: Portfolio weights aligned with ``mu``.
        mu: Annualized expected returns per asset.
        frequency: Annualization frequency. ``mu`` is assumed annualized at
            252 trading days; values other than 252 rescale accordingly.

    Returns:
        Expected annualized portfolio return.
    """
    weights = np.asarray(weights, dtype=float)
    annual = float(weights @ np.asarray(mu, dtype=float))
    return annual * (frequency / 252)


def portfolio_volatility(
    weights: np.ndarray, cov: pd.DataFrame, frequency: int = 252
) -> float:
    """Annualized portfolio volatility (standard deviation).

    Args:
        weights: Portfolio weights aligned with ``cov``.
        cov: Annualized covariance matrix.
        frequency: Annualization frequency. ``cov`` is assumed annualized at
            252 trading days; values other than 252 rescale accordingly.

    Returns:
        Annualized portfolio volatility.
    """
    weights = np.asarray(weights, dtype=float)
    variance = float(weights @ np.asarray(cov, dtype=float) @ weights)
    vol = float(np.sqrt(max(variance, 0.0)))
    return vol * np.sqrt(frequency / 252)


def sharpe_ratio(
    weights: np.ndarray,
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.04,
    frequency: int = 252,
) -> float:
    """Annualized Sharpe ratio of a portfolio.

    Args:
        weights: Portfolio weights aligned with ``mu`` and ``cov``.
        mu: Annualized expected returns per asset.
        cov: Annualized covariance matrix.
        risk_free_rate: Annualized risk-free rate.
        frequency: Annualization frequency.

    Returns:
        Sharpe ratio. Returns 0.0 when volatility is zero.
    """
    ret = portfolio_return(weights, mu, frequency)
    vol = portfolio_volatility(weights, cov, frequency)
    if vol == 0.0:
        return 0.0
    return (ret - risk_free_rate) / vol


def sortino_ratio(
    weights: np.ndarray,
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.04,
    frequency: int = 252,
) -> float:
    """Annualized Sortino ratio using downside deviation.

    Downside deviation is computed from the portfolio's daily returns,
    considering only periods where the return falls below zero.

    Args:
        weights: Portfolio weights aligned with the columns of ``returns_df``.
        returns_df: DataFrame of periodic (daily) asset returns
            (rows = dates, columns = tickers).
        risk_free_rate: Annualized risk-free rate.
        frequency: Number of trading periods per year used for annualization.

    Returns:
        Sortino ratio. Returns 0.0 when downside deviation is zero.
    """
    weights = np.asarray(weights, dtype=float)
    portfolio_daily = returns_df.to_numpy(dtype=float) @ weights

    annual_return = float(portfolio_daily.mean()) * frequency

    downside = portfolio_daily[portfolio_daily < 0.0]
    if downside.size == 0:
        return 0.0
    downside_deviation = float(np.sqrt(np.mean(downside**2))) * np.sqrt(frequency)
    if downside_deviation == 0.0:
        return 0.0
    return (annual_return - risk_free_rate) / downside_deviation


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """Maximum drawdown of a cumulative return (or price/equity) series.

    Args:
        cumulative_returns: Series representing the growth of the portfolio
            over time (e.g. an equity curve or cumulative wealth index).

    Returns:
        The maximum drawdown as a non-positive float (e.g. ``-0.25`` for a
        25% peak-to-trough decline). Returns 0.0 for an empty series.
    """
    if cumulative_returns.empty:
        return 0.0
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1.0
    return float(drawdown.min())
