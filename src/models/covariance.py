"""Covariance matrix estimators for portfolio optimization.

All estimators operate on a price DataFrame (rows = dates, columns = tickers)
and return an annualized covariance :class:`pandas.DataFrame` whose index and
columns are the tickers.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from a price DataFrame.

    Args:
        prices: DataFrame of asset prices (rows = dates, columns = tickers).

    Returns:
        DataFrame of daily log returns with rows containing NaNs dropped.
    """
    returns = np.log(prices / prices.shift(1))
    return returns.dropna(how="any")


def sample_covariance(
    prices: pd.DataFrame, frequency: int = 252
) -> pd.DataFrame:
    """Annualized sample covariance matrix of daily log returns.

    Args:
        prices: DataFrame of asset prices (rows = dates, columns = tickers).
        frequency: Number of trading periods per year used for annualization.

    Returns:
        Annualized covariance matrix indexed by ticker on both axes.
    """
    returns = _log_returns(prices)
    cov = returns.cov() * frequency
    logger.debug("Computed sample covariance for %d assets", cov.shape[0])
    return cov


def ledoit_wolf(prices: pd.DataFrame, frequency: int = 252) -> pd.DataFrame:
    """Annualized Ledoit-Wolf shrinkage covariance matrix.

    Uses :class:`sklearn.covariance.LedoitWolf` to shrink the sample
    covariance towards a structured estimator, improving conditioning.

    Args:
        prices: DataFrame of asset prices (rows = dates, columns = tickers).
        frequency: Number of trading periods per year used for annualization.

    Returns:
        Annualized covariance matrix indexed by ticker on both axes.
    """
    returns = _log_returns(prices)
    estimator = LedoitWolf().fit(returns.values)
    cov = pd.DataFrame(
        estimator.covariance_ * frequency,
        index=returns.columns,
        columns=returns.columns,
    )
    logger.debug(
        "Computed Ledoit-Wolf covariance (shrinkage=%.4f) for %d assets",
        estimator.shrinkage_,
        cov.shape[0],
    )
    return cov


def rolling_covariance(
    prices: pd.DataFrame, window: int = 126, frequency: int = 252
) -> pd.DataFrame:
    """Annualized covariance estimated from the most recent ``window`` days.

    Args:
        prices: DataFrame of asset prices (rows = dates, columns = tickers).
        window: Number of most-recent trading days of returns to use.
        frequency: Number of trading periods per year used for annualization.

    Returns:
        Annualized covariance matrix indexed by ticker on both axes.
    """
    returns = _log_returns(prices)
    recent = returns.iloc[-window:]
    cov = recent.cov() * frequency
    logger.debug(
        "Computed rolling covariance (window=%d, used=%d) for %d assets",
        window,
        recent.shape[0],
        cov.shape[0],
    )
    return cov
