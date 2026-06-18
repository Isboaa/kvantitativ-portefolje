"""Expected return estimators for portfolio optimization.

All estimators operate on a price DataFrame (rows = dates, columns = tickers)
and return an annualized expected-return :class:`pandas.Series` indexed by
ticker.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from a price DataFrame.

    Args:
        prices: DataFrame of asset prices (rows = dates, columns = tickers).

    Returns:
        DataFrame of daily log returns with the first (NaN) row dropped.
    """
    returns = np.log(prices / prices.shift(1))
    return returns.dropna(how="all")


def mean_historical_return(
    prices: pd.DataFrame, frequency: int = 252
) -> pd.Series:
    """Annualized mean of daily log returns.

    Args:
        prices: DataFrame of asset prices (rows = dates, columns = tickers).
        frequency: Number of trading periods per year used for annualization.

    Returns:
        Series of annualized expected returns indexed by ticker.
    """
    returns = _log_returns(prices)
    annualized = returns.mean() * frequency
    logger.debug("Computed mean historical return for %d assets", annualized.size)
    return annualized


def ema_historical_return(
    prices: pd.DataFrame, span: int = 60, frequency: int = 252
) -> pd.Series:
    """Annualized exponentially weighted mean of daily log returns.

    Gives more weight to recent observations via an exponential moving
    average controlled by ``span``.

    Args:
        prices: DataFrame of asset prices (rows = dates, columns = tickers).
        span: Decay span for the exponential moving average. Larger values
            place relatively more weight on older observations.
        frequency: Number of trading periods per year used for annualization.

    Returns:
        Series of annualized expected returns indexed by ticker.
    """
    returns = _log_returns(prices)
    annualized = returns.ewm(span=span).mean().iloc[-1] * frequency
    logger.debug(
        "Computed EMA historical return (span=%d) for %d assets",
        span,
        annualized.size,
    )
    return annualized
