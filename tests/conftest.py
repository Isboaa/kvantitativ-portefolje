"""Shared pytest fixtures for the quantitative portfolio test suite.

All fixtures are fully synthetic and seeded for reproducibility so the test
suite never touches the network or any real market-data API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Constants used to build the synthetic universe.
# ---------------------------------------------------------------------------
TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
N_TICKERS = len(TICKERS)
SEED = 42
TRADING_DAYS = 252


@pytest.fixture(scope="session")
def tickers() -> list[str]:
    """Return the list of synthetic ticker symbols."""
    return list(TICKERS)


@pytest.fixture(scope="session")
def trading_dates() -> pd.DatetimeIndex:
    """Roughly three years of business days."""
    # ~3 * 252 trading days. Use business-day frequency for realism.
    return pd.bdate_range(start="2021-01-01", periods=3 * TRADING_DAYS, freq="B")


@pytest.fixture
def sample_prices(trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """A DataFrame of synthetic daily adjusted close prices.

    Shape: (3 years of business days, 5 tickers). Generated from a geometric
    random walk so prices are strictly positive and exhibit realistic drift
    and volatility. Seeded for reproducibility.
    """
    rng = np.random.default_rng(SEED)
    n_days = len(trading_dates)

    # Per-ticker annualised drift and volatility -> daily params.
    annual_mu = np.array([0.08, 0.12, 0.05, 0.15, 0.10])
    annual_sigma = np.array([0.18, 0.25, 0.12, 0.30, 0.20])
    daily_mu = annual_mu / TRADING_DAYS
    daily_sigma = annual_sigma / np.sqrt(TRADING_DAYS)

    # Correlated daily log returns via a simple factor structure.
    market = rng.standard_normal((n_days, 1))
    idio = rng.standard_normal((n_days, N_TICKERS))
    betas = np.array([0.9, 1.1, 0.6, 1.3, 1.0])
    shocks = betas * market + idio
    # Normalise the combined shock to unit variance per column.
    shocks = shocks / shocks.std(axis=0, keepdims=True)

    log_returns = daily_mu + daily_sigma * shocks
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))

    return pd.DataFrame(prices, index=trading_dates, columns=list(TICKERS))


@pytest.fixture
def sample_returns(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns derived from ``sample_prices`` (no NaNs)."""
    return sample_prices.pct_change().dropna()


@pytest.fixture
def sample_mu(sample_prices: pd.DataFrame) -> pd.Series:
    """Annualised expected returns (mean historical) as a Series."""
    daily = sample_prices.pct_change().dropna()
    mu = daily.mean() * TRADING_DAYS
    mu.name = "expected_return"
    return mu


@pytest.fixture
def sample_cov(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """Annualised sample covariance matrix (symmetric, PSD)."""
    daily = sample_prices.pct_change().dropna()
    cov = daily.cov() * TRADING_DAYS
    # Symmetrise to guard against tiny floating-point asymmetry.
    cov = (cov + cov.T) / 2.0
    return cov


@pytest.fixture
def equal_weights() -> np.ndarray:
    """Equal-weight allocation vector that sums to 1."""
    return np.full(N_TICKERS, 1.0 / N_TICKERS)


@pytest.fixture
def flat_returns(trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """A returns DataFrame of all-zero returns for 5 tickers."""
    return pd.DataFrame(
        np.zeros((len(trading_dates), N_TICKERS)),
        index=trading_dates,
        columns=list(TICKERS),
    )
