"""Quantitative portfolio models.

This package provides estimators and optimizers for mean-variance portfolio
construction:

- :mod:`expected_returns`: annualized expected-return estimators.
- :mod:`covariance`: annualized covariance estimators.
- :mod:`optimizer`: long-only, fully-invested portfolio optimizers.
- :mod:`metrics`: portfolio performance metrics.
"""

from __future__ import annotations

from src.models.covariance import (
    ledoit_wolf,
    rolling_covariance,
    sample_covariance,
)
from src.models.expected_returns import (
    ema_historical_return,
    mean_historical_return,
)
from src.models.metrics import (
    max_drawdown,
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio,
    sortino_ratio,
)
from src.models.optimizer import (
    efficient_frontier,
    maximum_sharpe,
    minimum_variance,
    naive_portfolio,
)

__all__ = [
    "mean_historical_return",
    "ema_historical_return",
    "sample_covariance",
    "ledoit_wolf",
    "rolling_covariance",
    "minimum_variance",
    "maximum_sharpe",
    "efficient_frontier",
    "naive_portfolio",
    "portfolio_return",
    "portfolio_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
]
