"""Backtest-style performance visualizations.

Cumulative-return and rolling-Sharpe charts for a set of portfolios, with the
SPY index used as a benchmark. SPY prices are fetched via the data module so
the benchmark is aligned to the same calendar as the supplied price history.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualization._io import (
    daily_returns,
    portfolio_returns,
    save_figure,
)

logger = logging.getLogger(__name__)

_BENCHMARK_TICKER = "SPY"


def _fetch_benchmark_returns(index: pd.Index) -> Optional[pd.Series]:
    """Fetch SPY benchmark daily returns aligned to ``index``.

    The data module is imported lazily so that the visualization package does
    not hard-fail when the (sibling-branch) data module is unavailable, e.g. in
    isolated unit tests. Any failure results in ``None`` and a logged warning.

    Args:
        index: Date index the price history is aligned to; bounds the fetch
            window and the returned series.

    Returns:
        Series of SPY daily simple returns over the requested window, or
        ``None`` if the benchmark could not be retrieved.
    """
    try:
        from src.data import load_or_fetch
    except Exception as exc:  # pragma: no cover - depends on sibling module
        logger.warning("Benchmark unavailable: could not import data module: %s", exc)
        return None

    try:
        start = pd.Timestamp(index.min()).strftime("%Y-%m-%d")
        end = pd.Timestamp(index.max()).strftime("%Y-%m-%d")
        prices = load_or_fetch([_BENCHMARK_TICKER], start=start, end=end)
    except Exception as exc:  # pragma: no cover - network/IO dependent
        logger.warning("Benchmark unavailable: SPY fetch failed: %s", exc)
        return None

    if isinstance(prices, pd.DataFrame):
        if _BENCHMARK_TICKER in prices.columns:
            series = prices[_BENCHMARK_TICKER]
        else:
            series = prices.iloc[:, 0]
    else:
        series = prices
    return series.pct_change().dropna()


def plot_cumulative_returns(
    prices: pd.DataFrame,
    weights_dict: dict[str, np.ndarray],
) -> go.Figure:
    """Plot cumulative growth of each portfolio against the SPY benchmark.

    Each portfolio's daily return is the weighted sum of asset returns; the
    cumulative curve is the running product of ``(1 + r)``. The SPY benchmark
    is overlaid as a dashed line when it can be fetched.

    Args:
        prices: DataFrame of asset prices (rows = dates, columns = tickers).
        weights_dict: Mapping of portfolio name to a weight vector aligned with
            the columns of ``prices``.

    Returns:
        The constructed Plotly :class:`~plotly.graph_objects.Figure`.
    """
    returns = daily_returns(prices)
    fig = go.Figure()

    for name, weights in weights_dict.items():
        port_ret = portfolio_returns(returns, np.asarray(weights, dtype=float))
        cumulative = (1.0 + port_ret).cumprod()
        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative.to_numpy(),
                mode="lines",
                name=str(name),
                hovertemplate=(
                    f"{name}<br>%{{x|%Y-%m-%d}}: "
                    "%{y:.2f}x<extra></extra>"
                ),
            )
        )

    benchmark = _fetch_benchmark_returns(returns.index)
    if benchmark is not None and not benchmark.empty:
        benchmark = benchmark.reindex(returns.index).dropna()
        cumulative = (1.0 + benchmark).cumprod()
        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative.to_numpy(),
                mode="lines",
                name=f"{_BENCHMARK_TICKER} (benchmark)",
                line={"color": "#000000", "dash": "dash", "width": 2},
                hovertemplate=(
                    f"{_BENCHMARK_TICKER}<br>%{{x|%Y-%m-%d}}: "
                    "%{y:.2f}x<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Cumulative Returns",
        xaxis={"title": "Date"},
        yaxis={"title": "Growth of $1"},
        template="plotly_white",
        legend={"title": "Portfolio"},
    )

    save_figure(fig, "plot_cumulative_returns")
    return fig


def plot_rolling_sharpe(
    prices: pd.DataFrame,
    weights_dict: dict[str, np.ndarray],
    window: int = 63,
    risk_free_rate: float = 0.04,
) -> go.Figure:
    """Plot the rolling annualized Sharpe ratio for each portfolio.

    For each rolling window of ``window`` trading days, the Sharpe ratio is the
    annualized mean excess return divided by the annualized return volatility.
    The annual risk-free rate is converted to a per-day rate before computing
    excess returns; annualization uses 252 trading days.

    Args:
        prices: DataFrame of asset prices (rows = dates, columns = tickers).
        weights_dict: Mapping of portfolio name to a weight vector aligned with
            the columns of ``prices``.
        window: Rolling window length in trading days (default 63, ~one
            quarter).
        risk_free_rate: Annualized risk-free rate (decimal, e.g. ``0.04``).

    Returns:
        The constructed Plotly :class:`~plotly.graph_objects.Figure`.
    """
    frequency = 252
    daily_rf = risk_free_rate / frequency
    returns = daily_returns(prices)

    fig = go.Figure()
    for name, weights in weights_dict.items():
        port_ret = portfolio_returns(returns, np.asarray(weights, dtype=float))
        excess = port_ret - daily_rf
        roll_mean = excess.rolling(window).mean()
        roll_std = port_ret.rolling(window).std()
        rolling_sharpe = (roll_mean / roll_std) * np.sqrt(frequency)
        rolling_sharpe = rolling_sharpe.dropna()
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.to_numpy(),
                mode="lines",
                name=str(name),
                hovertemplate=(
                    f"{name}<br>%{{x|%Y-%m-%d}}: "
                    "%{y:.2f}<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=0.0, line={"color": "#888888", "dash": "dot"})
    fig.update_layout(
        title=f"Rolling Sharpe Ratio ({window}-day window)",
        xaxis={"title": "Date"},
        yaxis={"title": "Annualized Sharpe ratio"},
        template="plotly_white",
        legend={"title": "Portfolio"},
    )

    save_figure(fig, "plot_rolling_sharpe")
    return fig
