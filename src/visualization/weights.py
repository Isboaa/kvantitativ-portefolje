"""Portfolio weight visualizations.

Provides a grouped bar chart comparing the weights of the reference
portfolios, and a stacked area chart showing how weights evolve across the
efficient frontier.
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualization._io import save_figure

logger = logging.getLogger(__name__)

# Human-readable labels for the canonical reference-portfolio keys.
_DISPLAY_NAMES: dict[str, str] = {
    "min_variance": "Min variance",
    "tangency": "Tangency",
    "naive": "Naive 1/N",
}

_WEIGHT_KEYS = ("weights", "weight", "w")
# Columns in a frontier DataFrame that describe the portfolio (not a weight).
_NON_WEIGHT_COLS = {
    "return", "returns", "expected_return", "mu", "ret",
    "volatility", "risk", "std", "sigma", "stdev", "vol",
    "sharpe", "sharpe_ratio",
}


def _extract_weights(portfolio: Mapping[str, object]) -> np.ndarray:
    """Extract the weight vector from a portfolio mapping.

    Args:
        portfolio: Mapping that contains a weights entry under one of the
            recognized keys (``weights``/``weight``/``w``).

    Returns:
        1-D numpy array of weights.

    Raises:
        KeyError: If no recognized weight key is present.
    """
    for key in _WEIGHT_KEYS:
        if key in portfolio:
            return np.asarray(portfolio[key], dtype=float).ravel()
    raise KeyError(
        f"None of {_WEIGHT_KEYS} found in portfolio keys {list(portfolio)}"
    )


def _asset_labels(
    portfolio: Mapping[str, object], n_assets: int
) -> list[str]:
    """Derive asset (ticker) labels for a portfolio.

    Uses an explicit ``assets``/``tickers`` entry when available, then the
    index of a pandas Series weight vector, otherwise falls back to generic
    ``Asset 1..N`` labels.

    Args:
        portfolio: Portfolio mapping that may carry an asset-label entry.
        n_assets: Number of assets, used for the fallback labels.

    Returns:
        List of asset label strings of length ``n_assets``.
    """
    for key in ("assets", "tickers", "labels"):
        if key in portfolio:
            return [str(a) for a in portfolio[key]]
    for key in _WEIGHT_KEYS:
        weights = portfolio.get(key)
        if isinstance(weights, pd.Series) and len(weights.index) == n_assets:
            return [str(a) for a in weights.index]
    return [f"Asset {i + 1}" for i in range(n_assets)]


def plot_weights(portfolios: dict[str, dict]) -> go.Figure:
    """Grouped bar chart comparing weights across reference portfolios.

    Args:
        portfolios: Mapping of portfolio name (e.g. ``min_variance``,
            ``tangency``, ``naive``) to a portfolio dict containing a weight
            vector and optionally an ``assets``/``tickers`` label list.

    Returns:
        The constructed Plotly :class:`~plotly.graph_objects.Figure`.
    """
    if not portfolios:
        raise ValueError("`portfolios` must contain at least one portfolio")

    fig = go.Figure()
    labels: list[str] | None = None

    for name, portfolio in portfolios.items():
        weights = _extract_weights(portfolio)
        if labels is None:
            labels = _asset_labels(portfolio, weights.size)
        display = _DISPLAY_NAMES.get(name, str(name))
        fig.add_trace(
            go.Bar(
                x=labels,
                y=weights,
                name=display,
                hovertemplate=(
                    f"{display}<br>%{{x}}: %{{y:.2%}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Portfolio Weights Comparison",
        barmode="group",
        xaxis={"title": "Asset"},
        yaxis={"title": "Weight", "tickformat": ".0%"},
        template="plotly_white",
        legend={"title": "Portfolio"},
    )

    save_figure(fig, "plot_weights")
    return fig


def _frontier_weight_columns(frontier_df: pd.DataFrame) -> list[str]:
    """Identify the per-asset weight columns of a frontier DataFrame.

    Columns whose (lowercased) name is a known metric column (return, risk,
    sharpe) are excluded; the remainder are treated as asset weights.

    Args:
        frontier_df: Frontier DataFrame.

    Returns:
        List of column names corresponding to asset weights, preserving order.
    """
    cols: list[str] = []
    for col in frontier_df.columns:
        if str(col).lower() in _NON_WEIGHT_COLS:
            continue
        if pd.api.types.is_numeric_dtype(frontier_df[col]):
            cols.append(col)
    return cols


def plot_weights_area(frontier_df: pd.DataFrame) -> go.Figure:
    """Stacked area chart of how asset weights shift along the frontier.

    The x-axis represents the frontier index (ordered from the
    minimum-variance end towards higher risk/return). Each asset contributes a
    stacked band whose height is its weight at that frontier point.

    Args:
        frontier_df: DataFrame with one row per frontier portfolio. Per-asset
            weight columns are auto-detected by excluding the known metric
            columns (return, volatility/risk, sharpe). The frontier ordering is
            taken from a return/volatility column when present, otherwise from
            row order.

    Returns:
        The constructed Plotly :class:`~plotly.graph_objects.Figure`.
    """
    weight_cols = _frontier_weight_columns(frontier_df)
    if not weight_cols:
        raise ValueError(
            "No asset weight columns detected in `frontier_df`. Expected "
            "numeric per-asset weight columns alongside the metric columns."
        )

    # Order frontier points by volatility (then return) when available so the
    # area chart reads left-to-right from min-variance to high-risk.
    ordered = frontier_df
    for key in ("volatility", "risk", "std", "sigma", "vol"):
        if key in frontier_df.columns:
            ordered = frontier_df.sort_values(key)
            break

    x = np.arange(len(ordered))
    fig = go.Figure()
    for col in weight_cols:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=ordered[col].to_numpy(dtype=float),
                mode="lines",
                name=str(col),
                stackgroup="weights",
                hovertemplate=(
                    f"{col}<br>Frontier point %{{x}}: "
                    "%{y:.2%}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Weight Allocation Across the Efficient Frontier",
        xaxis={"title": "Frontier point (low to high risk)"},
        yaxis={"title": "Weight", "tickformat": ".0%"},
        template="plotly_white",
        legend={"title": "Asset"},
    )

    save_figure(fig, "plot_weights_area")
    return fig
