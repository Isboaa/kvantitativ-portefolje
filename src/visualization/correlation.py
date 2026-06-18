"""Correlation visualization for the asset universe."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.visualization._io import daily_returns, save_figure

logger = logging.getLogger(__name__)


def plot_correlation_heatmap(prices: pd.DataFrame) -> go.Figure:
    """Annotated correlation heatmap of asset daily returns.

    Correlations are computed from daily simple returns (not raw prices, which
    are non-stationary). The heatmap uses the diverging ``RdBu`` colorscale
    fixed to the ``[-1, 1]`` range so the zero-correlation midpoint maps to the
    neutral color, and each cell is annotated with its rounded value.

    Args:
        prices: DataFrame of asset prices (rows = dates, columns = tickers).

    Returns:
        The constructed Plotly :class:`~plotly.graph_objects.Figure`.
    """
    if prices.shape[1] < 2:
        raise ValueError(
            "At least two asset columns are required to compute correlations."
        )

    corr = daily_returns(prices).corr()

    fig = px.imshow(
        corr,
        x=corr.columns,
        y=corr.index,
        color_continuous_scale="RdBu",
        zmin=-1.0,
        zmax=1.0,
        text_auto=".2f",
        aspect="auto",
        title="Asset Return Correlation",
    )
    fig.update_xaxes(side="bottom")
    fig.update_layout(
        template="plotly_white",
        coloraxis_colorbar={"title": "Correlation"},
    )

    save_figure(fig, "plot_correlation_heatmap")
    return fig
