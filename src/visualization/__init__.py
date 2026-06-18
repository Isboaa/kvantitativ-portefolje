"""Plotly-based visualization for the quantitative portfolio project.

Every plotting function builds and returns a :class:`plotly.graph_objects.Figure`
and, as a side effect, writes the figure to
``outputs/{function_name}_{timestamp}.html``.

Public API:
    plot_efficient_frontier: Frontier curve, reference portfolios, assets, CML.
    plot_weights:            Grouped bar chart of reference-portfolio weights.
    plot_weights_area:       Stacked area chart of weights along the frontier.
    plot_correlation_heatmap: Annotated return-correlation heatmap.
    plot_cumulative_returns: Cumulative portfolio returns vs. SPY benchmark.
    plot_rolling_sharpe:     Rolling annualized Sharpe ratio per portfolio.
"""

from src.visualization.correlation import plot_correlation_heatmap
from src.visualization.efficient_frontier import plot_efficient_frontier
from src.visualization.performance import (
    plot_cumulative_returns,
    plot_rolling_sharpe,
)
from src.visualization.weights import plot_weights, plot_weights_area

__all__ = [
    "plot_efficient_frontier",
    "plot_weights",
    "plot_weights_area",
    "plot_correlation_heatmap",
    "plot_cumulative_returns",
    "plot_rolling_sharpe",
]
