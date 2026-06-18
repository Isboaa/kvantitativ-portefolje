"""Internal helpers shared across visualization modules.

These utilities are not part of the public API; they centralize the
figure-saving convention (``outputs/{function_name}_{timestamp}.html``) and a
few small data-shaping helpers used by the plotting functions.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Output directory is relative to the project root (the current working
# directory at runtime), per project conventions.
OUTPUT_DIR = Path("outputs")

# Consistent colors for the three reference portfolios across all figures.
PORTFOLIO_COLORS: dict[str, str] = {
    "min_variance": "#1f77b4",  # blue
    "tangency": "#FFD700",      # gold
    "naive": "#d62728",         # red
}


def save_figure(fig: go.Figure, function_name: str) -> Path:
    """Save a Plotly figure to ``outputs/{function_name}_{timestamp}.html``.

    The ``outputs`` directory is created if it does not already exist. The
    timestamp uses second resolution so repeated calls within the same run do
    not silently overwrite one another.

    Args:
        fig: The Plotly figure to persist.
        function_name: Name of the calling plotting function, used as the
            filename prefix.

    Returns:
        The path the figure was written to.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"{function_name}_{timestamp}.html"
    fig.write_html(str(path))
    logger.info("Saved figure to %s", path)
    return path


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from a price DataFrame.

    Simple (arithmetic) returns are used so that portfolio returns can be
    formed as a weighted sum of asset returns.

    Args:
        prices: DataFrame of asset prices (rows = dates, columns = tickers).

    Returns:
        DataFrame of daily simple returns with the leading NaN row dropped.
    """
    return prices.pct_change().dropna(how="all")


def portfolio_returns(
    returns: pd.DataFrame, weights: np.ndarray
) -> pd.Series:
    """Compute the daily return series of a weighted portfolio.

    Args:
        returns: DataFrame of asset daily returns (rows = dates, columns =
            tickers).
        weights: 1-D array of portfolio weights aligned with the columns of
            ``returns``.

    Returns:
        Series of daily portfolio returns indexed by date.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != returns.shape[1]:
        raise ValueError(
            f"weights length ({weights.shape[0]}) does not match number of "
            f"assets ({returns.shape[1]})"
        )
    return (returns * weights).sum(axis=1)
