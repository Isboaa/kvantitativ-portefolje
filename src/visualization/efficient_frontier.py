"""Efficient frontier visualization.

Renders the mean-variance efficient frontier together with the reference
portfolios (minimum-variance, tangency, naive 1/N), the individual assets and
the Capital Market Line, using Plotly exclusively.
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualization._io import PORTFOLIO_COLORS, save_figure

logger = logging.getLogger(__name__)

# Candidate column names for the frontier DataFrame, tried in order so the
# plot is robust to small naming differences from the optimization module.
_RETURN_KEYS = ("return", "returns", "expected_return", "mu", "ret")
_RISK_KEYS = ("volatility", "risk", "std", "sigma", "stdev", "vol")
_SHARPE_KEYS = ("sharpe", "sharpe_ratio")


def _pick(frame_or_map: Mapping[str, object], keys: tuple[str, ...]) -> str:
    """Return the first key from ``keys`` present in ``frame_or_map``.

    Args:
        frame_or_map: A mapping or DataFrame whose columns/keys are searched.
        keys: Candidate keys in priority order.

    Returns:
        The matching key.

    Raises:
        KeyError: If none of the candidate keys are present.
    """
    available = (
        list(frame_or_map.columns)
        if isinstance(frame_or_map, pd.DataFrame)
        else list(frame_or_map.keys())
    )
    for key in keys:
        if key in available:
            return key
    raise KeyError(
        f"None of {keys} found among available keys {available}"
    )


def _point_coords(
    point: Mapping[str, object],
) -> tuple[float, float, float | None]:
    """Extract (risk, return, sharpe) from a portfolio mapping.

    Args:
        point: Mapping describing a portfolio. Must contain a return value and
            a risk/volatility value; a Sharpe value is optional.

    Returns:
        Tuple of (volatility, expected_return, sharpe-or-None).
    """
    ret = float(point[_pick(point, _RETURN_KEYS)])
    risk = float(point[_pick(point, _RISK_KEYS)])
    try:
        sharpe: float | None = float(point[_pick(point, _SHARPE_KEYS)])
    except KeyError:
        sharpe = None
    return risk, ret, sharpe


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    min_var: Mapping[str, object],
    tangency: Mapping[str, object],
    naive: Mapping[str, object],
    assets_df: pd.DataFrame,
    risk_free_rate: float,
    save: bool = True,
) -> go.Figure:
    """Plot the efficient frontier with reference portfolios and the CML.

    The frontier curve is colored by Sharpe ratio (Viridis colorscale). The
    minimum-variance (blue), tangency (gold star) and naive 1/N (red)
    portfolios are marked as named points. Individual assets are scattered with
    their ticker labels, and the Capital Market Line is drawn from
    ``(0, risk_free_rate)`` through the tangency portfolio.

    Args:
        frontier_df: DataFrame with one row per frontier portfolio. Must
            contain a return column (one of ``return``/``expected_return``/...)
            and a risk column (one of ``volatility``/``risk``/...). A Sharpe
            column is used for coloring when present, otherwise it is derived.
        min_var: Mapping describing the minimum-variance portfolio (return and
            risk values).
        tangency: Mapping describing the tangency (max-Sharpe) portfolio.
        naive: Mapping describing the naive 1/N portfolio.
        assets_df: DataFrame indexed by ticker with a return column and a risk
            column for each individual asset.
        risk_free_rate: Annualized risk-free rate (decimal, e.g. ``0.04``).
        save: When ``True`` (default) the figure is also written to
            ``outputs/`` as HTML; pass ``False`` to only return it.

    Returns:
        The constructed Plotly :class:`~plotly.graph_objects.Figure`.
    """
    ret_col = _pick(frontier_df, _RETURN_KEYS)
    risk_col = _pick(frontier_df, _RISK_KEYS)
    rets = frontier_df[ret_col].to_numpy(dtype=float)
    risks = frontier_df[risk_col].to_numpy(dtype=float)

    try:
        sharpe_col = _pick(frontier_df, _SHARPE_KEYS)
        sharpes = frontier_df[sharpe_col].to_numpy(dtype=float)
    except KeyError:
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpes = np.where(risks > 0, (rets - risk_free_rate) / risks, 0.0)

    fig = go.Figure()

    # Frontier curve, colored by Sharpe ratio.
    fig.add_trace(
        go.Scatter(
            x=risks,
            y=rets,
            mode="markers+lines",
            name="Efficient frontier",
            line={"color": "rgba(120,120,120,0.4)"},
            marker={
                "size": 6,
                "color": sharpes,
                "colorscale": "Viridis",
                "colorbar": {"title": "Sharpe"},
                "showscale": True,
            },
            hovertemplate=(
                "Volatility: %{x:.2%}<br>Return: %{y:.2%}"
                "<br>Sharpe: %{marker.color:.2f}<extra></extra>"
            ),
        )
    )

    # Individual assets scatter with ticker labels.
    asset_ret_col = _pick(assets_df, _RETURN_KEYS)
    asset_risk_col = _pick(assets_df, _RISK_KEYS)
    fig.add_trace(
        go.Scatter(
            x=assets_df[asset_risk_col].to_numpy(dtype=float),
            y=assets_df[asset_ret_col].to_numpy(dtype=float),
            mode="markers+text",
            name="Assets",
            text=[str(t) for t in assets_df.index],
            textposition="top center",
            marker={"size": 9, "color": "#555555", "symbol": "circle-open"},
            hovertemplate=(
                "%{text}<br>Volatility: %{x:.2%}"
                "<br>Return: %{y:.2%}<extra></extra>"
            ),
        )
    )

    # Reference portfolios.
    mv_risk, mv_ret, _ = _point_coords(min_var)
    tan_risk, tan_ret, _ = _point_coords(tangency)
    nv_risk, nv_ret, _ = _point_coords(naive)

    _add_marker(fig, mv_risk, mv_ret, "Min variance",
                PORTFOLIO_COLORS["min_variance"], "diamond", 14)
    _add_marker(fig, tan_risk, tan_ret, "Tangency",
                PORTFOLIO_COLORS["tangency"], "star", 18)
    _add_marker(fig, nv_risk, nv_ret, "Naive 1/N",
                PORTFOLIO_COLORS["naive"], "x", 14)

    # Capital Market Line: from (0, rf) through the tangency portfolio,
    # extended slightly beyond it for visual context.
    if tan_risk > 0:
        x_max = max(float(np.max(risks)) if risks.size else tan_risk, tan_risk)
        x_max *= 1.05
        slope = (tan_ret - risk_free_rate) / tan_risk
        cml_x = np.array([0.0, x_max])
        cml_y = risk_free_rate + slope * cml_x
        fig.add_trace(
            go.Scatter(
                x=cml_x,
                y=cml_y,
                mode="lines",
                name="Capital Market Line",
                line={"color": PORTFOLIO_COLORS["tangency"],
                      "dash": "dash", "width": 2},
                hovertemplate=(
                    "Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra>CML</extra>"
                ),
            )
        )

    fig.update_layout(
        title="Efficient Frontier",
        xaxis={"title": "Annualized volatility", "tickformat": ".0%"},
        yaxis={"title": "Annualized expected return", "tickformat": ".0%"},
        template="plotly_white",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )

    if save:
        save_figure(fig, "plot_efficient_frontier")
    return fig


def _add_marker(
    fig: go.Figure,
    x: float,
    y: float,
    name: str,
    color: str,
    symbol: str,
    size: int,
) -> None:
    """Add a single named marker trace for a reference portfolio.

    Args:
        fig: Figure to add the trace to (modified in place).
        x: Volatility coordinate.
        y: Return coordinate.
        name: Legend/trace name.
        color: Marker color.
        symbol: Plotly marker symbol.
        size: Marker size in pixels.
    """
    fig.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            name=name,
            marker={
                "size": size,
                "color": color,
                "symbol": symbol,
                "line": {"width": 1, "color": "#333333"},
            },
            hovertemplate=(
                f"{name}<br>Volatility: %{{x:.2%}}"
                "<br>Return: %{y:.2%}<extra></extra>"
            ),
        )
    )
