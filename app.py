"""Streamlit web app for the quantitative portfolio optimizer.

An interactive front-end over the same pipeline ``main.py`` runs. Users pick a
set of stocks (from a curated list or by typing custom tickers), a date range
and estimation settings; the app fetches prices, optimizes the reference
portfolios and renders every chart with a plain-language explanation so a
non-specialist can understand what they are looking at.

Launch with::

    streamlit run app.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.data import load_or_fetch
from src.data.fetcher import RateLimitError
from src.models.covariance import ledoit_wolf, rolling_covariance, sample_covariance
from src.models.expected_returns import mean_historical_return
from src.models.metrics import max_drawdown, sortino_ratio
from src.models.optimizer import (
    efficient_frontier,
    maximum_sharpe,
    minimum_variance,
    naive_portfolio,
)
from src.visualization import (
    plot_correlation_heatmap,
    plot_cumulative_returns,
    plot_efficient_frontier,
    plot_rolling_sharpe,
    plot_weights,
    plot_weights_area,
)

CONFIG_PATH = Path("config.yaml")

# Covariance estimator dispatch (mirrors main.py). The rolling estimator also
# needs a window; the others ignore it.
_COV_METHODS = {
    "sample": lambda prices, freq, window: sample_covariance(prices, frequency=freq),
    "ledoit_wolf": lambda prices, freq, window: ledoit_wolf(prices, frequency=freq),
    "rolling": lambda prices, freq, window: rolling_covariance(
        prices, window=window, frequency=freq
    ),
}

# A curated universe so users can point-and-click common tickers. They can also
# type anything else in the "custom tickers" box.
POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX",
    "JPM", "GS", "BAC", "V", "MA", "XOM", "CVX", "JNJ", "PFE", "UNH",
    "PG", "KO", "PEP", "WMT", "HD", "DIS", "BRK-B", "SPY", "QQQ", "GLD",
]

# Human-friendly covariance-method labels for the selectbox.
_COV_LABELS = {
    "ledoit_wolf": "Ledoit-Wolf shrinkage (recommended)",
    "sample": "Sample covariance",
    "rolling": "Rolling window",
}


# --------------------------------------------------------------------------- #
# Plain-language explanations shown beneath each chart.
# --------------------------------------------------------------------------- #
INTRO = """
This tool builds **investment portfolios** out of the stocks you choose and
shows how they would have performed. It uses *Markowitz mean-variance
optimization* — the classic idea that, for any level of risk, there is a best
possible mix of assets. Pick your stocks in the sidebar and press
**Run analysis**.

Three reference strategies are compared throughout:

- **Minimum Variance** — the safest mix (lowest expected ups-and-downs).
- **Tangency (Max Sharpe)** — the best *risk-adjusted* mix (most return per unit
  of risk). Usually the headline strategy.
- **1/N Benchmark** — simply an equal amount in every stock; a naive baseline to
  beat.
"""

EXPLANATIONS: dict[str, str] = {
    "frontier": """
**What am I looking at?** Every point is a possible portfolio. The horizontal
axis is **risk** (volatility) and the vertical axis is **expected return**. The
curved line — the *efficient frontier* — traces the best return you can get for
each level of risk. **Up and to the left is better** (more return, less risk).

- ⭐ **Tangency** (gold star): the best return-per-risk mix.
- 🔷 **Min variance** (blue diamond): the lowest-risk mix.
- 🔺 **Balanced** (green triangle): a middle ground between the two.
- ✖️ **1/N** (red): equal weights, for comparison.
- Small circles are the **individual stocks** — notice how mixing them beats
  most single stocks.
- The dashed **Capital Market Line** shows what you get by blending the tangency
  portfolio with a risk-free asset (like cash/T-bills).
""",
    "weights": """
**What am I looking at?** How much of each stock every strategy holds. A taller
bar means a bigger bet on that stock. Compare the strategies side by side — the
minimum-variance mix usually leans on steadier stocks, while the tangency mix
concentrates in the best risk-adjusted performers.
""",
    "weights_area": """
**What am I looking at?** How the ideal mix *changes* as you accept more risk.
Reading left to right, you move from the safest portfolio toward higher
risk/return ones. Each colored band is one stock; watch how the bands grow and
shrink as some stocks earn their place only when you're willing to take on more
risk.
""",
    "correlation": """
**What am I looking at?** How closely each pair of stocks moves together, from
daily returns. **Blue (near +1)** means they tend to rise and fall *together*;
**red (near -1)** means they move in *opposite* directions; near 0 means little
relationship. Diversification works best when you combine stocks that *don't*
all move together — so lots of red/white is good for reducing risk.
""",
    "cumulative": """
**What am I looking at?** The growth of **\\$1** invested in each strategy over
the selected period — this is the "would I have made money?" chart. The dashed
black line is the **S&P 500 (SPY)** as a market benchmark. A line ending higher
made more money; a steeper drop means a rougher ride along the way.
""",
    "rolling_sharpe": """
**What am I looking at?** The **Sharpe ratio** (return earned per unit of risk)
measured over a moving 3-month window, so you can see *when* each strategy was
performing well. Higher is better; above 1 is strong. A line that stays high and
steady is more dependable than one that spikes and crashes.
""",
}

GLOSSARY = """
- **Return** — how much the portfolio grew, per year (annualized).
- **Volatility** — how much the value bounces around; the standard measure of
  risk. Lower is calmer.
- **Sharpe ratio** — return earned above the risk-free rate, per unit of
  volatility. Higher = better reward for the risk taken. Above 1 is good.
- **Sortino ratio** — like Sharpe, but only penalizes *downside* moves (it
  doesn't count upside swings as "risk").
- **Max Drawdown** — the worst peak-to-trough drop over the period. -30% means
  the portfolio once fell 30% from a high point.
- **Tangency portfolio** — the mix with the highest Sharpe ratio.
- **Risk-free rate** — the return on a "safe" asset like short-term Treasury
  bills; the baseline every risky bet is compared against.
"""

# Color for the Balanced portfolio marker on the efficient frontier (green).
BALANCED_COLOR = "#2ca02c"

# Maps the risk-appetite selector to a portfolio shown in the summary table.
RISK_TO_PORTFOLIO = {
    "Conservative": "Minimum Variance",
    "Balanced": "Balanced",
    "Aggressive": "Tangency (Max Sharpe)",
}

PICK_GUIDANCE = """
All of these are "sensible" portfolios — the right one depends on **how much
risk you're comfortable with**. There is no single correct answer.

| If you are… | Choose | Why |
|---|---|---|
| **Cautious** — you hate big drops and want the smoothest ride | **Minimum Variance** | The lowest-risk mix. It won't shoot the lights out, but it's the calmest. |
| **Balanced** — you want solid returns without the wildest swings | **Balanced** | A middle-of-the-road mix: more return than Minimum Variance, less risk than Tangency. |
| **Growth-focused** — you can stomach bigger swings for the best reward-per-risk | **Tangency (Max Sharpe)** | The highest *risk-adjusted* return, but often concentrated in a few winners and more volatile. |
| **Just want a simple baseline** | **1/N Benchmark** | Equal amounts in everything — no optimization, surprisingly hard to beat. |

**An honest caveat.** All of these are built from **past** returns, and history is
a *noisy* guide to the future. The Tangency mix in particular chases whatever did
best in your chosen window, so it can look great on paper yet be fragile going
forward. The **Minimum Variance** and **Balanced** mixes lean on *risk* estimates
(which are far more stable than return estimates), so they tend to hold up better
out-of-sample. Treat this tool as a way to *understand* trade-offs, **not** as
investment advice.
"""


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load config.yaml, tolerating a missing file."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _portfolio_metrics(
    result: dict[str, Any], returns_df: pd.DataFrame, risk_free_rate: float
) -> dict[str, Any]:
    """Augment an optimizer result with Sortino and max-drawdown.

    Mirrors ``main.py:_portfolio_metrics`` so app numbers match the CLI.
    """
    weights = result["weights"].to_numpy(dtype=float)
    portfolio_daily = returns_df.to_numpy(dtype=float) @ weights
    cumulative = pd.Series((1.0 + portfolio_daily).cumprod(), index=returns_df.index)
    return {
        "weights": result["weights"],
        "return": result["return"],
        "volatility": result["volatility"],
        "sharpe": result["sharpe"],
        "sortino": sortino_ratio(weights, returns_df, risk_free_rate=risk_free_rate),
        "max_drawdown": max_drawdown(cumulative),
    }


def _balanced_from_frontier(
    frontier: pd.DataFrame,
    mu_index: pd.Index,
    mv: dict[str, Any],
    tangency: dict[str, Any],
) -> dict[str, Any]:
    """Derive a 'balanced' portfolio from the efficient frontier.

    Picks the frontier point whose volatility is closest to the midpoint between
    the minimum-variance and tangency volatilities — a sensible middle ground
    between the safest and the best risk-adjusted mix. Reuses the already-
    computed frontier, so no extra optimization is needed. Falls back to the
    tangency portfolio when the frontier is degenerate.

    Args:
        frontier: Efficient-frontier DataFrame (return/volatility/sharpe + one
            weight column per ticker).
        mu_index: Ticker order used to align the weight columns.
        mv: The minimum-variance portfolio dict.
        tangency: The tangency portfolio dict.

    Returns:
        A portfolio dict shaped like the optimizer outputs (``weights`` Series,
        ``return``, ``volatility``, ``sharpe``).
    """
    if frontier is None or len(frontier) <= 1:
        return dict(tangency)

    lo = float(mv["volatility"])
    hi = float(tangency["volatility"])
    target = (lo + hi) / 2.0
    idx = (frontier["volatility"] - target).abs().idxmin()
    row = frontier.loc[idx]
    weights = pd.Series([float(row[t]) for t in mu_index], index=mu_index)
    return {
        "weights": weights,
        "return": float(row["return"]),
        "volatility": float(row["volatility"]),
        "sharpe": float(row["sharpe"]),
    }


@st.cache_data(show_spinner=False)
def run_pipeline(
    tickers: tuple[str, ...],
    start: str,
    end: str,
    cov_method: str,
    risk_free_rate: float,
    frequency: int,
    n_points: int,
    rolling_window: int,
) -> dict[str, Any]:
    """Run the full optimization pipeline and return everything the UI needs.

    Cached on its arguments so re-picking an identical configuration does not
    recompute. Tickers are passed as a tuple so the arguments are hashable.
    """
    prices = load_or_fetch(list(tickers), start, end)
    returns_df = prices.pct_change().dropna(how="any")

    mu = mean_historical_return(prices, frequency=frequency)
    cov = _COV_METHODS[cov_method](prices, frequency, rolling_window)

    mv = minimum_variance(mu, cov)
    tangency = maximum_sharpe(mu, cov, risk_free_rate=risk_free_rate)
    naive = naive_portfolio(mu, cov)
    frontier = efficient_frontier(mu, cov, n_points=n_points)
    balanced = _balanced_from_frontier(frontier, mu.index, mv, tangency)

    # Order shown throughout the app: safest -> balanced -> aggressive -> baseline.
    metrics = {
        "Minimum Variance": _portfolio_metrics(mv, returns_df, risk_free_rate),
        "Balanced": _portfolio_metrics(balanced, returns_df, risk_free_rate),
        "Tangency (Max Sharpe)": _portfolio_metrics(tangency, returns_df, risk_free_rate),
        "1/N Benchmark": _portfolio_metrics(naive, returns_df, risk_free_rate),
    }

    assets_df = pd.DataFrame(
        {
            "return": mu,
            "volatility": pd.Series(np.sqrt(np.diag(cov.to_numpy())), index=mu.index),
        }
    )
    weights_dict = {
        "Minimum Variance": mv["weights"].to_numpy(dtype=float),
        "Balanced": balanced["weights"].to_numpy(dtype=float),
        "Tangency": tangency["weights"].to_numpy(dtype=float),
        "1/N": naive["weights"].to_numpy(dtype=float),
    }

    return {
        "prices": prices,
        "mu": mu,
        "cov": cov,
        "mv": mv,
        "balanced": balanced,
        "tangency": tangency,
        "naive": naive,
        "frontier": frontier,
        "assets_df": assets_df,
        "weights_dict": weights_dict,
        "metrics": metrics,
        "risk_free_rate": risk_free_rate,
        "prices_used": list(prices.columns),
    }


def _metrics_table(metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build the formatted summary table shown at the top of the results."""
    rows = {
        name: {
            "Return": f"{m['return'] * 100:.2f}%",
            "Volatility": f"{m['volatility'] * 100:.2f}%",
            "Sharpe": f"{m['sharpe']:.2f}",
            "Sortino": f"{m['sortino']:.2f}",
            "Max Drawdown": f"{m['max_drawdown'] * 100:.2f}%",
        }
        for name, m in metrics.items()
    }
    return pd.DataFrame(rows).T


def _chart_section(title: str, key: str, fig) -> None:
    """Render a chart with its explanation in a consistent layout."""
    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("ℹ️ What does this chart show?", expanded=False):
        st.markdown(EXPLANATIONS[key])


def _add_balanced_marker(fig: go.Figure, balanced: dict[str, Any]) -> None:
    """Overlay the Balanced portfolio as a marker on the efficient-frontier fig.

    Done here rather than inside the plotting module so the visualization layer
    stays unchanged; the module already returns the Figure for us to annotate.
    """
    fig.add_trace(
        go.Scatter(
            x=[float(balanced["volatility"])],
            y=[float(balanced["return"])],
            mode="markers",
            name="Balanced",
            marker={
                "size": 15,
                "color": BALANCED_COLOR,
                "symbol": "triangle-up",
                "line": {"width": 1, "color": "#333333"},
            },
            hovertemplate=(
                "Balanced<br>Volatility: %{x:.2%}"
                "<br>Return: %{y:.2%}<extra></extra>"
            ),
        )
    )


def render_results(result: dict[str, Any]) -> None:
    """Render the metrics table and all six explained charts."""
    used = result["prices_used"]
    st.success(f"Analyzed **{len(used)}** stocks: {', '.join(used)}")

    st.header("📊 Performance summary")
    st.markdown(
        "How the strategies compare over the selected period. "
        "See the glossary at the bottom for what each column means."
    )

    # Pick-helper: map the user's risk appetite to a recommended portfolio and
    # highlight it. This reads already-computed results, so it never recomputes.
    st.markdown("**🎯 Which one fits you?** Set your risk appetite:")
    appetite = st.radio(
        "Risk appetite",
        options=list(RISK_TO_PORTFOLIO.keys()),
        index=1,
        horizontal=True,
        label_visibility="collapsed",
    )
    recommended = RISK_TO_PORTFOLIO[appetite]
    rec = result["metrics"][recommended]
    st.success(
        f"**Recommended for a {appetite.lower()} investor: {recommended}** — "
        f"expected return **{rec['return'] * 100:.1f}%**, "
        f"volatility **{rec['volatility'] * 100:.1f}%**, "
        f"Sharpe **{rec['sharpe']:.2f}**. The matching row is highlighted below."
    )

    table = _metrics_table(result["metrics"])

    def _highlight(row: pd.Series) -> list[str]:
        color = "background-color: #fff3cd" if row.name == recommended else ""
        return [color] * len(row)

    st.dataframe(table.style.apply(_highlight, axis=1), use_container_width=True)

    with st.expander("🤔 Which portfolio should I choose?", expanded=False):
        st.markdown(PICK_GUIDANCE)

    st.divider()
    frontier_fig = plot_efficient_frontier(
        result["frontier"],
        result["mv"],
        result["tangency"],
        result["naive"],
        result["assets_df"],
        result["risk_free_rate"],
        save=False,
    )
    _add_balanced_marker(frontier_fig, result["balanced"])
    _chart_section("Efficient frontier", "frontier", frontier_fig)

    st.divider()
    _chart_section(
        "Portfolio weights",
        "weights",
        plot_weights(
            {
                "Min variance": result["mv"],
                "Balanced": result["balanced"],
                "Tangency": result["tangency"],
                "Naive 1/N": result["naive"],
            },
            save=False,
        ),
    )

    st.divider()
    _chart_section(
        "Weights across the frontier",
        "weights_area",
        plot_weights_area(result["frontier"], save=False),
    )

    st.divider()
    _chart_section(
        "Correlation heatmap",
        "correlation",
        plot_correlation_heatmap(result["prices"], save=False),
    )

    st.divider()
    _chart_section(
        "Cumulative returns",
        "cumulative",
        plot_cumulative_returns(result["prices"], result["weights_dict"], save=False),
    )

    st.divider()
    _chart_section(
        "Rolling Sharpe ratio",
        "rolling_sharpe",
        plot_rolling_sharpe(
            result["prices"],
            result["weights_dict"],
            risk_free_rate=result["risk_free_rate"],
            save=False,
        ),
    )

    st.divider()
    with st.expander("📖 Glossary — what do these terms mean?"):
        st.markdown(GLOSSARY)


def _add_custom_tickers() -> None:
    """Move symbols typed in the custom box into the multiselect as chips.

    Runs as the text input's ``on_change`` callback, which Streamlit fires
    before the widgets re-render, so the new chip is visible on the same rerun.
    The box is cleared afterwards so it is ready for the next entry.
    """
    raw = st.session_state.get("custom_input", "")
    symbols = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]

    for symbol in symbols:
        if symbol not in POPULAR_TICKERS and symbol not in st.session_state.custom_options:
            st.session_state.custom_options.append(symbol)
        if symbol not in st.session_state.ticker_selection:
            st.session_state.ticker_selection.append(symbol)

    st.session_state.custom_input = ""


def sidebar_controls(config: dict[str, Any]) -> dict[str, Any] | None:
    """Render the sidebar and return the run configuration, or None.

    Returns a config dict only when the user has requested a run with a valid
    ticker selection; otherwise returns None.
    """
    st.sidebar.title("⚙️ Configuration")

    default_tickers = config.get("tickers") or ["AAPL", "MSFT", "GOOGL", "SPY"]

    # The selection is mirrored in session state so the custom-ticker box can
    # push into it (see _add_custom_tickers) and have the symbol show up as a
    # chip in the multiselect, exactly like a clicked one.
    #
    # It is kept under its own key rather than the widget's: adding a custom
    # ticker grows `options`, which changes the multiselect's identity and makes
    # Streamlit drop the widget's stored state. Feeding this mirror back in as
    # `default` is what survives that reset — otherwise every chip would vanish
    # the moment a custom ticker was added.
    if "custom_options" not in st.session_state:
        st.session_state.custom_options = []
    if "ticker_selection" not in st.session_state:
        st.session_state.ticker_selection = list(default_tickers)

    # Every selected ticker must appear in options or Streamlit raises; because
    # custom_options only ever grows, that invariant holds across reruns.
    options = sorted(
        set(POPULAR_TICKERS) | set(default_tickers) | set(st.session_state.custom_options)
    )

    st.sidebar.subheader("1. Choose stocks")
    selected = st.sidebar.multiselect(
        "Pick from common tickers",
        options=options,
        default=st.session_state.ticker_selection,
        key="ticker_widget",
        help="Select as many as you like. Add anything else in the box below.",
    )
    # Mirror the widget's own edits (e.g. removing a chip) back into state.
    st.session_state.ticker_selection = list(selected)
    st.sidebar.text_input(
        "Add custom ticker(s)",
        key="custom_input",
        placeholder="e.g. NVDA, COST, TSM",
        on_change=_add_custom_tickers,
        help="Comma- or space-separated symbols. Case-insensitive.",
    )
    st.sidebar.caption("Press Enter to add — it appears as a chip above.")

    tickers = list(selected)

    st.sidebar.subheader("2. Date range")
    date_range = config.get("date_range", {})
    default_start = pd.Timestamp(date_range.get("start", "2019-01-01")).date()
    default_end = pd.Timestamp(date_range.get("end", "2024-01-01")).date()
    start = st.sidebar.date_input(
        "Start date", value=default_start, min_value=date(2000, 1, 1)
    )
    end = st.sidebar.date_input("End date", value=default_end, min_value=date(2000, 1, 1))

    st.sidebar.subheader("3. Settings")
    optimization = config.get("optimization", {})
    default_cov = optimization.get("covariance_method", "ledoit_wolf")
    method_keys = list(_COV_LABELS.keys())
    cov_method = st.sidebar.selectbox(
        "Covariance method",
        options=method_keys,
        index=method_keys.index(default_cov) if default_cov in method_keys else 0,
        format_func=lambda k: _COV_LABELS[k],
        help="How the risk relationships between stocks are estimated.",
    )
    rf_default = float(optimization.get("risk_free_rate", 0.04)) * 100
    risk_free_rate = st.sidebar.slider(
        "Risk-free rate (%)",
        min_value=0.0,
        max_value=8.0,
        value=rf_default,
        step=0.25,
        help="The return on a 'safe' asset, used for Sharpe/tangency math.",
    ) / 100.0

    run = st.sidebar.button("🚀 Run analysis", type="primary", use_container_width=True)

    st.sidebar.caption(f"**{len(tickers)}** stock(s) selected.")

    if not run:
        return None

    if len(tickers) < 2:
        st.sidebar.error("Please select at least **2** stocks to build a portfolio.")
        return None
    if start >= end:
        st.sidebar.error("Start date must be **before** the end date.")
        return None

    return {
        "tickers": tuple(tickers),
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "cov_method": cov_method,
        "risk_free_rate": risk_free_rate,
        "frequency": int(optimization.get("frequency", 252)),
        "n_points": int(optimization.get("n_frontier_points", 100)),
        "rolling_window": int(optimization.get("rolling_window", 126)),
    }


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")
    st.title("📈 Quantitative Portfolio Optimizer")
    st.markdown(INTRO)

    config = load_config()
    run_config = sidebar_controls(config)

    if run_config is not None:
        try:
            with st.spinner("Fetching prices and optimizing portfolios…"):
                result = run_pipeline(**run_config)
            st.session_state["result"] = result
        except RateLimitError as exc:
            # Must come before ValueError: RateLimitError subclasses it.
            st.warning(
                f"⏳ {exc}\n\n"
                "Yahoo Finance limits how often new data can be downloaded. "
                "Wait about a minute and press **Run analysis** again — once a "
                "stock set has been fetched it is cached and loads instantly. "
                "The default preset also works fully offline."
            )
            return
        except ValueError as exc:
            st.error(
                f"Could not run the analysis: {exc}\n\n"
                "This usually means one or more ticker symbols were not "
                "recognized. Check the symbols and date range, then try again."
            )
            return
        except Exception as exc:  # pragma: no cover - surface unexpected errors
            st.error(f"Unexpected error: {exc}")
            return

    if "result" in st.session_state:
        render_results(st.session_state["result"])
    else:
        st.info("👈 Choose your stocks and settings in the sidebar, then press **Run analysis**.")


if __name__ == "__main__":
    main()
