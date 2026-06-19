"""End-to-end Markowitz mean-variance portfolio optimization pipeline.

Wires every ``src`` subpackage together:

1. Parse CLI arguments (defaults sourced from ``config.yaml``; CLI wins).
2. Load price data (parquet-cached yfinance download).
3. Estimate annualized expected returns and a covariance matrix.
4. Optimize the minimum-variance, tangency (max-Sharpe) and 1/N portfolios,
   plus the efficient frontier.
5. Compute performance metrics, render Plotly charts, print a summary table,
   log the run to ``HISTORY.md`` / ``CLAUDE.md`` and optionally git-commit.

Run with::

    python main.py
    python main.py --tickers AAPL MSFT --cov-method sample --no-push
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from tabulate import tabulate

from src.data import load_or_fetch
from src.models.covariance import ledoit_wolf, rolling_covariance, sample_covariance
from src.models.expected_returns import mean_historical_return
from src.models.metrics import max_drawdown, sortino_ratio
from src.models.optimizer import (
    efficient_frontier,
    maximum_sharpe,
    minimum_variance,
    naive_portfolio,
)
from src.utils.cli import build_parser
from src.utils.git_sync import auto_commit
from src.utils.history_logger import log_run
from src.visualization import (
    plot_correlation_heatmap,
    plot_cumulative_returns,
    plot_efficient_frontier,
    plot_rolling_sharpe,
    plot_weights,
    plot_weights_area,
)

logger = logging.getLogger("main")

CONFIG_PATH = Path("config.yaml")

# Covariance estimator dispatch keyed by the --cov-method choice.
_COV_METHODS: dict[str, Callable[[pd.DataFrame, int, int], pd.DataFrame]] = {
    "sample": lambda prices, freq, window: sample_covariance(prices, frequency=freq),
    "ledoit_wolf": lambda prices, freq, window: ledoit_wolf(prices, frequency=freq),
    "rolling": lambda prices, freq, window: rolling_covariance(
        prices, window=window, frequency=freq
    ),
}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML configuration file into a dict."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _flat_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested config.yaml into the flat keys build_parser expects."""
    date_range = config.get("date_range", {})
    optimization = config.get("optimization", {})
    return {
        "tickers": config.get("tickers"),
        "start": date_range.get("start"),
        "end": date_range.get("end"),
        "cov_method": optimization.get("covariance_method"),
        "risk_free_rate": optimization.get("risk_free_rate"),
        "output_dir": config.get("output", {}).get("dir"),
    }


def setup_logging(config: dict[str, Any]) -> None:
    """Configure root logging from the config's ``logging`` section."""
    log_config = config.get("logging", {})
    level = getattr(logging, str(log_config.get("level", "INFO")).upper(), logging.INFO)
    fmt = log_config.get("format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logging.basicConfig(level=level, format=fmt)


def _portfolio_metrics(
    result: dict[str, Any], returns_df: pd.DataFrame, risk_free_rate: float
) -> dict[str, Any]:
    """Augment an optimizer result dict with Sortino and max-drawdown."""
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


def _print_summary(metrics_by_name: dict[str, dict[str, Any]]) -> None:
    """Print a formatted summary table of all portfolios to stdout."""
    headers = ["Portfolio", "Return", "Volatility", "Sharpe", "Sortino", "Max DD"]
    rows = [
        [
            name,
            f"{m['return'] * 100:.2f}%",
            f"{m['volatility'] * 100:.2f}%",
            f"{m['sharpe']:.2f}",
            f"{m['sortino']:.2f}",
            f"{m['max_drawdown'] * 100:.2f}%",
        ]
        for name, m in metrics_by_name.items()
    ]
    print("\n" + tabulate(rows, headers=headers, tablefmt="github") + "\n")


def run(config: dict[str, Any], args: Any) -> dict[str, dict[str, Any]]:
    """Execute the full pipeline and return per-portfolio metrics."""
    optimization = config.get("optimization", {})
    frequency = int(optimization.get("frequency", 252))
    n_points = int(optimization.get("n_frontier_points", 100))
    rolling_window = int(optimization.get("rolling_window", 126))

    logger.info("Loading prices for %s (%s to %s)", args.tickers, args.start, args.end)
    prices = load_or_fetch(args.tickers, args.start, args.end)
    returns_df = prices.pct_change().dropna(how="any")

    mu = mean_historical_return(prices, frequency=frequency)
    cov = _COV_METHODS[args.cov_method](prices, frequency, rolling_window)
    logger.info("Estimated mu and %s covariance for %d assets", args.cov_method, len(mu))

    mv = minimum_variance(mu, cov)
    tangency = maximum_sharpe(mu, cov, risk_free_rate=args.risk_free_rate)
    naive = naive_portfolio(mu, cov)
    frontier = efficient_frontier(mu, cov, n_points=n_points)

    metrics = {
        "min_variance": _portfolio_metrics(mv, returns_df, args.risk_free_rate),
        "tangency": _portfolio_metrics(tangency, returns_df, args.risk_free_rate),
        "equal_weight": _portfolio_metrics(naive, returns_df, args.risk_free_rate),
    }

    _print_summary(
        {
            "Minimum Variance": metrics["min_variance"],
            "Tangency (Max Sharpe)": metrics["tangency"],
            "1/N Benchmark": metrics["equal_weight"],
        }
    )

    # Charts (each saves itself to outputs/{name}_{timestamp}.html).
    assets_df = pd.DataFrame(
        {
            "return": mu,
            "volatility": pd.Series(np.sqrt(np.diag(cov.to_numpy())), index=mu.index),
        }
    )
    weights_dict = {
        "Minimum Variance": mv["weights"].to_numpy(dtype=float),
        "Tangency": tangency["weights"].to_numpy(dtype=float),
        "1/N": naive["weights"].to_numpy(dtype=float),
    }

    logger.info("Rendering charts to %s", args.output_dir)
    plot_efficient_frontier(frontier, mv, tangency, naive, assets_df, args.risk_free_rate)
    plot_weights(
        {"Minimum Variance": mv, "Tangency": tangency, "1/N": naive}
    )
    plot_weights_area(frontier)
    plot_correlation_heatmap(prices)
    plot_cumulative_returns(prices, weights_dict)
    plot_rolling_sharpe(prices, weights_dict, risk_free_rate=args.risk_free_rate)

    return metrics


def main(argv: list[str] | None = None) -> None:
    """Entry point: load config, parse args, run pipeline, log and commit."""
    config = load_config()
    setup_logging(config)

    args = build_parser(_flat_defaults(config)).parse_args(argv)

    metrics = run(config, args)

    run_config = {
        "tickers": args.tickers,
        "start": args.start,
        "end": args.end,
        "cov_method": args.cov_method,
        "output_dir": args.output_dir,
    }
    log_run(run_config, metrics)

    if not args.no_push:
        auto_commit(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    else:
        logger.info("--no-push set; skipping git commit of run results.")


if __name__ == "__main__":
    main()
