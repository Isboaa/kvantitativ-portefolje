"""Command-line interface for the portfolio optimizer entrypoint.

Builds the :class:`argparse.ArgumentParser` consumed by ``main.py``. Defaults
are sourced from the project configuration where available, falling back to
sensible built-in values so the parser is usable even before config loads.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Built-in fallbacks used when no config value is supplied.
_DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA"]
_DEFAULT_START = "2020-01-01"
_DEFAULT_END = "2024-12-31"
_DEFAULT_COV_METHOD = "ledoit_wolf"
_DEFAULT_RISK_FREE_RATE = 0.04
_DEFAULT_OUTPUT_DIR = "outputs/"

_COV_CHOICES = ["sample", "ledoit_wolf", "rolling"]


def _config_default(config: dict[str, Any] | None, key: str, fallback: Any) -> Any:
    """Return ``config[key]`` if present and truthy, else ``fallback``."""
    if not config:
        return fallback
    value = config.get(key)
    return value if value not in (None, "", []) else fallback


def build_parser(config: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    """Build the argument parser for the optimizer CLI.

    Args:
        config: Optional configuration mapping used to source argument
            defaults (``tickers``, ``start``, ``end``). When omitted, built-in
            defaults are used.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    config = config or {}

    parser = argparse.ArgumentParser(
        prog="portfolio-optimizer",
        description="Run the quantitative portfolio optimizer.",
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=_config_default(config, "tickers", _DEFAULT_TICKERS),
        help="Ticker symbols to include in the portfolio.",
    )
    parser.add_argument(
        "--start",
        default=_config_default(config, "start", _DEFAULT_START),
        help="Start date for historical data (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        default=_config_default(config, "end", _DEFAULT_END),
        help="End date for historical data (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--cov-method",
        choices=_COV_CHOICES,
        default=_config_default(config, "cov_method", _DEFAULT_COV_METHOD),
        help="Covariance estimation method.",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=_config_default(config, "risk_free_rate", _DEFAULT_RISK_FREE_RATE),
        help="Annual risk-free rate used for Sharpe/tangency calculations.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing committed run results to the git remote.",
    )
    parser.add_argument(
        "--output-dir",
        default=_config_default(config, "output_dir", _DEFAULT_OUTPUT_DIR),
        help="Directory where output HTML and CSV files are written.",
    )

    return parser


def parse_args(argv: list[str] | None = None, config: dict[str, Any] | None = None) -> argparse.Namespace:
    """Parse command-line arguments into a clean namespace.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.
        config: Optional configuration mapping for argument defaults.

    Returns:
        The parsed :class:`argparse.Namespace`.
    """
    parser = build_parser(config)
    return parser.parse_args(argv)
