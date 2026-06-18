"""Run history logging and CLAUDE.md status synchronization.

This module appends human-readable run summaries to ``HISTORY.md`` at the
project root and keeps the ``## Current status`` and ``## Changelog`` sections
of ``CLAUDE.md`` up to date after every run.

All file operations degrade gracefully: if a target file is missing or cannot
be written, a warning is logged and execution continues — these helpers never
raise.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# File paths are kept relative to the project root per project conventions.
HISTORY_PATH = Path("HISTORY.md")
CLAUDE_PATH = Path("CLAUDE.md")

# Portfolios rendered (in order) in a HISTORY.md entry, mapping the display
# heading to the key looked up inside ``results``.
_PORTFOLIO_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Minimum Variance Portfolio", ("min_variance", "minimum_variance", "mvp")),
    ("Tangency Portfolio", ("tangency", "max_sharpe", "tangency_portfolio")),
    ("1/N Benchmark", ("equal_weight", "one_over_n", "benchmark", "1/n")),
)


def _fmt_pct(value: Any) -> str:
    """Format a fraction (e.g. 0.1234) as a percentage string ``12.34%``.

    Returns ``"N/A"`` when the value is missing or not numeric.
    """
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_ratio(value: Any) -> str:
    """Format a unitless ratio (e.g. Sharpe) as ``1.23``.

    Returns ``"N/A"`` when the value is missing or not numeric.
    """
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _get(metrics: Mapping[str, Any], *keys: str) -> Any:
    """Return the first present value among ``keys`` in ``metrics``."""
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def _resolve_portfolio(results: Mapping[str, Any], keys: tuple[str, ...]) -> Mapping[str, Any]:
    """Return the metrics mapping for the first matching key, else empty dict."""
    for key in keys:
        value = results.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _format_metrics_line(metrics: Mapping[str, Any]) -> str:
    """Build the ``- Return: ... | Volatility: ...`` line for a portfolio."""
    ret = _fmt_pct(_get(metrics, "return", "expected_return", "annual_return", "ret"))
    vol = _fmt_pct(_get(metrics, "volatility", "vol", "annual_volatility", "std"))
    sharpe = _fmt_ratio(_get(metrics, "sharpe", "sharpe_ratio"))
    sortino = _fmt_ratio(_get(metrics, "sortino", "sortino_ratio"))
    max_dd = _fmt_pct(_get(metrics, "max_drawdown", "max_dd", "maximum_drawdown"))
    return (
        f"- Return: {ret} | Volatility: {vol} | Sharpe: {sharpe} | "
        f"Sortino: {sortino} | Max DD: {max_dd}"
    )


def _format_tickers(config: Mapping[str, Any]) -> str:
    """Render the tickers list from config as a space-separated string."""
    tickers = config.get("tickers")
    if isinstance(tickers, (list, tuple)):
        return " ".join(str(t) for t in tickers)
    if tickers:
        return str(tickers)
    return "N/A"


def _format_date_range(config: Mapping[str, Any]) -> str:
    """Render the ``start`` to ``end`` date range from config."""
    start = config.get("start") or config.get("start_date") or "N/A"
    end = config.get("end") or config.get("end_date") or "N/A"
    return f"{start} to {end}"


def _cov_method(config: Mapping[str, Any]) -> str:
    """Return the covariance method name from config."""
    return str(config.get("cov_method") or config.get("covariance_method") or "N/A")


def _output_dir(config: Mapping[str, Any], results: Mapping[str, Any]) -> str:
    """Resolve where outputs were saved for the ``Outputs saved:`` line."""
    return str(
        config.get("output_dir")
        or results.get("output_dir")
        or "outputs/"
    )


def log_run(config: dict, results: dict) -> None:
    """Append a timestamped markdown summary of a run to ``HISTORY.md``.

    Args:
        config: Run configuration (tickers, date range, covariance method,
            output directory, ...).
        results: Per-portfolio metrics keyed by portfolio name. Each value is
            a mapping with keys such as ``return``, ``volatility``, ``sharpe``,
            ``sortino`` and ``max_drawdown``.

    The function never raises; on any failure it logs a warning and returns.
    """
    config = config or {}
    results = results or {}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = [
        "---",
        f"## Run — {timestamp}",
        f"**Tickers:** {_format_tickers(config)}",
        f"**Date range:** {_format_date_range(config)}",
        f"**Covariance method:** {_cov_method(config)}",
        "",
    ]

    for heading, keys in _PORTFOLIO_SECTIONS:
        metrics = _resolve_portfolio(results, keys)
        lines.append(f"### {heading}")
        lines.append(_format_metrics_line(metrics))
        lines.append("")

    lines.append(f"**Outputs saved:** {_output_dir(config, results)}")
    lines.append("---")
    lines.append("")

    entry = "\n".join(lines)

    try:
        with HISTORY_PATH.open("a", encoding="utf-8") as handle:
            handle.write(entry)
        logger.info("Appended run entry to %s", HISTORY_PATH)
    except OSError as exc:
        logger.warning("Failed to write run entry to %s: %s", HISTORY_PATH, exc)

    # Keep CLAUDE.md in sync as part of the same post-run hook.
    update_claude_md(config, results, timestamp)


def _best_sharpe(results: Mapping[str, Any]) -> str:
    """Return the tangency portfolio's Sharpe ratio formatted as ``1.23``."""
    tangency = _resolve_portfolio(
        results, ("tangency", "max_sharpe", "tangency_portfolio")
    )
    return _fmt_ratio(_get(tangency, "sharpe", "sharpe_ratio"))


def _build_status_block(config: Mapping[str, Any], results: Mapping[str, Any], timestamp: str) -> list[str]:
    """Build the lines for the ``## Current status`` section body."""
    return [
        "## Current status",
        f"Last run: {timestamp}",
        f"Tickers: {_format_tickers(config)}",
        f"Covariance method: {_cov_method(config)}",
        f"Best Sharpe (tangency): {_best_sharpe(results)}",
        "Outputs: outputs/",
    ]


def _replace_status_section(text: str, status_lines: list[str]) -> str:
    """Replace the ``## Current status`` block, or append it if absent.

    The block extends from the ``## Current status`` heading up to (but not
    including) the next top-level ``## `` heading or end of file.
    """
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("## Current status"):
            start = idx
            break

    if start is None:
        # No status section: append one (with a separating blank line).
        suffix = "" if text.endswith("\n") or not text else "\n"
        return text + suffix + "\n" + "\n".join(status_lines) + "\n"

    # Find the end of the block: the next "## " heading after the start.
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break

    # Preserve a blank separator line before any following section.
    tail = lines[end:]
    if tail and tail[0].strip() != "":
        tail = [""] + tail

    new_lines = lines[:start] + status_lines + tail
    return "\n".join(new_lines) + "\n"


def _append_changelog(text: str, timestamp: str, description: str) -> str:
    """Append a changelog bullet under ``## Changelog`` (creating it if needed)."""
    entry = f"- {timestamp}: {description}"
    if "## Changelog" in text:
        body = text if text.endswith("\n") else text + "\n"
        return body + entry + "\n"

    separator = "" if text.endswith("\n") or not text else "\n"
    return text + separator + "\n## Changelog\n" + entry + "\n"


def update_claude_md(config: dict, results: dict, timestamp: str) -> None:
    """Update the ``## Current status`` and ``## Changelog`` of ``CLAUDE.md``.

    Args:
        config: Run configuration (see :func:`log_run`).
        results: Per-portfolio metrics (see :func:`log_run`).
        timestamp: Human-readable run timestamp (``YYYY-MM-DD HH:MM:SS``).

    Rewrites the ``## Current status`` block beneath its heading and appends a
    dated entry to ``## Changelog`` (creating that section if it is missing).
    The function never raises; on any failure it logs a warning and returns.
    """
    config = config or {}
    results = results or {}

    try:
        text = CLAUDE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("%s not found; skipping CLAUDE.md update", CLAUDE_PATH)
        return
    except OSError as exc:
        logger.warning("Failed to read %s: %s", CLAUDE_PATH, exc)
        return

    status_lines = _build_status_block(config, results, timestamp)
    tickers = _format_tickers(config)
    description = f"ran optimizer on {tickers}, updated outputs"

    try:
        text = _replace_status_section(text, status_lines)
        text = _append_changelog(text, timestamp, description)
        CLAUDE_PATH.write_text(text, encoding="utf-8")
        logger.info("Updated current status and changelog in %s", CLAUDE_PATH)
    except OSError as exc:
        logger.warning("Failed to write %s: %s", CLAUDE_PATH, exc)
