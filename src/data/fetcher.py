"""Download adjusted closing prices via yfinance.

Runnable as a module for a quick smoke test::

    python -m src.data.fetcher --tickers AAPL MSFT TSLA \\
        --start 2020-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Number of download attempts and the base (seconds) for exponential back-off
# when Yahoo rate-limits or returns an empty frame.
_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 2.0


class RateLimitError(ValueError):
    """Raised when Yahoo Finance rate-limits the price download.

    Subclasses :class:`ValueError` so existing callers that catch ``ValueError``
    keep working, while newer callers (e.g. the web app) can catch this specific
    type to show an honest "wait and retry" message.
    """


def _is_rate_limited(exc: Exception | None) -> bool:
    """Best-effort detection of a Yahoo rate-limit condition.

    yfinance signals throttling in two ways depending on version: it may raise
    ``YFRateLimitError``, or it may swallow the error and merely return an empty
    frame while recording the reason in ``yf.shared._ERRORS``. This checks both.

    Args:
        exc: The exception raised by ``yf.download`` (if any).

    Returns:
        ``True`` if the failure looks like rate-limiting.
    """
    needles = ("rate limit", "too many requests", "yfratelimit")

    if exc is not None:
        if type(exc).__name__ == "YFRateLimitError":
            return True
        if any(n in str(exc).lower() for n in needles):
            return True

    # Inspect yfinance's per-ticker error registry when the call returned empty.
    errors = getattr(getattr(yf, "shared", None), "_ERRORS", None)
    if isinstance(errors, dict):
        blob = " ".join(str(v) for v in errors.values()).lower()
        if any(n in blob for n in needles):
            return True

    return False


def _download_with_retry(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Call ``yf.download`` with retries and exponential back-off.

    Returns the first non-empty frame. If every attempt fails, raises
    :class:`RateLimitError` when the failure looks like throttling, otherwise a
    plain :class:`ValueError`.

    Args:
        tickers: De-duplicated ticker symbols to download.
        start: Inclusive start date (YYYY-MM-DD).
        end: Exclusive end date (YYYY-MM-DD).

    Returns:
        The raw yfinance download frame.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        raw = None
        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="column",
            )
        except Exception as exc:  # noqa: BLE001 - retried/classified below
            last_exc = exc
            logger.warning("yfinance download attempt %d failed: %s", attempt + 1, exc)

        if raw is not None and not raw.empty:
            return raw

        if attempt < _MAX_ATTEMPTS - 1:
            delay = _BACKOFF_BASE ** attempt
            logger.info("Retrying download in %.0fs (attempt %d/%d)…",
                        delay, attempt + 2, _MAX_ATTEMPTS)
            time.sleep(delay)

    if _is_rate_limited(last_exc):
        raise RateLimitError(
            "Yahoo Finance is rate-limiting requests right now. Please wait a "
            "minute and try again. (Already-analyzed stocks are cached and still "
            "work offline.)"
        )
    if last_exc is not None:
        raise ValueError(f"yfinance download failed: {last_exc}") from last_exc
    raise ValueError(
        f"yfinance returned no data for tickers {tickers} in range {start}..{end}."
    )


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted closing prices for ``tickers`` between two dates.

    Adjusted closes are obtained from yfinance (``auto_adjust=True``), so the
    returned ``Close`` column already incorporates splits and dividends.

    Tickers that return no usable data are logged as a warning and dropped.
    The returned frame is forward-filled (to bridge sporadic gaps) and any rows
    still containing NaNs are removed, yielding a clean, NaN-free price matrix.

    Args:
        tickers: List of ticker symbols, e.g. ``["AAPL", "MSFT"]``.
        start: Inclusive start date in ``YYYY-MM-DD`` format.
        end: Exclusive end date in ``YYYY-MM-DD`` format.

    Returns:
        A DataFrame with a :class:`~pandas.DatetimeIndex`, one column per
        surviving ticker, and no NaN values.

    Raises:
        ValueError: If ``tickers`` is empty or no ticker returns any data.
    """
    if not tickers:
        raise ValueError("`tickers` must contain at least one symbol.")

    # De-duplicate while preserving a deterministic order.
    unique_tickers = sorted(set(tickers))
    logger.info(
        "Fetching %d ticker(s) from %s to %s: %s",
        len(unique_tickers),
        start,
        end,
        ", ".join(unique_tickers),
    )

    raw = _download_with_retry(unique_tickers, start, end)

    prices = _extract_close(raw, unique_tickers)

    # Drop tickers that returned no usable data at all.
    empty_cols = [t for t in prices.columns if prices[t].isna().all()]
    if empty_cols:
        logger.warning(
            "Dropping %d ticker(s) with no data: %s",
            len(empty_cols),
            ", ".join(empty_cols),
        )
        prices = prices.drop(columns=empty_cols)

    missing = [t for t in unique_tickers if t not in prices.columns]
    if missing:
        logger.warning(
            "Requested tickers absent from yfinance response: %s",
            ", ".join(missing),
        )

    if prices.shape[1] == 0:
        raise ValueError("No tickers returned usable price data.")

    # Forward-fill sporadic gaps, then drop any rows that are still incomplete
    # (e.g. leading NaNs before a ticker started trading).
    prices = prices.sort_index().ffill().dropna(how="any")

    prices.index = pd.DatetimeIndex(prices.index, name="Date")
    prices.columns.name = None

    logger.info(
        "Returning price data with shape %s covering %s to %s.",
        prices.shape,
        prices.index.min().date() if not prices.empty else "n/a",
        prices.index.max().date() if not prices.empty else "n/a",
    )
    return prices


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull the ``Close`` price columns out of a yfinance response.

    yfinance returns a single-level frame for one ticker and a MultiIndex
    (column level) frame for several. This normalises both into a frame whose
    columns are ticker symbols.

    Args:
        raw: The raw frame returned by :func:`yfinance.download`.
        tickers: The de-duplicated tickers that were requested.

    Returns:
        A DataFrame of close prices with ticker columns.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("yfinance response did not contain 'Close' prices.")
        close = raw["Close"].copy()
    else:
        # Single ticker: columns are price fields (Open/High/Low/Close/...).
        if "Close" not in raw.columns:
            raise ValueError("yfinance response did not contain a 'Close' column.")
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    return close


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser for the smoke test."""
    parser = argparse.ArgumentParser(
        description="Fetch adjusted closing prices via yfinance.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="One or more ticker symbols, e.g. AAPL MSFT TSLA.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Inclusive start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="Exclusive end date (YYYY-MM-DD).",
    )
    return parser


def main() -> None:
    """CLI entry point: fetch prices and print shape plus the first 5 rows."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_arg_parser().parse_args()
    prices = fetch_prices(args.tickers, args.start, args.end)
    print(f"Shape: {prices.shape}")
    print(prices.head())


if __name__ == "__main__":
    main()
