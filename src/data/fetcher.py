"""Download adjusted closing prices via yfinance.

Runnable as a module for a quick smoke test::

    python -m src.data.fetcher --tickers AAPL MSFT TSLA \\
        --start 2020-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


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

    raw = yf.download(
        tickers=unique_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if raw is None or raw.empty:
        raise ValueError(
            f"yfinance returned no data for tickers {unique_tickers} "
            f"in range {start}..{end}."
        )

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
