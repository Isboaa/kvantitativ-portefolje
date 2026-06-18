"""Parquet-backed caching layer for price data.

Avoids redundant network fetches by storing each unique
(ticker set, date range) request as a parquet file on disk.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from src.data.fetcher import fetch_prices

logger = logging.getLogger(__name__)


def _cache_key(tickers: list[str], start: str, end: str) -> str:
    """Compute a deterministic md5 cache key for a request.

    The key is derived from the sorted, de-duplicated ticker set together with
    the start and end dates, so requests with the same tickers (in any order)
    and date range map to the same cache file.

    Args:
        tickers: Requested ticker symbols.
        start: Inclusive start date (YYYY-MM-DD).
        end: Exclusive end date (YYYY-MM-DD).

    Returns:
        A hex-encoded md5 digest.
    """
    canonical = "|".join(sorted(set(tickers))) + f"|{start}|{end}"
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()


def load_or_fetch(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """Return cached prices if available, otherwise fetch and cache them.

    A parquet file keyed by the md5 hash of the sorted tickers plus the date
    range is looked up in ``cache_dir``. On a hit it is loaded and returned; on
    a miss :func:`~src.data.fetcher.fetch_prices` is called, the result is
    persisted as parquet, and then returned.

    Args:
        tickers: Ticker symbols to load.
        start: Inclusive start date (YYYY-MM-DD).
        end: Exclusive end date (YYYY-MM-DD).
        cache_dir: Directory (relative to project root) for parquet files.

    Returns:
        A clean price DataFrame as produced by ``fetch_prices``.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    key = _cache_key(tickers, start, end)
    parquet_file = cache_path / f"{key}.parquet"

    if parquet_file.exists():
        logger.info("Cache hit: loading prices from %s", parquet_file)
        prices = pd.read_parquet(parquet_file)
        prices.index = pd.DatetimeIndex(prices.index, name="Date")
        return prices

    logger.info("Cache miss for key %s; fetching from source.", key)
    prices = fetch_prices(tickers, start, end)
    prices.to_parquet(parquet_file)
    logger.info("Cached prices to %s", parquet_file)
    return prices
