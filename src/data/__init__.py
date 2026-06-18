"""Data fetching and caching for the quantitative portfolio project.

Public API:
    fetch_prices:  Download adjusted closing prices via yfinance.
    load_or_fetch: Parquet-backed cache wrapper around ``fetch_prices``.
"""

from src.data.cache import load_or_fetch
from src.data.fetcher import fetch_prices

__all__ = ["fetch_prices", "load_or_fetch"]
