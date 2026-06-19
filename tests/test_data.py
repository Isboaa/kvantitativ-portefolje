"""Tests for src.data.fetcher and src.data.cache.

All network access is mocked: ``yfinance`` is patched so no real HTTP requests
are made and the suite is fully deterministic.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.data import cache as cache_mod
from src.data import fetcher as fetcher_mod
from src.data.cache import load_or_fetch
from src.data.fetcher import fetch_prices

TICKERS = ["AAA", "BBB", "CCC"]
START = "2022-01-01"
END = "2022-03-01"


def _make_download_frame(tickers, dates) -> pd.DataFrame:
    """Build a yfinance-style multi-index download frame.

    yfinance returns columns as a MultiIndex of (field, ticker) when called
    with ``group_by`` defaults; we provide an "Adj Close"/"Close" layout.
    """
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    columns = pd.MultiIndex.from_product([fields, tickers])
    rng = np.random.default_rng(123)
    data = 100 + np.cumsum(rng.standard_normal((len(dates), len(columns))), axis=0)
    return pd.DataFrame(data, index=dates, columns=columns)


@pytest.fixture
def fake_dates() -> pd.DatetimeIndex:
    return pd.bdate_range(START, END)


@pytest.fixture
def patched_yfinance(fake_dates):
    """Patch the ``yfinance`` module referenced by the fetcher.

    Yields the mock download callable so tests can inspect call counts.
    """
    frame = _make_download_frame(TICKERS, fake_dates)
    fake_yf = mock.MagicMock()
    fake_yf.download.return_value = frame

    # Patch whichever attribute the fetcher module exposes for yfinance.
    target = "yf" if hasattr(fetcher_mod, "yf") else "yfinance"
    if not hasattr(fetcher_mod, target):
        target = "yf"
    with mock.patch.object(fetcher_mod, target, fake_yf, create=True):
        yield fake_yf


class TestFetchPrices:
    def test_returns_dataframe(self, patched_yfinance) -> None:
        df = fetch_prices(TICKERS, START, END)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_no_network_call_under_mock(self, patched_yfinance) -> None:
        fetch_prices(TICKERS, START, END)
        assert patched_yfinance.download.called

    def test_handles_missing_ticker_gracefully(self, fake_dates) -> None:
        """A ticker with no data should not crash the fetcher.

        We simulate yfinance returning data for only a subset of tickers.
        """
        available = ["AAA", "BBB"]
        requested = ["AAA", "BBB", "ZZZZ"]  # ZZZZ has no data
        frame = _make_download_frame(available, fake_dates)
        fake_yf = mock.MagicMock()
        fake_yf.download.return_value = frame

        target = "yf" if hasattr(fetcher_mod, "yf") else "yfinance"
        if not hasattr(fetcher_mod, target):
            target = "yf"
        with mock.patch.object(fetcher_mod, target, fake_yf, create=True):
            df = fetch_prices(requested, START, END)
        # Should return successfully with at most the available tickers.
        assert isinstance(df, pd.DataFrame)
        assert "ZZZZ" not in df.columns


class TestLoadOrFetch:
    def test_caches_and_returns_identical_frame(self, patched_yfinance, tmp_path) -> None:
        """Second call must return an identical frame without re-downloading."""
        cache_dir = str(tmp_path / "cache")

        # Patch the fetcher used inside the cache module if it imports it directly.
        first = load_or_fetch(TICKERS, START, END, cache_dir=cache_dir)
        calls_after_first = patched_yfinance.download.call_count

        second = load_or_fetch(TICKERS, START, END, cache_dir=cache_dir)
        calls_after_second = patched_yfinance.download.call_count

        # No additional download on the second call.
        assert calls_after_second == calls_after_first
        # Parquet round-trip does not preserve the DatetimeIndex freq metadata,
        # so compare data/values without the freq attribute.
        pd.testing.assert_frame_equal(first, second, check_freq=False)

    def test_creates_cache_artifact(self, patched_yfinance, tmp_path) -> None:
        cache_dir = tmp_path / "cache"
        load_or_fetch(TICKERS, START, END, cache_dir=str(cache_dir))
        # Some cache file should have been written.
        assert cache_dir.exists()
        assert any(cache_dir.iterdir())
