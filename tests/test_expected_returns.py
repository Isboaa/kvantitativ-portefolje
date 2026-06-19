"""Tests for src.models.expected_returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.expected_returns import (
    ema_historical_return,
    mean_historical_return,
)


class TestMeanHistoricalReturn:
    def test_returns_series_with_ticker_index(self, sample_prices: pd.DataFrame) -> None:
        mu = mean_historical_return(sample_prices)
        assert isinstance(mu, pd.Series)
        assert list(mu.index) == list(sample_prices.columns)

    def test_no_nans(self, sample_prices: pd.DataFrame) -> None:
        mu = mean_historical_return(sample_prices)
        assert not mu.isna().any()

    def test_annualisation_scales_with_frequency(self, sample_prices: pd.DataFrame) -> None:
        daily = mean_historical_return(sample_prices, frequency=1)
        annual = mean_historical_return(sample_prices, frequency=252)
        # Annualised should equal daily scaled by the frequency factor.
        np.testing.assert_allclose(annual.values, daily.values * 252, rtol=1e-6)

    def test_matches_manual_computation(self, sample_prices: pd.DataFrame) -> None:
        # The estimator annualizes the mean of daily *log* returns.
        log_returns = np.log(sample_prices / sample_prices.shift(1)).dropna()
        expected = log_returns.mean() * 252
        mu = mean_historical_return(sample_prices, frequency=252)
        np.testing.assert_allclose(mu.values, expected.values, rtol=1e-6)


class TestEmaHistoricalReturn:
    def test_returns_series_with_ticker_index(self, sample_prices: pd.DataFrame) -> None:
        mu = ema_historical_return(sample_prices)
        assert isinstance(mu, pd.Series)
        assert list(mu.index) == list(sample_prices.columns)

    def test_no_nans(self, sample_prices: pd.DataFrame) -> None:
        mu = ema_historical_return(sample_prices)
        assert not mu.isna().any()

    def test_finite_values(self, sample_prices: pd.DataFrame) -> None:
        mu = ema_historical_return(sample_prices, span=60)
        assert np.isfinite(mu.values).all()

    def test_span_affects_result(self, sample_prices: pd.DataFrame) -> None:
        short = ema_historical_return(sample_prices, span=20)
        long = ema_historical_return(sample_prices, span=200)
        # Different spans should generally produce different estimates.
        assert not np.allclose(short.values, long.values)
