"""Tests for src.models.metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.metrics import (
    max_drawdown,
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio,
    sortino_ratio,
)


class TestPortfolioReturn:
    def test_finite(self, equal_weights, sample_returns) -> None:
        r = portfolio_return(equal_weights, sample_returns)
        assert np.isfinite(float(r))

    def test_flat_returns_zero(self, equal_weights, flat_returns) -> None:
        r = portfolio_return(equal_weights, flat_returns)
        assert float(r) == pytest.approx(0.0, abs=1e-12)


class TestPortfolioVolatility:
    def test_non_negative(self, equal_weights, sample_returns) -> None:
        vol = portfolio_volatility(equal_weights, sample_returns)
        assert float(vol) >= 0.0

    def test_flat_returns_zero_volatility(self, equal_weights, flat_returns) -> None:
        vol = portfolio_volatility(equal_weights, flat_returns)
        assert float(vol) == pytest.approx(0.0, abs=1e-12)


class TestSharpeRatio:
    def test_flat_returns_zero(self, equal_weights, flat_returns) -> None:
        """Sharpe on a zero-return, zero-vol series should be 0 (not NaN/inf)."""
        sr = sharpe_ratio(equal_weights, flat_returns, risk_free_rate=0.0)
        assert float(sr) == pytest.approx(0.0, abs=1e-9)

    def test_finite_on_real_returns(self, equal_weights, sample_returns) -> None:
        sr = sharpe_ratio(equal_weights, sample_returns)
        assert np.isfinite(float(sr))


class TestSortinoRatio:
    def test_finite(self, equal_weights, sample_returns) -> None:
        sr = sortino_ratio(equal_weights, sample_returns)
        assert np.isfinite(float(sr))

    def test_at_least_sharpe_when_positively_skewed(self) -> None:
        """For positively-skewed returns, downside deviation <= total stdev,
        so the Sortino ratio should be >= the Sharpe ratio.
        """
        rng = np.random.default_rng(7)
        # Positively-skewed return stream: many small values, few large gains.
        skewed = rng.lognormal(mean=-6.0, sigma=1.0, size=2000) - 0.001
        returns_df = pd.DataFrame({"X": skewed})
        weights = np.array([1.0])

        sharpe = float(sharpe_ratio(weights, returns_df, risk_free_rate=0.0))
        sortino = float(sortino_ratio(weights, returns_df, risk_free_rate=0.0))
        assert sortino >= sharpe - 1e-8


class TestMaxDrawdown:
    def test_monotonically_increasing_is_zero(self) -> None:
        cumulative = pd.Series(np.linspace(1.0, 2.0, 100))
        dd = max_drawdown(cumulative)
        assert float(dd) == pytest.approx(0.0, abs=1e-9)

    def test_monotonically_decreasing_correct_value(self) -> None:
        # Falls from 100 to 50 -> max drawdown of -0.5 (or 0.5 in magnitude).
        cumulative = pd.Series(np.linspace(100.0, 50.0, 50))
        dd = float(max_drawdown(cumulative))
        assert abs(dd) == pytest.approx(0.5, abs=1e-6)

    def test_known_path(self) -> None:
        # Peak 100 -> trough 60 gives a 40% drawdown regardless of later recovery.
        cumulative = pd.Series([100.0, 110.0, 60.0, 80.0, 120.0])
        dd = float(max_drawdown(cumulative))
        # Largest decline is 110 -> 60 = -0.4545...
        expected = (60.0 - 110.0) / 110.0
        assert abs(dd) == pytest.approx(abs(expected), abs=1e-6)
