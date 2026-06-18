"""Tests for src.models.optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.optimizer import (
    efficient_frontier,
    maximum_sharpe,
    minimum_variance,
    naive_portfolio,
)

RESULT_KEYS = {"weights", "return", "volatility", "sharpe"}


def _weights_array(result: dict, n: int) -> np.ndarray:
    """Extract a weights vector as a numpy array from a result dict."""
    w = result["weights"]
    if isinstance(w, dict):
        w = np.array(list(w.values()), dtype=float)
    elif isinstance(w, pd.Series):
        w = w.to_numpy(dtype=float)
    else:
        w = np.asarray(w, dtype=float)
    assert w.shape == (n,)
    return w


class TestMinimumVariance:
    def test_result_has_expected_keys(self, sample_mu, sample_cov) -> None:
        res = minimum_variance(sample_mu, sample_cov)
        assert RESULT_KEYS.issubset(res.keys())

    def test_weights_sum_to_one(self, sample_mu, sample_cov) -> None:
        res = minimum_variance(sample_mu, sample_cov)
        w = _weights_array(res, len(sample_mu))
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_non_negative(self, sample_mu, sample_cov) -> None:
        res = minimum_variance(sample_mu, sample_cov)
        w = _weights_array(res, len(sample_mu))
        assert (w >= -1e-8).all()

    def test_minimises_volatility_vs_random(self, sample_mu, sample_cov) -> None:
        """Min-variance volatility must not exceed 1000 random long-only ports."""
        res = minimum_variance(sample_mu, sample_cov)
        mv_vol = float(res["volatility"])

        cov = sample_cov.to_numpy() if isinstance(sample_cov, pd.DataFrame) else np.asarray(sample_cov)
        n = cov.shape[0]
        rng = np.random.default_rng(0)
        for _ in range(1000):
            w = rng.random(n)
            w /= w.sum()
            vol = float(np.sqrt(w @ cov @ w))
            # Allow a tiny numerical tolerance.
            assert mv_vol <= vol + 1e-8


class TestMaximumSharpe:
    def test_result_has_expected_keys(self, sample_mu, sample_cov) -> None:
        res = maximum_sharpe(sample_mu, sample_cov)
        assert RESULT_KEYS.issubset(res.keys())

    def test_weights_sum_to_one(self, sample_mu, sample_cov) -> None:
        res = maximum_sharpe(sample_mu, sample_cov)
        w = _weights_array(res, len(sample_mu))
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_non_negative(self, sample_mu, sample_cov) -> None:
        res = maximum_sharpe(sample_mu, sample_cov)
        w = _weights_array(res, len(sample_mu))
        assert (w >= -1e-8).all()

    def test_sharpe_at_least_min_variance(self, sample_mu, sample_cov) -> None:
        ms = maximum_sharpe(sample_mu, sample_cov, risk_free_rate=0.04)
        mv = minimum_variance(sample_mu, sample_cov)
        assert float(ms["sharpe"]) >= float(mv["sharpe"]) - 1e-8


class TestEfficientFrontier:
    def test_returns_dataframe_with_columns(self, sample_mu, sample_cov) -> None:
        df = efficient_frontier(sample_mu, sample_cov, n_points=50)
        assert isinstance(df, pd.DataFrame)
        for col in ("return", "volatility", "sharpe"):
            assert col in df.columns
        for ticker in sample_mu.index:
            assert ticker in df.columns

    def test_returns_monotonically_increasing(self, sample_mu, sample_cov) -> None:
        df = efficient_frontier(sample_mu, sample_cov, n_points=50)
        rets = df["return"].to_numpy()
        diffs = np.diff(rets)
        assert (diffs >= -1e-8).all()

    def test_n_points_respected(self, sample_mu, sample_cov) -> None:
        df = efficient_frontier(sample_mu, sample_cov, n_points=30)
        assert len(df) == 30

    def test_long_only_weights(self, sample_mu, sample_cov) -> None:
        df = efficient_frontier(sample_mu, sample_cov, n_points=40)
        weight_cols = list(sample_mu.index)
        weights = df[weight_cols].to_numpy()
        assert (weights >= -1e-6).all()

    def test_weights_sum_to_one_per_row(self, sample_mu, sample_cov) -> None:
        df = efficient_frontier(sample_mu, sample_cov, n_points=40)
        weight_cols = list(sample_mu.index)
        row_sums = df[weight_cols].to_numpy().sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


class TestNaivePortfolio:
    def test_result_has_expected_keys(self, sample_mu, sample_cov) -> None:
        res = naive_portfolio(sample_mu, sample_cov)
        assert RESULT_KEYS.issubset(res.keys())

    def test_equal_weights(self, sample_mu, sample_cov) -> None:
        res = naive_portfolio(sample_mu, sample_cov)
        n = len(sample_mu)
        w = _weights_array(res, n)
        np.testing.assert_allclose(w, np.full(n, 1.0 / n), atol=1e-8)
