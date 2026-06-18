"""Tests for src.models.covariance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.covariance import (
    ledoit_wolf,
    rolling_covariance,
    sample_covariance,
)


def _as_matrix(cov) -> np.ndarray:
    """Coerce a covariance estimator output to a 2D numpy array."""
    if isinstance(cov, pd.DataFrame):
        return cov.to_numpy()
    return np.asarray(cov)


def _is_symmetric(mat: np.ndarray, atol: float = 1e-8) -> bool:
    return np.allclose(mat, mat.T, atol=atol)


def _is_psd(mat: np.ndarray, tol: float = 1e-8) -> bool:
    # Symmetrise before eigen-decomposition to avoid spurious complex parts.
    sym = (mat + mat.T) / 2.0
    eigvals = np.linalg.eigvalsh(sym)
    return bool((eigvals >= -tol).all())


class TestSampleCovariance:
    def test_symmetric(self, sample_prices: pd.DataFrame) -> None:
        mat = _as_matrix(sample_covariance(sample_prices))
        assert _is_symmetric(mat)

    def test_positive_semidefinite(self, sample_prices: pd.DataFrame) -> None:
        mat = _as_matrix(sample_covariance(sample_prices))
        assert _is_psd(mat)

    def test_shape_matches_universe(self, sample_prices: pd.DataFrame) -> None:
        mat = _as_matrix(sample_covariance(sample_prices))
        n = sample_prices.shape[1]
        assert mat.shape == (n, n)


class TestLedoitWolf:
    def test_symmetric(self, sample_prices: pd.DataFrame) -> None:
        out = ledoit_wolf(sample_prices)
        mat = _as_matrix(out[0] if isinstance(out, tuple) else out)
        assert _is_symmetric(mat)

    def test_positive_semidefinite(self, sample_prices: pd.DataFrame) -> None:
        out = ledoit_wolf(sample_prices)
        mat = _as_matrix(out[0] if isinstance(out, tuple) else out)
        assert _is_psd(mat)

    def test_shrinkage_coefficient_in_unit_interval(
        self, sample_prices: pd.DataFrame
    ) -> None:
        """The Ledoit-Wolf shrinkage intensity must lie in [0, 1].

        Implementations may expose it either as a second tuple element or as a
        ``shrinkage`` attribute on the returned object.
        """
        out = ledoit_wolf(sample_prices)
        shrinkage = None
        if isinstance(out, tuple) and len(out) >= 2:
            shrinkage = float(out[1])
        elif hasattr(out, "shrinkage"):
            shrinkage = float(out.shrinkage)
        if shrinkage is None:
            pytest.skip("Implementation does not expose the shrinkage coefficient")
        assert 0.0 <= shrinkage <= 1.0


class TestRollingCovariance:
    def test_symmetric(self, sample_prices: pd.DataFrame) -> None:
        out = rolling_covariance(sample_prices, window=126)
        mat = _as_matrix(out)
        # rolling_covariance may return a single most-recent matrix.
        if mat.ndim == 2:
            assert _is_symmetric(mat)

    def test_positive_semidefinite(self, sample_prices: pd.DataFrame) -> None:
        out = rolling_covariance(sample_prices, window=126)
        mat = _as_matrix(out)
        if mat.ndim == 2:
            assert _is_psd(mat)
