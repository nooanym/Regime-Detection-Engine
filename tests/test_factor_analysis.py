"""Tests for rde.analysis.factor_analysis (Phase 23)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.analysis.factor_analysis import (
    FactorConfig,
    RegimeFactorResult,
    RegimeFactors,
    factor_alignment,
    factor_return_decomposition,
    fit_regime_factors,
    project_to_factors,
    rolling_factor_exposure,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_data(T: int = 80, d: int = 4, K: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, d)) * 0.02
    raw = np.abs(rng.standard_normal((T, K)))
    post = raw / raw.sum(axis=1, keepdims=True)
    return X, post


def _fit(T: int = 80, d: int = 4, K: int = 2, nf: int = 2, seed: int = 0):
    X, post = _random_data(T, d, K, seed)
    cfg = FactorConfig(n_factors=nf)
    return fit_regime_factors(X, post, config=cfg), X, post


# ---------------------------------------------------------------------------
# fit_regime_factors
# ---------------------------------------------------------------------------


class TestFitRegimeFactors:
    def test_returns_result(self) -> None:
        result, _, _ = _fit()
        assert isinstance(result, RegimeFactorResult)

    def test_n_regime_factors(self) -> None:
        K = 3
        X, post = _random_data(K=K)
        result = fit_regime_factors(X, post)
        assert len(result.regime_factors) == K

    def test_global_factors_regime_minus_one(self) -> None:
        result, _, _ = _fit()
        assert result.global_factors.regime == -1

    def test_loadings_shape(self) -> None:
        d, nf = 5, 3
        X, post = _random_data(d=d)
        result = fit_regime_factors(X, post, config=FactorConfig(n_factors=nf))
        for rf in result.regime_factors:
            assert rf.loadings.shape == (d, nf)

    def test_global_loadings_shape(self) -> None:
        d, nf = 5, 3
        X, post = _random_data(d=d)
        result = fit_regime_factors(X, post, config=FactorConfig(n_factors=nf))
        assert result.global_factors.loadings.shape == (d, nf)

    def test_evr_sums_to_le_one(self) -> None:
        result, _, _ = _fit()
        for rf in result.regime_factors + [result.global_factors]:
            assert rf.explained_variance_ratio.sum() <= 1.0 + 1e-9

    def test_evr_non_negative(self) -> None:
        result, _, _ = _fit()
        for rf in result.regime_factors + [result.global_factors]:
            assert np.all(rf.explained_variance_ratio >= -1e-10)

    def test_loadings_orthonormal_columns(self) -> None:
        result, _, _ = _fit()
        for rf in result.regime_factors + [result.global_factors]:
            L = rf.loadings
            gram = L.T @ L
            np.testing.assert_allclose(gram, np.eye(L.shape[1]), atol=1e-8)

    def test_factor_stability_shape(self) -> None:
        nf = 2
        result, _, _ = _fit(nf=nf)
        assert result.factor_stability.shape == (nf,)

    def test_factor_stability_in_range(self) -> None:
        result, _, _ = _fit()
        assert np.all(result.factor_stability >= 0.0)
        assert np.all(result.factor_stability <= 1.0 + 1e-9)

    def test_accepts_dataframe(self) -> None:
        X, post = _random_data()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        result = fit_regime_factors(df, post)
        assert isinstance(result, RegimeFactorResult)

    def test_n_factors_capped_at_d(self) -> None:
        d = 3
        X, post = _random_data(d=d)
        result = fit_regime_factors(X, post, config=FactorConfig(n_factors=10))
        assert result.global_factors.loadings.shape[1] == d

    def test_eff_obs_positive(self) -> None:
        result, _, _ = _fit()
        for rf in result.regime_factors:
            assert rf.eff_obs >= 0.0

    def test_mean_shape(self) -> None:
        d = 4
        X, post = _random_data(d=d)
        result = fit_regime_factors(X, post)
        for rf in result.regime_factors + [result.global_factors]:
            assert rf.mean.shape == (d,)

    def test_low_eff_obs_falls_back_gracefully(self) -> None:
        T, d, K = 50, 3, 4
        rng = np.random.default_rng(5)
        X = rng.standard_normal((T, d)) * 0.01
        # One regime gets all mass, others get nearly zero
        post = np.zeros((T, K))
        post[:, 0] = 1.0
        cfg = FactorConfig(n_factors=2, min_eff_obs=5.0)
        result = fit_regime_factors(X, post, config=cfg)
        assert len(result.regime_factors) == K

    def test_single_regime(self) -> None:
        X, _ = _random_data()
        T = len(X)
        post = np.ones((T, 1))
        result = fit_regime_factors(X, post)
        assert len(result.regime_factors) == 1


# ---------------------------------------------------------------------------
# factor_alignment
# ---------------------------------------------------------------------------


class TestFactorAlignment:
    def test_returns_result(self) -> None:
        result, _, _ = _fit()
        realigned = factor_alignment(result)
        assert isinstance(realigned, RegimeFactorResult)

    def test_in_place_and_returned(self) -> None:
        result, _, _ = _fit()
        realigned = factor_alignment(result)
        assert realigned is result

    def test_loadings_still_orthonormal(self) -> None:
        result, _, _ = _fit()
        factor_alignment(result)
        for rf in result.regime_factors:
            gram = rf.loadings.T @ rf.loadings
            np.testing.assert_allclose(gram, np.eye(gram.shape[0]), atol=1e-8)

    def test_positive_dot_with_global(self) -> None:
        result, _, _ = _fit()
        factor_alignment(result)
        for rf in result.regime_factors:
            nf = min(rf.loadings.shape[1], result.global_factors.loadings.shape[1])
            for f in range(nf):
                dot = np.dot(rf.loadings[:, f], result.global_factors.loadings[:, f])
                assert dot >= -1e-10, f"Regime {rf.regime} factor {f}: dot={dot}"


# ---------------------------------------------------------------------------
# rolling_factor_exposure
# ---------------------------------------------------------------------------


class TestRollingFactorExposure:
    def test_returns_dataframe(self) -> None:
        result, X, post = _fit()
        df = rolling_factor_exposure(X, post, result)
        assert isinstance(df, pd.DataFrame)

    def test_shape(self) -> None:
        T, nf = 80, 2
        result, X, post = _fit(T=T, nf=nf)
        df = rolling_factor_exposure(X, post, result)
        assert df.shape == (T, nf)

    def test_column_names(self) -> None:
        nf = 2
        result, X, post = _fit(nf=nf)
        df = rolling_factor_exposure(X, post, result)
        for f in range(nf):
            assert f"factor_{f}" in df.columns

    def test_no_nan(self) -> None:
        result, X, post = _fit()
        df = rolling_factor_exposure(X, post, result)
        assert not df.isnull().any().any()

    def test_preserves_dataframe_index(self) -> None:
        T = 40
        X, post = _random_data(T=T)
        idx = pd.date_range("2024-01-01", periods=T, freq="D")
        df_X = pd.DataFrame(X, index=idx)
        result = fit_regime_factors(df_X, post, config=FactorConfig(n_factors=2))
        exposure = rolling_factor_exposure(df_X, post, result)
        assert (exposure.index == idx).all()

    def test_different_asset_idx(self) -> None:
        result, X, post = _fit(d=4)
        df0 = rolling_factor_exposure(X, post, result, asset_idx=0)
        df2 = rolling_factor_exposure(X, post, result, asset_idx=2)
        # Different columns generally
        assert not np.allclose(df0.values, df2.values)


# ---------------------------------------------------------------------------
# project_to_factors
# ---------------------------------------------------------------------------


class TestProjectToFactors:
    def test_returns_dataframe(self) -> None:
        result, X, post = _fit()
        scores = project_to_factors(X, post, result)
        assert isinstance(scores, pd.DataFrame)

    def test_shape(self) -> None:
        T, nf = 80, 2
        result, X, post = _fit(T=T, nf=nf)
        scores = project_to_factors(X, post, result)
        assert scores.shape == (T, nf)

    def test_no_nan(self) -> None:
        result, X, post = _fit()
        scores = project_to_factors(X, post, result)
        assert not scores.isnull().any().any()

    def test_column_names(self) -> None:
        nf = 3
        result, X, post = _fit(nf=nf)
        scores = project_to_factors(X, post, result)
        for f in range(nf):
            assert f"factor_{f}" in scores.columns

    def test_accepts_dataframe_input(self) -> None:
        result, X, post = _fit()
        df = pd.DataFrame(X)
        scores = project_to_factors(df, post, result)
        assert isinstance(scores, pd.DataFrame)


# ---------------------------------------------------------------------------
# factor_return_decomposition
# ---------------------------------------------------------------------------


class TestFactorReturnDecomposition:
    def test_returns_dataframe(self) -> None:
        result, X, post = _fit()
        scores = project_to_factors(X, post, result)
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.standard_normal(len(X)) * 0.01)
        decomp = factor_return_decomposition(returns, scores)
        assert isinstance(decomp, pd.DataFrame)

    def test_columns_include_residual(self) -> None:
        result, X, post = _fit()
        scores = project_to_factors(X, post, result)
        returns = pd.Series(np.zeros(len(X)))
        decomp = factor_return_decomposition(returns, scores)
        assert "residual" in decomp.columns

    def test_factor_columns_present(self) -> None:
        nf = 2
        result, X, post = _fit(nf=nf)
        scores = project_to_factors(X, post, result)
        returns = pd.Series(np.zeros(len(X)))
        decomp = factor_return_decomposition(returns, scores)
        for f in range(nf):
            assert f"factor_{f}" in decomp.columns

    def test_contributions_plus_residual_equals_returns(self) -> None:
        result, X, post = _fit()
        scores = project_to_factors(X, post, result)
        rng = np.random.default_rng(3)
        returns = pd.Series(rng.standard_normal(len(X)) * 0.01)
        decomp = factor_return_decomposition(returns, scores)
        reconstructed = decomp.sum(axis=1)
        np.testing.assert_allclose(reconstructed.values, returns.values, atol=1e-10)

    def test_zero_returns_all_residual(self) -> None:
        result, X, post = _fit()
        scores = project_to_factors(X, post, result)
        returns = pd.Series(np.zeros(len(X)))
        decomp = factor_return_decomposition(returns, scores)
        np.testing.assert_allclose(decomp.drop(columns="residual").values,
                                   np.zeros((len(X), scores.shape[1])), atol=1e-10)
