"""Tests for rde.analysis.correlation (Phase 30)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.analysis.correlation import (
    CorrelationConfig,
    RegimeCorrelation,
    RegimeCorrelationResult,
    correlation_breakdown,
    dynamic_conditional_correlation,
    regime_correlation,
    tail_dependence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _data(T: int = 80, d: int = 3, K: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, d)) * 0.01
    raw = np.abs(rng.standard_normal((T, K)))
    post = raw / raw.sum(axis=1, keepdims=True)
    return X, post


def _fit(T: int = 80, d: int = 3, K: int = 2, seed: int = 0,
         method: str = "pearson"):
    X, post = _data(T, d, K, seed)
    cfg = CorrelationConfig(method=method)
    return regime_correlation(X, post, config=cfg), X, post


# ---------------------------------------------------------------------------
# regime_correlation
# ---------------------------------------------------------------------------


class TestRegimeCorrelation:
    def test_returns_result(self) -> None:
        result, _, _ = _fit()
        assert isinstance(result, RegimeCorrelationResult)

    def test_n_regime_corrs(self) -> None:
        K = 3
        X, post = _data(K=K)
        result = regime_correlation(X, post)
        assert len(result.regime_corrs) == K

    def test_global_corr_regime_minus_one(self) -> None:
        result, _, _ = _fit()
        assert result.global_corr.regime == -1

    def test_corr_shape(self) -> None:
        d = 4
        X, post = _data(d=d)
        result = regime_correlation(X, post)
        for rc in result.regime_corrs + [result.global_corr]:
            assert rc.corr.shape == (d, d)

    def test_diagonal_ones(self) -> None:
        result, _, _ = _fit()
        for rc in result.regime_corrs + [result.global_corr]:
            np.testing.assert_allclose(np.diag(rc.corr), 1.0, atol=1e-10)

    def test_symmetric(self) -> None:
        result, _, _ = _fit()
        for rc in result.regime_corrs + [result.global_corr]:
            np.testing.assert_allclose(rc.corr, rc.corr.T, atol=1e-10)

    def test_corr_in_range(self) -> None:
        result, _, _ = _fit()
        for rc in result.regime_corrs + [result.global_corr]:
            assert np.all(rc.corr >= -1.0 - 1e-8)
            assert np.all(rc.corr <= 1.0 + 1e-8)

    def test_stability_non_negative(self) -> None:
        result, _, _ = _fit()
        assert result.stability >= 0.0

    def test_perfectly_stable_zero_stability(self) -> None:
        # If both regimes have same data (uniform posteriors), stability ≈ 0
        T, d, K = 60, 3, 2
        X, _ = _data(T, d)
        post = np.full((T, K), 0.5)
        result = regime_correlation(X, post)
        assert result.stability == pytest.approx(0.0, abs=1e-8)

    def test_accepts_dataframe(self) -> None:
        X, post = _data()
        df = pd.DataFrame(X, columns=[f"a{i}" for i in range(X.shape[1])])
        result = regime_correlation(df, post)
        assert isinstance(result, RegimeCorrelationResult)

    def test_spearman_method(self) -> None:
        result, _, _ = _fit(method="spearman")
        for rc in result.regime_corrs:
            np.testing.assert_allclose(np.diag(rc.corr), 1.0, atol=1e-8)

    def test_eff_obs_positive(self) -> None:
        result, _, _ = _fit()
        for rc in result.regime_corrs:
            assert rc.eff_obs >= 0.0

    def test_highly_correlated_data(self) -> None:
        T, K = 100, 2
        rng = np.random.default_rng(0)
        base = rng.standard_normal(T)
        X = np.column_stack([base + rng.standard_normal(T) * 0.01,
                             base + rng.standard_normal(T) * 0.01])
        post = np.full((T, K), 0.5)
        result = regime_correlation(X, post)
        # Both corr matrices should show high correlation
        for rc in result.regime_corrs:
            assert rc.corr[0, 1] > 0.9


# ---------------------------------------------------------------------------
# dynamic_conditional_correlation
# ---------------------------------------------------------------------------


class TestDynamicConditionalCorrelation:
    def test_returns_dataframe(self) -> None:
        result, X, post = _fit()
        dcc = dynamic_conditional_correlation(X, post, result)
        assert isinstance(dcc, pd.DataFrame)

    def test_shape(self) -> None:
        T, d = 80, 3
        result, X, post = _fit(T=T, d=d)
        dcc = dynamic_conditional_correlation(X, post, result)
        n_pairs = d * (d - 1) // 2
        assert dcc.shape == (T, n_pairs)

    def test_no_nan(self) -> None:
        result, X, post = _fit()
        dcc = dynamic_conditional_correlation(X, post, result)
        assert not dcc.isnull().any().any()

    def test_values_in_range(self) -> None:
        result, X, post = _fit()
        dcc = dynamic_conditional_correlation(X, post, result)
        assert (dcc.values >= -1.0 - 1e-8).all()
        assert (dcc.values <= 1.0 + 1e-8).all()

    def test_column_names(self) -> None:
        d = 3
        X, post = _data(d=d)
        result = regime_correlation(X, post)
        dcc = dynamic_conditional_correlation(X, post, result)
        assert "corr_0_1" in dcc.columns
        assert "corr_0_2" in dcc.columns
        assert "corr_1_2" in dcc.columns

    def test_one_hot_posterior_returns_regime_corr(self) -> None:
        T, d, K = 30, 2, 2
        X, _ = _data(T, d)
        post = np.zeros((T, K))
        post[:, 0] = 1.0
        result = regime_correlation(X, post)
        dcc = dynamic_conditional_correlation(X, post, result)
        expected = float(result.regime_corrs[0].corr[0, 1])
        np.testing.assert_allclose(dcc["corr_0_1"].values, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# tail_dependence
# ---------------------------------------------------------------------------


class TestTailDependence:
    def test_returns_dataframe(self) -> None:
        X, post = _data()
        df = tail_dependence(X, post)
        assert isinstance(df, pd.DataFrame)

    def test_columns(self) -> None:
        X, post = _data()
        df = tail_dependence(X, post)
        for col in ["regime", "pair", "lower_tail", "upper_tail"]:
            assert col in df.columns

    def test_values_in_range(self) -> None:
        X, post = _data()
        df = tail_dependence(X, post)
        assert (df["lower_tail"] >= 0).all()
        assert (df["lower_tail"] <= 1.0 + 1e-8).all()
        assert (df["upper_tail"] >= 0).all()
        assert (df["upper_tail"] <= 1.0 + 1e-8).all()

    def test_n_rows(self) -> None:
        d, K = 3, 2
        X, post = _data(d=d, K=K)
        df = tail_dependence(X, post)
        n_pairs = d * (d - 1) // 2
        assert len(df) == K * n_pairs

    def test_high_correlation_high_tail_dep(self) -> None:
        T, K = 500, 2
        rng = np.random.default_rng(0)
        base = rng.standard_normal(T)
        X = np.column_stack([base, base + rng.standard_normal(T) * 0.001])
        post = np.full((T, K), 0.5)
        df = tail_dependence(X, post)
        # Perfect correlation → high tail dependence
        lo = df["lower_tail"].mean()
        assert lo > 0.5


# ---------------------------------------------------------------------------
# correlation_breakdown
# ---------------------------------------------------------------------------


class TestCorrelationBreakdown:
    def test_returns_series(self) -> None:
        result, X, post = _fit()
        dcc = dynamic_conditional_correlation(X, post, result)
        bd = correlation_breakdown(dcc)
        assert isinstance(bd, pd.Series)

    def test_length(self) -> None:
        T = 80
        result, X, post = _fit(T=T)
        dcc = dynamic_conditional_correlation(X, post, result)
        bd = correlation_breakdown(dcc)
        assert len(bd) == T

    def test_non_negative_where_finite(self) -> None:
        result, X, post = _fit()
        dcc = dynamic_conditional_correlation(X, post, result)
        bd = correlation_breakdown(dcc)
        finite_vals = bd.dropna()
        assert (finite_vals >= 0).all()

    def test_stable_dcc_near_zero_breakdown(self) -> None:
        T, d, K = 100, 2, 2
        X, _ = _data(T, d)
        # Constant posterior → constant DCC → breakdown ≈ 0
        post = np.full((T, K), 0.5)
        result = regime_correlation(X, post)
        dcc = dynamic_conditional_correlation(X, post, result)
        bd = correlation_breakdown(dcc)
        # Constant DCC → rolling std = 0
        np.testing.assert_allclose(bd.dropna().values, 0.0, atol=1e-10)
