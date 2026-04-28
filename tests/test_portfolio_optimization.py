"""Tests for rde.analysis.portfolio_optimization (Phase 25)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.analysis.portfolio_optimization import (
    EfficientFrontier,
    MVOConfig,
    RegimeMVOResult,
    RegimeMVOWeights,
    black_litterman_posterior,
    blended_mvo,
    efficient_frontier,
    regime_mvo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _returns(T: int = 120, d: int = 3, K: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((T, d)) * 0.01
    raw = np.abs(rng.standard_normal((T, K)))
    post = raw / raw.sum(axis=1, keepdims=True)
    return R, post


def _fit(T: int = 120, d: int = 3, K: int = 2):
    R, post = _returns(T, d, K)
    return regime_mvo(R, post), R, post


# ---------------------------------------------------------------------------
# regime_mvo
# ---------------------------------------------------------------------------


class TestRegimeMVO:
    def test_returns_result(self) -> None:
        result, _, _ = _fit()
        assert isinstance(result, RegimeMVOResult)

    def test_n_regime_weights(self) -> None:
        K = 3
        R, post = _returns(K=K)
        result = regime_mvo(R, post)
        assert len(result.regime_weights) == K

    def test_global_weights_regime_minus_one(self) -> None:
        result, _, _ = _fit()
        assert result.global_weights.regime == -1

    def test_regime_indices(self) -> None:
        K = 3
        R, post = _returns(K=K)
        result = regime_mvo(R, post)
        for k, rw in enumerate(result.regime_weights):
            assert rw.regime == k

    def test_weights_sum_to_one(self) -> None:
        result, _, _ = _fit()
        for rw in result.regime_weights + [result.global_weights]:
            assert rw.min_var_weights.sum() == pytest.approx(1.0, abs=1e-8)
            assert rw.max_sharpe_weights.sum() == pytest.approx(1.0, abs=1e-8)

    def test_weights_non_negative(self) -> None:
        result, _, _ = _fit()
        for rw in result.regime_weights + [result.global_weights]:
            assert np.all(rw.min_var_weights >= -1e-10)
            assert np.all(rw.max_sharpe_weights >= -1e-10)

    def test_weights_le_one(self) -> None:
        result, _, _ = _fit()
        for rw in result.regime_weights + [result.global_weights]:
            assert np.all(rw.min_var_weights <= 1.0 + 1e-10)
            assert np.all(rw.max_sharpe_weights <= 1.0 + 1e-10)

    def test_mean_shape(self) -> None:
        d = 4
        R, post = _returns(d=d)
        result = regime_mvo(R, post)
        for rw in result.regime_weights + [result.global_weights]:
            assert rw.mean.shape == (d,)

    def test_cov_shape(self) -> None:
        d = 4
        R, post = _returns(d=d)
        result = regime_mvo(R, post)
        for rw in result.regime_weights + [result.global_weights]:
            assert rw.cov.shape == (d, d)

    def test_cov_symmetric(self) -> None:
        result, _, _ = _fit()
        for rw in result.regime_weights + [result.global_weights]:
            np.testing.assert_allclose(rw.cov, rw.cov.T, atol=1e-12)

    def test_eff_obs_positive(self) -> None:
        result, _, _ = _fit()
        for rw in result.regime_weights:
            assert rw.eff_obs >= 0.0

    def test_accepts_dataframe(self) -> None:
        R, post = _returns()
        df = pd.DataFrame(R, columns=["A", "B", "C"])
        result = regime_mvo(df, post)
        assert result.asset_names == ["A", "B", "C"]

    def test_numpy_input_no_asset_names(self) -> None:
        R, post = _returns()
        result = regime_mvo(R, post)
        assert result.asset_names is None

    def test_min_var_lower_or_equal_variance_than_max_sharpe(self) -> None:
        # Min-var should have variance ≤ max-sharpe (by definition)
        result, _, _ = _fit(d=5)
        for rw in result.regime_weights + [result.global_weights]:
            var_mv = float(rw.min_var_weights @ rw.cov @ rw.min_var_weights)
            var_ms = float(rw.max_sharpe_weights @ rw.cov @ rw.max_sharpe_weights)
            assert var_mv <= var_ms + 1e-8

    def test_high_eff_obs_regime_uses_own_moments(self) -> None:
        # With one regime dominating all mass, its mean should differ from global.
        T, d = 300, 3
        rng = np.random.default_rng(9)
        R = np.concatenate([
            rng.standard_normal((T // 2, d)) * 0.005 + 0.002,   # high return regime
            rng.standard_normal((T // 2, d)) * 0.015 - 0.002,   # low return regime
        ], axis=0)
        post = np.zeros((T, 2))
        post[:T // 2, 0] = 1.0
        post[T // 2:, 1] = 1.0
        result = regime_mvo(R, post)
        # Regime 0 mean should be positive; regime 1 should be negative
        assert result.regime_weights[0].mean.mean() > 0
        assert result.regime_weights[1].mean.mean() < 0


# ---------------------------------------------------------------------------
# blended_mvo
# ---------------------------------------------------------------------------


class TestBlendedMVO:
    def test_returns_ndarray(self) -> None:
        result, _, _ = _fit()
        w = blended_mvo(result, np.array([0.6, 0.4]))
        assert isinstance(w, np.ndarray)

    def test_shape(self) -> None:
        d = 3
        result, _, _ = _fit(d=d)
        w = blended_mvo(result, np.array([0.5, 0.5]))
        assert w.shape == (d,)

    def test_sums_to_one(self) -> None:
        result, _, _ = _fit()
        w = blended_mvo(result, np.array([0.7, 0.3]))
        assert w.sum() == pytest.approx(1.0, abs=1e-8)

    def test_non_negative(self) -> None:
        result, _, _ = _fit()
        w = blended_mvo(result, np.array([0.5, 0.5]))
        assert np.all(w >= -1e-10)

    def test_min_var_kind(self) -> None:
        result, _, _ = _fit()
        w = blended_mvo(result, np.array([1.0, 0.0]), kind="min_var")
        np.testing.assert_allclose(w, result.regime_weights[0].min_var_weights, rtol=1e-10)

    def test_one_hot_posterior_returns_regime_weights(self) -> None:
        result, _, _ = _fit()
        pt = np.array([1.0, 0.0])
        w = blended_mvo(result, pt, kind="max_sharpe")
        np.testing.assert_allclose(w, result.regime_weights[0].max_sharpe_weights, rtol=1e-10)


# ---------------------------------------------------------------------------
# efficient_frontier
# ---------------------------------------------------------------------------


class TestEfficientFrontier:
    def test_returns_efficient_frontier(self) -> None:
        result, _, _ = _fit()
        ef = efficient_frontier(result.global_weights)
        assert isinstance(ef, EfficientFrontier)

    def test_volatilities_non_negative(self) -> None:
        result, _, _ = _fit()
        ef = efficient_frontier(result.global_weights)
        assert np.all(ef.volatilities >= 0)

    def test_returns_increasing(self) -> None:
        result, _, _ = _fit()
        ef = efficient_frontier(result.global_weights)
        # Returns should be non-decreasing along the frontier
        assert np.all(np.diff(ef.returns) >= -1e-10)

    def test_n_frontier_points(self) -> None:
        n = 15
        result, _, _ = _fit()
        cfg = MVOConfig(n_frontier_points=n)
        ef = efficient_frontier(result.global_weights, config=cfg)
        assert len(ef.volatilities) == n
        assert len(ef.returns) == n
        assert ef.weights.shape[0] == n

    def test_weights_sum_to_one(self) -> None:
        result, _, _ = _fit()
        ef = efficient_frontier(result.global_weights)
        sums = ef.weights.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(len(sums)), atol=1e-8)

    def test_weights_shape(self) -> None:
        d = 4
        R, post = _returns(d=d)
        result = regime_mvo(R, post)
        ef = efficient_frontier(result.global_weights)
        assert ef.weights.shape[1] == d

    def test_frontier_weights_non_negative(self) -> None:
        result, _, _ = _fit()
        ef = efficient_frontier(result.global_weights)
        assert np.all(ef.weights >= -1e-8)


# ---------------------------------------------------------------------------
# black_litterman_posterior
# ---------------------------------------------------------------------------


class TestBlackLittermanPosterior:
    def test_returns_ndarray(self) -> None:
        d = 3
        mu = np.zeros(d)
        cov = np.eye(d) * 0.01
        views = np.array([0.02, 0.01, -0.01])
        conf = np.array([0.8, 0.5, 0.3])
        result = black_litterman_posterior(mu, cov, views, conf)
        assert isinstance(result, np.ndarray)

    def test_shape(self) -> None:
        d = 4
        mu = np.zeros(d)
        cov = np.eye(d)
        views = np.ones(d) * 0.01
        conf = np.ones(d) * 0.5
        result = black_litterman_posterior(mu, cov, views, conf)
        assert result.shape == (d,)

    def test_high_confidence_closer_to_views(self) -> None:
        """With very high confidence, posterior should approach views."""
        d = 2
        mu = np.zeros(d)
        cov = np.eye(d) * 0.01
        views = np.array([0.05, -0.05])
        lo_conf = np.array([0.01, 0.01])
        hi_conf = np.array([100.0, 100.0])
        lo = black_litterman_posterior(mu, cov, views, lo_conf)
        hi = black_litterman_posterior(mu, cov, views, hi_conf)
        # High confidence posterior should be closer to views
        assert np.linalg.norm(hi - views) < np.linalg.norm(lo - views)

    def test_low_confidence_closer_to_equilibrium(self) -> None:
        d = 2
        mu_eq = np.array([0.02, 0.01])
        cov = np.eye(d) * 0.01
        views = np.array([0.10, 0.10])
        lo_conf = np.array([0.001, 0.001])
        hi_conf = np.array([10.0, 10.0])
        lo = black_litterman_posterior(mu_eq, cov, views, lo_conf)
        hi = black_litterman_posterior(mu_eq, cov, views, hi_conf)
        assert np.linalg.norm(lo - mu_eq) < np.linalg.norm(hi - mu_eq)

    def test_with_posterior_regime_weight(self) -> None:
        d = 3
        mu = np.zeros(d)
        cov = np.eye(d) * 0.01
        views = np.ones(d) * 0.03
        conf = np.ones(d) * 0.5
        pt = np.array([0.9, 0.1])
        result = black_litterman_posterior(mu, cov, views, conf, posteriors=pt)
        assert result.shape == (d,)
        assert np.all(np.isfinite(result))

    def test_zero_views_returns_toward_equilibrium(self) -> None:
        d = 3
        mu_eq = np.array([0.02, 0.01, -0.01])
        cov = np.eye(d) * 0.01
        views = np.zeros(d)
        conf = np.ones(d) * 0.5
        result = black_litterman_posterior(mu_eq, cov, views, conf)
        # Result should be between mu_eq and 0
        assert np.all(np.abs(result) <= np.abs(mu_eq) + 1e-10)

    def test_finite_output(self) -> None:
        d = 5
        rng = np.random.default_rng(0)
        mu = rng.standard_normal(d) * 0.01
        A = rng.standard_normal((d, d))
        cov = A @ A.T + np.eye(d) * 0.01
        views = rng.standard_normal(d) * 0.02
        conf = rng.uniform(0.1, 0.9, d)
        result = black_litterman_posterior(mu, cov, views, conf)
        assert np.all(np.isfinite(result))
