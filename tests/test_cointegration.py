"""Tests for rde.analysis.cointegration (Phase 24)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.analysis.cointegration import (
    RegimeCointegrationResult,
    RegimeSpreadParams,
    SpreadConfig,
    fit_regime_spreads,
    half_life_ornstein_uhlenbeck,
    regime_half_lives,
    regime_spread_zscore,
    spread_zscore,
    weighted_ols,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cointegrated(T: int = 200, beta: float = 1.5, noise: float = 0.05, seed: int = 0):
    """Create y = alpha + beta * x + stationary_noise."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(T) * 0.01)
    spread = _ou_process(T, rng, theta=0.1, mu=0.0, sigma=noise)
    y = 0.1 + beta * x + spread
    return pd.Series(y), pd.Series(x)


def _ou_process(T: int, rng, theta: float = 0.1, mu: float = 0.0, sigma: float = 0.02):
    s = np.zeros(T)
    for t in range(1, T):
        s[t] = s[t - 1] + theta * (mu - s[t - 1]) + sigma * rng.standard_normal()
    return s


def _uniform_posteriors(T: int, K: int) -> np.ndarray:
    return np.full((T, K), 1.0 / K)


def _fit(T: int = 200, K: int = 2, seed: int = 0):
    y, x = _make_cointegrated(T=T, seed=seed)
    post = _uniform_posteriors(T, K)
    return fit_regime_spreads(y, x, post), y, x


# ---------------------------------------------------------------------------
# weighted_ols
# ---------------------------------------------------------------------------


class TestWeightedOLS:
    def test_recovers_true_beta(self) -> None:
        rng = np.random.default_rng(0)
        T = 500
        x = rng.standard_normal(T)
        y = 2.0 * x + 0.5 + rng.standard_normal(T) * 0.01
        alpha, beta = weighted_ols(y, x, np.ones(T))
        assert beta == pytest.approx(2.0, abs=0.05)
        assert alpha == pytest.approx(0.5, abs=0.05)

    def test_uniform_vs_equal_weight(self) -> None:
        rng = np.random.default_rng(1)
        T = 100
        x = rng.standard_normal(T)
        y = x + rng.standard_normal(T) * 0.1
        a1, b1 = weighted_ols(y, x, np.ones(T))
        a2, b2 = weighted_ols(y, x, np.ones(T) * 5.0)
        assert a1 == pytest.approx(a2, rel=1e-6)
        assert b1 == pytest.approx(b2, rel=1e-6)

    def test_returns_floats(self) -> None:
        a, b = weighted_ols(np.ones(10), np.ones(10), np.ones(10))
        assert isinstance(a, float)
        assert isinstance(b, float)

    def test_zero_variance_x_handled(self) -> None:
        a, b = weighted_ols(np.ones(10), np.ones(10), np.ones(10))
        assert np.isfinite(a)
        assert np.isfinite(b)


# ---------------------------------------------------------------------------
# half_life_ornstein_uhlenbeck
# ---------------------------------------------------------------------------


class TestHalfLifeOU:
    def test_fast_mean_reversion_short_half_life(self) -> None:
        rng = np.random.default_rng(0)
        # OU process with strong mean reversion (theta=0.5 → half-life ≈ 1.4 bars)
        s = _ou_process(1000, rng, theta=0.5, mu=0.0, sigma=0.01)
        hl = half_life_ornstein_uhlenbeck(s)
        assert hl < 10

    def test_random_walk_large_half_life(self) -> None:
        rng = np.random.default_rng(1)
        s = np.cumsum(rng.standard_normal(500))
        hl = half_life_ornstein_uhlenbeck(s)
        # Finite-sample Dickey-Fuller regression gives slightly negative b,
        # producing a large but potentially finite estimate.
        assert hl == np.inf or hl > 50

    def test_positive_half_life(self) -> None:
        rng = np.random.default_rng(2)
        s = _ou_process(500, rng, theta=0.05)
        hl = half_life_ornstein_uhlenbeck(s)
        if np.isfinite(hl):
            assert hl > 0

    def test_short_series_returns_inf(self) -> None:
        assert half_life_ornstein_uhlenbeck(np.array([1.0, 2.0])) == np.inf

    def test_empty_series_returns_inf(self) -> None:
        assert half_life_ornstein_uhlenbeck(np.array([])) == np.inf

    def test_slow_reversion_longer_than_fast(self) -> None:
        rng = np.random.default_rng(0)
        fast = _ou_process(2000, rng, theta=0.4)
        slow = _ou_process(2000, rng, theta=0.05)
        hl_fast = half_life_ornstein_uhlenbeck(fast)
        hl_slow = half_life_ornstein_uhlenbeck(slow)
        if np.isfinite(hl_fast) and np.isfinite(hl_slow):
            assert hl_slow > hl_fast


# ---------------------------------------------------------------------------
# fit_regime_spreads
# ---------------------------------------------------------------------------


class TestFitRegimeSpreads:
    def test_returns_result(self) -> None:
        result, _, _ = _fit()
        assert isinstance(result, RegimeCointegrationResult)

    def test_n_regime_params(self) -> None:
        K = 3
        y, x = _make_cointegrated()
        post = _uniform_posteriors(200, K)
        result = fit_regime_spreads(y, x, post)
        assert len(result.regime_params) == K

    def test_regime_indices(self) -> None:
        K = 3
        y, x = _make_cointegrated()
        post = _uniform_posteriors(200, K)
        result = fit_regime_spreads(y, x, post)
        for k, rp in enumerate(result.regime_params):
            assert rp.regime == k

    def test_spread_length(self) -> None:
        T = 200
        result, y, _ = _fit(T=T)
        assert len(result.spread) == T

    def test_global_beta_close_to_true(self) -> None:
        # OLS on I(1) series is super-consistent but finite-sample noisy;
        # use generous tolerance.
        y, x = _make_cointegrated(T=500, beta=1.5, noise=0.01, seed=42)
        post = _uniform_posteriors(500, 2)
        result = fit_regime_spreads(y, x, post)
        assert result.global_beta == pytest.approx(1.5, abs=0.2)

    def test_global_alpha_close_to_true(self) -> None:
        y, x = _make_cointegrated(T=500, beta=1.5, noise=0.01, seed=42)
        post = _uniform_posteriors(500, 2)
        result = fit_regime_spreads(y, x, post)
        assert result.global_alpha == pytest.approx(0.1, abs=0.2)

    def test_spread_is_stationary_like(self) -> None:
        y, x = _make_cointegrated(T=500, noise=0.02, seed=0)
        post = _uniform_posteriors(500, 2)
        result = fit_regime_spreads(y, x, post)
        # Spread std should be much less than y std (cointegration shrinks variance)
        assert result.spread.std() < y.std()

    def test_global_half_life_finite_for_cointegrated(self) -> None:
        y, x = _make_cointegrated(T=500, noise=0.02, seed=0)
        post = _uniform_posteriors(500, 2)
        result = fit_regime_spreads(y, x, post)
        assert np.isfinite(result.global_half_life)

    def test_global_half_life_positive(self) -> None:
        result, _, _ = _fit()
        if np.isfinite(result.global_half_life):
            assert result.global_half_life > 0.0

    def test_eigenvalues_shape(self) -> None:
        K = 3
        y, x = _make_cointegrated()
        post = _uniform_posteriors(200, K)
        result = fit_regime_spreads(y, x, post)
        assert result.eigenvalues.shape == (K,)

    def test_eigenvalues_positive(self) -> None:
        result, _, _ = _fit()
        assert np.all(result.eigenvalues > 0)

    def test_spread_std_non_negative(self) -> None:
        result, _, _ = _fit()
        for rp in result.regime_params:
            assert rp.spread_std >= 0.0

    def test_eff_obs_sums_to_t(self) -> None:
        T = 200
        result, _, _ = _fit(T=T)
        total_eff = sum(rp.eff_obs for rp in result.regime_params)
        assert total_eff == pytest.approx(T, rel=1e-6)

    def test_accepts_numpy_arrays(self) -> None:
        rng = np.random.default_rng(0)
        T = 100
        x = rng.standard_normal(T)
        y = x + rng.standard_normal(T) * 0.1
        post = _uniform_posteriors(T, 2)
        result = fit_regime_spreads(y, x, post)
        assert isinstance(result, RegimeCointegrationResult)


# ---------------------------------------------------------------------------
# spread_zscore
# ---------------------------------------------------------------------------


class TestSpreadZScore:
    def test_returns_series(self) -> None:
        s = pd.Series(np.random.default_rng(0).standard_normal(100))
        z = spread_zscore(s)
        assert isinstance(z, pd.Series)

    def test_length(self) -> None:
        s = pd.Series(np.random.default_rng(0).standard_normal(100))
        z = spread_zscore(s, window=10)
        assert len(z) == 100

    def test_stationary_mean_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        s = pd.Series(_ou_process(1000, rng, theta=0.1))
        z = spread_zscore(s, window=50)
        assert abs(z.dropna().mean()) < 0.5

    def test_constant_series_zero_zscore(self) -> None:
        s = pd.Series(np.ones(100) * 5.0)
        z = spread_zscore(s, window=10)
        # All std = 0 → z = 0 (divide by 0 guarded)
        assert np.all(z.dropna().abs() < 1e-10)

    def test_accepts_numpy_array(self) -> None:
        arr = np.random.default_rng(0).standard_normal(50)
        z = spread_zscore(arr, window=10)
        assert isinstance(z, pd.Series)


# ---------------------------------------------------------------------------
# regime_spread_zscore
# ---------------------------------------------------------------------------


class TestRegimeSpreadZScore:
    def test_returns_series(self) -> None:
        result, y, x = _fit()
        z = regime_spread_zscore(result.spread, _uniform_posteriors(200, 2), result.regime_params)
        assert isinstance(z, pd.Series)

    def test_length(self) -> None:
        T = 200
        result, y, x = _fit(T=T)
        z = regime_spread_zscore(result.spread, _uniform_posteriors(T, 2), result.regime_params)
        assert len(z) == T

    def test_finite_values(self) -> None:
        result, y, x = _fit()
        z = regime_spread_zscore(result.spread, _uniform_posteriors(200, 2), result.regime_params)
        assert np.all(np.isfinite(z.values))


# ---------------------------------------------------------------------------
# regime_half_lives
# ---------------------------------------------------------------------------


class TestRegimeHalfLives:
    def test_returns_dataframe(self) -> None:
        result, _, _ = _fit()
        df = regime_half_lives(result)
        assert isinstance(df, pd.DataFrame)

    def test_columns(self) -> None:
        result, _, _ = _fit()
        df = regime_half_lives(result)
        for col in ["regime", "half_life", "eff_obs", "beta"]:
            assert col in df.columns

    def test_n_rows(self) -> None:
        K = 3
        y, x = _make_cointegrated()
        post = _uniform_posteriors(200, K)
        result = fit_regime_spreads(y, x, post)
        df = regime_half_lives(result)
        assert len(df) == K
