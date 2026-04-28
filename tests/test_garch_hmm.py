"""Tests for rde.models.garch_hmm.

Coverage:
    - garch_conditional_vol: shape, positivity, constant-variance special case,
      shock persistence, initial value.
    - fit_regime_garch: return types, shapes, parameter constraints,
      mu accuracy, edge cases (zero posteriors, single state).
    - forecast_garch_vol: shape, convergence to unconditional vol, monotone decay.
    - Integration: 2-state fit on synthetic data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from rde.models.garch_hmm import (
    GARCHFit,
    GARCHParams,
    fit_regime_garch,
    forecast_garch_vol,
    garch_conditional_vol,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)
T = 500
K = 2


def _make_returns(seed: int = 42, size: int = T) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=size)


def _make_uniform_posteriors(T: int, K: int) -> np.ndarray:
    """Uniform responsibility weights across K states."""
    return np.full((T, K), 1.0 / K)


def _make_params(
    state: int = 0,
    omega: float = 1e-5,
    alpha: float = 0.1,
    beta: float = 0.85,
    mu: float = 0.0,
    bars_per_year: float = 8760.0,
) -> GARCHParams:
    persistence = alpha + beta
    if persistence < 1.0:
        uncond_vol = float(np.sqrt(omega / (1.0 - persistence)) * np.sqrt(bars_per_year))
    else:
        uncond_vol = float("nan")
    return GARCHParams(
        state=state,
        omega=omega,
        alpha=alpha,
        beta=beta,
        mu=mu,
        persistence=persistence,
        unconditional_vol=uncond_vol,
    )


# ---------------------------------------------------------------------------
# 1. garch_conditional_vol
# ---------------------------------------------------------------------------


class TestGarchConditionalVol:

    def test_output_shape(self) -> None:
        returns = _make_returns()
        params = _make_params()
        vol = garch_conditional_vol(returns, params)
        assert vol.shape == (T,), "Shape must equal (T,)"

    def test_all_positive(self) -> None:
        returns = _make_returns()
        params = _make_params()
        vol = garch_conditional_vol(returns, params)
        assert np.all(vol > 0), "All conditional vols must be strictly positive"

    def test_constant_vol_when_alpha_beta_zero(self) -> None:
        """When α=β=0 σ²_t = ω for all t ≥ 1 (t=0 is the unconditional init)."""
        returns = _make_returns()
        omega = 4e-4
        params = _make_params(omega=omega, alpha=0.0, beta=0.0)
        vol = garch_conditional_vol(returns, params)
        # All bars from t=1 onward must equal sqrt(omega)
        expected = np.sqrt(omega)
        np.testing.assert_allclose(vol[1:], expected, rtol=1e-10)

    def test_initial_value_is_sample_variance(self) -> None:
        """σ²_0 should be the unconditional (sample) variance of returns."""
        returns = _make_returns()
        params = _make_params(alpha=0.0, beta=0.0)
        vol = garch_conditional_vol(returns, params)
        expected_sigma0 = np.sqrt(max(np.var(returns), 1e-8))
        assert abs(vol[0] - expected_sigma0) < 1e-12

    def test_shock_persistence_slow_mean_reversion(self) -> None:
        """After a large shock, vol should remain elevated for many bars."""
        size = 1000
        returns = np.zeros(size)
        returns[10] = 0.1   # single spike; rest are zero
        params = _make_params(omega=1e-6, alpha=0.1, beta=0.85, mu=0.0)
        vol = garch_conditional_vol(returns, params)
        # Volatility at bar 11 (right after shock) should be >> bar 10
        assert vol[11] > vol[10] * 2, "Large shock should raise vol immediately"
        # And vol should still be elevated at bar 50 relative to bar 10
        assert vol[50] > vol[10], "Persistence should keep vol elevated"

    def test_high_persistence_slower_decay_than_low(self) -> None:
        """Higher α+β → slower decay after a shock at short horizons.

        We equalise unconditional variance between the two models (same omega)
        so the only difference is persistence (0.99 vs 0.65).  We check 15
        bars after the shock, where the high-persistence model is still
        significantly elevated while the low-persistence model has mostly
        mean-reverted.
        """
        returns = np.zeros(200)
        returns[10] = 0.05
        # Same omega → same unconditional variance; differs only in persistence.
        params_high = _make_params(omega=1e-6, alpha=0.15, beta=0.84, mu=0.0)  # 0.99
        params_low  = _make_params(omega=1e-6, alpha=0.15, beta=0.50, mu=0.0)  # 0.65
        vol_high = garch_conditional_vol(returns, params_high)
        vol_low  = garch_conditional_vol(returns, params_low)
        # 15 bars after the shock: high-persistence still elevated vs low-persistence
        assert vol_high[25] > vol_low[25], (
            f"High persistence should keep vol elevated at t=25: "
            f"vol_high={vol_high[25]:.6f}  vol_low={vol_low[25]:.6f}"
        )

    def test_no_nan_in_output(self) -> None:
        returns = _make_returns(seed=7, size=300)
        params = _make_params(omega=1e-6, alpha=0.05, beta=0.9)
        vol = garch_conditional_vol(returns, params)
        assert not np.any(np.isnan(vol))


# ---------------------------------------------------------------------------
# 2. fit_regime_garch
# ---------------------------------------------------------------------------


class TestFitRegimeGarch:

    def test_returns_garchfit_type(self) -> None:
        returns = _make_returns()
        posteriors = _make_uniform_posteriors(T, K)
        result = fit_regime_garch(returns, posteriors)
        assert isinstance(result, GARCHFit)

    def test_correct_number_of_params(self) -> None:
        returns = _make_returns()
        for k in (1, 2, 3):
            posteriors = _make_uniform_posteriors(T, k)
            result = fit_regime_garch(returns, posteriors)
            assert len(result.params) == k

    def test_persistence_in_valid_range(self) -> None:
        returns = _make_returns()
        posteriors = _make_uniform_posteriors(T, K)
        result = fit_regime_garch(returns, posteriors)
        for gp in result.params:
            assert 0.0 <= gp.persistence < 1.0, (
                f"State {gp.state}: persistence={gp.persistence} outside [0,1)"
            )

    def test_omega_positive(self) -> None:
        returns = _make_returns()
        posteriors = _make_uniform_posteriors(T, K)
        result = fit_regime_garch(returns, posteriors)
        for gp in result.params:
            assert gp.omega > 0.0, f"State {gp.state}: omega={gp.omega} not positive"

    def test_alpha_beta_non_negative(self) -> None:
        returns = _make_returns()
        posteriors = _make_uniform_posteriors(T, K)
        result = fit_regime_garch(returns, posteriors)
        for gp in result.params:
            assert gp.alpha >= 0.0
            assert gp.beta >= 0.0

    def test_weighted_cond_vol_shape_matches_T(self) -> None:
        returns = _make_returns()
        posteriors = _make_uniform_posteriors(T, K)
        result = fit_regime_garch(returns, posteriors)
        assert len(result.weighted_cond_vol) == T

    def test_cond_vol_shape_T_K(self) -> None:
        returns = _make_returns()
        posteriors = _make_uniform_posteriors(T, K)
        result = fit_regime_garch(returns, posteriors)
        assert result.cond_vol.shape == (T, K)

    def test_cond_vol_all_positive(self) -> None:
        returns = _make_returns()
        posteriors = _make_uniform_posteriors(T, K)
        result = fit_regime_garch(returns, posteriors)
        assert np.all(result.cond_vol.values > 0)

    def test_weighted_cond_vol_all_positive(self) -> None:
        returns = _make_returns()
        posteriors = _make_uniform_posteriors(T, K)
        result = fit_regime_garch(returns, posteriors)
        assert np.all(result.weighted_cond_vol.values > 0)

    def test_mu_near_weighted_mean_for_dominant_state(self) -> None:
        """μ_k should approximate the responsibility-weighted mean return."""
        rng = np.random.default_rng(1)
        n = 800
        returns = rng.normal(0.005, 0.01, size=n)
        # State 0 dominates (weight ~0.9)
        posteriors = np.zeros((n, 2))
        posteriors[:, 0] = 0.9
        posteriors[:, 1] = 0.1
        result = fit_regime_garch(returns, posteriors)
        expected_mu0 = float(np.dot(posteriors[:, 0], returns) / posteriors[:, 0].sum())
        assert abs(result.params[0].mu - expected_mu0) < 1e-10

    def test_all_zero_posteriors_state_no_crash(self) -> None:
        """A state with zero weight should fall back gracefully, not crash."""
        returns = _make_returns()
        posteriors = np.zeros((T, 2))
        posteriors[:, 0] = 1.0   # state 1 has zero weight
        result = fit_regime_garch(returns, posteriors)
        assert len(result.params) == 2
        # The zero-weight state should have alpha=beta=0 (constant variance fallback)
        assert result.params[1].alpha == 0.0
        assert result.params[1].beta == 0.0

    def test_single_state_works(self) -> None:
        returns = _make_returns()
        posteriors = np.ones((T, 1))
        result = fit_regime_garch(returns, posteriors)
        assert len(result.params) == 1
        assert result.cond_vol.shape == (T, 1)

    def test_accepts_pandas_series_with_index(self) -> None:
        index = pd.date_range("2024-01-01", periods=T, freq="h")
        returns = pd.Series(_make_returns(), index=index)
        posteriors = _make_uniform_posteriors(T, K)
        result = fit_regime_garch(returns, posteriors)
        assert isinstance(result.cond_vol, pd.DataFrame)
        assert isinstance(result.weighted_cond_vol, pd.Series)
        assert len(result.weighted_cond_vol) == T

    def test_raises_on_length_mismatch(self) -> None:
        returns = _make_returns(size=100)
        posteriors = _make_uniform_posteriors(200, K)
        with pytest.raises(ValueError, match="length"):
            fit_regime_garch(returns, posteriors)

    def test_raises_on_nan_returns(self) -> None:
        returns = _make_returns().copy()
        returns[50] = np.nan
        posteriors = _make_uniform_posteriors(T, K)
        with pytest.raises(ValueError, match="NaN"):
            fit_regime_garch(returns, posteriors)

    def test_params_sorted_by_state_index(self) -> None:
        returns = _make_returns()
        k = 4
        posteriors = _make_uniform_posteriors(T, k)
        result = fit_regime_garch(returns, posteriors)
        indices = [gp.state for gp in result.params]
        assert indices == sorted(indices)

    def test_unconditional_vol_finite_when_stationary(self) -> None:
        returns = _make_returns()
        posteriors = _make_uniform_posteriors(T, K)
        result = fit_regime_garch(returns, posteriors)
        for gp in result.params:
            if gp.persistence < 1.0:
                assert np.isfinite(gp.unconditional_vol)
                assert gp.unconditional_vol > 0

    def test_state_indices_consecutive(self) -> None:
        returns = _make_returns()
        k = 3
        posteriors = _make_uniform_posteriors(T, k)
        result = fit_regime_garch(returns, posteriors)
        assert [gp.state for gp in result.params] == list(range(k))

    def test_weighted_cond_vol_equals_gamma_dot_sigma(self) -> None:
        """weighted_cond_vol[t] must exactly equal Σ_k γ_t(k) * σ_t,k."""
        returns = _make_returns()
        posteriors = _make_uniform_posteriors(T, K)
        result = fit_regime_garch(returns, posteriors)
        manual = (posteriors * result.cond_vol.values).sum(axis=1)
        np.testing.assert_allclose(
            result.weighted_cond_vol.values, manual, rtol=1e-12
        )


# ---------------------------------------------------------------------------
# 3. forecast_garch_vol
# ---------------------------------------------------------------------------


class TestForecastGarchVol:

    def _stationary_params(self) -> GARCHParams:
        return _make_params(omega=1e-5, alpha=0.05, beta=0.90, mu=0.0)

    def test_shape_matches_horizons(self) -> None:
        params = self._stationary_params()
        horizons = [1, 5, 10, 24, 168]
        result = forecast_garch_vol(params, 0.0, 0.01, horizons)
        assert len(result) == len(horizons)
        assert list(result.index) == horizons

    def test_converges_to_unconditional_vol_as_h_increases(self) -> None:
        params = self._stationary_params()
        uncond_vol = float(
            np.sqrt(params.omega / (1.0 - params.persistence))
        )
        # Use a σ_t very different from unconditional so convergence is visible.
        last_sigma = uncond_vol * 5.0
        horizons = [1, 10, 100, 1000, 5000]
        result = forecast_garch_vol(params, 0.0, last_sigma, horizons)
        # At h=5000 the forecast should be very close to unconditional vol.
        assert abs(result[5000] - uncond_vol) < uncond_vol * 0.01

    def test_monotone_decay_when_sigma_above_unconditional(self) -> None:
        params = self._stationary_params()
        uncond_vol = float(
            np.sqrt(params.omega / (1.0 - params.persistence))
        )
        last_sigma = uncond_vol * 3.0
        horizons = list(range(1, 51))
        result = forecast_garch_vol(params, 0.0, last_sigma, horizons)
        # Each forecast should be ≤ the previous one (monotone decay toward uncond).
        diffs = np.diff(result.values)
        assert np.all(diffs <= 1e-15), "Forecast should monotonically decrease toward unconditional vol"

    def test_all_positive_values(self) -> None:
        params = self._stationary_params()
        horizons = [1, 2, 10, 50, 200]
        result = forecast_garch_vol(params, 0.001, 0.012, horizons)
        assert np.all(result.values > 0)

    def test_no_nan_in_stationary_forecast(self) -> None:
        params = self._stationary_params()
        horizons = [1, 10, 100]
        result = forecast_garch_vol(params, 0.0, 0.01, horizons)
        assert not np.any(np.isnan(result.values))

    def test_constant_forecast_when_alpha_beta_zero(self) -> None:
        """When α=β=0 every horizon forecast equals sqrt(ω)."""
        params = _make_params(omega=4e-4, alpha=0.0, beta=0.0)
        horizons = [1, 10, 100, 1000]
        result = forecast_garch_vol(params, 0.0, 0.01, horizons)
        expected = np.sqrt(params.omega)
        np.testing.assert_allclose(result.values, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. Integration: real 2-state fitted model on synthetic data
# ---------------------------------------------------------------------------


class TestIntegration:

    def test_fit_on_synthetic_2state_hmm(self) -> None:
        """Train a GaussianHMM, decode posteriors, then fit GARCH — should not raise."""
        rng = np.random.default_rng(99)
        n = 600
        # Simple 2-regime synthetic returns
        state_seq = rng.choice([0, 1], size=n, p=[0.7, 0.3])
        returns = np.where(
            state_seq == 0,
            rng.normal(0.001, 0.005, size=n),
            rng.normal(-0.002, 0.015, size=n),
        )

        # Fit a GaussianHMM and get posteriors
        X = returns.reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm = GaussianHMM(
                n_components=2,
                covariance_type="full",
                n_iter=200,
                random_state=0,
            )
            hmm.fit(X)
            posteriors = hmm.predict_proba(X)   # (T, 2)

        # This must not raise.
        result = fit_regime_garch(returns, posteriors)

        assert isinstance(result, GARCHFit)
        assert len(result.params) == 2
        assert result.cond_vol.shape == (n, 2)
        assert len(result.weighted_cond_vol) == n
        for gp in result.params:
            assert gp.omega > 0
            assert 0.0 <= gp.persistence < 1.0

    def test_garch_vol_matches_manual_recursion_after_fit(self) -> None:
        """garch_conditional_vol output must match a hand-coded recursion."""
        returns = _make_returns(seed=5, size=200)
        posteriors = _make_uniform_posteriors(200, 2)
        result = fit_regime_garch(returns, posteriors)
        gp0 = result.params[0]

        # Manually compute σ_t for state 0
        T_local = 200
        sigma2_manual = np.empty(T_local)
        sigma2_manual[0] = max(float(np.var(returns)), 1e-8)
        for t in range(1, T_local):
            eps2 = (returns[t - 1] - gp0.mu) ** 2
            sigma2_manual[t] = max(
                gp0.omega + gp0.alpha * eps2 + gp0.beta * sigma2_manual[t - 1],
                1e-12,
            )
        expected = np.sqrt(sigma2_manual)
        actual = result.cond_vol[0].values
        np.testing.assert_allclose(actual, expected, rtol=1e-10)
