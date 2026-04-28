"""Tests for rde.models.kalman — regime-switching Kalman filter (Phase 19)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.models.kalman import (
    KalmanConfig,
    KalmanFilterResult,
    RegimeSwitchingKalmanResult,
    kalman_filter,
    regime_switching_kalman_filter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine_series(T: int = 300, noise: float = 0.01, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
    t = np.linspace(0, 4 * np.pi, T)
    return pd.Series(np.sin(t) * 0.1 + rng.normal(0, noise, T), index=idx)


def _random_walk(T: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
    return pd.Series(rng.normal(0, 0.01, T).cumsum(), index=idx)


def _posteriors(T: int = 300, K: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(K), size=T)


# ---------------------------------------------------------------------------
# kalman_filter — basic shape and property tests
# ---------------------------------------------------------------------------


class TestKalmanFilter:
    def test_returns_result(self) -> None:
        s = _sine_series()
        result = kalman_filter(s)
        assert isinstance(result, KalmanFilterResult)

    def test_filtered_state_shape_velocity(self) -> None:
        T = 200
        s = _sine_series(T=T)
        result = kalman_filter(s, config=KalmanConfig(use_velocity=True))
        assert result.filtered_state.shape == (T, 2)
        assert "trend" in result.filtered_state.columns
        assert "velocity" in result.filtered_state.columns

    def test_filtered_state_shape_no_velocity(self) -> None:
        T = 200
        s = _sine_series(T=T)
        result = kalman_filter(s, config=KalmanConfig(use_velocity=False))
        assert result.filtered_state.shape == (T, 1)
        assert "trend" in result.filtered_state.columns

    def test_index_preserved(self) -> None:
        s = _sine_series()
        result = kalman_filter(s)
        pd.testing.assert_index_equal(result.filtered_state.index, s.index)

    def test_innovations_length(self) -> None:
        T = 150
        s = _sine_series(T=T)
        result = kalman_filter(s)
        assert len(result.innovations) == T

    def test_innovation_variance_positive(self) -> None:
        s = _sine_series()
        result = kalman_filter(s)
        assert (result.innovation_variance.values > 0).all()

    def test_filtered_smooths_noisy_series(self) -> None:
        """Filtered trend should have smaller std than raw noisy series."""
        T = 500
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
        trend = np.linspace(0, 0.1, T)
        noisy = pd.Series(trend + rng.normal(0, 0.05, T), index=idx)
        result = kalman_filter(noisy, config=KalmanConfig(use_velocity=False))
        filtered_std = float(result.filtered_state["trend"].std())
        noisy_std = float(noisy.std())
        assert filtered_std < noisy_std

    def test_accepts_numpy_array(self) -> None:
        arr = np.random.default_rng(1).normal(0, 0.01, 100)
        result = kalman_filter(arr)
        assert len(result.filtered_state) == 100

    def test_nan_observation_skipped(self) -> None:
        s = _sine_series(T=100)
        s.iloc[30:35] = np.nan
        result = kalman_filter(s)
        assert len(result.filtered_state) == 100
        assert not result.filtered_state["trend"].isna().any()

    def test_trend_tracks_ramp(self) -> None:
        """For a ramp signal, filtered trend should be correlated with truth."""
        T = 300
        idx = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
        truth = np.linspace(0, 1, T)
        obs = pd.Series(truth + np.random.default_rng(5).normal(0, 0.02, T), index=idx)
        result = kalman_filter(obs, config=KalmanConfig(use_velocity=False))
        corr = float(np.corrcoef(truth, result.filtered_state["trend"].values)[0, 1])
        assert corr > 0.98

    def test_low_process_noise_slower_adaptation(self) -> None:
        """With lower process noise, the filter adapts more slowly."""
        T = 200
        s = _sine_series(T=T)
        r_fast = kalman_filter(s, config=KalmanConfig(process_noise=1e-1, use_velocity=False))
        r_slow = kalman_filter(s, config=KalmanConfig(process_noise=1e-6, use_velocity=False))
        # Slow filter should have less variance (smoother)
        assert r_slow.filtered_state["trend"].std() < r_fast.filtered_state["trend"].std()

    def test_filtered_variance_shape(self) -> None:
        T = 100
        s = _sine_series(T=T)
        result = kalman_filter(s)
        assert result.filtered_variance.shape[0] == T
        assert result.filtered_variance.shape[1] == result.filtered_variance.shape[2]


# ---------------------------------------------------------------------------
# regime_switching_kalman_filter
# ---------------------------------------------------------------------------


class TestRegimeSwitchingKalman:
    def test_returns_result(self) -> None:
        s = _sine_series()
        P = _posteriors()
        result = regime_switching_kalman_filter(s, P)
        assert isinstance(result, RegimeSwitchingKalmanResult)

    def test_blended_state_length(self) -> None:
        T = 200
        s = _sine_series(T=T)
        P = _posteriors(T=T)
        result = regime_switching_kalman_filter(s, P)
        assert len(result.blended_state) == T

    def test_per_regime_trend_shape(self) -> None:
        T, K = 200, 3
        s = _sine_series(T=T)
        P = _posteriors(T=T, K=K)
        result = regime_switching_kalman_filter(s, P)
        assert result.per_regime_trend.shape == (T, K)

    def test_trend_signal_shape(self) -> None:
        T = 200
        s = _sine_series(T=T)
        P = _posteriors(T=T)
        result = regime_switching_kalman_filter(s, P)
        assert len(result.trend_signal) == T

    def test_trend_signal_in_minus_one_one(self) -> None:
        s = _sine_series()
        P = _posteriors()
        result = regime_switching_kalman_filter(s, P)
        vals = result.trend_signal.values
        assert (vals >= -1.0 - 1e-9).all()
        assert (vals <= 1.0 + 1e-9).all()

    def test_index_preserved(self) -> None:
        s = _sine_series()
        P = _posteriors()
        result = regime_switching_kalman_filter(s, P)
        pd.testing.assert_index_equal(result.blended_state.index, s.index)

    def test_regime_vars_accepted(self) -> None:
        s = _sine_series()
        P = _posteriors()
        K = P.shape[1]
        regime_vars = np.full(K, 0.01 ** 2)
        result = regime_switching_kalman_filter(s, P, regime_vars=regime_vars)
        assert isinstance(result, RegimeSwitchingKalmanResult)

    def test_blended_is_weighted_average(self) -> None:
        """When posteriors are one-hot to regime 0, blended = regime_0 trend."""
        T = 200
        s = _sine_series(T=T)
        P = np.zeros((T, 2))
        P[:, 0] = 1.0
        result = regime_switching_kalman_filter(s, P)
        blended = result.blended_state["trend"].values
        regime_0 = result.per_regime_trend["regime_0"].values
        np.testing.assert_allclose(blended, regime_0, atol=1e-10)

    def test_no_nan_in_output(self) -> None:
        s = _sine_series()
        P = _posteriors()
        result = regime_switching_kalman_filter(s, P)
        assert not result.blended_state.isna().any().any()
        assert not result.per_regime_trend.isna().any().any()
        assert not result.trend_signal.isna().any()

    def test_two_regimes_different_trends(self) -> None:
        """With very different regime variances, per-regime trends differ."""
        T = 300
        rng = np.random.default_rng(9)
        s = pd.Series(rng.normal(0, 0.01, T))
        P = np.zeros((T, 2))
        P[:, 0] = 0.9
        P[:, 1] = 0.1
        regime_vars = np.array([1e-6, 0.1])  # very different scales
        result = regime_switching_kalman_filter(s, P, regime_vars=regime_vars)
        r0 = result.per_regime_trend["regime_0"].values
        r1 = result.per_regime_trend["regime_1"].values
        # Different noise → different filtered trends (won't be identical)
        assert not np.allclose(r0, r1, atol=1e-6)
