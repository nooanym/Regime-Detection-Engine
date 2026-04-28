"""Tests for rde.analysis.signal_filtering (Phase 26)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.analysis.signal_filtering import (
    EMAConfig,
    HPConfig,
    KalmanSmootherConfig,
    adaptive_kalman_smoother,
    composite_signal,
    regime_ema,
    regime_hp_filter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series(T: int = 100, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.standard_normal(T) * 0.01))


def _post(T: int = 100, K: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    raw = np.abs(rng.standard_normal((T, K)))
    return raw / raw.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# regime_ema
# ---------------------------------------------------------------------------


class TestRegimeEMA:
    def test_returns_series(self) -> None:
        s = _series()
        post = _post()
        result = regime_ema(s, post)
        assert isinstance(result, pd.Series)

    def test_length(self) -> None:
        T = 80
        s = _series(T)
        post = _post(T)
        result = regime_ema(s, post)
        assert len(result) == T

    def test_no_nan(self) -> None:
        s = _series()
        post = _post()
        result = regime_ema(s, post)
        assert not result.isnull().any()

    def test_preserves_index(self) -> None:
        T = 50
        idx = pd.date_range("2024-01-01", periods=T, freq="D")
        s = pd.Series(np.ones(T), index=idx)
        post = _post(T)
        result = regime_ema(s, post)
        assert (result.index == idx).all()

    def test_constant_series_stays_constant(self) -> None:
        T = 100
        s = pd.Series(np.ones(T) * 5.0)
        post = _post(T)
        result = regime_ema(s, post)
        np.testing.assert_allclose(result.values, 5.0, atol=1e-10)

    def test_short_half_life_tracks_faster(self) -> None:
        """Short half-life EMA should have higher std (less smoothing)."""
        T = 200
        rng = np.random.default_rng(0)
        s = pd.Series(rng.standard_normal(T) * 0.01)
        post = np.full((T, 2), 0.5)
        fast = regime_ema(s, post, config=EMAConfig(half_lives=[2.0, 2.0]))
        slow = regime_ema(s, post, config=EMAConfig(half_lives=[50.0, 50.0]))
        assert fast.std() > slow.std()

    def test_accepts_numpy_input(self) -> None:
        arr = np.random.default_rng(0).standard_normal(60) * 0.01
        post = _post(60)
        result = regime_ema(arr, post)
        assert isinstance(result, pd.Series)

    def test_k_less_than_half_lives_handled(self) -> None:
        T = 50
        s = _series(T)
        K = 1  # fewer regimes than half-lives
        post = _post(T, K)
        cfg = EMAConfig(half_lives=[5.0, 20.0])
        result = regime_ema(s, post, config=cfg)
        assert len(result) == T
        assert not result.isnull().any()


# ---------------------------------------------------------------------------
# regime_hp_filter
# ---------------------------------------------------------------------------


class TestRegimeHPFilter:
    def test_returns_two_series(self) -> None:
        s = _series()
        post = _post()
        trend, cycle = regime_hp_filter(s, post)
        assert isinstance(trend, pd.Series)
        assert isinstance(cycle, pd.Series)

    def test_lengths(self) -> None:
        T = 80
        s = _series(T)
        post = _post(T)
        trend, cycle = regime_hp_filter(s, post)
        assert len(trend) == T
        assert len(cycle) == T

    def test_trend_plus_cycle_equals_series(self) -> None:
        s = _series()
        post = _post()
        trend, cycle = regime_hp_filter(s, post)
        np.testing.assert_allclose((trend + cycle).values, s.values, atol=1e-8)

    def test_no_nan(self) -> None:
        s = _series()
        post = _post()
        trend, cycle = regime_hp_filter(s, post)
        assert not trend.isnull().any()
        assert not cycle.isnull().any()

    def test_preserves_index(self) -> None:
        T = 60
        idx = pd.date_range("2024-01-01", periods=T, freq="D")
        s = pd.Series(np.ones(T), index=idx)
        post = _post(T)
        trend, cycle = regime_hp_filter(s, post)
        assert (trend.index == idx).all()

    def test_large_lambda_smoother_trend(self) -> None:
        """Larger lambda → trend has lower std (smoother)."""
        T = 200
        s = _series(T)
        post = np.full((T, 2), 0.5)
        _, cycle_lo = regime_hp_filter(s, post, config=HPConfig(lambdas=[10.0, 10.0]))
        _, cycle_hi = regime_hp_filter(s, post, config=HPConfig(lambdas=[10000.0, 10000.0]))
        # Larger lambda → smoother trend → larger cycle variation
        assert cycle_hi.std() > cycle_lo.std()

    def test_constant_series_zero_cycle(self) -> None:
        T = 50
        s = pd.Series(np.ones(T) * 3.0)
        post = _post(T)
        trend, cycle = regime_hp_filter(s, post)
        np.testing.assert_allclose(cycle.values, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# adaptive_kalman_smoother
# ---------------------------------------------------------------------------


class TestAdaptiveKalmanSmoother:
    def test_returns_series(self) -> None:
        s = _series()
        post = _post()
        result = adaptive_kalman_smoother(s, post)
        assert isinstance(result, pd.Series)

    def test_length(self) -> None:
        T = 80
        s = _series(T)
        post = _post(T)
        result = adaptive_kalman_smoother(s, post)
        assert len(result) == T

    def test_no_nan(self) -> None:
        s = _series()
        post = _post()
        result = adaptive_kalman_smoother(s, post)
        assert not result.isnull().any()

    def test_preserves_index(self) -> None:
        T = 50
        idx = pd.date_range("2024-01-01", periods=T, freq="D")
        s = pd.Series(np.random.default_rng(0).standard_normal(T), index=idx)
        post = _post(T)
        result = adaptive_kalman_smoother(s, post)
        assert (result.index == idx).all()

    def test_smoother_reduces_noise(self) -> None:
        T = 500
        rng = np.random.default_rng(0)
        truth = np.cumsum(rng.standard_normal(T) * 0.001)
        noisy = pd.Series(truth + rng.standard_normal(T) * 0.05)
        post = np.full((T, 2), 0.5)
        cfg = KalmanSmootherConfig(
            process_noise_per_regime=[1e-6, 1e-6],
            obs_noise=0.1,
        )
        smoothed = adaptive_kalman_smoother(noisy, post, config=cfg)
        # Smoothed should be closer to truth than raw noisy
        err_raw = float(np.mean((noisy.values - truth) ** 2))
        err_smooth = float(np.mean((smoothed.values - truth) ** 2))
        assert err_smooth < err_raw

    def test_high_process_noise_closer_to_raw(self) -> None:
        T = 100
        rng = np.random.default_rng(1)
        s = pd.Series(rng.standard_normal(T) * 0.01)
        post = np.full((T, 2), 0.5)
        cfg_hi = KalmanSmootherConfig(process_noise_per_regime=[10.0, 10.0], obs_noise=0.001)
        cfg_lo = KalmanSmootherConfig(process_noise_per_regime=[1e-8, 1e-8], obs_noise=0.001)
        hi = adaptive_kalman_smoother(s, post, config=cfg_hi)
        lo = adaptive_kalman_smoother(s, post, config=cfg_lo)
        # High process noise → trusts observations more → stays closer to raw
        err_hi = float(np.mean((hi.values - s.values) ** 2))
        err_lo = float(np.mean((lo.values - s.values) ** 2))
        assert err_hi < err_lo

    def test_constant_series_unchanged(self) -> None:
        T = 50
        s = pd.Series(np.ones(T) * 3.0)
        post = _post(T)
        result = adaptive_kalman_smoother(s, post)
        np.testing.assert_allclose(result.values, 3.0, atol=1e-4)

    def test_k_less_than_process_noises(self) -> None:
        T = 50
        K = 1
        s = _series(T)
        post = _post(T, K)
        cfg = KalmanSmootherConfig(process_noise_per_regime=[0.001, 0.0001])
        result = adaptive_kalman_smoother(s, post, config=cfg)
        assert len(result) == T
        assert not result.isnull().any()


# ---------------------------------------------------------------------------
# composite_signal
# ---------------------------------------------------------------------------


class TestCompositeSignal:
    def test_returns_series(self) -> None:
        T, K = 60, 2
        signals = {
            "ema": pd.Series(np.random.default_rng(0).standard_normal(T)),
            "hp": pd.Series(np.random.default_rng(1).standard_normal(T)),
        }
        post = _post(T, K)
        result = composite_signal(signals, post)
        assert isinstance(result, pd.Series)

    def test_length(self) -> None:
        T, K = 60, 2
        signals = {"a": pd.Series(np.ones(T)), "b": pd.Series(np.zeros(T))}
        post = _post(T, K)
        result = composite_signal(signals, post)
        assert len(result) == T

    def test_single_signal_equal_weights(self) -> None:
        T, K = 40, 2
        s = pd.Series(np.random.default_rng(0).standard_normal(T))
        post = _post(T, K)
        result = composite_signal({"s": s}, post)
        np.testing.assert_allclose(result.values, s.values, atol=1e-10)

    def test_equal_signals_same_as_individual(self) -> None:
        T, K = 40, 2
        vals = np.random.default_rng(0).standard_normal(T)
        signals = {"a": pd.Series(vals), "b": pd.Series(vals)}
        post = _post(T, K)
        result = composite_signal(signals, post)
        np.testing.assert_allclose(result.values, vals, atol=1e-10)

    def test_regime_weights_matrix(self) -> None:
        T, K = 40, 2
        rng = np.random.default_rng(0)
        s1 = pd.Series(rng.standard_normal(T))
        s2 = pd.Series(np.zeros(T))
        # Regime 0: all weight on s1; regime 1: all weight on s2
        rsw = np.array([[1.0, 0.0], [0.0, 1.0]])
        # Posteriors: all regime 0
        post = np.zeros((T, K))
        post[:, 0] = 1.0
        result = composite_signal({"s1": s1, "s2": s2}, post, rsw)
        np.testing.assert_allclose(result.values, s1.values, atol=1e-10)

    def test_no_nan(self) -> None:
        T, K = 60, 3
        rng = np.random.default_rng(0)
        signals = {f"s{i}": pd.Series(rng.standard_normal(T)) for i in range(3)}
        post = _post(T, K)
        result = composite_signal(signals, post)
        assert not result.isnull().any()
