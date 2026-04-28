"""Tests for rde.analysis.drawdown_control (Phase 20)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.analysis.drawdown_control import (
    DrawdownControlConfig,
    RegimePositionLimitConfig,
    apply_drawdown_control,
    apply_regime_position_limits,
    drawdown_scale_factor,
    regime_drawdown_budget,
    regime_position_limits,
    running_drawdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _equity(T: int = 300, drift: float = 0.0, crash_at: int | None = None) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
    r = np.full(T, drift)
    if crash_at is not None:
        r[crash_at] = -0.15  # 15% crash in one bar
    eq = 10000 * np.cumprod(1.0 + r)
    return pd.Series(eq, index=idx)


def _weights(T: int = 200, N: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
    w = rng.uniform(-0.5, 0.5, (T, N))
    return pd.DataFrame(w, index=idx, columns=[f"A{i}" for i in range(N)])


def _returns(T: int = 200, N: int = 2, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
    return pd.DataFrame(rng.normal(0, 0.01, (T, N)), index=idx, columns=[f"A{i}" for i in range(N)])


def _posteriors(T: int = 200, K: int = 2, seed: int = 2) -> np.ndarray:
    return np.random.default_rng(seed).dirichlet(np.ones(K), size=T)


# ---------------------------------------------------------------------------
# running_drawdown
# ---------------------------------------------------------------------------


class TestRunningDrawdown:
    def test_monotone_rising_zero_drawdown(self) -> None:
        eq = _equity(T=100, drift=0.001)
        dd = running_drawdown(eq)
        np.testing.assert_allclose(dd.values, 0.0, atol=1e-9)

    def test_drawdown_non_positive(self) -> None:
        eq = _equity(T=100, drift=0.0, crash_at=50)
        dd = running_drawdown(eq)
        assert (dd.values <= 1e-12).all()

    def test_crash_gives_negative_drawdown(self) -> None:
        eq = _equity(T=100, crash_at=50)
        dd = running_drawdown(eq)
        assert dd.values[51] < -0.1

    def test_accepts_numpy_array(self) -> None:
        eq = np.cumprod(1 + np.full(100, 0.001))
        dd = running_drawdown(eq)
        assert len(dd) == 100

    def test_index_preserved(self) -> None:
        eq = _equity()
        dd = running_drawdown(eq)
        pd.testing.assert_index_equal(dd.index, eq.index)

    def test_full_recovery_back_to_zero(self) -> None:
        """After crash and full recovery, drawdown returns to 0."""
        T = 200
        r = np.zeros(T)
        r[50] = -0.10  # crash
        # Need (1-0.10) * (1+rate)^n >= 1 → rate >= (1/0.9)^(1/n) - 1
        # With n=100 bars and rate=0.0015: 0.9 * 1.0015^100 = 0.9 * 1.162 = 1.046 > 1 ✓
        r[51:151] = 0.0015
        eq = pd.Series(10000 * np.cumprod(1 + r))
        dd = running_drawdown(eq)
        # After sufficient recovery, drawdown should be back near 0
        assert dd.values[-1] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# drawdown_scale_factor
# ---------------------------------------------------------------------------


class TestDrawdownScaleFactor:
    def test_no_drawdown_scale_one(self) -> None:
        dd = pd.Series(np.zeros(100))
        scale = drawdown_scale_factor(dd)
        np.testing.assert_allclose(scale.values, 1.0, atol=1e-9)

    def test_hard_stop_gives_zero(self) -> None:
        cfg = DrawdownControlConfig(max_drawdown=0.10, soft_limit=0.05, recovery_threshold=0.02)
        dd = pd.Series([-0.15] * 100)  # 15% drawdown throughout
        scale = drawdown_scale_factor(dd, cfg)
        # First bar: -0.15 >= max 0.10 → stop
        assert scale.values[0] == pytest.approx(0.0, abs=1e-9)

    def test_scale_in_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        dd = pd.Series(-np.abs(rng.uniform(0, 0.15, 200)))
        scale = drawdown_scale_factor(dd)
        assert (scale.values >= -1e-9).all()
        assert (scale.values <= 1.0 + 1e-9).all()

    def test_ramp_between_limits(self) -> None:
        cfg = DrawdownControlConfig(max_drawdown=0.10, soft_limit=0.05, recovery_threshold=0.02)
        dd = pd.Series([-0.075])  # midpoint of soft and hard
        scale = drawdown_scale_factor(dd, cfg)
        assert 0.0 < scale.values[0] < 1.0

    def test_recovery_needed_before_restart(self) -> None:
        """After stop, scale stays 0 until recovery_threshold met."""
        cfg = DrawdownControlConfig(max_drawdown=0.10, soft_limit=0.05, recovery_threshold=0.02)
        # Stop at t=0, then partial recovery at t=1 (still above recovery_threshold)
        dd = pd.Series([-0.12, -0.08, -0.01])
        scale = drawdown_scale_factor(dd, cfg)
        # t=0: stop (dd=0.12 > max=0.10)
        assert scale.values[0] == pytest.approx(0.0)
        # t=1: still stopped (0.08 > recovery 0.02)
        assert scale.values[1] == pytest.approx(0.0)
        # t=2: recovered (0.01 <= 0.02) → scale > 0
        assert scale.values[2] > 0.0


# ---------------------------------------------------------------------------
# apply_drawdown_control
# ---------------------------------------------------------------------------


class TestApplyDrawdownControl:
    def test_returns_dataframe(self) -> None:
        w = _weights()
        r = _returns()
        result = apply_drawdown_control(w, r)
        assert isinstance(result, pd.DataFrame)

    def test_same_shape(self) -> None:
        w = _weights()
        r = _returns()
        result = apply_drawdown_control(w, r)
        assert result.shape == w.shape

    def test_zero_weights_unchanged(self) -> None:
        T, N = 200, 2
        idx = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
        zero_w = pd.DataFrame(np.zeros((T, N)), index=idx, columns=["A0", "A1"])
        r = _returns(T=T)
        result = apply_drawdown_control(zero_w, r)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-10)

    def test_rising_market_weights_unchanged(self) -> None:
        """If portfolio never draws down, scale = 1 → weights unchanged."""
        T = 100
        idx = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
        # All positive returns so no drawdown
        w = pd.DataFrame(np.ones((T, 1)) * 0.5, index=idx, columns=["A0"])
        r = pd.DataFrame(np.full((T, 1), 0.001), index=idx, columns=["A0"])
        cfg = DrawdownControlConfig(max_drawdown=0.10)
        result = apply_drawdown_control(w, r, config=cfg)
        np.testing.assert_allclose(result.values, w.values, atol=1e-9)

    def test_scaled_weights_smaller_magnitude(self) -> None:
        """After a large crash, scaled weights should be smaller than originals."""
        T = 200
        rng = np.random.default_rng(7)
        idx = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
        w = pd.DataFrame(rng.uniform(0.1, 0.5, (T, 2)), index=idx, columns=["A0", "A1"])
        r_vals = rng.normal(0, 0.01, (T, 2))
        r_vals[50, :] = -0.08  # big loss at t=50
        r = pd.DataFrame(r_vals, index=idx, columns=["A0", "A1"])
        cfg = DrawdownControlConfig(max_drawdown=0.05, soft_limit=0.02)
        result = apply_drawdown_control(w, r, config=cfg)
        # After stop, total weight magnitude should be less than before
        assert np.abs(result.values[60:]).sum() <= np.abs(w.values[60:]).sum() + 1e-9


# ---------------------------------------------------------------------------
# regime_position_limits
# ---------------------------------------------------------------------------


class TestRegimePositionLimits:
    def test_equal_vols_equal_limits(self) -> None:
        vols = np.array([0.10, 0.10, 0.10])
        limits = regime_position_limits(vols)
        np.testing.assert_allclose(limits, limits[0], atol=1e-9)

    def test_higher_vol_smaller_limit(self) -> None:
        vols = np.array([0.05, 0.10, 0.20])
        limits = regime_position_limits(vols)
        assert limits[0] >= limits[1] >= limits[2] - 1e-9

    def test_limits_non_negative(self) -> None:
        vols = np.array([0.01, 0.05, 0.15])
        limits = regime_position_limits(vols)
        assert (limits >= 0).all()

    def test_base_limit_applied(self) -> None:
        """Lowest-vol regime gets base_limit."""
        vols = np.array([0.05, 0.10])
        cfg = RegimePositionLimitConfig(base_limit=2.0)
        limits = regime_position_limits(vols, cfg)
        assert limits[0] == pytest.approx(2.0, abs=1e-9)

    def test_min_limit_floor(self) -> None:
        vols = np.array([0.01, 100.0])
        cfg = RegimePositionLimitConfig(base_limit=1.0, min_limit=0.1)
        limits = regime_position_limits(vols, cfg)
        assert limits[1] >= 0.1


# ---------------------------------------------------------------------------
# apply_regime_position_limits
# ---------------------------------------------------------------------------


class TestApplyRegimePositionLimits:
    def test_returns_dataframe(self) -> None:
        w = _weights()
        P = _posteriors()
        vols = np.array([0.05, 0.15])
        result = apply_regime_position_limits(w, P, vols)
        assert isinstance(result, pd.DataFrame)

    def test_same_shape(self) -> None:
        w = _weights()
        P = _posteriors()
        vols = np.array([0.05, 0.15])
        result = apply_regime_position_limits(w, P, vols)
        assert result.shape == w.shape

    def test_magnitude_le_original(self) -> None:
        """Clipping can only reduce magnitude, never increase."""
        w = _weights()
        P = _posteriors()
        vols = np.array([0.05, 0.15])
        result = apply_regime_position_limits(w, P, vols)
        assert (np.abs(result.values) <= np.abs(w.values) + 1e-9).all()

    def test_tiny_base_limit_zeros_weights(self) -> None:
        w = _weights()
        P = _posteriors()
        vols = np.array([0.10, 0.10])
        cfg = RegimePositionLimitConfig(base_limit=1e-8)
        result = apply_regime_position_limits(w, P, vols, config=cfg)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# regime_drawdown_budget
# ---------------------------------------------------------------------------


class TestRegimeDrawdownBudget:
    def test_sums_to_total_budget(self) -> None:
        sharpes = np.array([0.5, 1.0, 2.0])
        budget = regime_drawdown_budget(sharpes, total_budget=0.10)
        assert budget.sum() == pytest.approx(0.10, abs=1e-9)

    def test_negative_sharpe_gets_zero(self) -> None:
        sharpes = np.array([-1.0, 1.0, 2.0])
        budget = regime_drawdown_budget(sharpes, total_budget=0.10)
        assert budget[0] == pytest.approx(0.0, abs=1e-9)

    def test_proportional_to_sharpe(self) -> None:
        sharpes = np.array([1.0, 2.0])
        budget = regime_drawdown_budget(sharpes, total_budget=0.10)
        assert budget[1] == pytest.approx(2 * budget[0], abs=1e-9)

    def test_all_negative_equal_allocation(self) -> None:
        sharpes = np.array([-1.0, -2.0])
        budget = regime_drawdown_budget(sharpes, total_budget=0.10)
        np.testing.assert_allclose(budget, 0.05, atol=1e-9)

    def test_non_negative_budgets(self) -> None:
        sharpes = np.array([1.0, -0.5, 2.0, 0.0])
        budget = regime_drawdown_budget(sharpes, total_budget=0.15)
        assert (budget >= -1e-12).all()
