"""Tests for rde.analysis.execution (Phase 22)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.analysis.execution import (
    ExecutionConfig,
    ImpactModelConfig,
    OrderSchedule,
    RegimeImpactParams,
    estimate_regime_impact,
    expected_slippage,
    optimal_order_schedule,
    slippage_attribution,
    twap_schedule,
    vwap_schedule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _params(K: int = 2) -> RegimeImpactParams:
    return RegimeImpactParams(
        n_regimes=K,
        half_spread=np.linspace(0.001, 0.003, K),
        impact_coeff=np.linspace(0.05, 0.15, K),
        avg_volume=np.linspace(1000.0, 2000.0, K),
    )


def _posteriors_uniform(K: int = 2) -> np.ndarray:
    return np.full(K, 1.0 / K)


def _features(T: int = 100, K: int = 2, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "log_hl_range": np.abs(rng.standard_normal(T)) * 0.02,
            "volume": rng.uniform(500, 2000, T),
        }
    )
    raw = np.abs(rng.standard_normal((T, K)))
    posteriors = raw / raw.sum(axis=1, keepdims=True)
    return df, posteriors


# ---------------------------------------------------------------------------
# RegimeImpactParams.blended
# ---------------------------------------------------------------------------


class TestRegimeImpactParamsBlended:
    def test_blended_uniform_is_mean(self) -> None:
        K = 3
        p = RegimeImpactParams(
            n_regimes=K,
            half_spread=np.array([0.001, 0.002, 0.003]),
            impact_coeff=np.array([0.1, 0.2, 0.3]),
            avg_volume=np.array([1000.0, 2000.0, 3000.0]),
        )
        hs, ic, av = p.blended(np.ones(K))
        assert hs == pytest.approx(0.002, rel=1e-6)
        assert ic == pytest.approx(0.2, rel=1e-6)
        assert av == pytest.approx(2000.0, rel=1e-6)

    def test_blended_one_hot_returns_regime(self) -> None:
        K = 3
        p = RegimeImpactParams(
            n_regimes=K,
            half_spread=np.array([0.001, 0.005, 0.003]),
            impact_coeff=np.array([0.1, 0.5, 0.3]),
            avg_volume=np.array([1000.0, 500.0, 2000.0]),
        )
        for k in range(K):
            hot = np.zeros(K)
            hot[k] = 1.0
            hs, ic, av = p.blended(hot)
            assert hs == pytest.approx(p.half_spread[k])
            assert ic == pytest.approx(p.impact_coeff[k])
            assert av == pytest.approx(p.avg_volume[k])

    def test_blended_returns_tuple_of_floats(self) -> None:
        p = _params()
        result = p.blended(_posteriors_uniform())
        assert len(result) == 3
        for v in result:
            assert isinstance(v, float)

    def test_blended_zero_posteriors_handled(self) -> None:
        p = _params()
        hs, ic, av = p.blended(np.zeros(2))
        assert np.isfinite(hs)
        assert np.isfinite(ic)
        assert np.isfinite(av)


# ---------------------------------------------------------------------------
# estimate_regime_impact
# ---------------------------------------------------------------------------


class TestEstimateRegimeImpact:
    def test_returns_regime_impact_params(self) -> None:
        df, post = _features()
        result = estimate_regime_impact(df, post)
        assert isinstance(result, RegimeImpactParams)

    def test_n_regimes_correct(self) -> None:
        K = 3
        df, post = _features(K=K)
        result = estimate_regime_impact(df, post)
        assert result.n_regimes == K

    def test_array_shapes(self) -> None:
        K = 4
        df, post = _features(K=K)
        result = estimate_regime_impact(df, post)
        assert result.half_spread.shape == (K,)
        assert result.impact_coeff.shape == (K,)
        assert result.avg_volume.shape == (K,)

    def test_half_spread_non_negative(self) -> None:
        df, post = _features()
        result = estimate_regime_impact(df, post)
        assert np.all(result.half_spread >= 0)

    def test_impact_coeff_non_negative(self) -> None:
        df, post = _features()
        result = estimate_regime_impact(df, post)
        assert np.all(result.impact_coeff >= 0)

    def test_avg_volume_positive(self) -> None:
        df, post = _features()
        result = estimate_regime_impact(df, post)
        assert np.all(result.avg_volume >= 1.0)

    def test_with_volume_col(self) -> None:
        df, post = _features()
        cfg = ImpactModelConfig(volume_col="volume")
        result = estimate_regime_impact(df, post, config=cfg)
        assert np.all(result.avg_volume >= 1.0)

    def test_missing_spread_proxy_returns_defaults(self) -> None:
        df = pd.DataFrame({"other_col": np.ones(50)})
        K = 2
        post = np.full((50, K), 1.0 / K)
        result = estimate_regime_impact(df, post)
        assert result.n_regimes == K
        assert np.all(result.half_spread >= 0)

    def test_missing_volume_col_avg_volume_ones(self) -> None:
        df, post = _features()
        cfg = ImpactModelConfig(volume_col="nonexistent")
        result = estimate_regime_impact(df, post, config=cfg)
        np.testing.assert_array_equal(result.avg_volume, np.ones(2))

    def test_volatile_regime_higher_impact(self) -> None:
        T = 200
        K = 2
        rng = np.random.default_rng(0)
        # Regime 0: low volatility spreads; regime 1: high volatility spreads
        spread = np.concatenate([
            np.abs(rng.standard_normal(T // 2)) * 0.001,
            np.abs(rng.standard_normal(T // 2)) * 0.02,
        ])
        df = pd.DataFrame({"log_hl_range": spread})
        post = np.zeros((T, K))
        post[:T // 2, 0] = 1.0
        post[T // 2:, 1] = 1.0
        result = estimate_regime_impact(df, post)
        # Volatile regime should have higher half-spread
        assert result.half_spread[1] > result.half_spread[0]


# ---------------------------------------------------------------------------
# expected_slippage
# ---------------------------------------------------------------------------


class TestExpectedSlippage:
    def test_returns_float(self) -> None:
        p = _params()
        s = expected_slippage(100.0, p, _posteriors_uniform())
        assert isinstance(s, float)

    def test_non_negative(self) -> None:
        p = _params()
        assert expected_slippage(100.0, p, _posteriors_uniform()) >= 0.0

    def test_zero_order_equals_half_spread(self) -> None:
        p = _params()
        pt = _posteriors_uniform()
        hs, _, _ = p.blended(pt)
        s = expected_slippage(0.0, p, pt)
        assert s == pytest.approx(hs, rel=1e-6)

    def test_larger_order_higher_slippage(self) -> None:
        p = _params()
        pt = _posteriors_uniform()
        s1 = expected_slippage(100.0, p, pt)
        s2 = expected_slippage(10000.0, p, pt)
        assert s2 > s1

    def test_sign_symmetric(self) -> None:
        p = _params()
        pt = _posteriors_uniform()
        assert expected_slippage(500.0, p, pt) == pytest.approx(
            expected_slippage(-500.0, p, pt), rel=1e-12
        )

    def test_sqrt_scaling(self) -> None:
        """Doubling order * 4 should roughly double the impact component."""
        p = RegimeImpactParams(
            n_regimes=1,
            half_spread=np.array([0.0]),
            impact_coeff=np.array([0.1]),
            avg_volume=np.array([1000.0]),
        )
        pt = np.array([1.0])
        s1 = expected_slippage(100.0, p, pt)
        s4 = expected_slippage(400.0, p, pt)
        assert s4 == pytest.approx(2 * s1, rel=1e-6)


# ---------------------------------------------------------------------------
# optimal_order_schedule
# ---------------------------------------------------------------------------


class TestOptimalOrderSchedule:
    def test_returns_order_schedule(self) -> None:
        p = _params()
        result = optimal_order_schedule(1000.0, p, _posteriors_uniform())
        assert isinstance(result, OrderSchedule)

    def test_sizes_sum_to_target(self) -> None:
        target = 500.0
        p = _params()
        result = optimal_order_schedule(target, p, _posteriors_uniform())
        assert result.sizes.sum() == pytest.approx(target, rel=1e-8)

    def test_sizes_length_equals_horizon(self) -> None:
        cfg = ExecutionConfig(horizon=15)
        p = _params()
        result = optimal_order_schedule(1000.0, p, _posteriors_uniform(), config=cfg)
        assert len(result.sizes) == 15

    def test_expected_cost_non_negative(self) -> None:
        p = _params()
        result = optimal_order_schedule(1000.0, p, _posteriors_uniform())
        assert result.expected_cost >= 0.0

    def test_completion_bar_in_range(self) -> None:
        cfg = ExecutionConfig(horizon=10)
        p = _params()
        result = optimal_order_schedule(1000.0, p, _posteriors_uniform(), config=cfg)
        assert 0 <= result.completion_bar < 10

    def test_urgency_one_front_loads(self) -> None:
        cfg = ExecutionConfig(horizon=10, urgency=1.0)
        p = _params()
        result = optimal_order_schedule(1000.0, p, _posteriors_uniform(), config=cfg)
        assert result.completion_bar == 0

    def test_urgency_one_first_slice_dominates(self) -> None:
        cfg = ExecutionConfig(horizon=10, urgency=1.0, min_child_size=0.0)
        p = _params()
        result = optimal_order_schedule(1000.0, p, _posteriors_uniform(), config=cfg)
        # Urgency clips to 1-1e-6 so first slice ≈ 999.999; rest of slices are negligible
        assert result.sizes[0] == pytest.approx(1000.0, abs=1.0)

    def test_negative_target_preserved(self) -> None:
        target = -750.0
        p = _params()
        result = optimal_order_schedule(target, p, _posteriors_uniform())
        assert result.sizes.sum() == pytest.approx(target, rel=1e-8)

    def test_high_urgency_smaller_completion_bar(self) -> None:
        p = _params()
        r_lo = optimal_order_schedule(1000.0, p, _posteriors_uniform(),
                                      config=ExecutionConfig(urgency=0.1))
        r_hi = optimal_order_schedule(1000.0, p, _posteriors_uniform(),
                                      config=ExecutionConfig(urgency=0.9))
        assert r_hi.completion_bar <= r_lo.completion_bar

    def test_zero_target_zero_cost(self) -> None:
        p = _params()
        result = optimal_order_schedule(0.0, p, _posteriors_uniform())
        assert result.expected_cost == pytest.approx(0.0, abs=1e-12)

    def test_higher_impact_higher_cost(self) -> None:
        lo = RegimeImpactParams(
            n_regimes=2,
            half_spread=np.array([0.001, 0.001]),
            impact_coeff=np.array([0.01, 0.01]),
            avg_volume=np.array([1000.0, 1000.0]),
        )
        hi = RegimeImpactParams(
            n_regimes=2,
            half_spread=np.array([0.01, 0.01]),
            impact_coeff=np.array([0.5, 0.5]),
            avg_volume=np.array([100.0, 100.0]),
        )
        pt = _posteriors_uniform()
        r_lo = optimal_order_schedule(1000.0, lo, pt)
        r_hi = optimal_order_schedule(1000.0, hi, pt)
        assert r_hi.expected_cost > r_lo.expected_cost


# ---------------------------------------------------------------------------
# twap_schedule
# ---------------------------------------------------------------------------


class TestTWAPSchedule:
    def test_returns_ndarray(self) -> None:
        result = twap_schedule(1000.0, 10)
        assert isinstance(result, np.ndarray)

    def test_length(self) -> None:
        assert len(twap_schedule(1000.0, 15)) == 15

    def test_sum_equals_target(self) -> None:
        assert twap_schedule(750.0, 10).sum() == pytest.approx(750.0)

    def test_equal_slices(self) -> None:
        result = twap_schedule(100.0, 5)
        np.testing.assert_allclose(result, np.full(5, 20.0))

    def test_negative_target(self) -> None:
        result = twap_schedule(-200.0, 4)
        assert result.sum() == pytest.approx(-200.0)
        np.testing.assert_allclose(result, np.full(4, -50.0))


# ---------------------------------------------------------------------------
# vwap_schedule
# ---------------------------------------------------------------------------


class TestVWAPSchedule:
    def test_returns_ndarray(self) -> None:
        vp = np.array([100.0, 200.0, 300.0, 400.0])
        result = vwap_schedule(1000.0, vp)
        assert isinstance(result, np.ndarray)

    def test_sum_equals_target(self) -> None:
        vp = np.array([1.0, 2.0, 3.0, 4.0])
        assert vwap_schedule(500.0, vp).sum() == pytest.approx(500.0)

    def test_proportional_to_volume(self) -> None:
        vp = np.array([1.0, 3.0])
        result = vwap_schedule(400.0, vp)
        assert result[0] == pytest.approx(100.0)
        assert result[1] == pytest.approx(300.0)

    def test_zero_volume_falls_back_to_twap(self) -> None:
        vp = np.zeros(5)
        result = vwap_schedule(100.0, vp)
        np.testing.assert_allclose(result, twap_schedule(100.0, 5))

    def test_uniform_volume_equals_twap(self) -> None:
        vp = np.ones(8)
        twap = twap_schedule(240.0, 8)
        vwap = vwap_schedule(240.0, vp)
        np.testing.assert_allclose(vwap, twap)


# ---------------------------------------------------------------------------
# slippage_attribution
# ---------------------------------------------------------------------------


class TestSlippageAttribution:
    def test_returns_dataframe(self) -> None:
        T, K = 20, 2
        sizes = pd.Series(np.ones(T) * 100.0)
        p = _params(K)
        post = np.full((T, K), 1.0 / K)
        result = slippage_attribution(sizes, p, post)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self) -> None:
        T, K = 20, 2
        sizes = pd.Series(np.ones(T) * 100.0)
        p = _params(K)
        post = np.full((T, K), 1.0 / K)
        result = slippage_attribution(sizes, p, post)
        for col in ["expected_slippage", "dominant_regime", "half_spread", "impact_component"]:
            assert col in result.columns

    def test_length_matches_input(self) -> None:
        T, K = 30, 3
        sizes = pd.Series(np.random.default_rng(0).standard_normal(T) * 50)
        p = _params(K)
        post = np.full((T, K), 1.0 / K)
        result = slippage_attribution(sizes, p, post)
        assert len(result) == T

    def test_slippage_non_negative(self) -> None:
        T, K = 20, 2
        sizes = pd.Series(np.ones(T) * 200.0)
        p = _params(K)
        post = np.full((T, K), 1.0 / K)
        result = slippage_attribution(sizes, p, post)
        assert (result["expected_slippage"] >= 0).all()

    def test_dominant_regime_valid_range(self) -> None:
        T, K = 20, 3
        sizes = pd.Series(np.ones(T) * 100.0)
        p = _params(K)
        post = np.full((T, K), 1.0 / K)
        result = slippage_attribution(sizes, p, post)
        assert result["dominant_regime"].between(0, K - 1).all()

    def test_slippage_equals_hs_plus_impact(self) -> None:
        T, K = 10, 2
        sizes = pd.Series(np.ones(T) * 500.0)
        p = _params(K)
        post = np.full((T, K), 1.0 / K)
        result = slippage_attribution(sizes, p, post)
        expected = result["half_spread"] + result["impact_component"]
        np.testing.assert_allclose(result["expected_slippage"].values, expected.values, rtol=1e-10)

    def test_preserves_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        sizes = pd.Series(np.ones(10) * 100.0, index=idx)
        p = _params()
        post = np.full((10, 2), 0.5)
        result = slippage_attribution(sizes, p, post)
        assert (result.index == idx).all()

    def test_one_hot_regime_matches_direct(self) -> None:
        T, K = 5, 2
        sizes = pd.Series(np.full(T, 300.0))
        p = _params(K)
        post = np.zeros((T, K))
        post[:, 1] = 1.0
        result = slippage_attribution(sizes, p, post)
        direct = expected_slippage(300.0, p, post[0])
        np.testing.assert_allclose(result["expected_slippage"].values,
                                   np.full(T, direct), rtol=1e-10)
