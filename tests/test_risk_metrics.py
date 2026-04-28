"""Tests for rde.analysis.risk_metrics (Phase 16)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.analysis.risk_metrics import (
    RegimeRiskConfig,
    RegimeRiskResult,
    _cvar,
    _max_drawdown,
    _sortino,
    _var,
    _weighted_quantile,
    compute_regime_risk,
    compute_regime_risk_weighted,
    regime_risk_divergence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _returns(T: int = 500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
    return pd.Series(rng.normal(0.0, 0.01, size=T), index=idx)


def _states(T: int = 500, K: int = 2, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, K, size=T)


def _posteriors(T: int = 500, K: int = 2, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(K), size=T)
    return raw


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


class TestVar:
    def test_all_positive_returns_zero_var(self) -> None:
        r = np.array([0.01, 0.02, 0.03, 0.04])
        # No loss → VaR at 95% should be ≤ 0 (negative loss)
        v = _var(r, 0.95)
        assert v <= 0.0

    def test_empty_returns_nan(self) -> None:
        assert np.isnan(_var(np.array([]), 0.95))

    def test_var_larger_at_higher_confidence(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.normal(0, 0.01, 1000)
        assert _var(r, 0.99) >= _var(r, 0.95)

    def test_var_is_positive_for_lossy_series(self) -> None:
        r = np.full(100, -0.01)
        assert _var(r, 0.95) > 0.0


class TestCVar:
    def test_cvar_ge_var(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.normal(0, 0.01, 1000)
        assert _cvar(r, 0.95) >= _var(r, 0.95) - 1e-12

    def test_empty_returns_nan(self) -> None:
        assert np.isnan(_cvar(np.array([]), 0.95))

    def test_cvar_positive_for_pure_losses(self) -> None:
        r = np.linspace(-0.05, -0.01, 100)
        assert _cvar(r, 0.95) > 0.0


class TestMaxDrawdown:
    def test_monotone_rising_zero_drawdown(self) -> None:
        r = np.full(100, 0.001)
        assert _max_drawdown(r) == pytest.approx(0.0, abs=1e-10)

    def test_single_crash_then_recovery(self) -> None:
        r = np.array([0.0, -0.10, 0.0, 0.0, 0.0, 0.11])
        mdd = _max_drawdown(r)
        assert 0.09 < mdd < 0.11

    def test_empty_returns_nan(self) -> None:
        assert np.isnan(_max_drawdown(np.array([])))

    def test_mdd_between_zero_and_one(self) -> None:
        rng = np.random.default_rng(42)
        r = rng.normal(0, 0.01, 500)
        mdd = _max_drawdown(r)
        assert 0.0 <= mdd <= 1.0


class TestSortino:
    def test_zero_downside_returns_inf_or_nan(self) -> None:
        r = np.full(100, 0.001)
        s = _sortino(r, 0.0, 8760.0)
        assert np.isinf(s) or np.isnan(s)

    def test_larger_downside_smaller_sortino(self) -> None:
        rng = np.random.default_rng(7)
        r_lo = rng.normal(0.001, 0.005, 500)
        r_hi = rng.normal(0.001, 0.015, 500)
        # hi has larger vol → smaller Sortino (generally)
        s_lo = _sortino(r_lo, 0.0, 8760.0)
        s_hi = _sortino(r_hi, 0.0, 8760.0)
        assert s_lo >= s_hi - 1e-6


class TestWeightedQuantile:
    def test_uniform_weights_match_numpy(self) -> None:
        rng = np.random.default_rng(0)
        v = rng.standard_normal(200)
        w = np.ones(200)
        q = 0.05
        wq = _weighted_quantile(v, w, q)
        nq = float(np.quantile(v, q))
        assert abs(wq - nq) < 0.1  # close but not exact (different methods)

    def test_zero_weights_returns_nan(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        w = np.zeros(3)
        assert np.isnan(_weighted_quantile(v, w, 0.5))

    def test_concentrated_weight_at_low_end(self) -> None:
        v = np.array([-0.05, -0.04, 0.00, 0.01, 0.02])
        w = np.array([10.0, 10.0, 0.01, 0.01, 0.01])  # weight on losses
        q05 = _weighted_quantile(v, w, 0.05)
        q50 = _weighted_quantile(v, w, 0.50)
        assert q05 <= q50


# ---------------------------------------------------------------------------
# compute_regime_risk (hard states)
# ---------------------------------------------------------------------------


class TestComputeRegimeRisk:
    def test_returns_regime_risk_result(self) -> None:
        r = _returns()
        s = _states()
        result = compute_regime_risk(r, s)
        assert isinstance(result, RegimeRiskResult)

    def test_metrics_has_correct_n_rows(self) -> None:
        r = _returns()
        s = _states(K=3)
        result = compute_regime_risk(r, s)
        assert len(result.metrics) == 3

    def test_metrics_columns_present(self) -> None:
        r = _returns()
        s = _states()
        result = compute_regime_risk(r, s)
        for col in ["mean_ann", "vol_ann", "sharpe", "var", "cvar", "max_drawdown"]:
            assert col in result.metrics.columns

    def test_n_obs_sums_to_T(self) -> None:
        T = 500
        r = _returns(T=T)
        s = _states(T=T)
        result = compute_regime_risk(r, s)
        assert int(result.metrics["n_obs"].sum()) == T

    def test_drawdown_series_keys_match_states(self) -> None:
        r = _returns()
        s = _states(K=3)
        result = compute_regime_risk(r, s)
        assert set(result.drawdown_series.keys()) == {0, 1, 2}

    def test_drawdown_series_non_positive(self) -> None:
        r = _returns()
        s = _states()
        result = compute_regime_risk(r, s)
        for k, dd in result.drawdown_series.items():
            assert (dd.values <= 1e-12).all(), f"State {k} has positive drawdown values"

    def test_var_positive_for_random_returns(self) -> None:
        rng = np.random.default_rng(99)
        r = pd.Series(rng.normal(0, 0.01, 500))
        s = (rng.uniform(size=500) > 0.5).astype(int)
        result = compute_regime_risk(r, s)
        for k in result.metrics.index:
            if not np.isnan(result.metrics.loc[k, "var"]):
                # VaR should be a real number (can be negative for pure-profit regime)
                assert np.isfinite(result.metrics.loc[k, "var"])

    def test_cvar_ge_var(self) -> None:
        r = _returns()
        s = _states()
        result = compute_regime_risk(r, s)
        for k in result.metrics.index:
            v = result.metrics.loc[k, "var"]
            cv = result.metrics.loc[k, "cvar"]
            if not (np.isnan(v) or np.isnan(cv)):
                assert cv >= v - 1e-9

    def test_length_mismatch_raises(self) -> None:
        r = _returns(T=100)
        s = _states(T=50)
        with pytest.raises(ValueError, match="length"):
            compute_regime_risk(r, s)

    def test_below_min_obs_gives_nan(self) -> None:
        T = 100
        r = _returns(T=T)
        s = np.zeros(T, dtype=int)
        s[:5] = 1  # state 1 has only 5 obs
        cfg = RegimeRiskConfig(min_obs=20)
        result = compute_regime_risk(r, s, config=cfg)
        assert np.isnan(result.metrics.loc[1, "sharpe"])

    def test_bull_regime_higher_mean(self) -> None:
        T = 1000
        rng = np.random.default_rng(5)
        r_bear = rng.normal(-0.001, 0.01, T // 2)
        r_bull = rng.normal(0.002, 0.01, T // 2)
        r = pd.Series(np.concatenate([r_bear, r_bull]))
        s = np.concatenate([np.zeros(T // 2, int), np.ones(T // 2, int)])
        result = compute_regime_risk(r, s)
        assert result.metrics.loc[1, "mean_ann"] > result.metrics.loc[0, "mean_ann"]

    def test_high_vol_regime_higher_var(self) -> None:
        T = 1000
        rng = np.random.default_rng(6)
        r_lo = rng.normal(0.0, 0.002, T // 2)
        r_hi = rng.normal(0.0, 0.020, T // 2)
        r = pd.Series(np.concatenate([r_lo, r_hi]))
        s = np.concatenate([np.zeros(T // 2, int), np.ones(T // 2, int)])
        result = compute_regime_risk(r, s)
        assert result.metrics.loc[1, "var"] > result.metrics.loc[0, "var"]


# ---------------------------------------------------------------------------
# compute_regime_risk_weighted (soft posteriors)
# ---------------------------------------------------------------------------


class TestComputeRegimeRiskWeighted:
    def test_returns_regime_risk_result(self) -> None:
        r = _returns()
        P = _posteriors()
        result = compute_regime_risk_weighted(r, P)
        assert isinstance(result, RegimeRiskResult)

    def test_metrics_n_rows_equals_K(self) -> None:
        K = 3
        r = _returns()
        P = _posteriors(K=K)
        result = compute_regime_risk_weighted(r, P)
        assert len(result.metrics) == K

    def test_columns_present(self) -> None:
        r = _returns()
        P = _posteriors()
        result = compute_regime_risk_weighted(r, P)
        for col in ["mean_ann", "vol_ann", "sharpe", "var", "cvar"]:
            assert col in result.metrics.columns

    def test_var_finite_for_noisy_returns(self) -> None:
        r = _returns()
        P = _posteriors()
        result = compute_regime_risk_weighted(r, P)
        for k in result.metrics.index:
            v = result.metrics.loc[k, "var"]
            if not np.isnan(v):
                assert np.isfinite(v)

    def test_cvar_ge_var(self) -> None:
        r = _returns()
        P = _posteriors()
        result = compute_regime_risk_weighted(r, P)
        for k in result.metrics.index:
            v = result.metrics.loc[k, "var"]
            cv = result.metrics.loc[k, "cvar"]
            if not (np.isnan(v) or np.isnan(cv)):
                assert cv >= v - 1e-9

    def test_length_mismatch_raises(self) -> None:
        r = _returns(T=100)
        P = _posteriors(T=50)
        with pytest.raises(ValueError, match="length"):
            compute_regime_risk_weighted(r, P)

    def test_drawdown_series_empty(self) -> None:
        r = _returns()
        P = _posteriors()
        result = compute_regime_risk_weighted(r, P)
        assert result.drawdown_series == {}

    def test_concentrated_posterior_close_to_hard(self) -> None:
        """When posteriors are nearly one-hot, weighted metrics ≈ hard metrics."""
        T = 500
        rng = np.random.default_rng(11)
        r = pd.Series(rng.normal(0, 0.01, T))
        s = rng.integers(0, 2, T)
        P = np.zeros((T, 2))
        P[np.arange(T), s] = 0.99
        P[np.arange(T), 1 - s] = 0.01

        res_hard = compute_regime_risk(r, s)
        res_soft = compute_regime_risk_weighted(r, P)

        for k in [0, 1]:
            mu_h = res_hard.metrics.loc[k, "mean_ann"]
            mu_s = res_soft.metrics.loc[k, "mean_ann"]
            if not (np.isnan(mu_h) or np.isnan(mu_s)):
                assert abs(mu_h - mu_s) < 0.5 * abs(mu_h) + 1e-6


# ---------------------------------------------------------------------------
# regime_risk_divergence
# ---------------------------------------------------------------------------


class TestRegimeRiskDivergence:
    def test_returns_dataframe(self) -> None:
        r = _returns()
        s = _states()
        result = compute_regime_risk(r, s)
        div = regime_risk_divergence(result)
        assert isinstance(div, pd.DataFrame)

    def test_columns_present(self) -> None:
        r = _returns()
        s = _states()
        result = compute_regime_risk(r, s)
        div = regime_risk_divergence(result)
        for col in ["state_a", "state_b", "metric", "abs_diff"]:
            assert col in div.columns

    def test_n_pairs_correct(self) -> None:
        K = 3
        r = _returns()
        s = _states(K=K)
        result = compute_regime_risk(r, s)
        div = regime_risk_divergence(result)
        metrics_cols = ["mean_ann", "vol_ann", "var", "cvar", "max_drawdown"]
        existing = [c for c in metrics_cols if c in result.metrics.columns]
        n_pairs = K * (K - 1) // 2
        assert len(div) == n_pairs * len(existing)

    def test_abs_diff_non_negative(self) -> None:
        r = _returns()
        s = _states()
        result = compute_regime_risk(r, s)
        div = regime_risk_divergence(result)
        valid = div["abs_diff"].dropna()
        assert (valid >= 0).all()

    def test_different_regimes_give_nonzero_divergence(self) -> None:
        T = 1000
        rng = np.random.default_rng(3)
        r_bear = rng.normal(-0.002, 0.01, T // 2)
        r_bull = rng.normal(0.002, 0.005, T // 2)
        r = pd.Series(np.concatenate([r_bear, r_bull]))
        s = np.concatenate([np.zeros(T // 2, int), np.ones(T // 2, int)])
        result = compute_regime_risk(r, s)
        div = regime_risk_divergence(result)
        mean_diff = div.loc[div["metric"] == "mean_ann", "abs_diff"].values[0]
        assert mean_diff > 0.5  # annualised, should be clearly non-zero
