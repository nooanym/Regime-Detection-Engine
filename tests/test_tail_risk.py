"""Tests for rde.analysis.tail_risk (Phase 28)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.analysis.tail_risk import (
    GPDParams,
    RegimeTailResult,
    TailConfig,
    fit_gpd,
    fit_regime_tails,
    gpd_es,
    gpd_var,
    stress_test_scenarios,
    tail_risk_decomposition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _returns(T: int = 500, K: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(T) * 0.01
    raw = np.abs(rng.standard_normal((T, K)))
    post = raw / raw.sum(axis=1, keepdims=True)
    return pd.Series(r), post


def _uniform_post(T: int, K: int) -> np.ndarray:
    return np.full((T, K), 1.0 / K)


# ---------------------------------------------------------------------------
# fit_gpd
# ---------------------------------------------------------------------------


class TestFitGPD:
    def test_returns_gpd_params(self) -> None:
        rng = np.random.default_rng(0)
        exc = rng.exponential(0.02, 100)
        result = fit_gpd(exc)
        assert isinstance(result, GPDParams)

    def test_sigma_positive(self) -> None:
        rng = np.random.default_rng(0)
        exc = rng.exponential(0.02, 100)
        result = fit_gpd(exc)
        assert result.sigma > 0

    def test_returns_none_for_empty(self) -> None:
        assert fit_gpd(np.array([])) is None

    def test_returns_none_for_single_value(self) -> None:
        assert fit_gpd(np.array([0.01])) is None

    def test_constant_exceedances_returns_none_or_params(self) -> None:
        # Constant series → zero variance → method of moments fails gracefully
        result = fit_gpd(np.ones(50) * 0.01)
        # Either None or params with sigma > 0
        if result is not None:
            assert result.sigma > 0

    def test_heavy_tail_positive_xi(self) -> None:
        # Pareto exceedances should give xi > 0
        rng = np.random.default_rng(1)
        # Pareto(1, alpha=2): heavy tail → xi = 1/alpha = 0.5
        exc = rng.pareto(2.0, 500) * 0.1
        result = fit_gpd(exc)
        assert result is not None
        assert result.xi > 0

    def test_n_exceedances_matches_input(self) -> None:
        n = 80
        rng = np.random.default_rng(0)
        exc = rng.exponential(0.01, n)
        result = fit_gpd(exc)
        assert result.n_exceedances == n


# ---------------------------------------------------------------------------
# gpd_var / gpd_es
# ---------------------------------------------------------------------------


class TestGPDVarES:
    def _params(self) -> GPDParams:
        return GPDParams(xi=0.2, sigma=0.01, threshold=0.03, n_exceedances=50)

    def test_var_above_threshold(self) -> None:
        p = self._params()
        var = gpd_var(p, 0.99, 500)
        assert var >= p.threshold

    def test_es_ge_var(self) -> None:
        p = self._params()
        var = gpd_var(p, 0.99, 500)
        es = gpd_es(p, 0.99, 500)
        assert es >= var - 1e-10

    def test_higher_confidence_higher_var(self) -> None:
        p = self._params()
        v99 = gpd_var(p, 0.99, 500)
        v999 = gpd_var(p, 0.999, 500)
        assert v999 >= v99

    def test_exponential_tail_xi_zero(self) -> None:
        # For xi=0, GPD = exponential: VaR should be finite and > threshold
        p = GPDParams(xi=0.0, sigma=0.01, threshold=0.02, n_exceedances=100)
        var = gpd_var(p, 0.99, 1000)
        assert np.isfinite(var)
        assert var >= p.threshold

    def test_es_infinite_for_xi_ge_one(self) -> None:
        p = GPDParams(xi=1.0, sigma=0.01, threshold=0.02, n_exceedances=50)
        es = gpd_es(p, 0.99, 500)
        assert es == np.inf


# ---------------------------------------------------------------------------
# fit_regime_tails
# ---------------------------------------------------------------------------


class TestFitRegimeTails:
    def test_returns_result(self) -> None:
        r, post = _returns()
        result = fit_regime_tails(r, post)
        assert isinstance(result, RegimeTailResult)

    def test_n_regime_tails(self) -> None:
        K = 3
        r, post = _returns(K=K)
        result = fit_regime_tails(r, post)
        assert len(result.regime_tails) == K

    def test_regime_indices(self) -> None:
        K = 3
        r, post = _returns(K=K)
        result = fit_regime_tails(r, post)
        for k, rt in enumerate(result.regime_tails):
            assert rt.regime == k

    def test_global_tail_regime_minus_one(self) -> None:
        r, post = _returns()
        result = fit_regime_tails(r, post)
        assert result.global_tail.regime == -1

    def test_var_positive(self) -> None:
        r, post = _returns(T=600)
        result = fit_regime_tails(r, post)
        for rt in result.regime_tails + [result.global_tail]:
            assert rt.var >= 0.0

    def test_es_ge_var(self) -> None:
        r, post = _returns(T=600)
        result = fit_regime_tails(r, post)
        for rt in result.regime_tails + [result.global_tail]:
            assert rt.es >= rt.var - 1e-10

    def test_eff_obs_positive(self) -> None:
        r, post = _returns()
        result = fit_regime_tails(r, post)
        for rt in result.regime_tails:
            assert rt.eff_obs >= 0.0

    def test_volatile_regime_higher_var(self) -> None:
        T = 800
        rng = np.random.default_rng(0)
        # Regime 0: low vol; regime 1: high vol
        r = np.concatenate([
            rng.standard_normal(T // 2) * 0.002,
            rng.standard_normal(T // 2) * 0.02,
        ])
        post = np.zeros((T, 2))
        post[:T // 2, 0] = 1.0
        post[T // 2:, 1] = 1.0
        result = fit_regime_tails(pd.Series(r), post,
                                  config=TailConfig(threshold_quantile=0.85,
                                                    min_tail_obs=5))
        assert result.regime_tails[1].var > result.regime_tails[0].var

    def test_accepts_numpy_array(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.standard_normal(300) * 0.01
        post = _uniform_post(300, 2)
        result = fit_regime_tails(r, post)
        assert isinstance(result, RegimeTailResult)


# ---------------------------------------------------------------------------
# tail_risk_decomposition
# ---------------------------------------------------------------------------


class TestTailRiskDecomposition:
    def test_returns_dataframe(self) -> None:
        r, post = _returns()
        result = fit_regime_tails(r, post)
        df = tail_risk_decomposition(result, np.array([0.6, 0.4]))
        assert isinstance(df, pd.DataFrame)

    def test_columns(self) -> None:
        r, post = _returns()
        result = fit_regime_tails(r, post)
        df = tail_risk_decomposition(result, np.array([0.5, 0.5]))
        for col in ["regime", "weight", "var", "es", "weighted_var", "weighted_es"]:
            assert col in df.columns

    def test_n_rows_equals_k(self) -> None:
        K = 3
        r, post = _returns(K=K)
        result = fit_regime_tails(r, post)
        df = tail_risk_decomposition(result, np.ones(K) / K)
        assert len(df) == K

    def test_weights_sum_to_one(self) -> None:
        r, post = _returns()
        result = fit_regime_tails(r, post)
        df = tail_risk_decomposition(result, np.array([0.7, 0.3]))
        assert df["weight"].sum() == pytest.approx(1.0, abs=1e-8)

    def test_weighted_var_le_var(self) -> None:
        r, post = _returns()
        result = fit_regime_tails(r, post)
        df = tail_risk_decomposition(result, np.array([0.5, 0.5]))
        assert (df["weighted_var"] <= df["var"] + 1e-10).all()


# ---------------------------------------------------------------------------
# stress_test_scenarios
# ---------------------------------------------------------------------------


class TestStressTestScenarios:
    def test_returns_dataframe(self) -> None:
        r, post = _returns()
        result = fit_regime_tails(r, post)
        df = stress_test_scenarios(result, np.array([0.5, 0.5]))
        assert isinstance(df, pd.DataFrame)

    def test_columns(self) -> None:
        r, post = _returns()
        result = fit_regime_tails(r, post)
        df = stress_test_scenarios(result, np.array([0.5, 0.5]))
        for col in ["scenario", "loss", "regime", "weight"]:
            assert col in df.columns

    def test_losses_above_threshold(self) -> None:
        r, post = _returns()
        result = fit_regime_tails(r, post)
        df = stress_test_scenarios(result, np.array([0.5, 0.5]))
        for k, rt in enumerate(result.regime_tails):
            regime_losses = df[df["regime"] == k]["loss"]
            if len(regime_losses) > 0:
                assert (regime_losses >= rt.threshold - 1e-10).all()

    def test_n_scenarios_approx_correct(self) -> None:
        r, post = _returns()
        result = fit_regime_tails(r, post)
        cfg = TailConfig(n_stress_scenarios=500)
        df = stress_test_scenarios(result, np.array([0.5, 0.5]), config=cfg)
        assert len(df) >= 490  # small rounding tolerance

    def test_one_hot_posterior_all_one_regime(self) -> None:
        r, post = _returns()
        result = fit_regime_tails(r, post)
        df = stress_test_scenarios(result, np.array([1.0, 0.0]),
                                   config=TailConfig(n_stress_scenarios=100))
        assert (df["regime"] == 0).all()

    def test_reproducible(self) -> None:
        r, post = _returns()
        result = fit_regime_tails(r, post)
        pt = np.array([0.5, 0.5])
        cfg = TailConfig(n_stress_scenarios=50, seed=7)
        df1 = stress_test_scenarios(result, pt, config=cfg)
        df2 = stress_test_scenarios(result, pt, config=cfg)
        np.testing.assert_array_equal(df1["loss"].values, df2["loss"].values)
