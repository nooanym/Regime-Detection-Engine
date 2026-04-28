"""Tests for rde.models.simulation (Phase 21)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.models.simulation import (
    ScenarioStats,
    SimulationConfig,
    SimulationResult,
    aggregate_scenarios,
    simulate_gaussian_regimes,
    simulate_garch_regimes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_params(K: int = 2, d: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((K, d)) * 0.01
    covars = np.array([np.eye(d) * 0.01 ** 2 for _ in range(K)])
    transmat = np.full((K, K), 0.1 / (K - 1))
    np.fill_diagonal(transmat, 0.9)
    startprob = np.full(K, 1.0 / K)
    return means, covars, transmat, startprob


class _FakeGARCHParams:
    def __init__(self, omega=1e-6, alpha=0.1, beta=0.85, mu=0.0):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.mu = mu


def _garch_params(K: int = 2):
    return [_FakeGARCHParams(omega=1e-6, alpha=0.1 + 0.05 * k, beta=0.8) for k in range(K)]


def _transmat(K: int = 2):
    mat = np.full((K, K), 0.1 / (K - 1))
    np.fill_diagonal(mat, 0.9)
    return mat


def _startprob(K: int = 2):
    return np.full(K, 1.0 / K)


# ---------------------------------------------------------------------------
# simulate_gaussian_regimes
# ---------------------------------------------------------------------------


class TestSimulateGaussianRegimes:
    def test_returns_simulation_result(self) -> None:
        means, covars, trans, start = _gaussian_params()
        result = simulate_gaussian_regimes(means, covars, trans, start)
        assert isinstance(result, SimulationResult)

    def test_returns_shape(self) -> None:
        cfg = SimulationConfig(n_paths=10, horizon=50)
        means, covars, trans, start = _gaussian_params()
        result = simulate_gaussian_regimes(means, covars, trans, start, config=cfg)
        assert result.returns.shape == (10, 50)

    def test_cum_returns_shape(self) -> None:
        cfg = SimulationConfig(n_paths=10, horizon=50)
        means, covars, trans, start = _gaussian_params()
        result = simulate_gaussian_regimes(means, covars, trans, start, config=cfg)
        assert result.cum_returns.shape == (10, 50)

    def test_regime_sequence_shape(self) -> None:
        cfg = SimulationConfig(n_paths=5, horizon=30)
        means, covars, trans, start = _gaussian_params()
        result = simulate_gaussian_regimes(means, covars, trans, start, config=cfg)
        assert result.regime_sequence.shape == (5, 30)

    def test_regime_labels_in_range(self) -> None:
        K = 3
        means, covars, trans, start = _gaussian_params(K=K)
        result = simulate_gaussian_regimes(means, covars, trans, start,
                                           config=SimulationConfig(n_paths=10, horizon=50))
        assert result.regime_sequence.min() >= 0
        assert result.regime_sequence.max() < K

    def test_cum_returns_is_cumsum(self) -> None:
        cfg = SimulationConfig(n_paths=3, horizon=20)
        means, covars, trans, start = _gaussian_params()
        result = simulate_gaussian_regimes(means, covars, trans, start, config=cfg)
        expected = np.cumsum(result.returns, axis=1)
        np.testing.assert_allclose(result.cum_returns, expected, atol=1e-10)

    def test_path_stats_n_rows(self) -> None:
        cfg = SimulationConfig(n_paths=7, horizon=50)
        means, covars, trans, start = _gaussian_params()
        result = simulate_gaussian_regimes(means, covars, trans, start, config=cfg)
        assert len(result.path_stats) == 7

    def test_path_stats_columns(self) -> None:
        means, covars, trans, start = _gaussian_params()
        result = simulate_gaussian_regimes(means, covars, trans, start)
        for col in ["total_return", "ann_vol", "sharpe", "max_drawdown"]:
            assert col in result.path_stats.columns

    def test_reproducible_with_same_seed(self) -> None:
        means, covars, trans, start = _gaussian_params()
        cfg = SimulationConfig(n_paths=5, horizon=30, seed=99)
        r1 = simulate_gaussian_regimes(means, covars, trans, start, config=cfg)
        r2 = simulate_gaussian_regimes(means, covars, trans, start, config=cfg)
        np.testing.assert_array_equal(r1.returns, r2.returns)

    def test_different_seeds_different_paths(self) -> None:
        means, covars, trans, start = _gaussian_params()
        r1 = simulate_gaussian_regimes(means, covars, trans, start,
                                       config=SimulationConfig(seed=0))
        r2 = simulate_gaussian_regimes(means, covars, trans, start,
                                       config=SimulationConfig(seed=1))
        assert not np.allclose(r1.returns, r2.returns)

    def test_higher_mean_regime_higher_avg_return(self) -> None:
        d = 1
        means = np.array([[0.001], [-0.001]])
        covars = np.array([np.eye(d) * 1e-4, np.eye(d) * 1e-4])
        # All time in regime 0 (high mean)
        trans = np.array([[1.0, 0.0], [0.0, 1.0]])
        start = np.array([1.0, 0.0])
        cfg = SimulationConfig(n_paths=50, horizon=200, seed=0)
        result = simulate_gaussian_regimes(means, covars, trans, start, config=cfg)
        assert result.returns.mean() > 0.0


# ---------------------------------------------------------------------------
# simulate_garch_regimes
# ---------------------------------------------------------------------------


class TestSimulateGARCHRegimes:
    def test_returns_simulation_result(self) -> None:
        params = _garch_params()
        result = simulate_garch_regimes(params, _transmat(), _startprob())
        assert isinstance(result, SimulationResult)

    def test_returns_shape(self) -> None:
        cfg = SimulationConfig(n_paths=10, horizon=50)
        params = _garch_params()
        result = simulate_garch_regimes(params, _transmat(), _startprob(), config=cfg)
        assert result.returns.shape == (10, 50)

    def test_regime_sequence_valid(self) -> None:
        K = 2
        cfg = SimulationConfig(n_paths=5, horizon=30)
        params = _garch_params(K)
        result = simulate_garch_regimes(params, _transmat(K), _startprob(K), config=cfg)
        assert result.regime_sequence.min() >= 0
        assert result.regime_sequence.max() < K

    def test_no_nan_returns(self) -> None:
        params = _garch_params()
        result = simulate_garch_regimes(params, _transmat(), _startprob())
        assert not np.any(np.isnan(result.returns))

    def test_reproducible(self) -> None:
        params = _garch_params()
        cfg = SimulationConfig(n_paths=5, horizon=30, seed=7)
        r1 = simulate_garch_regimes(params, _transmat(), _startprob(), config=cfg)
        r2 = simulate_garch_regimes(params, _transmat(), _startprob(), config=cfg)
        np.testing.assert_array_equal(r1.returns, r2.returns)

    def test_high_persistence_higher_vol_clustering(self) -> None:
        """High beta should produce more autocorrelated squared returns."""
        cfg = SimulationConfig(n_paths=1, horizon=2000, seed=0)
        params_lo = [_FakeGARCHParams(omega=1e-5, alpha=0.1, beta=0.3)]
        params_hi = [_FakeGARCHParams(omega=1e-5, alpha=0.1, beta=0.85)]
        trans = np.array([[1.0]])
        start = np.array([1.0])
        r_lo = simulate_garch_regimes(params_lo, trans, start, config=cfg)
        r_hi = simulate_garch_regimes(params_hi, trans, start, config=cfg)
        # High persistence → higher variance of rolling volatility
        sq_lo = pd.Series(r_lo.returns[0] ** 2).rolling(20).std().dropna().std()
        sq_hi = pd.Series(r_hi.returns[0] ** 2).rolling(20).std().dropna().std()
        assert sq_hi > sq_lo


# ---------------------------------------------------------------------------
# aggregate_scenarios
# ---------------------------------------------------------------------------


class TestAggregateScenarios:
    def test_returns_scenario_stats(self) -> None:
        means, covars, trans, start = _gaussian_params()
        result = simulate_gaussian_regimes(means, covars, trans, start,
                                           config=SimulationConfig(n_paths=20, horizon=50))
        stats = aggregate_scenarios(result)
        assert isinstance(stats, ScenarioStats)

    def test_std_non_negative(self) -> None:
        means, covars, trans, start = _gaussian_params()
        result = simulate_gaussian_regimes(means, covars, trans, start,
                                           config=SimulationConfig(n_paths=20))
        stats = aggregate_scenarios(result)
        assert stats.std_total_return >= 0.0

    def test_var_le_mean(self) -> None:
        """5th-percentile return should be ≤ mean return."""
        means, covars, trans, start = _gaussian_params()
        result = simulate_gaussian_regimes(means, covars, trans, start,
                                           config=SimulationConfig(n_paths=50, horizon=100))
        stats = aggregate_scenarios(result)
        assert stats.var_95 <= stats.mean_total_return + 1e-9

    def test_cvar_le_var(self) -> None:
        """CVaR (expected loss beyond VaR) should be ≤ VaR."""
        means, covars, trans, start = _gaussian_params()
        result = simulate_gaussian_regimes(means, covars, trans, start,
                                           config=SimulationConfig(n_paths=50))
        stats = aggregate_scenarios(result)
        assert stats.cvar_95 <= stats.var_95 + 1e-9

    def test_mdd_non_negative(self) -> None:
        means, covars, trans, start = _gaussian_params()
        result = simulate_gaussian_regimes(means, covars, trans, start,
                                           config=SimulationConfig(n_paths=20))
        stats = aggregate_scenarios(result)
        assert stats.mean_max_drawdown >= 0.0
        assert stats.p95_max_drawdown >= stats.mean_max_drawdown - 1e-9

    def test_regime_freq_sums_to_one(self) -> None:
        means, covars, trans, start = _gaussian_params()
        result = simulate_gaussian_regimes(means, covars, trans, start,
                                           config=SimulationConfig(n_paths=20, horizon=100))
        stats = aggregate_scenarios(result)
        assert stats.regime_freq.sum() == pytest.approx(1.0, abs=1e-9)

    def test_regime_freq_non_negative(self) -> None:
        means, covars, trans, start = _gaussian_params()
        result = simulate_gaussian_regimes(means, covars, trans, start)
        stats = aggregate_scenarios(result)
        assert (stats.regime_freq >= 0).all()
