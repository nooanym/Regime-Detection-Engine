"""Tests for rde.evaluation.regime_analytics (Phase 7)."""

import numpy as np
import pandas as pd
import pytest

from rde.evaluation.regime_analytics import (
    RegimeStats,
    _compute_per_run_mdd,
    _max_drawdown_from_returns,
    compute_regime_stats,
    regime_stats_to_dataframe,
    regime_transition_table,
    transition_forecast,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_returns_states(seed=0, T=500, n_states=3):
    rng = np.random.default_rng(seed)
    states = rng.integers(0, n_states, size=T)
    # State 0: low return, State 1: high return, State 2: crash
    mean_map = {0: -0.0002, 1: 0.0005, 2: -0.003}
    std_map  = {0: 0.005,   1: 0.008,  2: 0.025}
    returns = np.array([
        rng.normal(mean_map[int(s)], std_map[int(s)]) for s in states
    ])
    return returns, states


# ── _max_drawdown_from_returns ────────────────────────────────────────────────

class TestMaxDrawdown:
    def test_empty_returns(self):
        assert _max_drawdown_from_returns(np.array([])) == 0.0

    def test_single_bar(self):
        assert _max_drawdown_from_returns(np.array([0.01])) == 0.0

    def test_monotone_up(self):
        r = np.full(50, 0.001)
        assert _max_drawdown_from_returns(r) == pytest.approx(0.0, abs=1e-10)

    def test_monotone_down(self):
        # 10% drawdown over 100 bars of -0.001
        r = np.full(100, -0.001)
        mdd = _max_drawdown_from_returns(r)
        assert mdd > 0.0
        assert mdd < 1.0

    def test_known_drawdown(self):
        # +10%, -20%, net: +0.1 then -0.2 → peak=1.1, trough≈0.88
        r = np.array([np.log(1.1), np.log(0.8)])
        mdd = _max_drawdown_from_returns(r)
        # peak = 1.1, trough = 1.1 * 0.8 = 0.88, dd = (1.1 - 0.88) / 1.1 ≈ 0.2
        assert mdd == pytest.approx(0.2, abs=0.01)

    def test_in_range(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0, 0.01, 300)
        mdd = _max_drawdown_from_returns(r)
        assert 0.0 <= mdd <= 1.0


# ── _compute_per_run_mdd ──────────────────────────────────────────────────────

class TestPerRunMdd:
    def test_single_run(self):
        returns = np.array([0.01, -0.02, 0.01])
        states  = np.array([0, 0, 0])
        mdd = _compute_per_run_mdd(returns, states, 0)
        assert mdd > 0.0

    def test_no_runs_of_target(self):
        returns = np.array([0.01, 0.02])
        states  = np.array([1, 1])
        mdd = _compute_per_run_mdd(returns, states, 0)
        assert mdd == 0.0

    def test_multiple_runs_takes_max(self):
        # Two runs: one with small drawdown, one with larger
        returns = np.array([0.01, -0.005, 0.0, -0.03, 0.01])
        states  = np.array([0,    0,     1,   0,    0  ])
        mdd = _compute_per_run_mdd(returns, states, 0)
        # Run 1: [0.01, -0.005] and Run 2: [-0.03, 0.01]
        assert mdd > 0.0


# ── compute_regime_stats ─────────────────────────────────────────────────────

class TestComputeRegimeStats:
    def test_returns_one_per_state(self):
        returns, states = _make_returns_states()
        stats = compute_regime_stats(returns, states)
        assert len(stats) == 3

    def test_state_indices_correct(self):
        returns, states = _make_returns_states(n_states=3)
        result = compute_regime_stats(returns, states)
        assert [s.state for s in result] == [0, 1, 2]

    def test_counts_sum_to_T(self):
        returns, states = _make_returns_states(T=400)
        result = compute_regime_stats(returns, states)
        assert sum(s.count for s in result) == 400

    def test_weights_sum_to_one(self):
        returns, states = _make_returns_states()
        result = compute_regime_stats(returns, states)
        assert sum(s.weight for s in result) == pytest.approx(1.0, abs=1e-9)

    def test_labels_applied(self):
        returns, states = _make_returns_states(n_states=2)
        result = compute_regime_stats(returns, states, labels=["bear", "bull"])
        assert result[0].label == "bear"
        assert result[1].label == "bull"

    def test_default_labels(self):
        returns, states = _make_returns_states(n_states=2)
        result = compute_regime_stats(returns, states, labels=None)
        assert result[0].label == "State 0"

    def test_high_return_state_has_positive_sharpe(self):
        rng = np.random.default_rng(1)
        T = 1000
        # State 1 has consistently positive returns
        returns = np.where(
            np.arange(T) % 2 == 0,
            rng.normal(0.001, 0.003, T),   # state 0: mildly positive
            rng.normal(-0.002, 0.003, T),  # state 1: negative
        )
        states = np.array([i % 2 for i in range(T)])
        result = compute_regime_stats(returns, states)
        assert result[0].sharpe_ann > result[1].sharpe_ann

    def test_high_vol_state_has_higher_vol(self):
        rng = np.random.default_rng(2)
        T = 1000
        r0 = rng.normal(0.0, 0.002, T // 2)   # low vol
        r1 = rng.normal(0.0, 0.020, T // 2)   # high vol
        returns = np.concatenate([r0, r1])
        states  = np.array([0] * (T // 2) + [1] * (T // 2))
        result = compute_regime_stats(returns, states)
        assert result[1].vol_ann > result[0].vol_ann

    def test_win_rate_in_bounds(self):
        returns, states = _make_returns_states()
        result = compute_regime_stats(returns, states)
        for rs in result:
            assert 0.0 <= rs.win_rate <= 1.0

    def test_max_drawdown_in_bounds(self):
        returns, states = _make_returns_states()
        result = compute_regime_stats(returns, states)
        for rs in result:
            assert 0.0 <= rs.max_drawdown <= 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            compute_regime_stats(np.ones(10), np.zeros(9, dtype=int))

    def test_var_less_than_cvar(self):
        # CVaR (expected shortfall) should be <= VaR (more conservative)
        rng = np.random.default_rng(0)
        returns = rng.normal(-0.001, 0.01, 500)
        states  = np.zeros(500, dtype=int)
        result = compute_regime_stats(returns, states)
        assert result[0].cvar_95 <= result[0].var_95

    def test_annualisation_scaling(self):
        """Doubling bars_per_year should double mean and scale vol by sqrt(2)."""
        rng = np.random.default_rng(7)
        returns = rng.normal(0.0005, 0.01, 400)
        states  = np.zeros(400, dtype=int)
        r1 = compute_regime_stats(returns, states, bars_per_year=1000)[0]
        r2 = compute_regime_stats(returns, states, bars_per_year=2000)[0]
        assert r2.mean_return_ann == pytest.approx(2.0 * r1.mean_return_ann, rel=1e-9)
        assert r2.vol_ann == pytest.approx(np.sqrt(2) * r1.vol_ann, rel=1e-9)


# ── regime_stats_to_dataframe ─────────────────────────────────────────────────

class TestRegimeStatsToDataframe:
    def test_returns_dataframe(self):
        returns, states = _make_returns_states(n_states=3)
        stats = compute_regime_stats(returns, states)
        df = regime_stats_to_dataframe(stats)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_index_is_state(self):
        returns, states = _make_returns_states(n_states=2)
        df = regime_stats_to_dataframe(compute_regime_stats(returns, states))
        assert df.index.name == "state"
        assert list(df.index) == [0, 1]

    def test_expected_columns_present(self):
        returns, states = _make_returns_states(n_states=2)
        df = regime_stats_to_dataframe(compute_regime_stats(returns, states))
        for col in ["sharpe_ann", "vol_ann_%", "max_drawdown_%", "win_rate_%"]:
            assert col in df.columns


# ── transition_forecast ───────────────────────────────────────────────────────

class TestTransitionForecast:
    def _two_state_transmat(self):
        return np.array([[0.9, 0.1], [0.2, 0.8]])

    def test_shape(self):
        A = self._two_state_transmat()
        df = transition_forecast(A, current_state=0, horizons=[1, 6, 24])
        assert df.shape == (3, 2)

    def test_rows_sum_to_one(self):
        A = self._two_state_transmat()
        df = transition_forecast(A, current_state=0, horizons=[1, 6, 24, 168])
        np.testing.assert_allclose(df.sum(axis=1).values, 1.0, atol=1e-10)

    def test_horizon_1_matches_transmat_row(self):
        A = self._two_state_transmat()
        df = transition_forecast(A, current_state=0, horizons=[1])
        np.testing.assert_allclose(df.iloc[0].values, A[0], atol=1e-12)

    def test_converges_to_stationary(self):
        """At large horizons, forecast should approach the stationary distribution."""
        from rde.evaluation.transition import stationary_distribution
        A = self._two_state_transmat()
        stat = stationary_distribution(A)
        df = transition_forecast(A, current_state=0, horizons=[5000])
        np.testing.assert_allclose(df.iloc[0].values, stat, atol=1e-6)

    def test_different_starting_states_converge(self):
        """Both starting states should yield the same forecast at large horizon."""
        A = self._two_state_transmat()
        df0 = transition_forecast(A, current_state=0, horizons=[5000])
        df1 = transition_forecast(A, current_state=1, horizons=[5000])
        np.testing.assert_allclose(df0.values, df1.values, atol=1e-6)

    def test_absorbing_state_stays_put(self):
        """A self-absorbing state should stay with probability 1 at all horizons."""
        A = np.array([[1.0, 0.0], [0.3, 0.7]])
        df = transition_forecast(A, current_state=0, horizons=[1, 10, 100])
        np.testing.assert_allclose(df["S0"].values, 1.0, atol=1e-12)


# ── regime_transition_table ───────────────────────────────────────────────────

class TestRegimeTransitionTable:
    def test_shape(self):
        A = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])
        df = regime_transition_table(A, labels=["S0", "S1", "S2"])
        # 3 states × 5 default horizons = 15 rows, 3 columns
        assert df.shape == (15, 3)

    def test_rows_sum_to_one(self):
        A = np.array([[0.9, 0.1], [0.2, 0.8]])
        df = regime_transition_table(A)
        np.testing.assert_allclose(df.sum(axis=1).values, 1.0, atol=1e-10)

    def test_custom_horizons(self):
        A = np.array([[0.9, 0.1], [0.2, 0.8]])
        df = regime_transition_table(A, horizons=[1, 24])
        assert df.shape == (4, 2)   # 2 states × 2 horizons

    def test_multiindex_structure(self):
        A = np.array([[0.9, 0.1], [0.2, 0.8]])
        df = regime_transition_table(A, labels=["bear", "bull"])
        assert df.index.names == ["from_state", "horizon_bars"]
        assert "bear" in df.index.get_level_values("from_state")
