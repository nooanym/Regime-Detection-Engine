"""Tests for vol_target_overlay — Phase 37b Track B."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.research.strategies.vol_target_overlay import (
    OverlayConfig,
    OverlayBacktestResult,
    _posterior_to_exposure,
    compute_overlay_signal,
    overlay_tearsheet,
    run_overlay_backtest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_posteriors(T: int, K: int, concentrated: bool = False) -> np.ndarray:
    """Make test posteriors: concentrated (one state dominates) or diffuse."""
    rng = np.random.default_rng(0)
    if concentrated:
        p = np.zeros((T, K))
        p[:, 0] = 0.85
        p[:, 1:] = 0.15 / (K - 1)
    else:
        p = rng.dirichlet(np.ones(K), size=T)
    return p.astype(np.float64)


def _make_returns(T: int, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=T, freq="h")
    return pd.Series(rng.normal(0.0001, 0.005, T), index=idx)


# ---------------------------------------------------------------------------
# _posterior_to_exposure
# ---------------------------------------------------------------------------


class TestPosteriorToExposure:
    def test_high_confidence_gives_full_exposure(self):
        cfg = OverlayConfig(high_conf_threshold=0.70, low_conf_threshold=0.50, min_exposure=0.3)
        exp = _posterior_to_exposure(0.90, cfg)
        assert exp == pytest.approx(1.0)

    def test_low_confidence_gives_min_exposure(self):
        cfg = OverlayConfig(high_conf_threshold=0.70, low_conf_threshold=0.50, min_exposure=0.3)
        exp = _posterior_to_exposure(0.40, cfg)
        assert exp == pytest.approx(0.3)

    def test_midpoint_gives_linear_interpolation(self):
        cfg = OverlayConfig(high_conf_threshold=0.70, low_conf_threshold=0.50, min_exposure=0.3)
        # midpoint = 0.60 → t=0.5
        exp = _posterior_to_exposure(0.60, cfg)
        expected = 0.3 + 0.5 * (1.0 - 0.3)
        assert exp == pytest.approx(expected, rel=1e-6)

    def test_vol_cap_applied_when_exceeded(self):
        cfg = OverlayConfig(vol_threshold_ann=0.5)
        # Posterior is concentrated (would give 1.0), but vol is very high
        exp = _posterior_to_exposure(0.90, cfg, posterior_weighted_vol_ann=2.0)
        # cap = 0.5 / 2.0 = 0.25 — but clipped to min_exposure=0.3
        assert exp == pytest.approx(0.3)

    def test_vol_cap_not_applied_below_threshold(self):
        cfg = OverlayConfig(vol_threshold_ann=0.5)
        exp = _posterior_to_exposure(0.90, cfg, posterior_weighted_vol_ann=0.3)
        assert exp == pytest.approx(1.0)

    def test_exposure_clipped_to_min(self):
        cfg = OverlayConfig(min_exposure=0.5)
        # Even extremely low confidence won't go below min_exposure
        exp = _posterior_to_exposure(0.0, cfg)
        assert exp >= cfg.min_exposure

    def test_degenerate_range_zero(self):
        cfg = OverlayConfig(high_conf_threshold=0.6, low_conf_threshold=0.6)
        exp_above = _posterior_to_exposure(0.7, cfg)
        exp_below = _posterior_to_exposure(0.5, cfg)
        assert exp_above == pytest.approx(1.0)
        assert exp_below == pytest.approx(cfg.min_exposure)


# ---------------------------------------------------------------------------
# compute_overlay_signal
# ---------------------------------------------------------------------------


class TestComputeOverlaySignal:
    def test_shape(self):
        T, K = 100, 4
        post = _make_posteriors(T, K, concentrated=True)
        cfg = OverlayConfig()
        sig = compute_overlay_signal(post, cfg)
        assert sig.shape == (T,)

    def test_concentrated_posteriors_give_high_exposure(self):
        T, K = 200, 4
        post = _make_posteriors(T, K, concentrated=True)  # max_post ≈ 0.85
        cfg = OverlayConfig(high_conf_threshold=0.70)
        sig = compute_overlay_signal(post, cfg)
        assert float(sig.mean()) == pytest.approx(1.0)

    def test_diffuse_posteriors_give_low_exposure(self):
        T, K = 200, 4
        # Uniform → max_post = 0.25
        post = np.full((T, K), 1.0 / K)
        cfg = OverlayConfig(low_conf_threshold=0.50, min_exposure=0.3)
        sig = compute_overlay_signal(post, cfg)
        assert float(sig.mean()) == pytest.approx(0.3)

    def test_vol_penalty_reduces_exposure(self):
        T, K = 50, 3
        post = _make_posteriors(T, K, concentrated=True)
        regime_vols = np.array([2.0, 2.0, 2.0])  # high vol in every regime
        cfg = OverlayConfig(high_conf_threshold=0.70, vol_threshold_ann=0.5, min_exposure=0.3)
        sig_no_vol = compute_overlay_signal(post, cfg)
        sig_vol = compute_overlay_signal(post, cfg, regime_ann_vols=regime_vols)
        assert float(sig_vol.mean()) <= float(sig_no_vol.mean())

    def test_exposure_in_valid_range(self):
        T, K = 300, 5
        rng = np.random.default_rng(1)
        post = rng.dirichlet(np.ones(K) * 2, size=T)
        cfg = OverlayConfig()
        sig = compute_overlay_signal(post, cfg)
        assert float(sig.min()) >= cfg.min_exposure - 1e-9
        assert float(sig.max()) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# run_overlay_backtest
# ---------------------------------------------------------------------------


class TestRunOverlayBacktest:
    def test_returns_correct_type(self):
        T, K = 240, 4
        returns = _make_returns(T)
        post = _make_posteriors(T, K, concentrated=True)
        result = run_overlay_backtest(returns, post)
        assert isinstance(result, OverlayBacktestResult)

    def test_equity_starts_at_first_bar_and_is_positive(self):
        T, K = 480, 3
        returns = _make_returns(T)
        post = _make_posteriors(T, K, concentrated=True)
        result = run_overlay_backtest(returns, post)
        assert float(result.equity.iloc[0]) > 0.0
        assert (result.equity > 0).all()

    def test_passive_equity_matches_cumulative_returns(self):
        T, K = 120, 2
        returns = _make_returns(T)
        post = _make_posteriors(T, K, concentrated=True)
        result = run_overlay_backtest(returns, post)
        expected_final = float((1.0 + returns).prod())
        assert float(result.passive_equity.iloc[-1]) == pytest.approx(expected_final, rel=1e-6)

    def test_n_rebalances_bounded(self):
        T, K = 720, 4
        returns = _make_returns(T)
        post = _make_posteriors(T, K)
        cfg = OverlayConfig(rebalance_bars=24)
        result = run_overlay_backtest(returns, post, cfg)
        # Can only rebalance at multiples of 24 → max T/24 rebalances
        assert result.n_rebalances <= T // cfg.rebalance_bars + 1

    def test_positions_in_valid_range(self):
        T, K = 480, 4
        returns = _make_returns(T)
        post = _make_posteriors(T, K)
        cfg = OverlayConfig()
        result = run_overlay_backtest(returns, post, cfg)
        assert float(result.positions.min()) >= cfg.min_exposure - 1e-9
        assert float(result.positions.max()) <= 1.0 + 1e-9

    def test_concentrated_posteriors_give_full_exposure(self):
        T, K = 240, 3
        returns = _make_returns(T)
        post = _make_posteriors(T, K, concentrated=True)
        cfg = OverlayConfig(high_conf_threshold=0.70, rebalance_bars=1)
        result = run_overlay_backtest(returns, post, cfg)
        assert float(result.positions.mean()) == pytest.approx(1.0, abs=0.01)

    def test_rebalance_bars_reduces_turnover(self):
        T, K = 480, 4
        returns = _make_returns(T)
        post = _make_posteriors(T, K)
        cfg_freq = OverlayConfig(rebalance_bars=1)
        cfg_slow = OverlayConfig(rebalance_bars=24)
        r_freq = run_overlay_backtest(returns, post, cfg_freq)
        r_slow = run_overlay_backtest(returns, post, cfg_slow)
        assert r_slow.n_rebalances <= r_freq.n_rebalances

    def test_index_preserved(self):
        T, K = 120, 3
        returns = _make_returns(T)
        post = _make_posteriors(T, K)
        result = run_overlay_backtest(returns, post)
        pd.testing.assert_index_equal(result.equity.index, returns.index)
        pd.testing.assert_index_equal(result.positions.index, returns.index)

    def test_zero_cost_equals_weighted_returns(self):
        T, K = 48, 2
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0, 0.001, T), index=pd.date_range("2023-01-01", periods=T, freq="h"))
        # Constant concentrated posteriors → constant full exposure
        post = np.full((T, K), [0.9, 0.1])
        cfg = OverlayConfig(transaction_cost=0.0, rebalance_bars=1)
        result = run_overlay_backtest(returns, post, cfg)
        # At t=0 prev_pos=0 so that bar earns 0; from bar 1 onward, exposure=1.0
        # so strat returns[1:] == asset returns[1:]
        pd.testing.assert_series_equal(
            result.returns.iloc[1:], returns.iloc[1:], check_names=False, atol=1e-10
        )


# ---------------------------------------------------------------------------
# overlay_tearsheet
# ---------------------------------------------------------------------------


class TestOverlayTearsheet:
    def _make_result(self, T: int = 720, K: int = 4) -> OverlayBacktestResult:
        returns = _make_returns(T)
        post = _make_posteriors(T, K)
        return run_overlay_backtest(returns, post)

    def test_returns_series(self):
        result = self._make_result()
        ts = overlay_tearsheet(result)
        assert isinstance(ts, pd.Series)

    def test_required_keys_present(self):
        result = self._make_result()
        ts = overlay_tearsheet(result)
        for key in ("sharpe", "passive_sharpe", "sharpe_improvement", "trades_per_year", "mean_exposure"):
            assert key in ts.index, f"Missing key: {key}"

    def test_sharpe_improvement_is_difference(self):
        result = self._make_result()
        ts = overlay_tearsheet(result)
        assert float(ts["sharpe_improvement"]) == pytest.approx(
            float(ts["sharpe"]) - float(ts["passive_sharpe"]), abs=1e-6
        )

    def test_trades_per_year_nonnegative(self):
        result = self._make_result()
        ts = overlay_tearsheet(result)
        assert float(ts["trades_per_year"]) >= 0.0

    def test_mean_exposure_in_valid_range(self):
        result = self._make_result()
        ts = overlay_tearsheet(result)
        assert 0.0 <= float(ts["mean_exposure"]) <= 1.0
