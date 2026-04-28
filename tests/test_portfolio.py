"""Tests for rde.analysis.portfolio (Phase 14)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.analysis.portfolio import (
    AllocationConfig,
    KellyConfig,
    VolTargetConfig,
    kelly_weights,
    portfolio_returns,
    score_proportional_weights,
    vol_target_weights,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _scores(T: int = 200, N: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
    vals = np.clip(rng.normal(0.0, 0.6, size=(T, N)), -1.0, 1.0)
    return pd.DataFrame(vals, index=idx, columns=[f"A{i}" for i in range(N)])


def _returns(T: int = 200, N: int = 3, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
    vals = rng.normal(0.0, 0.01, size=(T, N))
    return pd.DataFrame(vals, index=idx, columns=[f"A{i}" for i in range(N)])


# ── score_proportional_weights ────────────────────────────────────────────────


def test_spw_shape():
    s = _scores()
    w = score_proportional_weights(s)
    assert w.shape == s.shape


def test_spw_long_only_no_negatives():
    s = _scores()
    w = score_proportional_weights(s, AllocationConfig(long_only=True))
    assert (w.values >= -1e-12).all()


def test_spw_long_short_has_negatives():
    s = _scores(seed=99)
    w = score_proportional_weights(s, AllocationConfig(long_only=False))
    # With random scores some should be negative
    assert (w.values < 0).any()


def test_spw_normalised_abs_sum_le_one():
    s = _scores()
    w = score_proportional_weights(s, AllocationConfig(normalise=True))
    abs_sums = np.abs(w.values).sum(axis=1)
    # rows with all-zero scores get weight 0 everywhere (sum = 0 is fine)
    nonzero_mask = abs_sums > 1e-9
    np.testing.assert_allclose(abs_sums[nonzero_mask], 1.0, atol=1e-9)


def test_spw_unnormalised_preserves_zero():
    s = _scores()
    s.iloc[5] = 0.0  # entire row is flat
    w = score_proportional_weights(s, AllocationConfig(normalise=False))
    np.testing.assert_allclose(w.iloc[5].values, 0.0)


def test_spw_index_preserved():
    s = _scores()
    w = score_proportional_weights(s)
    pd.testing.assert_index_equal(w.index, s.index)


def test_spw_columns_preserved():
    s = _scores()
    w = score_proportional_weights(s)
    assert list(w.columns) == list(s.columns)


def test_spw_all_bear_long_only_gives_zero_weights():
    s = _scores()
    s_neg = s.copy()
    s_neg.iloc[:] = -1.0  # all bear
    w = score_proportional_weights(s_neg, AllocationConfig(long_only=True))
    np.testing.assert_allclose(w.values, 0.0)


# ── vol_target_weights ────────────────────────────────────────────────────────


def test_vtw_shape():
    s = _scores()
    r = _returns()
    base = score_proportional_weights(s)
    w = vol_target_weights(base, r)
    assert w.shape == base.shape


def test_vtw_index_preserved():
    s = _scores()
    r = _returns()
    base = score_proportional_weights(s)
    w = vol_target_weights(base, r)
    pd.testing.assert_index_equal(w.index, base.index)


def test_vtw_zero_base_gives_zero_output():
    s = _scores()
    r = _returns()
    zero_base = pd.DataFrame(0.0, index=s.index, columns=s.columns)
    w = vol_target_weights(zero_base, r)
    np.testing.assert_allclose(w.values, 0.0)


def test_vtw_respects_max_leverage():
    s = _scores()
    r = _returns()
    cfg = VolTargetConfig(target_vol=5.0, max_leverage=1.0)  # very high target → capped
    base = score_proportional_weights(s)
    w = vol_target_weights(base, r, config=cfg)
    abs_sums = np.abs(w.values).sum(axis=1)
    assert (abs_sums <= 1.0 + 1e-9).all()


def test_vtw_high_target_gives_higher_leverage():
    s = _scores()
    r = _returns()
    base = score_proportional_weights(s)
    cfg_lo = VolTargetConfig(target_vol=0.05, max_leverage=10.0)
    cfg_hi = VolTargetConfig(target_vol=0.30, max_leverage=10.0)
    w_lo = vol_target_weights(base, r, config=cfg_lo)
    w_hi = vol_target_weights(base, r, config=cfg_hi)
    # Higher target vol → generally larger absolute weights (after warm-up)
    warm_up = 170  # wait for rolling window to fill
    assert np.abs(w_hi.values[warm_up:]).mean() >= np.abs(w_lo.values[warm_up:]).mean() - 1e-9


# ── kelly_weights ─────────────────────────────────────────────────────────────


def test_kw_shape():
    s = _scores()
    r = _returns()
    w = kelly_weights(r, s)
    assert w.shape == s.shape


def test_kw_long_only_no_negatives():
    s = _scores()
    r = _returns()
    w = kelly_weights(r, s, KellyConfig(long_only=True))
    assert (w.values >= -1e-12).all()


def test_kw_normalised_abs_sum():
    s = _scores()
    r = _returns()
    w = kelly_weights(r, s, KellyConfig(normalise=True))
    abs_sums = np.abs(w.values).sum(axis=1)
    nonzero = abs_sums > 1e-9
    np.testing.assert_allclose(abs_sums[nonzero], 1.0, atol=1e-9)


def test_kw_smaller_fraction_gives_smaller_weights():
    s = _scores()
    r = _returns()
    w_full = kelly_weights(r, s, KellyConfig(fraction=1.0, normalise=False))
    w_quarter = kelly_weights(r, s, KellyConfig(fraction=0.25, normalise=False))
    # Quarter-Kelly weights should be smaller in magnitude
    np.testing.assert_array_less(
        np.abs(w_quarter.values).mean(),
        np.abs(w_full.values).mean() + 1e-9,
    )


def test_kw_index_columns_preserved():
    s = _scores()
    r = _returns()
    w = kelly_weights(r, s)
    pd.testing.assert_index_equal(w.index, s.index)
    assert list(w.columns) == list(s.columns)


# ── portfolio_returns ─────────────────────────────────────────────────────────


def test_pr_shape():
    s = _scores()
    r = _returns()
    w = score_proportional_weights(s)
    pr = portfolio_returns(w, r)
    assert pr.shape == (len(w),)


def test_pr_first_bar_zero_gross():
    """First bar: lagged weight = 0 → gross return = 0; net can be 0 (no startup cost)."""
    s = _scores()
    r = _returns()
    w = score_proportional_weights(s)
    pr = portfolio_returns(w, r, transaction_cost_bps=0.0)
    assert pr.iloc[0] == pytest.approx(0.0)


def test_pr_zero_cost_higher_return():
    s = _scores()
    r = _returns()
    w = score_proportional_weights(s)
    pr_free = portfolio_returns(w, r, transaction_cost_bps=0.0)
    pr_cost = portfolio_returns(w, r, transaction_cost_bps=10.0)
    # With turnover, higher cost → lower cumulative net return
    assert pr_free.sum() >= pr_cost.sum()


def test_pr_flat_weights_zero_return():
    s = _scores()
    r = _returns()
    zero_w = pd.DataFrame(0.0, index=s.index, columns=s.columns)
    pr = portfolio_returns(zero_w, r)
    np.testing.assert_allclose(pr.values, 0.0)


def test_pr_fully_long_positive_market():
    """With fully positive weights on a rising market, return should be positive."""
    N, T = 3, 300
    idx = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
    w = pd.DataFrame(
        np.full((T, N), 1.0 / N),
        index=idx,
        columns=[f"A{i}" for i in range(N)],
    )
    r = pd.DataFrame(
        np.full((T, N), 0.001),  # +0.1% every bar
        index=idx,
        columns=[f"A{i}" for i in range(N)],
    )
    pr = portfolio_returns(w, r, transaction_cost_bps=0.0)
    assert pr.sum() > 0


def test_pr_index_preserved():
    s = _scores()
    r = _returns()
    w = score_proportional_weights(s)
    pr = portfolio_returns(w, r)
    pd.testing.assert_index_equal(pr.index, w.index)
