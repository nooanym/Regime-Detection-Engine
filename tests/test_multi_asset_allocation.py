"""Tests for rde.analysis.multi_asset_allocation (Phase 42)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.analysis.multi_asset_allocation import (
    MultiAssetConfig,
    MultiAssetResult,
    compare_allocations,
    equal_weight_baseline,
    global_min_var_baseline,
    run_multi_asset_allocation,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_returns(T: int = 300, N: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=T, freq="B")
    data = rng.standard_normal((T, N)) * 0.01
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(N)])


def _make_features(
    returns: pd.DataFrame,
    vol_window: int = 10,
) -> dict[str, pd.DataFrame]:
    """Build simple feature DataFrames (log_return + rolling vol) per asset."""
    features = {}
    for col in returns.columns:
        r = returns[col]
        vol = r.rolling(vol_window).std().fillna(r.std())
        df = pd.DataFrame({"log_return": r, "volatility": vol}, index=returns.index)
        features[col] = df
    return features


def _fast_config(n_states: int = 2) -> MultiAssetConfig:
    return MultiAssetConfig(
        ann_factor=252,
        rebalance_bars=50,
        lookback_bars=80,
        cov_window_bars=40,
        n_states=n_states,
        n_restarts=1,
        transaction_cost=0.001,
        target_vol=None,
    )


# ---------------------------------------------------------------------------
# MultiAssetResult structure
# ---------------------------------------------------------------------------


class TestRunMultiAssetAllocation:
    def test_returns_result_type(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = run_multi_asset_allocation(ret, feats, config=_fast_config())
        assert isinstance(result, MultiAssetResult)

    def test_weights_shape(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = run_multi_asset_allocation(ret, feats, config=_fast_config())
        assert result.weights.shape == ret.shape

    def test_weights_columns_match_assets(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = run_multi_asset_allocation(ret, feats, config=_fast_config())
        assert list(result.weights.columns) == list(ret.columns)

    def test_weights_long_only(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        cfg = _fast_config()
        cfg.min_weight = 0.0
        result = run_multi_asset_allocation(ret, feats, config=cfg)
        assert (result.weights.values >= -1e-9).all()

    def test_portfolio_returns_length(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = run_multi_asset_allocation(ret, feats, config=_fast_config())
        assert len(result.portfolio_returns) == len(ret)

    def test_equity_starts_near_one(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = run_multi_asset_allocation(ret, feats, config=_fast_config())
        assert abs(result.equity.iloc[0] - 1.0) < 0.1

    def test_equity_positive(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = run_multi_asset_allocation(ret, feats, config=_fast_config())
        assert (result.equity.values > 0.0).all()

    def test_rebalance_dates_populated(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = run_multi_asset_allocation(ret, feats, config=_fast_config())
        assert len(result.rebalance_dates) >= 1

    def test_metrics_finite(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = run_multi_asset_allocation(ret, feats, config=_fast_config())
        assert np.isfinite(result.sharpe)
        assert np.isfinite(result.calmar)
        assert np.isfinite(result.max_drawdown)
        assert np.isfinite(result.ann_return)
        assert np.isfinite(result.ann_vol)

    def test_min_var_mode(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        cfg = _fast_config()
        cfg.mvo_kind = "min_var"
        result = run_multi_asset_allocation(ret, feats, config=cfg)
        assert isinstance(result, MultiAssetResult)
        assert result.weights.shape == ret.shape

    def test_requires_at_least_two_assets(self) -> None:
        ret = _make_returns(N=1)
        feats = _make_features(ret)
        with pytest.raises(ValueError, match="at least 2"):
            run_multi_asset_allocation(ret, feats, config=_fast_config())

    def test_vol_target_overlay(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        cfg = _fast_config()
        cfg.target_vol = 0.10
        result = run_multi_asset_allocation(ret, feats, config=cfg)
        assert isinstance(result, MultiAssetResult)
        assert (result.weights.values >= -1e-9).all()

    def test_asset_returns_stored(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = run_multi_asset_allocation(ret, feats, config=_fast_config())
        assert result.asset_returns.shape == ret.shape

    def test_four_assets(self) -> None:
        ret = _make_returns(N=4)
        feats = _make_features(ret)
        result = run_multi_asset_allocation(ret, feats, config=_fast_config())
        assert result.weights.shape == (len(ret), 4)


# ---------------------------------------------------------------------------
# No-lookahead sanity: weights must not change for past periods
# ---------------------------------------------------------------------------


class TestNoLookahead:
    def test_early_weights_unchanged_after_later_rebalance(self) -> None:
        """Weights in the first rebalance period must equal the initial equal-weight."""
        ret = _make_returns(T=300)
        feats = _make_features(ret)
        cfg = _fast_config()
        result = run_multi_asset_allocation(ret, feats, config=cfg)
        # Before the first rebalance (bars 0..lookback_bars-1), weights are
        # initialised to equal weight and not changed by later rebalances.
        pre = result.weights.iloc[: cfg.lookback_bars - 1].values
        N = ret.shape[1]
        expected = np.full_like(pre, 1.0 / N)
        np.testing.assert_array_almost_equal(pre, expected, decimal=6)


# ---------------------------------------------------------------------------
# Equal-weight baseline
# ---------------------------------------------------------------------------


class TestEqualWeightBaseline:
    def test_returns_result(self) -> None:
        ret = _make_returns()
        result = equal_weight_baseline(ret, ann_factor=252)
        assert isinstance(result, MultiAssetResult)

    def test_weights_are_equal(self) -> None:
        ret = _make_returns(N=4)
        result = equal_weight_baseline(ret)
        N = ret.shape[1]
        np.testing.assert_array_almost_equal(
            result.weights.values, np.full_like(result.weights.values, 1.0 / N)
        )

    def test_equity_positive(self) -> None:
        ret = _make_returns()
        result = equal_weight_baseline(ret)
        assert (result.equity.values > 0.0).all()

    def test_metrics_finite(self) -> None:
        ret = _make_returns()
        result = equal_weight_baseline(ret)
        assert np.isfinite(result.sharpe)
        assert np.isfinite(result.max_drawdown)


# ---------------------------------------------------------------------------
# Global min-var baseline
# ---------------------------------------------------------------------------


class TestGlobalMinVarBaseline:
    def test_returns_result(self) -> None:
        ret = _make_returns()
        result = global_min_var_baseline(ret, ann_factor=252)
        assert isinstance(result, MultiAssetResult)

    def test_weights_shape(self) -> None:
        ret = _make_returns()
        result = global_min_var_baseline(ret)
        assert result.weights.shape == ret.shape

    def test_weights_long_only_by_default(self) -> None:
        ret = _make_returns()
        result = global_min_var_baseline(ret, min_weight=0.0)
        assert (result.weights.values >= -1e-9).all()

    def test_rebalance_dates_populated(self) -> None:
        ret = _make_returns()
        result = global_min_var_baseline(ret)
        assert len(result.rebalance_dates) >= 1

    def test_metrics_finite(self) -> None:
        ret = _make_returns()
        result = global_min_var_baseline(ret)
        assert np.isfinite(result.sharpe)
        assert np.isfinite(result.max_drawdown)


# ---------------------------------------------------------------------------
# compare_allocations
# ---------------------------------------------------------------------------


class TestCompareAllocations:
    def _get_results(self) -> dict[str, MultiAssetResult]:
        ret = _make_returns()
        feats = _make_features(ret)
        ew = equal_weight_baseline(ret)
        mv = global_min_var_baseline(ret)
        rc = run_multi_asset_allocation(ret, feats, config=_fast_config())
        return {"equal_weight": ew, "min_var": mv, "regime_mvo": rc}

    def test_returns_dataframe(self) -> None:
        results = self._get_results()
        df = compare_allocations(results)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self) -> None:
        results = self._get_results()
        df = compare_allocations(results)
        for col in ("name", "sharpe", "calmar", "max_drawdown", "ann_return", "ann_vol"):
            assert col in df.columns

    def test_row_count(self) -> None:
        results = self._get_results()
        df = compare_allocations(results)
        assert len(df) == len(results)

    def test_sorted_by_sharpe_descending(self) -> None:
        results = self._get_results()
        df = compare_allocations(results)
        sharpes = df["sharpe"].values
        assert all(sharpes[i] >= sharpes[i + 1] for i in range(len(sharpes) - 1))

    def test_all_metrics_finite(self) -> None:
        results = self._get_results()
        df = compare_allocations(results)
        for col in ("sharpe", "calmar", "max_drawdown", "ann_return", "ann_vol"):
            assert np.isfinite(df[col].values).all()
