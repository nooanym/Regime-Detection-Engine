"""Tests for rde.analysis.multi_asset_allocation (Phase 42)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.analysis.multi_asset_allocation import (
    MultiAssetConfig,
    MultiAssetResult,
    collect_rtmv_rebalance_cache,
    compare_allocations,
    equal_weight_baseline,
    global_min_var_baseline,
    regime_informed_min_var,
    regime_tilted_min_var,
    regime_tilted_min_var_from_cache,
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


# ---------------------------------------------------------------------------
# Regime-informed minimum-variance
# ---------------------------------------------------------------------------


class TestRegimeInformedMinVar:
    def test_returns_result_type(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = regime_informed_min_var(ret, feats, config=_fast_config())
        assert isinstance(result, MultiAssetResult)

    def test_name(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = regime_informed_min_var(ret, feats, config=_fast_config())
        assert result.name == "regime_informed_min_var"

    def test_weights_shape(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = regime_informed_min_var(ret, feats, config=_fast_config())
        assert result.weights.shape == ret.shape

    def test_weights_long_only(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = regime_informed_min_var(ret, feats, config=_fast_config())
        assert (result.weights.values >= -1e-9).all()

    def test_equity_positive(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = regime_informed_min_var(ret, feats, config=_fast_config())
        assert (result.equity.values > 0.0).all()

    def test_metrics_finite(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = regime_informed_min_var(ret, feats, config=_fast_config())
        for attr in ("sharpe", "calmar", "max_drawdown", "ann_return", "ann_vol"):
            assert np.isfinite(getattr(result, attr))

    def test_rebalance_dates_populated(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = regime_informed_min_var(ret, feats, config=_fast_config())
        assert len(result.rebalance_dates) >= 1

    def test_requires_at_least_two_assets(self) -> None:
        ret = _make_returns(N=1)
        feats = _make_features(ret)
        with pytest.raises(ValueError, match="at least 2"):
            regime_informed_min_var(ret, feats, config=_fast_config())

    def test_min_eligible_assets_must_be_two(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        with pytest.raises(ValueError, match="min_eligible_assets"):
            regime_informed_min_var(ret, feats, config=_fast_config(), min_eligible_assets=1)

    def test_high_threshold_falls_back_to_min_eligible(self) -> None:
        """With threshold=+inf all assets excluded; min_eligible fallback keeps 2."""
        ret = _make_returns(N=4)
        feats = _make_features(ret)
        cfg = _fast_config()
        result = regime_informed_min_var(
            ret, feats, config=cfg, exclusion_threshold=1e6, min_eligible_assets=2
        )
        assert isinstance(result, MultiAssetResult)
        # At every rebalance at least 2 assets have nonzero weight.
        reb_dates = result.rebalance_dates
        if reb_dates:
            for d in reb_dates:
                w = result.weights.loc[d].values
                assert (w > 1e-9).sum() >= 2

    def test_four_assets(self) -> None:
        ret = _make_returns(N=4)
        feats = _make_features(ret)
        result = regime_informed_min_var(ret, feats, config=_fast_config())
        assert result.weights.shape == (len(ret), 4)

    def test_portfolio_returns_length(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = regime_informed_min_var(ret, feats, config=_fast_config())
        assert len(result.portfolio_returns) == len(ret)


class TestRegimeTiltedMinVar:
    def test_returns_result_type(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = regime_tilted_min_var(ret, feats, config=_fast_config())
        assert isinstance(result, MultiAssetResult)

    def test_name(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = regime_tilted_min_var(ret, feats, config=_fast_config())
        assert result.name == "regime_tilted_min_var"

    def test_weights_shape(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = regime_tilted_min_var(ret, feats, config=_fast_config())
        assert result.weights.shape == (len(ret), ret.shape[1])

    def test_weights_long_only(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = regime_tilted_min_var(ret, feats, config=_fast_config())
        assert (result.weights.values >= -1e-9).all()

    def test_lambda_zero_resembles_min_var(self) -> None:
        """lambda=0 should be identical to global_min_var_baseline."""
        ret = _make_returns()
        feats = _make_features(ret)
        cfg = _fast_config()
        rtmv = regime_tilted_min_var(ret, feats, config=cfg, lambda_tilt=0.0)
        mv = global_min_var_baseline(
            ret,
            ann_factor=cfg.ann_factor,
            rebalance_bars=cfg.rebalance_bars,
            cov_window_bars=cfg.cov_window_bars,
            transaction_cost=cfg.transaction_cost,
        )
        assert abs(rtmv.sharpe - mv.sharpe) < 0.05

    def test_metrics_finite(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = regime_tilted_min_var(ret, feats, config=_fast_config())
        assert np.isfinite(result.sharpe)
        assert np.isfinite(result.calmar)
        assert np.isfinite(result.max_drawdown)

    def test_invalid_lambda_raises(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        with pytest.raises(ValueError, match="lambda_tilt"):
            regime_tilted_min_var(ret, feats, config=_fast_config(), lambda_tilt=1.5)

    def test_requires_at_least_two_assets(self) -> None:
        ret = _make_returns(N=1)
        feats = _make_features(ret)
        with pytest.raises(ValueError, match="2 assets"):
            regime_tilted_min_var(ret, feats, config=_fast_config())

    def test_portfolio_returns_length(self) -> None:
        ret = _make_returns()
        feats = _make_features(ret)
        result = regime_tilted_min_var(ret, feats, config=_fast_config())
        assert len(result.portfolio_returns) == len(ret)

    def test_lambda_grid_sharpe_monotone(self) -> None:
        """Higher lambda should shift results (not necessarily monotone, but all finite)."""
        ret = _make_returns()
        feats = _make_features(ret)
        cfg = _fast_config()
        for lam in [0.0, 0.1, 0.3, 0.5]:
            r = regime_tilted_min_var(ret, feats, config=cfg, lambda_tilt=lam)
            assert np.isfinite(r.sharpe), f"lambda={lam} gives non-finite Sharpe"

    def test_shuffle_seed_produces_different_result(self) -> None:
        """shuffle_seed=N changes portfolio returns vs no shuffle."""
        ret = _make_returns(T=350)
        feats = _make_features(ret)
        cfg = _fast_config()
        real = regime_tilted_min_var(ret, feats, config=cfg, lambda_tilt=0.2)
        shuffled = regime_tilted_min_var(
            ret, feats, config=cfg, lambda_tilt=0.2, shuffle_seed=42
        )
        assert not np.allclose(
            real.portfolio_returns.values,
            shuffled.portfolio_returns.values,
        ), "Shuffled result should differ from unshuffled"

    def test_shuffle_result_is_finite(self) -> None:
        ret = _make_returns(T=350)
        feats = _make_features(ret)
        result = regime_tilted_min_var(
            ret, feats, config=_fast_config(), lambda_tilt=0.2, shuffle_seed=7
        )
        assert np.isfinite(result.sharpe)
        assert len(result.portfolio_returns) == len(ret)

    def test_shuffle_seed_none_unchanged(self) -> None:
        """shuffle_seed=None (default) must produce the same result as before."""
        ret = _make_returns(T=350)
        feats = _make_features(ret)
        cfg = _fast_config()
        r1 = regime_tilted_min_var(ret, feats, config=cfg, lambda_tilt=0.1)
        r2 = regime_tilted_min_var(ret, feats, config=cfg, lambda_tilt=0.1, shuffle_seed=None)
        assert np.allclose(r1.portfolio_returns.values, r2.portfolio_returns.values)


# ---------------------------------------------------------------------------
# Phase 47: cache functions and CV validation helpers
# ---------------------------------------------------------------------------


class TestCollectRTMVRebalanceCache:
    """Tests for the HMM fit cache used by Phase 47."""

    def test_cache_structure(self) -> None:
        ret = _make_returns(T=350)
        feats = _make_features(ret)
        cfg = _fast_config()
        cache = collect_rtmv_rebalance_cache(ret, feats, config=cfg)
        assert len(cache) > 0
        entry = cache[0]
        assert "t" in entry and "w_minvar" in entry and "exp_returns" in entry
        assert entry["w_minvar"].shape == (len(ret.columns),)
        assert entry["exp_returns"].shape == (len(ret.columns),)

    def test_cache_weights_sum_to_one(self) -> None:
        ret = _make_returns(T=350)
        feats = _make_features(ret)
        cache = collect_rtmv_rebalance_cache(ret, feats, config=_fast_config())
        for entry in cache:
            assert abs(entry["w_minvar"].sum() - 1.0) < 1e-6

    def test_from_cache_matches_direct(self) -> None:
        """regime_tilted_min_var and regime_tilted_min_var_from_cache give same result."""
        ret = _make_returns(T=350)
        feats = _make_features(ret)
        cfg = _fast_config()
        r_direct = regime_tilted_min_var(ret, feats, config=cfg, lambda_tilt=0.2)
        cache = collect_rtmv_rebalance_cache(ret, feats, config=cfg)
        r_cached = regime_tilted_min_var_from_cache(ret, cache, config=cfg, lambda_tilt=0.2)
        assert np.isfinite(r_direct.sharpe)
        assert np.isfinite(r_cached.sharpe)
        # Sharpes should be identical (same HMM seeds, same data)
        assert abs(r_direct.sharpe - r_cached.sharpe) < 1e-6

    def test_from_cache_shuffle_differs(self) -> None:
        ret = _make_returns(T=350)
        feats = _make_features(ret)
        cfg = _fast_config()
        cache = collect_rtmv_rebalance_cache(ret, feats, config=cfg)
        r_real = regime_tilted_min_var_from_cache(ret, cache, config=cfg, lambda_tilt=0.2)
        r_shuf = regime_tilted_min_var_from_cache(
            ret, cache, config=cfg, lambda_tilt=0.2, shuffle_seed=42
        )
        assert not np.allclose(
            r_real.portfolio_returns.values, r_shuf.portfolio_returns.values,
        )


class TestRTMVCVValidation:
    """Tests for the Phase 47 CV validation helpers."""

    def test_fold_sharpes_length(self) -> None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from run_rtmv_cv_validation import _fold_sharpes

        ret = _make_returns()
        cfg = _fast_config()
        mv = global_min_var_baseline(
            ret,
            ann_factor=cfg.ann_factor,
            rebalance_bars=cfg.rebalance_bars,
            cov_window_bars=cfg.cov_window_bars,
            transaction_cost=cfg.transaction_cost,
        )
        fold_sh = _fold_sharpes(mv.portfolio_returns, n_folds=4, ann_factor=252)
        assert len(fold_sh) == 4
        assert all(np.isfinite(s) for s in fold_sh)

    def test_interpolate_break_even_exact_zero(self) -> None:
        from scripts.run_rtmv_cv_validation import _interpolate_break_even
        be = _interpolate_break_even([10, 50, 100, 200], [0.10, 0.05, -0.02, -0.10])
        assert 50.0 < be < 100.0

    def test_interpolate_break_even_never(self) -> None:
        from scripts.run_rtmv_cv_validation import _interpolate_break_even
        be = _interpolate_break_even([10, 50, 100], [0.05, 0.03, 0.01])
        assert be == float("inf")

    def test_rtmv_cv_result_dataclass(self) -> None:
        from scripts.run_rtmv_cv_validation import RTMVCVResult
        r = RTMVCVResult(lambda_tilt=0.30, n_folds=5)
        assert r.lambda_tilt == 0.30
        assert r.n_folds == 5
        assert r.fold_sharpe_rtmv == []
        assert r.pct_folds_rtmv_wins == 0.0
