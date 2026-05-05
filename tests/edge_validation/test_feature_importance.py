"""Tests for rde.evaluation.feature_importance (Phase 37.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.evaluation.feature_importance import (
    FeatureImportanceResult,
    FoldImportance,
    _permutation_importance_single_fold,
    run_permutation_importance,
)
from rde.evaluation.purged_cv import _default_strategy_config
from rde.models.hmm import train_hmm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_df(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    log_ret = rng.normal(0, 0.002, n)
    return pd.DataFrame(
        {
            "log_return": log_ret,
            "volatility_w24": np.abs(log_ret) + rng.uniform(0.001, 0.003, n),
            "smoothed_return_w12": pd.Series(log_ret).rolling(12, min_periods=1).mean().values,
        },
        index=idx,
    )


FEATURE_COLS = ["log_return", "volatility_w24", "smoothed_return_w12"]
FAST_KW = {"n_restarts": 1, "n_iter": 10, "seed_base": 0}


# ---------------------------------------------------------------------------
# FoldImportance
# ---------------------------------------------------------------------------


class TestFoldImportance:
    def test_fields(self):
        fi = FoldImportance(fold_id=0, importances={"log_return": 0.2, "vol": -0.1})
        assert fi.fold_id == 0
        assert fi.importances["log_return"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# FeatureImportanceResult
# ---------------------------------------------------------------------------


class TestFeatureImportanceResult:
    def _make(self) -> FeatureImportanceResult:
        fis = [
            FoldImportance(0, {"a": 0.3, "b": -0.1}),
            FoldImportance(1, {"a": 0.2, "b": 0.1}),
            FoldImportance(2, {"a": 0.4, "b": 0.2}),
        ]
        return FeatureImportanceResult(
            feature_names=["a", "b"],
            fold_importances=fis,
            mean_importance={"a": 0.3, "b": 0.067},
            std_importance={"a": 0.1, "b": 0.153},
            positive_fold_fraction={"a": 1.0, "b": 0.667},
            is_stable={"a": True, "b": False},
        )

    def test_is_stable_threshold(self):
        r = self._make()
        assert r.is_stable["a"] is True
        assert r.is_stable["b"] is False

    def test_positive_fold_fraction_range(self):
        r = self._make()
        for v in r.positive_fold_fraction.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# _permutation_importance_single_fold
# ---------------------------------------------------------------------------


class TestPermutationImportanceSingleFold:
    def _fitted(self):
        df = _synthetic_df(500)
        X = df[FEATURE_COLS].values
        return train_hmm(X, n_states=2, feature_names=FEATURE_COLS, **FAST_KW)

    def test_returns_dict_with_all_features(self):
        fitted = self._fitted()
        df = _synthetic_df(500)
        X_test = df[FEATURE_COLS].values[350:]
        ret_test = df["log_return"].values[350:]
        strategy = _default_strategy_config(fitted, 8760, 0.0001)

        from rde.inference.online import OnlineDecoder
        decoder = OnlineDecoder(fitted)
        posteriors = decoder.batch_filter(X_test)
        regimes = posteriors.argmax(axis=1)
        from rde.analysis.backtest import backtest_tearsheet, run_backtest
        bt = run_backtest(ret_test, regimes, strategy)
        ts = backtest_tearsheet(bt, ann_factor=8760)
        model_sharpe = float(ts["sharpe"])

        imp = _permutation_importance_single_fold(
            X_test_raw=X_test,
            regimes_model=regimes,
            model_sharpe=model_sharpe,
            returns_test=ret_test,
            fitted=fitted,
            feature_cols=FEATURE_COLS,
            ann_factor=8760,
            transaction_cost=0.0001,
            n_permutations=3,
            rng=np.random.default_rng(0),
        )
        assert set(imp.keys()) == set(FEATURE_COLS)
        assert all(np.isfinite(v) for v in imp.values())

    def test_importance_is_finite(self):
        fitted = self._fitted()
        df = _synthetic_df(500)
        X_test = df[FEATURE_COLS].values[350:]
        ret_test = df["log_return"].values[350:]

        from rde.inference.online import OnlineDecoder
        from rde.analysis.backtest import backtest_tearsheet, run_backtest

        strategy = _default_strategy_config(fitted, 8760, 0.0001)
        decoder = OnlineDecoder(fitted)
        posteriors = decoder.batch_filter(X_test)
        regimes = posteriors.argmax(axis=1)
        bt = run_backtest(ret_test, regimes, strategy)
        ts = backtest_tearsheet(bt, ann_factor=8760)
        model_sharpe = float(ts["sharpe"])

        imp = _permutation_importance_single_fold(
            X_test_raw=X_test,
            regimes_model=regimes,
            model_sharpe=model_sharpe,
            returns_test=ret_test,
            fitted=fitted,
            feature_cols=FEATURE_COLS,
            ann_factor=8760,
            transaction_cost=0.0001,
            n_permutations=3,
        )
        for feat, val in imp.items():
            assert np.isfinite(val), f"{feat}: importance not finite"


# ---------------------------------------------------------------------------
# run_permutation_importance
# ---------------------------------------------------------------------------


class TestRunPermutationImportance:
    def test_returns_feature_importance_result(self):
        df = _synthetic_df(2000)
        result = run_permutation_importance(
            df, FEATURE_COLS, "log_return", n_states=2,
            train_bars=600, test_bars=200, embargo_bars=24,
            n_permutations=2,
            train_kwargs=FAST_KW,
            seed=0,
        )
        assert isinstance(result, FeatureImportanceResult)

    def test_feature_names_match(self):
        df = _synthetic_df(2000)
        result = run_permutation_importance(
            df, FEATURE_COLS, "log_return", n_states=2,
            train_bars=600, test_bars=200,
            n_permutations=2,
            train_kwargs=FAST_KW,
        )
        assert result.feature_names == FEATURE_COLS

    def test_fold_count_matches_splits(self):
        df = _synthetic_df(2000)
        result = run_permutation_importance(
            df, FEATURE_COLS, "log_return", n_states=2,
            train_bars=600, test_bars=200, embargo_bars=24,
            n_permutations=2,
            train_kwargs=FAST_KW,
        )
        assert len(result.fold_importances) > 0

    def test_mean_importance_finite(self):
        df = _synthetic_df(2000)
        result = run_permutation_importance(
            df, FEATURE_COLS, "log_return", n_states=2,
            train_bars=600, test_bars=200,
            n_permutations=2,
            train_kwargs=FAST_KW,
        )
        for feat in FEATURE_COLS:
            assert np.isfinite(result.mean_importance[feat]), f"{feat} mean not finite"

    def test_positive_fold_fraction_in_range(self):
        df = _synthetic_df(2000)
        result = run_permutation_importance(
            df, FEATURE_COLS, "log_return", n_states=2,
            train_bars=600, test_bars=200,
            n_permutations=2,
            train_kwargs=FAST_KW,
        )
        for feat, frac in result.positive_fold_fraction.items():
            assert 0.0 <= frac <= 1.0, f"{feat}: fraction {frac} out of range"

    def test_is_stable_keys_match_features(self):
        df = _synthetic_df(2000)
        result = run_permutation_importance(
            df, FEATURE_COLS, "log_return", n_states=2,
            train_bars=600, test_bars=200,
            n_permutations=2,
            train_kwargs=FAST_KW,
        )
        assert set(result.is_stable.keys()) == set(FEATURE_COLS)

    def test_empty_result_on_insufficient_data(self):
        df = _synthetic_df(100)
        result = run_permutation_importance(
            df, FEATURE_COLS, "log_return", n_states=2,
            train_bars=600, test_bars=200,
            n_permutations=2,
            train_kwargs=FAST_KW,
        )
        assert len(result.fold_importances) == 0
        assert all(not v for v in result.is_stable.values())
