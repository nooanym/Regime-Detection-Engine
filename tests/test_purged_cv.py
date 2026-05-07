"""Tests for rde.evaluation.purged_cv.

Fast unit tests cover the split generators and FoldResult serialisation.
The slow integration test runs a full purged-CV pass on synthetic data.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from rde.evaluation.purged_cv import (
    FoldResult,
    combinatorial_purged_splits,
    purged_k_fold_splits,
    run_purged_cv,
    run_combinatorial_purged_cv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_df(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """3-feature synthetic DataFrame matching the FeaturePipeline output schema."""
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


# ---------------------------------------------------------------------------
# purged_k_fold_splits
# ---------------------------------------------------------------------------


class TestPurgedKFoldSplits:
    def test_basic_structure(self):
        folds = list(purged_k_fold_splits(1000, train_bars=500, test_bars=100))
        assert len(folds) > 0
        for train, test in folds:
            assert len(train) == 500
            assert len(test) == 100

    def test_embargo_gap(self):
        embargo = 24
        folds = list(purged_k_fold_splits(1000, train_bars=500, test_bars=100, embargo_bars=embargo))
        for train, test in folds:
            # Last train bar + embargo_bars must be < first test bar
            assert train[-1] + embargo < test[0]

    def test_no_overlap(self):
        folds = list(purged_k_fold_splits(1000, train_bars=500, test_bars=100))
        for train, test in folds:
            assert len(np.intersect1d(train, test)) == 0

    def test_advances_by_test_bars(self):
        folds = list(purged_k_fold_splits(2000, train_bars=600, test_bars=200))
        assert len(folds) >= 2
        # Second fold's train starts 200 bars after first fold's train start
        assert folds[1][0][0] == folds[0][0][0] + 200

    def test_stops_before_overflow(self):
        folds = list(purged_k_fold_splits(100, train_bars=60, test_bars=30))
        for _, test in folds:
            assert test[-1] < 100


# ---------------------------------------------------------------------------
# combinatorial_purged_splits
# ---------------------------------------------------------------------------


class TestCombinatorialPurgedSplits:
    def test_n_combinations(self):
        n_groups, n_test = 5, 2
        folds = list(combinatorial_purged_splits(1000, n_groups=n_groups, n_test_groups=n_test))
        expected = len(list(itertools.combinations(range(n_groups), n_test)))
        assert len(folds) == expected

    def test_no_overlap_between_train_and_test(self):
        for train, test, _ in combinatorial_purged_splits(1000, n_groups=5, n_test_groups=2):
            assert len(np.intersect1d(train, test)) == 0

    def test_test_group_ids_cover_all_combinations(self):
        n_groups, n_test = 4, 2
        seen = set()
        for _, _, groups in combinatorial_purged_splits(800, n_groups=n_groups, n_test_groups=n_test):
            seen.add(groups)
        expected = set(itertools.combinations(range(n_groups), n_test))
        assert seen == expected

    def test_embargo_applied(self):
        embargo = 10
        for train, test, _ in combinatorial_purged_splits(
            500, n_groups=4, n_test_groups=1, embargo_bars=embargo
        ):
            # No training bar should be within embargo_bars of any test bar start
            test_starts = []
            # Infer test group starts as points where test index jumps
            if len(test) > 1:
                gaps = np.diff(test)
                test_starts = [test[0]] + list(test[1:][gaps > 1])
            else:
                test_starts = [test[0]]
            for ts in test_starts:
                for forbidden in range(max(0, ts - embargo), ts):
                    assert forbidden not in set(train), (
                        f"Bar {forbidden} is within embargo of test start {ts} but in train"
                    )


# ---------------------------------------------------------------------------
# FoldResult
# ---------------------------------------------------------------------------


class TestFoldResult:
    def _make(self, fold_id: int = 0) -> FoldResult:
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        te = pd.Timestamp("2024-03-01", tz="UTC")
        return FoldResult(
            fold_id=fold_id,
            cv_type="purged_kfold",
            train_start=ts, train_end=te,
            test_start=te, test_end=pd.Timestamp("2024-04-01", tz="UTC"),
            n_train=1000, n_test=200,
            sharpe=0.8, calmar=1.2, max_drawdown=0.05,
            hit_rate=0.52, ann_return=0.12, ann_vol=0.15,
            n_trades=10, bic=5000.0, log_likelihood=-2400.0,
            n_params=60, n_states_used=4,
            regime_conditional_returns={0: -0.001, 1: 0.0005, 2: 0.001, 3: 0.002},
        )

    def test_to_dict_keys(self):
        fr = self._make()
        d = fr.to_dict()
        assert "sharpe" in d
        assert "regime_0_mean_return" in d
        assert "regime_3_mean_return" in d
        assert "transmat_frobenius_from_mean" in d

    def test_to_dict_values(self):
        fr = self._make()
        d = fr.to_dict()
        assert d["fold_id"] == 0
        assert d["sharpe"] == pytest.approx(0.8)
        assert d["regime_2_mean_return"] == pytest.approx(0.001)

    def test_default_dispersion_is_nan(self):
        fr = self._make()
        assert np.isnan(fr.transmat_frobenius_from_mean)
        assert np.isnan(fr.means_frobenius_from_mean)


# ---------------------------------------------------------------------------
# run_purged_cv — fast (small synthetic data, 1 restart)
# ---------------------------------------------------------------------------


class TestRunPurgedCv:
    def test_returns_fold_list_and_dataframe(self):
        df = _synthetic_df(2000)
        folds, summary = run_purged_cv(
            df, FEATURE_COLS, "log_return", n_states=2,
            train_bars=600, test_bars=200, embargo_bars=24,
            ann_factor=8760,
            train_kwargs={"n_restarts": 1, "n_iter": 20, "seed_base": 0},
        )
        assert len(folds) > 0
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == len(folds)

    def test_summary_has_required_columns(self):
        df = _synthetic_df(2000)
        _, summary = run_purged_cv(
            df, FEATURE_COLS, "log_return", n_states=2,
            train_bars=600, test_bars=200,
            train_kwargs={"n_restarts": 1, "n_iter": 20, "seed_base": 0},
        )
        for col in ["sharpe", "calmar", "max_drawdown", "bic", "n_states_used"]:
            assert col in summary.columns, f"Missing column: {col}"

    def test_fold_metrics_finite(self):
        df = _synthetic_df(2000)
        folds, _ = run_purged_cv(
            df, FEATURE_COLS, "log_return", n_states=2,
            train_bars=600, test_bars=200,
            train_kwargs={"n_restarts": 1, "n_iter": 20, "seed_base": 0},
        )
        for fr in folds:
            assert np.isfinite(fr.sharpe), f"Fold {fr.fold_id}: Sharpe not finite"
            assert np.isfinite(fr.bic), f"Fold {fr.fold_id}: BIC not finite"
            assert 0.0 <= fr.max_drawdown <= 1.0, f"Fold {fr.fold_id}: drawdown out of range"

    def test_dispersion_filled_after_multiple_folds(self):
        df = _synthetic_df(3000)
        folds, _ = run_purged_cv(
            df, FEATURE_COLS, "log_return", n_states=2,
            train_bars=700, test_bars=200,
            train_kwargs={"n_restarts": 1, "n_iter": 20, "seed_base": 0},
        )
        if len(folds) >= 2:
            for fr in folds:
                assert np.isfinite(fr.transmat_frobenius_from_mean)
                assert np.isfinite(fr.means_frobenius_from_mean)

    def test_raises_on_insufficient_data(self):
        df = _synthetic_df(100)
        with pytest.raises(ValueError, match="at least"):
            run_purged_cv(
                df, FEATURE_COLS, "log_return", n_states=2,
                train_bars=600, test_bars=200,
                train_kwargs={"n_restarts": 1, "n_iter": 5},
            )

    def test_saves_parquet(self, tmp_path):
        df = _synthetic_df(2000)
        run_purged_cv(
            df, FEATURE_COLS, "log_return", n_states=2,
            train_bars=600, test_bars=200,
            train_kwargs={"n_restarts": 1, "n_iter": 20, "seed_base": 0},
            results_dir=tmp_path,
        )
        parquets = list(tmp_path.glob("purged_cv_*.parquet"))
        assert len(parquets) == 1
        loaded = pd.read_parquet(parquets[0])
        assert "sharpe" in loaded.columns


# ---------------------------------------------------------------------------
# run_combinatorial_purged_cv — fast
# ---------------------------------------------------------------------------


class TestRunCombinatorialPurgedCv:
    def test_n_folds_matches_combinations(self):
        df = _synthetic_df(2400)
        n_groups, n_test = 4, 2
        folds, _ = run_combinatorial_purged_cv(
            df, FEATURE_COLS, "log_return", n_states=2,
            n_groups=n_groups, n_test_groups=n_test,
            train_kwargs={"n_restarts": 1, "n_iter": 20, "seed_base": 0},
        )
        expected = len(list(itertools.combinations(range(n_groups), n_test)))
        assert len(folds) == expected

    def test_cv_type_label(self):
        df = _synthetic_df(2400)
        folds, _ = run_combinatorial_purged_cv(
            df, FEATURE_COLS, "log_return", n_states=2,
            n_groups=4, n_test_groups=2,
            train_kwargs={"n_restarts": 1, "n_iter": 20, "seed_base": 0},
        )
        assert all(fr.cv_type == "combinatorial_purged" for fr in folds)

    def test_sharpe_distribution(self):
        """Combinatorial CV yields a distribution of Sharpe values, not all identical."""
        df = _synthetic_df(3000)
        folds, _ = run_combinatorial_purged_cv(
            df, FEATURE_COLS, "log_return", n_states=2,
            n_groups=4, n_test_groups=2,
            train_kwargs={"n_restarts": 1, "n_iter": 30, "seed_base": 0},
        )
        sharpes = [fr.sharpe for fr in folds if np.isfinite(fr.sharpe)]
        assert len(sharpes) >= 2
        # Not all identical (different test windows → different Sharpe)
        assert not all(s == sharpes[0] for s in sharpes)


# ---------------------------------------------------------------------------
# Slow integration test — realistic data size
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_purged_cv_on_realistic_data():
    """Full purged CV on 17k-bar synthetic data (matching BTC size)."""
    df = _synthetic_df(17_000, seed=99)
    folds, summary = run_purged_cv(
        df, FEATURE_COLS, "log_return", n_states=4,
        train_bars=4000, test_bars=500, embargo_bars=24,
        ann_factor=8760,
        train_kwargs={"n_restarts": 3, "n_iter": 100, "seed_base": 42},
    )
    assert len(folds) >= 4
    sharpes = [fr.sharpe for fr in folds]
    # Sharpe should be finite for all folds
    assert all(np.isfinite(s) for s in sharpes)
    # Summary parquet schema
    for col in ["sharpe", "calmar", "bic", "transmat_frobenius_from_mean"]:
        assert col in summary.columns
