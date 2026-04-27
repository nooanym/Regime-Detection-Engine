"""Tests for Phase 2 evaluation modules: stability and walk-forward harness."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from rde.evaluation.stability import stability_across_restarts
from rde.evaluation.walk_forward import WalkForwardHarness
from rde.features.pipeline import FeaturePipeline
from rde.features.returns import LogReturns, SmoothedReturns
from rde.features.volatility import RollingVolatility


# ── helpers ────────────────────────────────────────────────────────────────────

def _synthetic_df(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 30_000.0 + np.cumsum(rng.normal(0, 150, n))
    close = np.maximum(close, 1.0)
    return pd.DataFrame(
        {
            "Open": close - rng.uniform(0, 80, n),
            "High": close + rng.uniform(0, 150, n),
            "Low": close - rng.uniform(0, 150, n),
            "Close": close,
            "Volume": rng.integers(1_000, 100_000, n).astype(float),
        },
        index=dates,
    )


def _feature_df(n: int = 800):
    df = _synthetic_df(n)
    pipeline = FeaturePipeline([LogReturns(), RollingVolatility(24), SmoothedReturns(12)])
    return pipeline.transform(df), ["log_return", "volatility_w24", "smoothed_return_w12"]


# ── stability tests ────────────────────────────────────────────────────────────

class TestStabilityAcrossRestarts:
    def test_returns_dict_with_required_keys(self):
        df_feat, feat_cols = _feature_df(500)
        X = df_feat[feat_cols].values
        result = stability_across_restarts(X, n_states=2, n_runs=3, n_restarts=2, n_iter=50)
        assert set(result.keys()) >= {
            "ari_matrix", "mean_ari", "min_ari",
            "transmat_std", "means_std", "models", "state_sequences",
        }

    def test_ari_matrix_shape(self):
        df_feat, feat_cols = _feature_df(500)
        X = df_feat[feat_cols].values
        result = stability_across_restarts(X, n_states=2, n_runs=3, n_restarts=2, n_iter=50)
        assert result["ari_matrix"].shape == (3, 3)

    def test_ari_matrix_diagonal_is_one(self):
        df_feat, feat_cols = _feature_df(500)
        X = df_feat[feat_cols].values
        result = stability_across_restarts(X, n_states=2, n_runs=3, n_restarts=2, n_iter=50)
        np.testing.assert_allclose(np.diag(result["ari_matrix"]), 1.0, atol=1e-6)

    def test_ari_matrix_symmetric(self):
        df_feat, feat_cols = _feature_df(500)
        X = df_feat[feat_cols].values
        result = stability_across_restarts(X, n_states=2, n_runs=3, n_restarts=2, n_iter=50)
        mat = result["ari_matrix"]
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)

    def test_mean_ari_in_valid_range(self):
        df_feat, feat_cols = _feature_df(500)
        X = df_feat[feat_cols].values
        result = stability_across_restarts(X, n_states=2, n_runs=2, n_restarts=2, n_iter=50)
        assert -1.0 <= result["mean_ari"] <= 1.0

    def test_returns_correct_number_of_models(self):
        df_feat, feat_cols = _feature_df(500)
        X = df_feat[feat_cols].values
        result = stability_across_restarts(X, n_states=2, n_runs=4, n_restarts=2, n_iter=50)
        assert len(result["models"]) == 4
        assert len(result["state_sequences"]) == 4

    def test_state_sequences_shape_and_range(self):
        df_feat, feat_cols = _feature_df(500)
        X = df_feat[feat_cols].values
        result = stability_across_restarts(X, n_states=2, n_runs=2, n_restarts=2, n_iter=50)
        for seq in result["state_sequences"]:
            assert seq.shape == (len(X),)
            assert set(seq).issubset({0, 1})

    def test_transmat_std_non_negative(self):
        df_feat, feat_cols = _feature_df(500)
        X = df_feat[feat_cols].values
        result = stability_across_restarts(X, n_states=2, n_runs=2, n_restarts=2, n_iter=50)
        assert result["transmat_std"] >= 0.0

    def test_single_run_ari_is_one(self):
        """With n_runs=1 there are no pairs; mean_ari and min_ari default to 1.0."""
        df_feat, feat_cols = _feature_df(500)
        X = df_feat[feat_cols].values
        result = stability_across_restarts(X, n_states=2, n_runs=1, n_restarts=2, n_iter=50)
        assert result["mean_ari"] == pytest.approx(1.0)


# ── walk-forward tests ─────────────────────────────────────────────────────────

class TestWalkForwardHarness:
    def test_init_monthly(self):
        h = WalkForwardHarness(recalibration="monthly", train_window="30d")
        assert h._recalibration == "monthly"
        assert h._train_window_days == 30

    def test_init_invalid_recalibration(self):
        with pytest.raises(ValueError, match="monthly"):
            WalkForwardHarness(recalibration="weekly")

    def test_run_returns_dataframe_with_same_index(self):
        df_feat, feat_cols = _feature_df(800)
        h = WalkForwardHarness(recalibration="monthly", train_window="14d")
        result = h.run(df_feat, n_states=2, feature_cols=feat_cols,
                       n_restarts=2, n_iter=50, seed_base=0)
        pd.testing.assert_index_equal(result.index, df_feat.index)

    def test_run_result_has_required_columns(self):
        df_feat, feat_cols = _feature_df(800)
        h = WalkForwardHarness(recalibration="monthly", train_window="14d")
        result = h.run(df_feat, n_states=2, feature_cols=feat_cols,
                       n_restarts=2, n_iter=50, seed_base=0)
        assert "regime" in result.columns
        assert "regime_proba_0" in result.columns
        assert "regime_proba_1" in result.columns

    def test_warmup_bars_are_nan(self):
        """The first train_window of bars must have NaN labels (no model trained before them)."""
        df_feat, feat_cols = _feature_df(800)
        train_days = 14
        h = WalkForwardHarness(recalibration="monthly", train_window=f"{train_days}d")
        result = h.run(df_feat, n_states=2, feature_cols=feat_cols,
                       n_restarts=2, n_iter=50, seed_base=0)
        warmup_end = df_feat.index[0] + pd.Timedelta(days=train_days)
        warmup_mask = df_feat.index < warmup_end
        assert result.loc[warmup_mask, "regime"].isna().all()

    def test_oos_bars_are_labeled(self):
        """At least some bars after the first boundary must be labelled."""
        df_feat, feat_cols = _feature_df(800)
        h = WalkForwardHarness(recalibration="monthly", train_window="14d")
        result = h.run(df_feat, n_states=2, feature_cols=feat_cols,
                       n_restarts=2, n_iter=50, seed_base=0)
        assert result["regime"].notna().sum() > 0

    def test_no_future_leakage_invariant(self):
        """For every labelled bar t, the first labelled bar's boundary must be
        strictly after all training data preceding t. We verify this structurally
        by checking that warm-up bars are NaN — a proxy for the leakage invariant."""
        df_feat, feat_cols = _feature_df(800)
        train_days = 14
        h = WalkForwardHarness(recalibration="monthly", train_window=f"{train_days}d")
        result = h.run(df_feat, n_states=2, feature_cols=feat_cols,
                       n_restarts=2, n_iter=50, seed_base=0)
        first_labeled_t = result["regime"].first_valid_index()
        assert first_labeled_t > df_feat.index[0] + pd.Timedelta(days=train_days - 1)

    def test_oos_regime_values_in_range(self):
        """All non-NaN regime labels must be valid state indices."""
        df_feat, feat_cols = _feature_df(800)
        h = WalkForwardHarness(recalibration="monthly", train_window="14d")
        result = h.run(df_feat, n_states=2, feature_cols=feat_cols,
                       n_restarts=2, n_iter=50, seed_base=0)
        labeled = result["regime"].dropna()
        assert set(labeled.astype(int)).issubset({0, 1})

    def test_oos_posteriors_sum_to_one(self):
        """Posterior probabilities for labelled bars must sum to 1 per bar."""
        df_feat, feat_cols = _feature_df(800)
        h = WalkForwardHarness(recalibration="monthly", train_window="14d")
        result = h.run(df_feat, n_states=2, feature_cols=feat_cols,
                       n_restarts=2, n_iter=50, seed_base=0)
        labeled_mask = result["regime"].notna()
        proba_cols = ["regime_proba_0", "regime_proba_1"]
        row_sums = result.loc[labeled_mask, proba_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-5)

    def test_train_window_too_long_raises(self):
        """Raise ValueError if train_window exceeds the data span."""
        df_feat, feat_cols = _feature_df(100)  # ~100 hours ≈ 4 days
        h = WalkForwardHarness(recalibration="monthly", train_window="30d")
        with pytest.raises(ValueError, match="train_window"):
            h.run(df_feat, n_states=2, feature_cols=feat_cols, n_restarts=1, n_iter=20)

    def test_requires_datetime_index(self):
        """Raise ValueError if df_features lacks a DatetimeIndex."""
        df_feat, feat_cols = _feature_df(200)
        df_no_dt = df_feat.reset_index(drop=True)
        h = WalkForwardHarness(recalibration="monthly", train_window="7d")
        with pytest.raises(ValueError, match="DatetimeIndex"):
            h.run(df_no_dt, n_states=2, feature_cols=feat_cols)
