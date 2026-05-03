"""Tests for rde.analysis.regime_concordance (Phase 34)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.analysis.regime_concordance import (
    ConcordanceConfig,
    ConcordanceResult,
    PairwiseConcordance,
    compute_concordance,
    concordance_heatmap_data,
    rolling_concordance_series,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scores(n: int, seed: int = 0, n_assets: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    data = {f"A{i}": rng.uniform(-1, 1, size=n) for i in range(n_assets)}
    return pd.DataFrame(data, index=idx)


def _make_correlated(n: int, correlation: float) -> pd.DataFrame:
    """Two assets with known correlation structure."""
    rng = np.random.default_rng(42)
    a = rng.uniform(-1, 1, size=n)
    if correlation == 1.0:
        b = a.copy()
    elif correlation == -1.0:
        b = -a.copy()
    else:
        noise = rng.uniform(-1, 1, size=n)
        b = correlation * a + (1 - abs(correlation)) * noise
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"A": a, "B": b}, index=idx)


# ---------------------------------------------------------------------------
# ConcordanceConfig
# ---------------------------------------------------------------------------

class TestConcordanceConfig:
    def test_defaults(self):
        cfg = ConcordanceConfig()
        assert cfg.min_common_bars == 30
        assert cfg.rolling_window == 20

    def test_custom(self):
        cfg = ConcordanceConfig(min_common_bars=10, rolling_window=5)
        assert cfg.min_common_bars == 10
        assert cfg.rolling_window == 5


# ---------------------------------------------------------------------------
# compute_concordance — basic correctness
# ---------------------------------------------------------------------------

class TestComputeConcordance:
    def test_returns_concordance_result(self):
        df = _make_scores(50)
        result = compute_concordance(df)
        assert isinstance(result, ConcordanceResult)

    def test_perfect_correlation_agreement(self):
        # When both assets always have the same sign, agreement = 1.0
        df = _make_correlated(100, correlation=1.0)
        result = compute_concordance(df)
        assert result.pairwise[0].direction_agreement == pytest.approx(1.0)

    def test_perfect_anti_correlation_agreement(self):
        # When signs are always opposite, agreement = 0.0
        df = _make_correlated(100, correlation=-1.0)
        result = compute_concordance(df)
        assert result.pairwise[0].direction_agreement == pytest.approx(0.0)

    def test_random_agreement_near_half(self):
        # Independent assets agree ~50% of the time by chance
        rng = np.random.default_rng(42)
        n = 10_000
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame(
            {"A": rng.standard_normal(n), "B": rng.standard_normal(n)},
            index=idx,
        )
        result = compute_concordance(df, ConcordanceConfig(min_common_bars=100))
        agr = result.pairwise[0].direction_agreement
        assert 0.4 < agr < 0.6

    def test_score_correlation_matches_numpy(self):
        df = _make_scores(100, seed=5)
        result = compute_concordance(df)
        expected = float(np.corrcoef(df["A0"].values, df["A1"].values)[0, 1])
        assert result.pairwise[0].score_correlation == pytest.approx(expected, abs=1e-6)

    def test_rolling_agreement_length(self):
        df = _make_scores(100)
        result = compute_concordance(df)
        ra = result.pairwise[0].rolling_agreement
        assert len(ra) == 100

    def test_rolling_agreement_first_values_nan(self):
        cfg = ConcordanceConfig(rolling_window=10)
        df = _make_scores(50)
        result = compute_concordance(df, cfg)
        ra = result.pairwise[0].rolling_agreement
        assert ra.iloc[:9].isna().all()
        assert not ra.iloc[9:].isna().any()

    def test_lead_lag_in_range(self):
        df = _make_scores(200)
        result = compute_concordance(df)
        assert -10 <= result.pairwise[0].lead_lag_days <= 10

    def test_lead_lag_detects_known_shift(self):
        # b[t] = raw[t+3], a[t] = raw[t]  →  a[t] = b[t-3]  →  b leads a
        # Implementation convention: positive lag = b leads a → expect +3
        n = 500
        rng = np.random.default_rng(0)
        raw = (rng.standard_normal(n + 10) > 0).astype(float) * 2 - 1
        a = raw[0: n]
        b = raw[3: n + 3]
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame({"A": a, "B": b}, index=idx)
        result = compute_concordance(df)
        assert result.pairwise[0].lead_lag_days == 3

    def test_global_agreement_is_mean_of_pairwise(self):
        df = _make_scores(50, n_assets=3)
        result = compute_concordance(df)
        expected = np.mean([p.direction_agreement for p in result.pairwise])
        assert result.global_agreement == pytest.approx(expected, abs=1e-9)

    def test_sync_matrix_symmetric(self):
        df = _make_scores(50, n_assets=3)
        result = compute_concordance(df)
        mat = result.regime_sync_matrix
        assert mat.shape == (3, 3)
        for i in mat.index:
            for j in mat.columns:
                assert mat.loc[i, j] == pytest.approx(mat.loc[j, i], abs=1e-9)

    def test_sync_matrix_diagonal_ones(self):
        df = _make_scores(50, n_assets=3)
        result = compute_concordance(df)
        mat = result.regime_sync_matrix
        for sym in mat.index:
            assert mat.loc[sym, sym] == pytest.approx(1.0)

    def test_raises_fewer_than_2_assets(self):
        df = _make_scores(50, n_assets=1)
        with pytest.raises(ValueError, match="at least 2"):
            compute_concordance(df)

    def test_raises_too_few_bars(self):
        df = _make_scores(5, n_assets=2)
        cfg = ConcordanceConfig(min_common_bars=30)
        with pytest.raises(ValueError, match="at least 30"):
            compute_concordance(df, cfg)

    def test_three_asset_pairwise_count(self):
        df = _make_scores(100, n_assets=3)
        result = compute_concordance(df)
        assert len(result.pairwise) == 3  # C(3,2) = 3

    def test_n_common_bars_stored(self):
        df = _make_scores(75)
        result = compute_concordance(df)
        assert result.n_common_bars == 75


# ---------------------------------------------------------------------------
# concordance_heatmap_data
# ---------------------------------------------------------------------------

class TestConcordanceHeatmapData:
    def test_returns_dataframe(self):
        df = _make_scores(50)
        result = compute_concordance(df)
        hm = concordance_heatmap_data(result)
        assert isinstance(hm, pd.DataFrame)

    def test_values_in_0_1(self):
        df = _make_scores(50, n_assets=3)
        result = compute_concordance(df)
        hm = concordance_heatmap_data(result)
        assert (hm.values >= 0).all()
        assert (hm.values <= 1).all()

    def test_is_copy_not_view(self):
        df = _make_scores(50)
        result = compute_concordance(df)
        hm = concordance_heatmap_data(result)
        hm.iloc[0, 0] = -99.0
        assert result.regime_sync_matrix.iloc[0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# rolling_concordance_series
# ---------------------------------------------------------------------------

class TestRollingConcordanceSeries:
    def test_column_count_two_assets(self):
        df = _make_scores(50)
        out = rolling_concordance_series(df, window=5)
        assert out.shape[1] == 1  # C(2,2) = 1

    def test_column_count_three_assets(self):
        df = _make_scores(50, n_assets=3)
        out = rolling_concordance_series(df, window=5)
        assert out.shape[1] == 3  # C(3,2) = 3

    def test_column_names_contain_double_underscore(self):
        df = _make_scores(50, n_assets=3)
        out = rolling_concordance_series(df, window=5)
        for col in out.columns:
            assert "__" in col

    def test_initial_values_nan(self):
        df = _make_scores(50)
        out = rolling_concordance_series(df, window=10)
        col = out.columns[0]
        assert out[col].iloc[:9].isna().all()

    def test_non_nan_values_in_01(self):
        df = _make_scores(100)
        out = rolling_concordance_series(df, window=10)
        valid = out.dropna()
        assert (valid.values >= 0).all()
        assert (valid.values <= 1).all()
