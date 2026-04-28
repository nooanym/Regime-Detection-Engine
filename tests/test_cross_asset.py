"""Tests for rde.analysis.cross_asset (Phase 13)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rde.analysis.cross_asset import (
    AssetRegimeData,
    CrossAssetResult,
    _sharpe_scores,
    compute_cross_asset,
    load_asset_regime_data,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_analytics(n_states: int, sharpes: list[float] | None = None) -> pd.DataFrame:
    if sharpes is None:
        sharpes = list(range(n_states))
    return pd.DataFrame({
        "state": list(range(n_states)),
        "label": [f"S{i}" for i in range(n_states)],
        "sharpe_ann": sharpes,
        "mean_return_ann": [s * 0.1 for s in sharpes],
        "vol_ann": [0.2] * n_states,
        "max_drawdown": [0.05] * n_states,
        "skewness": [0.0] * n_states,
        "excess_kurtosis": [0.0] * n_states,
        "var_95": [-0.01] * n_states,
        "cvar_95": [-0.02] * n_states,
        "win_rate": [0.5] * n_states,
        "count": [100] * n_states,
        "weight": [1 / n_states] * n_states,
    })


def _make_regimes_df(n: int = 500, n_states: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "log_return": rng.normal(0.0, 0.01, n),
        "regime": rng.integers(0, n_states, n),
        "regime_label": [f"S{rng.integers(0, n_states)}" for _ in range(n)],
        "Close": np.cumsum(rng.normal(0, 1, n)) + 100,
    }, index=idx)


def _write_asset_results(tmp_path: Path, symbol: str, n: int = 500, seed: int = 0) -> Path:
    result_dir = tmp_path / symbol
    result_dir.mkdir()
    reg_df = _make_regimes_df(n=n, seed=seed)
    reg_df.to_parquet(result_dir / "regimes.parquet")
    ana_df = _make_analytics(3)
    ana_df.to_parquet(result_dir / "regime_analytics.parquet")
    return result_dir


def _make_asset_data(symbol: str, n: int = 500, seed: int = 0) -> AssetRegimeData:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    returns = pd.Series(rng.normal(0, 0.01, n), index=idx, name=symbol)
    score = pd.Series(rng.uniform(-1, 1, n), index=idx, name=symbol)
    label = pd.Series([f"S{i % 3}" for i in range(n)], index=idx, name=symbol)
    return AssetRegimeData(
        symbol=symbol, returns=returns, regime_score=score, regime_label=label
    )


# ── _sharpe_scores ────────────────────────────────────────────────────────────


def test_sharpe_scores_shape():
    ana = _make_analytics(4)
    scores = _sharpe_scores(ana)
    assert scores.shape == (4,)


def test_sharpe_scores_single_state():
    ana = _make_analytics(1, sharpes=[2.5])
    scores = _sharpe_scores(ana)
    assert scores[0] == pytest.approx(0.0)


def test_sharpe_scores_range():
    ana = _make_analytics(5)
    scores = _sharpe_scores(ana)
    assert scores.min() == pytest.approx(-1.0)
    assert scores.max() == pytest.approx(1.0)


def test_sharpe_scores_ordering():
    ana = _make_analytics(3, sharpes=[-1.0, 0.0, 2.0])
    scores = _sharpe_scores(ana)
    assert scores[0] < scores[1] < scores[2]


def test_sharpe_scores_empty():
    ana = pd.DataFrame(columns=["state", "label", "sharpe_ann"])
    scores = _sharpe_scores(ana)
    assert scores.shape == (0,)


def test_sharpe_scores_nan_filled_with_zero():
    ana = _make_analytics(3, sharpes=[float("nan"), 0.0, 1.0])
    scores = _sharpe_scores(ana)
    assert np.all(np.isfinite(scores))


# ── load_asset_regime_data ────────────────────────────────────────────────────


def test_load_returns_asset_regime_data(tmp_path):
    rd = _write_asset_results(tmp_path, "BTC-USD")
    result = load_asset_regime_data(rd)
    assert isinstance(result, AssetRegimeData)
    assert result.symbol == "BTC-USD"


def test_load_series_lengths_match(tmp_path):
    rd = _write_asset_results(tmp_path, "ETH-USD", n=200)
    d = load_asset_regime_data(rd)
    assert len(d.returns) == len(d.regime_score) == len(d.regime_label)


def test_load_regime_score_in_range(tmp_path):
    rd = _write_asset_results(tmp_path, "SPY")
    d = load_asset_regime_data(rd)
    assert d.regime_score.min() >= -1.0 - 1e-9
    assert d.regime_score.max() <= 1.0 + 1e-9


def test_load_missing_regimes_raises(tmp_path):
    rd = tmp_path / "X"
    rd.mkdir()
    _make_analytics(3).to_parquet(rd / "regime_analytics.parquet")
    with pytest.raises(FileNotFoundError, match="regimes.parquet"):
        load_asset_regime_data(rd)


def test_load_missing_analytics_raises(tmp_path):
    rd = tmp_path / "X"
    rd.mkdir()
    _make_regimes_df().to_parquet(rd / "regimes.parquet")
    with pytest.raises(FileNotFoundError, match="regime_analytics.parquet"):
        load_asset_regime_data(rd)


def test_load_missing_log_return_raises(tmp_path):
    rd = tmp_path / "X"
    rd.mkdir()
    df = _make_regimes_df().drop(columns=["log_return"])
    df.to_parquet(rd / "regimes.parquet")
    _make_analytics(3).to_parquet(rd / "regime_analytics.parquet")
    with pytest.raises(KeyError, match="log_return"):
        load_asset_regime_data(rd)


# ── compute_cross_asset ───────────────────────────────────────────────────────


def test_compute_returns_cross_asset_result():
    data = [_make_asset_data("A"), _make_asset_data("B")]
    result = compute_cross_asset(data)
    assert isinstance(result, CrossAssetResult)


def test_compute_requires_two_assets():
    with pytest.raises(ValueError, match="2 assets"):
        compute_cross_asset([_make_asset_data("A")])


def test_compute_assets_list():
    data = [_make_asset_data("BTC"), _make_asset_data("ETH"), _make_asset_data("SPY")]
    result = compute_cross_asset(data)
    assert result.assets == ["BTC", "ETH", "SPY"]


def test_compute_n_common_bars():
    data = [_make_asset_data("A", n=500), _make_asset_data("B", n=500)]
    result = compute_cross_asset(data)
    assert result.n_common_bars == 500


def test_compute_partial_overlap():
    idx_a = pd.date_range("2024-01-01", periods=400, freq="h", tz="UTC")
    idx_b = pd.date_range("2024-01-10", periods=400, freq="h", tz="UTC")
    a = AssetRegimeData(
        "A",
        returns=pd.Series(np.zeros(400), index=idx_a, name="A"),
        regime_score=pd.Series(np.zeros(400), index=idx_a, name="A"),
        regime_label=pd.Series(["x"] * 400, index=idx_a, name="A"),
    )
    b = AssetRegimeData(
        "B",
        returns=pd.Series(np.zeros(400), index=idx_b, name="B"),
        regime_score=pd.Series(np.zeros(400), index=idx_b, name="B"),
        regime_label=pd.Series(["x"] * 400, index=idx_b, name="B"),
    )
    result = compute_cross_asset([a, b])
    assert result.n_common_bars < 400


def test_compute_empty_overlap_raises():
    idx_a = pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC")
    idx_b = pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC")
    a = AssetRegimeData("A", pd.Series(np.zeros(100), index=idx_a, name="A"),
                        pd.Series(np.zeros(100), index=idx_a, name="A"),
                        pd.Series(["x"] * 100, index=idx_a, name="A"))
    b = AssetRegimeData("B", pd.Series(np.zeros(100), index=idx_b, name="B"),
                        pd.Series(np.zeros(100), index=idx_b, name="B"),
                        pd.Series(["x"] * 100, index=idx_b, name="B"))
    with pytest.raises(ValueError, match="No overlapping"):
        compute_cross_asset([a, b])


def test_return_corr_shape():
    data = [_make_asset_data(s) for s in ["A", "B", "C"]]
    result = compute_cross_asset(data)
    assert result.return_corr.shape == (3, 3)


def test_return_corr_diagonal_ones():
    data = [_make_asset_data(s) for s in ["A", "B"]]
    result = compute_cross_asset(data)
    np.testing.assert_allclose(np.diag(result.return_corr.values), 1.0)


def test_score_corr_symmetric():
    data = [_make_asset_data(s) for s in ["A", "B", "C"]]
    result = compute_cross_asset(data)
    mat = result.score_corr.values
    np.testing.assert_allclose(mat, mat.T, atol=1e-12)


def test_cond_return_corr_keys():
    data = [_make_asset_data(s) for s in ["A", "B"]]
    result = compute_cross_asset(data)
    assert set(result.cond_return_corr.keys()) == {
        "bull_bull", "bull_bear", "bear_bull", "bear_bear"
    }


def test_cond_return_corr_matrix_shape():
    data = [_make_asset_data(s) for s in ["A", "B", "C"]]
    result = compute_cross_asset(data)
    for mat in result.cond_return_corr.values():
        assert mat.shape == (3, 3)


def test_co_occurrence_keys_are_pairs():
    data = [_make_asset_data(s) for s in ["A", "B", "C"]]
    result = compute_cross_asset(data)
    expected_pairs = {("A", "B"), ("A", "C"), ("B", "C")}
    assert set(result.co_occurrence.keys()) == expected_pairs


def test_co_occurrence_counts_sum_to_n():
    data = [_make_asset_data("A"), _make_asset_data("B")]
    result = compute_cross_asset(data)
    counts = result.co_occurrence[("A", "B")]
    assert int(counts.values.sum()) == result.n_common_bars


def test_co_occurrence_shape():
    data = [_make_asset_data("A"), _make_asset_data("B")]
    result = compute_cross_asset(data)
    assert result.co_occurrence[("A", "B")].shape == (2, 2)


def test_aligned_returns_shape():
    data = [_make_asset_data(s) for s in ["A", "B"]]
    result = compute_cross_asset(data)
    assert result.aligned_returns.shape[1] == 2
    assert len(result.aligned_returns) == result.n_common_bars


def test_aligned_scores_shape():
    data = [_make_asset_data(s) for s in ["A", "B", "C"]]
    result = compute_cross_asset(data)
    assert result.aligned_scores.shape == (result.n_common_bars, 3)


# ── load + compute round-trip ─────────────────────────────────────────────────


def test_roundtrip_from_disk(tmp_path):
    rd_a = _write_asset_results(tmp_path, "BTC-USD", n=300, seed=0)
    rd_b = _write_asset_results(tmp_path, "ETH-USD", n=300, seed=1)
    data = [load_asset_regime_data(rd_a), load_asset_regime_data(rd_b)]
    result = compute_cross_asset(data)
    assert result.n_common_bars == 300
    assert result.return_corr.shape == (2, 2)
