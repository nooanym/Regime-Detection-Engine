"""Smoke test: runs the full Phase 1 pipeline on synthetic data (no yfinance calls).

Every test here uses a deterministic synthetic DataFrame so CI never hits the network.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from rde.config.schema import (
    AssetConfig,
    Config,
    FeatureConfig,
    ModelConfig,
    RunConfig,
    SelectionConfig,
)
from rde.evaluation.persistence import empirical_dwell_times, expected_dwell_times
from rde.evaluation.transition import stationary_distribution, transition_entropy
from rde.features.pipeline import FeaturePipeline
from rde.features.returns import LogReturns, SmoothedReturns
from rde.features.volatility import RollingVolatility
from rde.inference.forward import forward_log_likelihood
from rde.inference.smoothing import forward_backward_posteriors
from rde.inference.viterbi import viterbi_decode
from rde.labeling.ranking import HEURISTIC_DISCLAIMER, rank_states
from rde.models.hmm import train_hmm
from rde.models.selection import select_n_states
from rde.viz.static_plots import (
    plot_per_regime_returns,
    plot_price_with_regimes,
    plot_regime_timeline,
    plot_transition_heatmap,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _synthetic_df(n: int = 600) -> pd.DataFrame:
    """Generate a deterministic synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(0)
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


def _fitted_model(n_states: int = 2, n: int = 600):
    df = _synthetic_df(n)
    pipeline = FeaturePipeline([LogReturns(), RollingVolatility(24), SmoothedReturns(12)])
    df_feat = pipeline.transform(df)
    X = df_feat[["log_return", "volatility_w24", "smoothed_return_w12"]].values
    feature_names = ["log_return", "volatility_w24", "smoothed_return_w12"]
    fitted = train_hmm(
        X, n_states=n_states, n_restarts=3, n_iter=50, seed_base=0,
        feature_names=feature_names,
    )
    return fitted, df_feat, X


# ── feature tests ──────────────────────────────────────────────────────────────

def test_log_returns_produces_correct_column():
    df = _synthetic_df(100)
    result = LogReturns().transform(df)
    assert "log_return" in result.columns
    expected = np.log(df["Close"] / df["Close"].shift(1))
    pd.testing.assert_series_equal(result["log_return"], expected, check_names=False)


def test_rolling_volatility_column():
    df = _synthetic_df(100)
    df = LogReturns().transform(df)
    result = RollingVolatility(24).transform(df)
    assert "volatility_w24" in result.columns


def test_smoothed_returns_column():
    df = _synthetic_df(100)
    df = LogReturns().transform(df)
    result = SmoothedReturns(12).transform(df)
    assert "smoothed_return_w12" in result.columns


def test_pipeline_drops_nan_rows():
    df = _synthetic_df(200)
    pipeline = FeaturePipeline([LogReturns(), RollingVolatility(24), SmoothedReturns(12)])
    result = pipeline.transform(df)
    assert result.isna().sum().sum() == 0
    assert len(result) < len(df)  # leading NaNs dropped


def test_pipeline_output_columns():
    df = _synthetic_df(200)
    pipeline = FeaturePipeline([LogReturns(), RollingVolatility(24), SmoothedReturns(12)])
    result = pipeline.transform(df)
    for col in ["log_return", "volatility_w24", "smoothed_return_w12"]:
        assert col in result.columns


# ── model tests ────────────────────────────────────────────────────────────────

def test_train_hmm_returns_fitted_model():
    fitted, _, X = _fitted_model(n_states=2)
    assert fitted.n_states == 2
    assert fitted.log_likelihood < 0
    assert len(fitted.all_restart_scores) == 3
    assert fitted.seed in range(0, 3)


def test_train_hmm_aic_bic_finite():
    fitted, _, _ = _fitted_model(n_states=2)
    assert np.isfinite(fitted.aic)
    assert np.isfinite(fitted.bic)


def test_train_hmm_multi_restart_finds_best():
    """Best restart score must equal max of all scores."""
    fitted, _, _ = _fitted_model(n_states=2)
    valid_scores = [s for s in fitted.all_restart_scores if np.isfinite(s)]
    assert fitted.log_likelihood == pytest.approx(max(valid_scores), rel=1e-6)


def test_select_n_states_returns_dataframe():
    df = _synthetic_df()
    pipeline = FeaturePipeline([LogReturns(), RollingVolatility(24), SmoothedReturns(12)])
    df_feat = pipeline.transform(df)
    X = df_feat[["log_return", "volatility_w24", "smoothed_return_w12"]].values
    fitted, scores_df = select_n_states(
        X, candidate_states=[2, 3], criterion="bic",
        n_restarts=2, n_iter=30, seed_base=0,
    )
    assert fitted.n_states in (2, 3)
    assert set(scores_df.columns) >= {"n_states", "log_likelihood", "aic", "bic"}
    assert len(scores_df) == 2


def test_select_n_states_picks_lower_bic():
    """Selected model must have the minimum BIC among candidates."""
    df = _synthetic_df()
    pipeline = FeaturePipeline([LogReturns(), RollingVolatility(24), SmoothedReturns(12)])
    df_feat = pipeline.transform(df)
    X = df_feat[["log_return", "volatility_w24", "smoothed_return_w12"]].values
    fitted, scores_df = select_n_states(
        X, candidate_states=[2, 3], criterion="bic",
        n_restarts=2, n_iter=30, seed_base=0,
    )
    best_row = scores_df.loc[scores_df["bic"].idxmin()]
    assert fitted.n_states == int(best_row["n_states"])


# ── inference tests ────────────────────────────────────────────────────────────

def test_viterbi_shape_and_range():
    fitted, df_feat, X = _fitted_model(n_states=2)
    X_scaled = fitted.scaler.transform(X)
    states = viterbi_decode(fitted.hmm, X_scaled)
    assert states.shape == (len(X),)
    assert set(states).issubset(set(range(fitted.n_states)))


def test_forward_backward_shape_and_sums_to_one():
    fitted, df_feat, X = _fitted_model(n_states=2)
    X_scaled = fitted.scaler.transform(X)
    posteriors = forward_backward_posteriors(fitted.hmm, X_scaled)
    assert posteriors.shape == (len(X), 2)
    np.testing.assert_allclose(posteriors.sum(axis=1), 1.0, atol=1e-6)


def test_forward_log_likelihood_consistent():
    fitted, df_feat, X = _fitted_model(n_states=2)
    X_scaled = fitted.scaler.transform(X)
    ll = forward_log_likelihood(fitted.hmm, X_scaled)
    assert np.isfinite(ll)
    assert ll == pytest.approx(fitted.log_likelihood, rel=1e-5)


# ── evaluation tests ───────────────────────────────────────────────────────────

def test_stationary_distribution_sums_to_one():
    transmat = np.array([[0.9, 0.1], [0.2, 0.8]])
    stat = stationary_distribution(transmat)
    assert stat.shape == (2,)
    np.testing.assert_allclose(stat.sum(), 1.0, atol=1e-6)


def test_stationary_distribution_is_invariant():
    transmat = np.array([[0.9, 0.1], [0.2, 0.8]])
    stat = stationary_distribution(transmat)
    np.testing.assert_allclose(stat @ transmat, stat, atol=1e-6)


def test_expected_dwell_times_shape():
    transmat = np.array([[0.9, 0.1], [0.2, 0.8]])
    edt = expected_dwell_times(transmat)
    assert edt.shape == (2,)
    assert edt[0] == pytest.approx(10.0, rel=1e-4)
    assert edt[1] == pytest.approx(5.0, rel=1e-4)


def test_empirical_dwell_times_run_lengths():
    states = np.array([0, 0, 0, 1, 1, 0, 0])
    emp = empirical_dwell_times(states)
    np.testing.assert_array_equal(sorted(emp[0]), sorted([3, 2]))
    np.testing.assert_array_equal(emp[1], [2])


def test_transition_entropy_non_negative():
    transmat = np.array([[0.9, 0.1], [0.5, 0.5]])
    ent = transition_entropy(transmat)
    assert (ent >= 0).all()


# ── labeling tests ─────────────────────────────────────────────────────────────

def test_rank_states_returns_two_labels_for_two_states():
    fitted, _, _ = _fitted_model(n_states=2)
    labels = rank_states(fitted)
    assert len(labels) == 2
    label_names = {ls.label for ls in labels}
    assert label_names == {"bearish", "bullish"}


def test_rank_states_ranks_are_unique():
    fitted, _, _ = _fitted_model(n_states=2)
    labels = rank_states(fitted)
    return_ranks = [ls.rank_return for ls in labels]
    assert sorted(return_ranks) == list(range(len(labels)))


def test_heuristic_disclaimer_present():
    assert "heuristic" in HEURISTIC_DISCLAIMER.lower()
    assert "ground truth" in HEURISTIC_DISCLAIMER.lower()


# ── viz tests ──────────────────────────────────────────────────────────────────

def test_plot_price_with_regimes_returns_figure():
    fitted, df_feat, X = _fitted_model(n_states=2, n=300)
    X_scaled = fitted.scaler.transform(X)
    states = viterbi_decode(fitted.hmm, X_scaled)
    fig = plot_price_with_regimes(df_feat, states, symbol="TEST")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_transition_heatmap_returns_figure():
    fitted, _, _ = _fitted_model(n_states=2)
    fig = plot_transition_heatmap(fitted.hmm.transmat_, symbol="TEST")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_regime_timeline_returns_figure():
    fitted, df_feat, X = _fitted_model(n_states=2, n=300)
    X_scaled = fitted.scaler.transform(X)
    states = viterbi_decode(fitted.hmm, X_scaled)
    fig = plot_regime_timeline(df_feat, states, symbol="TEST")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_per_regime_returns_returns_figure():
    fitted, df_feat, X = _fitted_model(n_states=2, n=300)
    X_scaled = fitted.scaler.transform(X)
    states = viterbi_decode(fitted.hmm, X_scaled)
    fig = plot_per_regime_returns(df_feat, states, symbol="TEST")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ── config tests ───────────────────────────────────────────────────────────────

def test_config_schema_defaults():
    cfg = Config(
        asset=AssetConfig(symbol="BTC-USD"),
        features=[FeatureConfig(name="LogReturns")],
        model=ModelConfig(),
        selection=SelectionConfig(),
        run=RunConfig(),
    )
    assert cfg.asset.period == "730d"
    assert cfg.model.n_restarts == 10
    assert cfg.selection.criterion == "bic"


def test_load_btc_yaml(tmp_path):
    import shutil
    from rde.config.loader import load_config
    src = "configs/btc.yaml"
    dst = tmp_path / "btc.yaml"
    shutil.copy(src, dst)
    cfg = load_config(dst)
    assert cfg.asset.symbol == "BTC-USD"
    assert cfg.asset.period == "730d"
    assert cfg.model.candidate_states == [2, 3, 4, 5, 6, 7, 8]
    assert cfg.selection.criterion == "bic"


# ── end-to-end smoke ───────────────────────────────────────────────────────────

def test_full_pipeline_smoke():
    """Data → features → model → inference → labels → plots — no yfinance."""
    df = _synthetic_df(700)
    pipeline = FeaturePipeline([LogReturns(), RollingVolatility(24), SmoothedReturns(12)])
    df_feat = pipeline.transform(df)
    feature_names = ["log_return", "volatility_w24", "smoothed_return_w12"]
    X = df_feat[feature_names].values

    fitted = train_hmm(
        X, n_states=2, n_restarts=3, n_iter=100, seed_base=42, feature_names=feature_names
    )
    X_scaled = fitted.scaler.transform(X)
    states = viterbi_decode(fitted.hmm, X_scaled)
    posteriors = forward_backward_posteriors(fitted.hmm, X_scaled)
    labelled = rank_states(fitted)
    stat = stationary_distribution(fitted.hmm.transmat_)

    # Correctness assertions
    assert states.shape == (len(df_feat),)
    assert posteriors.shape == (len(df_feat), 2)
    np.testing.assert_allclose(posteriors.sum(axis=1), 1.0, atol=1e-6)
    assert len(labelled) == 2
    np.testing.assert_allclose(stat.sum(), 1.0, atol=1e-6)

    # Plots
    label_list = [ls.label for ls in sorted(labelled, key=lambda x: x.index)]
    figs = [
        plot_price_with_regimes(df_feat, states, labels=label_list, symbol="SYNTHETIC"),
        plot_transition_heatmap(fitted.hmm.transmat_, labels=label_list, symbol="SYNTHETIC"),
        plot_regime_timeline(df_feat, states, labels=label_list, symbol="SYNTHETIC"),
        plot_per_regime_returns(df_feat, states, labels=label_list, symbol="SYNTHETIC"),
    ]
    for fig in figs:
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
