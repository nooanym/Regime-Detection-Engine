"""Tests for static (matplotlib) and interactive (Plotly) visualisations.

All tests use synthetic data — no network calls, no display needed.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from rde.features.pipeline import FeaturePipeline
from rde.features.returns import LogReturns, SmoothedReturns
from rde.features.volatility import RollingVolatility
from rde.inference.viterbi import viterbi_decode
from rde.models.hmm import train_hmm
from rde.viz.static_plots import (
    plot_per_regime_returns,
    plot_price_with_regimes,
    plot_regime_timeline,
    plot_transition_heatmap,
)
from rde.viz.interactive import (
    plot_per_regime_returns as iplot_per_regime_returns,
    plot_price_with_regimes as iplot_price_with_regimes,
    plot_regime_timeline as iplot_regime_timeline,
    plot_transition_heatmap as iplot_transition_heatmap,
)


# ── fixtures ───────────────────────────────────────────────────────────────────

def _synthetic_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 50_000.0 + np.cumsum(rng.normal(0, 200, n))
    close = np.maximum(close, 1.0)
    return pd.DataFrame(
        {
            "Open": close - 50, "High": close + 100,
            "Low": close - 100, "Close": close,
            "Volume": rng.integers(1_000, 50_000, n).astype(float),
        },
        index=dates,
    )


@pytest.fixture(scope="module")
def regime_data():
    """Fitted model + decoded states on synthetic data, shared across tests."""
    df = _synthetic_df(400)
    pipeline = FeaturePipeline([LogReturns(), RollingVolatility(24), SmoothedReturns(12)])
    df_feat = pipeline.transform(df)
    feat_cols = ["log_return", "volatility_w24", "smoothed_return_w12"]
    X = df_feat[feat_cols].values
    fitted = train_hmm(X, n_states=3, n_restarts=2, n_iter=50, seed_base=0,
                       feature_names=feat_cols)
    X_scaled = fitted.scaler.transform(X)
    states = viterbi_decode(fitted.hmm, X_scaled)
    return df_feat, states, fitted.hmm.transmat_


# ── helpers ────────────────────────────────────────────────────────────────────

def _close_mpl(fig):
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# STATIC PLOTS
# ══════════════════════════════════════════════════════════════════════════════

class TestStaticPricePlot:
    def test_returns_mpl_figure(self, regime_data):
        df, states, _ = regime_data
        fig = plot_price_with_regimes(df, states, symbol="TEST")
        assert isinstance(fig, plt.Figure)
        _close_mpl(fig)

    def test_with_custom_labels(self, regime_data):
        df, states, _ = regime_data
        fig = plot_price_with_regimes(df, states, labels=["bear", "neutral", "bull"], symbol="TEST")
        assert isinstance(fig, plt.Figure)
        _close_mpl(fig)

    def test_legend_has_state_entries(self, regime_data):
        df, states, _ = regime_data
        labels = ["A", "B", "C"]
        fig = plot_price_with_regimes(df, states, labels=labels, symbol="TEST")
        ax = fig.axes[0]
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        for lbl in labels:
            assert lbl in legend_texts
        _close_mpl(fig)

    def test_two_state_model(self):
        df = _synthetic_df(200)
        pipeline = FeaturePipeline([LogReturns(), RollingVolatility(24), SmoothedReturns(12)])
        df_feat = pipeline.transform(df)
        X = df_feat[["log_return", "volatility_w24", "smoothed_return_w12"]].values
        fitted = train_hmm(X, n_states=2, n_restarts=2, n_iter=30, seed_base=1)
        states = viterbi_decode(fitted.hmm, fitted.scaler.transform(X))
        fig = plot_price_with_regimes(df_feat, states, symbol="TEST")
        assert isinstance(fig, plt.Figure)
        _close_mpl(fig)


class TestStaticTransitionHeatmap:
    def test_returns_mpl_figure(self, regime_data):
        _, _, transmat = regime_data
        fig = plot_transition_heatmap(transmat, symbol="TEST")
        assert isinstance(fig, plt.Figure)
        _close_mpl(fig)

    def test_with_labels(self, regime_data):
        _, _, transmat = regime_data
        fig = plot_transition_heatmap(transmat, labels=["low", "mid", "high"], symbol="TEST")
        assert isinstance(fig, plt.Figure)
        _close_mpl(fig)

    def test_two_state_transmat(self):
        transmat = np.array([[0.9, 0.1], [0.2, 0.8]])
        fig = plot_transition_heatmap(transmat, symbol="TEST")
        assert isinstance(fig, plt.Figure)
        _close_mpl(fig)

    def test_annotations_present(self, regime_data):
        """Each cell should contain a text annotation."""
        _, _, transmat = regime_data
        fig = plot_transition_heatmap(transmat, symbol="TEST")
        ax = fig.axes[0]
        n = transmat.shape[0]
        text_count = sum(1 for child in ax.get_children()
                         if hasattr(child, "get_text") and child.get_text())
        assert text_count >= n * n
        _close_mpl(fig)


class TestStaticRegimeTimeline:
    def test_returns_mpl_figure(self, regime_data):
        df, states, _ = regime_data
        fig = plot_regime_timeline(df, states, symbol="TEST")
        assert isinstance(fig, plt.Figure)
        _close_mpl(fig)

    def test_y_axis_hidden(self, regime_data):
        df, states, _ = regime_data
        fig = plot_regime_timeline(df, states, symbol="TEST")
        ax = fig.axes[0]
        assert ax.get_yticks().size == 0
        _close_mpl(fig)


class TestStaticPerRegimeReturns:
    def test_returns_mpl_figure(self, regime_data):
        df, states, _ = regime_data
        fig = plot_per_regime_returns(df, states, symbol="TEST")
        assert isinstance(fig, plt.Figure)
        _close_mpl(fig)

    def test_correct_number_of_histogram_patches(self, regime_data):
        """At least one patch per state that has observations."""
        df, states, _ = regime_data
        fig = plot_per_regime_returns(df, states, symbol="TEST")
        ax = fig.axes[0]
        assert len(ax.patches) > 0
        _close_mpl(fig)

    def test_missing_return_col_raises(self, regime_data):
        df, states, _ = regime_data
        with pytest.raises((KeyError, Exception)):
            plot_per_regime_returns(df, states, return_col="nonexistent", symbol="TEST")


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE (PLOTLY) PLOTS
# ══════════════════════════════════════════════════════════════════════════════

class TestInteractivePricePlot:
    def test_returns_plotly_figure(self, regime_data):
        df, states, _ = regime_data
        fig = iplot_price_with_regimes(df, states, symbol="TEST")
        assert isinstance(fig, go.Figure)

    def test_has_price_trace(self, regime_data):
        df, states, _ = regime_data
        fig = iplot_price_with_regimes(df, states, symbol="TEST")
        trace_names = [t.name for t in fig.data]
        assert "Close" in trace_names

    def test_has_vrects_for_shading(self, regime_data):
        df, states, _ = regime_data
        fig = iplot_price_with_regimes(df, states, symbol="TEST")
        assert len(fig.layout.shapes) > 0

    def test_with_custom_labels(self, regime_data):
        df, states, _ = regime_data
        fig = iplot_price_with_regimes(df, states, labels=["bear", "neutral", "bull"], symbol="TEST")
        assert isinstance(fig, go.Figure)

    def test_title_contains_symbol(self, regime_data):
        df, states, _ = regime_data
        fig = iplot_price_with_regimes(df, states, symbol="BTC-USD")
        assert "BTC-USD" in fig.layout.title.text

    def test_two_state_model(self):
        df = _synthetic_df(200)
        pipeline = FeaturePipeline([LogReturns(), RollingVolatility(24), SmoothedReturns(12)])
        df_feat = pipeline.transform(df)
        X = df_feat[["log_return", "volatility_w24", "smoothed_return_w12"]].values
        fitted = train_hmm(X, n_states=2, n_restarts=2, n_iter=30, seed_base=1)
        states = viterbi_decode(fitted.hmm, fitted.scaler.transform(X))
        fig = iplot_price_with_regimes(df_feat, states, symbol="TEST")
        assert isinstance(fig, go.Figure)


class TestInteractiveTransitionHeatmap:
    def test_returns_plotly_figure(self, regime_data):
        _, _, transmat = regime_data
        fig = iplot_transition_heatmap(transmat, symbol="TEST")
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self, regime_data):
        _, _, transmat = regime_data
        fig = iplot_transition_heatmap(transmat, symbol="TEST")
        assert any(isinstance(t, go.Heatmap) for t in fig.data)

    def test_heatmap_z_matches_transmat(self, regime_data):
        _, _, transmat = regime_data
        fig = iplot_transition_heatmap(transmat, symbol="TEST")
        heatmap = next(t for t in fig.data if isinstance(t, go.Heatmap))
        np.testing.assert_allclose(np.array(heatmap.z), transmat, atol=1e-9)

    def test_axis_labels_match_states(self, regime_data):
        _, _, transmat = regime_data
        labels = ["low", "mid", "high"]
        fig = iplot_transition_heatmap(transmat, labels=labels, symbol="TEST")
        heatmap = next(t for t in fig.data if isinstance(t, go.Heatmap))
        assert list(heatmap.x) == labels
        assert list(heatmap.y) == labels

    def test_title_contains_symbol(self, regime_data):
        _, _, transmat = regime_data
        fig = iplot_transition_heatmap(transmat, symbol="ETH-USD")
        assert "ETH-USD" in fig.layout.title.text


class TestInteractiveRegimeTimeline:
    def test_returns_plotly_figure(self, regime_data):
        df, states, _ = regime_data
        fig = iplot_regime_timeline(df, states, symbol="TEST")
        assert isinstance(fig, go.Figure)

    def test_has_vrects(self, regime_data):
        df, states, _ = regime_data
        fig = iplot_regime_timeline(df, states, symbol="TEST")
        assert len(fig.layout.shapes) > 0

    def test_y_axis_hidden(self, regime_data):
        df, states, _ = regime_data
        fig = iplot_regime_timeline(df, states, symbol="TEST")
        assert fig.layout.yaxis.visible is False

    def test_title_contains_symbol(self, regime_data):
        df, states, _ = regime_data
        fig = iplot_regime_timeline(df, states, symbol="SPY")
        assert "SPY" in fig.layout.title.text


class TestInteractivePerRegimeReturns:
    def test_returns_plotly_figure(self, regime_data):
        df, states, _ = regime_data
        fig = iplot_per_regime_returns(df, states, symbol="TEST")
        assert isinstance(fig, go.Figure)

    def test_has_histogram_traces(self, regime_data):
        df, states, _ = regime_data
        fig = iplot_per_regime_returns(df, states, symbol="TEST")
        hist_traces = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(hist_traces) > 0

    def test_number_of_histograms_matches_states(self, regime_data):
        df, states, _ = regime_data
        n_states = int(states.max()) + 1
        fig = iplot_per_regime_returns(df, states, symbol="TEST")
        hist_traces = [t for t in fig.data if isinstance(t, go.Histogram)]
        assert len(hist_traces) == n_states

    def test_barmode_is_overlay(self, regime_data):
        df, states, _ = regime_data
        fig = iplot_per_regime_returns(df, states, symbol="TEST")
        assert fig.layout.barmode == "overlay"

    def test_with_custom_labels(self, regime_data):
        df, states, _ = regime_data
        fig = iplot_per_regime_returns(df, states,
                                       labels=["bear", "neutral", "bull"], symbol="TEST")
        hist_traces = [t for t in fig.data if isinstance(t, go.Histogram)]
        trace_names = {t.name for t in hist_traces}
        assert trace_names == {"bear", "neutral", "bull"}


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG VALIDATION — ETH and SPY
# ══════════════════════════════════════════════════════════════════════════════

class TestAssetConfigs:
    def test_eth_config_loads(self):
        from rde.config.loader import load_config
        cfg = load_config("configs/eth.yaml")
        assert cfg.asset.symbol == "ETH-USD"
        assert cfg.asset.period == "730d"
        assert cfg.asset.interval == "1h"
        assert cfg.model.candidate_states == [2, 3, 4, 5, 6]

    def test_spy_config_loads(self):
        from rde.config.loader import load_config
        cfg = load_config("configs/spy.yaml")
        assert cfg.asset.symbol == "SPY"
        assert cfg.asset.period == "730d"
        assert cfg.selection.criterion == "bic"

    def test_btc_config_loads(self):
        from rde.config.loader import load_config
        cfg = load_config("configs/btc.yaml")
        assert cfg.asset.symbol == "BTC-USD"
        assert cfg.model.n_restarts == 10
        assert cfg.model.seed_base == 42

    def test_invalid_period_raises(self, tmp_path):
        from rde.config.loader import load_config
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "asset:\n  symbol: BTC-USD\n  period: 800d\n  interval: 1h\n"
            "features: []\nmodel: {}\nselection: {}\nrun: {}\n"
        )
        with pytest.raises(ValueError, match="730d"):
            load_config(bad)

    def test_invalid_criterion_raises(self, tmp_path):
        from rde.config.loader import load_config
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "asset:\n  symbol: BTC-USD\n"
            "features: []\nmodel: {}\nselection:\n  criterion: rmse\nrun: {}\n"
        )
        with pytest.raises(ValueError, match="criterion"):
            load_config(bad)

    def test_invalid_covariance_type_raises(self, tmp_path):
        from rde.config.loader import load_config
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "asset:\n  symbol: BTC-USD\n"
            "features: []\nmodel:\n  covariance_type: sparse\nselection: {}\nrun: {}\n"
        )
        with pytest.raises(ValueError, match="covariance_type"):
            load_config(bad)

    def test_candidate_states_below_two_raises(self, tmp_path):
        from rde.config.loader import load_config
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "asset:\n  symbol: BTC-USD\n"
            "features: []\nmodel:\n  candidate_states: [1, 2, 3]\nselection: {}\nrun: {}\n"
        )
        with pytest.raises(ValueError, match="candidate_states"):
            load_config(bad)
