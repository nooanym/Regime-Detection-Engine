"""Tests for app/panels_live.py (Phase 34)."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_APP_DIR = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(_APP_DIR))

try:
    import plotly.graph_objects as go
    from panels_live import (
        _build_live_posterior_figure,
        _build_live_price_figure,
        _load_live_model,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(not _AVAILABLE, reason="panels_live not importable")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_timestamps(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")


def _make_posteriors(n: int, K: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(K), size=n)
    return raw  # shape (n, K), rows sum to 1


def _make_regimes_df(n: int) -> pd.DataFrame:
    idx = _make_timestamps(n)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Close": rng.uniform(40_000, 70_000, size=n),
            "regime": rng.integers(0, 3, size=n),
            "regime_label": ["Bull" if i % 2 == 0 else "Bear" for i in range(n)],
        },
        index=idx,
    )


def _make_fake_fitted_model(K: int = 3, n_features: int = 3, tmp_path: Path = None) -> tuple:
    """Build a minimal fake FittedModel-like object and pickle it."""
    from sklearn.preprocessing import StandardScaler
    from hmmlearn.hmm import GaussianHMM

    scaler = StandardScaler()
    scaler.fit(np.random.default_rng(0).standard_normal((100, n_features)))

    hmm = GaussianHMM(n_components=K, covariance_type="diag", n_iter=5, random_state=0)
    X = np.random.default_rng(0).standard_normal((100, n_features))
    hmm.fit(X)

    # Minimal FittedModel stub
    fitted = MagicMock()
    fitted.hmm = hmm
    fitted.scaler = scaler
    fitted.n_states = K
    fitted.feature_names = ["log_return", "volatility_w24", "smoothed_return_w12"]

    if tmp_path is not None:
        pkl_path = tmp_path / "model.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(fitted, fh)

    return fitted


# ---------------------------------------------------------------------------
# _load_live_model
# ---------------------------------------------------------------------------

class TestLoadLiveModel:
    def test_returns_none_when_pkl_absent(self, tmp_path):
        result = _load_live_model("BTC-USD", tmp_path)
        assert result is None

    def test_returns_model_when_pkl_exists(self, tmp_path):
        fake_model = MagicMock()
        pkl_path = tmp_path / "model.pkl"
        with patch("panels_live._load_live_model", return_value=fake_model):
            # Direct path: write a real pickle, bypass the MagicMock issue
            # by testing via load_model patch in rde.models.persistence
            pass
        with patch("rde.models.persistence.load_model", return_value=fake_model):
            pkl_path.write_bytes(b"\x80\x05\x95\x00\x00\x00\x00\x00\x00\x00\x00.")  # minimal
            # patch load_model so pickle format doesn't matter
            result = _load_live_model("BTC-USD", tmp_path)
        assert result is not None

    def test_returns_none_on_corrupt_pkl(self, tmp_path):
        pkl_path = tmp_path / "model.pkl"
        pkl_path.write_bytes(b"not a pickle")
        result = _load_live_model("BTC-USD", tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# _build_live_posterior_figure
# ---------------------------------------------------------------------------

class TestBuildLivePosteriorFigure:
    def test_returns_figure(self):
        K, n = 3, 100
        ts = _make_timestamps(n)
        post = _make_posteriors(n, K)
        labels = {0: "Bear", 1: "Neutral", 2: "Bull"}
        fig = _build_live_posterior_figure(ts, post, labels, "BTC-USD")
        assert isinstance(fig, go.Figure)

    def test_title_contains_asset(self):
        K, n = 2, 50
        ts = _make_timestamps(n)
        post = _make_posteriors(n, K)
        fig = _build_live_posterior_figure(ts, post, {0: "A", 1: "B"}, "ETH-USD")
        assert "ETH-USD" in fig.layout.title.text

    def test_one_trace_per_state(self):
        K, n = 4, 80
        ts = _make_timestamps(n)
        post = _make_posteriors(n, K)
        fig = _build_live_posterior_figure(ts, post, {k: f"R{k}" for k in range(K)}, "SPY")
        assert len(fig.data) == K

    def test_n_bars_clips_to_available(self):
        K, n = 2, 50
        ts = _make_timestamps(n)
        post = _make_posteriors(n, K)
        # n_bars larger than available data — should not raise
        fig = _build_live_posterior_figure(ts, post, {0: "A", 1: "B"}, "X", n_bars=1000)
        assert isinstance(fig, go.Figure)

    def test_n_bars_limits_display(self):
        K, n = 2, 200
        ts = _make_timestamps(n)
        post = _make_posteriors(n, K)
        fig = _build_live_posterior_figure(ts, post, {0: "A", 1: "B"}, "X", n_bars=20)
        # Each trace should have at most 20 x-values
        for trace in fig.data:
            assert len(trace.x) <= 20

    def test_posteriors_approximately_sum_to_one(self):
        K, n = 3, 60
        ts = _make_timestamps(n)
        post = _make_posteriors(n, K)
        # Verify the input posteriors sum to 1 (test data integrity)
        np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# _build_live_price_figure
# ---------------------------------------------------------------------------

class TestBuildLivePriceFigure:
    def test_returns_figure(self):
        n, K = 100, 3
        df = _make_regimes_df(n)
        post = _make_posteriors(n, K)
        fig = _build_live_price_figure(df, post, "BTC-USD")
        assert isinstance(fig, go.Figure)

    def test_title_contains_asset(self):
        n, K = 50, 2
        df = _make_regimes_df(n)
        post = _make_posteriors(n, K)
        fig = _build_live_price_figure(df, post, "SOL-USD")
        assert "SOL-USD" in fig.layout.title.text

    def test_at_most_k_traces(self):
        n, K = 100, 3
        df = _make_regimes_df(n)
        post = _make_posteriors(n, K)
        fig = _build_live_price_figure(df, post, "X")
        assert len(fig.data) <= K

    def test_n_bars_clips_correctly(self):
        n, K = 200, 2
        df = _make_regimes_df(n)
        post = _make_posteriors(n, K)
        fig = _build_live_price_figure(df, post, "X", n_bars=30)
        total_pts = sum(len(t.x) for t in fig.data)
        assert total_pts <= 30
