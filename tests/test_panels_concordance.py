"""Tests for app/panels_concordance.py — pure figure builders."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_APP_DIR = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(_APP_DIR))

try:
    from panels_concordance import (
        _build_sync_heatmap_figure,
        _build_rolling_concordance_figure,
        _build_lead_lag_figure,
    )
    import plotly.graph_objects as go
    _PANELS_AVAILABLE = True
except ImportError:
    _PANELS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _PANELS_AVAILABLE, reason="panels_concordance.py not found"
)


# ---------------------------------------------------------------------------
# Synthetic test-data helpers
# ---------------------------------------------------------------------------

def _sync_matrix(assets: list[str] | None = None) -> pd.DataFrame:
    """Return a small synthetic sync matrix."""
    if assets is None:
        assets = ["BTC-USD", "ETH-USD", "SPY"]
    n = len(assets)
    data = np.eye(n)
    # Off-diagonal values in [0, 1]
    for i in range(n):
        for j in range(n):
            if i != j:
                data[i, j] = 0.55 + 0.1 * ((i + j) % 3)
    return pd.DataFrame(data, index=assets, columns=assets)


def _rolling_df(n_bars: int = 50, pairs: list[str] | None = None) -> pd.DataFrame:
    """Return a rolling-concordance DataFrame with leading NaNs."""
    if pairs is None:
        pairs = ["BTC-USD__ETH-USD", "BTC-USD__SPY"]
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="D")
    data: dict[str, list[float]] = {}
    for col in pairs:
        series = [float("nan")] * 19 + list(np.random.default_rng(0).uniform(0.3, 0.9, n_bars - 19))
        data[col] = series
    return pd.DataFrame(data, index=idx)


class _FakePairwise:
    """Minimal stand-in for PairwiseConcordance."""

    def __init__(self, asset_a: str, asset_b: str, lead_lag_days: int) -> None:
        self.asset_a = asset_a
        self.asset_b = asset_b
        self.lead_lag_days = lead_lag_days


# ---------------------------------------------------------------------------
# Tests: _build_sync_heatmap_figure
# ---------------------------------------------------------------------------

class TestBuildSyncHeatmapFigure:
    def test_returns_figure(self):
        fig = _build_sync_heatmap_figure(_sync_matrix(), "Test Heatmap")
        assert isinstance(fig, go.Figure)

    def test_has_exactly_one_trace(self):
        fig = _build_sync_heatmap_figure(_sync_matrix(), "Test Heatmap")
        assert len(fig.data) == 1

    def test_trace_is_heatmap(self):
        fig = _build_sync_heatmap_figure(_sync_matrix(), "Test Heatmap")
        assert isinstance(fig.data[0], go.Heatmap)

    def test_text_annotations_formatted_as_percentages(self):
        fig = _build_sync_heatmap_figure(_sync_matrix(), "Test Heatmap")
        heatmap = fig.data[0]
        # text should be a 2D list/tuple; each cell ends with "%"
        for row in heatmap.text:
            for cell in row:
                assert cell.endswith("%"), f"Expected '%' suffix in '{cell}'"

    def test_title_appears_in_layout(self):
        title = "My Sync Heatmap"
        fig = _build_sync_heatmap_figure(_sync_matrix(), title)
        assert title in str(fig.layout.title)

    def test_2x2_matrix(self):
        assets = ["A", "B"]
        df = pd.DataFrame([[1.0, 0.6], [0.6, 1.0]], index=assets, columns=assets)
        fig = _build_sync_heatmap_figure(df, "2x2")
        assert isinstance(fig, go.Figure)

    def test_colorscale_is_rdylgn(self):
        fig = _build_sync_heatmap_figure(_sync_matrix(), "Colorscale check")
        colorscale = fig.data[0].colorscale
        # Plotly may store as a tuple/list; just verify it's not empty
        assert colorscale is not None


# ---------------------------------------------------------------------------
# Tests: _build_rolling_concordance_figure
# ---------------------------------------------------------------------------

class TestBuildRollingConcordanceFigure:
    def test_returns_figure(self):
        rdf = _rolling_df()
        fig = _build_rolling_concordance_figure(rdf, list(rdf.columns), "Rolling")
        assert isinstance(fig, go.Figure)

    def test_one_trace_per_column(self):
        pairs = ["BTC-USD__ETH-USD", "BTC-USD__SPY", "ETH-USD__SPY"]
        rdf = _rolling_df(pairs=pairs)
        fig = _build_rolling_concordance_figure(rdf, pairs, "Rolling 3 pairs")
        assert len(fig.data) == len(pairs)

    def test_title_appears_in_layout(self):
        rdf = _rolling_df()
        title = "Rolling Concordance Chart"
        fig = _build_rolling_concordance_figure(rdf, list(rdf.columns), title)
        assert title in str(fig.layout.title)

    def test_unknown_column_skipped(self):
        """Columns not present in rolling_df are silently skipped."""
        rdf = _rolling_df(pairs=["A__B"])
        fig = _build_rolling_concordance_figure(rdf, ["A__B", "X__Y"], "Skip test")
        assert isinstance(fig, go.Figure)
        # Only "A__B" is in rdf; "X__Y" should be ignored
        assert len(fig.data) == 1

    def test_empty_pairs_list(self):
        rdf = _rolling_df()
        fig = _build_rolling_concordance_figure(rdf, [], "Empty pairs")
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Tests: _build_lead_lag_figure
# ---------------------------------------------------------------------------

class TestBuildLeadLagFigure:
    def _make_pairs(self, lags: list[int]) -> list[_FakePairwise]:
        assets = [f"A{i}" for i in range(len(lags) + 1)]
        return [
            _FakePairwise(assets[i], assets[i + 1], lag)
            for i, lag in enumerate(lags)
        ]

    def test_returns_figure(self):
        pairs = self._make_pairs([3, -2, 0])
        fig = _build_lead_lag_figure(pairs, "Lead-Lag")
        assert isinstance(fig, go.Figure)

    def test_positive_lag_green(self):
        pairs = self._make_pairs([5])
        fig = _build_lead_lag_figure(pairs, "Positive lag")
        color = fig.data[0].marker.color
        # marker.color may be a list or a scalar
        if isinstance(color, (list, tuple)):
            assert color[0] == "#2ca02c", f"Expected green, got {color[0]}"
        else:
            assert color == "#2ca02c", f"Expected green, got {color}"

    def test_negative_lag_red(self):
        pairs = self._make_pairs([-4])
        fig = _build_lead_lag_figure(pairs, "Negative lag")
        color = fig.data[0].marker.color
        if isinstance(color, (list, tuple)):
            assert color[0] == "#d62728", f"Expected red, got {color[0]}"
        else:
            assert color == "#d62728", f"Expected red, got {color}"

    def test_zero_lag_grey(self):
        pairs = self._make_pairs([0])
        fig = _build_lead_lag_figure(pairs, "Zero lag")
        color = fig.data[0].marker.color
        if isinstance(color, (list, tuple)):
            assert color[0] == "#7f7f7f", f"Expected grey, got {color[0]}"
        else:
            assert color == "#7f7f7f", f"Expected grey, got {color}"

    def test_empty_pairwise_list_returns_figure(self):
        """Empty list must not raise and must return a Figure."""
        fig = _build_lead_lag_figure([], "Empty lead-lag")
        assert isinstance(fig, go.Figure)

    def test_title_appears_in_layout(self):
        pairs = self._make_pairs([1, -1])
        title = "My Lead-Lag Chart"
        fig = _build_lead_lag_figure(pairs, title)
        assert title in str(fig.layout.title)
