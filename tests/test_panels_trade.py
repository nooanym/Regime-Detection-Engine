"""Tests for app/panels_trade.py — Trade History panel figure builders."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_APP_DIR = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(_APP_DIR))

try:
    from panels_trade import (
        _build_equity_curve_figure,
        _build_drawdown_figure,
        _build_fill_scatter_figure,
        _build_regime_pnl_figure,
    )
    import plotly.graph_objects as go
    _PANELS_AVAILABLE = True
except ImportError:
    _PANELS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _PANELS_AVAILABLE, reason="panels_trade.py not found"
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_ASSET = "TEST-USD"


def _make_snapshots(n: int = 50, start_equity: float = 10_000.0) -> pd.DataFrame:
    """Create a minimal portfolio_snapshots DataFrame."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    equity = start_equity + np.cumsum(rng.normal(10, 50, size=n))
    price = 40_000.0 + np.cumsum(rng.normal(0, 200, size=n))
    regime = rng.integers(0, 3, size=n)
    return pd.DataFrame(
        {"equity": equity, "price": price, "regime": regime},
        index=idx,
    )


def _make_trade_log(n: int = 10) -> pd.DataFrame:
    """Create a minimal trade_log DataFrame with alternating buy/sell."""
    idx = pd.date_range("2024-01-01 06:00", periods=n, freq="5h", tz="UTC")
    sides = ["buy" if i % 2 == 0 else "sell" for i in range(n)]
    prices = 40_000.0 + np.arange(n) * 100.0
    return pd.DataFrame(
        {"side": sides, "price": prices, "quantity": [0.01] * n},
        index=idx,
    )


def _empty_snapshots() -> pd.DataFrame:
    return pd.DataFrame(columns=["equity", "price", "regime"])


def _empty_trade_log() -> pd.DataFrame:
    return pd.DataFrame(columns=["side", "price", "quantity"])


# ---------------------------------------------------------------------------
# Tests: _build_equity_curve_figure
# ---------------------------------------------------------------------------


class TestBuildEquityCurveFigure:
    def test_returns_figure(self):
        fig = _build_equity_curve_figure(_make_snapshots(), _ASSET)
        assert isinstance(fig, go.Figure)

    def test_title_contains_asset(self):
        fig = _build_equity_curve_figure(_make_snapshots(), _ASSET)
        assert _ASSET in str(fig.layout.title.text)

    def test_has_reference_line(self):
        """Equity curve + dashed reference line → at least 2 traces/shapes."""
        fig = _build_equity_curve_figure(_make_snapshots(), _ASSET)
        # The equity line is a trace; the hline adds a layout shape
        assert len(fig.data) >= 1
        # hline adds to fig.layout.shapes
        assert len(fig.layout.shapes) >= 1

    def test_empty_snapshots_does_not_raise(self):
        fig = _build_equity_curve_figure(_empty_snapshots(), _ASSET)
        assert isinstance(fig, go.Figure)

    def test_single_row_does_not_raise(self):
        single = _make_snapshots(n=1)
        fig = _build_equity_curve_figure(single, _ASSET)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Tests: _build_drawdown_figure
# ---------------------------------------------------------------------------


class TestBuildDrawdownFigure:
    def test_returns_figure(self):
        fig = _build_drawdown_figure(_make_snapshots(), _ASSET)
        assert isinstance(fig, go.Figure)

    def test_title_contains_asset(self):
        fig = _build_drawdown_figure(_make_snapshots(), _ASSET)
        assert _ASSET in str(fig.layout.title.text)

    def test_drawdown_values_lte_zero(self):
        """All drawdown values must be ≤ 0 (or NaN for empty)."""
        snapshots = _make_snapshots(50)
        fig = _build_drawdown_figure(snapshots, _ASSET)
        assert len(fig.data) == 1
        y_vals = np.array(fig.data[0].y, dtype=float)
        finite = y_vals[np.isfinite(y_vals)]
        assert np.all(finite <= 0.0 + 1e-9)

    def test_empty_snapshots_does_not_raise(self):
        fig = _build_drawdown_figure(_empty_snapshots(), _ASSET)
        assert isinstance(fig, go.Figure)

    def test_filled_area_trace(self):
        """The drawdown trace should have a fill property set."""
        fig = _build_drawdown_figure(_make_snapshots(), _ASSET)
        assert len(fig.data) >= 1
        trace = fig.data[0]
        assert trace.fill is not None and trace.fill != "none"


# ---------------------------------------------------------------------------
# Tests: _build_fill_scatter_figure
# ---------------------------------------------------------------------------


class TestBuildFillScatterFigure:
    def test_returns_figure(self):
        fig = _build_fill_scatter_figure(_make_snapshots(), _make_trade_log(), _ASSET)
        assert isinstance(fig, go.Figure)

    def test_title_contains_asset(self):
        fig = _build_fill_scatter_figure(_make_snapshots(), _make_trade_log(), _ASSET)
        assert _ASSET in str(fig.layout.title.text)

    def test_buy_and_sell_distinct_colors(self):
        """Buy trace color must differ from sell trace color."""
        fig = _build_fill_scatter_figure(_make_snapshots(), _make_trade_log(), _ASSET)
        # Price line + buy + sell = at least 3 traces
        assert len(fig.data) >= 3
        names = [t.name for t in fig.data]
        assert "Buy" in names
        assert "Sell" in names
        buy_trace = next(t for t in fig.data if t.name == "Buy")
        sell_trace = next(t for t in fig.data if t.name == "Sell")
        buy_color = buy_trace.marker.color
        sell_color = sell_trace.marker.color
        assert buy_color != sell_color

    def test_empty_inputs_do_not_raise(self):
        fig = _build_fill_scatter_figure(_empty_snapshots(), _empty_trade_log(), _ASSET)
        assert isinstance(fig, go.Figure)

    def test_buy_only_trade_log(self):
        tl = _make_trade_log(4)
        tl["side"] = "buy"
        fig = _build_fill_scatter_figure(_make_snapshots(), tl, _ASSET)
        assert isinstance(fig, go.Figure)
        names = [t.name for t in fig.data]
        assert "Buy" in names
        assert "Sell" not in names


# ---------------------------------------------------------------------------
# Tests: _build_regime_pnl_figure
# ---------------------------------------------------------------------------


class TestBuildRegimePnlFigure:
    def test_returns_figure(self):
        fig = _build_regime_pnl_figure(_make_snapshots(), _ASSET)
        assert isinstance(fig, go.Figure)

    def test_title_contains_asset(self):
        fig = _build_regime_pnl_figure(_make_snapshots(), _ASSET)
        assert _ASSET in str(fig.layout.title.text)

    def test_n_bars_equals_n_regimes(self):
        snapshots = _make_snapshots(60)
        n_regimes = int(snapshots["regime"].nunique())
        fig = _build_regime_pnl_figure(snapshots, _ASSET)
        assert len(fig.data) == 1  # single Bar trace
        assert len(fig.data[0].x) == n_regimes

    def test_empty_snapshots_does_not_raise(self):
        fig = _build_regime_pnl_figure(_empty_snapshots(), _ASSET)
        assert isinstance(fig, go.Figure)

    def test_single_regime(self):
        snapshots = _make_snapshots(20)
        snapshots["regime"] = 0  # only one regime
        fig = _build_regime_pnl_figure(snapshots, _ASSET)
        assert isinstance(fig, go.Figure)
