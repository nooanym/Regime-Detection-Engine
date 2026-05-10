"""Tests for app/panels_rtmv.py — RTMV Portfolio panel figure builders."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_APP_DIR = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(_APP_DIR))

try:
    from panels_rtmv import (
        _build_equity_chart,
        _build_weights_chart,
        _build_fills_table,
    )
    import plotly.graph_objects as go
    _PANELS_AVAILABLE = True
except ImportError:
    _PANELS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _PANELS_AVAILABLE, reason="panels_rtmv.py not found"
)

ASSETS = ["SPY", "GLD", "TLT", "IEF"]


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_snapshots(n: int = 60, with_rebalances: bool = True) -> pd.DataFrame:
    """Synthetic snapshots DataFrame matching RTMVRebalancer output."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)

    equity = 100_000.0 + np.cumsum(rng.normal(50, 200, n))
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / (peak + 1e-10)

    data = {
        "equity": equity,
        "cash": equity * 0.0,
        "drawdown": drawdown,
        "n_rebalances": np.arange(n) // 10,
        "is_halted": np.zeros(n, dtype=bool),
        "rebalance": np.zeros(n, dtype=bool),
    }

    if with_rebalances:
        data["rebalance"][::10] = True

    weights = np.random.dirichlet(np.ones(len(ASSETS)), size=n)
    for i, a in enumerate(ASSETS):
        data[f"weight_{a}"] = weights[:, i]
        data[f"target_{a}"] = weights[:, i]
        data[f"pos_{a}"] = rng.uniform(1, 10, n)
        data[f"price_{a}"] = rng.uniform(100, 500, n)

    return pd.DataFrame(data, index=dates)


def _make_fills(n: int = 30) -> pd.DataFrame:
    """Synthetic fills DataFrame."""
    rng = np.random.default_rng(99)
    dates = pd.date_range("2020-01-01", periods=n, freq="21B")
    return pd.DataFrame({
        "timestamp": dates,
        "asset": rng.choice(ASSETS, n),
        "side": rng.choice(["buy", "sell"], n),
        "quantity": rng.uniform(0.5, 10.0, n).round(4),
        "price": rng.uniform(100.0, 500.0, n).round(2),
        "slippage": rng.uniform(0.01, 1.0, n).round(4),
        "notional": rng.uniform(100.0, 5000.0, n).round(2),
    })


# ---------------------------------------------------------------------------
# _build_equity_chart tests
# ---------------------------------------------------------------------------


class TestBuildEquityChart:
    def test_returns_figure(self) -> None:
        snaps = _make_snapshots()
        fig = _build_equity_chart(snaps)
        assert isinstance(fig, go.Figure)

    def test_has_equity_trace(self) -> None:
        snaps = _make_snapshots()
        fig = _build_equity_chart(snaps)
        trace_names = [t.name for t in fig.data]
        assert "RTMV Portfolio" in trace_names

    def test_has_drawdown_trace(self) -> None:
        snaps = _make_snapshots()
        fig = _build_equity_chart(snaps)
        trace_names = [t.name for t in fig.data]
        assert "Drawdown %" in trace_names

    def test_has_rebalance_markers_when_present(self) -> None:
        snaps = _make_snapshots(with_rebalances=True)
        fig = _build_equity_chart(snaps)
        trace_names = [t.name for t in fig.data]
        assert "Rebalance" in trace_names

    def test_no_rebalance_trace_when_absent(self) -> None:
        snaps = _make_snapshots(with_rebalances=False)
        fig = _build_equity_chart(snaps)
        trace_names = [t.name for t in fig.data]
        assert "Rebalance" not in trace_names

    def test_no_crash_on_empty_equity(self) -> None:
        snaps = _make_snapshots()
        snaps["equity"] = float("nan")
        fig = _build_equity_chart(snaps)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# _build_weights_chart tests
# ---------------------------------------------------------------------------


class TestBuildWeightsChart:
    def test_returns_figure_with_weight_cols(self) -> None:
        snaps = _make_snapshots()
        fig = _build_weights_chart(snaps)
        assert isinstance(fig, go.Figure)

    def test_returns_none_without_weight_cols(self) -> None:
        snaps = pd.DataFrame({"equity": [1.0, 2.0]})
        result = _build_weights_chart(snaps)
        assert result is None

    def test_one_trace_per_asset(self) -> None:
        snaps = _make_snapshots()
        fig = _build_weights_chart(snaps)
        weight_cols = [c for c in snaps.columns if c.startswith("weight_")]
        assert len(fig.data) == len(weight_cols)

    def test_asset_names_in_traces(self) -> None:
        snaps = _make_snapshots()
        fig = _build_weights_chart(snaps)
        trace_names = {t.name for t in fig.data}
        for a in ASSETS:
            assert a in trace_names


# ---------------------------------------------------------------------------
# _build_fills_table tests
# ---------------------------------------------------------------------------


class TestBuildFillsTable:
    def test_returns_figure(self) -> None:
        fills = _make_fills()
        fig = _build_fills_table(fills)
        assert isinstance(fig, go.Figure)

    def test_table_trace_present(self) -> None:
        fills = _make_fills()
        fig = _build_fills_table(fills)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Table)

    def test_max_50_rows(self) -> None:
        fills = _make_fills(n=100)
        fig = _build_fills_table(fills)
        table = fig.data[0]
        # First column's cell values should be at most 50 entries
        assert len(table.cells.values[0]) <= 50

    def test_no_crash_on_empty_fills(self) -> None:
        fills = pd.DataFrame(
            columns=["timestamp", "asset", "side", "quantity", "price", "slippage", "notional"]
        )
        fig = _build_fills_table(fills)
        assert isinstance(fig, go.Figure)
