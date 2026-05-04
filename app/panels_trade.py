"""
Regime Detection Engine — Trade History Panel (Phase 36)

Two-tier architecture:
  _build_*_figure(data) — pure functions returning go.Figure (no Streamlit calls)
  _panel_trade_history(asset, results_dir) — UI layer calling builders + st.*

Reads Phase 35 output from results/<asset>/live/:
  trade_log.parquet        — one row per fill (timestamp index)
  portfolio_snapshots.parquet — one row per bar post-warmup (timestamp index)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Shared color palette (must match streamlit_app.py)
# ---------------------------------------------------------------------------
_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


# ---------------------------------------------------------------------------
# Pure figure builders
# ---------------------------------------------------------------------------


def _build_equity_curve_figure(snapshots_df: pd.DataFrame, asset: str) -> go.Figure:
    """Line chart of portfolio equity over time with a starting-value reference line.

    Parameters
    ----------
    snapshots_df : pd.DataFrame
        Portfolio snapshots with a DatetimeIndex and an ``equity`` column.
    asset : str
        Asset symbol, used in the plot title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    if len(snapshots_df) > 0 and "equity" in snapshots_df.columns:
        equity = snapshots_df["equity"]
        start_equity = float(equity.iloc[0])

        # Equity line
        fig.add_trace(
            go.Scattergl(
                x=snapshots_df.index,
                y=equity.values,
                mode="lines",
                name="Equity",
                line=dict(color="#1f77b4", width=1.5),
            )
        )

        # Horizontal reference at starting equity
        fig.add_hline(
            y=start_equity,
            line=dict(dash="dash", color="#7f7f7f", width=1),
            annotation_text=f"Start: {start_equity:,.0f}",
            annotation_position="right",
        )

    fig.update_layout(
        title=f"{asset} — Paper Trading Equity Curve",
        xaxis=dict(title="Date", rangeslider=dict(visible=True), type="date"),
        yaxis=dict(title="Portfolio Value ($)"),
        hovermode="x unified",
        margin=dict(t=60, b=40),
    )
    return fig


def _build_drawdown_figure(snapshots_df: pd.DataFrame, asset: str) -> go.Figure:
    """Filled area chart of running drawdown expressed as a percentage.

    Parameters
    ----------
    snapshots_df : pd.DataFrame
        Portfolio snapshots with a DatetimeIndex and an ``equity`` column.
    asset : str
        Asset symbol, used in the plot title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    if len(snapshots_df) > 0 and "equity" in snapshots_df.columns:
        equity = snapshots_df["equity"]
        drawdown = (equity / equity.cummax() - 1.0) * 100.0

        fig.add_trace(
            go.Scatter(
                x=snapshots_df.index,
                y=drawdown.values,
                mode="lines",
                name="Drawdown (%)",
                fill="tozeroy",
                line=dict(color="#d62728", width=1),
                fillcolor="rgba(214, 39, 40, 0.4)",
            )
        )

    fig.update_layout(
        title=f"{asset} — Drawdown (%)",
        xaxis=dict(title="Date", rangeslider=dict(visible=True), type="date"),
        yaxis=dict(title="Drawdown (%)"),
        hovermode="x unified",
        margin=dict(t=60, b=40),
    )
    return fig


def _build_fill_scatter_figure(
    snapshots_df: pd.DataFrame,
    trade_log_df: pd.DataFrame,
    asset: str,
) -> go.Figure:
    """Price line overlaid with buy/sell fill markers.

    Parameters
    ----------
    snapshots_df : pd.DataFrame
        Portfolio snapshots with a ``price`` column and DatetimeIndex.
    trade_log_df : pd.DataFrame
        Trade log with ``price`` and ``side`` columns and DatetimeIndex.
    asset : str
        Asset symbol, used in the plot title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # Price line from snapshots
    if len(snapshots_df) > 0 and "price" in snapshots_df.columns:
        fig.add_trace(
            go.Scattergl(
                x=snapshots_df.index,
                y=snapshots_df["price"].values,
                mode="lines",
                name="Price",
                line=dict(color="#7f7f7f", width=1),
            )
        )

    # Buy fills
    if len(trade_log_df) > 0 and "side" in trade_log_df.columns and "price" in trade_log_df.columns:
        buys = trade_log_df[trade_log_df["side"] == "buy"]
        if len(buys) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buys.index,
                    y=buys["price"].values,
                    mode="markers",
                    name="Buy",
                    marker=dict(
                        symbol="triangle-up",
                        color="#2ca02c",
                        size=10,
                        line=dict(color="#2ca02c", width=1),
                    ),
                )
            )

        # Sell fills
        sells = trade_log_df[trade_log_df["side"] == "sell"]
        if len(sells) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sells.index,
                    y=sells["price"].values,
                    mode="markers",
                    name="Sell",
                    marker=dict(
                        symbol="triangle-down",
                        color="#d62728",
                        size=10,
                        line=dict(color="#d62728", width=1),
                    ),
                )
            )

    fig.update_layout(
        title=f"{asset} — Fill Locations",
        xaxis=dict(title="Date", rangeslider=dict(visible=True), type="date"),
        yaxis=dict(title="Price ($)"),
        hovermode="x unified",
        margin=dict(t=60, b=40),
    )
    return fig


def _build_regime_pnl_figure(snapshots_df: pd.DataFrame, asset: str) -> go.Figure:
    """Bar chart of mean bar P&L grouped by regime.

    Parameters
    ----------
    snapshots_df : pd.DataFrame
        Portfolio snapshots with ``equity`` and ``regime`` columns.
    asset : str
        Asset symbol, used in the plot title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    if (
        len(snapshots_df) > 1
        and "equity" in snapshots_df.columns
        and "regime" in snapshots_df.columns
    ):
        df = snapshots_df.copy()
        df["equity_change"] = df["equity"].diff()

        grouped = (
            df.groupby("regime")["equity_change"]
            .mean()
            .sort_index()
        )

        regime_indices = grouped.index.tolist()
        x_labels = [f"Regime {r}" for r in regime_indices]
        y_values = grouped.values.tolist()
        colors = [_PALETTE[int(r) % len(_PALETTE)] for r in regime_indices]

        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=y_values,
                marker_color=colors,
                name="Mean Bar P&L",
            )
        )

    fig.update_layout(
        title=f"{asset} — Mean Bar P&L by Regime",
        xaxis=dict(title="Regime"),
        yaxis=dict(title="Mean Equity Change ($)"),
        showlegend=False,
        margin=dict(t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit panel
# ---------------------------------------------------------------------------


def _panel_trade_history(asset: str, results_dir: Path) -> None:
    """Streamlit panel: Trade History from Phase 35 paper-trading output.

    Loads ``trade_log.parquet`` and ``portfolio_snapshots.parquet`` from
    ``results_dir / "live"/`` and renders four Plotly charts plus summary
    metrics.

    Parameters
    ----------
    asset : str
        Asset symbol for display labels.
    results_dir : Path
        The asset's results directory (e.g. ``results/BTC-USD``).
    """
    st.subheader("Trade History (Paper Trading)")

    live_dir = results_dir / "live"
    trade_log_path = live_dir / "trade_log.parquet"
    snapshots_path = live_dir / "portfolio_snapshots.parquet"

    # Guard: missing files
    if not trade_log_path.exists() or not snapshots_path.exists():
        missing = []
        if not trade_log_path.exists():
            missing.append("`trade_log.parquet`")
        if not snapshots_path.exists():
            missing.append("`portfolio_snapshots.parquet`")
        st.info(
            f"Paper-trading output not found ({', '.join(missing)}). "
            "Generate it by running:\n\n"
            f"```\nuv run rde trade --model results/{asset}/model.pkl "
            f"--config configs/{asset.lower().replace('-', '')}.yaml --backtest\n```"
        )
        return

    # Load data
    try:
        trade_log_df = pd.read_parquet(trade_log_path)
        snapshots_df = pd.read_parquet(snapshots_path)
    except Exception as exc:
        st.error(f"Failed to load trade history files: {exc}")
        return

    # ── Summary metrics ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    # Total Return %
    if len(snapshots_df) > 1 and "equity" in snapshots_df.columns:
        equity = snapshots_df["equity"]
        total_return_pct = float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0)
    else:
        total_return_pct = float("nan")

    # Max Drawdown %
    if len(snapshots_df) > 1 and "equity" in snapshots_df.columns:
        eq = snapshots_df["equity"]
        max_dd_pct = float((eq / eq.cummax() - 1.0).min() * 100.0)
    else:
        max_dd_pct = float("nan")

    # Total Trades
    total_trades = len(trade_log_df)

    # Sharpe (annualised from daily equity returns)
    sharpe = float("nan")
    if len(snapshots_df) > 1 and "equity" in snapshots_df.columns:
        daily_eq = snapshots_df["equity"].resample("D").last().dropna()
        if len(daily_eq) > 1:
            daily_ret = daily_eq.pct_change().dropna()
            if len(daily_ret) > 1 and daily_ret.std(ddof=1) > 0:
                sharpe = float(
                    daily_ret.mean() / daily_ret.std(ddof=1) * math.sqrt(252)
                )

    col1.metric(
        "Total Return",
        f"{total_return_pct:+.2f}%" if math.isfinite(total_return_pct) else "N/A",
    )
    col2.metric(
        "Max Drawdown",
        f"{max_dd_pct:.2f}%" if math.isfinite(max_dd_pct) else "N/A",
    )
    col3.metric("Total Trades", f"{total_trades:,}")
    col4.metric(
        "Sharpe (ann)",
        f"{sharpe:+.3f}" if math.isfinite(sharpe) else "N/A",
    )

    # ── Figures ────────────────────────────────────────────────────────────────
    equity_fig = _build_equity_curve_figure(snapshots_df, asset)
    st.plotly_chart(equity_fig, use_container_width=True)

    drawdown_fig = _build_drawdown_figure(snapshots_df, asset)
    st.plotly_chart(drawdown_fig, use_container_width=True)

    fill_fig = _build_fill_scatter_figure(snapshots_df, trade_log_df, asset)
    st.plotly_chart(fill_fig, use_container_width=True)

    regime_pnl_fig = _build_regime_pnl_figure(snapshots_df, asset)
    st.plotly_chart(regime_pnl_fig, use_container_width=True)
