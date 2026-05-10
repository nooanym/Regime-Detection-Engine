"""RTMV Portfolio Monitoring Panel (Phase 49).

Reads ``results/rtmv_live/snapshots.parquet`` and
``results/rtmv_live/fills.parquet`` written by
:mod:`rde.trading.rtmv_rebalancer` and renders a Streamlit monitoring panel.

Public API
----------
_panel_rtmv_portfolio(results_dir)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Plotly builders (pure functions, no Streamlit)
# ---------------------------------------------------------------------------


def _build_equity_chart(snaps: pd.DataFrame) -> go.Figure:
    """Equity curve with rebalance event markers."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
    )

    # Row 1: equity curve
    fig.add_trace(
        go.Scatter(
            x=snaps.index,
            y=snaps["equity"],
            mode="lines",
            name="RTMV Portfolio",
            line=dict(color="#1f77b4", width=2),
        ),
        row=1, col=1,
    )

    # Rebalance markers
    if "rebalance" in snaps.columns:
        reb = snaps[snaps["rebalance"].astype(bool)]
        if not reb.empty:
            fig.add_trace(
                go.Scatter(
                    x=reb.index,
                    y=reb["equity"],
                    mode="markers",
                    name="Rebalance",
                    marker=dict(symbol="diamond", size=7, color="#ff7f0e"),
                    showlegend=True,
                ),
                row=1, col=1,
            )

    # Row 2: drawdown
    if "drawdown" in snaps.columns:
        dd = snaps["drawdown"] * -100  # as positive % for area chart
        fig.add_trace(
            go.Scatter(
                x=snaps.index,
                y=dd,
                mode="lines",
                name="Drawdown %",
                fill="tozeroy",
                line=dict(color="#d62728", width=1),
                fillcolor="rgba(214, 39, 40, 0.15)",
            ),
            row=2, col=1,
        )

    fig.update_layout(
        title="RTMV Portfolio — Equity & Drawdown",
        height=460,
        margin=dict(t=44, b=20, l=60, r=20),
        legend=dict(orientation="h", x=0, y=1.08),
    )
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="DD (%)", row=2, col=1, autorange="reversed")
    return fig


def _build_weights_chart(snaps: pd.DataFrame) -> go.Figure | None:
    """Stacked area chart of actual portfolio weights over time."""
    weight_cols = [c for c in snaps.columns if c.startswith("weight_")]
    if not weight_cols:
        return None

    assets = [c[len("weight_"):] for c in weight_cols]
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]

    fig = go.Figure()
    for i, (col, asset) in enumerate(zip(weight_cols, assets)):
        w = snaps[col].ffill().fillna(0.0)
        fig.add_trace(go.Scatter(
            x=snaps.index,
            y=w,
            mode="lines",
            name=asset,
            stackgroup="weights",
            line=dict(width=0.5, color=palette[i % len(palette)]),
            fillcolor=palette[i % len(palette)].replace(")", ", 0.6)").replace("rgb", "rgba")
            if palette[i % len(palette)].startswith("rgb")
            else palette[i % len(palette)],
        ))

    fig.update_layout(
        title="Portfolio Weight Trajectory",
        height=280,
        margin=dict(t=44, b=20, l=60, r=20),
        yaxis=dict(title="Weight", tickformat=".0%"),
    )
    return fig


def _build_fills_table(fills: pd.DataFrame) -> go.Figure:
    """Compact table of recent fills."""
    recent = fills.tail(50).copy()
    if "timestamp" in recent.columns:
        recent["date"] = pd.to_datetime(recent["timestamp"]).dt.strftime("%Y-%m-%d")
    elif recent.index.name == "timestamp":
        recent["date"] = recent.index.strftime("%Y-%m-%d")
    else:
        recent["date"] = ""

    for col in ["quantity", "price", "slippage", "notional"]:
        if col in recent.columns:
            recent[col] = recent[col].round(4)

    cols = [c for c in ["date", "asset", "side", "quantity", "price", "slippage", "notional"]
            if c in recent.columns]

    fig = go.Figure(go.Table(
        header=dict(
            values=[c.capitalize() for c in cols],
            fill_color="#1f77b4",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=[recent[c].tolist() for c in cols],
            fill_color=[["#f8f9fa" if i % 2 == 0 else "white"
                         for i in range(len(recent))]],
            align="left",
            font=dict(size=11),
        ),
    ))
    fig.update_layout(
        title="Recent Fills (last 50)",
        height=300,
        margin=dict(t=44, b=10, l=10, r=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Panel entry point
# ---------------------------------------------------------------------------


def _panel_rtmv_portfolio(results_dir: Path) -> None:
    """Render the RTMV Portfolio Monitoring panel.

    Parameters
    ----------
    results_dir : Path
        Repository ``results/`` directory.  Expects ``rtmv_live/``
        subdirectory with ``snapshots.parquet`` and ``fills.parquet``.
    """
    import streamlit as st

    st.subheader("RTMV Portfolio Monitoring")

    rtmv_dir = results_dir / "rtmv_live"
    snaps_path = rtmv_dir / "snapshots.parquet"
    fills_path = rtmv_dir / "fills.parquet"

    if not snaps_path.exists():
        st.info(
            "No RTMV backtest found. Run:  \n"
            "```\nuv run python scripts/run_rtmv_live.py --mode backtest\n```"
        )
        return

    snaps = pd.read_parquet(snaps_path)
    if snaps.index.name != "timestamp" and "timestamp" in snaps.columns:
        snaps = snaps.set_index("timestamp")

    if snaps.empty:
        st.warning("Snapshots file is empty.")
        return

    # ------------------------------------------------------------------
    # Summary metrics row
    # ------------------------------------------------------------------
    eq = snaps["equity"].dropna()
    ret = eq.pct_change().dropna()
    ann = 252
    mu = float(ret.mean()) * ann
    sig = float(ret.std()) * (ann ** 0.5) + 1e-15
    sharpe = mu / sig
    peak = eq.cummax()
    mdd = float(((eq - peak) / (peak + 1e-15)).min())
    calmar = mu / (abs(mdd) + 1e-15)

    n_reb = int(snaps["n_rebalances"].iloc[-1]) if "n_rebalances" in snaps.columns else "—"
    is_halted = bool(snaps["is_halted"].iloc[-1]) if "is_halted" in snaps.columns else False
    current_eq = float(eq.iloc[-1])
    as_of = str(eq.index[-1])[:10]

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Equity", f"${current_eq:,.0f}")
    col2.metric("Sharpe", f"{sharpe:.3f}")
    col3.metric("Max DD", f"{mdd:.1%}")
    col4.metric("Ann Return", f"{mu:.1%}")
    col5.metric("Rebalances", str(n_reb))
    halt_label = "🔴 HALTED" if is_halted else "🟢 Active"
    col6.metric("Status", halt_label, delta=f"as of {as_of}", delta_color="off")

    # ------------------------------------------------------------------
    # Current weights
    # ------------------------------------------------------------------
    target_cols = [c for c in snaps.columns if c.startswith("target_")]
    if target_cols:
        last_reb = snaps[snaps.get("rebalance", False)] if "rebalance" in snaps.columns else snaps
        if not last_reb.empty:
            last_row = last_reb.iloc[-1]
            weights_now = {c[len("target_"):]: float(last_row[c]) for c in target_cols}
            st.markdown("**Current target weights (last rebalance)**")
            weight_cols_display = st.columns(len(weights_now))
            for i, (a, w) in enumerate(sorted(weights_now.items(), key=lambda x: -x[1])):
                weight_cols_display[i].metric(a, f"{w:.1%}")

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    eq_fig = _build_equity_chart(snaps)
    st.plotly_chart(eq_fig, use_container_width=True)

    wt_fig = _build_weights_chart(snaps)
    if wt_fig is not None:
        st.plotly_chart(wt_fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Fills table
    # ------------------------------------------------------------------
    if fills_path.exists():
        fills = pd.read_parquet(fills_path)
        if not fills.empty:
            fills_fig = _build_fills_table(fills)
            st.plotly_chart(fills_fig, use_container_width=True)
        else:
            st.caption("No fills recorded yet.")
    else:
        st.caption("Fills file not found — no trades executed yet.")
