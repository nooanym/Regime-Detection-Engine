"""
Regime Detection Engine — Live Dashboard

Reads results from results/<asset>/ directories and visualises them
interactively. Does NOT import from the rde package; all data is loaded
directly from parquet and txt files.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Regime Detection Engine",
    page_icon="📊",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# ---------------------------------------------------------------------------
# Fixed color palette (tab10-like, deterministic by state index)
# ---------------------------------------------------------------------------
_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # yellow-green
    "#17becf",  # teal
]


def _label_color(labels: list[str]) -> dict[str, str]:
    """Map sorted unique labels to palette colors."""
    sorted_labels = sorted(set(labels))
    return {lbl: _PALETTE[i % len(_PALETTE)] for i, lbl in enumerate(sorted_labels)}


# ---------------------------------------------------------------------------
# Data loading helpers (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _list_assets() -> list[str]:
    """Return sorted list of asset folders found under results/."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        d.name
        for d in RESULTS_DIR.iterdir()
        if d.is_dir() and d.name.lower() != "cache"
    )


@st.cache_data(show_spinner=False)
def _load_regimes(asset: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / asset / "regimes.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _load_analytics(asset: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / asset / "regime_analytics.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _load_signals(asset: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / asset / "signals.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _load_diagnostics(asset: str) -> str | None:
    path = RESULTS_DIR / asset / "diagnostics.txt"
    if not path.exists():
        return None
    return path.read_text()


@st.cache_data(show_spinner=False)
def _load_comparison() -> dict[str, pd.DataFrame | None]:
    comp_dir = RESULTS_DIR / "comparison"
    keys = ["return_corr", "score_corr", "cond_return_corr", "co_occurrence"]
    out: dict[str, pd.DataFrame | None] = {}
    for k in keys:
        p = comp_dir / f"{k}.parquet"
        out[k] = pd.read_parquet(p) if p.exists() else None
    return out


@st.cache_data(show_spinner=False)
def _load_wf_backtest(asset: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    bt_path = RESULTS_DIR / asset / "wf_backtest.parquet"
    fold_path = RESULTS_DIR / asset / "wf_fold_summary.parquet"
    bt = pd.read_parquet(bt_path) if bt_path.exists() else None
    folds = pd.read_parquet(fold_path) if fold_path.exists() else None
    return bt, folds


@st.cache_data(show_spinner=False)
def _load_analysis_report(asset: str) -> dict | None:
    """Load Phase 31 analysis report JSON if it exists."""
    import json
    path = RESULTS_DIR / asset / "analysis" / "analysis_report.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from panels_analysis import (
        _panel_execution,
        _panel_factor_analysis,
        _panel_tail_risk,
        _panel_transition,
        _panel_correlation,
        _panel_cointegration,
        _panel_portfolio,
        _panel_analysis_markdown,
    )
    _ANALYSIS_PANELS_AVAILABLE = True
except ImportError:
    _ANALYSIS_PANELS_AVAILABLE = False

try:
    from panels_live import _panel_live_feed
    _LIVE_PANEL_AVAILABLE = True
except ImportError:
    _LIVE_PANEL_AVAILABLE = False

try:
    from panels_trade import _panel_trade_history
    _TRADE_PANEL_AVAILABLE = True
except ImportError:
    _TRADE_PANEL_AVAILABLE = False

try:
    from panels_concordance import _panel_concordance
    _CONCORDANCE_PANEL_AVAILABLE = True
except ImportError:
    _CONCORDANCE_PANEL_AVAILABLE = False

try:
    from panels_rtmv import _panel_rtmv_portfolio
    _RTMV_PANEL_AVAILABLE = True
except ImportError:
    _RTMV_PANEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------


def _panel_overview(regimes_df: pd.DataFrame) -> None:
    st.subheader("Regime Overview")

    n_states = int(regimes_df["regime"].nunique())
    n_obs = len(regimes_df)
    date_start = str(regimes_df.index[0])[:19]
    date_end = str(regimes_df.index[-1])[:19]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("States Detected", n_states)
    col2.metric("Total Observations", f"{n_obs:,}")
    col3.metric("Start", date_start)
    col4.metric("End", date_end)

    # Pie chart: regime distribution by label
    label_counts = regimes_df["regime_label"].value_counts().reset_index()
    label_counts.columns = ["label", "count"]
    color_map = _label_color(label_counts["label"].tolist())
    colors = [color_map[lbl] for lbl in label_counts["label"]]

    fig = go.Figure(
        go.Pie(
            labels=label_counts["label"],
            values=label_counts["count"],
            marker=dict(colors=colors),
            textinfo="label+percent",
            hole=0.35,
        )
    )
    fig.update_layout(
        title="Regime Distribution",
        legend_title="Regime Label",
        margin=dict(t=50, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def _panel_price_with_regimes(regimes_df: pd.DataFrame, asset: str) -> None:
    st.subheader("Price with Regimes")

    labels = regimes_df["regime_label"].unique().tolist()
    color_map = _label_color(labels)

    fig = go.Figure()

    for label in sorted(labels):
        mask = regimes_df["regime_label"] == label
        subset = regimes_df[mask]
        # Use a scatter with markers=None so gaps between non-contiguous
        # segments don't draw phantom connecting lines
        fig.add_trace(
            go.Scattergl(
                x=subset.index,
                y=subset["Close"],
                mode="lines",
                name=label,
                line=dict(color=color_map[label], width=1),
                legendgroup=label,
            )
        )

    fig.update_layout(
        title=f"{asset} — Close Price by Regime",
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True),
            type="date",
        ),
        yaxis=dict(title="Price (USD)"),
        legend_title="Regime Label",
        hovermode="x unified",
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _panel_analytics_table(analytics_df: pd.DataFrame) -> None:
    st.subheader("Regime Analytics")

    def _style_sharpe(val: float) -> str:
        if val > 0:
            return "color: #2ca02c; font-weight: bold"
        elif val < 0:
            return "color: #d62728; font-weight: bold"
        return ""

    numeric_cols = analytics_df.select_dtypes(include="number").columns.tolist()

    styled = analytics_df.style.format(
        {c: "{:.4f}" for c in numeric_cols if c not in ("count",)},
        na_rep="—",
    )

    if "sharpe_ann" in analytics_df.columns:
        styled = styled.applymap(_style_sharpe, subset=["sharpe_ann"])

    st.dataframe(styled, use_container_width=True)


def _panel_signal_chart(regimes_df: pd.DataFrame, signals_df: pd.DataFrame, asset: str) -> None:
    st.subheader("Signal Chart")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=(f"{asset} Close Price", "Signal"),
    )

    # Top: Close price
    fig.add_trace(
        go.Scattergl(
            x=regimes_df.index,
            y=regimes_df["Close"],
            mode="lines",
            name="Close",
            line=dict(color="#1f77b4", width=1),
        ),
        row=1,
        col=1,
    )

    # Bottom: Signal
    fig.add_trace(
        go.Scattergl(
            x=signals_df.index,
            y=signals_df["signal"],
            mode="lines",
            name="Signal",
            line=dict(color="#ff7f0e", width=1),
        ),
        row=2,
        col=1,
    )

    # Reference lines on signal panel
    for y_val, dash, color, name in [
        (0.25, "dash", "#2ca02c", "+0.25"),
        (-0.25, "dash", "#d62728", "-0.25"),
        (0.0, "solid", "#7f7f7f", "0"),
    ]:
        fig.add_hline(
            y=y_val,
            line=dict(dash=dash, color=color, width=1),
            row=2,
            col=1,
            annotation_text=name,
            annotation_position="right",
        )

    fig.update_layout(
        height=600,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.05),
        margin=dict(t=60, b=40),
        xaxis2=dict(rangeslider=dict(visible=True), type="date"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _panel_transition_heatmap(regimes_df: pd.DataFrame) -> None:
    st.subheader("Regime Transition Heatmap")

    states = regimes_df["regime"].values
    labels_series = regimes_df["regime_label"]

    # Build empirical transition matrix
    unique_states = sorted(regimes_df["regime"].unique())
    n = len(unique_states)
    state_to_idx = {s: i for i, s in enumerate(unique_states)}
    counts = np.zeros((n, n), dtype=float)

    for t in range(len(states) - 1):
        i = state_to_idx[states[t]]
        j = state_to_idx[states[t + 1]]
        counts[i, j] += 1

    # Row-normalise (avoid divide-by-zero)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transmat = counts / row_sums

    # Build axis labels: "state_idx (label)"
    axis_labels = []
    for s in unique_states:
        # Get the most common label for this regime integer
        lbl = labels_series[regimes_df["regime"] == s].mode()
        lbl_str = lbl.iloc[0] if len(lbl) > 0 else str(s)
        axis_labels.append(f"{s} ({lbl_str})")

    text_vals = [[f"{transmat[i, j]:.2f}" for j in range(n)] for i in range(n)]

    fig = go.Figure(
        go.Heatmap(
            z=transmat,
            x=axis_labels,
            y=axis_labels,
            text=text_vals,
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Probability"),
        )
    )
    fig.update_layout(
        title="Empirical Regime Transition Matrix",
        xaxis_title="To State",
        yaxis_title="From State",
        yaxis=dict(autorange="reversed"),
        margin=dict(t=60, b=60, l=100, r=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _panel_performance(regimes_df: pd.DataFrame, signals_df: pd.DataFrame, asset: str) -> None:
    st.subheader("Performance Metrics")

    # Align indices
    common_idx = regimes_df.index.intersection(signals_df.index)
    r = regimes_df.loc[common_idx, "log_return"]
    sig = signals_df.loc[common_idx, "signal"]

    # Strategy equity curve
    position = sig.clip(-1, 1).shift(1).fillna(0)
    strategy_ret = (position * r).fillna(0)
    equity = (1 + strategy_ret).cumprod() * 10_000

    # Buy-and-hold
    bh_ret = r.fillna(0)
    bh_equity = (1 + bh_ret).cumprod() * 10_000

    # Summary metrics
    strat_total = float((equity.iloc[-1] / 10_000 - 1) * 100)
    bh_total = float((bh_equity.iloc[-1] / 10_000 - 1) * 100)
    outperformance = strat_total - bh_total

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Strategy Total Return",
        f"{strat_total:.2f}%",
        delta=f"{outperformance:+.2f}% vs B&H",
    )
    col2.metric("Buy-and-Hold Total Return", f"{bh_total:.2f}%")
    col3.metric(
        "Outperformance",
        f"{outperformance:+.2f}%",
        delta_color="normal",
    )

    # Equity curve chart
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="Strategy",
            line=dict(color="#2ca02c", width=1.5),
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=bh_equity.index,
            y=bh_equity.values,
            mode="lines",
            name="Buy & Hold",
            line=dict(color="#1f77b4", width=1.5, dash="dot"),
        )
    )
    fig.update_layout(
        title=f"{asset} — Strategy vs Buy-and-Hold (starting $10,000)",
        xaxis=dict(title="Date", rangeslider=dict(visible=True), type="date"),
        yaxis=dict(title="Portfolio Value ($)"),
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _panel_cross_asset(comp: dict[str, pd.DataFrame | None]) -> None:
    st.subheader("Cross-Asset Regime Correlation")
    st.caption(
        "Run `uv run rde compare --result results/BTC-USD --result results/ETH-USD "
        "--result results/SPY` to generate this data."
    )

    ret_corr = comp.get("return_corr")
    score_corr = comp.get("score_corr")
    cond_corr = comp.get("cond_return_corr")

    if ret_corr is None:
        st.warning(
            "`results/comparison/return_corr.parquet` not found. "
            "Run `rde compare` to generate comparison outputs."
        )
        return

    col1, col2 = st.columns(2)

    def _corr_heatmap(df: pd.DataFrame, title: str) -> go.Figure:
        fig = go.Figure(
            go.Heatmap(
                z=df.values,
                x=df.columns.tolist(),
                y=df.index.tolist(),
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=[[f"{v:+.3f}" for v in row] for row in df.values],
                texttemplate="%{text}",
                showscale=True,
            )
        )
        fig.update_layout(
            title=title,
            margin=dict(t=50, b=20, l=60, r=20),
            height=300,
        )
        return fig

    with col1:
        st.plotly_chart(
            _corr_heatmap(ret_corr, "Return Correlation (unconditional)"),
            use_container_width=True,
        )
    with col2:
        if score_corr is not None:
            st.plotly_chart(
                _corr_heatmap(score_corr, "Regime-Score Correlation"),
                use_container_width=True,
            )

    if cond_corr is not None and len(cond_corr) > 0:
        st.markdown("**Conditional return correlations by joint regime direction**")
        directions = cond_corr["direction"].unique() if "direction" in cond_corr.columns else []
        pairs = (
            cond_corr[["asset_a", "asset_b"]].drop_duplicates()
            if "asset_a" in cond_corr.columns else pd.DataFrame()
        )
        if not pairs.empty:
            for _, row in pairs.iterrows():
                sym_a, sym_b = row["asset_a"], row["asset_b"]
                pair_df = cond_corr[
                    (cond_corr["asset_a"] == sym_a) & (cond_corr["asset_b"] == sym_b)
                ][["direction", "correlation"]].set_index("direction")
                st.markdown(f"*{sym_a} × {sym_b}*")
                st.dataframe(
                    pair_df.style.format({"correlation": "{:+.4f}"})
                    .background_gradient(cmap="RdBu", vmin=-1, vmax=1),
                    use_container_width=False,
                )


def _panel_wf_backtest(
    wf_bt_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    asset: str,
) -> None:
    st.subheader("Walk-Forward OOS Backtest")
    st.caption(
        "All signals are generated using only data available at each bar "
        "(causal online filtering). No in-sample leakage."
    )

    # ── Metrics row ──────────────────────────────────────────────────────────
    net_ret = wf_bt_df["net_return"].dropna()
    equity = wf_bt_df["equity_curve"].dropna()
    bars_per_year = 8760.0
    if len(net_ret) > 1:
        sharpe = (net_ret.mean() / net_ret.std(ddof=1)) * np.sqrt(bars_per_year)
        peak = equity.cummax()
        dd = ((equity - peak) / peak).min()
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (bars_per_year / len(equity)) - 1.0
        total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    else:
        sharpe = dd = cagr = total_ret = float("nan")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("OOS Bars", f"{len(net_ret):,}")
    c2.metric("OOS Folds", str(len(fold_df)) if fold_df is not None else "—")
    c3.metric("Sharpe (ann)", f"{sharpe:+.3f}" if np.isfinite(sharpe) else "N/A")
    c4.metric("Max Drawdown", f"{-dd*100:.1f}%" if np.isfinite(dd) else "N/A")
    c5.metric("Total Return", f"{total_ret*100:+.1f}%" if np.isfinite(total_ret) else "N/A")

    # ── Equity curve vs buy-and-hold ─────────────────────────────────────────
    initial = equity.iloc[0] if len(equity) > 0 else 10_000.0
    common_idx = regimes_df.index.intersection(equity.index)
    if len(common_idx) > 1 and "Close" in regimes_df.columns:
        bh_close = regimes_df.loc[common_idx, "Close"]
        bh_equity = initial * bh_close / bh_close.iloc[0]
    else:
        bh_equity = None

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=equity.index, y=equity.values,
            mode="lines", name="OOS Strategy",
            line=dict(color="#2ca02c", width=1.5),
        )
    )
    if bh_equity is not None:
        fig.add_trace(
            go.Scattergl(
                x=bh_equity.index, y=bh_equity.values,
                mode="lines", name="Buy & Hold",
                line=dict(color="#1f77b4", width=1.5, dash="dot"),
            )
        )
    fig.update_layout(
        title=f"{asset} — Walk-Forward OOS Equity Curve",
        xaxis=dict(title="Date", rangeslider=dict(visible=True), type="date"),
        yaxis=dict(title="Portfolio Value ($)"),
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-fold summary ─────────────────────────────────────────────────────
    if fold_df is not None and len(fold_df) > 0:
        st.markdown("**Per-fold summary**")
        display_df = fold_df.copy()
        display_df.index = display_df.index.strftime("%Y-%m-%d")
        st.dataframe(
            display_df.style.format(
                {"train_log_lik": "{:.1f}", "train_bars": "{:.0f}", "oos_bars": "{:.0f}"}
            ),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def _data_freshness(asset: str) -> str:
    """Return a human-readable age string for the regimes parquet."""
    import time
    path = RESULTS_DIR / asset / "regimes.parquet"
    if not path.exists():
        return "no data"
    age_s = time.time() - path.stat().st_mtime
    if age_s < 3600:
        return f"{int(age_s / 60)}m ago"
    if age_s < 86400:
        return f"{int(age_s / 3600)}h ago"
    return f"{int(age_s / 86400)}d ago"


def _panel_current_state(
    regimes_df: pd.DataFrame,
    signals_df: pd.DataFrame | None,
    asset: str,
) -> None:
    """Show the most recent regime, price, signal, and posterior probabilities."""
    st.subheader("Current State")

    latest = regimes_df.iloc[-1]
    current_regime = int(latest["regime"])
    current_label = str(latest["regime_label"])
    current_price = float(latest["Close"])
    current_date = str(regimes_df.index[-1])[:16]

    # Streak: consecutive bars in current regime
    states = regimes_df["regime"].values
    streak = 1
    for i in range(len(states) - 2, -1, -1):
        if states[i] == current_regime:
            streak += 1
        else:
            break
    streak_label = f"≈{streak/24:.1f}d" if streak >= 48 else f"≈{streak}h"

    # Latest signal
    current_signal = None
    if signals_df is not None and "signal" in signals_df.columns:
        common = regimes_df.index.intersection(signals_df.index)
        if len(common) > 0:
            current_signal = float(signals_df.loc[common[-1], "signal"])

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Price", f"${current_price:,.0f}")
    col2.metric("Regime", f"#{current_regime}", delta=current_label, delta_color="off")
    col3.metric("Streak", f"{streak} bars", delta=streak_label, delta_color="off")
    if current_signal is not None:
        direction = "↑ Long" if current_signal > 0.25 else ("↓ Short" if current_signal < -0.25 else "→ Flat")
        delta_color = "normal" if current_signal > 0.25 else ("inverse" if current_signal < -0.25 else "off")
        col4.metric("Signal", f"{current_signal:+.3f}", delta=direction, delta_color=delta_color)
    col5.metric("As Of", current_date)

    # Posterior probability bar chart
    proba_cols = [c for c in regimes_df.columns if c.startswith("regime_proba_")]
    if proba_cols:
        probas = [float(latest[c]) for c in proba_cols]
        labels = [regimes_df.loc[regimes_df["regime"] == k, "regime_label"].iloc[0]
                  if (regimes_df["regime"] == k).any() else f"Regime {k}"
                  for k in range(len(probas))]
        fig = go.Figure(go.Bar(
            x=[f"#{k} {labels[k]}" for k in range(len(probas))],
            y=probas,
            marker_color=[_PALETTE[k % len(_PALETTE)] for k in range(len(probas))],
            text=[f"{p:.1%}" for p in probas],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"{asset} — Posterior Regime Probabilities (latest bar: {current_date})",
            yaxis=dict(title="Probability", range=[0, 1.15]),
            height=260,
            margin=dict(t=40, b=20, l=40, r=20),
            xaxis_tickangle=-20,
        )
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("Regime Detection Engine — Live Dashboard")

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Controls")

        assets = _list_assets()
        if not assets:
            st.error(
                "No results found under `results/`. "
                "Run `uv run rde run --config configs/<asset>.yaml` first."
            )
            st.stop()

        asset = st.selectbox("Asset", assets, index=0)

        # Data freshness badge
        freshness = _data_freshness(asset)
        st.caption(f"Data: **{freshness}** — rerun app to refresh cache")

        all_panels = [
            "Current State",           # Phase 33
            "Live Feed",               # Phase 34 — OnlineDecoder
            "Regime Overview",
            "Price with Regimes",
            "Regime Analytics Table",
            "Signal Chart",
            "Regime Transition Heatmap",
            "Performance Metrics",
            "Walk-Forward OOS Backtest",
            "Cross-Asset Correlation",
            # Phase 32 — analysis pipeline panels
            "Execution & Market Impact",
            "Factor Analysis",
            "Tail Risk (EVT)",
            "Regime Transitions",
            "Correlation Structure",
            "Cointegration",
            "Portfolio Optimisation",
            "Full Analysis Report",
            # Phase 36 — paper-trading + concordance
            "Trade History",
            "Regime Concordance",
            # Phase 49 — RTMV portfolio monitoring
            "RTMV Portfolio",
        ]
        selected_panels = st.multiselect(
            "Panels to display",
            options=all_panels,
            default=all_panels[:10],   # Current State + Live Feed + original 8
        )

        st.markdown("---")

        # Date range filter
        st.markdown("**Date range**")

    # ------------------------------------------------------------------
    # Load data (before date filter so we know the full range)
    # ------------------------------------------------------------------
    with st.spinner(f"Loading results for {asset}…"):
        regimes_df = _load_regimes(asset)
        analytics_df = _load_analytics(asset)
        signals_df = _load_signals(asset)
        analysis_report = _load_analysis_report(asset)
        feature_names_from_report = (
            analysis_report.get("feature_names", []) if analysis_report else []
        )

    if regimes_df is None:
        st.error(f"`results/{asset}/regimes.parquet` not found. Run the pipeline first.")
        st.stop()

    # Date range filter (in sidebar, but needs data loaded first)
    with st.sidebar:
        import datetime as _dt
        idx = regimes_df.index
        min_dt = idx.min().to_pydatetime().date()
        max_dt = idx.max().to_pydatetime().date()
        date_from, date_to = st.date_input(
            "From / To",
            value=(min_dt, max_dt),
            min_value=min_dt,
            max_value=max_dt,
            label_visibility="collapsed",
        )
        if isinstance(date_from, _dt.date) and isinstance(date_to, _dt.date):
            _mask = (idx.date >= date_from) & (idx.date <= date_to)
            regimes_df = regimes_df.loc[_mask]
            if signals_df is not None:
                signals_df = signals_df.loc[signals_df.index.isin(regimes_df.index)]

    # ------------------------------------------------------------------
    # Render selected panels
    # ------------------------------------------------------------------

    if "Current State" in selected_panels:
        st.markdown("---")
        _panel_current_state(_load_regimes(asset), _load_signals(asset), asset)

    if "Live Feed" in selected_panels:
        st.markdown("---")
        if _LIVE_PANEL_AVAILABLE:
            _panel_live_feed(asset, RESULTS_DIR / asset)
        else:
            st.warning("panels_live.py not found — refresh the app.")

    if "Regime Overview" in selected_panels:
        st.markdown("---")
        _panel_overview(regimes_df)

    if "Price with Regimes" in selected_panels:
        st.markdown("---")
        _panel_price_with_regimes(regimes_df, asset)

    if "Regime Analytics Table" in selected_panels:
        st.markdown("---")
        if analytics_df is not None:
            _panel_analytics_table(analytics_df)
        else:
            st.warning(
                f"`results/{asset}/regime_analytics.parquet` not found — skipping analytics table."
            )

    if "Signal Chart" in selected_panels:
        st.markdown("---")
        if signals_df is not None:
            _panel_signal_chart(regimes_df, signals_df, asset)
        else:
            st.warning(
                f"`results/{asset}/signals.parquet` not found — skipping signal chart."
            )

    if "Regime Transition Heatmap" in selected_panels:
        st.markdown("---")
        _panel_transition_heatmap(regimes_df)

    if "Performance Metrics" in selected_panels:
        st.markdown("---")
        if signals_df is not None:
            _panel_performance(regimes_df, signals_df, asset)
        else:
            st.warning(
                f"`results/{asset}/signals.parquet` not found — skipping performance metrics."
            )

    if "Walk-Forward OOS Backtest" in selected_panels:
        st.markdown("---")
        wf_bt_df, wf_fold_df = _load_wf_backtest(asset)
        if wf_bt_df is not None:
            _panel_wf_backtest(wf_bt_df, wf_fold_df, regimes_df, asset)
        else:
            st.warning(
                f"`results/{asset}/wf_backtest.parquet` not found. "
                "Run with `--wf-backtest` to generate it."
            )

    if "Cross-Asset Correlation" in selected_panels:
        st.markdown("---")
        comp = _load_comparison()
        if any(v is not None for v in comp.values()):
            _panel_cross_asset(comp)
        else:
            st.warning(
                "No cross-asset comparison data found. "
                "Run `uv run rde compare --result results/BTC-USD --result results/ETH-USD` "
                "to generate it."
            )

    if "Execution & Market Impact" in selected_panels:
        st.markdown("---")
        if _ANALYSIS_PANELS_AVAILABLE:
            if analysis_report is not None:
                _panel_execution(analysis_report)
            else:
                st.warning(
                    f"No analysis report found for {asset}. "
                    "Run `uv run rde analyse --config configs/<asset>.yaml` to generate it."
                )
        else:
            st.warning("panels_analysis.py not found — run `rde analyse` and refresh.")

    if "Factor Analysis" in selected_panels:
        st.markdown("---")
        if _ANALYSIS_PANELS_AVAILABLE and analysis_report is not None:
            _panel_factor_analysis(analysis_report, feature_names_from_report)
        elif _ANALYSIS_PANELS_AVAILABLE:
            st.warning(f"No analysis report found for {asset}.")

    if "Tail Risk (EVT)" in selected_panels:
        st.markdown("---")
        if _ANALYSIS_PANELS_AVAILABLE and analysis_report is not None:
            _panel_tail_risk(analysis_report)
        elif _ANALYSIS_PANELS_AVAILABLE:
            st.warning(f"No analysis report found for {asset}.")

    if "Regime Transitions" in selected_panels:
        st.markdown("---")
        if _ANALYSIS_PANELS_AVAILABLE and analysis_report is not None:
            _panel_transition(analysis_report)
        elif _ANALYSIS_PANELS_AVAILABLE:
            st.warning(f"No analysis report found for {asset}.")

    if "Correlation Structure" in selected_panels:
        st.markdown("---")
        if _ANALYSIS_PANELS_AVAILABLE and analysis_report is not None:
            _panel_correlation(analysis_report, feature_names_from_report)
        elif _ANALYSIS_PANELS_AVAILABLE:
            st.warning(f"No analysis report found for {asset}.")

    if "Cointegration" in selected_panels:
        st.markdown("---")
        if _ANALYSIS_PANELS_AVAILABLE and analysis_report is not None:
            _panel_cointegration(analysis_report, feature_names_from_report)
        elif _ANALYSIS_PANELS_AVAILABLE:
            st.warning(f"No analysis report found for {asset}.")

    if "Portfolio Optimisation" in selected_panels:
        st.markdown("---")
        if _ANALYSIS_PANELS_AVAILABLE and analysis_report is not None:
            _panel_portfolio(analysis_report, feature_names_from_report)
        elif _ANALYSIS_PANELS_AVAILABLE:
            st.warning(f"No analysis report found for {asset}.")

    if "Full Analysis Report" in selected_panels:
        st.markdown("---")
        if _ANALYSIS_PANELS_AVAILABLE:
            _panel_analysis_markdown(RESULTS_DIR / asset)
        else:
            st.warning("panels_analysis.py not found.")

    if "Trade History" in selected_panels:
        st.markdown("---")
        if _TRADE_PANEL_AVAILABLE:
            _panel_trade_history(asset, RESULTS_DIR / asset)
        else:
            st.warning("panels_trade.py not found — refresh.")

    if "Regime Concordance" in selected_panels:
        st.markdown("---")
        if _CONCORDANCE_PANEL_AVAILABLE:
            _panel_concordance(RESULTS_DIR / "comparison")
        else:
            st.warning("panels_concordance.py not found.")

    if "RTMV Portfolio" in selected_panels:
        st.markdown("---")
        if _RTMV_PANEL_AVAILABLE:
            _panel_rtmv_portfolio(RESULTS_DIR)
        else:
            st.warning("panels_rtmv.py not found — refresh.")

    # ------------------------------------------------------------------
    # Raw diagnostics expander
    # ------------------------------------------------------------------
    diagnostics_txt = _load_diagnostics(asset)
    if diagnostics_txt is not None:
        with st.expander("Raw Diagnostics (diagnostics.txt)", expanded=False):
            st.code(diagnostics_txt, language="text")


if __name__ == "__main__":
    main()
