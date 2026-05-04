"""
Regime Detection Engine — Regime Concordance Panel

Two-tier architecture:
  _build_*_figure(data, ...) — pure functions returning go.Figure (no Streamlit calls)
  _panel_concordance(results_dir)  — UI layer calling builders + st.*

All panel functions guard against missing / None data and fewer than 2 assets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# Make rde.* importable when run from the app/ directory.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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

_GREEN = "#2ca02c"
_RED = "#d62728"
_GREY = "#7f7f7f"


# ---------------------------------------------------------------------------
# 1. Sync heatmap
# ---------------------------------------------------------------------------


def _build_sync_heatmap_figure(sync_matrix_df: pd.DataFrame, title: str) -> go.Figure:
    """Plotly heatmap of the regime sync matrix (values in [0, 1]).

    Parameters
    ----------
    sync_matrix_df:
        Square DataFrame whose (i, j) entry is the direction-agreement fraction
        for assets i and j.  Diagonal entries should be 1.0.
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    assets = list(sync_matrix_df.columns)
    z = sync_matrix_df.values.tolist()
    text = [[f"{v:.0%}" for v in row] for row in sync_matrix_df.values]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=assets,
            y=assets,
            text=text,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Agreement"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Asset",
        yaxis_title="Asset",
        yaxis=dict(autorange="reversed"),
        margin=dict(t=60, b=60, l=80, r=40),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Rolling concordance line chart
# ---------------------------------------------------------------------------


def _build_rolling_concordance_figure(
    rolling_df: pd.DataFrame,
    asset_pairs: list[str],
    title: str,
) -> go.Figure:
    """Line chart of rolling direction-agreement for each asset pair.

    Parameters
    ----------
    rolling_df:
        DataFrame with one column per asset pair (e.g. ``"BTC-USD__ETH-USD"``).
        Values are rolling fractions in [0, 1]; leading NaNs are expected.
    asset_pairs:
        Column names to plot.  Traces are drawn in the order given.
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    for i, col in enumerate(asset_pairs):
        if col not in rolling_df.columns:
            continue
        series = rolling_df[col].dropna()
        # Friendly label: replace "__" separator with " / "
        label = col.replace("__", " / ")
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=label,
                line=dict(color=_PALETTE[i % len(_PALETTE)], width=1.5),
            )
        )
    fig.update_layout(
        title=title,
        xaxis=dict(title="Date", type="date"),
        yaxis=dict(title="Direction Agreement", range=[0, 1]),
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
        margin=dict(t=60, b=60, l=60, r=20),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Lead-lag bar chart
# ---------------------------------------------------------------------------


def _build_lead_lag_figure(pairwise_list: list, title: str) -> go.Figure:
    """Horizontal bar chart of lead-lag days for each asset pair.

    Parameters
    ----------
    pairwise_list:
        List of ``PairwiseConcordance`` objects (or anything with
        ``.asset_a``, ``.asset_b``, ``.lead_lag_days`` attributes).
        An empty list is handled gracefully.
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    if not pairwise_list:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    labels = [f"{p.asset_a} → {p.asset_b}" for p in pairwise_list]
    values = [p.lead_lag_days for p in pairwise_list]
    colors = [
        _GREEN if v > 0 else (_RED if v < 0 else _GREY)
        for v in values
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(title="Lead-Lag (bars)"),
        yaxis=dict(title=""),
        margin=dict(t=60, b=40, l=160, r=40),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Panel entry point
# ---------------------------------------------------------------------------


def _panel_concordance(results_dir: Path) -> None:
    """Render the Regime Concordance panel.

    Parameters
    ----------
    results_dir:
        Path to the ``results/comparison/`` directory produced by
        ``rde compare``.  Must contain ``aligned_scores.parquet``.
    """
    # Import streamlit inside the function body so the module can be imported
    # without a running Streamlit context (required by the test suite).
    import streamlit as st

    st.subheader("Regime Concordance")
    st.caption(
        "Cross-asset regime synchronisation. "
        "Run `uv run rde compare --result results/BTC-USD --result results/ETH-USD "
        "--result results/SPY --daily` to generate this data."
    )

    scores_path = results_dir / "aligned_scores.parquet"
    if not scores_path.exists():
        st.info(
            "`results/comparison/aligned_scores.parquet` not found. "
            "Run `uv run rde compare --result results/<asset> ...` to generate concordance data."
        )
        return

    aligned_scores_df = pd.read_parquet(scores_path)

    try:
        from rde.analysis.regime_concordance import (
            compute_concordance,
            rolling_concordance_series,
        )
        result = compute_concordance(aligned_scores_df)
    except Exception as exc:
        st.warning(
            f"Could not compute concordance ({exc}). "
            "Ensure at least 2 assets and 30 common bars are present."
        )
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    n_pairs = len(result.pairwise)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Global Concordance", f"{result.global_agreement:.1%}")
    c2.metric("N Assets", str(len(result.assets)))
    c3.metric("Common Bars", f"{result.n_common_bars:,}")
    c4.metric("Pairs", str(n_pairs))

    # ── Sync heatmap ──────────────────────────────────────────────────────────
    st.plotly_chart(
        _build_sync_heatmap_figure(
            result.regime_sync_matrix,
            title="Regime Synchronisation Matrix (direction agreement)",
        ),
        use_container_width=True,
    )

    # ── Rolling concordance ───────────────────────────────────────────────────
    rolling_df = rolling_concordance_series(aligned_scores_df)
    pair_cols = list(rolling_df.columns)
    if pair_cols:
        st.plotly_chart(
            _build_rolling_concordance_figure(
                rolling_df,
                asset_pairs=pair_cols,
                title="Rolling Direction Agreement (20-bar window)",
            ),
            use_container_width=True,
        )

    # ── Lead-lag ──────────────────────────────────────────────────────────────
    if result.pairwise:
        st.plotly_chart(
            _build_lead_lag_figure(
                result.pairwise,
                title="Lead-Lag Analysis (positive = asset_a leads asset_b)",
            ),
            use_container_width=True,
        )

    # ── Pairwise table ────────────────────────────────────────────────────────
    st.markdown("**Pairwise Concordance Table**")
    table_rows = [
        {
            "Asset A": p.asset_a,
            "Asset B": p.asset_b,
            "Direction Agreement %": f"{p.direction_agreement:.1%}",
            "Score Correlation": round(p.score_correlation, 4),
            "Lead-Lag (bars)": p.lead_lag_days,
        }
        for p in result.pairwise
    ]
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
