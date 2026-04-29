"""
Regime Detection Engine — Analysis Panels

Two-tier architecture:
  _build_*_figure(data) — pure functions returning go.Figure (no Streamlit calls)
  _panel_*(report, feature_names) — UI layer calling builders + st.*

All panel functions guard against missing / None data.
"""

from __future__ import annotations

import math
from pathlib import Path

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
# 1. Execution / Market Impact
# ---------------------------------------------------------------------------

def _build_slippage_figure(execution: dict, n_regimes: int) -> go.Figure:
    """Bar chart of expected slippage by regime.

    Parameters
    ----------
    execution:
        The ``execution`` sub-dict from the analysis report.
    n_regimes:
        Number of regimes (used for x-axis range guard).

    Returns
    -------
    go.Figure
    """
    slippage_map: dict = execution.get("slippage_by_regime", {})
    regimes = sorted(slippage_map.keys(), key=lambda k: int(k))
    x_labels = [f"Regime {k}" for k in regimes]
    y_values = [slippage_map[k] for k in regimes]
    colors = [_PALETTE[int(k) % len(_PALETTE)] for k in regimes]

    fig = go.Figure(
        go.Bar(x=x_labels, y=y_values, marker_color=colors)
    )
    fig.update_layout(
        title="Expected Slippage by Regime (order size = 0.01)",
        yaxis_title="Slippage (fraction)",
        showlegend=False,
    )
    return fig


def _panel_execution(report: dict) -> None:
    """Render the Execution & Market Impact panel."""
    execution = report.get("execution")
    if execution is None:
        st.info("No execution data found. Run the analysis pipeline to generate it.")
        return

    st.subheader("Execution & Market Impact")
    st.metric("Regimes Modelled", execution.get("n_regimes", "—"))

    n_regimes = execution.get("n_regimes", len(execution.get("slippage_by_regime", {})))
    if execution.get("slippage_by_regime"):
        st.plotly_chart(_build_slippage_figure(execution, n_regimes), use_container_width=True)

    twap = execution.get("twap_example")
    if twap:
        slices = list(range(1, len(twap) + 1))
        fig = go.Figure(
            go.Bar(x=[f"Slice {i}" for i in slices], y=twap, marker_color=_PALETTE[0])
        )
        fig.update_layout(
            title="TWAP Schedule Example",
            yaxis_title="Notional per Slice",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 2. Factor Analysis
# ---------------------------------------------------------------------------

def _build_factor_figure(factor_result: dict, feature_names: list[str]) -> go.Figure:
    """Grouped bar chart of explained variance ratio per factor per regime.

    Parameters
    ----------
    factor_result:
        The ``factor_result`` sub-dict from the analysis report.
    feature_names:
        List of feature names (used for context / fallback labels).

    Returns
    -------
    go.Figure
    """
    evr_list: list[list[float]] = factor_result.get("explained_variance_ratios", [])
    if not evr_list:
        return go.Figure()

    n_factors = max(len(row) for row in evr_list)
    factor_labels = [str(i + 1) for i in range(n_factors)]

    fig = go.Figure()
    for k, evr in enumerate(evr_list):
        padded = list(evr) + [0.0] * (n_factors - len(evr))
        fig.add_trace(
            go.Bar(
                name=f"Regime {k}",
                x=factor_labels,
                y=padded,
                marker_color=_PALETTE[k % len(_PALETTE)],
            )
        )

    fig.update_layout(
        title="Explained Variance Ratio by Factor and Regime",
        barmode="group",
        xaxis_title="Factor Index",
        yaxis=dict(title="Explained Variance Ratio", range=[0, 1]),
    )
    return fig


def _panel_factor_analysis(report: dict, feature_names: list[str]) -> None:
    """Render the Factor Analysis panel."""
    factor_result = report.get("factor_result")
    if factor_result is None:
        st.info("No factor analysis data found. Run the analysis pipeline to generate it.")
        return

    st.subheader("Factor Analysis")
    fig = _build_factor_figure(factor_result, feature_names)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 3. Tail Risk (EVT)
# ---------------------------------------------------------------------------

def _build_tail_risk_figure(regime_tails: list[dict], metric: str = "var_95") -> go.Figure:
    """Bar chart of VaR or ES per regime.

    Parameters
    ----------
    regime_tails:
        List of per-regime tail dicts from the analysis report.
    metric:
        ``"var_95"`` or ``"es_95"``.

    Returns
    -------
    go.Figure
    """
    x_labels, y_values, colors = [], [], []
    for row in regime_tails:
        val = row.get(metric)
        if val is None:
            continue
        k = row.get("regime", 0)
        x_labels.append(f"Regime {k}")
        y_values.append(val)
        colors.append(_PALETTE[k % len(_PALETTE)])

    title = "Regime VaR 95%" if metric == "var_95" else "Regime ES 95%"
    fig = go.Figure(go.Bar(x=x_labels, y=y_values, marker_color=colors))
    fig.update_layout(
        title=title,
        yaxis_title="Loss (fraction)",
        showlegend=False,
    )
    return fig


def _panel_tail_risk(report: dict) -> None:
    """Render the Tail Risk (EVT) panel."""
    tail_risk = report.get("tail_risk")
    if tail_risk is None:
        st.info("No tail risk data found. Run the analysis pipeline to generate it.")
        return

    st.subheader("Tail Risk (EVT)")
    regime_tails: list[dict] = tail_risk.get("regime_tails", [])

    if regime_tails:
        # Table with formatted floats
        rows = []
        for row in regime_tails:
            rows.append(
                {
                    "regime": row.get("regime"),
                    "xi": f"{row['xi']:.4f}" if row.get("xi") is not None else "—",
                    "sigma": f"{row['sigma']:.4f}" if row.get("sigma") is not None else "—",
                    "var_95": f"{row['var_95']:.4f}" if row.get("var_95") is not None else "—",
                    "es_95": f"{row['es_95']:.4f}" if row.get("es_95") is not None else "—",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(_build_tail_risk_figure(regime_tails, "var_95"), use_container_width=True)
        with col2:
            st.plotly_chart(_build_tail_risk_figure(regime_tails, "es_95"), use_container_width=True)


# ---------------------------------------------------------------------------
# 4. Transition Analysis
# ---------------------------------------------------------------------------

def _build_transition_duration_figure(expected_durations: list, n_states: int) -> go.Figure:
    """Horizontal bar chart of expected dwell time per regime.

    Parameters
    ----------
    expected_durations:
        List of expected duration values (one per state).
    n_states:
        Number of states.

    Returns
    -------
    go.Figure
    """
    y_labels = [f"Regime {k}" for k in range(len(expected_durations))]
    colors = [_PALETTE[k % len(_PALETTE)] for k in range(len(expected_durations))]

    fig = go.Figure(
        go.Bar(
            x=expected_durations,
            y=y_labels,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(
        title="Expected Regime Dwell Time (bars)",
        xaxis_title="Expected Duration (bars)",
        showlegend=False,
    )
    return fig


def _build_transition_entropy_figure(mean_entropy: float, n_states: int) -> go.Figure:
    """Gauge indicator for mean transition entropy.

    Parameters
    ----------
    mean_entropy:
        Mean Shannon entropy (nats) of outgoing transitions.
    n_states:
        Number of HMM states — determines the max reference value log(K).

    Returns
    -------
    go.Figure
    """
    max_entropy = math.log(n_states) if n_states > 1 else 1.0
    gauge_max = max_entropy * 1.1

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=mean_entropy,
            title={"text": "Mean Transition Entropy (nats)"},
            gauge={
                "axis": {"range": [0, gauge_max]},
                "bar": {"color": _PALETTE[0]},
                "threshold": {
                    "line": {"color": "red", "width": 2},
                    "thickness": 0.75,
                    "value": max_entropy,
                },
            },
        )
    )
    return fig


def _panel_transition(report: dict) -> None:
    """Render the Regime Transition Analysis panel."""
    transition = report.get("transition")
    if transition is None:
        st.info("No transition data found. Run the analysis pipeline to generate it.")
        return

    st.subheader("Regime Transition Analysis")
    expected_durations = transition.get("expected_durations", [])
    mean_entropy: float = transition.get("mean_transition_entropy", 0.0)
    n_states = report.get("n_states", max(len(expected_durations), 2))

    col1, col2 = st.columns(2)
    with col1:
        if expected_durations:
            st.plotly_chart(
                _build_transition_duration_figure(expected_durations, n_states),
                use_container_width=True,
            )
    with col2:
        st.plotly_chart(
            _build_transition_entropy_figure(mean_entropy, n_states),
            use_container_width=True,
        )
        st.metric("Mean Transition Entropy", f"{mean_entropy:.4f}")


# ---------------------------------------------------------------------------
# 5. Correlation Structure
# ---------------------------------------------------------------------------

def _build_corr_heatmap_figure(
    corr_matrix: list[list], title: str, feature_names: list[str]
) -> go.Figure:
    """Annotated correlation heatmap.

    Parameters
    ----------
    corr_matrix:
        List-of-lists (square) correlation matrix.
    title:
        Figure title.
    feature_names:
        Axis tick labels.

    Returns
    -------
    go.Figure
    """
    n = len(corr_matrix)
    text_annotations = [
        [f"{corr_matrix[i][j]:+.2f}" for j in range(n)] for i in range(n)
    ]

    fig = go.Figure(
        go.Heatmap(
            z=corr_matrix,
            x=feature_names,
            y=feature_names,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=text_annotations,
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig.update_layout(title=title, height=300)
    return fig


def _panel_correlation(report: dict, feature_names: list[str]) -> None:
    """Render the Correlation Structure panel."""
    correlation = report.get("correlation")
    if correlation is None:
        st.info("No correlation data found. Run the analysis pipeline to generate it.")
        return

    st.subheader("Correlation Structure")
    stability = correlation.get("stability")
    if stability is not None:
        st.metric("Correlation Stability (Frobenius)", f"{stability:.4f}")

    global_corr = correlation.get("global_corr")
    if global_corr:
        st.plotly_chart(
            _build_corr_heatmap_figure(global_corr, "Global Correlation Matrix", feature_names),
            use_container_width=True,
        )

    regime_corrs: list[list[list]] = correlation.get("regime_corrs", [])
    if regime_corrs:
        with st.expander("Per-Regime Correlation Matrices", expanded=False):
            chunk_size = 3
            for start in range(0, len(regime_corrs), chunk_size):
                chunk = regime_corrs[start: start + chunk_size]
                cols = st.columns(len(chunk))
                for col_idx, (k, mat) in enumerate(
                    zip(range(start, start + len(chunk)), chunk)
                ):
                    with cols[col_idx]:
                        st.plotly_chart(
                            _build_corr_heatmap_figure(
                                mat, f"Regime {k} Correlation", feature_names
                            ),
                            use_container_width=True,
                        )

    tail_dep = correlation.get("tail_dependence")
    if tail_dep:
        st.dataframe(pd.DataFrame(tail_dep), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 6. Cointegration & Spread Analysis
# ---------------------------------------------------------------------------

def _build_cointegration_figure(
    regime_half_lives: dict, global_half_life: float | None
) -> go.Figure:
    """Bar chart of OU half-life by regime plus global.

    Parameters
    ----------
    regime_half_lives:
        Dict mapping str regime index -> half-life (float or None).
    global_half_life:
        Global (pooled) half-life, or None.

    Returns
    -------
    go.Figure
    """
    x_labels, y_values, colors = [], [], []
    sorted_keys = sorted(regime_half_lives.keys(), key=lambda k: int(k))
    for k in sorted_keys:
        val = regime_half_lives[k]
        if val is None:
            continue
        x_labels.append(f"Regime {k}")
        y_values.append(val)
        colors.append(_PALETTE[int(k) % len(_PALETTE)])

    if global_half_life is not None:
        x_labels.append("Global")
        y_values.append(global_half_life)
        colors.append("#7f7f7f")

    fig = go.Figure(go.Bar(x=x_labels, y=y_values, marker_color=colors))
    fig.update_layout(
        title="Ornstein-Uhlenbeck Half-Life by Regime (bars)",
        yaxis_title="Half-Life (bars)",
        showlegend=False,
    )
    return fig


def _panel_cointegration(report: dict, feature_names: list[str]) -> None:
    """Render the Cointegration & Spread Analysis panel."""
    cointegration = report.get("cointegration")
    if cointegration is None:
        st.info("No cointegration data found. Run the analysis pipeline to generate it.")
        return

    st.subheader("Cointegration & Spread Analysis")
    pair = cointegration.get("pair", "unknown")
    st.caption(f"Pair: {pair}")

    regime_half_lives = cointegration.get("regime_half_lives", {})
    global_half_life = cointegration.get("global_half_life")

    fig = _build_cointegration_figure(regime_half_lives, global_half_life)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 7. Portfolio Optimisation
# ---------------------------------------------------------------------------

def _build_portfolio_figure(
    blended_weights: list,
    regime_weights: list[list],
    feature_names: list[str],
) -> go.Figure:
    """Grouped bar chart of portfolio weights: global blended + per-regime.

    Parameters
    ----------
    blended_weights:
        Global blended weight vector.
    regime_weights:
        List of per-regime weight vectors.
    feature_names:
        Asset / feature labels for the x-axis.

    Returns
    -------
    go.Figure
    """
    n = len(feature_names)
    # Pad / truncate to n
    def _pad(w: list) -> list:
        return (list(w) + [0.0] * n)[:n]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Global (Blended)",
            x=feature_names,
            y=_pad(blended_weights),
            marker_color="#7f7f7f",
        )
    )
    for k, rw in enumerate(regime_weights):
        fig.add_trace(
            go.Bar(
                name=f"Regime {k}",
                x=feature_names,
                y=_pad(rw),
                marker_color=_PALETTE[k % len(_PALETTE)],
            )
        )

    fig.update_layout(
        title="Portfolio Weights: Blended vs Per-Regime",
        barmode="group",
        yaxis=dict(title="Weight", range=[0, 1]),
    )
    return fig


def _panel_portfolio(report: dict, feature_names: list[str]) -> None:
    """Render the Portfolio Optimisation panel."""
    portfolio = report.get("portfolio")
    if portfolio is None:
        st.info("No portfolio optimisation data found. Run the analysis pipeline to generate it.")
        return

    st.subheader("Portfolio Optimisation")
    blended_weights = portfolio.get("blended_weights", [])
    regime_weights = portfolio.get("regime_weights", [])

    if blended_weights or regime_weights:
        fig = _build_portfolio_figure(blended_weights, regime_weights, feature_names)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 8. Full Analysis Markdown Report
# ---------------------------------------------------------------------------

def _panel_analysis_markdown(asset_path: Path) -> None:
    """Render the Full Analysis Report panel from a markdown file.

    Parameters
    ----------
    asset_path:
        Path to the asset results directory (e.g. ``results/BTC-USD/``).
    """
    st.subheader("Full Analysis Report")
    md_path = asset_path / "analysis" / "analysis.md"
    if not md_path.exists():
        st.info("Run `rde analyse --config ...` to generate.")
        return

    content = md_path.read_text(encoding="utf-8")
    with st.expander("View Markdown Report", expanded=False):
        st.markdown(content)
