"""Interactive (Plotly) versions of all four required regime visualisations.

Each function mirrors the signature of its static counterpart in ``static_plots``
but returns a ``plotly.graph_objects.Figure`` instead of a matplotlib Figure.
Color palette is identical to the static version (matplotlib tab10 hex values)
so the two renderings are visually consistent.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# matplotlib tab10 hex values — deterministic per state index (≤ 10 states)
_TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _state_colors(n_states: int) -> list[str]:
    return [_TAB10[i % len(_TAB10)] for i in range(n_states)]


def _resolve_labels(n_states: int, labels: Optional[list[str]]) -> list[str]:
    if labels is not None and len(labels) == n_states:
        return labels
    return [f"State {i}" for i in range(n_states)]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a '#rrggbb' hex colour to an 'rgba(r,g,b,a)' CSS string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _run_length_spans(
    index: pd.Index, states: np.ndarray
) -> list[tuple[int, object, object]]:
    """Return (state, x0, x1) run-length spans; same logic as static_plots."""
    spans = []
    if len(states) == 0:
        return spans
    current = int(states[0])
    start = index[0]
    for i in range(1, len(states)):
        s = int(states[i])
        if s != current:
            spans.append((current, start, index[i]))
            current = s
            start = index[i]
    spans.append((current, start, index[-1]))
    return spans


def plot_price_with_regimes(
    df: pd.DataFrame,
    states: np.ndarray,
    labels: Optional[list[str]] = None,
    symbol: str = "Asset",
) -> go.Figure:
    """Interactive Close price chart with regime-coloured background shading.

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with DatetimeIndex and 'Close' column.
    states : np.ndarray
        Integer state sequence of shape (T,), aligned with df.index.
    labels : list[str] | None
        Human-readable label per state.
    symbol : str
        Asset name used in the title and hover text.

    Returns
    -------
    plotly.graph_objects.Figure

    """
    n_states = int(states.max()) + 1
    colors = _state_colors(n_states)
    state_labels = _resolve_labels(n_states, labels)

    fig = go.Figure()

    # Background shading — one vrect per run (efficient via run-length encoding)
    seen_states: set[int] = set()
    for state_idx, x0, x1 in _run_length_spans(df.index, states):
        show_legend = state_idx not in seen_states
        seen_states.add(state_idx)
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=_hex_to_rgba(colors[state_idx], 0.25),
            line_width=0,
            name=state_labels[state_idx],
            showlegend=show_legend,
            legendgroup=str(state_idx),
        )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            line=dict(color="black", width=1),
            name="Close",
            showlegend=False,
            hovertemplate="%{x}<br>Close: %{y:,.2f}<extra></extra>",
        )
    )

    date_start = df.index[0].strftime("%Y-%m-%d") if len(df) else ""
    date_end   = df.index[-1].strftime("%Y-%m-%d") if len(df) else ""
    fig.update_layout(
        title=f"{symbol} — Price with Regimes ({date_start} to {date_end})",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Regime",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )
    return fig


def plot_transition_heatmap(
    transmat: np.ndarray,
    labels: Optional[list[str]] = None,
    symbol: str = "Asset",
) -> go.Figure:
    """Interactive annotated heatmap of the transition matrix.

    Parameters
    ----------
    transmat : np.ndarray
        Row-stochastic matrix of shape (n_states, n_states).
    labels : list[str] | None
        State labels.
    symbol : str
        Asset name for the title.

    Returns
    -------
    plotly.graph_objects.Figure

    """
    n_states = transmat.shape[0]
    state_labels = _resolve_labels(n_states, labels)

    text = [[f"{transmat[i, j]:.3f}" for j in range(n_states)] for i in range(n_states)]

    fig = go.Figure(
        go.Heatmap(
            z=transmat,
            x=state_labels,
            y=state_labels,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Probability"),
            hovertemplate="From: %{y}<br>To: %{x}<br>P: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{symbol} — Transition Matrix",
        xaxis_title="To",
        yaxis_title="From",
        template="plotly_white",
        height=420,
        yaxis_autorange="reversed",
    )
    return fig


def plot_regime_timeline(
    df: pd.DataFrame,
    states: np.ndarray,
    labels: Optional[list[str]] = None,
    symbol: str = "Asset",
) -> go.Figure:
    """Interactive horizontal timeline strip coloured by regime.

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with DatetimeIndex.
    states : np.ndarray
        Integer state sequence, aligned with df.index.
    labels : list[str] | None
        State labels.
    symbol : str
        Asset name.

    Returns
    -------
    plotly.graph_objects.Figure

    """
    n_states = int(states.max()) + 1
    colors = _state_colors(n_states)
    state_labels = _resolve_labels(n_states, labels)

    fig = go.Figure()

    seen_states: set[int] = set()
    for state_idx, x0, x1 in _run_length_spans(df.index, states):
        show_legend = state_idx not in seen_states
        seen_states.add(state_idx)
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=_hex_to_rgba(colors[state_idx], 0.8),
            line_width=0,
            name=state_labels[state_idx],
            showlegend=show_legend,
            legendgroup=str(state_idx),
        )

    # Invisible scatter to anchor the x-axis range
    fig.add_trace(
        go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[0, 0],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=f"{symbol} — Regime Timeline",
        xaxis_title="Date",
        yaxis=dict(visible=False),
        legend_title="Regime",
        template="plotly_white",
        height=180,
    )
    return fig


def plot_per_regime_returns(
    df: pd.DataFrame,
    states: np.ndarray,
    return_col: str = "log_return",
    labels: Optional[list[str]] = None,
    symbol: str = "Asset",
) -> go.Figure:
    """Interactive overlapping return-distribution histograms by regime.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame containing return_col.
    states : np.ndarray
        Integer state sequence, aligned with df.index.
    return_col : str
        Column name of the return series.
    labels : list[str] | None
        State labels.
    symbol : str
        Asset name.

    Returns
    -------
    plotly.graph_objects.Figure

    """
    n_states = int(states.max()) + 1
    colors = _state_colors(n_states)
    state_labels = _resolve_labels(n_states, labels)

    fig = go.Figure()

    for s in range(n_states):
        mask = states == s
        returns = df[return_col].values[mask]
        if len(returns) == 0:
            continue
        fig.add_trace(
            go.Histogram(
                x=returns,
                name=state_labels[s],
                marker_color=colors[s],
                opacity=0.55,
                nbinsx=80,
                histnorm="probability density",
                hovertemplate=(
                    f"<b>{state_labels[s]}</b><br>"
                    "Return: %{x:.5f}<br>Density: %{y:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="overlay",
        title=f"{symbol} — Return Distribution by Regime",
        xaxis_title="Log Return",
        yaxis_title="Density",
        legend_title="Regime",
        template="plotly_white",
        height=450,
    )
    return fig
