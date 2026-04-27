"""Static matplotlib plots for regime visualisation (price, heatmap, timeline, returns)."""

import logging
from typing import Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CMAP = matplotlib.colormaps["tab10"]


def _state_colors(n_states: int) -> list:
    return [_CMAP(i) for i in range(n_states)]


def _resolve_labels(n_states: int, labels: Optional[list[str]]) -> list[str]:
    if labels is not None and len(labels) == n_states:
        return labels
    return [f"State {i}" for i in range(n_states)]


def _run_length_spans(
    index: pd.Index, states: np.ndarray
) -> list[tuple[int, object, object]]:
    """Return list of (state, start, end) run-length spans for efficient axvspan calls."""
    spans = []
    if len(states) == 0:
        return spans
    current_state = int(states[0])
    start_pos = index[0]
    for i in range(1, len(states)):
        s = int(states[i])
        if s != current_state:
            spans.append((current_state, start_pos, index[i]))
            current_state = s
            start_pos = index[i]
    spans.append((current_state, start_pos, index[-1]))
    return spans


def plot_price_with_regimes(
    df: pd.DataFrame,
    states: np.ndarray,
    labels: Optional[list[str]] = None,
    symbol: str = "Asset",
) -> plt.Figure:
    """Plot Close price as a line with regime-colored background shading.

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with DatetimeIndex and 'Close' column.
    states : np.ndarray
        Integer state sequence of shape (T,), aligned with df.index.
    labels : list[str] | None
        Human-readable label per state. Falls back to "State N" if None.
    symbol : str
        Asset name used in the plot title.

    Returns
    -------
    matplotlib.figure.Figure

    """
    n_states = int(states.max()) + 1
    colors = _state_colors(n_states)
    state_labels = _resolve_labels(n_states, labels)

    fig, ax = plt.subplots(figsize=(16, 6))

    for state_idx, start, end in _run_length_spans(df.index, states):
        ax.axvspan(start, end, alpha=0.25, color=colors[state_idx], linewidth=0)

    ax.plot(df.index, df["Close"].values, color="black", linewidth=0.8, zorder=5)

    legend_patches = [
        mpatches.Patch(color=colors[s], alpha=0.5, label=state_labels[s])
        for s in range(n_states)
    ]
    ax.legend(handles=legend_patches, loc="upper left")

    date_start = df.index[0].strftime("%Y-%m-%d") if len(df) else ""
    date_end = df.index[-1].strftime("%Y-%m-%d") if len(df) else ""
    ax.set_title(f"{symbol} — Price with Regimes ({date_start} to {date_end})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.tight_layout()
    return fig


def plot_transition_heatmap(
    transmat: np.ndarray,
    labels: Optional[list[str]] = None,
    symbol: str = "Asset",
) -> plt.Figure:
    """Plot an annotated heatmap of the transition matrix.

    Parameters
    ----------
    transmat : np.ndarray
        Row-stochastic matrix of shape (n_states, n_states).
    labels : list[str] | None
        State labels. Falls back to "State N".
    symbol : str
        Asset name for the title.

    Returns
    -------
    matplotlib.figure.Figure

    """
    n_states = transmat.shape[0]
    state_labels = _resolve_labels(n_states, labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(transmat, vmin=0.0, vmax=1.0, cmap="Blues")
    plt.colorbar(im, ax=ax, label="Probability")

    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    ax.set_xticklabels(state_labels, rotation=30, ha="right")
    ax.set_yticklabels(state_labels)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title(f"{symbol} — Transition Matrix")

    for i in range(n_states):
        for j in range(n_states):
            ax.text(j, i, f"{transmat[i, j]:.2f}", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_regime_timeline(
    df: pd.DataFrame,
    states: np.ndarray,
    labels: Optional[list[str]] = None,
    symbol: str = "Asset",
) -> plt.Figure:
    """Plot a compact horizontal timeline strip colored by regime.

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
    matplotlib.figure.Figure

    """
    n_states = int(states.max()) + 1
    colors = _state_colors(n_states)
    state_labels = _resolve_labels(n_states, labels)

    fig, ax = plt.subplots(figsize=(16, 1.5))

    for state_idx, start, end in _run_length_spans(df.index, states):
        ax.axvspan(start, end, alpha=0.8, color=colors[state_idx], linewidth=0)

    legend_patches = [
        mpatches.Patch(color=colors[s], alpha=0.7, label=state_labels[s])
        for s in range(n_states)
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Date")
    ax.set_title(f"{symbol} — Regime Timeline")
    plt.tight_layout()
    return fig


def plot_per_regime_returns(
    df: pd.DataFrame,
    states: np.ndarray,
    return_col: str = "log_return",
    labels: Optional[list[str]] = None,
    symbol: str = "Asset",
) -> plt.Figure:
    """Plot overlapping return histograms conditional on regime.

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
    matplotlib.figure.Figure

    """
    n_states = int(states.max()) + 1
    colors = _state_colors(n_states)
    state_labels = _resolve_labels(n_states, labels)

    fig, ax = plt.subplots(figsize=(10, 5))

    for s in range(n_states):
        mask = states == s
        returns = df[return_col].values[mask]
        if len(returns) > 0:
            ax.hist(
                returns,
                bins=80,
                alpha=0.5,
                color=colors[s],
                label=state_labels[s],
                density=True,
            )

    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.set_title(f"{symbol} — Return Distribution by Regime")
    ax.legend()
    plt.tight_layout()
    return fig
