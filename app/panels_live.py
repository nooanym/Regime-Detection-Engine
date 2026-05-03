"""
Regime Detection Engine — Live Feed Panel (Phase 34)

Two-tier architecture:
  _build_*_figure(data) — pure functions returning go.Figure (no Streamlit calls)
  _panel_live_feed(asset, results_dir) — UI layer

Uses OnlineDecoder to compute fresh causal posteriors on the latest bars from
regimes.parquet without re-fitting the HMM.  Requires model.pkl to exist
(generate with: uv run rde run --config configs/<asset>.yaml --save-model results/<asset>/model.pkl).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Allow running from app/ dir
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

if TYPE_CHECKING:
    from rde.models.hmm import FittedModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette (must match streamlit_app.py)
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
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_live_model(asset: str, results_dir: Path) -> "FittedModel | None":
    """Load model.pkl for *asset* if it exists, else return None.

    Parameters
    ----------
    asset : str
        Asset symbol (unused but kept for future per-asset routing).
    results_dir : Path
        The asset's results directory, e.g. ``results/BTC-USD``.

    Returns
    -------
    FittedModel or None
    """
    pkl_path = results_dir / "model.pkl"
    if not pkl_path.exists():
        return None
    try:
        from rde.models.persistence import load_model  # noqa: PLC0415
        return load_model(pkl_path)
    except Exception as exc:
        logger.warning("Could not load model from %s: %s", pkl_path, exc)
        return None


# ---------------------------------------------------------------------------
# Pure figure builders
# ---------------------------------------------------------------------------

def _build_live_posterior_figure(
    timestamps: pd.DatetimeIndex,
    online_posteriors: np.ndarray,
    regime_labels: dict[int, str],
    asset: str,
    n_bars: int = 96,
) -> go.Figure:
    """Stacked area chart of online filtered posteriors over the last *n_bars*.

    Parameters
    ----------
    timestamps : pd.DatetimeIndex
        Bar timestamps matching rows of *online_posteriors*.
    online_posteriors : np.ndarray
        Shape (T, K). Each row sums to 1.
    regime_labels : dict[int, str]
        Map from state index → human-readable label.
    asset : str
        Used in the plot title.
    n_bars : int
        Number of trailing bars to display.

    Returns
    -------
    go.Figure
    """
    T, K = online_posteriors.shape
    display_n = min(n_bars, T)
    ts_disp = timestamps[-display_n:]
    post_disp = online_posteriors[-display_n:]

    fig = go.Figure()
    for k in range(K):
        label = regime_labels.get(k, f"Regime {k}")
        fig.add_trace(
            go.Scatter(
                x=ts_disp,
                y=post_disp[:, k],
                mode="lines",
                name=f"#{k} {label}",
                stackgroup="one",
                line=dict(width=0.5, color=_PALETTE[k % len(_PALETTE)]),
                fillcolor=_PALETTE[k % len(_PALETTE)],
                opacity=0.75,
            )
        )

    fig.update_layout(
        title=f"{asset} — Online Filtered Posteriors (last {display_n} bars)",
        xaxis_title="Time",
        yaxis=dict(title="Probability", range=[0, 1]),
        legend_title="Regime",
        hovermode="x unified",
        margin=dict(t=50, b=30, l=50, r=20),
        height=260,
    )
    return fig


def _build_live_price_figure(
    regimes_df: pd.DataFrame,
    online_posteriors: np.ndarray,
    asset: str,
    n_bars: int = 96,
) -> go.Figure:
    """Price line chart for the last *n_bars*, coloured by argmax online posterior.

    Parameters
    ----------
    regimes_df : pd.DataFrame
        Must have a ``Close`` column and a DatetimeIndex.
    online_posteriors : np.ndarray
        Shape (T, K). Aligned with regimes_df rows.
    asset : str
        Used in the plot title.
    n_bars : int
        Number of trailing bars to display.

    Returns
    -------
    go.Figure
    """
    T = len(regimes_df)
    K = online_posteriors.shape[1]
    display_n = min(n_bars, T)

    df_disp = regimes_df.iloc[-display_n:].copy()
    post_disp = online_posteriors[-display_n:]
    live_regime = np.argmax(post_disp, axis=1)

    fig = go.Figure()
    for k in range(K):
        mask = live_regime == k
        if not mask.any():
            continue
        subset = df_disp[mask]
        fig.add_trace(
            go.Scattergl(
                x=subset.index,
                y=subset["Close"],
                mode="markers",
                marker=dict(
                    size=3,
                    color=_PALETTE[k % len(_PALETTE)],
                    opacity=0.9,
                ),
                name=f"Regime #{k}",
                showlegend=True,
            )
        )

    fig.update_layout(
        title=f"{asset} — Price Coloured by Online Regime (last {display_n} bars)",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        legend_title="Online Regime",
        hovermode="x unified",
        margin=dict(t=50, b=30, l=60, r=20),
        height=300,
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit panel
# ---------------------------------------------------------------------------

def _panel_live_feed(asset: str, results_dir: Path) -> None:
    """Streamlit panel: Live Feed using OnlineDecoder.

    1. Loads regimes.parquet and model.pkl from *results_dir*.
    2. Re-runs batch_filter on the last 500 bars (warm-up window) and
       displays the last 96.
    3. Shows metrics: current online regime, confidence, last bar timestamp.
    4. Shows: posterior area chart + price chart coloured by live regime.

    If model.pkl is absent, shows an info box with instructions to generate it.

    Parameters
    ----------
    asset : str
        Asset symbol for display.
    results_dir : Path
        The asset's results directory.
    """
    st.subheader("Live Feed (Online Decoder)")

    # ── Load regimes ──────────────────────────────────────────────────────────
    reg_path = results_dir / "regimes.parquet"
    if not reg_path.exists():
        st.error(f"`{reg_path}` not found. Run the pipeline first.")
        return

    regimes_df = pd.read_parquet(reg_path)

    # ── Load model ────────────────────────────────────────────────────────────
    fitted = _load_live_model(asset, results_dir)
    if fitted is None:
        st.info(
            f"No `model.pkl` found for {asset}. To enable the Live Feed panel, "
            "generate the model file with:\n\n"
            f"```\nuv run rde run --config configs/{asset.lower().replace('-','')}.yaml "
            f"--save-model results/{asset}/model.pkl\n```"
        )
        return

    # ── Reconstruct feature matrix ────────────────────────────────────────────
    feature_names = fitted.feature_names
    missing = [f for f in feature_names if f not in regimes_df.columns]
    if missing:
        st.warning(
            f"Feature columns missing from regimes.parquet: {missing}. "
            "Showing pre-computed regime labels only."
        )
        return

    X_raw = regimes_df[feature_names].values.astype(float)
    # Remove rows with NaN (shouldn't exist post-pipeline, but guard anyway)
    valid_mask = ~np.isnan(X_raw).any(axis=1)
    X_clean = X_raw[valid_mask]
    ts_clean = regimes_df.index[valid_mask]
    df_clean = regimes_df[valid_mask]

    if len(X_clean) == 0:
        st.error("No valid feature rows found in regimes.parquet.")
        return

    # Use last 500 bars for online filtering (warm-up + display)
    warmup = min(500, len(X_clean))
    X_window = X_clean[-warmup:]
    ts_window = ts_clean[-warmup:]
    df_window = df_clean.iloc[-warmup:]

    # ── Run OnlineDecoder ─────────────────────────────────────────────────────
    try:
        from rde.inference.online import OnlineDecoder  # noqa: PLC0415
        decoder = OnlineDecoder(fitted)
        online_posteriors = decoder.batch_filter(X_window)
    except Exception as exc:
        st.error(f"OnlineDecoder failed: {exc}")
        return

    # ── Build regime labels map ───────────────────────────────────────────────
    K = fitted.n_states
    regime_labels: dict[int, str] = {}
    for k in range(K):
        mask = df_window["regime"] == k
        if mask.any():
            regime_labels[k] = str(df_window.loc[mask, "regime_label"].iloc[0])
        else:
            regime_labels[k] = f"Regime {k}"

    # ── Latest bar metrics ────────────────────────────────────────────────────
    latest_posterior = online_posteriors[-1]
    live_regime_idx = int(np.argmax(latest_posterior))
    confidence = float(latest_posterior[live_regime_idx])
    last_ts = str(ts_window[-1])[:16]
    live_label = regime_labels.get(live_regime_idx, f"Regime {live_regime_idx}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Online Regime", f"#{live_regime_idx}", delta=live_label, delta_color="off")
    col2.metric("Confidence", f"{confidence:.1%}")
    col3.metric("States", K)
    col4.metric("As Of", last_ts)

    # ── Charts ────────────────────────────────────────────────────────────────
    posterior_fig = _build_live_posterior_figure(
        ts_window, online_posteriors, regime_labels, asset
    )
    st.plotly_chart(posterior_fig, use_container_width=True)

    price_fig = _build_live_price_figure(
        df_window, online_posteriors, asset
    )
    st.plotly_chart(price_fig, use_container_width=True)
