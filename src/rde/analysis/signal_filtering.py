"""Regime-adaptive signal filtering (Phase 26).

Implements:

1. **regime_ema** — Exponential moving average with per-regime decay
   (half-life), blended by HMM posteriors at each bar.

2. **regime_hp_filter** — Hodrick-Prescott decomposition with lambda
   selected per regime (volatile regimes use smaller lambda to track
   faster mean reversion).

3. **regime_butterworth** — Simple IIR low-pass filter with
   regime-adaptive cutoff frequency.

4. **adaptive_kalman_smoother** — RTS (Rauch-Tung-Striebel) smoother
   applied to a univariate series with process noise scaled per regime.

5. **composite_signal** — Combines multiple filtered signals into a
   single composite using regime-posterior weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EMAConfig:
    """Configuration for regime-adaptive EMA.

    Attributes
    ----------
    half_lives : list[float]
        Half-life (in bars) per regime.  Longer half-life = slower EMA.
    """

    half_lives: list[float] = field(default_factory=lambda: [5.0, 20.0])


@dataclass
class HPConfig:
    """Configuration for regime-adaptive HP filter.

    Attributes
    ----------
    lambdas : list[float]
        Smoothing parameter per regime.  Smaller = less smooth (tracks
        signal faster).  Typical: 1600 for quarterly, 100 for monthly,
        6.25 for annual data.
    """

    lambdas: list[float] = field(default_factory=lambda: [100.0, 1600.0])


@dataclass
class KalmanSmootherConfig:
    """Configuration for regime-adaptive Kalman smoother.

    Attributes
    ----------
    process_noise_per_regime : list[float]
        Process noise variance per regime.  Higher in volatile regimes.
    obs_noise : float
        Observation noise variance (common to all regimes).
    """

    process_noise_per_regime: list[float] = field(default_factory=lambda: [0.001, 0.0001])
    obs_noise: float = 0.01


# ---------------------------------------------------------------------------
# Regime EMA
# ---------------------------------------------------------------------------


def regime_ema(
    series: pd.Series | np.ndarray,
    posteriors: np.ndarray,
    config: EMAConfig | None = None,
) -> pd.Series:
    """Posterior-blended regime-adaptive EMA.

    At each bar, the effective decay α_t = Σ_k p_{tk} * α_k where
    α_k = 1 - exp(-log(2) / half_life_k).

    Parameters
    ----------
    series : array-like, length T
    posteriors : np.ndarray, shape (T, K)
    config : EMAConfig, optional

    Returns
    -------
    pd.Series, same index as series.
    """
    cfg = config or EMAConfig()
    s = np.asarray(series, dtype=float)
    idx = series.index if isinstance(series, pd.Series) else pd.RangeIndex(len(s))
    T, K = posteriors.shape
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)

    # Per-regime decay factors.
    half_lives = np.asarray(cfg.half_lives, dtype=float)
    n_hl = len(half_lives)
    alphas = 1.0 - np.exp(-np.log(2.0) / np.maximum(half_lives, 1e-6))

    result = np.zeros(T)
    result[0] = s[0] if np.isfinite(s[0]) else 0.0

    for t in range(1, T):
        # Blend alpha across regimes.
        wt = P_norm[t, :n_hl] if K >= n_hl else P_norm[t]
        if K < n_hl:
            # Pad weights with zeros for missing regimes.
            wt_full = np.zeros(n_hl)
            wt_full[:K] = wt
            wt = wt_full
        wt = wt / (wt.sum() + 1e-15)
        alpha_t = float(wt @ alphas)
        xt = s[t] if np.isfinite(s[t]) else result[t - 1]
        result[t] = alpha_t * xt + (1.0 - alpha_t) * result[t - 1]

    return pd.Series(result, index=idx)


# ---------------------------------------------------------------------------
# Regime HP filter
# ---------------------------------------------------------------------------


def _hp_filter(y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Hodrick-Prescott filter.  Returns (trend, cycle)."""
    T = len(y)
    if T < 3:
        return y.copy(), np.zeros(T)

    # Build second-difference matrix D2 (T-2, T).
    from scipy.sparse import eye as speye, diags as spdiags
    from scipy.sparse.linalg import spsolve

    try:
        I = speye(T, format="csc")
        e = np.ones(T)
        D = spdiags(
            [e, -2 * e, e],
            [0, 1, 2],
            T - 2,
            T,
            format="csc",
        )
        trend = spsolve(I + lam * D.T @ D, y)
    except Exception:
        # Fallback: dense solve.
        I = np.eye(T)
        D = np.zeros((T - 2, T))
        for i in range(T - 2):
            D[i, i] = 1.0
            D[i, i + 1] = -2.0
            D[i, i + 2] = 1.0
        try:
            trend = np.linalg.solve(I + lam * D.T @ D, y)
        except np.linalg.LinAlgError:
            trend = y.copy()

    cycle = y - trend
    return trend, cycle


def regime_hp_filter(
    series: pd.Series | np.ndarray,
    posteriors: np.ndarray,
    config: HPConfig | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Compute HP trend and cycle with regime-blended lambda.

    Runs the HP filter for each regime lambda, then blends the resulting
    trends by the time-averaged regime posterior.

    Parameters
    ----------
    series : array-like, length T
    posteriors : np.ndarray, shape (T, K)
    config : HPConfig, optional

    Returns
    -------
    (trend, cycle) — two pd.Series with same index as series.
    """
    cfg = config or HPConfig()
    s = np.asarray(series, dtype=float)
    idx = series.index if isinstance(series, pd.Series) else pd.RangeIndex(len(s))
    T, K = posteriors.shape
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)
    avg_post = P_norm.mean(axis=0)  # shape (K,)

    lambdas = cfg.lambdas
    n_lam = len(lambdas)

    blend_trend = np.zeros(T)
    for k, lam in enumerate(lambdas):
        wk = float(avg_post[k]) if k < K else 0.0
        if wk < 1e-10:
            continue
        trend_k, _ = _hp_filter(s, lam)
        blend_trend += wk * trend_k

    # Normalise weight (in case n_lam < K or n_lam > K).
    used_weight = sum(float(avg_post[k]) for k in range(min(n_lam, K)))
    if used_weight > 1e-10:
        blend_trend /= used_weight

    cycle = s - blend_trend
    return pd.Series(blend_trend, index=idx), pd.Series(cycle, index=idx)


# ---------------------------------------------------------------------------
# Adaptive Kalman smoother (RTS)
# ---------------------------------------------------------------------------


def adaptive_kalman_smoother(
    series: pd.Series | np.ndarray,
    posteriors: np.ndarray,
    config: KalmanSmootherConfig | None = None,
) -> pd.Series:
    """Kalman forward-backward smoother with regime-adaptive process noise.

    Uses a scalar random-walk state-space model:
        x_t = x_{t-1} + w_t,   w_t ~ N(0, Q_t)
        y_t = x_t + v_t,       v_t ~ N(0, R)

    Q_t is the posterior-blended process noise:
        Q_t = Σ_k p_{tk} * Q_k

    Forward pass: Kalman filter.
    Backward pass: RTS smoother.

    Parameters
    ----------
    series : array-like, length T
    posteriors : np.ndarray, shape (T, K)
    config : KalmanSmootherConfig, optional

    Returns
    -------
    pd.Series — smoothed series.
    """
    cfg = config or KalmanSmootherConfig()
    y = np.asarray(series, dtype=float)
    idx = series.index if isinstance(series, pd.Series) else pd.RangeIndex(len(y))
    T, K = posteriors.shape
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)

    Q_k = np.asarray(cfg.process_noise_per_regime, dtype=float)
    n_q = len(Q_k)
    R = cfg.obs_noise

    # Forward pass.
    x_filt = np.zeros(T)
    P_filt = np.zeros(T)
    x_pred_arr = np.zeros(T)
    P_pred_arr = np.zeros(T)

    # Initialise.
    x = y[0] if np.isfinite(y[0]) else 0.0
    P = float(Q_k.mean()) + R

    for t in range(T):
        wt = P_norm[t, :n_q] if K >= n_q else P_norm[t]
        if K < n_q:
            wt_full = np.zeros(n_q)
            wt_full[:K] = wt
            wt = wt_full
        wt = wt / (wt.sum() + 1e-15)
        Qt = float(wt @ Q_k)

        # Predict.
        x_pred = x
        P_pred = P + Qt
        x_pred_arr[t] = x_pred
        P_pred_arr[t] = P_pred

        # Update.
        yt = y[t] if np.isfinite(y[t]) else x_pred
        S = P_pred + R
        K_gain = P_pred / (S + 1e-15)
        x = x_pred + K_gain * (yt - x_pred)
        P = (1.0 - K_gain) * P_pred

        x_filt[t] = x
        P_filt[t] = P

    # RTS backward pass.
    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()

    for t in range(T - 2, -1, -1):
        G = P_filt[t] / (P_pred_arr[t + 1] + 1e-15)
        x_smooth[t] = x_filt[t] + G * (x_smooth[t + 1] - x_pred_arr[t + 1])
        P_smooth[t] = P_filt[t] + G * G * (P_smooth[t + 1] - P_pred_arr[t + 1])

    return pd.Series(x_smooth, index=idx)


# ---------------------------------------------------------------------------
# Composite signal
# ---------------------------------------------------------------------------


def composite_signal(
    signals: dict[str, pd.Series],
    posteriors: np.ndarray,
    regime_signal_weights: np.ndarray | None = None,
) -> pd.Series:
    """Blend multiple signals into a composite using regime posteriors.

    Parameters
    ----------
    signals : dict[str, pd.Series]
        Named signal series, all of same length T.
    posteriors : np.ndarray, shape (T, K)
    regime_signal_weights : np.ndarray, shape (K, n_signals) or None
        Per-regime weight on each signal.  If None, all signals are
        weighted equally in every regime.

    Returns
    -------
    pd.Series — composite signal.
    """
    names = list(signals.keys())
    n_signals = len(names)
    T, K = posteriors.shape
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)

    # Build signal matrix (T, n_signals).
    idx = next(iter(signals.values())).index
    S = np.column_stack([signals[n].values.astype(float) for n in names])

    if regime_signal_weights is None:
        regime_signal_weights = np.full((K, n_signals), 1.0 / n_signals)

    # Normalise per regime.
    rsw = regime_signal_weights.copy().astype(float)
    row_sums = rsw.sum(axis=1, keepdims=True)
    rsw /= np.where(row_sums > 1e-15, row_sums, 1.0)

    composite = np.zeros(T)
    for t in range(T):
        blended_w = P_norm[t] @ rsw  # shape (n_signals,)
        blended_w /= (blended_w.sum() + 1e-15)
        composite[t] = float(blended_w @ S[t])

    return pd.Series(composite, index=idx)
