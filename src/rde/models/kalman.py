"""Regime-switching Kalman filter (Phase 19).

Implements:

1. **Scalar Kalman filter** — exact linear-Gaussian state-space tracking
   of a latent trend, with observation noise and process noise calibrated
   from recent data.

2. **Regime-switching Kalman filter (RSKF)** — at each step, maintains a
   separate Kalman estimate per regime, weighted by the HMM posterior
   responsibilities.  Each regime has its own (F, H, Q, R) matrices so
   volatile regimes adapt differently from quiet regimes.

3. **Regime-conditional trend signals** — maps filtered trend estimates
   to directional signals in [-1, +1] appropriate for trading.

Mathematical background
-----------------------
State equation:   z_t = F_k z_{t-1} + w_t,  w_t ~ N(0, Q_k)
Observation eq.:  y_t = H_k z_t + v_t,      v_t ~ N(0, R_k)

For a scalar random-walk trend (most common):
  F_k = [[1, dt], [0, 1]]  (position + velocity)
  H_k = [1, 0]             (observe position only)
  Q_k = σ_w,k² * I         (process noise)
  R_k = σ_v,k²             (observation noise = regime variance)

References
----------
Kalman, R. E. (1960). A new approach to linear filtering and prediction
    problems. *Journal of Basic Engineering*, 82(1), 82–95.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class KalmanConfig:
    """Configuration for the regime-switching Kalman filter.

    Attributes
    ----------
    process_noise : float
        Base process noise variance Q for the trend component.
    obs_noise_scale : float
        Multiplier on the regime's empirical variance to set observation
        noise R_k = obs_noise_scale * regime_var_k.
    initial_variance : float
        Initial state covariance P_0.
    dt : float
        Time step (default 1.0 = one bar).
    use_velocity : bool
        If True, use a 2-state [position, velocity] model.  If False,
        use a 1-state [position] random-walk model.
    """

    process_noise: float = 1e-4
    obs_noise_scale: float = 1.0
    initial_variance: float = 1.0
    dt: float = 1.0
    use_velocity: bool = True


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class KalmanFilterResult:
    """Output of :func:`kalman_filter`.

    Attributes
    ----------
    filtered_state : pd.DataFrame
        Shape (T, n_state_dims).  Posterior mean at each step.
        Columns: ``["trend"]`` or ``["trend", "velocity"]``.
    filtered_variance : np.ndarray
        Shape (T, n_state_dims, n_state_dims).  Posterior covariance.
    innovations : pd.Series
        Observation minus predicted observation at each step.
    innovation_variance : pd.Series
        Variance of the innovation (Kalman gain denominator).
    """

    filtered_state: pd.DataFrame
    filtered_variance: np.ndarray
    innovations: pd.Series
    innovation_variance: pd.Series


@dataclass
class RegimeSwitchingKalmanResult:
    """Output of :func:`regime_switching_kalman_filter`.

    Attributes
    ----------
    blended_state : pd.DataFrame
        Posterior state blended across regimes by HMM responsibilities.
    per_regime_trend : pd.DataFrame
        Shape (T, K).  Filtered trend for each regime separately.
    trend_signal : pd.Series
        Normalised trend signal in [-1, +1].  Positive = long, negative = short.
    """

    blended_state: pd.DataFrame
    per_regime_trend: pd.DataFrame
    trend_signal: pd.Series


# ---------------------------------------------------------------------------
# Scalar / univariate Kalman filter
# ---------------------------------------------------------------------------


def kalman_filter(
    observations: pd.Series | np.ndarray,
    obs_noise_var: float | None = None,
    config: KalmanConfig | None = None,
) -> KalmanFilterResult:
    """Run a linear Kalman filter on a univariate observation series.

    Parameters
    ----------
    observations : pd.Series or np.ndarray
        Univariate time series.  NaNs are skipped (innovation = 0,
        no update step performed for that bar).
    obs_noise_var : float, optional
        Observation noise variance R.  If None, estimated from the
        sample variance of the observations.
    config : KalmanConfig, optional

    Returns
    -------
    KalmanFilterResult
    """
    cfg = config or KalmanConfig()

    if isinstance(observations, pd.Series):
        idx = observations.index
        y = observations.values.astype(float)
    else:
        idx = pd.RangeIndex(len(observations))
        y = np.asarray(observations, dtype=float)

    T = len(y)
    R = obs_noise_var if obs_noise_var is not None else float(np.nanvar(y))

    if cfg.use_velocity:
        n = 2
        F = np.array([[1.0, cfg.dt], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = cfg.process_noise * np.eye(n)
        x = np.zeros(n)
        P = cfg.initial_variance * np.eye(n)
        state_cols = ["trend", "velocity"]
    else:
        n = 1
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[cfg.process_noise]])
        x = np.zeros(n)
        P = np.array([[cfg.initial_variance]])
        state_cols = ["trend"]

    R_mat = np.array([[R]])

    filtered_states = np.zeros((T, n))
    filtered_vars = np.zeros((T, n, n))
    innovations = np.zeros(T)
    innov_vars = np.zeros(T)

    for t in range(T):
        # Predict.
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        yt = y[t]
        if np.isnan(yt):
            # No observation: pass through prediction.
            x = x_pred
            P = P_pred
            innovations[t] = 0.0
            innov_vars[t] = float((H @ P_pred @ H.T + R_mat)[0, 0])
        else:
            # Update.
            innov = float(yt - (H @ x_pred)[0])
            S = float((H @ P_pred @ H.T + R_mat)[0, 0])
            K = (P_pred @ H.T) / S  # shape (n, 1)
            x = x_pred + K.flatten() * innov
            P = (np.eye(n) - np.outer(K.flatten(), H.flatten())) @ P_pred
            innovations[t] = innov
            innov_vars[t] = S

        filtered_states[t] = x
        filtered_vars[t] = P

    return KalmanFilterResult(
        filtered_state=pd.DataFrame(filtered_states, index=idx, columns=state_cols),
        filtered_variance=filtered_vars,
        innovations=pd.Series(innovations, index=idx, name="innovation"),
        innovation_variance=pd.Series(innov_vars, index=idx, name="innovation_var"),
    )


# ---------------------------------------------------------------------------
# Regime-switching Kalman filter
# ---------------------------------------------------------------------------


def regime_switching_kalman_filter(
    observations: pd.Series | np.ndarray,
    posteriors: np.ndarray,
    regime_vars: np.ndarray | None = None,
    config: KalmanConfig | None = None,
) -> RegimeSwitchingKalmanResult:
    """Run a separate Kalman filter per regime, blend by HMM responsibilities.

    Each regime k has its own observation noise variance
    R_k = obs_noise_scale * regime_vars[k].  The blended trend at each
    step is Σ_k P(s_t=k | x_{1:t}) * trend_k.

    Parameters
    ----------
    observations : pd.Series or np.ndarray
        Univariate observations, length T.
    posteriors : np.ndarray
        Shape (T, K).  HMM regime posterior probabilities.
    regime_vars : np.ndarray, optional
        Shape (K,).  Per-regime observation variance.  If None, computed
        from the observations weighted by each regime's responsibility.
    config : KalmanConfig, optional

    Returns
    -------
    RegimeSwitchingKalmanResult
    """
    cfg = config or KalmanConfig()

    if isinstance(observations, pd.Series):
        idx = observations.index
        y = observations.values.astype(float)
    else:
        idx = pd.RangeIndex(len(observations))
        y = np.asarray(observations, dtype=float)

    T, K = posteriors.shape
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)

    if regime_vars is None:
        regime_vars = np.zeros(K)
        for k in range(K):
            wk = P_norm[:, k]
            mu_k = (wk * y).sum() / (wk.sum() + 1e-15)
            regime_vars[k] = float((wk * (y - mu_k) ** 2).sum() / (wk.sum() + 1e-15))
            regime_vars[k] = max(regime_vars[k], 1e-10)

    per_regime_trends = np.zeros((T, K))

    for k in range(K):
        res_k = kalman_filter(
            observations,
            obs_noise_var=cfg.obs_noise_scale * regime_vars[k],
            config=cfg,
        )
        per_regime_trends[:, k] = res_k.filtered_state["trend"].values

    blended_trend = (P_norm * per_regime_trends).sum(axis=1)
    blended_df = pd.DataFrame({"trend": blended_trend}, index=idx)

    # Normalise to [-1, +1] using a rolling z-score with window=50.
    trend_s = pd.Series(blended_trend, index=idx)
    roll_mu = trend_s.rolling(50, min_periods=1).mean()
    roll_std = trend_s.rolling(50, min_periods=1).std().fillna(1e-10)
    trend_signal = (trend_s - roll_mu) / (roll_std + 1e-10)
    trend_signal = trend_signal.clip(-1.0, 1.0)

    return RegimeSwitchingKalmanResult(
        blended_state=blended_df,
        per_regime_trend=pd.DataFrame(
            per_regime_trends, index=idx, columns=[f"regime_{k}" for k in range(K)]
        ),
        trend_signal=trend_signal.rename("trend_signal"),
    )
