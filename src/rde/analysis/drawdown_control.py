"""Regime-adaptive drawdown control and position limits (Phase 20).

Implements three complementary risk-gating mechanisms that use the live
HMM regime posterior to scale down or zero out positions before the
portfolio reaches a pre-specified drawdown limit:

1. **DrawdownController** — tracks running drawdown from peak equity;
   scales positions down linearly as drawdown approaches the limit and
   cuts to zero at the limit.

2. **RegimePositionLimits** — per-regime maximum absolute position
   derived from the regime's risk metrics (VaR, volatility, max drawdown).
   Higher-vol regimes get smaller position limits.

3. **RegimeDrawdownBudget** — allocates a fraction of total drawdown budget
   to each regime based on its Sharpe ratio; regimes with negative Sharpe
   receive zero budget.

All functions operate on DataFrames / Series with a DatetimeIndex and are
designed to be used in pipeline with :func:`rde.analysis.portfolio` weight
functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DrawdownControlConfig:
    """Configuration for running drawdown control.

    Attributes
    ----------
    max_drawdown : float
        Hard stop-loss: positions cut to zero when running drawdown
        reaches this level.  E.g. 0.10 = 10%.
    soft_limit : float
        Linear ramp starts here (must be < max_drawdown).
        At soft_limit, scale = 1.0.  At max_drawdown, scale = 0.0.
    recovery_threshold : float
        Drawdown must recover to this level before positions are restored.
        Prevents flip-flopping around the stop-loss boundary.
        E.g. 0.05 means re-engage when drawdown < 5%.
    """

    max_drawdown: float = 0.10
    soft_limit: float = 0.05
    recovery_threshold: float = 0.03


@dataclass
class RegimePositionLimitConfig:
    """Configuration for regime-conditional position limits.

    Attributes
    ----------
    base_limit : float
        Maximum absolute position in the highest-quality regime.
    vol_scale_power : float
        Exponent for vol-based scaling.  limit_k = base_limit / (σ_k / σ_min)^power.
    min_limit : float
        Floor on any regime limit (avoid degenerate 0-limit regimes).
    """

    base_limit: float = 1.0
    vol_scale_power: float = 1.0
    min_limit: float = 0.0


# ---------------------------------------------------------------------------
# Running drawdown tracking
# ---------------------------------------------------------------------------


def running_drawdown(equity: pd.Series | np.ndarray) -> pd.Series:
    """Compute running drawdown from peak equity (non-positive series).

    Parameters
    ----------
    equity : pd.Series or np.ndarray
        Cumulative equity curve (absolute values, e.g. starting at 10000).

    Returns
    -------
    pd.Series
        Values in (-∞, 0].  0 = at all-time high.
    """
    if isinstance(equity, pd.Series):
        idx = equity.index
        eq = equity.values.astype(float)
    else:
        idx = pd.RangeIndex(len(equity))
        eq = np.asarray(equity, dtype=float)

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-15)
    return pd.Series(dd, index=idx, name="drawdown")


def drawdown_scale_factor(
    drawdown_series: pd.Series,
    config: DrawdownControlConfig | None = None,
) -> pd.Series:
    """Compute a position scale factor in [0, 1] from running drawdown.

    Scale = 1 when drawdown ≤ soft_limit.
    Scale = 0 when drawdown ≤ -max_drawdown.
    Linear ramp between soft_limit and max_drawdown.
    Once stopped (scale = 0), stays 0 until drawdown recovers above
    recovery_threshold.

    Parameters
    ----------
    drawdown_series : pd.Series
        Running drawdown (non-positive values).
    config : DrawdownControlConfig, optional

    Returns
    -------
    pd.Series
        Values in [0, 1].
    """
    cfg = config or DrawdownControlConfig()
    dd = np.asarray(drawdown_series.values, dtype=float)
    T = len(dd)
    scale = np.ones(T)
    stopped = False

    for t in range(T):
        d = -dd[t]  # make positive: 0 = no drawdown, 0.1 = 10% drawdown

        if stopped:
            if d <= cfg.recovery_threshold:
                stopped = False
            else:
                scale[t] = 0.0
                continue

        if d >= cfg.max_drawdown:
            scale[t] = 0.0
            stopped = True
        elif d >= cfg.soft_limit:
            ramp = (d - cfg.soft_limit) / (cfg.max_drawdown - cfg.soft_limit + 1e-15)
            scale[t] = max(0.0, 1.0 - ramp)
        else:
            scale[t] = 1.0

    return pd.Series(scale, index=drawdown_series.index, name="dd_scale")


def apply_drawdown_control(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    config: DrawdownControlConfig | None = None,
) -> pd.DataFrame:
    """Apply drawdown control to a weight DataFrame.

    Simulates portfolio equity using ``weights.shift(1) * returns``, tracks
    running drawdown, and scales weights by the drawdown scale factor.

    Parameters
    ----------
    weights : pd.DataFrame
        Per-bar position weights, shape (T, N).
    returns : pd.DataFrame
        Per-bar log returns, shape (T, N), same index as weights.
    config : DrawdownControlConfig, optional

    Returns
    -------
    pd.DataFrame
        Scaled weights, same shape as input.
    """
    cfg = config or DrawdownControlConfig()
    # Simulate portfolio returns with the unscaled weights.
    port_ret = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    equity = (1.0 + port_ret).cumprod()

    dd = running_drawdown(equity)
    scale = drawdown_scale_factor(dd, cfg)

    return weights.multiply(scale, axis=0)


# ---------------------------------------------------------------------------
# Regime-conditional position limits
# ---------------------------------------------------------------------------


def regime_position_limits(
    regime_vols: np.ndarray,
    config: RegimePositionLimitConfig | None = None,
) -> np.ndarray:
    """Compute per-regime maximum absolute position limits.

    Limits are inversely proportional to relative regime volatility:

        limit_k = base_limit / (σ_k / σ_min)^power

    where σ_min is the minimum regime volatility (lowest-vol regime gets
    the base_limit, all others get less).

    Parameters
    ----------
    regime_vols : np.ndarray
        Shape (K,).  Per-regime annualised volatility.
    config : RegimePositionLimitConfig, optional

    Returns
    -------
    np.ndarray
        Shape (K,).  Per-regime maximum absolute position.
    """
    cfg = config or RegimePositionLimitConfig()
    vols = np.asarray(regime_vols, dtype=float)
    sigma_min = vols.min()
    if sigma_min < 1e-10:
        sigma_min = 1e-10

    limits = cfg.base_limit / (vols / sigma_min) ** cfg.vol_scale_power
    limits = np.maximum(limits, cfg.min_limit)
    return limits


def apply_regime_position_limits(
    weights: pd.DataFrame,
    posteriors: np.ndarray,
    regime_vols: np.ndarray,
    config: RegimePositionLimitConfig | None = None,
) -> pd.DataFrame:
    """Scale weights down so each column respects regime-conditional limits.

    The effective limit at bar t is the responsibility-weighted average of
    per-regime limits: L_t = Σ_k P(s_t=k) * limit_k.

    Parameters
    ----------
    weights : pd.DataFrame
        Shape (T, N).
    posteriors : np.ndarray
        Shape (T, K).
    regime_vols : np.ndarray
        Shape (K,).  Per-regime annualised volatility.
    config : RegimePositionLimitConfig, optional

    Returns
    -------
    pd.DataFrame
        Weights clipped to blended regime limits, shape (T, N).
    """
    limits_k = regime_position_limits(regime_vols, config)  # (K,)
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)
    blended_limit = P_norm @ limits_k  # (T,) — scalar per bar

    w = weights.values.copy()
    for t in range(len(w)):
        lim = float(blended_limit[t])
        w[t] = np.clip(w[t], -lim, lim)

    return pd.DataFrame(w, index=weights.index, columns=weights.columns)


# ---------------------------------------------------------------------------
# Regime drawdown budget allocation
# ---------------------------------------------------------------------------


def regime_drawdown_budget(
    regime_sharpes: np.ndarray,
    total_budget: float = 0.10,
) -> np.ndarray:
    """Allocate total drawdown budget across regimes by Sharpe ratio.

    Regimes with non-positive Sharpe receive zero budget.  The rest
    receive budget proportional to their (clipped positive) Sharpe:

        budget_k = total_budget * max(sharpe_k, 0) / Σ_j max(sharpe_j, 0)

    Parameters
    ----------
    regime_sharpes : np.ndarray
        Shape (K,).  Per-regime Sharpe ratios.
    total_budget : float
        Total drawdown budget to distribute.  Default 0.10 = 10%.

    Returns
    -------
    np.ndarray
        Shape (K,).  Per-regime drawdown budget.
    """
    s = np.asarray(regime_sharpes, dtype=float)
    pos = np.maximum(s, 0.0)
    total_pos = pos.sum()
    if total_pos < 1e-12:
        # All Sharpes non-positive: equal allocation
        return np.full(len(s), total_budget / len(s))
    return total_budget * pos / total_pos
