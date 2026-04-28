"""Regime-aware execution strategy (Phase 22).

Models the cost of executing a trade as a function of the current market
regime.  Volatile regimes have higher spread and market impact; quiet
regimes allow larger order sizes at lower cost.

Implements:

1. **RegimeImpactModel** — per-regime linear market impact:
   cost_k = spread_k/2 + impact_k * sqrt(order_size)
   Parameters are estimated from regime-conditional bid-ask spread proxies
   (high-low range as a spread estimate) and volume.

2. **OptimalOrderSlicer** — given a target position change and a regime
   posterior, computes the schedule (number of child orders, size per bar)
   that minimises expected total impact cost subject to a completion horizon.

3. **TWAPSchedule** / **VWAPSchedule** — time-weighted and volume-weighted
   reference benchmarks.

4. **RegimeSlippageEstimator** — estimates expected slippage for a given
   order using the blended regime impact model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ImpactModelConfig:
    """Parameters for per-regime linear square-root impact model.

    Attributes
    ----------
    spread_proxy : str
        Column name used as the half-spread proxy.  Default ``"log_hl_range"``
        (log of high/low range, a common volatility proxy for spread).
    volume_col : str | None
        Column for volume.  Used for VWAP and volume-scaled impact.
    base_impact : float
        Baseline market impact coefficient (fraction of price per unit
        of order size / average daily volume).
    """

    spread_proxy: str = "log_hl_range"
    volume_col: str | None = None
    base_impact: float = 0.1


@dataclass
class ExecutionConfig:
    """Configuration for the optimal order slicer.

    Attributes
    ----------
    horizon : int
        Maximum number of bars over which to complete the order.
    urgency : float
        Fraction in (0, 1].  Higher = finish faster (more aggressive
        front-loading).  1.0 = execute everything in the first bar.
    min_child_size : float
        Minimum order size per child slice (fraction of target).
    """

    horizon: int = 10
    urgency: float = 0.5
    min_child_size: float = 0.01


# ---------------------------------------------------------------------------
# Per-regime impact parameters
# ---------------------------------------------------------------------------


@dataclass
class RegimeImpactParams:
    """Estimated impact model parameters per regime.

    Attributes
    ----------
    n_regimes : int
    half_spread : np.ndarray, shape (K,)
        Estimated half bid-ask spread per regime (as a fraction of price).
    impact_coeff : np.ndarray, shape (K,)
        Square-root impact coefficient per regime.
    avg_volume : np.ndarray, shape (K,)
        Average volume per regime (used to normalise order size).
    """

    n_regimes: int
    half_spread: np.ndarray
    impact_coeff: np.ndarray
    avg_volume: np.ndarray

    def blended(self, posteriors: np.ndarray) -> tuple[float, float, float]:
        """Return responsibility-weighted (half_spread, impact, avg_vol)."""
        w = posteriors / (posteriors.sum() + 1e-15)
        return (
            float(w @ self.half_spread),
            float(w @ self.impact_coeff),
            float(w @ self.avg_volume),
        )


def estimate_regime_impact(
    features: pd.DataFrame,
    posteriors: np.ndarray,
    config: ImpactModelConfig | None = None,
) -> RegimeImpactParams:
    """Estimate per-regime impact model parameters from historical data.

    Uses responsibility-weighted statistics to estimate:
    - half_spread from the ``spread_proxy`` column (mean per regime).
    - impact_coeff = base_impact * (regime_vol / global_vol).
    - avg_volume from the volume column if present.

    Parameters
    ----------
    features : pd.DataFrame
        Feature frame including the spread proxy column.
    posteriors : np.ndarray, shape (T, K)
    config : ImpactModelConfig, optional

    Returns
    -------
    RegimeImpactParams
    """
    cfg = config or ImpactModelConfig()
    T, K = posteriors.shape
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)

    half_spread = np.zeros(K)
    impact_coeff = np.zeros(K)
    avg_volume = np.ones(K)

    global_vol = 1.0
    if cfg.spread_proxy in features.columns:
        spread_vals = features[cfg.spread_proxy].values.astype(float)
        global_vol = float(np.nanstd(spread_vals)) + 1e-10
        for k in range(K):
            wk = P_norm[:, k]
            eff_n = wk.sum()
            if eff_n < 1.0:
                half_spread[k] = global_vol * 0.5
            else:
                half_spread[k] = float((wk * spread_vals).sum() / eff_n) * 0.5
                regime_vol = float(
                    np.sqrt((wk * (spread_vals - half_spread[k] * 2) ** 2).sum() / eff_n)
                )
                impact_coeff[k] = cfg.base_impact * (regime_vol / global_vol + 1e-10)

    if cfg.volume_col and cfg.volume_col in features.columns:
        vol_vals = features[cfg.volume_col].values.astype(float)
        for k in range(K):
            wk = P_norm[:, k]
            eff_n = wk.sum()
            avg_volume[k] = float((wk * vol_vals).sum() / max(eff_n, 1.0))

    # Ensure non-negative
    half_spread = np.abs(half_spread)
    impact_coeff = np.abs(impact_coeff)
    avg_volume = np.maximum(avg_volume, 1.0)

    return RegimeImpactParams(
        n_regimes=K,
        half_spread=half_spread,
        impact_coeff=impact_coeff,
        avg_volume=avg_volume,
    )


# ---------------------------------------------------------------------------
# Expected slippage estimator
# ---------------------------------------------------------------------------


def expected_slippage(
    order_size: float,
    impact_params: RegimeImpactParams,
    current_posteriors: np.ndarray,
) -> float:
    """Estimate expected one-way slippage for an order.

    Slippage = half_spread + impact_coeff * sqrt(|order_size| / avg_volume)

    Parameters
    ----------
    order_size : float
        Signed order size (positive = buy, negative = sell).
        Units: same as avg_volume (typically number of units).
    impact_params : RegimeImpactParams
    current_posteriors : np.ndarray, shape (K,)

    Returns
    -------
    float
        Estimated fractional slippage (0.01 = 1% of price).
    """
    hs, ic, av = impact_params.blended(current_posteriors)
    normalized = abs(order_size) / (av + 1e-10)
    return hs + ic * np.sqrt(normalized)


# ---------------------------------------------------------------------------
# Optimal order schedule
# ---------------------------------------------------------------------------


@dataclass
class OrderSchedule:
    """Output of :func:`optimal_order_schedule`.

    Attributes
    ----------
    sizes : np.ndarray, shape (horizon,)
        Child order size at each bar (signed).
    expected_cost : float
        Total expected impact cost for the schedule.
    completion_bar : int
        First bar at which the cumulative schedule reaches the target.
    """

    sizes: np.ndarray
    expected_cost: float
    completion_bar: int


def optimal_order_schedule(
    target_delta: float,
    impact_params: RegimeImpactParams,
    current_posteriors: np.ndarray,
    config: ExecutionConfig | None = None,
) -> OrderSchedule:
    """Compute a regime-aware optimal execution schedule.

    Uses a simple urgency-decaying schedule:

        size_t = target * urgency * (1 - urgency)^t   (geometric decay)

    normalised to sum to target_delta.  Higher urgency front-loads the
    order; lower urgency spreads it more evenly.

    Expected cost = Σ_t |size_t| * slippage(size_t, params, posteriors)

    Parameters
    ----------
    target_delta : float
        Total signed position change to execute.
    impact_params : RegimeImpactParams
    current_posteriors : np.ndarray, shape (K,)
    config : ExecutionConfig, optional

    Returns
    -------
    OrderSchedule
    """
    cfg = config or ExecutionConfig()
    h = cfg.horizon
    u = np.clip(cfg.urgency, 1e-6, 1.0 - 1e-6)

    # Geometric decay weights.
    weights = np.array([(1 - u) ** t * u for t in range(h)])
    # Handle u=1: all weight at t=0.
    if weights.sum() < 1e-10:
        weights[0] = 1.0
    weights /= weights.sum()

    sizes = weights * target_delta

    # Apply minimum child size floor (zero out slices below threshold).
    min_size = abs(target_delta) * cfg.min_child_size
    mask = np.abs(sizes) < min_size
    sizes[mask] = 0.0
    # Redistribute zeroed mass to the first non-zero slot.
    residual = target_delta - sizes.sum()
    if abs(residual) > 1e-12:
        nonzero = np.where(~mask)[0]
        if len(nonzero) > 0:
            sizes[nonzero[0]] += residual

    # Compute expected cost.
    cost = 0.0
    for t in range(h):
        if abs(sizes[t]) > 1e-12:
            cost += abs(sizes[t]) * expected_slippage(
                sizes[t], impact_params, current_posteriors
            )

    # Find completion bar.
    cum = np.cumsum(np.abs(sizes))
    target_abs = abs(target_delta)
    completion = int(np.searchsorted(cum, target_abs * 0.999))
    completion = min(completion, h - 1)

    return OrderSchedule(sizes=sizes, expected_cost=cost, completion_bar=completion)


# ---------------------------------------------------------------------------
# TWAP and VWAP reference schedules
# ---------------------------------------------------------------------------


def twap_schedule(target_delta: float, horizon: int) -> np.ndarray:
    """Time-Weighted Average Price execution: equal slices.

    Parameters
    ----------
    target_delta : float
    horizon : int

    Returns
    -------
    np.ndarray, shape (horizon,)
    """
    return np.full(horizon, target_delta / horizon)


def vwap_schedule(
    target_delta: float,
    volume_profile: np.ndarray,
) -> np.ndarray:
    """Volume-Weighted Average Price execution: slices proportional to volume.

    Parameters
    ----------
    target_delta : float
    volume_profile : np.ndarray, shape (horizon,)
        Forecast volume at each bar (e.g. rolling average).

    Returns
    -------
    np.ndarray, shape (horizon,)
    """
    total_vol = volume_profile.sum()
    if total_vol < 1e-10:
        return twap_schedule(target_delta, len(volume_profile))
    return target_delta * volume_profile / total_vol


# ---------------------------------------------------------------------------
# Regime slippage attribution
# ---------------------------------------------------------------------------


def slippage_attribution(
    order_sizes: pd.Series,
    impact_params: RegimeImpactParams,
    posteriors: np.ndarray,
) -> pd.DataFrame:
    """Compute per-bar expected slippage and regime attribution.

    Parameters
    ----------
    order_sizes : pd.Series
        Signed order size at each bar, length T.
    impact_params : RegimeImpactParams
    posteriors : np.ndarray, shape (T, K)

    Returns
    -------
    pd.DataFrame
        Columns: ``expected_slippage``, ``dominant_regime``,
        ``half_spread``, ``impact_component``.
    """
    T = len(order_sizes)
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)

    slippage_vals = np.zeros(T)
    hs_vals = np.zeros(T)
    impact_vals = np.zeros(T)
    dominant = np.zeros(T, dtype=int)

    for t in range(T):
        sz = float(order_sizes.iloc[t])
        pt = P_norm[t]
        hs, ic, av = impact_params.blended(pt)
        normalized = abs(sz) / (av + 1e-10)
        hs_vals[t] = hs
        impact_vals[t] = ic * np.sqrt(normalized)
        slippage_vals[t] = hs + impact_vals[t]
        dominant[t] = int(np.argmax(pt))

    return pd.DataFrame(
        {
            "expected_slippage": slippage_vals,
            "dominant_regime": dominant,
            "half_spread": hs_vals,
            "impact_component": impact_vals,
        },
        index=order_sizes.index,
    )
