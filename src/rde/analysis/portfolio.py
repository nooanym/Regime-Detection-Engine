"""Regime-conditional portfolio allocation.

Translates per-asset regime scores into portfolio weights using three
complementary methods that can be used independently or composed:

1. **Score-proportional weights** — weights ∝ positive regime scores;
   assets in bear regimes get zero weight (long-only) or negative weight
   (long-short). Normalised so |weights| sums to 1.

2. **Volatility targeting** — rescales the weight vector so the expected
   annualised portfolio volatility equals a target level.  Uses a rolling
   covariance estimate so no future data is used.

3. **Kelly-inspired sizing** — scales each asset's weight by
   ``E[r] / Var[r]``, the single-asset Kelly fraction (full Kelly is
   aggressive; multiply by a `fraction` parameter < 1 for fractional
   Kelly).  Regime stats from the most recent training window supply
   the E[r] and Var[r] estimates.

All methods return a ``pd.DataFrame`` of weights aligned to the same
index as the input scores, so they can be fed directly into the backtest
engine (one-asset case) or to a multi-asset equity curve calculator.

Notes
-----
- Weights are computed from information available **at bar t** and applied
  to returns at bar **t+1** (no lookahead).
- The covariance rolling window defaults to 168 bars (one week at hourly
  frequency).  The first 168 bars receive equal weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MIN_WEIGHT = 1e-8  # floor to avoid divide-by-zero in normalisation


@dataclass
class AllocationConfig:
    """Settings for :func:`score_proportional_weights`.

    Attributes
    ----------
    long_only : bool
        If True, assets with score ≤ 0 receive zero weight.
        If False, they receive a negative weight (short position).
    normalise : bool
        If True, rescale weights so their absolute sum equals 1.
    """

    long_only: bool = True
    normalise: bool = True


@dataclass
class VolTargetConfig:
    """Settings for :func:`vol_target_weights`.

    Attributes
    ----------
    target_vol : float
        Target annualised portfolio volatility (e.g. 0.15 = 15%).
    cov_window : int
        Rolling window (bars) for covariance estimation.
    bars_per_year : float
        Used to annualise per-bar covariance.
    max_leverage : float
        Upper bound on the sum of absolute weights. Prevents extreme
        leverage in low-vol regimes.
    """

    target_vol: float = 0.15
    cov_window: int = 168
    bars_per_year: float = 8760.0
    max_leverage: float = 2.0


@dataclass
class KellyConfig:
    """Settings for :func:`kelly_weights`.

    Attributes
    ----------
    fraction : float
        Fraction of the full Kelly bet (default 0.25 = quarter-Kelly).
        Full Kelly (1.0) maximises log-wealth but has extreme drawdowns.
    min_var : float
        Floor on per-asset variance to avoid division by near-zero values.
    long_only : bool
        Clip negative Kelly weights to zero.
    normalise : bool
        Rescale so absolute weights sum to 1 after clipping.
    """

    fraction: float = 0.25
    min_var: float = 1e-8
    long_only: bool = True
    normalise: bool = True


def score_proportional_weights(
    scores: pd.DataFrame,
    config: AllocationConfig | None = None,
) -> pd.DataFrame:
    """Convert regime scores to portfolio weights.

    Parameters
    ----------
    scores : pd.DataFrame
        Shape (T, N). Each column is one asset's regime score in [-1, 1].
    config : AllocationConfig, optional

    Returns
    -------
    pd.DataFrame
        Shape (T, N). Portfolio weights at each bar.

    Notes
    -----
    For ``long_only=True``, only positive scores contribute; the resulting
    weights form a long-only portfolio.  For ``long_only=False``, negative
    scores map to short positions.
    """
    if config is None:
        config = AllocationConfig()

    vals = scores.values.copy().astype(float)

    if config.long_only:
        vals = np.clip(vals, 0.0, None)

    if config.normalise:
        row_abs_sum = np.abs(vals).sum(axis=1, keepdims=True)
        row_abs_sum = np.where(row_abs_sum < _MIN_WEIGHT, 1.0, row_abs_sum)
        vals = vals / row_abs_sum

    return pd.DataFrame(vals, index=scores.index, columns=scores.columns)


def vol_target_weights(
    base_weights: pd.DataFrame,
    returns: pd.DataFrame,
    config: VolTargetConfig | None = None,
) -> pd.DataFrame:
    """Rescale base weights to hit a portfolio volatility target.

    At each bar t the scalar multiplier is:

        multiplier_t = target_vol / portfolio_vol_t

    where ``portfolio_vol_t = sqrt(w_t @ Cov_t @ w_t) * sqrt(bars_per_year)``
    is estimated from the rolling covariance of ``returns`` up to bar t-1.

    Parameters
    ----------
    base_weights : pd.DataFrame
        Shape (T, N). Unnormalised or normalised weight suggestions.
    returns : pd.DataFrame
        Shape (T, N). Per-bar log returns for each asset. Must share the
        same index and columns as ``base_weights``.
    config : VolTargetConfig, optional

    Returns
    -------
    pd.DataFrame
        Shape (T, N). Rescaled weights; absolute sum capped at
        ``config.max_leverage``.
    """
    if config is None:
        config = VolTargetConfig()

    if not returns.index.equals(base_weights.index):
        returns = returns.reindex(base_weights.index)

    assets = base_weights.columns.tolist()
    ret_vals = returns[assets].values.astype(float)
    w_vals = base_weights[assets].values.astype(float)
    T = len(w_vals)
    scaled = np.zeros_like(w_vals)

    ann_factor = config.bars_per_year

    for t in range(T):
        w = w_vals[t]
        abs_sum = np.abs(w).sum()
        if abs_sum < _MIN_WEIGHT:
            scaled[t] = 0.0
            continue

        start = max(0, t - config.cov_window)
        window = ret_vals[start:t]

        if len(window) < 2 or np.all(window == 0):
            cov = np.eye(len(assets)) * (0.01 ** 2 / ann_factor)
        else:
            cov = np.cov(window.T, ddof=1)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])

        port_var = float(w @ cov @ w) * ann_factor
        port_vol = np.sqrt(max(port_var, 1e-12))

        multiplier = min(config.target_vol / port_vol, config.max_leverage / abs_sum)
        scaled[t] = w * multiplier

    return pd.DataFrame(scaled, index=base_weights.index, columns=base_weights.columns)


def kelly_weights(
    returns: pd.DataFrame,
    scores: pd.DataFrame,
    config: KellyConfig | None = None,
    cov_window: int = 168,
) -> pd.DataFrame:
    """Compute fractional Kelly weights from rolling return statistics.

    The single-asset Kelly fraction for asset i at bar t is:

        f_i = mu_i / sigma_i^2

    where mu_i and sigma_i^2 are the rolling mean and variance of returns
    up to bar t-1.  The sign of the bet is taken from the regime score.

    Parameters
    ----------
    returns : pd.DataFrame
        Shape (T, N). Per-bar log returns.
    scores : pd.DataFrame
        Shape (T, N). Regime scores in [-1, 1] giving the bet direction.
    config : KellyConfig, optional
    cov_window : int
        Rolling window for mean/variance estimation.

    Returns
    -------
    pd.DataFrame
        Shape (T, N). Kelly-scaled portfolio weights.
    """
    if config is None:
        config = KellyConfig()

    assets = scores.columns.tolist()
    ret_vals = returns[assets].values.astype(float)
    score_vals = scores[assets].values.astype(float)
    T = len(score_vals)
    kelly_w = np.zeros_like(score_vals)

    for t in range(T):
        start = max(0, t - cov_window)
        window = ret_vals[start:t]

        if len(window) < 2:
            kelly_w[t] = 0.0
            continue

        mu = window.mean(axis=0)
        var = window.var(axis=0, ddof=1)
        var = np.maximum(var, config.min_var)

        raw = config.fraction * (mu / var) * np.sign(score_vals[t])
        kelly_w[t] = raw

    if config.long_only:
        kelly_w = np.clip(kelly_w, 0.0, None)

    if config.normalise:
        row_abs_sum = np.abs(kelly_w).sum(axis=1, keepdims=True)
        row_abs_sum = np.where(row_abs_sum < _MIN_WEIGHT, 1.0, row_abs_sum)
        kelly_w = kelly_w / row_abs_sum

    return pd.DataFrame(kelly_w, index=scores.index, columns=scores.columns)


def portfolio_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_cost_bps: float = 1.0,
) -> pd.Series:
    """Compute portfolio returns from a weight matrix and per-asset returns.

    Applies the no-lookahead shift: ``weights.shift(1)`` is used so each
    bar's return is earned from the previous bar's weights.

    Parameters
    ----------
    weights : pd.DataFrame
        Shape (T, N). Portfolio weights at each bar.
    returns : pd.DataFrame
        Shape (T, N). Per-bar log returns.
    transaction_cost_bps : float
        Cost per unit of weight change per asset.

    Returns
    -------
    pd.Series
        Shape (T,). Net portfolio return at each bar.
    """
    assets = weights.columns.tolist()
    ret_vals = returns[assets].reindex(weights.index).values.astype(float)
    w_vals = weights.values.astype(float)

    lagged_w = np.vstack([np.zeros((1, w_vals.shape[1])), w_vals[:-1]])
    gross_ret = (lagged_w * ret_vals).sum(axis=1)

    cost_per_unit = transaction_cost_bps / 10_000.0
    turnover = np.abs(w_vals - lagged_w).sum(axis=1)
    costs = turnover * cost_per_unit
    costs[0] = 0.0  # no startup cost

    net_ret = gross_ret - costs
    return pd.Series(net_ret, index=weights.index, name="portfolio_return")
