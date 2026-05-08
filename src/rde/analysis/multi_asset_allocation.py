"""Regime-conditional multi-asset portfolio allocation (Phase 42).

Walk-forward allocation across N assets where each asset has its own HMM
fitted on a training window.  At each monthly rebalance date:

1. For each asset, fit HMM on training window features.
2. Run OnlineDecoder through the training window; take the final-bar posterior.
3. Compute posterior-weighted expected return per asset:
   ``E[r_i] = sum_k(p_k * mu_k[log_return_feature])``
4. Estimate joint covariance from the rolling return history.
5. Apply max-Sharpe MVO (or min-var) → base weights.
6. Optionally rescale for a portfolio vol target.
7. Hold weights for the next rebalance period; compute net-of-cost returns.

Key invariant: at rebalance date t, only data in [t - lookback_bars, t) is
used.  Weights are applied to returns at bar t, t+1, ... (no lookahead).

Public API
----------
MultiAssetConfig
MultiAssetResult
run_multi_asset_allocation
equal_weight_baseline
global_min_var_baseline
compare_allocations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from rde.analysis.portfolio_optimization import (
    MVOConfig,
    _box_project,
    _max_sharpe_weights,
    _min_variance_weights,
    _weighted_moments,
)
from rde.analysis.portfolio import portfolio_returns
from rde.inference.online import OnlineDecoder
from rde.models.hmm import train_hmm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MultiAssetConfig:
    """Configuration for :func:`run_multi_asset_allocation`.

    Attributes
    ----------
    ann_factor : int
        Bars per year (252 for daily, 8760 for hourly).
    rebalance_bars : int
        Number of bars between portfolio rebalances.
    lookback_bars : int
        Training window length for each per-asset HMM fit.
    cov_window_bars : int
        Rolling window length for joint covariance estimation.
    n_states : int
        Number of HMM states for each asset.
    n_restarts : int
        HMM training restarts (higher → more robust but slower).
    transaction_cost : float
        One-way cost as a fraction of weight change per asset.
    target_vol : float or None
        If set, rescale final weights so expected portfolio vol equals this
        level (annualised).  ``None`` disables vol targeting.
    mvo_kind : str
        ``"max_sharpe"`` — regime-informed expected-return MVO.
        ``"min_var"``    — ignore expected returns, minimise variance.
    min_weight : float
        Minimum weight per asset (0.0 = long-only floor).
    max_weight : float
        Maximum weight per asset (caps single-asset concentration).
    regularisation : float
        Ridge regularisation on covariance diagonal for invertibility.
    log_return_col : str
        Name of the log-return feature column in each asset's feature
        DataFrame; used to read regime-conditional expected returns.
    """

    ann_factor: int = 252
    rebalance_bars: int = 21
    lookback_bars: int = 504
    cov_window_bars: int = 63
    n_states: int = 8
    n_restarts: int = 3
    transaction_cost: float = 0.001
    target_vol: float | None = 0.10
    mvo_kind: str = "max_sharpe"
    min_weight: float = 0.0
    max_weight: float = 0.60
    regularisation: float = 1e-4
    log_return_col: str = "log_return"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MultiAssetResult:
    """Output of :func:`run_multi_asset_allocation` and baselines.

    Attributes
    ----------
    name : str
        Strategy label.
    weights : pd.DataFrame
        Shape (T, N).  Portfolio weights at each bar (pre-shift; applied to
        returns at bar t+1 via the ``portfolio_returns`` function).
    portfolio_returns : pd.Series
        Net-of-cost per-bar portfolio returns.
    equity : pd.Series
        Cumulative equity curve starting at 1.0.
    asset_returns : pd.DataFrame
        Shape (T, N).  Per-asset return series on the aligned index.
    rebalance_dates : list[pd.Timestamp]
        Dates on which the portfolio was rebalanced.
    sharpe : float
    calmar : float
    max_drawdown : float
    ann_return : float
    ann_vol : float
    """

    name: str
    weights: pd.DataFrame
    portfolio_returns: pd.Series
    equity: pd.Series
    asset_returns: pd.DataFrame
    rebalance_dates: list[pd.Timestamp]
    sharpe: float
    calmar: float
    max_drawdown: float
    ann_return: float
    ann_vol: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_metrics(
    port_returns: pd.Series,
    ann_factor: int,
) -> tuple[float, float, float, float, float]:
    """Return (sharpe, calmar, max_drawdown, ann_return, ann_vol)."""
    r = port_returns.values.astype(float)
    mu = float(r.mean()) * ann_factor
    sig = float(r.std(ddof=1)) * np.sqrt(ann_factor) + 1e-15
    sharpe = (mu - 0.0) / sig

    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-15)
    mdd = float(abs(dd.min()))
    calmar = mu / (mdd + 1e-15)

    return sharpe, calmar, mdd, mu, sig


def _vol_target_scale(
    weights: np.ndarray,
    cov: np.ndarray,
    target_vol: float,
    ann_factor: int,
    max_leverage: float = 2.0,
) -> np.ndarray:
    """Rescale weights to hit target portfolio vol."""
    port_var = float(weights @ cov @ weights) * ann_factor
    port_vol = np.sqrt(max(port_var, 1e-12))
    scale = min(target_vol / port_vol, max_leverage / (np.abs(weights).sum() + 1e-15))
    return weights * scale


def _posterior_expected_return(
    decoder: OnlineDecoder,
    X_train: np.ndarray,
    log_return_idx: int,
) -> float:
    """Run decoder through training data; return posterior-weighted mean return.

    Uses :meth:`OnlineDecoder.batch_filter` so the final posterior reflects
    all training bars causally.  The expected return is:

        E[r] = sum_k( p_k * mu_k[log_return_idx] )

    where ``mu_k`` is the k-th row of the HMM emission means (original space).
    """
    posteriors = decoder.batch_filter(X_train)  # (T, K)
    last_posterior = posteriors[-1]  # (K,)

    # OnlineDecoder stores the scaled HMM as _model and its scaler as _scaler.
    means_scaled = decoder._model.means_  # (K, n_features)
    means_orig = decoder._scaler.inverse_transform(means_scaled)  # (K, n_features)
    regime_returns = means_orig[:, log_return_idx]  # (K,)
    return float(last_posterior @ regime_returns)


def _fit_and_decode(
    X_train: np.ndarray,
    n_states: int,
    n_restarts: int,
    train_kwargs: dict,
) -> tuple[OnlineDecoder, np.ndarray]:
    """Fit HMM and return (decoder, posteriors_on_train)."""
    fitted = train_hmm(
        X_train,
        n_states=n_states,
        n_restarts=n_restarts,
        **train_kwargs,
    )
    decoder = OnlineDecoder(fitted)
    posteriors = decoder.batch_filter(X_train)
    return decoder, posteriors


# ---------------------------------------------------------------------------
# Main allocation function
# ---------------------------------------------------------------------------


def run_multi_asset_allocation(
    asset_returns: pd.DataFrame,
    asset_features: dict[str, pd.DataFrame],
    config: MultiAssetConfig | None = None,
    train_kwargs: dict | None = None,
) -> MultiAssetResult:
    """Walk-forward regime-conditional multi-asset allocation.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Shape (T, N).  Per-bar log returns for each asset.  Index must be a
        DatetimeIndex; columns are asset names.
    asset_features : dict[str, pd.DataFrame]
        Keyed by asset name (matching ``asset_returns`` columns).  Each
        DataFrame has one row per bar and feature columns used for HMM
        fitting.  Must cover the same date range as ``asset_returns``.
    config : MultiAssetConfig, optional
    train_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`rde.models.hmm.train_hmm`
        (e.g. ``n_iter``, ``covariance_type``).

    Returns
    -------
    MultiAssetResult
    """
    cfg = config or MultiAssetConfig()
    tkwargs = train_kwargs or {}
    assets = list(asset_returns.columns)
    N = len(assets)

    if N < 2:
        raise ValueError("multi_asset_allocation requires at least 2 assets.")

    # Align feature DataFrames to the returns index.
    aligned_features = {
        a: asset_features[a].reindex(asset_returns.index).ffill().dropna()
        for a in assets
    }

    # Weight matrix: initialise to equal weight; will be filled at each rebalance.
    T = len(asset_returns)
    weight_arr = np.full((T, N), 1.0 / N)
    rebalance_dates: list[pd.Timestamp] = []

    for t in range(cfg.lookback_bars, T, cfg.rebalance_bars):
        train_slice = slice(t - cfg.lookback_bars, t)
        cov_slice = slice(max(0, t - cfg.cov_window_bars), t)

        # --- Joint covariance from rolling return window ---
        ret_window = asset_returns.iloc[cov_slice].values.astype(float)  # (cov_w, N)
        if ret_window.shape[0] < max(4, N + 1):
            continue  # not enough history; keep equal-weight
        cov_joint = np.cov(ret_window.T, ddof=1)
        if cov_joint.ndim == 0:
            cov_joint = np.array([[float(cov_joint)]])
        cov_joint += np.eye(N) * cfg.regularisation

        # --- Expected returns per asset from HMM posteriors ---
        exp_returns = np.zeros(N)
        for i, asset in enumerate(assets):
            feat_df = aligned_features[asset]
            feat_train = feat_df.iloc[train_slice].values.astype(float)

            if feat_train.shape[0] < cfg.n_states * 2:
                continue  # not enough training data

            # Identify log_return column index.
            feat_cols = list(feat_df.columns)
            if cfg.log_return_col in feat_cols:
                lr_idx = feat_cols.index(cfg.log_return_col)
            else:
                lr_idx = 0  # fallback

            try:
                fitted = train_hmm(
                    feat_train,
                    n_states=cfg.n_states,
                    n_restarts=cfg.n_restarts,
                    **tkwargs,
                )
                decoder = OnlineDecoder(fitted)
                exp_returns[i] = _posterior_expected_return(decoder, feat_train, lr_idx)
            except Exception as exc:
                logger.warning("HMM fit failed for %s at bar %d: %s", asset, t, exc)

        # --- MVO weights ---
        if cfg.mvo_kind == "min_var":
            weights = _min_variance_weights(cov_joint, cfg.min_weight, cfg.max_weight)
        else:
            weights = _max_sharpe_weights(
                exp_returns, cov_joint, 0.0, cfg.min_weight, cfg.max_weight
            )

        # --- Vol targeting overlay ---
        if cfg.target_vol is not None:
            weights = _vol_target_scale(
                weights, cov_joint, cfg.target_vol, cfg.ann_factor
            )
            # After scaling, re-project onto weight constraints.
            total = weights.sum()
            if total > 1e-10:
                weights = np.clip(weights / total, cfg.min_weight, cfg.max_weight)
                weights = _box_project(weights, cfg.min_weight, cfg.max_weight, N)

        # Fill the next rebalance period with these weights.
        end = min(t + cfg.rebalance_bars, T)
        weight_arr[t:end] = weights[None, :]
        rebalance_dates.append(asset_returns.index[t])

    weights_df = pd.DataFrame(weight_arr, index=asset_returns.index, columns=assets)

    port_ret = portfolio_returns(
        weights_df,
        asset_returns,
        transaction_cost_bps=cfg.transaction_cost * 10_000,
    )
    equity = (1.0 + port_ret).cumprod()

    sharpe, calmar, mdd, ann_ret, ann_vol = _compute_metrics(port_ret, cfg.ann_factor)

    return MultiAssetResult(
        name="regime_conditional_mvo",
        weights=weights_df,
        portfolio_returns=port_ret,
        equity=equity,
        asset_returns=asset_returns.copy(),
        rebalance_dates=rebalance_dates,
        sharpe=sharpe,
        calmar=calmar,
        max_drawdown=mdd,
        ann_return=ann_ret,
        ann_vol=ann_vol,
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def equal_weight_baseline(
    asset_returns: pd.DataFrame,
    ann_factor: int = 252,
    transaction_cost: float = 0.001,
) -> MultiAssetResult:
    """Always equal-weight, buy-and-hold (rebalanced monthly to equal weight).

    Uses :func:`rde.analysis.portfolio.portfolio_returns` with a constant
    weight matrix for a clean cost comparison.
    """
    T, N = asset_returns.shape
    w_arr = np.full((T, N), 1.0 / N)
    weights_df = pd.DataFrame(w_arr, index=asset_returns.index, columns=asset_returns.columns)

    port_ret = portfolio_returns(
        weights_df,
        asset_returns,
        transaction_cost_bps=transaction_cost * 10_000,
    )
    equity = (1.0 + port_ret).cumprod()
    sharpe, calmar, mdd, ann_ret, ann_vol = _compute_metrics(port_ret, ann_factor)

    return MultiAssetResult(
        name="equal_weight",
        weights=weights_df,
        portfolio_returns=port_ret,
        equity=equity,
        asset_returns=asset_returns.copy(),
        rebalance_dates=[],
        sharpe=sharpe,
        calmar=calmar,
        max_drawdown=mdd,
        ann_return=ann_ret,
        ann_vol=ann_vol,
    )


def global_min_var_baseline(
    asset_returns: pd.DataFrame,
    ann_factor: int = 252,
    rebalance_bars: int = 21,
    cov_window_bars: int = 63,
    transaction_cost: float = 0.001,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    regularisation: float = 1e-4,
) -> MultiAssetResult:
    """Monthly-rebalanced minimum-variance portfolio (no regime information).

    Provides the upper bound of what a *purely covariance-driven* allocator
    can achieve, establishing whether regime-conditioned expected returns add
    value above and beyond variance reduction.
    """
    T, N = asset_returns.shape
    assets = list(asset_returns.columns)
    weight_arr = np.full((T, N), 1.0 / N)
    rebalance_dates: list[pd.Timestamp] = []

    for t in range(cov_window_bars, T, rebalance_bars):
        cov_slice = slice(max(0, t - cov_window_bars), t)
        ret_window = asset_returns.iloc[cov_slice].values.astype(float)
        if ret_window.shape[0] < max(4, N + 1):
            continue
        cov_joint = np.cov(ret_window.T, ddof=1)
        if cov_joint.ndim == 0:
            cov_joint = np.array([[float(cov_joint)]])
        cov_joint += np.eye(N) * regularisation
        weights = _min_variance_weights(cov_joint, min_weight, max_weight)
        end = min(t + rebalance_bars, T)
        weight_arr[t:end] = weights[None, :]
        rebalance_dates.append(asset_returns.index[t])

    weights_df = pd.DataFrame(weight_arr, index=asset_returns.index, columns=assets)
    port_ret = portfolio_returns(
        weights_df,
        asset_returns,
        transaction_cost_bps=transaction_cost * 10_000,
    )
    equity = (1.0 + port_ret).cumprod()
    sharpe, calmar, mdd, ann_ret, ann_vol = _compute_metrics(port_ret, ann_factor)

    return MultiAssetResult(
        name="global_min_var",
        weights=weights_df,
        portfolio_returns=port_ret,
        equity=equity,
        asset_returns=asset_returns.copy(),
        rebalance_dates=rebalance_dates,
        sharpe=sharpe,
        calmar=calmar,
        max_drawdown=mdd,
        ann_return=ann_ret,
        ann_vol=ann_vol,
    )


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def compare_allocations(
    results: dict[str, MultiAssetResult],
    ann_factor: int = 252,
) -> pd.DataFrame:
    """Tabulate performance metrics for a set of allocation strategies.

    Parameters
    ----------
    results : dict[str, MultiAssetResult]
        Keyed by strategy label.  Values from :func:`run_multi_asset_allocation`
        or the baseline functions.
    ann_factor : int

    Returns
    -------
    pd.DataFrame
        Columns: name, sharpe, calmar, max_drawdown, ann_return, ann_vol.
        Sorted descending by Sharpe.
    """
    rows = []
    for name, r in results.items():
        rows.append({
            "name": name,
            "sharpe": round(r.sharpe, 3),
            "calmar": round(r.calmar, 3),
            "max_drawdown": round(r.max_drawdown, 4),
            "ann_return": round(r.ann_return, 4),
            "ann_vol": round(r.ann_vol, 4),
            "n_rebalances": len(r.rebalance_dates),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("sharpe", ascending=False).reset_index(drop=True)
