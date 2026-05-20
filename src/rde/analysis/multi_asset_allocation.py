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
regime_informed_min_var
equal_weight_baseline
global_min_var_baseline
compare_allocations
compute_rtmv_weights_now
compute_rtmv_weights_now_joint
compute_momentum_scores
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
        # Scale the entire position down so portfolio vol hits target_vol.
        # The unallocated fraction goes to cash — do NOT renormalise back
        # to sum=1 (that would undo the vol scaling entirely).
        if cfg.target_vol is not None:
            weights = _vol_target_scale(
                weights, cov_joint, cfg.target_vol, cfg.ann_factor
            )
            weights = np.clip(weights, cfg.min_weight, cfg.max_weight)

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
# Regime-informed minimum-variance (Phase 42c probe)
# ---------------------------------------------------------------------------


def regime_informed_min_var(
    asset_returns: pd.DataFrame,
    asset_features: dict[str, pd.DataFrame],
    config: MultiAssetConfig | None = None,
    exclusion_threshold: float = 0.0,
    min_eligible_assets: int = 2,
    train_kwargs: dict | None = None,
) -> MultiAssetResult:
    """Walk-forward minimum-variance with regime-based asset exclusion.

    At each monthly rebalance, assets whose HMM posterior-weighted expected
    return falls below ``exclusion_threshold`` are excluded (weight set to
    zero).  The remaining *eligible* assets receive global-min-var weights
    estimated from the rolling joint covariance.

    This probes whether regime-conditional exclusions add value on top of
    the structural variance advantage of global min-var (Sharpe ~0.94 in the
    Phase 42b backtest with BTC/ETH/SPY/GLD).

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Shape (T, N).  Per-bar log returns.  Must have a DatetimeIndex.
    asset_features : dict[str, pd.DataFrame]
        Per-asset feature DataFrames (same index as ``asset_returns``).
    config : MultiAssetConfig, optional
        Fields used: ``ann_factor``, ``rebalance_bars``, ``lookback_bars``,
        ``cov_window_bars``, ``n_states``, ``n_restarts``,
        ``transaction_cost``, ``min_weight``, ``max_weight``,
        ``regularisation``, ``log_return_col``.
        ``mvo_kind`` and ``target_vol`` are ignored (always min-var,
        no vol target).
    exclusion_threshold : float
        Assets with posterior-weighted E[r] below this value are excluded.
        Default ``0.0`` excludes assets in net-negative-return regimes.
    min_eligible_assets : int
        Minimum assets to keep eligible.  If the threshold would exclude
        more than ``N - min_eligible_assets`` assets, the
        ``min_eligible_assets`` assets with the highest E[r] are retained.
    train_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`rde.models.hmm.train_hmm`.

    Returns
    -------
    MultiAssetResult
        ``name = "regime_informed_min_var"``
    """
    cfg = config or MultiAssetConfig()
    tkwargs = train_kwargs or {}
    assets = list(asset_returns.columns)
    N = len(assets)

    if N < 2:
        raise ValueError("regime_informed_min_var requires at least 2 assets.")
    if min_eligible_assets < 2:
        raise ValueError("min_eligible_assets must be at least 2.")

    aligned_features = {
        a: asset_features[a].reindex(asset_returns.index).ffill().dropna()
        for a in assets
    }

    T = len(asset_returns)
    weight_arr = np.full((T, N), 1.0 / N)
    rebalance_dates: list[pd.Timestamp] = []

    for t in range(cfg.lookback_bars, T, cfg.rebalance_bars):
        train_slice = slice(t - cfg.lookback_bars, t)
        cov_slice = slice(max(0, t - cfg.cov_window_bars), t)

        # Joint covariance from rolling return window.
        ret_window = asset_returns.iloc[cov_slice].values.astype(float)
        if ret_window.shape[0] < max(4, N + 1):
            continue
        cov_joint = np.cov(ret_window.T, ddof=1)
        if cov_joint.ndim == 0:
            cov_joint = np.array([[float(cov_joint)]])
        cov_joint += np.eye(N) * cfg.regularisation

        # Posterior-weighted expected return per asset.
        exp_returns = np.zeros(N)
        for i, asset in enumerate(assets):
            feat_df = aligned_features[asset]
            feat_train = feat_df.iloc[train_slice].values.astype(float)
            if feat_train.shape[0] < cfg.n_states * 2:
                continue
            feat_cols = list(feat_df.columns)
            lr_idx = (
                feat_cols.index(cfg.log_return_col)
                if cfg.log_return_col in feat_cols
                else 0
            )
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
                logger.warning(
                    "HMM fit failed for %s at bar %d: %s", asset, t, exc
                )

        # Determine eligible asset set.
        eligible_mask = exp_returns >= exclusion_threshold
        if int(eligible_mask.sum()) < min_eligible_assets:
            # Relax: keep top-min_eligible by expected return.
            top_idx = np.argsort(exp_returns)[::-1][:min_eligible_assets]
            eligible_mask = np.zeros(N, dtype=bool)
            eligible_mask[top_idx] = True

        eligible_idx = np.where(eligible_mask)[0]
        n_el = len(eligible_idx)

        # Min-var on eligible sub-universe; embed back into full weight vector.
        if n_el == N:
            weights = _min_variance_weights(cov_joint, cfg.min_weight, cfg.max_weight)
        else:
            cov_sub = cov_joint[np.ix_(eligible_idx, eligible_idx)]
            sub_w = _min_variance_weights(cov_sub, cfg.min_weight, cfg.max_weight)
            weights = np.zeros(N)
            weights[eligible_idx] = sub_w

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
        name="regime_informed_min_var",
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


def regime_tilted_min_var(
    asset_returns: pd.DataFrame,
    asset_features: dict[str, pd.DataFrame],
    config: MultiAssetConfig | None = None,
    lambda_tilt: float = 0.1,
    train_kwargs: dict | None = None,
    shuffle_seed: int | None = None,
) -> MultiAssetResult:
    """Walk-forward min-var tilted toward high-expected-return assets.

    Combines the structural variance advantage of global min-var with a
    gentle regime signal tilt:

        w = (1 - lambda) * w_minvar + lambda * w_regime

    where ``w_regime`` allocates proportionally to each asset's
    HMM posterior-weighted expected return (assets with E[r] ≤ 0 get zero
    regime weight; fallback to equal-weight when all E[r] ≤ 0).

    At ``lambda=0`` this is exactly ``global_min_var_baseline``.
    At ``lambda=1`` it is pure regime-proportional allocation.
    Practical range: 0.05–0.30.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Shape (T, N).  Per-bar log returns with DatetimeIndex.
    asset_features : dict[str, pd.DataFrame]
        Per-asset feature DataFrames (same index as ``asset_returns``).
    config : MultiAssetConfig, optional
        Uses ``ann_factor``, ``rebalance_bars``, ``lookback_bars``,
        ``cov_window_bars``, ``n_states``, ``n_restarts``,
        ``transaction_cost``, ``min_weight``, ``max_weight``,
        ``regularisation``, ``log_return_col``.
        ``mvo_kind`` and ``target_vol`` are ignored.
    lambda_tilt : float
        Mixing weight in [0, 1].  0 = pure min-var; 1 = pure regime.
    train_kwargs : dict, optional
        Forwarded to :func:`rde.models.hmm.train_hmm`.

    Returns
    -------
    MultiAssetResult
        ``name = "regime_tilted_min_var"``
    """
    cfg = config or MultiAssetConfig()
    tkwargs = train_kwargs or {}
    assets = list(asset_returns.columns)
    N = len(assets)
    _shuffle_rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None

    if N < 2:
        raise ValueError("regime_tilted_min_var requires at least 2 assets.")
    if not (0.0 <= lambda_tilt <= 1.0):
        raise ValueError("lambda_tilt must be in [0, 1].")

    aligned_features = {
        a: asset_features[a].reindex(asset_returns.index).ffill().dropna()
        for a in assets
    }

    T = len(asset_returns)
    weight_arr = np.full((T, N), 1.0 / N)
    rebalance_dates: list[pd.Timestamp] = []

    for t in range(cfg.lookback_bars, T, cfg.rebalance_bars):
        train_slice = slice(t - cfg.lookback_bars, t)
        cov_slice = slice(max(0, t - cfg.cov_window_bars), t)

        # Joint covariance from rolling return window.
        ret_window = asset_returns.iloc[cov_slice].values.astype(float)
        if ret_window.shape[0] < max(4, N + 1):
            continue
        cov_joint = np.cov(ret_window.T, ddof=1)
        if cov_joint.ndim == 0:
            cov_joint = np.array([[float(cov_joint)]])
        cov_joint += np.eye(N) * cfg.regularisation

        # Global min-var weights.
        w_minvar = _min_variance_weights(cov_joint, cfg.min_weight, cfg.max_weight)

        # Posterior-weighted expected return per asset.
        exp_returns = np.zeros(N)
        for i, asset in enumerate(assets):
            feat_df = aligned_features[asset]
            feat_train = feat_df.iloc[train_slice].values.astype(float)
            if feat_train.shape[0] < cfg.n_states * 2:
                continue
            feat_cols = list(feat_df.columns)
            lr_idx = (
                feat_cols.index(cfg.log_return_col)
                if cfg.log_return_col in feat_cols
                else 0
            )
            try:
                fitted = train_hmm(
                    feat_train,
                    n_states=cfg.n_states,
                    n_restarts=cfg.n_restarts,
                    **tkwargs,
                )
                decoder = OnlineDecoder(fitted)
                exp_returns[i] = _posterior_expected_return(
                    decoder, feat_train, lr_idx
                )
            except Exception as exc:
                logger.warning(
                    "HMM fit failed for %s at bar %d: %s", asset, t, exc
                )

        # Optionally shuffle expected returns (for shuffle robustness test).
        if _shuffle_rng is not None:
            exp_returns = _shuffle_rng.permutation(exp_returns)

        # Regime weights: proportional to positive expected return.
        e_r_pos = np.maximum(exp_returns, 0.0)
        if e_r_pos.sum() > 1e-15:
            w_regime = e_r_pos / e_r_pos.sum()
        else:
            w_regime = np.ones(N) / N

        # Convex combination, then clip and renormalize.
        w_raw = (1.0 - lambda_tilt) * w_minvar + lambda_tilt * w_regime
        w_raw = np.clip(w_raw, cfg.min_weight, cfg.max_weight)
        total = w_raw.sum()
        if total > 1e-15:
            w_raw /= total

        end = min(t + cfg.rebalance_bars, T)
        weight_arr[t:end] = w_raw[None, :]
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
        name="regime_tilted_min_var",
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


def collect_rtmv_rebalance_cache(
    asset_returns: pd.DataFrame,
    asset_features: dict[str, pd.DataFrame],
    config: MultiAssetConfig | None = None,
    train_kwargs: dict | None = None,
) -> list[dict]:
    """Run one HMM fitting pass and cache per-rebalance (w_minvar, exp_returns).

    This allows the Phase 47 validation to reuse HMM fits across the cost
    sensitivity sweep and shuffle test instead of refitting from scratch each
    time.  The returned cache is a list of dicts, one per rebalance step:

        {"t": int, "w_minvar": np.ndarray(N,), "exp_returns": np.ndarray(N,)}

    Parameters
    ----------
    asset_returns, asset_features, config, train_kwargs
        Same as :func:`regime_tilted_min_var`.

    Returns
    -------
    list[dict]
        Ordered by rebalance bar index.
    """
    cfg = config or MultiAssetConfig()
    tkwargs = train_kwargs or {}
    assets = list(asset_returns.columns)
    N = len(assets)
    aligned_features = {
        a: asset_features[a].reindex(asset_returns.index).ffill().dropna()
        for a in assets
    }
    T = len(asset_returns)
    cache: list[dict] = []

    for t in range(cfg.lookback_bars, T, cfg.rebalance_bars):
        train_slice = slice(t - cfg.lookback_bars, t)
        cov_slice = slice(max(0, t - cfg.cov_window_bars), t)

        ret_window = asset_returns.iloc[cov_slice].values.astype(float)
        if ret_window.shape[0] < max(4, N + 1):
            continue
        cov_joint = np.cov(ret_window.T, ddof=1)
        if cov_joint.ndim == 0:
            cov_joint = np.array([[float(cov_joint)]])
        cov_joint += np.eye(N) * cfg.regularisation
        w_minvar = _min_variance_weights(cov_joint, cfg.min_weight, cfg.max_weight)

        exp_returns = np.zeros(N)
        for i, asset in enumerate(assets):
            feat_df = aligned_features[asset]
            feat_train = feat_df.iloc[train_slice].values.astype(float)
            if feat_train.shape[0] < cfg.n_states * 2:
                continue
            feat_cols = list(feat_df.columns)
            lr_idx = (
                feat_cols.index(cfg.log_return_col)
                if cfg.log_return_col in feat_cols
                else 0
            )
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

        cache.append({"t": t, "w_minvar": w_minvar.copy(), "exp_returns": exp_returns.copy()})

    return cache


def regime_tilted_min_var_from_cache(
    asset_returns: pd.DataFrame,
    rebalance_cache: list[dict],
    config: MultiAssetConfig | None = None,
    lambda_tilt: float = 0.1,
    shuffle_seed: int | None = None,
) -> MultiAssetResult:
    """Apply RTMV blending using pre-computed (w_minvar, exp_returns) cache.

    Skips HMM fitting entirely — uses cached values from
    :func:`collect_rtmv_rebalance_cache`.  Used for cost sweeps and shuffle
    tests where only the post-fit blending logic changes.

    Parameters
    ----------
    asset_returns : pd.DataFrame
    rebalance_cache : list[dict]
        Output of :func:`collect_rtmv_rebalance_cache`.
    config : MultiAssetConfig, optional
        Uses ``ann_factor``, ``transaction_cost``, ``min_weight``,
        ``max_weight``, ``rebalance_bars``.
    lambda_tilt : float
    shuffle_seed : int, optional
        If set, permutes ``exp_returns`` at each rebalance step.

    Returns
    -------
    MultiAssetResult
    """
    cfg = config or MultiAssetConfig()
    assets = list(asset_returns.columns)
    N = len(assets)
    T = len(asset_returns)
    _shuffle_rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None

    weight_arr = np.full((T, N), 1.0 / N)
    rebalance_dates: list[pd.Timestamp] = []

    for entry in rebalance_cache:
        t = entry["t"]
        w_minvar = entry["w_minvar"].copy()
        exp_returns = entry["exp_returns"].copy()

        if _shuffle_rng is not None:
            exp_returns = _shuffle_rng.permutation(exp_returns)

        e_r_pos = np.maximum(exp_returns, 0.0)
        if e_r_pos.sum() > 1e-15:
            w_regime = e_r_pos / e_r_pos.sum()
        else:
            w_regime = np.ones(N) / N

        w_raw = (1.0 - lambda_tilt) * w_minvar + lambda_tilt * w_regime
        w_raw = np.clip(w_raw, cfg.min_weight, cfg.max_weight)
        total = w_raw.sum()
        if total > 1e-15:
            w_raw /= total

        end = min(t + cfg.rebalance_bars, T)
        weight_arr[t:end] = w_raw[None, :]
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
        name="regime_tilted_min_var",
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


# ---------------------------------------------------------------------------
# Live / single-step weight computation (Phase 48)
# ---------------------------------------------------------------------------


def compute_momentum_scores(
    asset_returns_window: pd.DataFrame,
    momentum_lookback: int = 252,
    momentum_skip: int = 21,
) -> dict[str, float]:
    """Compute 12m-1m cross-sectional momentum score per asset.

    Uses the cumulative log-return in the window from bar ``-(lookback)`` to
    bar ``-(skip)`` (inclusive), which equals ``log(P[t-skip] / P[t-lookback])``.
    Assets where the window is too short to compute the score receive a score
    of 0.0 (neutral).

    Parameters
    ----------
    asset_returns_window : pd.DataFrame
        Shape ``(W, N)``.  Per-bar log returns.
    momentum_lookback : int
        Number of bars for the lookback period (default 252 = 12 months).
    momentum_skip : int
        Number of recent bars to skip to avoid short-term reversal
        (default 21 = 1 month).

    Returns
    -------
    dict[str, float]
        Map from asset name to momentum score (positive = trend-following,
        negative = counter-trend).
    """
    assets = list(asset_returns_window.columns)
    T = len(asset_returns_window)
    scores: dict[str, float] = {}
    for a in assets:
        ret_series = asset_returns_window[a].values
        # Need at least lookback + 1 bars to compute from -(lookback) to -(skip).
        if T < momentum_lookback + 1:
            scores[a] = 0.0
            continue
        # Cumulative log return from bar -(lookback) to bar -(skip):
        # log(P[t-skip] / P[t-lookback]) = sum of log returns in (t-lookback, t-skip].
        start_idx = T - momentum_lookback
        end_idx = T - momentum_skip
        if end_idx <= start_idx:
            scores[a] = 0.0
            continue
        scores[a] = float(ret_series[start_idx:end_idx].sum())
    return scores


def compute_rtmv_weights_now(
    asset_returns_window: pd.DataFrame,
    asset_features_window: dict[str, pd.DataFrame],
    config: MultiAssetConfig | None = None,
    lambda_tilt: float = 0.05,
    train_kwargs: dict | None = None,
    adaptive_lambda: bool = False,
    lambda_min: float = 0.02,
    lambda_max: float = 0.15,
    return_posteriors: bool = False,
    return_models: bool = False,
    lambda_by_state_rank: list[float] | None = None,
    lambda_proxy_asset: str | None = None,
    momentum_overlay: bool = False,
    momentum_lookback: int = 252,
    momentum_skip: int = 21,
    momentum_lambda_high: float = 0.15,
    momentum_lambda_low: float = 0.01,
    momentum_tilt_scale: float = 0.0,
) -> dict[str, float] | tuple:
    """Compute RTMV target weights from a fixed-length lookback window.

    This is the live-compatible version of the per-step logic inside
    :func:`regime_tilted_min_var`.  Call at each monthly rebalance boundary
    with the last ``lookback_bars`` rows of returns and features.

    Parameters
    ----------
    asset_returns_window : pd.DataFrame
        Shape ``(W, N)``.  Recent log returns, one column per asset.
        ``W`` should be at least ``config.cov_window_bars``.
    asset_features_window : dict[str, pd.DataFrame]
        Per-asset feature DataFrames aligned to the same index.
    config : MultiAssetConfig, optional
        Uses ``cov_window_bars``, ``n_states``, ``n_restarts``,
        ``min_weight``, ``max_weight``, ``regularisation``,
        ``log_return_col``.  ``rebalance_bars`` and ``lookback_bars``
        are ignored (the caller controls the window size).
    lambda_tilt : float
        Blending weight in [0, 1].  Used when ``adaptive_lambda=False``.
    train_kwargs : dict, optional
        Forwarded to :func:`rde.models.hmm.train_hmm`.
    adaptive_lambda : bool
        If True, derive λ from average per-asset last-bar posterior entropy:
        confident regimes (low entropy) → λ_max; uncertain (high entropy) →
        λ_min.  Overrides ``lambda_tilt``.
    lambda_min : float
        Floor for the adaptive λ schedule (used when posteriors are diffuse).
    lambda_max : float
        Ceiling for the adaptive λ schedule (used when posteriors are sharp).
    return_posteriors : bool
        If ``True``, also return a per-asset dict of last-bar HMM posteriors
        (shape ``(n_states,)`` each). Used by posterior-triggered rebalancers
        (Phase 50b/53) to compute KL divergence between rebalances.  Assets
        where HMM fitting failed map to a uniform posterior.
    return_models : bool
        If ``True``, also return the per-asset ``FittedModel`` objects.
        Used by Phase 53 to initialise ``OnlineDecoder`` instances that are
        stepped cheaply between rebalances (no refit).
    lambda_by_state_rank : list[float], optional
        Regime-conditional λ schedule (Phase 51/51b).  Length must equal
        ``config.n_states``.  Element ``k`` is the λ to use when the
        portfolio-level dominant state has return rank ``k`` (rank 0 =
        lowest mean-return state, rank n_states-1 = highest).
        Overrides ``lambda_tilt`` and ``adaptive_lambda`` when provided.
    lambda_proxy_asset : str, optional
        Phase 51b: when set alongside ``lambda_by_state_rank``, use *only*
        this asset's dominant state rank to select λ (ignores all other
        assets).  Avoids the Phase 51 averaging problem where SPY rank-0
        (bear) cancels TLT rank-2 (bond-bull) to produce a neutral signal.
        Falls back to the full average if the proxy asset's HMM fit fails.
    momentum_overlay : bool
        Phase 57 (legacy variant): when True, apply cross-sectional momentum
        adjustment to each asset's effective λ.  Assets with positive 12m-1m
        momentum receive ``momentum_lambda_high`` when HMM also agrees
        (bullish rank); assets with negative momentum receive
        ``momentum_lambda_low``.  The final per-asset λ is blended with the
        portfolio-level λ using the HMM last-bar posterior confidence as the
        blend weight.
    momentum_lookback : int
        Bars in the momentum lookback window (default 252 ≈ 12 months daily).
    momentum_skip : int
        Bars to skip at the recent end to avoid short-term reversal
        (default 21 ≈ 1 month daily).
    momentum_lambda_high : float
        λ to apply to assets with positive momentum + bullish HMM (default
        0.15).  Must be in [0, 1].
    momentum_lambda_low : float
        λ to apply to assets with negative momentum + bearish HMM (default
        0.01).  Must be in [0, 1].
    momentum_tilt_scale : float
        Phase 57 (additive z-score variant): scale for the additive
        cross-sectional momentum tilt.  When > 0, the 12m-1m momentum signal
        is normalised to a cross-sectional z-score and added to the combined
        RTMV weight vector::

            w_final = w_combined + momentum_tilt_scale * z_momentum

        The result is clipped to [0, ∞) and renormalised to sum to 1.
        Default 0.0 disables this path entirely (backward-compatible).
        This parameter is independent of ``momentum_overlay``; do not set
        both > 0 simultaneously.

    Returns
    -------
    dict[str, float]
        Target weight per asset, values sum to 1.0.
        Falls back to equal-weight if insufficient data.
    tuple[dict[str, float], dict[str, np.ndarray]]
        When ``return_posteriors=True``.  Second element maps asset name to
        a ``(n_states,)`` array — the last-bar filtered posterior.
    """
    cfg = config or MultiAssetConfig()
    tkwargs = train_kwargs or {}
    assets = list(asset_returns_window.columns)
    N = len(assets)

    if N < 2:
        raise ValueError("compute_rtmv_weights_now requires at least 2 assets.")
    if not (0.0 <= lambda_tilt <= 1.0):
        raise ValueError("lambda_tilt must be in [0, 1].")
    if adaptive_lambda:
        if not (0.0 <= lambda_min <= lambda_max <= 1.0):
            raise ValueError("Require 0 <= lambda_min <= lambda_max <= 1.")
    if lambda_by_state_rank is not None:
        if len(lambda_by_state_rank) != (config or MultiAssetConfig()).n_states:
            raise ValueError(
                f"lambda_by_state_rank length ({len(lambda_by_state_rank)}) "
                f"must equal n_states ({(config or MultiAssetConfig()).n_states})."
            )
        if not all(0.0 <= v <= 1.0 for v in lambda_by_state_rank):
            raise ValueError("All lambda_by_state_rank values must be in [0, 1].")

    ret_arr = asset_returns_window.values.astype(float)
    uniform_post = np.ones(cfg.n_states) / cfg.n_states
    if ret_arr.shape[0] < max(4, N + 1):
        logger.warning("Insufficient data (%d bars) — returning equal weights.", ret_arr.shape[0])
        weights = {a: 1.0 / N for a in assets}
        if return_posteriors and return_models:
            return weights, {a: uniform_post.copy() for a in assets}, {}
        if return_posteriors:
            return weights, {a: uniform_post.copy() for a in assets}
        if return_models:
            return weights, {}
        return weights

    # Joint covariance from the tail of the window.
    cov_window = ret_arr[-cfg.cov_window_bars:] if len(ret_arr) > cfg.cov_window_bars else ret_arr
    cov_joint = np.cov(cov_window.T, ddof=1)
    if cov_joint.ndim == 0:
        cov_joint = np.array([[float(cov_joint)]])
    cov_joint += np.eye(N) * cfg.regularisation

    w_minvar = _min_variance_weights(cov_joint, cfg.min_weight, cfg.max_weight)

    # Posterior-weighted expected return per asset; collect last-bar posteriors
    # so we can derive an adaptive λ from their average entropy and (optionally)
    # return the per-asset posterior for KL-triggered rebalancing.
    exp_returns = np.zeros(N)
    last_posteriors: list[np.ndarray] = []
    posteriors_per_asset: dict[str, np.ndarray] = {a: uniform_post.copy() for a in assets}
    fitted_models_per_asset: dict[str, object] = {}  # FittedModel per asset (Phase 53)
    dominant_state_ranks: list[int] = []  # per-asset dominant state return rank
    dominant_rank_per_asset: dict[str, int] = {}  # Phase 57: per-asset rank for momentum overlay
    proxy_asset_rank: int | None = None   # rank for the proxy asset (Phase 51b)
    for i, asset in enumerate(assets):
        feat_df = asset_features_window.get(asset)
        if feat_df is None or len(feat_df) < cfg.n_states * 2:
            continue
        feat_arr = feat_df.values.astype(float)
        feat_cols = list(feat_df.columns)
        lr_idx = (
            feat_cols.index(cfg.log_return_col)
            if cfg.log_return_col in feat_cols
            else 0
        )
        try:
            fitted = train_hmm(
                feat_arr,
                n_states=cfg.n_states,
                n_restarts=cfg.n_restarts,
                **tkwargs,
            )
            fitted_models_per_asset[asset] = fitted
            decoder = OnlineDecoder(fitted)
            posteriors = decoder.batch_filter(feat_arr)  # (T, K)
            last_posteriors.append(posteriors[-1])
            posteriors_per_asset[asset] = posteriors[-1].copy()
            means_scaled = decoder._model.means_
            means_orig = decoder._scaler.inverse_transform(means_scaled)
            exp_returns[i] = float(posteriors[-1] @ means_orig[:, lr_idx])

            # Regime-conditional λ: record the return rank of the dominant state.
            state_means = means_orig[:, lr_idx]  # (K,) mean log-return per state
            dominant_state = int(np.argmax(posteriors[-1]))
            rank = int(np.argsort(state_means).tolist().index(dominant_state))
            dominant_rank_per_asset[asset] = rank  # Phase 57: always track per-asset rank
            if lambda_by_state_rank is not None:
                dominant_state_ranks.append(rank)
                if lambda_proxy_asset is not None and asset == lambda_proxy_asset:
                    proxy_asset_rank = rank
        except Exception as exc:
            logger.warning("HMM fit failed for %s: %s", asset, exc)

    # Determine effective lambda.
    if lambda_by_state_rank is not None and dominant_state_ranks:
        if lambda_proxy_asset is not None and proxy_asset_rank is not None:
            # Phase 51b: proxy-asset rank only — avoids cross-asset averaging.
            portfolio_rank = max(0, min(proxy_asset_rank, len(lambda_by_state_rank) - 1))
        else:
            # Phase 51 original: mean across all assets (fallback when proxy fails).
            mean_rank = float(np.mean(dominant_state_ranks))
            portfolio_rank = int(round(mean_rank))
            portfolio_rank = max(0, min(portfolio_rank, len(lambda_by_state_rank) - 1))
        lambda_effective = lambda_by_state_rank[portfolio_rank]
    elif adaptive_lambda and last_posteriors:
        H_max = float(np.log(cfg.n_states)) if cfg.n_states > 1 else 1.0
        entropies = []
        for p in last_posteriors:
            p_safe = np.clip(p, 1e-15, 1.0)
            entropies.append(float(-np.sum(p_safe * np.log(p_safe))))
        h_norm = float(np.mean(entropies)) / (H_max + 1e-15)
        h_norm = float(np.clip(h_norm, 0.0, 1.0))
        lambda_effective = lambda_max * (1.0 - h_norm) + lambda_min * h_norm
    else:
        lambda_effective = lambda_tilt

    e_r_pos = np.maximum(exp_returns, 0.0)
    if e_r_pos.sum() > 1e-15:
        w_regime = e_r_pos / e_r_pos.sum()
    else:
        w_regime = np.ones(N) / N

    # Phase 57: momentum overlay — per-asset lambda adjustment.
    # When momentum_overlay=True, each asset's contribution to the tilt is
    # scaled by a per-asset lambda that depends on whether the 12m-1m momentum
    # signal agrees with the HMM dominant-state rank.  The blended portfolio
    # weight is then a convex combination of w_minvar and the momentum-adjusted
    # regime weights.
    if momentum_overlay and len(asset_returns_window) >= momentum_lookback + 1:
        mom_scores = compute_momentum_scores(
            asset_returns_window,
            momentum_lookback=momentum_lookback,
            momentum_skip=momentum_skip,
        )
        n_states_cfg = cfg.n_states
        bullish_rank = n_states_cfg - 1  # highest-return state rank = bullish
        bearish_rank = 0                 # lowest-return state rank = bearish

        # Build per-asset lambda based on momentum + HMM agreement.
        lambda_per_asset = np.full(N, lambda_effective)
        for i, asset in enumerate(assets):
            mom = mom_scores.get(asset, 0.0)
            hmm_rank = dominant_rank_per_asset.get(asset, -1)
            if mom > 0 and hmm_rank == bullish_rank:
                lambda_per_asset[i] = momentum_lambda_high
            elif mom < 0 and hmm_rank == bearish_rank:
                lambda_per_asset[i] = momentum_lambda_low
            # else: keep portfolio-level lambda_effective

        # Compute per-asset blended regime component then sum.
        # w_i = (1 - lambda_i) * w_minvar_i + lambda_i * w_regime_i
        # This extends the scalar blending to a per-asset schedule.
        w_raw = (1.0 - lambda_per_asset) * w_minvar + lambda_per_asset * w_regime
        logger.debug(
            "Momentum overlay: scores=%s  hmm_ranks=%s  lambda_per_asset=%s",
            {a: f"{mom_scores.get(a, 0):.3f}" for a in assets},
            dominant_rank_per_asset,
            {a: f"{lam:.3f}" for a, lam in zip(assets, lambda_per_asset)},
        )
    else:
        w_raw = (1.0 - lambda_effective) * w_minvar + lambda_effective * w_regime

    w_raw = np.clip(w_raw, cfg.min_weight, cfg.max_weight)
    total = w_raw.sum()
    if total > 1e-15:
        w_raw /= total

    # Phase 57 additive z-score momentum tilt.
    # Applied after the RTMV blending so momentum shifts allocations
    # without interacting with the regime λ schedule.
    if momentum_tilt_scale > 0.0 and len(asset_returns_window) >= 2:
        n_12m = min(252, len(asset_returns_window))
        n_1m = min(21, len(asset_returns_window))
        mom_12m = asset_returns_window.iloc[-n_12m:].sum()
        mom_1m = asset_returns_window.iloc[-n_1m:].sum()
        mom_signal = mom_12m - mom_1m
        mu_mom = float(mom_signal.mean())
        sigma_mom = float(mom_signal.std())
        if sigma_mom > 1e-12:
            z_mom = (mom_signal - mu_mom) / sigma_mom
        else:
            z_mom = pd.Series(0.0, index=mom_signal.index)
        w_final = w_raw + momentum_tilt_scale * z_mom.values
        w_final = np.clip(w_final, 0.0, None)
        total_final = w_final.sum()
        if total_final > 1e-15:
            w_final /= total_final
        else:
            w_final = w_raw  # fallback: revert to pre-momentum weights
        logger.debug(
            "Momentum z-score tilt (scale=%.3f): z=%s  w_pre=%s  w_post=%s",
            momentum_tilt_scale,
            {a: f"{z:.3f}" for a, z in zip(assets, z_mom.values)},
            {a: f"{w:.3f}" for a, w in zip(assets, w_raw)},
            {a: f"{w:.3f}" for a, w in zip(assets, w_final)},
        )
        w_raw = w_final

    weights = {a: float(w) for a, w in zip(assets, w_raw)}
    if return_posteriors and return_models:
        return weights, posteriors_per_asset, fitted_models_per_asset
    if return_posteriors:
        return weights, posteriors_per_asset
    if return_models:
        return weights, fitted_models_per_asset
    return weights


# ---------------------------------------------------------------------------
# Phase 54: Joint 5D HMM — single model on the stacked return vector
# ---------------------------------------------------------------------------


def compute_rtmv_weights_now_joint(
    asset_returns_window: pd.DataFrame,
    config: MultiAssetConfig | None = None,
    lambda_tilt: float = 0.05,
    train_kwargs: dict | None = None,
    lambda_by_state_rank: list[float] | None = None,
    spy_asset: str = "SPY",
) -> dict[str, float]:
    """Compute RTMV weights using a single joint HMM on the stacked return vector.

    Replaces the 5 independent per-asset HMMs with a single
    ``GaussianHMM(n_features=N, n_states=k, covariance_type="diag")`` fitted
    on the full ``(T, N)`` return matrix.  The dominant state from the joint
    model is used to select ``lambda_tilt`` via ``lambda_by_state_rank``,
    ranked by the joint state's mean SPY (or first-asset) return.

    The regime-tilt weight vector is derived from the posterior-weighted
    expected return per asset (from joint model state means for each feature
    dimension).

    Parameters
    ----------
    asset_returns_window : pd.DataFrame
        Shape ``(W, N)``.  Recent log returns, one column per asset.
        ``W`` should be at least ``config.cov_window_bars``.
    config : MultiAssetConfig, optional
        Uses ``cov_window_bars``, ``n_states``, ``n_restarts``,
        ``min_weight``, ``max_weight``, ``regularisation``.
    lambda_tilt : float
        Blending weight in [0, 1].  Overridden by ``lambda_by_state_rank``
        when provided and non-empty.
    train_kwargs : dict, optional
        Forwarded to :func:`rde.models.hmm.train_hmm`.  ``covariance_type``
        defaults to ``"diag"`` for the joint model unless overridden here.
    lambda_by_state_rank : list[float], optional
        Length must equal ``config.n_states``.  Element ``k`` is the lambda
        when the joint model's dominant state has SPY-return rank ``k``
        (0 = lowest mean return, n_states-1 = highest).
    spy_asset : str
        Column name of the proxy asset used to rank states.  Defaults to
        ``"SPY"``; falls back to the first column if not present.

    Returns
    -------
    dict[str, float]
        Target weight per asset, values sum to 1.0.
        Falls back to equal-weight on insufficient data or HMM failure.

    Notes
    -----
    covariance_type="diag" is the correct default for a 5-feature joint HMM
    on ~5000 daily bars.  A full 5x5 covariance per state requires
    5*6/2=15 extra parameters per state and can be rank-deficient on
    concentrated bond periods.
    """
    cfg = config or MultiAssetConfig()
    tkwargs = {"covariance_type": "diag"}
    tkwargs.update(train_kwargs or {})
    assets = list(asset_returns_window.columns)
    N = len(assets)

    if N < 2:
        raise ValueError("compute_rtmv_weights_now_joint requires at least 2 assets.")

    ret_arr = asset_returns_window.values.astype(float)
    T = ret_arr.shape[0]

    # Equal-weight fallback if insufficient data.
    if T < max(4, N + 1):
        logger.warning(
            "Joint HMM: insufficient data (%d bars) — equal weights.", T
        )
        return {a: 1.0 / N for a in assets}

    # Joint covariance from the tail of the window.
    cov_window = ret_arr[-cfg.cov_window_bars:] if T > cfg.cov_window_bars else ret_arr
    cov_joint = np.cov(cov_window.T, ddof=1)
    if cov_joint.ndim == 0:
        cov_joint = np.array([[float(cov_joint)]])
    cov_joint += np.eye(N) * cfg.regularisation

    w_minvar = _min_variance_weights(cov_joint, cfg.min_weight, cfg.max_weight)

    # Identify the SPY (proxy) column index for ranking states.
    spy_idx = assets.index(spy_asset) if spy_asset in assets else 0

    # Fit a single joint HMM on the full N-dimensional return matrix.
    try:
        fitted = train_hmm(
            ret_arr,
            n_states=cfg.n_states,
            n_restarts=cfg.n_restarts,
            feature_names=assets,
            **tkwargs,
        )
    except Exception as exc:
        logger.warning("Joint HMM fit failed: %s — falling back to min-var.", exc)
        weights = w_minvar / (w_minvar.sum() + 1e-15)
        return {a: float(w) for a, w in zip(assets, weights)}

    decoder = OnlineDecoder(fitted)
    posteriors = decoder.batch_filter(ret_arr)  # (T, K)
    last_posterior = posteriors[-1]  # (K,)

    # Recover per-asset expected returns from joint state means.
    means_scaled = decoder._model.means_  # (K, N)
    means_orig = decoder._scaler.inverse_transform(means_scaled)  # (K, N)

    # E[r_i] = sum_k p_k * mu_k[i]
    exp_returns = last_posterior @ means_orig  # (N,)

    # Determine effective lambda from state rank if schedule provided.
    if lambda_by_state_rank and len(lambda_by_state_rank) == cfg.n_states:
        spy_means = means_orig[:, spy_idx]  # mean SPY return per state (K,)
        dominant_state = int(np.argmax(last_posterior))
        rank = int(np.argsort(spy_means).tolist().index(dominant_state))
        rank = max(0, min(rank, len(lambda_by_state_rank) - 1))
        lambda_effective = lambda_by_state_rank[rank]
        logger.debug(
            "Joint HMM: dominant_state=%d rank=%d lambda=%.3f post=%s",
            dominant_state, rank, lambda_effective,
            [f"{p:.3f}" for p in last_posterior],
        )
    else:
        lambda_effective = lambda_tilt

    # Regime weights: proportional to positive expected return.
    e_r_pos = np.maximum(exp_returns, 0.0)
    if e_r_pos.sum() > 1e-15:
        w_regime = e_r_pos / e_r_pos.sum()
    else:
        w_regime = np.ones(N) / N

    w_raw = (1.0 - lambda_effective) * w_minvar + lambda_effective * w_regime
    w_raw = np.clip(w_raw, cfg.min_weight, cfg.max_weight)
    total = w_raw.sum()
    if total > 1e-15:
        w_raw /= total

    return {a: float(w) for a, w in zip(assets, w_raw)}
