"""Vol-target overlay strategy (Phase 37b Track B).

Reframes regime classification as a volatility-targeting exposure overlay on a
passive long position, rather than a directional long/flat/short signal.

**Rationale:** The 1.2 diagnostic confirmed that BTC's regime structure is
stable (inter-half ARI = 0.74). The problem is not that the model is wrong —
it reliably identifies risk-on/risk-off regimes. The problem is that using
regime labels as directional signals (long in bull, flat otherwise) produces
too many trades and too little edge margin to survive execution costs.

The overlay approach:
- Base position: 100 % long BTC at all times.
- Exposure multiplier: when the forward-filtering posterior is concentrated
  in a single regime (max_posterior > HIGH_CONF_THRESHOLD), full exposure (1.0).
  When the posterior is diffuse (max_posterior < LOW_CONF_THRESHOLD), exposure
  scales down to MIN_EXPOSURE. Linearly interpolated in between.
- Optional vol-regime penalty: when the posterior-weighted regime vol exceeds
  a threshold, reduce exposure independently of confidence.
- Signal is evaluated daily (24-bar aggregation of posteriors) to reduce
  trade frequency to << 20 per year.

This produces a low-turnover signal — typical hold periods of days to weeks —
that clears the ≥ 5 bps cost break-even required for Trading212 deployment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rde.analysis.backtest import BacktestResult, backtest_tearsheet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class OverlayConfig:
    """Configuration for the vol-target overlay.

    Attributes
    ----------
    high_conf_threshold : float
        Max posterior above which exposure = 1.0 (default 0.7).
    low_conf_threshold : float
        Max posterior below which exposure = min_exposure (default 0.5).
    min_exposure : float
        Minimum exposure fraction when confidence is lowest (default 0.3).
    vol_threshold_ann : float | None
        If set, reduce exposure by (vol_threshold_ann / posterior_weighted_vol)
        when posterior-weighted annualised vol exceeds this level (default None).
    rebalance_bars : int
        Number of bars between exposure recalculations. Controls turnover.
        Default 24 (daily for hourly data).
    transaction_cost : float
        One-way cost per unit of position change (default 0.0001 = 1 bp).
    ann_factor : int
        Bars per year for annualisation (default 8760 for hourly crypto).
    """

    high_conf_threshold: float = 0.70
    low_conf_threshold: float = 0.50
    min_exposure: float = 0.30
    vol_threshold_ann: float | None = None
    rebalance_bars: int = 24
    transaction_cost: float = 0.0001
    ann_factor: int = 8760


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class OverlayBacktestResult:
    """Output of :func:`run_overlay_backtest`.

    Attributes
    ----------
    equity : pd.Series
    returns : pd.Series
    positions : pd.Series
        Fractional exposure at each bar.
    confidence : pd.Series
        Max posterior (regime confidence) at each bar.
    n_rebalances : int
        Number of times the position actually changed.
    passive_equity : pd.Series
        Buy-and-hold equity curve (always 1x long), for comparison.
    """

    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    confidence: pd.Series
    n_rebalances: int
    passive_equity: pd.Series


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def _posterior_to_exposure(
    max_posterior: float,
    cfg: OverlayConfig,
    posterior_weighted_vol_ann: float | None = None,
) -> float:
    """Map max posterior → target exposure in [min_exposure, 1.0].

    1. Confidence interpolation: linear between (low_conf_threshold, min_exposure)
       and (high_conf_threshold, 1.0).
    2. Optional vol cap: if posterior_weighted_vol_ann > vol_threshold_ann,
       cap exposure at vol_threshold_ann / posterior_weighted_vol_ann.
    """
    # Confidence-based exposure
    lo, hi = cfg.low_conf_threshold, cfg.high_conf_threshold
    rng = hi - lo
    if rng <= 0:
        conf_exposure = 1.0 if max_posterior >= hi else cfg.min_exposure
    else:
        t = (max_posterior - lo) / rng
        t = float(np.clip(t, 0.0, 1.0))
        conf_exposure = cfg.min_exposure + t * (1.0 - cfg.min_exposure)

    if (
        cfg.vol_threshold_ann is not None
        and posterior_weighted_vol_ann is not None
        and posterior_weighted_vol_ann > cfg.vol_threshold_ann
    ):
        vol_cap = cfg.vol_threshold_ann / max(posterior_weighted_vol_ann, 1e-9)
        conf_exposure = min(conf_exposure, vol_cap)

    return float(np.clip(conf_exposure, cfg.min_exposure, 1.0))


def compute_overlay_signal(
    posteriors: np.ndarray,
    cfg: OverlayConfig,
    regime_ann_vols: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-bar target exposure from filtered posteriors.

    Parameters
    ----------
    posteriors : np.ndarray
        Shape (T, K) — forward-filtered state posteriors from OnlineDecoder.
    cfg : OverlayConfig
    regime_ann_vols : np.ndarray | None
        Shape (K,) — annualised realised vol per regime (from training data).
        Required only when ``cfg.vol_threshold_ann`` is set.

    Returns
    -------
    np.ndarray
        Shape (T,) — target exposure at each bar, in [min_exposure, 1.0].
    """
    T, K = posteriors.shape
    max_post = posteriors.max(axis=1)  # (T,)

    if cfg.vol_threshold_ann is not None and regime_ann_vols is not None:
        pw_vol = (posteriors * regime_ann_vols[np.newaxis, :]).sum(axis=1)
    else:
        pw_vol = None

    exposures = np.zeros(T)
    for t in range(T):
        pv = float(pw_vol[t]) if pw_vol is not None else None
        exposures[t] = _posterior_to_exposure(float(max_post[t]), cfg, pv)

    return exposures


def run_overlay_backtest(
    returns: pd.Series,
    posteriors: np.ndarray,
    cfg: OverlayConfig | None = None,
    regime_ann_vols: np.ndarray | None = None,
) -> OverlayBacktestResult:
    """Backtest the vol-target overlay on a return series.

    The position is recalculated every ``cfg.rebalance_bars`` bars by averaging
    posteriors over the look-back window (causal — no future leakage).
    Transaction costs are incurred on every position change.

    Parameters
    ----------
    returns : pd.Series
        Per-bar asset returns, length T.
    posteriors : np.ndarray
        Shape (T, K) — causal filtered posteriors (from OnlineDecoder.batch_filter).
    cfg : OverlayConfig | None
    regime_ann_vols : np.ndarray | None

    Returns
    -------
    OverlayBacktestResult
    """
    cfg = cfg or OverlayConfig()
    r = returns.values
    T = len(r)
    idx = returns.index

    # Compute bar-level exposure signal (no rebalance smoothing yet)
    raw_exposure = compute_overlay_signal(posteriors, cfg, regime_ann_vols)

    # Apply rebalance_bars: position only changes at multiples of rebalance_bars
    positions = np.zeros(T)
    cur_pos = 1.0  # start fully invested
    for t in range(T):
        if t % cfg.rebalance_bars == 0:
            # Average posteriors over look-back window [t-rebalance_bars, t]
            lb = max(0, t - cfg.rebalance_bars + 1)
            avg_post = posteriors[lb: t + 1].mean(axis=0)
            max_post = float(avg_post.max())
            if cfg.vol_threshold_ann is not None and regime_ann_vols is not None:
                pw_vol = float((avg_post * regime_ann_vols).sum())
            else:
                pw_vol = None
            cur_pos = _posterior_to_exposure(max_post, cfg, pw_vol)
        positions[t] = cur_pos

    # Simulate PnL
    equity = np.ones(T + 1)
    strat_returns = np.zeros(T)
    n_rebalances = 0
    prev_pos = 0.0

    for t in range(T):
        target = positions[t]
        delta = abs(target - prev_pos)
        cost = delta * cfg.transaction_cost
        if delta > 1e-8:
            n_rebalances += 1
        bar_ret = prev_pos * r[t] - cost
        strat_returns[t] = bar_ret
        equity[t + 1] = equity[t] * (1.0 + bar_ret)
        prev_pos = target

    # Passive B&H equity (1x long, no cost after bar 0)
    passive_eq = np.ones(T + 1)
    for t in range(T):
        passive_eq[t + 1] = passive_eq[t] * (1.0 + r[t])

    equity_s = pd.Series(equity[1:], index=idx)
    passive_s = pd.Series(passive_eq[1:], index=idx)

    logger.info(
        "Overlay backtest: n_rebalances=%d, trades_per_year≈%.1f",
        n_rebalances,
        n_rebalances * cfg.ann_factor / max(T, 1),
    )

    return OverlayBacktestResult(
        equity=equity_s,
        returns=pd.Series(strat_returns, index=idx),
        positions=pd.Series(positions, index=idx),
        confidence=pd.Series(posteriors.max(axis=1), index=idx),
        n_rebalances=n_rebalances,
        passive_equity=passive_s,
    )


def overlay_tearsheet(
    result: OverlayBacktestResult,
    ann_factor: int = 8760,
) -> pd.Series:
    """Compute overlay strategy metrics plus comparison against passive BTC.

    Returns
    -------
    pd.Series
        Includes all standard backtest metrics plus overlay-specific ones:
        passive_sharpe, sharpe_improvement, trades_per_year.
    """
    from rde.analysis.backtest import BacktestResult, backtest_tearsheet

    bt = BacktestResult(
        equity=result.equity,
        returns=result.returns,
        positions=result.positions,
        trades=pd.DataFrame(columns=["bar", "direction", "size", "cost"]),
        regime_labels=np.zeros(len(result.returns), dtype=int),
    )
    ts = backtest_tearsheet(bt, ann_factor=ann_factor)

    # Passive metrics
    passive_r = result.passive_equity.pct_change().fillna(0).values
    passive_sharpe = float(
        passive_r.mean() / (passive_r.std(ddof=1) + 1e-15) * np.sqrt(ann_factor)
    )
    T = len(result.returns)
    trades_per_year = result.n_rebalances * ann_factor / max(T, 1)

    extra = pd.Series({
        "passive_sharpe": passive_sharpe,
        "sharpe_improvement": float(ts["sharpe"]) - passive_sharpe,
        "trades_per_year": trades_per_year,
        "mean_exposure": float(result.positions.mean()),
        "min_exposure": float(result.positions.min()),
    })
    return pd.concat([ts, extra])
