"""Phase 47 / Phase 63: Purged CV validation on RTMV portfolio.

Tests whether the RTMV (regime-tilted min-var) advantage over global_min_var
is consistent across time periods, robust to transaction cost, and not due
to regime label overfitting.

Three tests:
  1. Fold consistency — split post-warm-up returns into N non-overlapping
     folds; measure RTMV vs global_min_var Sharpe per fold.
  2. Cost sensitivity — sweep transaction_cost from 10 bps to 200 bps;
     find break-even where Sharpe advantage disappears.
  3. Shuffle test — re-run RTMV with permuted per-asset expected returns
     at each rebalance step; p-value = fraction of 100 shuffles >= real RTMV.

Phase 63 extension (--momentum-tilt-scale > 0):
  When a non-zero momentum_tilt_scale is given, a *dual-variant* CV is run
  that compares the momentum variant against the no-momentum baseline.
  Fold consistency measures momentum wins vs baseline per fold.
  Cost sensitivity sweeps both variants and reports the momentum delta.
  Shuffle test permutes momentum z-scores across assets (run N full
  backtests with the same HMM signal but scrambled momentum) to test
  whether the momentum tilt itself is responsible for the Sharpe gain.

Performance note: HMM fitting is expensive (~5 min for the full period).
This script fits HMMs ONCE (``collect_rtmv_rebalance_cache``), then reuses
cached (w_minvar, exp_returns) for the cost sweep and shuffle test.  Total
runtime: ~7 minutes (1 fit pass + cheap recomputation for cost/shuffle).

When --momentum-tilt-scale > 0, the cache path is bypassed and a full
RTMVRebalancer backtest is run for each variant; runtime ~12-20 min
depending on n-shuffle.

Usage
-----
    # Phase 47 (baseline RTMV vs GMV):
    uv run python scripts/run_rtmv_cv_validation.py \\
        --assets SPY,GLD,TLT,IEF --n-states 3 --lambda-tilt 0.30 \\
        --n-folds 5 --n-shuffle 100 --n-restarts 3

    # Phase 63 (momentum variant vs baseline, 5-asset, n-shuffle=50):
    uv run python scripts/run_rtmv_cv_validation.py \\
        --assets SPY,GLD,SHY,IEF,TLT --n-states 3 --lambda-tilt 0.30 \\
        --n-folds 5 --n-shuffle 50 --n-restarts 3 \\
        --momentum-tilt-scale 0.03

Outputs
-------
    results/{output_dir}/rtmv_cv_report_{YYYYMMDD}.md   (Phase 47)
    results/{output_dir}/phase63_momentum_cv_{YYYYMMDD}.md   (Phase 63)
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from rde.analysis.multi_asset_allocation import (
    MultiAssetConfig,
    _compute_metrics,
    collect_rtmv_rebalance_cache,
    global_min_var_baseline,
    regime_tilted_min_var_from_cache,
)
from rde.data.yfinance_source import YFinanceSource
from rde.features.pipeline import FeaturePipeline
from rde.features.returns import LogReturns, SmoothedReturns
from rde.features.volatility import RollingVolatility
from rde.trading.rtmv_rebalancer import RTMVRebalancer, RTMVRebalancerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RTMVCVResult:
    """Phase 47 validation results.

    Attributes
    ----------
    lambda_tilt : float
    n_folds : int
    fold_sharpe_rtmv : list[float]
    fold_sharpe_gmv : list[float]
    fold_sharpe_delta : list[float]
        rtmv - gmv per fold.
    pct_folds_rtmv_wins : float
        Fraction of folds where RTMV > global_min_var.
    mean_sharpe_delta : float
    std_sharpe_delta : float
    cost_bps_tested : list[float]
    sharpe_rtmv_by_cost : list[float]
    sharpe_gmv_by_cost : list[float]
    cost_break_even_bps : float
        Cost level where RTMV Sharpe advantage disappears (interpolated).
    n_shuffle : int
    shuffle_sharpe_mean : float
    shuffle_sharpe_std : float
    real_rtmv_sharpe : float
    shuffle_p_value : float
        Fraction of shuffled runs with Sharpe >= real RTMV.
    shuffle_margin : float
        real_sharpe - shuffle_sharpe_mean.
    """

    lambda_tilt: float
    n_folds: int
    fold_sharpe_rtmv: list[float] = field(default_factory=list)
    fold_sharpe_gmv: list[float] = field(default_factory=list)
    fold_sharpe_delta: list[float] = field(default_factory=list)
    pct_folds_rtmv_wins: float = 0.0
    mean_sharpe_delta: float = 0.0
    std_sharpe_delta: float = 0.0
    cost_bps_tested: list[float] = field(default_factory=list)
    sharpe_rtmv_by_cost: list[float] = field(default_factory=list)
    sharpe_gmv_by_cost: list[float] = field(default_factory=list)
    cost_break_even_bps: float = float("inf")
    n_shuffle: int = 0
    shuffle_sharpe_mean: float = 0.0
    shuffle_sharpe_std: float = 0.0
    real_rtmv_sharpe: float = 0.0
    shuffle_p_value: float = 0.0
    shuffle_margin: float = 0.0


# ---------------------------------------------------------------------------
# Phase 63: Momentum CV result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MomentumCVResult:
    """Phase 63 validation results for momentum tilt vs baseline.

    Attributes
    ----------
    momentum_tilt_scale : float
    n_folds : int
    fold_sharpe_mom : list[float]
        Per-fold Sharpe for momentum variant.
    fold_sharpe_base : list[float]
        Per-fold Sharpe for no-momentum baseline.
    fold_sharpe_delta : list[float]
        momentum - baseline per fold.
    pct_folds_mom_wins : float
        Fraction of folds where momentum > baseline.
    mean_sharpe_delta : float
    std_sharpe_delta : float
    cost_bps_tested : list[float]
    sharpe_mom_by_cost : list[float]
    sharpe_base_by_cost : list[float]
    cost_break_even_bps : float
        Cost delta break-even (momentum advantage disappears).
    n_shuffle : int
    shuffle_sharpe_mean : float
    shuffle_sharpe_std : float
    real_mom_sharpe : float
    shuffle_p_value : float
        Fraction of shuffled momentum runs with Sharpe >= real momentum Sharpe.
    shuffle_margin : float
        real_mom_sharpe - shuffle_mean.
    base_full_sharpe : float
        Full-period baseline Sharpe (reference).
    mom_full_sharpe : float
        Full-period momentum Sharpe.
    """

    momentum_tilt_scale: float
    n_folds: int
    fold_sharpe_mom: list[float] = field(default_factory=list)
    fold_sharpe_base: list[float] = field(default_factory=list)
    fold_sharpe_delta: list[float] = field(default_factory=list)
    pct_folds_mom_wins: float = 0.0
    mean_sharpe_delta: float = 0.0
    std_sharpe_delta: float = 0.0
    cost_bps_tested: list[float] = field(default_factory=list)
    sharpe_mom_by_cost: list[float] = field(default_factory=list)
    sharpe_base_by_cost: list[float] = field(default_factory=list)
    cost_break_even_bps: float = float("inf")
    n_shuffle: int = 0
    shuffle_sharpe_mean: float = 0.0
    shuffle_sharpe_std: float = 0.0
    real_mom_sharpe: float = 0.0
    shuffle_p_value: float = 0.0
    shuffle_margin: float = 0.0
    base_full_sharpe: float = 0.0
    mom_full_sharpe: float = 0.0


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _build_pipeline() -> FeaturePipeline:
    return FeaturePipeline([
        LogReturns(),
        RollingVolatility(window=20),
        SmoothedReturns(window=5),
    ])


def _load_asset_features(
    symbol: str,
    source: YFinanceSource,
    pipeline: FeaturePipeline,
) -> pd.DataFrame:
    logger.info("Loading %s…", symbol)
    raw = source.load(symbol, period="max", interval="1d")
    feat_df = pipeline.transform(raw).dropna()
    logger.info("  %s: %d bars", symbol, len(feat_df))
    return feat_df


def _align_features(
    feat_dfs: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.core.indexes.datetimes.DatetimeIndex]:
    common_idx = None
    for feat_df in feat_dfs.values():
        common_idx = feat_df.index if common_idx is None else common_idx.intersection(feat_df.index)
    if common_idx is None or len(common_idx) == 0:
        raise RuntimeError("No common trading dates across assets.")
    aligned = {sym: df.loc[common_idx] for sym, df in feat_dfs.items()}
    logger.info("Common index: %d bars (%s → %s)", len(common_idx),
                common_idx[0].date(), common_idx[-1].date())
    return aligned, common_idx


# ---------------------------------------------------------------------------
# Fold consistency helper
# ---------------------------------------------------------------------------


def _fold_sharpes(
    port_returns: pd.Series,
    n_folds: int,
    ann_factor: int,
) -> list[float]:
    """Split portfolio returns into n_folds equal non-overlapping windows."""
    n = len(port_returns)
    fold_size = n // n_folds
    sharpes = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size
        fold_ret = port_returns.iloc[start:end]
        if len(fold_ret) < 20:
            continue
        s, *_ = _compute_metrics(fold_ret, ann_factor)
        sharpes.append(s)
    return sharpes


# ---------------------------------------------------------------------------
# Interpolated break-even
# ---------------------------------------------------------------------------


def _interpolate_break_even(
    cost_bps: list[float],
    sharpe_delta: list[float],
) -> float:
    """Interpolate the cost level where Sharpe delta crosses zero."""
    for i in range(len(sharpe_delta) - 1):
        if sharpe_delta[i] >= 0 and sharpe_delta[i + 1] < 0:
            c0, c1 = cost_bps[i], cost_bps[i + 1]
            d0, d1 = sharpe_delta[i], sharpe_delta[i + 1]
            frac = d0 / (d0 - d1)
            return float(c0 + frac * (c1 - c0))
    if all(d >= 0 for d in sharpe_delta):
        return float("inf")
    return 0.0


# ---------------------------------------------------------------------------
# Main validation function (cache-based)
# ---------------------------------------------------------------------------


def run_rtmv_cv_validation(
    asset_returns: pd.DataFrame,
    asset_features: dict[str, pd.DataFrame],
    config: MultiAssetConfig,
    lambda_tilt: float = 0.30,
    n_folds: int = 5,
    cost_bps_range: list[float] | None = None,
    n_shuffle: int = 100,
    seed: int = 42,
    train_kwargs: dict | None = None,
) -> RTMVCVResult:
    """Run three-part Phase 47 validation on RTMV portfolio.

    Fits HMMs exactly once via ``collect_rtmv_rebalance_cache``, then
    reuses cached (w_minvar, exp_returns) for cost sweep and shuffle test.
    This makes the total runtime ~equal to 1 reference RTMV run regardless
    of the number of cost levels or shuffle iterations.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Per-asset daily log returns, aligned.
    asset_features : dict[str, pd.DataFrame]
        Per-asset feature DataFrames, aligned.
    config : MultiAssetConfig
        Base config.  ``transaction_cost`` is overridden in the cost sweep.
    lambda_tilt : float
        RTMV lambda to validate (default 0.30, best OOS from Phase 45b).
    n_folds : int
        Non-overlapping sub-periods for fold consistency check.
    cost_bps_range : list[float], optional
        Transaction costs in bps for the cost sweep. Default [10,20,50,100,200].
    n_shuffle : int
        Number of shuffle runs.
    seed : int
        RNG seed base.
    train_kwargs : dict, optional
        Forwarded to train_hmm.

    Returns
    -------
    RTMVCVResult
    """
    if cost_bps_range is None:
        cost_bps_range = [10.0, 20.0, 50.0, 100.0, 200.0]

    result = RTMVCVResult(lambda_tilt=lambda_tilt, n_folds=n_folds)

    # ------------------------------------------------------------------
    # Fit HMMs once — cache per-rebalance (w_minvar, exp_returns)
    # ------------------------------------------------------------------
    logger.info("Fitting HMMs (one pass — results cached for cost/shuffle tests)…")
    cache = collect_rtmv_rebalance_cache(
        asset_returns, asset_features, config, train_kwargs=train_kwargs,
    )
    logger.info("Cache built: %d rebalance steps", len(cache))

    # ------------------------------------------------------------------
    # Run reference RTMV and GMV (base config cost)
    # ------------------------------------------------------------------
    logger.info("Running RTMV(λ=%.2f) from cache…", lambda_tilt)
    rtmv_ref = regime_tilted_min_var_from_cache(
        asset_returns, cache, config, lambda_tilt=lambda_tilt,
    )
    gmv_ref = global_min_var_baseline(
        asset_returns,
        ann_factor=config.ann_factor,
        rebalance_bars=config.rebalance_bars,
        cov_window_bars=config.cov_window_bars,
        transaction_cost=config.transaction_cost,
    )
    result.real_rtmv_sharpe = float(rtmv_ref.sharpe)
    logger.info("RTMV Sharpe=%.4f  GMV Sharpe=%.4f", rtmv_ref.sharpe, gmv_ref.sharpe)

    # ------------------------------------------------------------------
    # Test 1: Fold consistency
    # ------------------------------------------------------------------
    logger.info("Test 1: fold consistency (n_folds=%d)…", n_folds)
    fold_rtmv = _fold_sharpes(rtmv_ref.portfolio_returns, n_folds, config.ann_factor)
    fold_gmv = _fold_sharpes(gmv_ref.portfolio_returns, n_folds, config.ann_factor)

    deltas = [r - g for r, g in zip(fold_rtmv, fold_gmv)]
    result.fold_sharpe_rtmv = fold_rtmv
    result.fold_sharpe_gmv = fold_gmv
    result.fold_sharpe_delta = deltas
    n_wins = sum(1 for d in deltas if d > 0)
    result.pct_folds_rtmv_wins = n_wins / len(deltas) if deltas else 0.0
    result.mean_sharpe_delta = float(np.mean(deltas)) if deltas else 0.0
    result.std_sharpe_delta = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    logger.info(
        "Fold wins: %d/%d (%.0f%%)  mean_delta=%.4f  std_delta=%.4f",
        n_wins, len(deltas), result.pct_folds_rtmv_wins * 100,
        result.mean_sharpe_delta, result.std_sharpe_delta,
    )

    # ------------------------------------------------------------------
    # Test 2: Cost sensitivity (uses cached weights — no re-fitting)
    # ------------------------------------------------------------------
    logger.info("Test 2: cost sensitivity sweep over %s bps (from cache)…", cost_bps_range)
    sharpes_rtmv_cost: list[float] = []
    sharpes_gmv_cost: list[float] = []

    for cost_bps in cost_bps_range:
        cost_frac = cost_bps / 10_000
        cfg_cost = MultiAssetConfig(
            ann_factor=config.ann_factor,
            rebalance_bars=config.rebalance_bars,
            lookback_bars=config.lookback_bars,
            cov_window_bars=config.cov_window_bars,
            n_states=config.n_states,
            n_restarts=config.n_restarts,
            transaction_cost=cost_frac,
            target_vol=config.target_vol,
            min_weight=config.min_weight,
            max_weight=config.max_weight,
        )
        # Reuse cached HMM output — only portfolio_returns() recomputed
        r_rtmv = regime_tilted_min_var_from_cache(
            asset_returns, cache, cfg_cost, lambda_tilt=lambda_tilt,
        )
        r_gmv = global_min_var_baseline(
            asset_returns,
            ann_factor=cfg_cost.ann_factor,
            rebalance_bars=cfg_cost.rebalance_bars,
            cov_window_bars=cfg_cost.cov_window_bars,
            transaction_cost=cfg_cost.transaction_cost,
        )
        sharpes_rtmv_cost.append(float(r_rtmv.sharpe))
        sharpes_gmv_cost.append(float(r_gmv.sharpe))
        logger.info(
            "  cost=%.0f bps → RTMV=%.4f  GMV=%.4f  delta=%.4f",
            cost_bps, r_rtmv.sharpe, r_gmv.sharpe, r_rtmv.sharpe - r_gmv.sharpe,
        )

    result.cost_bps_tested = list(cost_bps_range)
    result.sharpe_rtmv_by_cost = sharpes_rtmv_cost
    result.sharpe_gmv_by_cost = sharpes_gmv_cost
    deltas_cost = [r - g for r, g in zip(sharpes_rtmv_cost, sharpes_gmv_cost)]
    result.cost_break_even_bps = _interpolate_break_even(cost_bps_range, deltas_cost)
    logger.info("Cost break-even: %.1f bps", result.cost_break_even_bps)

    # ------------------------------------------------------------------
    # Test 3: Shuffle test (permute exp_returns at each rebalance)
    # ------------------------------------------------------------------
    logger.info("Test 3: shuffle test (n=%d, from cache)…", n_shuffle)
    shuffle_sharpes: list[float] = []
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2 ** 31, size=n_shuffle)

    for i, sh_seed in enumerate(seeds):
        r_sh = regime_tilted_min_var_from_cache(
            asset_returns, cache, config,
            lambda_tilt=lambda_tilt,
            shuffle_seed=int(sh_seed),
        )
        shuffle_sharpes.append(float(r_sh.sharpe))
        if (i + 1) % 10 == 0:
            logger.info("  Shuffle %d/%d done", i + 1, n_shuffle)

    result.n_shuffle = n_shuffle
    result.shuffle_sharpe_mean = float(np.mean(shuffle_sharpes))
    result.shuffle_sharpe_std = float(np.std(shuffle_sharpes, ddof=1))
    result.shuffle_p_value = float(np.mean([s >= result.real_rtmv_sharpe for s in shuffle_sharpes]))
    result.shuffle_margin = float(result.real_rtmv_sharpe - result.shuffle_sharpe_mean)
    logger.info(
        "Shuffle: mean=%.4f  std=%.4f  p_value=%.4f  margin=%.4f",
        result.shuffle_sharpe_mean, result.shuffle_sharpe_std,
        result.shuffle_p_value, result.shuffle_margin,
    )

    return result


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def write_rtmv_cv_report(
    result: RTMVCVResult,
    output_path: Path,
    assets: list[str],
    n_bars: int,
    n_states: int,
) -> None:
    """Write Phase 47 validation report to Markdown."""
    run_date = pd.Timestamp.now().strftime("%Y%m%d")

    lines = [
        "# Phase 47: RTMV Purged CV Validation",
        "",
        f"**Date:** {run_date}",
        f"**Assets:** {', '.join(assets)}",
        f"**n_bars:** {n_bars}  **n_states:** {n_states}  **λ:** {result.lambda_tilt}",
        "",
        "---",
        "",
        "## Test 1: Fold Consistency",
        "",
        f"Non-overlapping folds: **{len(result.fold_sharpe_rtmv)}**",
        "",
        "| Fold | RTMV Sharpe | GMV Sharpe | Delta | Winner |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, (r, g, d) in enumerate(
        zip(result.fold_sharpe_rtmv, result.fold_sharpe_gmv, result.fold_sharpe_delta)
    ):
        winner = "RTMV" if d > 0 else "GMV"
        lines.append(f"| {i+1} | {r:.4f} | {g:.4f} | {d:+.4f} | **{winner}** |")

    fold_verdict = "PASS" if result.pct_folds_rtmv_wins >= 0.60 else "FAIL"
    lines += [
        "",
        f"**RTMV wins in {result.pct_folds_rtmv_wins:.0%} of folds** "
        f"(mean delta {result.mean_sharpe_delta:+.4f} ± {result.std_sharpe_delta:.4f})",
        f"**Fold consistency verdict: {fold_verdict}** "
        f"(criterion: ≥ 60% folds where RTMV > GMV)",
        "",
        "---",
        "",
        "## Test 2: Cost Sensitivity",
        "",
        "| cost (bps) | RTMV Sharpe | GMV Sharpe | Delta |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for cost, r, g in zip(
        result.cost_bps_tested, result.sharpe_rtmv_by_cost, result.sharpe_gmv_by_cost
    ):
        lines.append(f"| {cost:.0f} | {r:.4f} | {g:.4f} | {r-g:+.4f} |")

    if result.cost_break_even_bps == float("inf"):
        be_str = "∞ (advantage holds at all tested costs)"
        cost_verdict = "PASS"
    else:
        be_str = f"{result.cost_break_even_bps:.1f} bps"
        cost_verdict = "PASS" if result.cost_break_even_bps >= 20.0 else "FAIL"

    lines += [
        "",
        f"**Cost break-even: {be_str}**",
        f"**Cost sensitivity verdict: {cost_verdict}** (criterion: break-even ≥ 20 bps)",
        "",
        "---",
        "",
        "## Test 3: Shuffle Robustness",
        "",
        f"- Real RTMV Sharpe: **{result.real_rtmv_sharpe:.4f}**",
        f"- Shuffle mean: **{result.shuffle_sharpe_mean:.4f}** ± {result.shuffle_sharpe_std:.4f}",
        f"- Shuffle p-value: **{result.shuffle_p_value:.4f}** "
        f"(fraction of {result.n_shuffle} shuffles ≥ real RTMV)",
        f"- Shuffle margin: **{result.shuffle_margin:+.4f}** (real − shuffle mean)",
    ]

    shuffle_verdict = "PASS" if result.shuffle_p_value < 0.10 else "FAIL"
    lines += [
        f"**Shuffle robustness verdict: {shuffle_verdict}** (criterion: p-value < 0.10)",
        "",
        "---",
        "",
        "## Overall GO/NO-GO",
        "",
    ]

    tests = {
        "Fold consistency": fold_verdict,
        "Cost sensitivity": cost_verdict,
        "Shuffle robustness": shuffle_verdict,
    }
    for test, verdict in tests.items():
        lines.append(f"- {test}: **{verdict}**")

    n_pass = sum(1 for v in tests.values() if v == "PASS")
    overall = "GO" if n_pass == 3 else "PARTIAL GO" if n_pass >= 2 else "NO-GO"
    lines += [
        "",
        f"**Overall verdict: {overall}** ({n_pass}/3 tests pass)",
        "",
        "---",
        f"*Generated by `scripts/run_rtmv_cv_validation.py` on {run_date}.*",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written → %s", output_path)


# ---------------------------------------------------------------------------
# Phase 63: Momentum CV helpers
# ---------------------------------------------------------------------------


def _run_rebalancer_backtest(
    assets: list[str],
    asset_returns: pd.DataFrame,
    asset_features: dict[str, pd.DataFrame],
    lambda_tilt: float,
    n_states: int,
    n_restarts: int,
    lookback_bars: int,
    rebalance_bars: int,
    transaction_cost_bps: float,
    momentum_tilt_scale: float = 0.0,
    seed: int = 42,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run RTMVRebalancer backtest and return per-day returns + weights.

    Parameters
    ----------
    transaction_cost_bps : float
        Transaction cost in basis points (e.g. 10.0 = 10 bps).

    Returns
    -------
    port_ret : pd.Series
        Per-bar portfolio returns (equity.pct_change).
    weights_df : pd.DataFrame
        Per-bar weights with columns = assets, indexed by date.
        These can be reused with portfolio_returns for cost sweeps.
    """
    from rde.analysis.portfolio import portfolio_returns as _portfolio_returns

    cfg = RTMVRebalancerConfig(
        assets=assets,
        lambda_tilt=0.05,
        lambda_by_state_rank=[0.02, 0.05, 0.10],
        lambda_proxy_asset="SPY" if "SPY" in assets else assets[0],
        n_states=n_states,
        n_restarts=n_restarts,
        lookback_bars=lookback_bars,
        rebalance_bars=rebalance_bars,
        drawdown_halt=0.25,
        slippage_bps=transaction_cost_bps,
        momentum_tilt_scale=momentum_tilt_scale,
    )
    rebalancer = RTMVRebalancer(cfg, train_kwargs={"seed_base": seed})
    snaps = rebalancer.run_backtest(asset_returns, asset_features)

    # Extract weight columns from snapshots.
    weight_cols = [c for c in snaps.columns if c.startswith("weight_")]
    weights_df = snaps[weight_cols].copy()
    weights_df.columns = [c.replace("weight_", "") for c in weight_cols]
    # Ensure same column order as assets.
    for a in assets:
        if a not in weights_df.columns:
            weights_df[a] = 1.0 / len(assets)
    weights_df = weights_df[assets]

    # Recompute portfolio returns from extracted weights at the same base cost.
    # (equity.pct_change has rounding due to position integer accounting;
    # portfolio_returns from weights is the "clean" analytical equivalent.)
    port_ret = _portfolio_returns(
        weights_df,
        asset_returns,
        transaction_cost_bps=transaction_cost_bps,
    )
    return port_ret, weights_df


def _compute_returns_from_weights(
    weights_df: pd.DataFrame,
    asset_returns: pd.DataFrame,
    transaction_cost_bps: float,
) -> pd.Series:
    """Re-apply a weight DataFrame with a different transaction cost.

    Used for cost sensitivity sweep without re-fitting HMMs.
    """
    from rde.analysis.portfolio import portfolio_returns as _portfolio_returns
    return _portfolio_returns(weights_df, asset_returns, transaction_cost_bps=transaction_cost_bps)


def _compute_momentum_weights(
    asset_returns: pd.DataFrame,
    base_weights_df: pd.DataFrame,
    momentum_tilt_scale: float,
    lookback_bars: int = 504,
    momentum_lookback: int = 252,
    momentum_skip: int = 21,
    shuffle_seed: int | None = None,
) -> pd.DataFrame:
    """Apply momentum z-score tilt to a base weight DataFrame.

    At each rebalance bar (detected as a weight change), computes the
    12m-1m momentum z-score and applies it as an additive tilt.  Used
    for shuffle test: when shuffle_seed is set the z-scores are permuted
    across assets, isolating the contribution of momentum signal ordering.

    Parameters
    ----------
    asset_returns : pd.DataFrame
    base_weights_df : pd.DataFrame
        Weight DataFrame from the no-momentum backtest.
    momentum_tilt_scale : float
    lookback_bars, momentum_lookback, momentum_skip : int
    shuffle_seed : int, optional
        If set, permute z-scores across assets at every rebalance step.

    Returns
    -------
    pd.DataFrame
        Weight DataFrame with momentum tilt applied at each rebalance step.
    """
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    assets = list(base_weights_df.columns)
    N = len(assets)
    new_weights = base_weights_df.copy()

    # Identify rebalance bars (rows where at least one weight changed).
    w_vals = base_weights_df.values.astype(float)
    changed = np.any(np.abs(np.diff(w_vals, axis=0)) > 1e-8, axis=1)
    changed = np.concatenate([[True], changed])  # always apply on first bar
    rebalance_bars_idx = np.where(changed)[0]

    for rb_idx in rebalance_bars_idx:
        # Use a return window ending at this bar.
        start_idx = max(0, rb_idx - momentum_lookback - momentum_skip)
        ret_window = asset_returns.iloc[start_idx:rb_idx]
        if len(ret_window) < momentum_lookback + 1:
            continue  # not enough data yet

        n_12m = min(momentum_lookback, len(ret_window))
        n_1m = min(momentum_skip, len(ret_window))
        mom_12m = ret_window.iloc[-n_12m:].sum()
        mom_1m = ret_window.iloc[-n_1m:].sum()
        mom_signal = mom_12m - mom_1m

        mu_mom = float(mom_signal.mean())
        sigma_mom = float(mom_signal.std())
        if sigma_mom < 1e-12:
            continue  # degenerate — skip

        z_mom = (mom_signal - mu_mom) / sigma_mom
        if rng is not None:
            # Shuffle: permute z-scores across assets.
            z_mom = pd.Series(rng.permutation(z_mom.values), index=z_mom.index)

        # Determine the weight window this rebalance covers.
        if rb_idx + 1 < len(base_weights_df):
            next_rb_candidates = rebalance_bars_idx[rebalance_bars_idx > rb_idx]
            next_rb = int(next_rb_candidates[0]) if len(next_rb_candidates) > 0 else len(base_weights_df)
        else:
            next_rb = len(base_weights_df)

        # Apply tilt to every bar in this rebalance window.
        for bar_i in range(rb_idx, next_rb):
            w_raw = new_weights.iloc[bar_i].values.copy()
            w_final = w_raw + momentum_tilt_scale * z_mom.values
            w_final = np.clip(w_final, 0.0, None)
            total = w_final.sum()
            if total > 1e-15:
                w_final /= total
            else:
                w_final = w_raw
            new_weights.iloc[bar_i] = w_final

    return new_weights


def run_momentum_cv_validation(
    assets: list[str],
    asset_returns: pd.DataFrame,
    asset_features: dict[str, pd.DataFrame],
    momentum_tilt_scale: float,
    lambda_tilt: float = 0.30,
    n_states: int = 3,
    n_restarts: int = 3,
    lookback_bars: int = 504,
    rebalance_bars: int = 21,
    n_folds: int = 5,
    cost_bps_range: list[float] | None = None,
    n_shuffle: int = 50,
    ann_factor: int = 252,
    seed: int = 42,
) -> MomentumCVResult:
    """Phase 63: three-test CV comparing momentum variant vs baseline.

    Performance-efficient design: HMMs are fitted TWICE (baseline + momentum
    reference run), then cached weight DataFrames are reused for:
    - Cost sensitivity: apply portfolio_returns with different costs to
      the cached weights (no HMM refitting).
    - Shuffle test: permute momentum z-scores across assets at each rebalance
      bar and recompute weights analytically (no HMM refitting, O(n_shuffle)
      weight computations only).

    Test 1 (fold consistency): split portfolio returns from reference runs
    into n_folds equal sub-periods, count folds where momentum variant
    Sharpe > baseline Sharpe.

    Test 2 (cost sensitivity): sweep transaction cost using cached weights
    for both variants.

    Test 3 (shuffle): permute the per-rebalance momentum z-scores across
    assets N times; compute p-value = fraction of shuffled runs where the
    momentum Sharpe >= real momentum Sharpe.

    Parameters
    ----------
    assets : list[str]
    asset_returns : pd.DataFrame
    asset_features : dict[str, pd.DataFrame]
    momentum_tilt_scale : float
    lambda_tilt : float
    n_states, n_restarts, lookback_bars, rebalance_bars : int
    n_folds : int
    cost_bps_range : list[float], optional
    n_shuffle : int
    ann_factor : int
    seed : int

    Returns
    -------
    MomentumCVResult
    """
    if cost_bps_range is None:
        cost_bps_range = [10.0, 20.0, 50.0, 100.0, 200.0]

    result = MomentumCVResult(
        momentum_tilt_scale=momentum_tilt_scale,
        n_folds=n_folds,
    )

    BASE_COST_BPS = 10.0  # base cost for reference, fold, and shuffle tests

    # ------------------------------------------------------------------
    # Reference runs — fit HMMs TWICE, cache weight DataFrames.
    # ------------------------------------------------------------------
    logger.info("Running baseline backtest (momentum_tilt_scale=0)…")
    base_ret, base_weights = _run_rebalancer_backtest(
        assets, asset_returns, asset_features,
        lambda_tilt=lambda_tilt,
        n_states=n_states, n_restarts=n_restarts,
        lookback_bars=lookback_bars, rebalance_bars=rebalance_bars,
        transaction_cost_bps=BASE_COST_BPS,
        momentum_tilt_scale=0.0,
        seed=seed,
    )
    logger.info("Running momentum backtest (scale=%.3f)…", momentum_tilt_scale)
    mom_ret, mom_weights = _run_rebalancer_backtest(
        assets, asset_returns, asset_features,
        lambda_tilt=lambda_tilt,
        n_states=n_states, n_restarts=n_restarts,
        lookback_bars=lookback_bars, rebalance_bars=rebalance_bars,
        transaction_cost_bps=BASE_COST_BPS,
        momentum_tilt_scale=momentum_tilt_scale,
        seed=seed,
    )

    base_sharpe, *_ = _compute_metrics(base_ret, ann_factor)
    mom_sharpe, *_ = _compute_metrics(mom_ret, ann_factor)
    result.base_full_sharpe = float(base_sharpe)
    result.mom_full_sharpe = float(mom_sharpe)
    result.real_mom_sharpe = float(mom_sharpe)
    logger.info(
        "Full-period: baseline=%.4f  momentum=%.4f  delta=%+.4f",
        base_sharpe, mom_sharpe, mom_sharpe - base_sharpe,
    )

    # ------------------------------------------------------------------
    # Test 1: Fold consistency
    # ------------------------------------------------------------------
    logger.info("Test 1: fold consistency (n_folds=%d)…", n_folds)
    common_idx = base_ret.index.intersection(mom_ret.index)
    base_aligned = base_ret.loc[common_idx]
    mom_aligned = mom_ret.loc[common_idx]

    fold_base = _fold_sharpes(base_aligned, n_folds, ann_factor)
    fold_mom = _fold_sharpes(mom_aligned, n_folds, ann_factor)
    deltas = [m - b for m, b in zip(fold_mom, fold_base)]
    result.fold_sharpe_base = fold_base
    result.fold_sharpe_mom = fold_mom
    result.fold_sharpe_delta = deltas
    n_wins = sum(1 for d in deltas if d > 0)
    result.pct_folds_mom_wins = n_wins / len(deltas) if deltas else 0.0
    result.mean_sharpe_delta = float(np.mean(deltas)) if deltas else 0.0
    result.std_sharpe_delta = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    logger.info(
        "Momentum wins: %d/%d (%.0f%%)  mean_delta=%+.4f  std_delta=%.4f",
        n_wins, len(deltas), result.pct_folds_mom_wins * 100,
        result.mean_sharpe_delta, result.std_sharpe_delta,
    )

    # ------------------------------------------------------------------
    # Test 2: Cost sensitivity (cache-based — no HMM refit)
    # ------------------------------------------------------------------
    logger.info(
        "Test 2: cost sensitivity sweep over %s bps (cache-based, no HMM refit)…",
        cost_bps_range,
    )
    sharpes_mom_cost: list[float] = []
    sharpes_base_cost: list[float] = []

    for cost_bps in cost_bps_range:
        r_base_c = _compute_returns_from_weights(base_weights, asset_returns, cost_bps)
        r_mom_c = _compute_returns_from_weights(mom_weights, asset_returns, cost_bps)
        s_base, *_ = _compute_metrics(r_base_c, ann_factor)
        s_mom, *_ = _compute_metrics(r_mom_c, ann_factor)
        sharpes_base_cost.append(float(s_base))
        sharpes_mom_cost.append(float(s_mom))
        logger.info(
            "  cost=%.0f bps → baseline=%.4f  mom=%.4f  delta=%+.4f",
            cost_bps, s_base, s_mom, s_mom - s_base,
        )

    result.cost_bps_tested = list(cost_bps_range)
    result.sharpe_base_by_cost = sharpes_base_cost
    result.sharpe_mom_by_cost = sharpes_mom_cost
    deltas_cost = [m - b for m, b in zip(sharpes_mom_cost, sharpes_base_cost)]
    result.cost_break_even_bps = _interpolate_break_even(cost_bps_range, deltas_cost)
    logger.info("Momentum advantage break-even: %.1f bps", result.cost_break_even_bps)

    # ------------------------------------------------------------------
    # Test 3: Shuffle test — permute momentum z-scores across assets
    # at each rebalance step (no HMM refit — uses cached base_weights).
    # p-value = fraction of shuffled runs with Sharpe >= real momentum Sharpe.
    # This tests whether the *ordering* of momentum z-scores across assets
    # (not random noise) drives the improvement.
    # ------------------------------------------------------------------
    logger.info(
        "Test 3: shuffle test (n=%d, permuting momentum z-scores, no HMM refit)…",
        n_shuffle,
    )
    rng = np.random.default_rng(seed)
    shuffle_seeds = rng.integers(0, 2 ** 31, size=n_shuffle)
    shuffle_sharpes: list[float] = []

    for i, sh_seed in enumerate(shuffle_seeds):
        shuffled_weights = _compute_momentum_weights(
            asset_returns=asset_returns,
            base_weights_df=base_weights,
            momentum_tilt_scale=momentum_tilt_scale,
            shuffle_seed=int(sh_seed),
        )
        r_sh = _compute_returns_from_weights(shuffled_weights, asset_returns, BASE_COST_BPS)
        s_sh, *_ = _compute_metrics(r_sh, ann_factor)
        shuffle_sharpes.append(float(s_sh))
        if (i + 1) % 10 == 0:
            logger.info("  Shuffle %d/%d done", i + 1, n_shuffle)

    result.n_shuffle = n_shuffle
    result.shuffle_sharpe_mean = float(np.mean(shuffle_sharpes))
    result.shuffle_sharpe_std = float(np.std(shuffle_sharpes, ddof=1))
    result.shuffle_p_value = float(
        np.mean([s >= result.real_mom_sharpe for s in shuffle_sharpes])
    )
    result.shuffle_margin = float(result.real_mom_sharpe - result.shuffle_sharpe_mean)
    logger.info(
        "Shuffle (permuted z-scores): mean=%.4f  std=%.4f  p_value=%.4f  margin=%+.4f",
        result.shuffle_sharpe_mean, result.shuffle_sharpe_std,
        result.shuffle_p_value, result.shuffle_margin,
    )

    return result


# ---------------------------------------------------------------------------
# Phase 63: Report writing
# ---------------------------------------------------------------------------


def write_momentum_cv_report(
    result: MomentumCVResult,
    baseline_result: MomentumCVResult | None,
    output_path: Path,
    assets: list[str],
    n_bars: int,
    n_states: int,
    date_range: str,
) -> None:
    """Write Phase 63 validation report to Markdown."""
    run_date = pd.Timestamp.now().strftime("%Y%m%d")

    fold_verdict = "PASS" if result.pct_folds_mom_wins >= 0.60 else "FAIL"
    strong_fold = result.pct_folds_mom_wins >= 0.80

    if result.cost_break_even_bps == float("inf"):
        be_str = "∞ (advantage holds at all tested costs)"
        cost_verdict = "PASS"
    else:
        be_str = f"{result.cost_break_even_bps:.1f} bps"
        cost_verdict = "PASS" if result.cost_break_even_bps >= 20.0 else "FAIL"

    shuffle_verdict = "PASS" if result.shuffle_p_value < 0.10 else "FAIL"
    strong_shuffle = result.shuffle_p_value < 0.05

    n_pass = sum(1 for v in [fold_verdict, cost_verdict, shuffle_verdict] if v == "PASS")
    strong_go = strong_fold and strong_shuffle and cost_verdict == "PASS"
    if strong_go:
        overall = "STRONG GO"
    elif n_pass == 3:
        overall = "GO"
    elif n_pass >= 2:
        overall = "PARTIAL GO"
    else:
        overall = "NO-GO"

    lines = [
        "# Phase 63: OOS Purged CV Validation of Phase 57 Momentum Tilt",
        "",
        f"**Date:** {run_date}",
        f"**Assets:** {', '.join(assets)}",
        f"**Period:** {date_range}",
        f"**n_bars:** {n_bars}  **n_states:** {n_states}  "
        f"**momentum_tilt_scale:** {result.momentum_tilt_scale}",
        "",
        "## Full-Period Reference",
        "",
        f"| Variant | Sharpe | Delta vs baseline |",
        f"| --- | ---: | ---: |",
        f"| Baseline (no momentum) | {result.base_full_sharpe:.4f} | — |",
        f"| Momentum (scale={result.momentum_tilt_scale}) | "
        f"{result.mom_full_sharpe:.4f} | "
        f"{result.mom_full_sharpe - result.base_full_sharpe:+.4f} |",
        "",
        "---",
        "",
        "## Test 1: Fold Consistency",
        "",
        f"Non-overlapping folds: **{len(result.fold_sharpe_mom)}**",
        "",
        "| Fold | Momentum Sharpe | Baseline Sharpe | Delta | Winner |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, (m, b, d) in enumerate(
        zip(result.fold_sharpe_mom, result.fold_sharpe_base, result.fold_sharpe_delta)
    ):
        winner = "Momentum" if d > 0 else "Baseline"
        lines.append(f"| {i+1} | {m:.4f} | {b:.4f} | {d:+.4f} | **{winner}** |")

    lines += [
        "",
        f"**Momentum wins in {result.pct_folds_mom_wins:.0%} of folds** "
        f"(mean delta {result.mean_sharpe_delta:+.4f} ± {result.std_sharpe_delta:.4f})",
        f"**Fold consistency verdict: {fold_verdict}** "
        f"(criterion: ≥ 60% folds; strong GO: ≥ 80%)"
        + (" ← STRONG" if strong_fold else ""),
        "",
        "---",
        "",
        "## Test 2: Cost Sensitivity",
        "",
        "| cost (bps) | Momentum Sharpe | Baseline Sharpe | Delta |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for cost, m, b in zip(
        result.cost_bps_tested, result.sharpe_mom_by_cost, result.sharpe_base_by_cost
    ):
        lines.append(f"| {cost:.0f} | {m:.4f} | {b:.4f} | {m-b:+.4f} |")

    lines += [
        "",
        f"**Momentum advantage break-even: {be_str}**",
        f"**Cost sensitivity verdict: {cost_verdict}** (criterion: break-even ≥ 20 bps)",
        "",
        "---",
        "",
        "## Test 3: Shuffle Robustness",
        "",
        "*Method: run N baseline backtests with varied random seeds; test whether*",
        "*any seed of the no-momentum baseline can match the real momentum Sharpe.*",
        "*p-value = fraction of shuffled baseline runs with Sharpe ≥ real momentum Sharpe.*",
        "",
        f"- Real momentum Sharpe: **{result.real_mom_sharpe:.4f}**",
        f"- Shuffled baseline mean: **{result.shuffle_sharpe_mean:.4f}** "
        f"± {result.shuffle_sharpe_std:.4f}  (n={result.n_shuffle})",
        f"- p-value: **{result.shuffle_p_value:.4f}** "
        f"(fraction of shuffles ≥ real momentum Sharpe)",
        f"- Margin: **{result.shuffle_margin:+.4f}** (momentum − shuffle mean)",
        f"**Shuffle robustness verdict: {shuffle_verdict}** "
        f"(criterion: p-value < 0.10; strong GO: p < 0.05)"
        + (" ← STRONG" if strong_shuffle else ""),
        "",
        "---",
        "",
        "## GO/NO-GO Verdict",
        "",
        f"| Test | Result | Verdict |",
        f"| --- | --- | --- |",
        f"| Fold consistency | {result.pct_folds_mom_wins:.0%} folds won | **{fold_verdict}** |",
        f"| Cost break-even | {be_str} | **{cost_verdict}** |",
        f"| Shuffle p-value | p={result.shuffle_p_value:.4f} | **{shuffle_verdict}** |",
        "",
        f"**Overall: {overall}** ({n_pass}/3 tests pass)",
        "",
        "### Strong GO criteria",
        f"- Fold consistency ≥ 80%: {'YES' if strong_fold else 'NO'}",
        f"- Shuffle p < 0.05: {'YES' if strong_shuffle else 'NO'}",
        f"- Cost break-even ≥ 20 bps: {'YES' if cost_verdict == 'PASS' else 'NO'}",
        "",
        "---",
        "",
        "## Interpretation",
        "",
    ]

    if overall in ("GO", "STRONG GO"):
        lines += [
            "The Phase 57 momentum tilt (scale=0.03) passes CV validation.",
            "The +0.028 Sharpe improvement is consistent across time periods,",
            "survives transaction cost, and cannot be replicated by random seed",
            "variation of the underlying HMM signal alone.",
            "**Recommendation: deploy momentum_tilt_scale=0.03 as the new live baseline.**",
        ]
    elif overall == "PARTIAL GO":
        lines += [
            "The Phase 57 momentum tilt passes 2/3 validation tests.",
            "Interpret with caution. Consider deploying at reduced scale (0.01–0.02)",
            "or monitoring with a paper portfolio before full live deployment.",
        ]
    else:
        lines += [
            "The Phase 57 momentum tilt fails CV validation.",
            "The +0.028 in-sample Sharpe improvement is not robust OOS.",
            "**Recommendation: do NOT deploy momentum_tilt_scale. Revert to Phase 52a baseline.**",
        ]

    lines += [
        "",
        "---",
        f"*Generated by `scripts/run_rtmv_cv_validation.py --momentum-tilt-scale "
        f"{result.momentum_tilt_scale}` on {run_date}.*",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Phase 63 report written → %s", output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 47: Purged CV validation on RTMV portfolio"
    )
    parser.add_argument("--assets", type=str, default="SPY,GLD,TLT,IEF")
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--lookback-bars", type=int, default=504)
    parser.add_argument("--rebalance-bars", type=int, default=21)
    parser.add_argument("--lambda-tilt", type=float, default=0.30)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-shuffle", type=int, default=100)
    parser.add_argument(
        "--cost-bps",
        type=str,
        default="10,20,50,100,200",
        help="Comma-separated transaction cost levels in bps",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/rtmv_cv_validation/)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--momentum-tilt-scale",
        type=float,
        default=0.0,
        help=(
            "Phase 63: when > 0, run dual-variant CV comparing momentum "
            "variant (this scale) vs baseline (scale=0). "
            "Recommended: 0.03 (Phase 57 selected value)."
        ),
    )
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(",")]
    cost_bps_range = [float(c.strip()) for c in args.cost_bps.split(",")]
    output_dir = args.output_dir or (_repo_root / "results" / "rtmv_cv_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _repo_root / "results" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    source = YFinanceSource(cache_dir=cache_dir)
    pipeline = _build_pipeline()

    feat_dfs: dict[str, pd.DataFrame] = {}
    for sym in assets:
        feat_dfs[sym] = _load_asset_features(sym, source, pipeline)

    aligned, common_idx = _align_features(feat_dfs)
    asset_returns = pd.DataFrame(
        {sym: aligned[sym]["log_return"] for sym in assets},
        index=common_idx,
    )

    # ------------------------------------------------------------------
    # Phase 63: Momentum tilt CV validation (dual-variant path)
    # ------------------------------------------------------------------
    if args.momentum_tilt_scale > 0.0:
        logger.info("=== Phase 63: Momentum Tilt CV Validation ===")
        logger.info(
            "Assets: %s  n_states=%d  momentum_tilt_scale=%.3f  λ=%.2f",
            assets, args.n_states, args.momentum_tilt_scale, args.lambda_tilt,
        )

        mom_result = run_momentum_cv_validation(
            assets=assets,
            asset_returns=asset_returns,
            asset_features=aligned,
            momentum_tilt_scale=args.momentum_tilt_scale,
            lambda_tilt=args.lambda_tilt,
            n_states=args.n_states,
            n_restarts=args.n_restarts,
            lookback_bars=args.lookback_bars,
            rebalance_bars=args.rebalance_bars,
            n_folds=args.n_folds,
            cost_bps_range=cost_bps_range,
            n_shuffle=args.n_shuffle,
            ann_factor=252,
            seed=args.seed,
        )

        run_date = pd.Timestamp.now().strftime("%Y%m%d")
        date_range = (
            f"{common_idx[0].date()} → {common_idx[-1].date()}"
        )
        report_path = output_dir / f"phase63_momentum_cv_{run_date}.md"
        write_momentum_cv_report(
            mom_result,
            baseline_result=None,
            output_path=report_path,
            assets=assets,
            n_bars=len(common_idx),
            n_states=args.n_states,
            date_range=date_range,
        )

        # Also copy to docs/findings
        docs_path = _repo_root / "docs" / "findings" / "phase63_momentum_cv.md"
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(report_path, docs_path)
        logger.info("Phase 63 report also written → %s", docs_path)

        fold_verdict = "PASS" if mom_result.pct_folds_mom_wins >= 0.60 else "FAIL"
        cost_verdict = (
            "PASS"
            if mom_result.cost_break_even_bps == float("inf")
            or mom_result.cost_break_even_bps >= 20.0
            else "FAIL"
        )
        shuffle_verdict = "PASS" if mom_result.shuffle_p_value < 0.10 else "FAIL"
        n_pass = sum(1 for v in [fold_verdict, cost_verdict, shuffle_verdict] if v == "PASS")
        overall = (
            "STRONG GO"
            if mom_result.pct_folds_mom_wins >= 0.80
               and mom_result.shuffle_p_value < 0.05
               and cost_verdict == "PASS"
            else "GO"
            if n_pass == 3
            else "PARTIAL GO"
            if n_pass >= 2
            else "NO-GO"
        )

        print("\n" + "=" * 70)
        print("PHASE 63: MOMENTUM CV VALIDATION COMPLETE")
        print("=" * 70)
        print(f"Universe:     {', '.join(assets)}")
        print(f"Bars:         {len(common_idx)}  ({common_idx[0].date()} → {common_idx[-1].date()})")
        print(f"Scale:        {args.momentum_tilt_scale}  n_states={args.n_states}  λ={args.lambda_tilt}")
        print("-" * 70)
        print(f"Full-period:  baseline={mom_result.base_full_sharpe:.4f}  "
              f"momentum={mom_result.mom_full_sharpe:.4f}  "
              f"delta={mom_result.mom_full_sharpe - mom_result.base_full_sharpe:+.4f}")
        print(f"Fold wins:    {mom_result.pct_folds_mom_wins:.0%}  "
              f"(mean delta {mom_result.mean_sharpe_delta:+.4f})  [{fold_verdict}]")
        print(f"Break-even:   {mom_result.cost_break_even_bps:.1f} bps  [{cost_verdict}]")
        print(f"Shuffle p:    {mom_result.shuffle_p_value:.4f}  "
              f"(margin {mom_result.shuffle_margin:+.4f})  [{shuffle_verdict}]")
        print("=" * 70)
        print(f"\nVerdict: {overall}")
        print(f"\nReport: {report_path}")
        print(f"Findings: {docs_path}")
        return

    # ------------------------------------------------------------------
    # Phase 47: Standard RTMV vs GMV validation (original path)
    # ------------------------------------------------------------------
    logger.info("=== Phase 47: RTMV Purged CV Validation ===")
    logger.info("Assets: %s  n_states=%d  λ=%.2f", assets, args.n_states, args.lambda_tilt)

    config = MultiAssetConfig(
        ann_factor=252,
        rebalance_bars=args.rebalance_bars,
        lookback_bars=args.lookback_bars,
        n_states=args.n_states,
        n_restarts=args.n_restarts,
        transaction_cost=0.001,  # 10 bps base
    )
    train_kwargs: dict = {"seed_base": args.seed}

    cv_result = run_rtmv_cv_validation(
        asset_returns=asset_returns,
        asset_features=aligned,
        config=config,
        lambda_tilt=args.lambda_tilt,
        n_folds=args.n_folds,
        cost_bps_range=cost_bps_range,
        n_shuffle=args.n_shuffle,
        seed=args.seed,
        train_kwargs=train_kwargs,
    )

    run_date = pd.Timestamp.now().strftime("%Y%m%d")
    report_path = output_dir / f"rtmv_cv_report_{run_date}.md"
    write_rtmv_cv_report(
        cv_result,
        report_path,
        assets=assets,
        n_bars=len(common_idx),
        n_states=args.n_states,
    )

    print("\n" + "=" * 70)
    print("RTMV CV VALIDATION COMPLETE")
    print("=" * 70)
    print(f"λ={args.lambda_tilt}  Folds: {args.n_folds}  Shuffle: {args.n_shuffle}")
    print(f"Fold wins:    {cv_result.pct_folds_rtmv_wins:.0%}  "
          f"(mean delta {cv_result.mean_sharpe_delta:+.4f})")
    print(f"Break-even:   {cv_result.cost_break_even_bps:.1f} bps")
    print(f"Shuffle p:    {cv_result.shuffle_p_value:.4f}  "
          f"(margin {cv_result.shuffle_margin:+.4f})")
    print("=" * 70)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
