"""Phase 47: Purged CV validation on RTMV portfolio.

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

Performance note: HMM fitting is expensive (~5 min for the full period).
This script fits HMMs ONCE (``collect_rtmv_rebalance_cache``), then reuses
cached (w_minvar, exp_returns) for the cost sweep and shuffle test.  Total
runtime: ~7 minutes (1 fit pass + cheap recomputation for cost/shuffle).

Usage
-----
    uv run python scripts/run_rtmv_cv_validation.py \\
        --assets SPY,GLD,TLT,IEF --n-states 3 --lambda-tilt 0.30 \\
        --n-folds 5 --n-shuffle 100 --n-restarts 3

Outputs
-------
    results/{output_dir}/rtmv_cv_report_{YYYYMMDD}.md
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
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(",")]
    cost_bps_range = [float(c.strip()) for c in args.cost_bps.split(",")]
    output_dir = args.output_dir or (_repo_root / "results" / "rtmv_cv_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _repo_root / "results" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Phase 47: RTMV Purged CV Validation ===")
    logger.info("Assets: %s  n_states=%d  λ=%.2f", assets, args.n_states, args.lambda_tilt)

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
