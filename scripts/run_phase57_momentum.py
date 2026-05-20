"""Phase 57: Cross-asset momentum overlay on RTMV.

Tests whether adding a (12m-1m) cross-sectional momentum z-score tilt
improves the Phase 52a 5-asset spy_rank_bull baseline.

Variants
--------
  baseline       : Phase 52a config (spy_rank_bull, no momentum tilt)
  mom_01         : momentum_tilt_scale=0.01
  mom_03         : momentum_tilt_scale=0.03
  mom_05         : momentum_tilt_scale=0.05

Universe: SPY, GLD, SHY, IEF, TLT (Phase 52a GO universe)
GO threshold: Sharpe >= 0.9830 (baseline 0.9726 + 0.0104)

Usage
-----
    uv run python scripts/run_phase57_momentum.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from rde.data.yfinance_source import YFinanceSource
from rde.features.pipeline import FeaturePipeline
from rde.features.returns import LogReturns, SmoothedReturns
from rde.features.volatility import RollingVolatility
from rde.trading.rtmv_rebalancer import RTMVRebalancer, RTMVRebalancerConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASSETS = ["SPY", "GLD", "SHY", "IEF", "TLT"]

# Phase 52a reference (spy_rank_bull on 5-asset universe)
PHASE52A_BASELINE_SHARPE = 0.9726

GO_THRESHOLD = 0.9830  # baseline + 0.0104

# Phase 51b / 52a winning config
LAMBDA_BY_STATE_RANK = [0.02, 0.05, 0.10]
LAMBDA_PROXY_ASSET = "SPY"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data(
    assets: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Download daily data for all assets and build feature DataFrames.

    Returns
    -------
    asset_returns : pd.DataFrame
        Log-return series on the common trading date index.
    asset_features : dict[str, pd.DataFrame]
        Per-asset feature DataFrames (log_return, volatility_w20,
        smoothed_return_w5).
    """
    source = YFinanceSource(cache_dir=_repo_root / "results" / "cache")
    pipeline = FeaturePipeline([
        LogReturns(),
        RollingVolatility(window=20),
        SmoothedReturns(window=5),
    ])

    feat_dfs: dict[str, pd.DataFrame] = {}
    for asset in assets:
        raw = source.load(asset, period="max", interval="1d")
        feat_dfs[asset] = pipeline.transform(raw).dropna()

    # Intersect to common trading dates.
    common_dates = feat_dfs[assets[0]].index
    for a in assets[1:]:
        common_dates = common_dates.intersection(feat_dfs[a].index)
    common_dates = common_dates.sort_values()

    for a in assets:
        feat_dfs[a] = feat_dfs[a].loc[common_dates]

    asset_returns = pd.DataFrame(
        {a: feat_dfs[a]["log_return"] for a in assets},
        index=common_dates,
    )
    return asset_returns, feat_dfs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _metrics(snaps: pd.DataFrame) -> dict:
    """Extract performance metrics from a snapshots DataFrame."""
    eq = snaps["equity"].dropna()
    ret = eq.pct_change().dropna()
    ann = 252
    mu = float(ret.mean()) * ann
    sig = float(ret.std()) * (ann ** 0.5) + 1e-15
    sharpe = mu / sig
    peak = eq.cummax()
    mdd = float(((eq - peak) / (peak + 1e-15)).min())
    calmar = mu / (abs(mdd) + 1e-15)
    n_reb = int(snaps["n_rebalances"].iloc[-1]) if "n_rebalances" in snaps.columns else 0
    return {
        "Sharpe": round(sharpe, 4),
        "Calmar": round(calmar, 4),
        "MDD": round(mdd, 4),
        "Ann Return": round(mu, 4),
        "Ann Vol": round(sig, 4),
        "N Rebalances": n_reb,
    }


# ---------------------------------------------------------------------------
# Run variant
# ---------------------------------------------------------------------------


def _run_variant(
    assets: list[str],
    asset_returns: pd.DataFrame,
    asset_features: dict[str, pd.DataFrame],
    momentum_tilt_scale: float = 0.0,
) -> dict:
    """Run one RTMV backtest variant and return performance metrics."""
    cfg = RTMVRebalancerConfig(
        assets=assets,
        # Phase 51b / 52a winning configuration:
        lambda_tilt=0.05,
        lambda_by_state_rank=LAMBDA_BY_STATE_RANK,
        lambda_proxy_asset=LAMBDA_PROXY_ASSET,
        n_states=3,
        n_restarts=3,
        lookback_bars=504,
        rebalance_bars=21,
        drawdown_halt=0.25,
        # Phase 57 additive momentum tilt:
        momentum_tilt_scale=momentum_tilt_scale,
    )
    rebalancer = RTMVRebalancer(cfg)
    snaps = rebalancer.run_backtest(asset_returns, asset_features)
    return _metrics(snaps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Phase 57 momentum overlay backtest and print results."""
    print("Loading data…", flush=True)
    asset_returns, asset_features = _load_data(ASSETS)
    n_bars = len(asset_returns)
    date_from = asset_returns.index[0].date()
    date_to = asset_returns.index[-1].date()
    print(
        f"  Universe: {', '.join(ASSETS)}\n"
        f"  {n_bars} bars  ({date_from} → {date_to})\n",
        flush=True,
    )

    variants = [
        ("baseline", 0.00),
        ("mom_01",   0.01),
        ("mom_03",   0.03),
        ("mom_05",   0.05),
    ]

    rows: list[dict] = []
    for name, scale in variants:
        print(f"Running {name} (momentum_tilt_scale={scale})…", flush=True)
        m = _run_variant(ASSETS, asset_returns, asset_features, momentum_tilt_scale=scale)
        m["variant"] = name
        m["momentum_tilt_scale"] = scale
        rows.append(m)
        print(
            f"  Sharpe={m['Sharpe']:.4f}  MDD={m['MDD']:.1%}  "
            f"Calmar={m['Calmar']:.4f}  Ann Return={m['Ann Return']:.1%}  "
            f"N Rebalances={m['N Rebalances']}",
            flush=True,
        )

    # Print summary table.
    print("\n" + "=" * 80)
    print("PHASE 57 — CROSS-ASSET MOMENTUM OVERLAY ON RTMV (5-asset spy_rank_bull)")
    print(f"Universe: {', '.join(ASSETS)}   Bars: {n_bars}   ({date_from} → {date_to})")
    print("=" * 80)
    header = (
        f"{'Variant':<12}{'Scale':>8}{'Sharpe':>10}{'Calmar':>10}"
        f"{'MDD':>10}{'Ann Ret':>10}{'Ann Vol':>10}{'N Reb':>8}"
    )
    print(header)
    print("-" * 80)
    for r in rows:
        print(
            f"{r['variant']:<12}{r['momentum_tilt_scale']:>8.2f}"
            f"{r['Sharpe']:>10.4f}{r['Calmar']:>10.4f}"
            f"{r['MDD']:>10.1%}{r['Ann Return']:>10.1%}"
            f"{r['Ann Vol']:>10.1%}{r['N Rebalances']:>8d}"
        )
    print("=" * 80)

    # Verdict.
    best = max(rows, key=lambda r: float(r["Sharpe"]))
    best_sharpe = float(best["Sharpe"])
    delta = best_sharpe - PHASE52A_BASELINE_SHARPE
    verdict = "GO" if best_sharpe >= GO_THRESHOLD else "NO-GO"

    print(f"\nPhase 52a baseline (spy_rank_bull, 5-asset): Sharpe = {PHASE52A_BASELINE_SHARPE:.4f}")
    print(f"Best Phase 57 result: {best['variant']} (scale={best['momentum_tilt_scale']})  Sharpe = {best_sharpe:.4f}")
    print(f"Delta vs baseline: {'+'if delta>=0 else ''}{delta:.4f}")
    print(f"GO threshold: {GO_THRESHOLD:.4f} (baseline + 0.0104)")
    print(f"\nVerdict: {verdict}")

    # Save results.
    out_dir = _repo_root / "results" / "phase57"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / "momentum_tilt_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    # Return structured dict for programmatic access.
    return {
        "rows": rows,
        "best": best,
        "best_sharpe": best_sharpe,
        "delta": delta,
        "verdict": verdict,
    }


if __name__ == "__main__":
    main()
