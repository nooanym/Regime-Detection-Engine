"""Phase 50b: compare calendar-only vs KL-triggered RTMV rebalancing.

Runs three configurations of the RTMV daily rebalancer on SPY/GLD/TLT/IEF
and prints a comparison table covering Sharpe, Calmar, MDD, ann return,
n_rebalances, and ann turnover.

Configurations
--------------
1. Calendar only      : rebalance_bars=21,  kl_threshold=0.0   (baseline)
2. KL moderate        : rebalance_bars=42,  kl_threshold=0.30
3. KL aggressive      : rebalance_bars=63,  kl_threshold=0.15

Usage
-----
    uv run python scripts/compare_kl_rebalance.py
    uv run python scripts/compare_kl_rebalance.py --start-date 2010-01-01

Output
------
Prints a Markdown-formatted comparison table to stdout and writes a CSV
copy to ``results/phase50b/comparison.csv``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from rde.analysis.multi_asset_allocation import MultiAssetConfig
from rde.data.yfinance_source import YFinanceSource
from rde.features.pipeline import FeaturePipeline
from rde.features.returns import LogReturns, SmoothedReturns
from rde.features.volatility import RollingVolatility
from rde.trading.rtmv_rebalancer import RTMVRebalancer, RTMVRebalancerConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_kl_rebalance")
logger.setLevel(logging.INFO)


@dataclass
class RunConfig:
    label: str
    rebalance_bars: int
    kl_threshold: float


CONFIGS: list[RunConfig] = [
    RunConfig("calendar_only",  rebalance_bars=21, kl_threshold=0.00),
    RunConfig("kl_moderate",    rebalance_bars=42, kl_threshold=0.30),
    RunConfig("kl_aggressive",  rebalance_bars=63, kl_threshold=0.15),
]


def _build_pipeline() -> FeaturePipeline:
    return FeaturePipeline([
        LogReturns(),
        RollingVolatility(window=20),
        SmoothedReturns(window=5),
    ])


def _load_data(
    assets: list[str],
    start_date: str | None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    source = YFinanceSource(cache_dir=Path("results/cache"))
    pipeline = _build_pipeline()

    feat_dfs: dict[str, pd.DataFrame] = {}
    for asset in assets:
        logger.info("Loading %s…", asset)
        raw = source.load(asset, period="max", interval="1d")
        feat_df = pipeline.transform(raw).dropna()
        feat_dfs[asset] = feat_df

    common = feat_dfs[assets[0]].index
    for a in assets[1:]:
        common = common.intersection(feat_dfs[a].index)
    common = common.sort_values()
    if start_date:
        common = common[common >= pd.Timestamp(start_date)]
    for a in assets:
        feat_dfs[a] = feat_dfs[a].loc[common]

    asset_returns = pd.DataFrame(
        {a: feat_dfs[a]["log_return"] for a in assets},
        index=common,
    )
    logger.info(
        "Aligned: %d bars  %s → %s",
        len(common), common[0].date(), common[-1].date(),
    )
    return asset_returns, feat_dfs


def _ann_turnover(rebalancer: RTMVRebalancer, snaps: pd.DataFrame) -> float:
    """One-way annualised turnover.

    Sum of absolute fill notionals (qty * exec_price) divided by the mean
    portfolio equity, scaled to a yearly rate using the bar count and 252
    trading days per year.  Lower is better.
    """
    fills = rebalancer.portfolio.fills
    if not fills:
        return 0.0
    eq = snaps["equity"].dropna()
    mean_eq = float(eq.mean()) if len(eq) else float("nan")
    if mean_eq <= 0 or np.isnan(mean_eq):
        return float("nan")
    notional = sum(abs(f.quantity * f.fill_price) for f in fills)
    n_bars = len(eq)
    years = max(n_bars / 252.0, 1e-9)
    return notional / mean_eq / years


def _metrics(snaps: pd.DataFrame) -> dict[str, float]:
    eq = snaps["equity"].dropna()
    ret = eq.pct_change().dropna()
    ann = 252
    mu = float(ret.mean()) * ann
    sig = float(ret.std()) * np.sqrt(ann) + 1e-15
    sharpe = mu / sig
    peak = eq.cummax()
    dd = (eq - peak) / (peak + 1e-15)
    mdd = float(dd.min())
    calmar = mu / (abs(mdd) + 1e-15)
    return {
        "sharpe": sharpe,
        "calmar": calmar,
        "max_drawdown": mdd,
        "ann_return": mu,
        "ann_vol": sig,
        "final_equity": float(eq.iloc[-1]),
    }


def _run_one(
    label: str,
    rebalance_bars: int,
    kl_threshold: float,
    asset_returns: pd.DataFrame,
    asset_features: dict[str, pd.DataFrame],
    assets: list[str],
) -> dict:
    cfg = RTMVRebalancerConfig(
        assets=assets,
        lambda_tilt=0.05,
        rebalance_bars=rebalance_bars,
        lookback_bars=504,
        n_states=3,
        n_restarts=3,
        initial_capital=100_000.0,
        slippage_bps=5.0,
        drawdown_halt=1.0,  # disable halt for this comparison
        rebalance_kl_threshold=kl_threshold,
    )
    ma_cfg = MultiAssetConfig(
        ann_factor=252,
        rebalance_bars=rebalance_bars,
        lookback_bars=504,
        n_states=3,
        n_restarts=3,
        cov_window_bars=63,
    )
    rebalancer = RTMVRebalancer(cfg, ma_config=ma_cfg)
    logger.info(
        "Running %s: rebalance_bars=%d  kl_threshold=%.3f",
        label, rebalance_bars, kl_threshold,
    )
    snaps = rebalancer.run_backtest(asset_returns, asset_features)
    m = _metrics(snaps)
    return {
        "label": label,
        "rebalance_bars": rebalance_bars,
        "kl_threshold": kl_threshold,
        "n_rebalances": rebalancer.state.n_rebalances,
        "n_kl_rebalances": rebalancer.state.n_kl_rebalances,
        "ann_turnover": _ann_turnover(rebalancer, snaps),
        **m,
    }


def _print_table(rows: list[dict]) -> str:
    header = (
        "| Strategy | Sharpe | Calmar | MaxDD | AnnRet | AnnVol | "
        "N Rebal | N KL | Turnover/yr |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    body = ""
    for r in rows:
        body += (
            f"| {r['label']} "
            f"| {r['sharpe']:.3f} "
            f"| {r['calmar']:.3f} "
            f"| {r['max_drawdown']:.1%} "
            f"| {r['ann_return']:.1%} "
            f"| {r['ann_vol']:.1%} "
            f"| {r['n_rebalances']} "
            f"| {r['n_kl_rebalances']} "
            f"| {r['ann_turnover']:.2f} |\n"
        )
    return header + body


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 50b KL rebalance comparison")
    p.add_argument("--assets", default="SPY,GLD,TLT,IEF")
    p.add_argument("--start-date", default=None)
    p.add_argument("--output-dir", default="results/phase50b")
    args = p.parse_args()

    assets = [a.strip() for a in args.assets.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    asset_returns, asset_features = _load_data(assets, args.start_date)

    rows: list[dict] = []
    for c in CONFIGS:
        row = _run_one(
            c.label,
            c.rebalance_bars,
            c.kl_threshold,
            asset_returns,
            asset_features,
            assets,
        )
        rows.append(row)
        print(
            f"  {c.label}: Sharpe={row['sharpe']:.3f}  "
            f"MDD={row['max_drawdown']:.1%}  "
            f"N_reb={row['n_rebalances']}  N_KL={row['n_kl_rebalances']}  "
            f"Turnover={row['ann_turnover']:.2f}/yr",
            flush=True,
        )

    table = _print_table(rows)
    print("\n" + "=" * 70)
    print("PHASE 50B: KL-TRIGGERED REBALANCE COMPARISON")
    print("=" * 70)
    print(f"Assets:  {', '.join(assets)}")
    print(f"Period:  {asset_returns.index[0].date()} → {asset_returns.index[-1].date()}")
    print(f"Bars:    {len(asset_returns)}")
    print()
    print(table)

    # CSV.
    df = pd.DataFrame(rows)
    csv_path = output_dir / "comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV → {csv_path}")


if __name__ == "__main__":
    main()
