"""Phase 50a: compare fixed λ=0.05 vs adaptive λ on the SPY/GLD/TLT/IEF universe.

Runs two RTMV backtests:

1. Fixed λ=0.05 (current Phase 48 baseline; backtest target Sharpe ~0.875).
2. Adaptive λ ∈ [lambda_min, lambda_max] driven by per-asset last-bar
   posterior entropy.

Prints a side-by-side comparison of Sharpe, Calmar, MDD, Ann Return,
and rebalance count.

Usage
-----
    uv run python scripts/compare_adaptive_lambda.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from rde.analysis.multi_asset_allocation import MultiAssetConfig
from rde.data.yfinance_source import YFinanceSource
from rde.features.pipeline import FeaturePipeline
from rde.features.returns import LogReturns, SmoothedReturns
from rde.features.volatility import RollingVolatility
from rde.trading.rtmv_rebalancer import RTMVRebalancer, RTMVRebalancerConfig

logging.basicConfig(
    level=logging.WARNING,  # quieter output for table comparison
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

ASSETS = ["SPY", "GLD", "TLT", "IEF"]
N_STATES = 3
N_RESTARTS = 3
LOOKBACK = 504
REBALANCE = 21
CAPITAL = 100_000.0


def _build_pipeline() -> FeaturePipeline:
    return FeaturePipeline([
        LogReturns(),
        RollingVolatility(window=20),
        SmoothedReturns(window=5),
    ])


def _load_aligned_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    source = YFinanceSource(cache_dir=Path("results/cache"))
    pipeline = _build_pipeline()
    feat_dfs: dict[str, pd.DataFrame] = {}
    for asset in ASSETS:
        print(f"Loading {asset}…", flush=True)
        raw = source.load(asset, period="max", interval="1d")
        feat_dfs[asset] = pipeline.transform(raw).dropna()

    common_dates = feat_dfs[ASSETS[0]].index
    for a in ASSETS[1:]:
        common_dates = common_dates.intersection(feat_dfs[a].index)
    common_dates = common_dates.sort_values()

    for a in ASSETS:
        feat_dfs[a] = feat_dfs[a].loc[common_dates]

    asset_returns = pd.DataFrame(
        {a: feat_dfs[a]["log_return"] for a in ASSETS},
        index=common_dates,
    )
    print(f"Aligned: {len(common_dates)} bars  "
          f"({common_dates[0].date()} → {common_dates[-1].date()})")
    return asset_returns, feat_dfs


def _metrics(snaps: pd.DataFrame) -> dict[str, float]:
    eq = snaps["equity"].dropna()
    ret = eq.pct_change().dropna()
    ann = 252
    mu = float(ret.mean()) * ann
    sig = float(ret.std()) * (ann ** 0.5) + 1e-15
    sharpe = mu / sig
    peak = eq.cummax()
    dd = (eq - peak) / (peak + 1e-15)
    mdd = float(abs(dd.min()))
    calmar = mu / (mdd + 1e-15)
    return {
        "sharpe": sharpe,
        "calmar": calmar,
        "mdd": mdd,
        "ann_return": mu,
        "ann_vol": sig,
        "final_equity": float(eq.iloc[-1]),
    }


def _run_backtest(
    asset_returns: pd.DataFrame,
    asset_features: dict[str, pd.DataFrame],
    *,
    label: str,
    adaptive_lambda: bool,
    lambda_tilt: float,
    lambda_min: float,
    lambda_max: float,
) -> tuple[dict[str, float], int]:
    cfg = RTMVRebalancerConfig(
        assets=ASSETS,
        lambda_tilt=lambda_tilt,
        rebalance_bars=REBALANCE,
        lookback_bars=LOOKBACK,
        n_states=N_STATES,
        n_restarts=N_RESTARTS,
        initial_capital=CAPITAL,
        adaptive_lambda=adaptive_lambda,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        output_dir=Path("results/rtmv_adaptive_compare") / label,
    )
    ma_cfg = MultiAssetConfig(
        ann_factor=252,
        rebalance_bars=REBALANCE,
        lookback_bars=LOOKBACK,
        n_states=N_STATES,
        n_restarts=N_RESTARTS,
    )
    rb = RTMVRebalancer(cfg, ma_config=ma_cfg)
    print(f"\nRunning {label}  adaptive={adaptive_lambda}  "
          f"λ={lambda_tilt} (or [{lambda_min},{lambda_max}])…", flush=True)
    snaps = rb.run_backtest(asset_returns, asset_features)
    metrics = _metrics(snaps)
    return metrics, rb.state.n_rebalances


def main() -> None:
    asset_returns, asset_features = _load_aligned_data()

    fixed_metrics, fixed_n = _run_backtest(
        asset_returns,
        asset_features,
        label="fixed_l05",
        adaptive_lambda=False,
        lambda_tilt=0.05,
        lambda_min=0.02,
        lambda_max=0.15,
    )

    adaptive_metrics, adaptive_n = _run_backtest(
        asset_returns,
        asset_features,
        label="adaptive",
        adaptive_lambda=True,
        lambda_tilt=0.05,  # ignored when adaptive_lambda=True
        lambda_min=0.02,
        lambda_max=0.15,
    )

    print("\n" + "=" * 72)
    print("ADAPTIVE λ vs FIXED λ=0.05  —  RTMV (SPY/GLD/TLT/IEF)")
    print("=" * 72)
    print(f"Period:   {asset_returns.index[0].date()} → {asset_returns.index[-1].date()}")
    print(f"Bars:     {len(asset_returns)}")
    print(f"n_states={N_STATES}  n_restarts={N_RESTARTS}  "
          f"lookback={LOOKBACK}  rebalance={REBALANCE}")
    print()

    rows = [
        ("Sharpe (ann)", "sharpe", "{:.3f}"),
        ("Calmar", "calmar", "{:.3f}"),
        ("Max Drawdown", "mdd", "{:.1%}"),
        ("Ann Return", "ann_return", "{:.1%}"),
        ("Ann Vol", "ann_vol", "{:.1%}"),
        ("Final Equity", "final_equity", "${:,.0f}"),
    ]
    print(f"{'Metric':<16} {'Fixed λ=0.05':>16} {'Adaptive λ':>16} {'Δ':>14}")
    print("-" * 72)
    for name, key, fmt in rows:
        fixed_v = fixed_metrics[key]
        adapt_v = adaptive_metrics[key]
        delta = adapt_v - fixed_v
        if "%" in fmt:
            delta_str = f"{delta:+.2%}"
        elif "$" in fmt:
            delta_str = f"${delta:+,.0f}"
        else:
            delta_str = f"{delta:+.3f}"
        print(f"{name:<16} {fmt.format(fixed_v):>16} {fmt.format(adapt_v):>16} {delta_str:>14}")

    print(f"{'N Rebalances':<16} {fixed_n:>16} {adaptive_n:>16} {adaptive_n - fixed_n:>+14}")
    print("=" * 72)

    # Verdict.
    sharpe_delta = adaptive_metrics["sharpe"] - fixed_metrics["sharpe"]
    print(
        f"\nVerdict: adaptive λ {'beats' if sharpe_delta > 0 else 'loses to'} fixed λ=0.05 "
        f"by {sharpe_delta:+.3f} Sharpe."
    )


if __name__ == "__main__":
    main()
