"""Phase 50c: Drawdown halt sweep + risk parity baseline comparison.

Two parts:

1. Sweep ``RTMVRebalancerConfig.drawdown_halt`` over a grid and report
   the equity-curve metrics (Sharpe, Calmar, MDD, ann return, halted
   bars) for each threshold. Helps decide whether the production 20%
   halt is mis-calibrated.

2. Run the inverse-vol risk parity baseline on the same SPY/GLD/TLT/IEF
   universe and compare to the Phase 48 RTMV(λ=0.05) baseline numbers.
   Risk parity is a stronger benchmark than equal-weight or pure
   min-var on the MDD axis, so beating it is a higher bar.

Usage
-----
    uv run python scripts/sweep_drawdown_halt.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from rde.analysis.multi_asset_allocation import MultiAssetConfig
from rde.data.yfinance_source import YFinanceSource
from rde.evaluation.baselines import multi_asset_risk_parity
from rde.features.pipeline import FeaturePipeline
from rde.features.returns import LogReturns, SmoothedReturns
from rde.features.volatility import RollingVolatility
from rde.trading.rtmv_rebalancer import RTMVRebalancer, RTMVRebalancerConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

ASSETS = ["SPY", "GLD", "TLT", "IEF"]
ANN = 252


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load cached yfinance daily bars + features for SPY/GLD/TLT/IEF."""
    source = YFinanceSource(cache_dir=Path("results/cache"))
    pipeline = FeaturePipeline([
        LogReturns(),
        RollingVolatility(window=20),
        SmoothedReturns(window=5),
    ])
    feat_dfs: dict[str, pd.DataFrame] = {}
    for asset in ASSETS:
        raw = source.load(asset, period="max", interval="1d")
        feat_df = pipeline.transform(raw).dropna()
        feat_dfs[asset] = feat_df
        print(f"  {asset}: {len(feat_df)} bars  ({feat_df.index[0].date()} → {feat_df.index[-1].date()})")

    common = feat_dfs[ASSETS[0]].index
    for a in ASSETS[1:]:
        common = common.intersection(feat_dfs[a].index)
    common = common.sort_values()
    for a in ASSETS:
        feat_dfs[a] = feat_dfs[a].loc[common]
    rets = pd.DataFrame(
        {a: feat_dfs[a]["log_return"] for a in ASSETS}, index=common
    )
    print(f"\nAligned dataset: {len(common)} bars  "
          f"({common[0].date()} → {common[-1].date()})\n")
    return rets, feat_dfs


# ---------------------------------------------------------------------------
# Metrics from a snapshots DataFrame
# ---------------------------------------------------------------------------


def _metrics_from_snapshots(snaps: pd.DataFrame) -> dict[str, float]:
    eq = snaps["equity"].dropna()
    ret = eq.pct_change().dropna()
    mu = float(ret.mean()) * ANN
    sig = float(ret.std()) * np.sqrt(ANN) + 1e-15
    sharpe = mu / sig
    peak = eq.cummax()
    mdd = float(((eq - peak) / (peak + 1e-15)).min())  # negative
    calmar = mu / (abs(mdd) + 1e-15)
    n_reb = int(snaps["n_rebalances"].iloc[-1]) if "n_rebalances" in snaps.columns else 0
    halted = int(snaps["is_halted"].sum()) if "is_halted" in snaps.columns else 0
    return {
        "sharpe": sharpe,
        "calmar": calmar,
        "mdd": mdd,
        "ann_return": mu,
        "ann_vol": sig,
        "n_rebalances": n_reb,
        "halted_bars": halted,
        "final_equity": float(eq.iloc[-1]),
    }


# ---------------------------------------------------------------------------
# A. Halt sweep
# ---------------------------------------------------------------------------


def _run_halt_sweep(
    rets: pd.DataFrame, feats: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    halt_grid = [0.10, 0.15, 0.20, 0.25, 0.30, 1.0]
    rows: list[dict] = []
    for halt in halt_grid:
        print(f"--- Sweeping drawdown_halt={halt:.2f} ---")
        cfg = RTMVRebalancerConfig(
            assets=ASSETS,
            lambda_tilt=0.05,
            rebalance_bars=21,
            lookback_bars=504,
            n_states=3,
            n_restarts=3,
            drawdown_halt=halt,
            output_dir=Path("results/halt_sweep_tmp"),
        )
        ma_cfg = MultiAssetConfig(
            ann_factor=ANN,
            rebalance_bars=21,
            lookback_bars=504,
            n_states=3,
            n_restarts=3,
        )
        rebalancer = RTMVRebalancer(cfg, ma_config=ma_cfg)
        snaps = rebalancer.run_backtest(rets, feats)
        m = _metrics_from_snapshots(snaps)
        m["drawdown_halt"] = halt
        rows.append(m)
        print(
            f"  Sharpe={m['sharpe']:.3f}  Calmar={m['calmar']:.3f}  "
            f"MDD={m['mdd']:.1%}  AnnRet={m['ann_return']:.1%}  "
            f"Reb={m['n_rebalances']}  Halted={m['halted_bars']}\n"
        )
    df = pd.DataFrame(rows)
    return df.sort_values("calmar", ascending=False).reset_index(drop=True)


def _print_halt_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("DRAWDOWN HALT SWEEP — sorted by Calmar")
    print("=" * 78)
    print()
    header = (
        "| halt | Sharpe | Calmar |   MDD   | AnnRet  | AnnVol  | "
        "Reb | Halted | Final $ |"
    )
    sep = (
        "|------|--------|--------|---------|---------|---------|"
        "-----|--------|---------|"
    )
    print(header)
    print(sep)
    for _, r in df.iterrows():
        print(
            f"| {r['drawdown_halt']:.2f} | {r['sharpe']:6.3f} | "
            f"{r['calmar']:6.3f} | {r['mdd']:7.2%} | "
            f"{r['ann_return']:6.2%} | {r['ann_vol']:6.2%} | "
            f"{int(r['n_rebalances']):3d} | {int(r['halted_bars']):6d} | "
            f"${r['final_equity']:>7,.0f} |"
        )
    print()


# ---------------------------------------------------------------------------
# B. Risk parity comparison
# ---------------------------------------------------------------------------


# Phase 48 baseline (results/rtmv_live/report_20260510.md)
RTMV_REF = {
    "sharpe": 0.875,
    "calmar": 0.299,
    "mdd": -0.215,
    "ann_return": 0.064,
    "ann_vol": 0.073,
}


def _run_risk_parity(rets: pd.DataFrame) -> dict[str, float]:
    rets_dict = {a: rets[a] for a in rets.columns}
    br = multi_asset_risk_parity(
        rets_dict,
        ann_factor=ANN,
        vol_window=63,
        rebalance_bars=21,
        transaction_cost=0.0001,
    )
    return {
        "sharpe": br.sharpe,
        "calmar": br.calmar,
        "mdd": -br.max_drawdown,  # convert to negative for symmetry
        "ann_return": br.ann_return,
        "ann_vol": br.ann_vol,
        "n_trades": br.n_trades,
        "final_equity": float(br.equity.iloc[-1]),
    }


def _print_risk_parity_comparison(rp: dict[str, float]) -> None:
    print("\n" + "=" * 78)
    print("RISK PARITY vs RTMV(λ=0.05) — same SPY/GLD/TLT/IEF universe")
    print("=" * 78)
    print()
    print("| Strategy             | Sharpe | Calmar |   MDD   | AnnRet  | AnnVol  |")
    print("|----------------------|--------|--------|---------|---------|---------|")
    print(
        f"| Risk Parity (inv-vol)| {rp['sharpe']:6.3f} | {rp['calmar']:6.3f} | "
        f"{rp['mdd']:7.2%} | {rp['ann_return']:6.2%} | {rp['ann_vol']:6.2%} |"
    )
    print(
        f"| RTMV (λ=0.05)        | {RTMV_REF['sharpe']:6.3f} | {RTMV_REF['calmar']:6.3f} | "
        f"{RTMV_REF['mdd']:7.2%} | {RTMV_REF['ann_return']:6.2%} | {RTMV_REF['ann_vol']:6.2%} |"
    )
    print()
    sharpe_win = RTMV_REF["sharpe"] > rp["sharpe"]
    calmar_win = RTMV_REF["calmar"] > rp["calmar"]
    mdd_win = abs(RTMV_REF["mdd"]) < abs(rp["mdd"])
    print("Verdict:")
    print(
        f"  - Sharpe: RTMV {'BEATS' if sharpe_win else 'LOSES TO'} risk parity "
        f"(Δ={RTMV_REF['sharpe'] - rp['sharpe']:+.3f})"
    )
    print(
        f"  - Calmar: RTMV {'BEATS' if calmar_win else 'LOSES TO'} risk parity "
        f"(Δ={RTMV_REF['calmar'] - rp['calmar']:+.3f})"
    )
    print(
        f"  - MDD:    RTMV {'BEATS' if mdd_win else 'LOSES TO'} risk parity "
        f"(Δ={abs(RTMV_REF['mdd']) - abs(rp['mdd']):+.2%})"
    )
    print()


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def _recommendation(df: pd.DataFrame) -> tuple[float, dict[str, float]]:
    best = df.iloc[0]
    halt = float(best["drawdown_halt"])
    print("=" * 78)
    print("RECOMMENDATION")
    print("=" * 78)
    print(
        f"Optimal halt threshold: {halt:.0%}  "
        f"(Calmar={best['calmar']:.3f}, Sharpe={best['sharpe']:.3f}, "
        f"MDD={best['mdd']:.1%}, halted_bars={int(best['halted_bars'])})"
    )

    baseline_row = df[df["drawdown_halt"] == 0.20]
    if not baseline_row.empty:
        b = baseline_row.iloc[0]
        if abs(halt - 0.20) < 1e-9:
            print(
                "Phase 48 baseline (halt=20%) is already the best — "
                "NO change recommended."
            )
        else:
            print(
                f"Phase 48 baseline (halt=20%): "
                f"Sharpe={b['sharpe']:.3f}, Calmar={b['calmar']:.3f}, "
                f"MDD={b['mdd']:.1%}."
            )
            print(
                f"Switching to halt={halt:.0%} would change Calmar by "
                f"{best['calmar'] - b['calmar']:+.3f} and "
                f"halted_bars by {int(best['halted_bars']) - int(b['halted_bars']):+d}."
            )
    print()
    return halt, dict(best)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 78)
    print("Phase 50c: Drawdown halt sweep + risk parity baseline")
    print("=" * 78)
    print(f"Assets: {', '.join(ASSETS)}\n")

    print("Loading aligned data…")
    rets, feats = _load_data()

    print("\n--- Part A: Drawdown halt sweep ---\n")
    sweep_df = _run_halt_sweep(rets, feats)
    _print_halt_table(sweep_df)

    out_dir = Path("results/halt_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = out_dir / "halt_sweep.parquet"
    sweep_df.to_parquet(sweep_path)
    print(f"Sweep written → {sweep_path}\n")

    print("\n--- Part B: Risk parity baseline ---\n")
    rp = _run_risk_parity(rets)
    _print_risk_parity_comparison(rp)

    rp_path = out_dir / "risk_parity.parquet"
    pd.DataFrame([rp]).to_parquet(rp_path)
    print(f"Risk parity result written → {rp_path}\n")

    _recommendation(sweep_df)


if __name__ == "__main__":
    main()
