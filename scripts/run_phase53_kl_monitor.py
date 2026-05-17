"""Phase 53: Online-Posterior KL Monitor Calibration.

Tests the reference-posterior KL trigger at several thresholds on the
5-asset universe (SPY/GLD/SHY/IEF/TLT) with spy_rank_bull schedule.

The Phase 50b KL trigger fired on 87% of bars because it compared posteriors
from two DIFFERENT HMM fits (model drift dominated the KL signal).  Phase 53
fixes this by anchoring the reference posterior at refit time and using the
SAME fitted model's OnlineDecoder for bar-by-bar updates.

GO criteria
-----------
  - OOS Sharpe ≥ 0.920  (vs 5-asset spy_rank_bull baseline 0.9726 in-sample)
  - KL-triggered rebalances ≤ 4/year (total rebalances ≤ 16/year)
  - Sharpe improvement ≥ 0 vs calendar-only 5-asset spy_rank_bull

Usage
-----
    uv run python scripts/run_phase53_kl_monitor.py
    uv run python scripts/run_phase53_kl_monitor.py --thresholds 0.05,0.10,0.20
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

_env_file = _repo_root / ".env"
if _env_file.exists():
    import os
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from rde.data.yfinance_source import YFinanceSource
from rde.features.pipeline import FeaturePipeline
from rde.features.returns import LogReturns, SmoothedReturns
from rde.features.volatility import RollingVolatility
from rde.trading.rtmv_rebalancer import RTMVRebalancer, RTMVRebalancerConfig

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

ASSETS = ["SPY", "GLD", "SHY", "IEF", "TLT"]

# baseline: calendar-only 5-asset spy_rank_bull (from Phase 52a)
BASELINE_SHARPE = 0.9726


def _load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    source = YFinanceSource(cache_dir=_repo_root / "results" / "cache")
    pipeline = FeaturePipeline([
        LogReturns(),
        RollingVolatility(window=20),
        SmoothedReturns(window=5),
    ])
    feat_dfs: dict[str, pd.DataFrame] = {}
    for asset in ASSETS:
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
    return asset_returns, feat_dfs


def _metrics(snaps: pd.DataFrame) -> dict:
    eq = snaps["equity"].dropna()
    ret = eq.pct_change().dropna()
    ann = 252
    mu = float(ret.mean()) * ann
    sig = float(ret.std()) * (ann ** 0.5) + 1e-15
    sharpe = mu / sig
    peak = eq.cummax()
    mdd = float(((eq - peak) / (peak + 1e-15)).min())
    calmar = mu / (abs(mdd) + 1e-15)
    n_reb = int(snaps["n_rebalances"].iloc[-1])
    n_kl53 = int(snaps["n_kl53_triggers"].iloc[-1]) if "n_kl53_triggers" in snaps.columns else 0
    n_years = len(eq) / 252
    return {
        "Sharpe": round(sharpe, 4),
        "Calmar": round(calmar, 4),
        "MDD": f"{mdd:.1%}",
        "Ann Return": f"{mu:.1%}",
        "Rebalances": n_reb,
        "KL53 triggers": n_kl53,
        "Reb/yr": round(n_reb / max(n_years, 1), 1),
        "KL53/yr": round(n_kl53 / max(n_years, 1), 1),
    }


def _run(
    asset_returns: pd.DataFrame,
    asset_features: dict,
    threshold: float,
    min_confidence: float = 0.50,
) -> dict:
    cfg = RTMVRebalancerConfig(
        assets=ASSETS,
        lambda_tilt=0.05,
        lambda_by_state_rank=[0.02, 0.05, 0.10],
        lambda_proxy_asset="SPY",
        kl_trigger_threshold=threshold,
        kl_min_bars_between_triggers=5,
        kl_min_dominant_confidence=min_confidence,
        n_states=3,
        n_restarts=3,
        lookback_bars=504,
        rebalance_bars=21,
        drawdown_halt=0.25,
    )
    snaps = RTMVRebalancer(cfg).run_backtest(asset_returns, asset_features)
    return _metrics(snaps)


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 53: KL monitor calibration")
    p.add_argument(
        "--thresholds", default="0.0,0.05,0.10,0.20,0.50,1.0,2.0",
        help="Comma-separated KL thresholds to test (0.0 = calendar-only baseline)",
    )
    p.add_argument(
        "--min-confidence", type=float, default=0.50,
        help="Minimum dominant-state posterior confidence required to trigger (default 0.50)",
    )
    args = p.parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",")]
    min_conf = args.min_confidence

    print(f"Loading 5-asset data…  (min_confidence={min_conf})", flush=True)
    asset_returns, asset_features = _load_data()
    n_bars = len(asset_returns)
    n_years = n_bars / 252
    print(f"  {n_bars} bars  ({asset_returns.index[0].date()} → {asset_returns.index[-1].date()})")

    rows: list[dict] = []
    for t in thresholds:
        label = "calendar-only" if t == 0.0 else f"kl53={t}"
        print(f"\nRunning {label}…", flush=True)
        m = _run(asset_returns, asset_features, t, min_confidence=min_conf)
        m["threshold"] = t
        m["label"] = label
        m["min_confidence"] = min_conf
        rows.append(m)
        print(f"  Sharpe={m['Sharpe']}  MDD={m['MDD']}  Reb/yr={m['Reb/yr']}  KL53/yr={m['KL53/yr']}")

    print("\n" + "=" * 80)
    print("PHASE 53 — KL MONITOR CALIBRATION (5-asset, spy_rank_bull)")
    print("=" * 80)
    hdr = f"{'Threshold':<14}{'Sharpe':>8}{'Calmar':>8}{'MDD':>8}{'Ann Ret':>9}{'Reb/yr':>8}{'KL53/yr':>9}"
    print(hdr)
    print("-" * 80)
    for r in rows:
        print(
            f"{r['label']:<14}{r['Sharpe']:>8}{r['Calmar']:>8}{r['MDD']:>8}"
            f"{r['Ann Return']:>9}{r['Reb/yr']:>8}{r['KL53/yr']:>9}"
        )
    print("=" * 80)

    baseline = next((r for r in rows if r["threshold"] == 0.0), None)
    best_improvement = None
    for r in rows:
        if r["threshold"] == 0.0:
            continue
        delta = r["Sharpe"] - (baseline["Sharpe"] if baseline else BASELINE_SHARPE)
        reb_ok = r["KL53/yr"] <= 4.0
        sharpe_ok = r["Sharpe"] >= 0.920
        verdict = "GO" if (delta > 0 and reb_ok and sharpe_ok) else "NO-GO"
        print(f"\n  kl53={r['threshold']}: delta={delta:+.4f}  KL53/yr={r['KL53/yr']}  → {verdict}")
        if delta > 0 and reb_ok and (best_improvement is None or delta > best_improvement[0]):
            best_improvement = (delta, r)

    if best_improvement:
        delta, r = best_improvement
        print(f"\nBest threshold: {r['threshold']}  Sharpe={r['Sharpe']}  delta={delta:+.4f}")

    out_dir = _repo_root / "results" / "phase53"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / "kl_calibration.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")


if __name__ == "__main__":
    main()
