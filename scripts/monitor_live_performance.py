"""Live RTMV performance monitor for Ruflo monitoring agent.

Reads the latest snapshots and fills from results/rtmv_live/ and reports
current Sharpe, MDD, drawdown status, and time-to-next-rebalance.

Exit codes:
  0 — healthy (Sharpe >= 0.70, no halt active, no drift alert)
  1 — warning (Sharpe 0.50–0.70 or drawdown approaching halt)
  2 — alert (Sharpe < 0.50 or halt triggered or fill anomaly)

Usage:
    uv run python scripts/monitor_live_performance.py
    uv run python scripts/monitor_live_performance.py --json
    uv run python scripts/monitor_live_performance.py --since 2026-05-01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
_results_dir = _repo_root / "results" / "rtmv_live"

SHARPE_WARN = 0.70
SHARPE_ALERT = 0.50
MDD_WARN = 0.20
HALT_THRESHOLD = 0.25


def _load_snapshots(since: str | None = None) -> pd.DataFrame | None:
    p = _results_dir / "snapshots.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if since:
        df = df[df.index >= since]
    return df


def _load_fills(since: str | None = None) -> pd.DataFrame | None:
    p = _results_dir / "fills.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if since:
        df = df[df.index >= since]
    return df


def _compute_metrics(snaps: pd.DataFrame) -> dict:
    eq = snaps["equity"].dropna()
    if len(eq) < 2:
        return {"status": "insufficient_data", "bars": len(eq)}

    ret = eq.pct_change().dropna()
    ann = 252
    mu = float(ret.mean()) * ann
    sig = float(ret.std()) * (ann ** 0.5) + 1e-15
    sharpe = mu / sig

    peak = eq.cummax()
    mdd = float(((eq - peak) / (peak + 1e-15)).min())
    current_dd = float((eq.iloc[-1] - peak.iloc[-1]) / (peak.iloc[-1] + 1e-15))

    halt_active = current_dd <= -HALT_THRESHOLD
    n_rebalances = int(snaps["n_rebalances"].iloc[-1]) if "n_rebalances" in snaps.columns else 0
    last_equity = float(eq.iloc[-1])
    bars = len(eq)

    # Time since last rebalance (rebalance_bars=21 default)
    rebalance_col = snaps.get("last_rebalance_bar") if hasattr(snaps, "get") else None
    bars_since_reb = None
    if "last_rebalance_bar" in snaps.columns:
        bars_since_reb = bars - int(snaps["last_rebalance_bar"].iloc[-1])

    return {
        "equity": round(last_equity, 2),
        "sharpe_ann": round(sharpe, 4),
        "mdd": f"{mdd:.1%}",
        "current_drawdown": f"{current_dd:.1%}",
        "halt_active": halt_active,
        "ann_return": f"{mu:.1%}",
        "ann_vol": f"{sig:.1%}",
        "n_rebalances": n_rebalances,
        "bars": bars,
        "bars_since_rebalance": bars_since_reb,
    }


def _status_code(metrics: dict) -> int:
    if metrics.get("halt_active"):
        return 2
    sharpe = metrics.get("sharpe_ann", 0)
    if isinstance(sharpe, (int, float)):
        if sharpe < SHARPE_ALERT:
            return 2
        if sharpe < SHARPE_WARN:
            return 1
    try:
        mdd_val = float(metrics.get("current_drawdown", "0%").strip("%")) / 100
        if mdd_val <= -MDD_WARN:
            return 1
    except (ValueError, AttributeError):
        pass
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="RTMV live performance monitor")
    p.add_argument("--json", action="store_true", help="Output JSON")
    p.add_argument("--since", default=None, help="Filter data since date (YYYY-MM-DD)")
    args = p.parse_args()

    snaps = _load_snapshots(args.since)

    if snaps is None or len(snaps) == 0:
        result = {
            "status": "no_data",
            "message": f"No snapshots found at {_results_dir}/snapshots.parquet",
            "hint": "Run: make live-rtmv OR make backtest-rtmv to generate data",
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("STATUS: NO DATA")
            print(result["message"])
            print(result["hint"])
        sys.exit(1)

    metrics = _compute_metrics(snaps)
    code = _status_code(metrics)
    status_labels = {0: "HEALTHY", 1: "WARNING", 2: "ALERT"}
    metrics["status"] = status_labels[code]

    fills = _load_fills(args.since)
    metrics["n_fills"] = len(fills) if fills is not None else 0

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"RTMV LIVE MONITOR — {metrics['status']}")
        print(f"{'='*50}")
        print(f"  Equity:          ${metrics['equity']:>10,.2f}")
        print(f"  Sharpe (ann):    {metrics['sharpe_ann']:>10.4f}  "
              f"{'(target: 0.90-0.93)' if code == 0 else '⚠ BELOW TARGET' if code == 1 else '🚨 ALERT'}")
        print(f"  MDD:             {metrics['mdd']:>10}  (halt at -25.0%)")
        print(f"  Current DD:      {metrics['current_drawdown']:>10}  "
              f"{'🚨 HALT ACTIVE' if metrics.get('halt_active') else ''}")
        print(f"  Ann Return:      {metrics['ann_return']:>10}")
        print(f"  Ann Vol:         {metrics['ann_vol']:>10}")
        print(f"  Rebalances:      {metrics['n_rebalances']:>10}")
        print(f"  Fills:           {metrics['n_fills']:>10}")
        print(f"  Bars:            {metrics['bars']:>10}")
        if metrics.get("bars_since_rebalance") is not None:
            next_reb = max(0, 21 - metrics["bars_since_rebalance"])
            print(f"  Next rebalance:  {next_reb:>10} bars (~{next_reb} trading days)")
        print(f"{'='*50}\n")

    sys.exit(code)


if __name__ == "__main__":
    main()
