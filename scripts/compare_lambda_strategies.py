"""Phase 51: Regime-Conditional λ Comparison.

Runs four RTMV variants on the SPY/GLD/TLT/IEF universe and compares:
  - fixed_l05   : fixed λ=0.05 (Phase 50 baseline)
  - fixed_l10   : fixed λ=0.10
  - rank_bear   : λ_by_rank=[0.10, 0.05, 0.02] — high tilt in bear, low in bull
  - rank_bull   : λ_by_rank=[0.02, 0.05, 0.10] — low tilt in bear, high in bull

Hypothesis: rank_bull should win (amplify regime tilt when equity momentum is
strong; reduce to near-min-var in bear markets where the HMM lags).

Usage
-----
    uv run python scripts/compare_lambda_strategies.py
    uv run python scripts/compare_lambda_strategies.py --start-date 2010-01-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

# Load .env if present.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_data(
    assets: list[str],
    start_date: str | None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
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

    common_dates = feat_dfs[assets[0]].index
    for a in assets[1:]:
        common_dates = common_dates.intersection(feat_dfs[a].index)
    common_dates = common_dates.sort_values()
    if start_date:
        common_dates = common_dates[common_dates >= pd.Timestamp(start_date)]
    for a in assets:
        feat_dfs[a] = feat_dfs[a].loc[common_dates]

    asset_returns = pd.DataFrame(
        {a: feat_dfs[a]["log_return"] for a in assets},
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
    n_reb = int(snaps["n_rebalances"].iloc[-1]) if "n_rebalances" in snaps.columns else 0
    return {
        "Sharpe": round(sharpe, 4),
        "Calmar": round(calmar, 4),
        "MDD": f"{mdd:.1%}",
        "Ann Return": f"{mu:.1%}",
        "Ann Vol": f"{sig:.1%}",
        "Rebalances": n_reb,
        "Final Equity": f"${eq.iloc[-1]:,.0f}",
    }


def _run_variant(
    name: str,
    assets: list[str],
    asset_returns: pd.DataFrame,
    asset_features: dict[str, pd.DataFrame],
    lambda_tilt: float = 0.05,
    lambda_by_state_rank: list[float] | None = None,
) -> dict:
    cfg = RTMVRebalancerConfig(
        assets=assets,
        lambda_tilt=lambda_tilt,
        lambda_by_state_rank=lambda_by_state_rank or [],
        n_states=3,
        n_restarts=3,
        lookback_bars=504,
        rebalance_bars=21,
        drawdown_halt=0.25,
    )
    rebalancer = RTMVRebalancer(cfg)
    snaps = rebalancer.run_backtest(asset_returns, asset_features)
    m = _metrics(snaps)
    m["variant"] = name
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 51: regime-conditional λ comparison")
    p.add_argument("--assets", default="SPY,GLD,TLT,IEF")
    p.add_argument("--start-date", default=None)
    args = p.parse_args()
    assets = [a.strip() for a in args.assets.split(",")]
    n = len(assets)  # n_states=3 so schedule length must be 3

    print(f"\nLoading data for {assets}…")
    asset_returns, asset_features = _load_data(assets, args.start_date)
    print(f"Loaded: {len(asset_returns)} bars  "
          f"({asset_returns.index[0].date()} → {asset_returns.index[-1].date()})\n")

    variants = [
        ("fixed_l05",   dict(lambda_tilt=0.05)),
        ("fixed_l10",   dict(lambda_tilt=0.10)),
        ("rank_bear",   dict(lambda_tilt=0.05, lambda_by_state_rank=[0.10, 0.05, 0.02])),
        ("rank_bull",   dict(lambda_tilt=0.05, lambda_by_state_rank=[0.02, 0.05, 0.10])),
    ]

    results = []
    for name, kwargs in variants:
        print(f"Running {name}…", flush=True)
        m = _run_variant(name, assets, asset_returns, asset_features, **kwargs)
        results.append(m)
        print(f"  Sharpe={m['Sharpe']}  MDD={m['MDD']}  Calmar={m['Calmar']}")

    # Print comparison table.
    print("\n" + "=" * 72)
    print("PHASE 51 — REGIME-CONDITIONAL λ COMPARISON")
    print("=" * 72)
    cols = ["variant", "Sharpe", "Calmar", "MDD", "Ann Return", "Ann Vol", "Rebalances"]
    header = f"{'Variant':<14}" + "".join(f"{c:>12}" for c in cols[1:])
    print(header)
    print("-" * 72)
    for r in results:
        row = f"{r['variant']:<14}" + "".join(f"{str(r[c]):>12}" for c in cols[1:])
        print(row)
    print("=" * 72)

    # Save results.
    out_dir = _repo_root / "results" / "phase51"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).set_index("variant")
    csv_path = out_dir / "lambda_comparison.csv"
    df.to_csv(csv_path)
    print(f"\nResults saved → {csv_path}")

    # Verdict.
    baseline = next(r for r in results if r["variant"] == "fixed_l05")
    best = max(results, key=lambda r: float(r["Sharpe"]))
    delta = float(best["Sharpe"]) - float(baseline["Sharpe"])
    print(f"\nBest variant: {best['variant']} (Sharpe {best['Sharpe']})")
    print(f"vs fixed_l05: {'+'if delta>=0 else ''}{delta:.4f} Sharpe")
    verdict = "GO" if delta > 0.010 else "MARGINAL" if delta > 0.0 else "NO-GO"
    print(f"Verdict: {verdict}  (threshold: >+0.010 Sharpe over fixed_l05)")


if __name__ == "__main__":
    main()
