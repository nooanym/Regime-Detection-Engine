"""Phase 52a: Bond Maturity Ladder Universe Expansion.

Tests whether adding duration-differentiated bond ETFs improves RTMV
performance by giving the HMM finer yield-curve regime discrimination.

Universes
---------
  4-asset  : SPY, GLD, TLT, IEF              (current baseline)
  5-asset  : SPY, GLD, SHY, IEF, TLT         (add 1-3Y SHY)
  6-asset  : SPY, GLD, SHY, IEF, TLT, TIP    (full ladder + TIPS)

Variants per universe
---------------------
  gmv            : λ=0.0 (pure global min-var — regime-agnostic reference)
  fixed_l05      : λ=0.05 (Phase 50 baseline)
  spy_rank_bull  : λ_by_rank=[0.02,0.05,0.10], proxy=SPY (Phase 51b GO)

GO threshold: best new-universe RTMV Sharpe > 4-asset spy_rank_bull (0.8948)

Usage
-----
    uv run python scripts/run_phase52_universe_expansion.py
    uv run python scripts/run_phase52_universe_expansion.py --universes 4,5
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

# Universe definitions
UNIVERSES = {
    "4-asset": ["SPY", "GLD", "TLT", "IEF"],
    "5-asset": ["SPY", "GLD", "SHY", "IEF", "TLT"],
    "6-asset": ["SPY", "GLD", "SHY", "IEF", "TLT", "TIP"],
}

# 4-asset Phase 51b reference result (from compare_lambda_strategies.py run)
PHASE51B_REFERENCE = {
    "gmv":           None,   # will be computed
    "fixed_l05":     0.8838,
    "spy_rank_bull": 0.8948,
}


def _load_data(
    assets: list[str],
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
    }


def _run_variant(
    assets: list[str],
    asset_returns: pd.DataFrame,
    asset_features: dict[str, pd.DataFrame],
    lambda_tilt: float = 0.05,
    lambda_by_state_rank: list[float] | None = None,
    lambda_proxy_asset: str | None = None,
) -> dict:
    cfg = RTMVRebalancerConfig(
        assets=assets,
        lambda_tilt=lambda_tilt,
        lambda_by_state_rank=lambda_by_state_rank or [],
        lambda_proxy_asset=lambda_proxy_asset,
        n_states=3,
        n_restarts=3,
        lookback_bars=504,
        rebalance_bars=21,
        drawdown_halt=0.25,
    )
    rebalancer = RTMVRebalancer(cfg)
    snaps = rebalancer.run_backtest(asset_returns, asset_features)
    return _metrics(snaps)


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 52a: bond maturity ladder expansion")
    p.add_argument(
        "--universes", default="4,5,6",
        help="Which universe sizes to test: 4, 5, 6 (default: all three)",
    )
    args = p.parse_args()

    sizes = [s.strip() for s in args.universes.split(",")]
    universe_keys = [k for k in UNIVERSES if k.split("-")[0] in sizes]

    # Variant definitions: (label, kwargs for _run_variant)
    variants = [
        ("gmv",           dict(lambda_tilt=0.0)),
        ("fixed_l05",     dict(lambda_tilt=0.05)),
        ("spy_rank_bull", dict(
            lambda_tilt=0.05,
            lambda_by_state_rank=[0.02, 0.05, 0.10],
            lambda_proxy_asset="SPY",
        )),
    ]

    all_rows: list[dict] = []

    for universe_key in universe_keys:
        assets = UNIVERSES[universe_key]
        print(f"\nLoading {universe_key} ({', '.join(assets)})…", flush=True)
        asset_returns, asset_features = _load_data(assets)
        n_bars = len(asset_returns)
        date_from = asset_returns.index[0].date()
        date_to = asset_returns.index[-1].date()
        print(f"  {n_bars} bars  ({date_from} → {date_to})")

        for var_name, kwargs in variants:
            print(f"  Running {var_name}…", flush=True)
            m = _run_variant(assets, asset_returns, asset_features, **kwargs)
            m["universe"] = universe_key
            m["variant"] = var_name
            m["n_bars"] = n_bars
            all_rows.append(m)
            print(f"    Sharpe={m['Sharpe']}  MDD={m['MDD']}  Calmar={m['Calmar']}")

    # Print summary table
    print("\n" + "=" * 80)
    print("PHASE 52a — BOND LADDER UNIVERSE EXPANSION")
    print("=" * 80)
    cols = ["universe", "variant", "Sharpe", "Calmar", "MDD", "Ann Return", "Ann Vol", "Rebalances"]
    header = f"{'Universe':<12}{'Variant':<16}" + "".join(f"{c:>11}" for c in cols[2:])
    print(header)
    print("-" * 80)
    for r in all_rows:
        row = (
            f"{r['universe']:<12}"
            f"{r['variant']:<16}"
            + "".join(f"{str(r[c]):>11}" for c in cols[2:])
        )
        print(row)
    print("=" * 80)

    # Verdict
    baseline_sharpe = PHASE51B_REFERENCE["spy_rank_bull"]  # 0.8948
    new_universe_rows = [r for r in all_rows if r["universe"] != "4-asset"]
    if new_universe_rows:
        best = max(new_universe_rows, key=lambda r: float(r["Sharpe"]))
        delta = float(best["Sharpe"]) - baseline_sharpe
        print(f"\nBest new-universe result: {best['universe']} / {best['variant']}  "
              f"Sharpe={best['Sharpe']}")
        print(f"vs 4-asset spy_rank_bull baseline (0.8948): "
              f"{'+'if delta>=0 else ''}{delta:.4f}")
        verdict = "GO" if delta > 0.005 else "MARGINAL" if delta > 0.0 else "NO-GO"
        print(f"Verdict: {verdict}  (threshold: >+0.005 Sharpe vs 4-asset spy_rank_bull)")

    # Save
    out_dir = _repo_root / "results" / "phase52"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "universe_expansion.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")


if __name__ == "__main__":
    main()
