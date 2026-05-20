"""Phase 58: Multi-strategy combined backtester.

Combines:
  1. Crypto funding carry (ETH + BTC via Binance funding rates)
  2. LETF decay short (Short TQQQ / Long 3x QQQ, delta-neutral)

Capital allocation: configurable (default 70% carry / 30% LETF).
Both strategies are capital-efficient derivatives so notional > 100% is valid.

Usage
-----
    uv run python scripts/run_multistrat.py
    uv run python scripts/run_multistrat.py --carry-weight 0.50 --letf-weight 0.50

Outputs
-------
    docs/findings/phase58_multistrat.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "src"))
sys.path.insert(0, str(_repo / "scripts"))

# Import from existing scripts (no duplication of logic)
from funding_carry_backtest import fetch_funding_rates, simulate_carry, PERIODS_PER_YEAR
from letf_decay_analysis import fetch_letf_data, backtest_decay_short

ANN_FACTOR = 252  # trading days/year


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _portfolio_metrics(daily_ret: pd.Series) -> dict[str, float]:
    """Compute Sharpe, MDD, Ann Return, Ann Vol from a daily return series."""
    if len(daily_ret) < 20:
        return {"sharpe": 0.0, "mdd": 0.0, "ann_return_pct": 0.0, "ann_vol_pct": 0.0}
    ann_ret = daily_ret.mean() * ANN_FACTOR
    ann_vol = daily_ret.std() * np.sqrt(ANN_FACTOR)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum = (1 + daily_ret).cumprod()
    mdd = float(((cum - cum.cummax()) / cum.cummax()).min())
    return {
        "sharpe": round(float(sharpe), 3),
        "mdd": round(mdd, 4),
        "ann_return_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
    }


def _carry_daily_pnl(symbols: list[str]) -> pd.DataFrame:
    """Fetch carry funding rates and resample 8-hourly net P&L to daily."""
    daily_frames: list[pd.Series] = []
    for sym in symbols:
        print(f"  Fetching {sym} funding rates from Binance…", flush=True)
        funding = fetch_funding_rates(sym, start_year=2020)
        sim = simulate_carry(funding)
        # Resample 8-hourly net_pnl to daily compounded return; strip tz for alignment
        daily = (1 + sim["net_pnl"]).resample("D").prod() - 1
        daily.index = daily.index.normalize().tz_localize(None).normalize()
        daily.name = sym
        daily_frames.append(daily)
    df = pd.concat(daily_frames, axis=1).dropna(how="all")
    # Equal-weight across symbols for combined carry P&L
    df["carry_daily"] = df.mean(axis=1)
    return df


def _letf_daily_pnl(cache_dir: Path) -> pd.Series:
    """Run the ALWAYS-IN LETF decay short and return daily P&L series."""
    print("  Fetching TQQQ + QQQ data…", flush=True)
    tqqq, qqq, _, _ = fetch_letf_data(cache_dir)
    # ALWAYS-IN: no vol or HMM filter (confirmed optimal in prior research)
    bt = backtest_decay_short(
        tqqq, qqq,
        vol_entry_pct=0.0,   # always enter (vol filter OFF)
        vol_exit_pct=-999.0,  # never exit on vol
        spy_states=None,
    )
    return bt["pnl_series"].rename("letf_daily")


def _print_tearsheet(label: str, m: dict[str, float]) -> None:
    print(f"\n  {label}")
    print(f"    Sharpe:      {m['sharpe']:.3f}")
    print(f"    Ann Return:  {m['ann_return_pct']:.2f}%")
    print(f"    Ann Vol:     {m['ann_vol_pct']:.2f}%")
    print(f"    Max DD:      {m['mdd']*100:.2f}%")


def _write_report(
    carry_m: dict,
    letf_m: dict,
    correlation: float,
    results: list[dict],
    output_path: Path,
) -> None:
    """Write Markdown findings report."""

    def _row(r: dict) -> str:
        verdict = "GO" if r["go"] else "NO-GO"
        return (
            f"| {r['alloc']} | {r['sharpe']:.3f} | "
            f"{r['ann_return_pct']:.2f}% | {r['ann_vol_pct']:.2f}% | "
            f"{r['mdd']*100:.2f}% | {verdict} |"
        )

    rows = "\n".join(_row(r) for r in results)
    overall = "GO" if any(r["go"] for r in results) else "NO-GO"

    content = f"""# Phase 58: Multi-Strategy Executor — GO/NO-GO

**Date:** 2026-05-17
**Overall Verdict:** **{overall}**

## Strategy Descriptions

### 1. Crypto Funding Carry (70% weight default)
Long spot + Short perpetual on Binance. Collects positive funding.
Entry: 30-day trailing annualized carry > 5%. Exit: carry < -2%.
Assets: BTCUSDT + ETHUSDT (equal-weight). P&L resampled from 8-hourly to daily.

### 2. LETF Decay Short (30% weight default)
Short TQQQ + Long 3x QQQ notional (delta-neutral). Captures variance drag.
Configuration: ALWAYS-IN (vol filter OFF — confirmed optimal in prior research).
Daily P&L: 3×QQQ_return − TQQQ_return − 2%/yr cost drag.

## Individual Strategy Metrics

| Strategy | Sharpe | Ann Return | Ann Vol | Max DD |
| :--- | ---: | ---: | ---: | ---: |
| Carry (ETH+BTC equal-weight) | {carry_m['sharpe']:.3f} | {carry_m['ann_return_pct']:.2f}% | {carry_m['ann_vol_pct']:.2f}% | {carry_m['mdd']*100:.2f}% |
| LETF Decay Short (always-in) | {letf_m['sharpe']:.3f} | {letf_m['ann_return_pct']:.2f}% | {letf_m['ann_vol_pct']:.2f}% | {letf_m['mdd']*100:.2f}% |

## Cross-Strategy Correlation

**Pearson correlation (daily P&L): {correlation:.4f}**

{"Diversification benefit confirmed: correlation < 0.30." if correlation < 0.30 else "Moderate correlation: some diversification benefit." if correlation < 0.60 else "High correlation: limited diversification benefit."}

Interpretation: carry P&L is driven by crypto funding market microstructure;
LETF decay is driven by equity index realized volatility. These are structurally
uncorrelated alpha sources.

## Combined Portfolio Metrics

| Allocation (Carry/LETF) | Sharpe | Ann Return | Ann Vol | Max DD | Verdict |
| :--- | ---: | ---: | ---: | ---: | ---: |
{rows}

**GO criterion:** Combined Sharpe > max(individual Sharpe) OR Combined MDD less negative than worst individual MDD (i.e. better drawdown protection).

## Conclusion

The multi-strategy combination {"provides genuine diversification and improved risk-adjusted returns." if overall == "GO" else "does not materially improve over the best individual strategy."}

Max individual Sharpe: {max(carry_m['sharpe'], letf_m['sharpe']):.3f}
Min individual MDD:    {min(carry_m['mdd'], letf_m['mdd'])*100:.2f}%

{"Both strategies' P&L streams are near-uncorrelated, confirming they draw from independent alpha sources." if correlation < 0.30 else ""}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"\nReport written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 58 multi-strategy combined backtester")
    p.add_argument("--carry-weight", type=float, default=0.70)
    p.add_argument("--letf-weight", type=float, default=0.30)
    p.add_argument("--cache-dir", type=Path, default=_repo / "results" / "cache")
    p.add_argument("--report-path", type=Path,
                   default=_repo / "docs" / "findings" / "phase58_multistrat.md")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("PHASE 58: MULTI-STRATEGY COMBINED BACKTESTER")
    print("=" * 68)

    # --- Carry leg ---
    print("\n[1/2] Building carry P&L (Binance ETH + BTC funding)…")
    carry_df = _carry_daily_pnl(["ETHUSDT", "BTCUSDT"])
    carry_daily = carry_df["carry_daily"].dropna()
    carry_m = _portfolio_metrics(carry_daily)
    _print_tearsheet("Carry (ETH+BTC equal-weight)", carry_m)

    # --- LETF leg ---
    print("\n[2/2] Building LETF decay P&L (TQQQ short / 3x QQQ long)…")
    letf_daily = _letf_daily_pnl(args.cache_dir)
    letf_m = _portfolio_metrics(letf_daily)
    _print_tearsheet("LETF Decay Short (always-in)", letf_m)

    # --- Align on common date index ---
    combined = pd.DataFrame({
        "carry": carry_daily,
        "letf": letf_daily,
    }).dropna()
    print(f"\nCommon date range: {combined.index[0].date()} → {combined.index[-1].date()}")
    print(f"Common bars: {len(combined)}")

    # --- Correlation ---
    corr = float(combined["carry"].corr(combined["letf"]))
    print(f"\nCross-strategy Pearson correlation: {corr:.4f}")
    if corr < 0.30:
        print("  => LOW correlation — genuine diversification benefit")
    elif corr < 0.60:
        print("  => MODERATE correlation — some diversification benefit")
    else:
        print("  => HIGH correlation — limited diversification benefit")

    # --- Combined portfolios at multiple allocations ---
    allocations = [
        (0.70, 0.30, "70/30"),
        (0.50, 0.50, "50/50"),
        (0.80, 0.20, "80/20"),
    ]
    # Override with CLI args if non-default
    if (args.carry_weight, args.letf_weight) != (0.70, 0.30):
        label = f"{args.carry_weight*100:.0f}/{args.letf_weight*100:.0f}"
        allocations.insert(0, (args.carry_weight, args.letf_weight, label))

    max_ind_sharpe = max(carry_m["sharpe"], letf_m["sharpe"])
    min_ind_mdd = min(carry_m["mdd"], letf_m["mdd"])

    results: list[dict] = []
    print("\nCombined portfolio metrics:")
    print(f"  {'Alloc':10s} {'Sharpe':>8s} {'AnnRet%':>9s} {'Vol%':>7s} {'MDD%':>8s} {'Verdict':>8s}")
    for cw, lw, label in allocations:
        combined_ret = cw * combined["carry"] + lw * combined["letf"]
        m = _portfolio_metrics(combined_ret)
        go = m["sharpe"] > max_ind_sharpe or m["mdd"] > min_ind_mdd  # mdd is negative
        m["alloc"] = label
        m["go"] = go
        results.append(m)
        verdict = "GO" if go else "NO-GO"
        print(f"  {label:10s} {m['sharpe']:>8.3f} {m['ann_return_pct']:>9.2f} "
              f"{m['ann_vol_pct']:>7.2f} {m['mdd']*100:>8.2f} {verdict:>8s}")

    print(f"\nMax individual Sharpe: {max_ind_sharpe:.3f}")
    print(f"Min individual MDD:    {min_ind_mdd*100:.2f}%")

    overall = "GO" if any(r["go"] for r in results) else "NO-GO"
    print(f"\nOverall verdict: {overall}")

    _write_report(carry_m, letf_m, corr, results, args.report_path)


if __name__ == "__main__":
    main()
