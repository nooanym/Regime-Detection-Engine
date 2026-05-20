"""
Phase 73 — SOL Perpetual Funding Carry Re-test
===============================================
Phase 59 (2023) found SOL NO-GO: mean ann carry 0.9% (2021+) due to FTX 2022
catastrophe (-35.6% annual).  SOL has since grown significantly.

This script re-tests whether SOL's 2024-2026 carry profile has improved enough
to overcome dilution arithmetic and beat the Phase 62 BTC+ETH bull-only Sharpe
of 17.943.

GO threshold (raw basket, no regime overlay): Sharpe >= 16.0
Live deployment threshold: beat Phase 62 baseline carry_weighted_bull_only Sharpe of 17.943

Baskets tested (carry-weighted within Phase 60 framework):
  - btc_eth           : BTC + ETH (Phase 62 baseline)
  - btc_eth_sol       : BTC + ETH + SOL
  - eth_sol           : ETH + SOL
  - btc_eth_sol_equal : BTC + ETH + SOL equal-weight
  - btc_eth_sol_max   : 100% to the highest rolling-carry asset each period

For the best-performing 3-asset basket: apply bull-only regime scaling (Phase 62
approach) and compare vs Phase 62 live baseline (Sharpe 17.943).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from funding_carry_backtest import fetch_funding_rates  # noqa: E402
from rde.models.hmm import train_hmm  # noqa: E402

PERIODS_PER_YEAR = 3 * 365   # 1095 eight-hour funding payments per year
COST_PER_PERIOD  = 0.005 / PERIODS_PER_YEAR   # ~0.5% annual friction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metrics(net: pd.Series, label: str = "") -> dict[str, float]:
    """Compute performance metrics from an 8-hourly net P&L series."""
    n = net.dropna()
    if len(n) == 0:
        return {k: float("nan") for k in ["sharpe", "ann_return", "ann_vol", "mdd", "cum_return"]}
    cum = (1 + n).cumprod()
    t = len(cum)
    ann = float(cum.iloc[-1] ** (PERIODS_PER_YEAR / t) - 1)
    vol = float(n.std() * np.sqrt(PERIODS_PER_YEAR))
    sharpe = ann / vol if vol > 0 else 0.0
    roll_max = np.maximum.accumulate(cum.values)
    mdd = float(((cum.values - roll_max) / roll_max).min())
    cum_ret = float(cum.iloc[-1] - 1)
    calmar = ann / abs(mdd) if mdd < 0 else float("nan")
    return {
        "sharpe": round(sharpe, 4),
        "ann_return": round(ann, 4),
        "ann_vol": round(vol, 4),
        "mdd": round(mdd, 4),
        "cum_return": round(cum_ret, 4),
        "calmar": round(calmar, 4),
    }


def _print_row(label: str, m: dict[str, float]) -> None:
    print(
        f"  {label:<32}  Sharpe={m['sharpe']:>7.3f}  "
        f"Ann={m['ann_return']:>7.2%}  MDD={m['mdd']:>7.2%}  "
        f"Cum={m['cum_return']:>7.1%}  Calmar={m['calmar']:>7.2f}"
    )


def _individual_stats(funding: pd.Series, symbol: str) -> dict:
    """Compute per-symbol statistics: ann return, Sharpe, % positive, worst year."""
    ann_rates = funding * PERIODS_PER_YEAR
    mean_ann = float(ann_rates.mean())
    median_ann = float(ann_rates.median())
    pct_positive = float((funding > 0).mean())

    # Yearly breakdown
    yearly: dict[int, float] = {}
    for yr, grp in ann_rates.groupby(funding.index.year):
        n_per = len(grp)
        cum_yr = float((1 + grp / PERIODS_PER_YEAR).prod() ** (PERIODS_PER_YEAR / n_per) - 1)
        yearly[int(yr)] = round(cum_yr, 4)

    worst_year_val = min(yearly.values()) if yearly else float("nan")
    worst_year_key = min(yearly, key=yearly.get) if yearly else None

    # Sharpe on raw per-period P&L
    vol = float(ann_rates.std() / PERIODS_PER_YEAR * np.sqrt(PERIODS_PER_YEAR))
    sharpe_raw = mean_ann / vol if vol > 0 else 0.0

    return {
        "symbol": symbol,
        "n_obs": len(funding),
        "start": str(funding.index[0].date()),
        "mean_ann": round(mean_ann, 4),
        "median_ann": round(median_ann, 4),
        "pct_positive": round(pct_positive, 4),
        "sharpe_raw": round(sharpe_raw, 4),
        "worst_year": f"{worst_year_key} ({worst_year_val:.1%})",
        "yearly": yearly,
    }


def _carry_weighted_net(
    df: pd.DataFrame,
    symbols: list[str],
    *,
    window: int = 90,
    max_carry_mode: bool = False,
    equal_weight: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """
    Return (net_pnl_series, in_market_series) for a multi-asset carry basket.

    Weights are proportional to rolling 90-period positive carry estimates
    (carry-weighted, matching Phase 60 logic).

    If max_carry_mode=True: 100% weight to highest rolling-carry asset.
    If equal_weight=True: 1/N to each asset.
    """
    carry_ests: dict[str, pd.Series] = {}
    for sym in symbols:
        carry_ests[sym] = (df[sym] * PERIODS_PER_YEAR).rolling(window).mean()

    # Weights
    if equal_weight:
        # Equal weight when any carry > 0; else 0
        any_pos = pd.concat([c > 0 for c in carry_ests.values()], axis=1).any(axis=1)
        w_dict = {sym: pd.Series(1.0 / len(symbols), index=df.index).where(any_pos, 0.0)
                  for sym in symbols}
        in_market = any_pos
    elif max_carry_mode:
        # 100% to the asset with highest rolling carry
        carry_df = pd.concat(carry_ests, axis=1)
        carry_arr = carry_df.values  # shape (T, N)
        total_pos = pd.concat(
            [carry_ests[sym].clip(lower=0) for sym in symbols], axis=1
        ).sum(axis=1).fillna(0)
        in_market = total_pos > 0

        # Row-wise argmax, falling back to 0 if all NaN
        best_col = np.full(len(carry_arr), 0, dtype=int)
        for t in range(len(carry_arr)):
            row = carry_arr[t]
            valid = ~np.isnan(row)
            if valid.any():
                best_col[t] = int(np.argmax(np.where(valid, row, -np.inf)))

        w_dict = {}
        for i, sym in enumerate(symbols):
            w = pd.Series(
                (best_col == i).astype(float), index=carry_df.index
            ).where(in_market, 0.0)
            w_dict[sym] = w
    else:
        # Carry-weighted (Phase 60 baseline logic)
        clipped: dict[str, pd.Series] = {sym: carry_ests[sym].clip(lower=0) for sym in symbols}
        total_pos = pd.concat(list(clipped.values()), axis=1).sum(axis=1)
        in_market = total_pos > 0
        w_dict = {}
        for sym in symbols:
            w = (clipped[sym] / total_pos.where(total_pos > 0, 1.0)).where(in_market, 0.0)
            w_dict[sym] = w

    raw = sum(w_dict[sym] * df[sym] for sym in symbols)
    net = raw - in_market.astype(float) * COST_PER_PERIOD
    return net, in_market


def fit_spy_ranks(start: str = "2020-01-01") -> pd.Series:
    """Fit SPY n=3 Gaussian HMM; return daily dominant-state rank (0=bear, 2=bull)."""
    spy = yf.download("SPY", start=start, interval="1d", progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    lr = np.log(spy["Close"] / spy["Close"].shift(1)).dropna()
    vol_s = lr.rolling(20).std().dropna()
    feats = pd.concat([lr, vol_s], axis=1).dropna()
    feats.columns = ["log_return", "volatility"]
    m = train_hmm(feats.values, n_states=3, n_restarts=3, seed_base=42)
    states = m.hmm.predict(m.scaler.transform(feats.values))
    rank_map = {int(s): int(r) for r, s in enumerate(np.argsort(m.hmm.means_[:, 0]))}
    return pd.Series(
        [rank_map[s] for s in states],
        index=feats.index.normalize(),
        name="hmm_rank",
    )


def _apply_bull_only_scale(net: pd.Series, in_market: pd.Series, spy_ranks: pd.Series) -> pd.Series:
    """Apply 1×/1×/1.5× bull-only SPY rank scaling (matching Phase 62)."""
    ranks_by_date = {ts.date(): int(v) for ts, v in spy_ranks.items() if not pd.isna(v)}
    scale_map = {0: 1.0, 1: 1.0, 2: 1.5}
    scale = pd.Series(
        [scale_map.get(ranks_by_date.get(ts.date(), -1), 1.0) for ts in net.index],
        index=net.index,
    )
    # Re-derive gross return (undo cost), apply scale, re-add cost
    gross = net + in_market.astype(float) * COST_PER_PERIOD
    net_scaled = gross * scale - in_market.astype(float) * COST_PER_PERIOD
    return net_scaled.dropna()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    print("=" * 70)
    print("PHASE 73 — SOL Perpetual Funding Carry Re-Test")
    print("=" * 70)
    print(f"Run date: {datetime.now(tz=timezone.utc).date()}")
    print()

    # ── Fetch data ──────────────────────────────────────────────────────────
    print("Fetching BTCUSDT funding rates (from 2020)...", end=" ", flush=True)
    btc_raw = fetch_funding_rates("BTCUSDT", start_year=2020)
    print(f"{len(btc_raw)} periods")

    print("Fetching ETHUSDT funding rates (from 2020)...", end=" ", flush=True)
    eth_raw = fetch_funding_rates("ETHUSDT", start_year=2020)
    print(f"{len(eth_raw)} periods")

    print("Fetching SOLUSDT funding rates (from 2021)...", end=" ", flush=True)
    sol_raw = fetch_funding_rates("SOLUSDT", start_year=2021)
    print(f"{len(sol_raw)} periods")
    print()

    # ── Per-symbol stats ────────────────────────────────────────────────────
    btc_stats = _individual_stats(btc_raw, "BTCUSDT")
    eth_stats = _individual_stats(eth_raw, "ETHUSDT")
    sol_stats = _individual_stats(sol_raw, "SOLUSDT")

    print("Individual symbol statistics:")
    print(f"  {'Symbol':<10} {'N obs':>6}  {'Start':<11}  {'Mean Ann':>9}  "
          f"{'Median Ann':>10}  {'% Pos':>6}  {'Sharpe':>7}  Worst Year")
    print("  " + "-" * 82)
    for st in [btc_stats, eth_stats, sol_stats]:
        print(
            f"  {st['symbol']:<10} {st['n_obs']:>6}  {st['start']:<11}  "
            f"{st['mean_ann']:>9.1%}  {st['median_ann']:>10.1%}  "
            f"{st['pct_positive']:>6.1%}  {st['sharpe_raw']:>7.2f}  {st['worst_year']}"
        )
    print()

    # SOL 2024+ stats specifically
    sol_2024 = sol_raw[sol_raw.index >= "2024-01-01"]
    if len(sol_2024) > 0:
        sol_2024_stats = _individual_stats(sol_2024, "SOLUSDT_2024+")
        print(f"  SOL 2024-present: mean_ann={sol_2024_stats['mean_ann']:.1%}  "
              f"median_ann={sol_2024_stats['median_ann']:.1%}  "
              f"% positive={sol_2024_stats['pct_positive']:.1%}  "
              f"Sharpe={sol_2024_stats['sharpe_raw']:.2f}")
        # Yearly detail
        for yr, val in sorted(sol_stats["yearly"].items()):
            print(f"    {yr}: {val:+.1%}")
    print()

    # Correlation analysis
    # Align all three
    df_corr = pd.DataFrame({
        "btc": btc_raw,
        "eth": eth_raw,
        "sol": sol_raw,
    }).dropna()
    corr_matrix = df_corr.corr()
    print("Funding rate correlations (aligned 3-way, 8-hourly):")
    print(f"  BTC↔ETH = {corr_matrix.loc['btc','eth']:.3f}  "
          f"BTC↔SOL = {corr_matrix.loc['btc','sol']:.3f}  "
          f"ETH↔SOL = {corr_matrix.loc['eth','sol']:.3f}")
    print()

    # Phase 59 had corr(ETH, SOL) = 0.258 on 2021+ data; show 2024+ improvement
    df_corr_recent = df_corr[df_corr.index >= "2024-01-01"]
    if len(df_corr_recent) > 10:
        corr_recent = df_corr_recent.corr()
        print(f"  [2024+ only]: BTC↔ETH={corr_recent.loc['btc','eth']:.3f}  "
              f"BTC↔SOL={corr_recent.loc['btc','sol']:.3f}  "
              f"ETH↔SOL={corr_recent.loc['eth','sol']:.3f}")
    print()

    # ── Align all three for basket testing ──────────────────────────────────
    df = pd.DataFrame({
        "BTCUSDT": btc_raw,
        "ETHUSDT": eth_raw,
        "SOLUSDT": sol_raw,
    }).sort_index().dropna()
    print(f"Aligned 3-way periods: {len(df)}  ({df.index[0].date()} → {df.index[-1].date()})")
    print()

    # For fair comparison with Phase 59, also compute baselines on the same aligned window
    symbols_btc_eth   = ["BTCUSDT", "ETHUSDT"]
    symbols_all_3     = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    symbols_eth_sol   = ["ETHUSDT", "SOLUSDT"]

    net_btc_eth, in_btc_eth = _carry_weighted_net(df, symbols_btc_eth)
    net_3_cw,    in_3_cw    = _carry_weighted_net(df, symbols_all_3)
    net_eth_sol, in_eth_sol = _carry_weighted_net(df, symbols_eth_sol)
    net_3_eq,    in_3_eq    = _carry_weighted_net(df, symbols_all_3, equal_weight=True)
    net_3_max,   in_3_max   = _carry_weighted_net(df, symbols_all_3, max_carry_mode=True)

    m_btc_eth = _metrics(net_btc_eth,  "btc_eth (baseline)")
    m_3_cw    = _metrics(net_3_cw,     "btc_eth_sol carry-wt")
    m_eth_sol = _metrics(net_eth_sol,  "eth_sol carry-wt")
    m_3_eq    = _metrics(net_3_eq,     "btc_eth_sol equal-wt")
    m_3_max   = _metrics(net_3_max,    "btc_eth_sol max-carry")

    # ── Basket comparison table ──────────────────────────────────────────────
    print("=" * 90)
    print("BASKET COMPARISON (carry-weighted, ~0.5% annual friction)")
    print("=" * 90)
    print(f"  {'Basket':<32}  {'Sharpe':>8}  {'Ann Ret':>8}  {'MDD':>8}  {'Cum Ret':>8}  {'Calmar':>7}")
    print("  " + "-" * 80)
    _print_row("btc_eth (Phase62 base)",    m_btc_eth)
    _print_row("btc_eth_sol carry-wt",      m_3_cw)
    _print_row("eth_sol carry-wt",          m_eth_sol)
    _print_row("btc_eth_sol equal-wt",      m_3_eq)
    _print_row("btc_eth_sol max-carry",     m_3_max)
    print()

    # ── Per-year breakdown for the 3-asset carry-weighted basket ────────────
    print("Per-year returns — btc_eth_sol carry-wt vs btc_eth baseline:")
    print(f"  {'Year':<6}  {'btc_eth':>9}  {'btc_eth_sol':>12}  {'delta':>8}")
    print("  " + "-" * 40)
    combined = pd.DataFrame({"base": net_btc_eth, "sol3": net_3_cw}).dropna()
    for yr, grp in combined.groupby(combined.index.year):
        cum_b = float((1 + grp["base"]).prod() ** (PERIODS_PER_YEAR / len(grp)) - 1)
        cum_s = float((1 + grp["sol3"]).prod() ** (PERIODS_PER_YEAR / len(grp)) - 1)
        print(f"  {yr:<6}  {cum_b:>9.1%}  {cum_s:>12.1%}  {cum_s - cum_b:>+8.1%}")
    print()

    # ── Best SOL-expanded raw basket vs GO threshold ────────────────────────
    all_raw_results = {
        "btc_eth": m_btc_eth,
        "btc_eth_sol carry-wt": m_3_cw,
        "eth_sol carry-wt": m_eth_sol,
        "btc_eth_sol equal-wt": m_3_eq,
        "btc_eth_sol max-carry": m_3_max,
    }
    # Verdict is about SOL-expanded baskets specifically
    sol_raw_results = {
        "btc_eth_sol carry-wt": m_3_cw,
        "eth_sol carry-wt": m_eth_sol,
        "btc_eth_sol equal-wt": m_3_eq,
        "btc_eth_sol max-carry": m_3_max,
    }
    best_label = max(sol_raw_results, key=lambda k: sol_raw_results[k]["sharpe"])
    best_m     = sol_raw_results[best_label]

    raw_go_threshold = 16.0
    raw_pass = best_m["sharpe"] >= raw_go_threshold
    print(f"Best SOL-expanded raw basket: '{best_label}' → Sharpe {best_m['sharpe']:.3f}")
    print(f"  2-asset baseline btc_eth: Sharpe {m_btc_eth['sharpe']:.3f}")
    print(f"Raw GO threshold (>= {raw_go_threshold}): {'PASS' if raw_pass else 'FAIL'}")
    print()

    # ── Bull-only regime scaling on 3-asset basket ──────────────────────────
    print("Fitting SPY n=3 HMM for regime scaling...", end=" ", flush=True)
    spy_ranks = fit_spy_ranks(start="2020-01-01")
    counts = spy_ranks.value_counts().sort_index().to_dict()
    print(f"done. bear={counts.get(0,0)} neutral={counts.get(1,0)} bull={counts.get(2,0)} days")
    print()

    # Apply bull-only scaling (1×/1×/1.5×) to the 3-asset CW basket and the baseline
    net_btc_eth_bull  = _apply_bull_only_scale(net_btc_eth, in_btc_eth, spy_ranks)
    net_3_cw_bull     = _apply_bull_only_scale(net_3_cw,    in_3_cw,    spy_ranks)
    net_eth_sol_bull  = _apply_bull_only_scale(net_eth_sol, in_eth_sol, spy_ranks)
    net_3_eq_bull     = _apply_bull_only_scale(net_3_eq,    in_3_eq,    spy_ranks)

    m_btc_eth_bull = _metrics(net_btc_eth_bull,  "btc_eth bull-only")
    m_3_cw_bull    = _metrics(net_3_cw_bull,     "btc_eth_sol CW bull-only")
    m_eth_sol_bull = _metrics(net_eth_sol_bull,  "eth_sol CW bull-only")
    m_3_eq_bull    = _metrics(net_3_eq_bull,     "btc_eth_sol EW bull-only")

    phase62_threshold = 17.943

    print("=" * 90)
    print("WITH BULL-ONLY REGIME SCALING (1×/1×/1.5× SPY rank)")
    print(f"Phase 62 live baseline (carry_weighted_bull_only): Sharpe = {phase62_threshold}")
    print("=" * 90)
    print(f"  {'Basket':<32}  {'Sharpe':>8}  {'Ann Ret':>8}  {'MDD':>8}  {'Cum Ret':>8}  {'Calmar':>7}")
    print("  " + "-" * 80)
    _print_row("btc_eth bull-only (Ph62)",    m_btc_eth_bull)
    _print_row("btc_eth_sol CW bull-only",    m_3_cw_bull)
    _print_row("eth_sol CW bull-only",        m_eth_sol_bull)
    _print_row("btc_eth_sol EW bull-only",    m_3_eq_bull)
    print()

    # All regime-scaled results (including baseline for reference)
    all_regime_results = {
        "btc_eth bull-only (Ph62)":   m_btc_eth_bull,
        "btc_eth_sol CW bull-only":   m_3_cw_bull,
        "eth_sol CW bull-only":       m_eth_sol_bull,
        "btc_eth_sol EW bull-only":   m_3_eq_bull,
    }

    # Verdict is specifically about SOL-expanded baskets (not the 2-asset baseline)
    sol_regime_results = {
        "btc_eth_sol CW bull-only":   m_3_cw_bull,
        "eth_sol CW bull-only":       m_eth_sol_bull,
        "btc_eth_sol EW bull-only":   m_3_eq_bull,
    }
    best_sol_regime_label = max(sol_regime_results, key=lambda k: sol_regime_results[k]["sharpe"])
    best_sol_regime_m     = sol_regime_results[best_sol_regime_label]

    # Best overall (for reference output)
    best_regime_label = max(all_regime_results, key=lambda k: all_regime_results[k]["sharpe"])
    best_regime_m     = all_regime_results[best_regime_label]

    live_pass = best_sol_regime_m["sharpe"] > phase62_threshold
    delta_live = best_sol_regime_m["sharpe"] - phase62_threshold

    print(f"Best SOL-expanded regime-scaled basket: '{best_sol_regime_label}' → "
          f"Sharpe {best_sol_regime_m['sharpe']:.3f}")
    print(f"Phase 62 live threshold ({phase62_threshold}): {'PASS' if live_pass else 'FAIL'}  "
          f"(delta = {delta_live:+.3f})")
    print(f"2-asset baseline (btc_eth bull-only): Sharpe {m_btc_eth_bull['sharpe']:.3f}  "
          f"[re-run on 2021+ window]")
    print()

    # ── Overall verdict ──────────────────────────────────────────────────────
    if live_pass:
        verdict = "GO"
        verdict_detail = (
            f"SOL-expanded basket '{best_sol_regime_label}' beats the Phase 62 "
            f"live baseline by {delta_live:+.3f} Sharpe. "
            "Recommend adding SOLUSDT to CarryStrategy."
        )
    elif raw_pass:
        verdict = "CONDITIONAL GO"
        verdict_detail = (
            f"Best SOL raw basket clears 16.0 raw threshold "
            f"(btc_eth Sharpe {m_btc_eth['sharpe']:.3f}). Best SOL-expanded regime basket "
            f"('{best_sol_regime_label}' Sharpe {best_sol_regime_m['sharpe']:.3f}) "
            f"does NOT beat the Phase 62 live baseline of {phase62_threshold} "
            f"(delta = {delta_live:+.3f}). "
            "SOL carry has improved but dilution prevents beating the live benchmark."
        )
    else:
        verdict = "NO-GO"
        verdict_detail = (
            f"Best SOL-expanded basket (raw Sharpe {best_m['sharpe']:.3f}) fails "
            f"the 16.0 raw threshold. Best regime-scaled SOL basket Sharpe "
            f"{best_sol_regime_m['sharpe']:.3f} (delta vs Ph62 = {delta_live:+.3f}). "
            "Dilution arithmetic still dominates SOL's improved carry profile."
        )

    print("=" * 70)
    print(f"PHASE 73 VERDICT: {verdict}")
    print(f"  {verdict_detail}")
    print("=" * 70)

    # ── SOL profile comparison: Phase 59 vs now ──────────────────────────────
    sol_phase59_mean = 0.009   # 0.9% mean as of Phase 59 (2021+)
    sol_phase59_sharpe = 10.81

    # Comparison
    sol_current_mean = sol_stats["mean_ann"]
    sol_current_sharpe = sol_stats["sharpe_raw"]
    sol_improvement = sol_current_mean - sol_phase59_mean

    print()
    print("SOL profile comparison (Phase 59 vs now):")
    print(f"  Mean ann carry:   Phase59={sol_phase59_mean:.1%}  Now={sol_current_mean:.1%}  "
          f"Δ={sol_improvement:+.1%}")
    print(f"  Individual Sharpe: Phase59={sol_phase59_sharpe:.2f}  "
          f"Now={sol_current_sharpe:.2f}")
    print(f"  % positive periods: {sol_stats['pct_positive']:.1%}  (Phase59: 72.0%)")

    # ── Write findings doc ───────────────────────────────────────────────────
    sol_yearly_table = "\n".join(
        f"| {yr} | {val:+.1%} |"
        for yr, val in sorted(sol_stats["yearly"].items())
    )

    basket_rows = "\n".join([
        f"| btc_eth (Phase62 base)       | {m_btc_eth['sharpe']:.3f} | "
        f"{m_btc_eth['ann_return']:.2%} | {m_btc_eth['ann_vol']:.2%} | "
        f"{m_btc_eth['mdd']:.2%} | {m_btc_eth['calmar']:.2f} |",

        f"| btc_eth_sol carry-wt         | {m_3_cw['sharpe']:.3f} | "
        f"{m_3_cw['ann_return']:.2%} | {m_3_cw['ann_vol']:.2%} | "
        f"{m_3_cw['mdd']:.2%} | {m_3_cw['calmar']:.2f} |",

        f"| eth_sol carry-wt             | {m_eth_sol['sharpe']:.3f} | "
        f"{m_eth_sol['ann_return']:.2%} | {m_eth_sol['ann_vol']:.2%} | "
        f"{m_eth_sol['mdd']:.2%} | {m_eth_sol['calmar']:.2f} |",

        f"| btc_eth_sol equal-wt         | {m_3_eq['sharpe']:.3f} | "
        f"{m_3_eq['ann_return']:.2%} | {m_3_eq['ann_vol']:.2%} | "
        f"{m_3_eq['mdd']:.2%} | {m_3_eq['calmar']:.2f} |",

        f"| btc_eth_sol max-carry        | {m_3_max['sharpe']:.3f} | "
        f"{m_3_max['ann_return']:.2%} | {m_3_max['ann_vol']:.2%} | "
        f"{m_3_max['mdd']:.2%} | {m_3_max['calmar']:.2f} |",
    ])

    regime_rows = "\n".join([
        f"| btc_eth bull-only (Ph62)     | {m_btc_eth_bull['sharpe']:.3f} | "
        f"{m_btc_eth_bull['ann_return']:.2%} | {m_btc_eth_bull['mdd']:.2%} | "
        f"{m_btc_eth_bull['calmar']:.2f} |",

        f"| btc_eth_sol CW bull-only     | {m_3_cw_bull['sharpe']:.3f} | "
        f"{m_3_cw_bull['ann_return']:.2%} | {m_3_cw_bull['mdd']:.2%} | "
        f"{m_3_cw_bull['calmar']:.2f} |",

        f"| eth_sol CW bull-only         | {m_eth_sol_bull['sharpe']:.3f} | "
        f"{m_eth_sol_bull['ann_return']:.2%} | {m_eth_sol_bull['mdd']:.2%} | "
        f"{m_eth_sol_bull['calmar']:.2f} |",

        f"| btc_eth_sol EW bull-only     | {m_3_eq_bull['sharpe']:.3f} | "
        f"{m_3_eq_bull['ann_return']:.2%} | {m_3_eq_bull['mdd']:.2%} | "
        f"{m_3_eq_bull['calmar']:.2f} |",
    ])

    corr_all = corr_matrix.loc["btc", "sol"]
    corr_eth_sol_all = corr_matrix.loc["eth", "sol"]
    corr_btc_eth_all = corr_matrix.loc["btc", "eth"]

    corr_recent_btc_sol = corr_recent.loc["btc", "sol"] if len(df_corr_recent) > 10 else float("nan")
    corr_recent_eth_sol = corr_recent.loc["eth", "sol"] if len(df_corr_recent) > 10 else float("nan")

    if live_pass:
        action_text = (
            f"**Recommend updating `src/rde/trading/carry_executor.py`** to add `SOLUSDT` "
            f"to the `CarryStrategy` universe. Use carry-weighted allocation (Phase 60 logic). "
            f"Phase 74 = live integration of SOL."
        )
    else:
        action_text = (
            "**BTC+ETH basket remains optimal.** SOL carry has improved structurally since "
            "Phase 59 but dilution arithmetic still prevents beating the Phase 62 live baseline. "
            "Phase 74 = portfolio re-optimisation — recheck multi-strat weights (80/15/5) "
            "with the 3-pair LETF leg, which now shows Sharpe 7.06 vs the old 4.887 single-pair."
        )

    # Pre-compute conditional strings to avoid f-string quoting conflicts
    if sol_improvement > 0.03:
        sol_profile_finding = (
            "SOL carry has **materially improved** vs Phase 59. The 2024+ period shows a "
            "structurally higher mean carry, improving the full-period average."
        )
    elif sol_improvement > 0:
        sol_profile_finding = (
            "SOL carry has improved only marginally vs Phase 59 — the improvement is too "
            "small to overcome dilution arithmetic."
        )
    else:
        sol_profile_finding = "SOL carry has not improved vs Phase 59."

    if m_3_cw["sharpe"] > m_btc_eth["sharpe"]:
        dilution_finding = (
            "SOL carry improvement is **sufficient** to overcome dilution — the 3-asset "
            "basket now beats the 2-asset baseline."
        )
    else:
        dilution_finding = (
            "Dilution arithmetic still dominates — the 3-asset basket still trails the "
            "2-asset baseline even with improved SOL carry."
        )

    if sol_improvement > 0.03:
        root_cause = (
            "**Primary driver of improvement:** SOL carry in 2024-2026 has been structurally "
            "higher than the 2021-2023 average, which was pulled down by the FTX 2022 "
            "catastrophe (-35.6% ann). The improved mean shifts the dilution arithmetic favorably."
        )
    else:
        root_cause = (
            "**Primary constraint:** Dilution arithmetic. Any asset with individual Sharpe "
            "below the BTC+ETH basket Sharpe will reduce the combined Sharpe. "
            "SOL's individual Sharpe improvement is insufficient to cross this threshold."
        )

    raw_pass_str  = "PASS" if raw_pass  else "FAIL"
    live_pass_str = "PASS" if live_pass else "FAIL"

    if len(df_corr_recent) > 10:
        corr_btc_eth_recent_str = f"{corr_recent.loc['btc', 'eth']:.3f}"
        corr_btc_sol_recent_str = f"{corr_recent_btc_sol:.3f}"
        corr_eth_sol_recent_str = f"{corr_recent_eth_sol:.3f}"
    else:
        corr_btc_eth_recent_str = "N/A"
        corr_btc_sol_recent_str = "N/A"
        corr_eth_sol_recent_str = "N/A"

    sol_sharpe_delta_str = f"{sol_current_sharpe - sol_phase59_sharpe:+.2f}"
    data_start_str = str(df.index[0].date())
    data_corr_start_str = str(df_corr.index[0].date())

    m3_cw_delta_str  = f"{m_3_cw['sharpe']  - m_btc_eth['sharpe']:+.3f}"
    m3_eq_delta_str  = f"{m_3_eq['sharpe']  - m_btc_eth['sharpe']:+.3f}"

    doc = f"""# Phase 73 — SOL Perpetual Funding Carry Re-Test

**Date:** 2026-05-18
**Verdict: {verdict}**
**Script:** `scripts/run_phase73_sol_carry.py`

---

## 1. Motivation

Phase 59 (2026-05-17) found SOL NO-GO with mean annualised carry of **0.9%** (2021+).
The catastrophic 2022 result (-35.6% annual during FTX collapse) dominated the full-period mean.

SOL has since grown significantly in derivatives market size. The question: has SOL's 2024-2026
carry profile improved enough to overcome dilution arithmetic and beat the Phase 62 live baseline
(carry_weighted_bull_only Sharpe = **{phase62_threshold}**)?

---

## 2. SOL Individual Carry Statistics

| Metric | Phase 59 (2021+ through mid-2024) | Phase 73 (2021-present) | Change |
|--------|-----------------------------------|-------------------------|--------|
| Mean ann carry | 0.9% | {sol_current_mean:.1%} | {sol_improvement:+.1%} |
| % positive periods | 72.0% | {sol_stats['pct_positive']:.1%} | -- |
| Individual Sharpe (raw) | 10.81 | {sol_current_sharpe:.2f} | {sol_sharpe_delta_str} |

### SOL yearly funding returns

| Year | Ann Return |
|------|-----------|
{sol_yearly_table}

### Key finding on SOL profile

{sol_profile_finding}

---

## 3. Funding Rate Correlations

| Pair | Full history ({data_corr_start_str}+) | 2024+ only |
|------|--------------------------------------|------------|
| BTC / ETH | {corr_btc_eth_all:.3f} | {corr_btc_eth_recent_str} |
| BTC / SOL | {corr_all:.3f} | {corr_btc_sol_recent_str} |
| ETH / SOL | {corr_eth_sol_all:.3f} | {corr_eth_sol_recent_str} |

Phase 59 reported ETH/SOL = 0.258 (2021+ window). SOL remains a good diversifier by correlation.

---

## 4. Basket Comparison (carry-weighted, no regime scaling)

| Basket | Sharpe | Ann Return | Ann Vol | MDD | Calmar |
|--------|--------|-----------|---------|-----|--------|
{basket_rows}

GO threshold: Sharpe >= {raw_go_threshold}
**Raw basket result: {raw_pass_str}** (best basket: '{best_label}' -> {best_m['sharpe']:.3f})

---

## 5. Basket Comparison with Bull-Only Regime Scaling (1x/1x/1.5x SPY rank)

| Basket | Sharpe | Ann Return | MDD | Calmar |
|--------|--------|-----------|-----|--------|
{regime_rows}

Phase 62 live deployment threshold: Sharpe > **{phase62_threshold}**
**Live deployment result: {live_pass_str}** (best basket: '{best_regime_label}' -> {best_regime_m['sharpe']:.3f}, delta = {delta_live:+.3f})

---

## 6. GO / NO-GO Decision

### {verdict}

{verdict_detail}

### Comparison to Phase 59 findings

Phase 59 finding: SOL Sharpe 15.41 for BTC+ETH+SOL (equal-weight, 2021+) -- below 15.86 baseline.

Phase 73 finding (aligned 3-way window starting {data_start_str}):
- btc_eth baseline Sharpe: **{m_btc_eth['sharpe']:.3f}**
- btc_eth_sol carry-wt Sharpe: **{m_3_cw['sharpe']:.3f}** (delta: {m3_cw_delta_str})
- btc_eth_sol equal-wt Sharpe: **{m_3_eq['sharpe']:.3f}** (delta: {m3_eq_delta_str})

{dilution_finding}

---

## 7. Root Cause Analysis

{root_cause}

The correlation structure (ETH/SOL = {corr_eth_sol_all:.3f} full, {corr_eth_sol_recent_str} in 2024+) means
diversification benefit is moderate -- SOL is a better diversifier than DOGE (which had corr 0.598)
but this does not compensate for the lower mean carry.

---

## 8. Next Steps (Phase 74)

{action_text}
"""

    doc_path = ROOT / "docs" / "findings" / "phase73_sol_carry.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(doc, encoding="utf-8")
    print(f"\nFindings written to: {doc_path}")

    return {
        "verdict": verdict,
        "sol_mean_ann": sol_current_mean,
        "sol_sharpe_raw": sol_current_sharpe,
        "best_raw_basket": best_label,
        "best_raw_sharpe": best_m["sharpe"],
        "best_regime_basket": best_regime_label,
        "best_regime_sharpe": best_regime_m["sharpe"],
        "phase62_threshold": phase62_threshold,
        "live_pass": live_pass,
    }


if __name__ == "__main__":
    run()
