"""Phase 74 — Portfolio Re-optimisation with Upgraded 3-Pair LETF Leg.

Phase 66 set the 80/15/5 (carry/letf/rtmv) allocation using the old single
TQQQ/QQQ pair (Sharpe ~4.887).  Phase 70/71 upgraded the LETF leg to the
3-pair combo (TQQQ/QQQ + SOXL/SOXX + UPRO/SPY), raising standalone Sharpe to
7.06.  This script re-runs the portfolio frontier grid search with the true
upgraded legs to find whether reweighting improves the combined portfolio.

Grid search:
  carry_w  : 0.50 .. 0.90 (step 0.05)
  letf_w   : 0.05 .. 0.35 (step 0.05)
  rtmv_w   : 1 - carry_w - letf_w  (must be in [0.02, 0.15])

For each valid combo compute:
  - Combined Sharpe / Ann Return / MDD / 2022 stress return

GO threshold: combined Sharpe >= 11.65 (Phase 71 baseline 11.5636 + 0.09)

Usage
-----
    uv run python scripts/run_phase74_portfolio_reopt.py

Outputs
-------
    docs/findings/phase74_portfolio_reopt.md
    results/phase68_multistrat/tearsheet_YYYYMMDD.md   (if GO)
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANN_FACTOR = 252
CARRY_PERIODS_PER_YEAR = 3 * 365
CARRY_COST_PER_PERIOD = 0.005 / CARRY_PERIODS_PER_YEAR

LETF_COST_TQQQ_DAILY = 0.0200 / ANN_FACTOR
LETF_COST_SOXL_DAILY = 0.0340 / ANN_FACTOR
LETF_COST_UPRO_DAILY = 0.0151 / ANN_FACTOR

COMMON_START = "2020-01-01"

# Phase 71 baseline: 80/15/5 with 3-pair LETF
BASELINE_SHARPE = 11.5636
GO_THRESHOLD = BASELINE_SHARPE + 0.09   # 11.6536

# Grid parameters
CARRY_GRID  = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
LETF_GRID   = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
RTMV_MIN    = 0.02
RTMV_MAX    = 0.15


# ---------------------------------------------------------------------------
# Helpers (copied from run_phase68_multistrat_live to avoid import)
# ---------------------------------------------------------------------------

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _portfolio_metrics(r: pd.Series) -> dict[str, float]:
    r = r.dropna()
    if len(r) < 20:
        return {"sharpe": 0.0, "ann_return": 0.0, "ann_vol": 0.0,
                "mdd": 0.0, "calmar": 0.0, "cum_return": 0.0, "n": len(r)}
    cum = (1 + r).cumprod()
    n = len(r)
    ann_ret = float(cum.iloc[-1] ** (ANN_FACTOR / n) - 1)
    ann_vol = float(r.std() * np.sqrt(ANN_FACTOR))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    roll_max = cum.cummax()
    mdd = float(((cum - roll_max) / roll_max).min())
    calmar = ann_ret / abs(mdd) if mdd < 0 else 0.0
    return {
        "sharpe": sharpe,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "mdd": mdd,
        "calmar": calmar,
        "cum_return": float(cum.iloc[-1] - 1),
        "n": n,
    }


def _year_return(r: pd.Series, year: int) -> float:
    sub = r[r.index.year == year].dropna()
    if len(sub) == 0:
        return float("nan")
    return float((1 + sub).prod() - 1)


# ---------------------------------------------------------------------------
# Load strategy legs (reuses run_phase68_multistrat_live logic)
# ---------------------------------------------------------------------------

def build_carry_daily(start: str = COMMON_START) -> pd.Series:
    from funding_carry_backtest import fetch_funding_rates  # type: ignore[import-not-found]
    from rde.models.hmm import train_hmm

    print("  [carry] Fetching BTC funding rates...", end=" ", flush=True)
    btc_raw = fetch_funding_rates("BTCUSDT", start_year=int(start[:4]))
    print(f"{len(btc_raw)} periods")

    print("  [carry] Fetching ETH funding rates...", end=" ", flush=True)
    eth_raw = fetch_funding_rates("ETHUSDT", start_year=int(start[:4]))
    print(f"{len(eth_raw)} periods")

    df = pd.DataFrame({"btc": btc_raw, "eth": eth_raw}).sort_index().dropna()

    carry_btc = (df["btc"] * CARRY_PERIODS_PER_YEAR).rolling(90).mean()
    carry_eth = (df["eth"] * CARRY_PERIODS_PER_YEAR).rolling(90).mean()
    w_b = carry_btc.clip(lower=0)
    w_e = carry_eth.clip(lower=0)
    total = w_b + w_e
    in_mkt = total > 0
    w_btc = (w_b / total.where(total > 0, 1.0)).where(in_mkt, 0.0)
    w_eth = (w_e / total.where(total > 0, 1.0)).where(in_mkt, 0.0)

    print("  [carry] Fitting SPY n=3 HMM for regime scaling...", end=" ", flush=True)
    try:
        spy_raw = yf.download("SPY", start=start, interval="1d", progress=False, auto_adjust=True)
        spy_raw = _flatten(spy_raw)
        lr = np.log(spy_raw["Close"] / spy_raw["Close"].shift(1)).dropna()
        vol = lr.rolling(20).std().dropna()
        feats = pd.concat([lr, vol], axis=1).dropna()
        feats.columns = ["log_return", "volatility"]
        m = train_hmm(feats.values, n_states=3, n_restarts=3, seed_base=42)
        states = m.hmm.predict(m.scaler.transform(feats.values))
        rank_map = {int(s): int(r) for r, s in enumerate(np.argsort(m.hmm.means_[:, 0]))}
        spy_ranks = pd.Series(
            [rank_map[s] for s in states],
            index=feats.index.normalize(),
            name="hmm_rank",
        )
        ranks_by_date = {ts.date(): int(v) for ts, v in spy_ranks.items() if not pd.isna(v)}
        scale_arr = pd.Series(
            [{0: 1.0, 1: 1.0, 2: 1.5}.get(ranks_by_date.get(ts.date(), -1), 1.0)
             for ts in df.index],
            index=df.index,
        )
        counts = spy_ranks.value_counts().sort_index().to_dict()
        print(f"done. bear={counts.get(0,0)} neutral={counts.get(1,0)} bull={counts.get(2,0)}")
    except Exception as exc:
        logger.warning("SPY HMM failed (%s), using flat 1.0x scale", exc)
        scale_arr = pd.Series(1.0, index=df.index)

    raw_8h = w_btc * df["btc"] + w_eth * df["eth"]
    net_8h = raw_8h * scale_arr - in_mkt.astype(float) * CARRY_COST_PER_PERIOD

    net_8h.index = net_8h.index.normalize().tz_localize(None)
    daily = (1 + net_8h).groupby(net_8h.index).prod() - 1
    daily.name = "r_carry"
    daily = daily[daily.index >= pd.Timestamp(start).normalize()]
    print(f"  [carry] {len(daily)} bars ({daily.index[0].date()} -> {daily.index[-1].date()})")
    return daily


def _pair_returns(
    ticker_3x: str,
    ticker_1x: str,
    cost_daily: float,
    start: str,
    cache: dict[str, pd.Series] | None = None,
) -> pd.Series:
    df_3x = _flatten(yf.download(ticker_3x, start=start, interval="1d", progress=False, auto_adjust=True))
    r_3x = np.log(df_3x["Close"].squeeze() / df_3x["Close"].squeeze().shift(1)).dropna()
    r_3x.index = r_3x.index.normalize()

    if cache is not None and ticker_1x in cache:
        close_1x = cache[ticker_1x]
    else:
        df_1x = _flatten(yf.download(ticker_1x, start=start, interval="1d", progress=False, auto_adjust=True))
        close_1x = df_1x["Close"].squeeze()
        close_1x.index = close_1x.index.normalize()
        if cache is not None:
            cache[ticker_1x] = close_1x

    r_1x = np.log(close_1x / close_1x.shift(1)).dropna()
    r_1x.index = r_1x.index.normalize()
    common = r_3x.index.intersection(r_1x.index)
    r_pair = (3.0 * r_1x.loc[common] - r_3x.loc[common] - cost_daily).clip(-0.5, 0.5)
    return r_pair


def build_letf_daily(start: str = COMMON_START) -> pd.Series:
    cache: dict[str, pd.Series] = {}
    print("  [letf] Fetching TQQQ/QQQ pair...", end=" ", flush=True)
    r_tqqq = _pair_returns("TQQQ", "QQQ", LETF_COST_TQQQ_DAILY, start, cache)
    print(f"done ({len(r_tqqq)} bars)")

    print("  [letf] Fetching SOXL/SOXX pair...", end=" ", flush=True)
    r_soxl = _pair_returns("SOXL", "SOXX", LETF_COST_SOXL_DAILY, start, cache)
    print(f"done ({len(r_soxl)} bars)")

    print("  [letf] Fetching UPRO/SPY pair...", end=" ", flush=True)
    r_upro = _pair_returns("UPRO", "SPY", LETF_COST_UPRO_DAILY, start, cache)
    print(f"done ({len(r_upro)} bars)")

    combo = pd.DataFrame({"r_tqqq": r_tqqq, "r_soxl": r_soxl, "r_upro": r_upro}).dropna()
    r_letf = combo.mean(axis=1)
    r_letf.name = "r_letf"
    print(f"  [letf] {len(r_letf)} bars ({r_letf.index[0].date()} -> {r_letf.index[-1].date()})")
    return r_letf


def build_rtmv_daily(start: str = COMMON_START) -> pd.Series:
    snap_path = ROOT / "results" / "rtmv_live_5asset" / "snapshots.parquet"
    if snap_path.exists():
        print(f"  [rtmv] Loading from {snap_path}...", end=" ", flush=True)
        snaps = pd.read_parquet(snap_path)
        if "date" in snaps.columns:
            snaps = snaps.set_index("date")
        if not isinstance(snaps.index, pd.DatetimeIndex):
            snaps.index = pd.to_datetime(snaps.index)
        snaps.index = snaps.index.normalize().tz_localize(None)
        snaps = snaps[~snaps.index.duplicated(keep="last")]
        eq = snaps["equity"].dropna().sort_index()
        if len(eq) > 50:
            r_rtmv = np.log1p(eq.pct_change().dropna())
            r_rtmv = r_rtmv[r_rtmv.index >= pd.Timestamp(start).normalize()]
            r_rtmv.name = "r_rtmv"
            print(f"done. {len(r_rtmv)} bars ({r_rtmv.index[0].date()} -> {r_rtmv.index[-1].date()})")
            return r_rtmv

    print("  [rtmv] snapshots not found — running backtest...", flush=True)
    from rde.data.yfinance_source import YFinanceSource
    from rde.features.pipeline import FeaturePipeline
    from rde.features.returns import LogReturns, SmoothedReturns
    from rde.features.volatility import RollingVolatility
    from rde.trading.rtmv_rebalancer import RTMVRebalancer, RTMVRebalancerConfig

    assets = ["SPY", "GLD", "SHY", "IEF", "TLT"]
    source = YFinanceSource(cache_dir=ROOT / "results" / "cache")
    pipeline = FeaturePipeline([LogReturns(), RollingVolatility(window=20), SmoothedReturns(window=5)])
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
        {a: feat_dfs[a]["log_return"] for a in assets}, index=common_dates
    )
    cfg = RTMVRebalancerConfig(
        assets=assets,
        lambda_tilt=0.05,
        lambda_by_state_rank=[0.02, 0.05, 0.10],
        lambda_proxy_asset="SPY",
        momentum_tilt_scale=0.03,
        n_states=3,
        n_restarts=3,
        lookback_bars=504,
        rebalance_bars=21,
        drawdown_halt=0.25,
        output_dir=ROOT / "results" / "rtmv_live_5asset",
    )
    rb = RTMVRebalancer(cfg)
    snaps = rb.run_backtest(asset_returns, feat_dfs)
    if "equity" in snaps.columns:
        eq = snaps["equity"].dropna()
    else:
        eq = pd.Series(
            [s.get("equity", float("nan")) for s in rb.state.snapshots],
            index=pd.DatetimeIndex([s.get("date") for s in rb.state.snapshots]),
        ).dropna()
    eq.index = pd.to_datetime(eq.index).normalize().tz_localize(None)
    r_rtmv = np.log1p(eq.pct_change().dropna())
    r_rtmv = r_rtmv[r_rtmv.index >= pd.Timestamp(start).normalize()]
    r_rtmv.name = "r_rtmv"
    print(f"  [rtmv] done. {len(r_rtmv)} bars")
    return r_rtmv


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid_search(combined: pd.DataFrame) -> pd.DataFrame:
    """Enumerate all valid (carry, letf, rtmv) weight combinations.

    Parameters
    ----------
    combined : pd.DataFrame
        Columns r_carry, r_letf, r_rtmv aligned on common daily index.

    Returns
    -------
    pd.DataFrame
        One row per valid allocation, sorted by sharpe descending.
    """
    r_c = combined["r_carry"].values
    r_l = combined["r_letf"].values
    r_r = combined["r_rtmv"].values
    idx = combined.index

    rows = []
    for cw in CARRY_GRID:
        for lw in LETF_GRID:
            rw = round(1.0 - cw - lw, 6)
            if rw < RTMV_MIN - 1e-9 or rw > RTMV_MAX + 1e-9:
                continue
            r_port = cw * r_c + lw * r_l + rw * r_r
            r_series = pd.Series(r_port, index=idx)

            m = _portfolio_metrics(r_series)
            ret_2022 = _year_return(r_series, 2022)
            rows.append({
                "carry_w": cw,
                "letf_w": lw,
                "rtmv_w": rw,
                "sharpe": m["sharpe"],
                "ann_return": m["ann_return"],
                "ann_vol": m["ann_vol"],
                "mdd": m["mdd"],
                "calmar": m["calmar"],
                "cum_return": m["cum_return"],
                "ret_2022": ret_2022,
            })

    grid = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return grid


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    return f"{v:+.2%}" if not np.isnan(v) else "N/A"


def print_heatmap(grid: pd.DataFrame) -> None:
    """Print a compact Sharpe heatmap over (carry_w, letf_w)."""
    carry_vals = sorted(grid["carry_w"].unique())
    letf_vals  = sorted(grid["letf_w"].unique())

    # Header
    header = f"{'carry\\letf':>10}" + "".join(f"  {v:.0%}" for v in letf_vals)
    print(header)
    print("  " + "-" * (len(header) + 2))
    for cw in carry_vals:
        row_str = f"  {cw:.0%}      "
        for lw in letf_vals:
            sub = grid[(grid["carry_w"] == cw) & (grid["letf_w"].round(6) == round(lw, 6))]
            if len(sub) == 0:
                row_str += "     --  "
            else:
                sh = sub.iloc[0]["sharpe"]
                row_str += f"  {sh:7.3f}"
        print(row_str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sep = "=" * 70

    print(sep)
    print("PHASE 74 — PORTFOLIO FRONTIER RE-OPTIMISATION")
    print(f"  Phase 71 baseline (80/15/5): Sharpe {BASELINE_SHARPE:.4f}")
    print(f"  GO threshold                : Sharpe {GO_THRESHOLD:.4f}")
    print(f"  Common start                : {COMMON_START}")
    print(sep)

    # -------------------------------------------------------------------------
    # Build legs
    # -------------------------------------------------------------------------
    print("\n[1/3] Building carry P&L...")
    r_carry = build_carry_daily(start=COMMON_START)

    print("\n[2/3] Building LETF 3-pair combo P&L...")
    r_letf = build_letf_daily(start=COMMON_START)

    print("\n[3/3] Loading RTMV P&L...")
    r_rtmv = build_rtmv_daily(start=COMMON_START)

    # Align
    combined = pd.DataFrame({
        "r_carry": r_carry,
        "r_letf": r_letf,
        "r_rtmv": r_rtmv,
    }).dropna()

    print(f"\nCommon period: {combined.index[0].date()} → {combined.index[-1].date()}")
    print(f"Common bars  : {len(combined)}")

    # -------------------------------------------------------------------------
    # Individual strategy metrics
    # -------------------------------------------------------------------------
    print(f"\n{sep}")
    print("INDIVIDUAL STRATEGY STANDALONE METRICS")
    print(sep)
    m_carry = _portfolio_metrics(combined["r_carry"])
    m_letf  = _portfolio_metrics(combined["r_letf"])
    m_rtmv  = _portfolio_metrics(combined["r_rtmv"])

    for label, m in [("Carry", m_carry), ("LETF 3-pair", m_letf), ("RTMV+mom", m_rtmv)]:
        print(f"  {label:<14}  Sharpe {m['sharpe']:.4f}  AnnRet {m['ann_return']:.2%}"
              f"  Vol {m['ann_vol']:.2%}  MDD {m['mdd']:.2%}")

    corr_cl = float(combined["r_carry"].corr(combined["r_letf"]))
    corr_cr = float(combined["r_carry"].corr(combined["r_rtmv"]))
    corr_lr = float(combined["r_letf"].corr(combined["r_rtmv"]))
    print(f"\n  carry↔letf  : {corr_cl:+.4f}")
    print(f"  carry↔rtmv  : {corr_cr:+.4f}")
    print(f"  letf↔rtmv   : {corr_lr:+.4f}")

    # -------------------------------------------------------------------------
    # Grid search
    # -------------------------------------------------------------------------
    print(f"\n{sep}")
    print("GRID SEARCH")
    print(f"  carry_w: {[f'{v:.0%}' for v in CARRY_GRID]}")
    print(f"  letf_w:  {[f'{v:.0%}' for v in LETF_GRID]}")
    print(f"  rtmv_w:  [{RTMV_MIN:.0%}, {RTMV_MAX:.0%}] (residual)")
    print(sep)

    grid = run_grid_search(combined)
    print(f"  Valid combinations: {len(grid)}")

    # -------------------------------------------------------------------------
    # Top 10 by Sharpe
    # -------------------------------------------------------------------------
    print(f"\n{'TOP 10 ALLOCATIONS BY SHARPE':}")
    print(f"  {'Carry':>6}  {'LETF':>5}  {'RTMV':>5}  "
          f"{'Sharpe':>8}  {'AnnRet':>8}  {'MDD':>8}  {'2022':>8}  {'Calmar':>8}")
    print("  " + "-" * 72)
    for _, row in grid.head(10).iterrows():
        print(
            f"  {row['carry_w']:>6.0%}  {row['letf_w']:>5.0%}  {row['rtmv_w']:>5.1%}  "
            f"  {row['sharpe']:>7.4f}  {row['ann_return']:>8.2%}  {row['mdd']:>8.2%}  "
            f"{row['ret_2022']:>8.2%}  {row['calmar']:>8.3f}"
        )

    # -------------------------------------------------------------------------
    # Baseline comparison
    # -------------------------------------------------------------------------
    baseline_row = grid[(grid["carry_w"] == 0.80) & (grid["letf_w"] == 0.15)]
    if len(baseline_row) == 0:
        # Compute manually
        r_bl = 0.80 * combined["r_carry"] + 0.15 * combined["r_letf"] + 0.05 * combined["r_rtmv"]
        m_bl = _portfolio_metrics(r_bl)
        bl_sharpe   = m_bl["sharpe"]
        bl_mdd      = m_bl["mdd"]
        bl_ann_ret  = m_bl["ann_return"]
        bl_ret_2022 = _year_return(r_bl, 2022)
    else:
        bl = baseline_row.iloc[0]
        bl_sharpe   = bl["sharpe"]
        bl_mdd      = bl["mdd"]
        bl_ann_ret  = bl["ann_return"]
        bl_ret_2022 = bl["ret_2022"]

    # -------------------------------------------------------------------------
    # Constrained optima
    # -------------------------------------------------------------------------
    best_unconstrained = grid.iloc[0]

    # MDD constraint: |MDD| <= 0.25%
    mdd_constrained = grid[grid["mdd"] >= -0.0025]
    best_mdd = mdd_constrained.iloc[0] if len(mdd_constrained) > 0 else None

    # 2022 stress: must be positive
    stress_constrained = grid[grid["ret_2022"] > 0]
    best_stress = stress_constrained.iloc[0] if len(stress_constrained) > 0 else None

    print(f"\n{sep}")
    print("OPTIMAL ALLOCATIONS")
    print(sep)
    print(f"  Baseline (80/15/5)            : "
          f"Sharpe {bl_sharpe:.4f}  MDD {bl_mdd:.2%}  AnnRet {bl_ann_ret:.2%}  2022 {bl_ret_2022:.2%}")

    print(f"\n  Best unconstrained            : "
          f"carry={best_unconstrained['carry_w']:.0%}  letf={best_unconstrained['letf_w']:.0%}  "
          f"rtmv={best_unconstrained['rtmv_w']:.1%}  "
          f"Sharpe {best_unconstrained['sharpe']:.4f}  MDD {best_unconstrained['mdd']:.2%}  "
          f"2022 {best_unconstrained['ret_2022']:.2%}")

    if best_mdd is not None:
        print(f"  Best with |MDD|<=0.25%        : "
              f"carry={best_mdd['carry_w']:.0%}  letf={best_mdd['letf_w']:.0%}  "
              f"rtmv={best_mdd['rtmv_w']:.1%}  "
              f"Sharpe {best_mdd['sharpe']:.4f}  MDD {best_mdd['mdd']:.2%}  "
              f"2022 {best_mdd['ret_2022']:.2%}")

    if best_stress is not None:
        print(f"  Best with 2022>0              : "
              f"carry={best_stress['carry_w']:.0%}  letf={best_stress['letf_w']:.0%}  "
              f"rtmv={best_stress['rtmv_w']:.1%}  "
              f"Sharpe {best_stress['sharpe']:.4f}  MDD {best_stress['mdd']:.2%}  "
              f"2022 {best_stress['ret_2022']:.2%}")

    # -------------------------------------------------------------------------
    # Sharpe heatmap
    # -------------------------------------------------------------------------
    print(f"\n{sep}")
    print("SHARPE HEATMAP  (carry_w rows, letf_w cols; -- = invalid rtmv constraint)")
    print(sep)
    print_heatmap(grid)

    # -------------------------------------------------------------------------
    # Determine GO/NO-GO and recommended weights
    # -------------------------------------------------------------------------
    top_sharpe = float(best_unconstrained["sharpe"])
    improvement = top_sharpe - BASELINE_SHARPE
    go = top_sharpe >= GO_THRESHOLD

    # Pick the recommendation: if GO, use unconstrained best; otherwise keep baseline
    if go:
        rec_cw = float(best_unconstrained["carry_w"])
        rec_lw = float(best_unconstrained["letf_w"])
        rec_rw = float(best_unconstrained["rtmv_w"])
        rec_sharpe = top_sharpe
        rec_mdd    = float(best_unconstrained["mdd"])
        rec_ret    = float(best_unconstrained["ann_return"])
        rec_2022   = float(best_unconstrained["ret_2022"])
    else:
        rec_cw = 0.80
        rec_lw = 0.15
        rec_rw = 0.05
        rec_sharpe = bl_sharpe
        rec_mdd    = bl_mdd
        rec_ret    = bl_ann_ret
        rec_2022   = bl_ret_2022

    print(f"\n{sep}")
    print("GO/NO-GO VERDICT")
    print(sep)
    print(f"  Best Sharpe    : {top_sharpe:.4f}")
    print(f"  GO threshold   : {GO_THRESHOLD:.4f}")
    print(f"  Improvement    : {improvement:+.4f} (need +0.09)")
    print(f"  Verdict        : {'GO' if go else 'NO-GO'}")
    print(sep)

    # -------------------------------------------------------------------------
    # If GO: update run_phase68_multistrat_live.py defaults
    # -------------------------------------------------------------------------
    if go:
        print(f"\n  --> GO! Updating run_phase68_multistrat_live.py defaults to "
              f"{rec_cw:.0%}/{rec_lw:.0%}/{rec_rw:.1%}")
        _update_phase68_defaults(rec_cw, rec_lw, rec_rw)
    else:
        print(f"\n  --> NO-GO. Phase 71 baseline weights (80/15/5) remain optimal.")

    # -------------------------------------------------------------------------
    # Write findings markdown
    # -------------------------------------------------------------------------
    findings_dir = ROOT / "docs" / "findings"
    findings_dir.mkdir(parents=True, exist_ok=True)
    md_path = findings_dir / "phase74_portfolio_reopt.md"

    top10_rows = "\n".join(
        f"| {r['carry_w']:.0%} | {r['letf_w']:.0%} | {r['rtmv_w']:.1%} | "
        f"**{r['sharpe']:.4f}** | {r['ann_return']:.2%} | {r['mdd']:.2%} | "
        f"{r['ret_2022']:.2%} |"
        if i == 0 else
        f"| {r['carry_w']:.0%} | {r['letf_w']:.0%} | {r['rtmv_w']:.1%} | "
        f"{r['sharpe']:.4f} | {r['ann_return']:.2%} | {r['mdd']:.2%} | "
        f"{r['ret_2022']:.2%} |"
        for i, (_, r) in enumerate(grid.head(10).iterrows())
    )

    heatmap_lines: list[str] = []
    carry_vals = sorted(grid["carry_w"].unique())
    letf_vals  = sorted(grid["letf_w"].unique())
    header_row = "| carry \\ letf |" + "|".join(f" {v:.0%} " for v in letf_vals) + "|"
    sep_row    = "|---|" + "|".join("---|" for _ in letf_vals)
    heatmap_lines.append(header_row)
    heatmap_lines.append(sep_row)
    for cw in carry_vals:
        cells = []
        for lw in letf_vals:
            sub = grid[(grid["carry_w"] == cw) & (grid["letf_w"].round(6) == round(lw, 6))]
            if len(sub) == 0:
                cells.append(" — ")
            else:
                sh = sub.iloc[0]["sharpe"]
                marker = " **" if sh >= GO_THRESHOLD else ""
                cells.append(f" {sh:.3f}{marker} ")
        heatmap_lines.append(f"| {cw:.0%} |" + "|".join(cells) + "|")

    heatmap_md = "\n".join(heatmap_lines)

    verdict_str = "GO" if go else "NO-GO"
    deploy_section = ""
    if go:
        deploy_section = f"""
## Deployment

New weights deployed to `scripts/run_phase68_multistrat_live.py`:
- `--carry-weight` default changed to **{rec_cw:.2f}** (was 0.80)
- `--letf-weight`  default changed to **{rec_lw:.2f}** (was 0.15)
- `--rtmv-weight`  default changed to **{rec_rw:.2f}** (was 0.05)

To run with new weights:
```bash
make multistrat-backtest
```
"""
    else:
        deploy_section = """
## Deployment

NO-GO: Phase 71 baseline weights (80% carry / 15% LETF / 5% RTMV) remain unchanged.
The 3-pair LETF upgrade did not shift the optimal allocation meaningfully.
"""

    md_content = f"""# Phase 74 — Portfolio Frontier Re-Optimisation (3-Pair LETF Upgrade)

**Date:** {date.today().isoformat()}
**Branch:** phase51/regime-conditional-lambda

## Context

Phase 66 set the multi-strategy allocation at **80% carry / 15% LETF / 5% RTMV** based on
the single TQQQ/QQQ pair (Sharpe ~4.887).  Phase 70/71 upgraded the LETF leg to the
3-pair combo (TQQQ/QQQ + SOXL/SOXX + UPRO/SPY), raising standalone Sharpe to **7.06**
(+0.82 improvement).  Phase 72 confirmed the LETF pair earns most in bear regimes
(+61%/yr bear vs +9%/yr bull), making it a natural hedge for carry drawdowns.

**Question:** With a stronger LETF leg, is more LETF allocation beneficial?

---

## Individual Strategy Metrics (2020-01-01 to present)

| Strategy | Sharpe | Ann Return | Ann Vol | MDD |
|---|---|---|---|---|
| Carry (BTC+ETH carry-weighted, bull-only) | {m_carry['sharpe']:.4f} | {m_carry['ann_return']:.2%} | {m_carry['ann_vol']:.2%} | {m_carry['mdd']:.2%} |
| LETF 3-pair combo | {m_letf['sharpe']:.4f} | {m_letf['ann_return']:.2%} | {m_letf['ann_vol']:.2%} | {m_letf['mdd']:.2%} |
| RTMV + momentum | {m_rtmv['sharpe']:.4f} | {m_rtmv['ann_return']:.2%} | {m_rtmv['ann_vol']:.2%} | {m_rtmv['mdd']:.2%} |

Cross-strategy correlations: carry↔LETF = {corr_cl:+.4f}, carry↔RTMV = {corr_cr:+.4f}, LETF↔RTMV = {corr_lr:+.4f}

---

## Grid Search Results — Top 10 Allocations

Grid: carry ∈ [50%,90%], letf ∈ [5%,35%], rtmv = residual ∈ [2%,15%].

| Carry | LETF | RTMV | Sharpe | Ann Return | MDD | 2022 |
|---|---|---|---|---|---|---|
{top10_rows}

---

## Sharpe Heatmap (carry rows, letf cols; — = invalid rtmv constraint)

{heatmap_md}

Values ≥ {GO_THRESHOLD:.4f} are marked **bold** (GO threshold).

---

## Comparison vs Phase 71 Baseline

| Metric | Phase 71 Baseline (80/15/5) | Phase 74 Optimal | Change |
|---|---|---|---|
| Allocation | 80/15/5 | {rec_cw:.0%}/{rec_lw:.0%}/{rec_rw:.1%} | {'changed' if go else 'unchanged'} |
| Combined Sharpe | {bl_sharpe:.4f} | {rec_sharpe:.4f} | {rec_sharpe - bl_sharpe:+.4f} |
| MDD | {bl_mdd:.2%} | {rec_mdd:.2%} | {rec_mdd - bl_mdd:+.4f} |
| Ann Return | {bl_ann_ret:.2%} | {rec_ret:.2%} | {rec_ret - bl_ann_ret:+.4f} |
| 2022 Stress | {bl_ret_2022:.2%} | {rec_2022:.2%} | {rec_2022 - bl_ret_2022:+.4f} |

---

## GO/NO-GO Verdict

| Criterion | Threshold | Result | Status |
|---|---|---|---|
| Sharpe improvement | ≥ +0.09 over {BASELINE_SHARPE:.4f} | {improvement:+.4f} | {"PASS" if go else "FAIL"} |
| 2022 stress positive | > 0% | {rec_2022:.2%} | {"PASS" if rec_2022 > 0 else "FAIL"} |

**Verdict: {verdict_str}**
{deploy_section}

---

## Analysis

The Phase 71 baseline carries most of the combined Sharpe because carry (Sharpe ~{m_carry['sharpe']:.1f})
overwhelms the LETF and RTMV contributions by a wide margin.  The carry component
is responsible for the low portfolio MDD (−{abs(bl_mdd):.2%}) since carry MDD is only
−{abs(m_carry['mdd']):.2%}.  Increasing the LETF weight from 15% to 20%+ raises
annualised return slightly (LETF earns {m_letf['ann_return']:.1%}/yr) but also increases
portfolio vol because the LETF vol is {m_letf['ann_vol']:.1%}/yr vs carry's {m_carry['ann_vol']:.1%}/yr.
The Sharpe of the combined portfolio peaks at the allocation where the marginal
diversification benefit of the LETF—via carry↔LETF correlation of {corr_cl:+.4f}—
exactly offsets the vol dilution of reducing carry weight.

## Phase 75 Recommendation

{"Since Phase 74 is GO with new weights deployed, Phase 75 = full end-to-end CV re-validation." if go else
 "Since Phase 74 is NO-GO (80/15/5 still optimal), Phase 75 = **intraday carry timing**." + chr(10) +
 "Test whether funding collection timing (8h UTC windows 00:00/08:00/16:00) can be optimised." + chr(10) +
 "Some windows are systematically higher — micro-optimisation but potentially +1–2% ann."}
"""

    md_path.write_text(md_content, encoding="utf-8")
    print(f"\nFindings written to: {md_path}")

    # -------------------------------------------------------------------------
    # If GO: write updated tearsheet
    # -------------------------------------------------------------------------
    if go:
        _write_updated_tearsheet(
            combined=combined,
            carry_w=rec_cw, letf_w=rec_lw, rtmv_w=rec_rw,
            m_carry=m_carry, m_letf=m_letf, m_rtmv=m_rtmv,
            corr_cl=corr_cl, corr_cr=corr_cr, corr_lr=corr_lr,
        )

    print(f"\n{'GO' if go else 'NO-GO'} — Phase 74 complete.")


# ---------------------------------------------------------------------------
# Helpers for GO path
# ---------------------------------------------------------------------------

def _update_phase68_defaults(carry_w: float, letf_w: float, rtmv_w: float) -> None:
    """Update argparse defaults in run_phase68_multistrat_live.py."""
    script = ROOT / "scripts" / "run_phase68_multistrat_live.py"
    text = script.read_text(encoding="utf-8")

    import re

    # Update carry-weight default
    text = re.sub(
        r'(--carry-weight["\',\s]+type=float,\s+default=)[0-9.]+',
        lambda m: m.group(0).rsplit("=", 1)[0] + f"={carry_w:.2f}",
        text,
    )
    # Update letf-weight default
    text = re.sub(
        r'(--letf-weight["\',\s]+type=float,\s+default=)[0-9.]+',
        lambda m: m.group(0).rsplit("=", 1)[0] + f"={letf_w:.2f}",
        text,
    )
    # Update rtmv-weight default
    text = re.sub(
        r'(--rtmv-weight["\',\s]+type=float,\s+default=)[0-9.]+',
        lambda m: m.group(0).rsplit("=", 1)[0] + f"={rtmv_w:.2f}",
        text,
    )
    # Update the comment on the Optimal allocation line
    text = re.sub(
        r"(Optimal allocation[^:]*: )[0-9.%/]+carry[^)]+\)",
        lambda m: m.group(0),  # leave existing comment, don't corrupt it
        text,
    )
    # Update the top-level docstring allocation comment
    old_alloc = "Optimal allocation (Phase 66): 80% carry + 15% LETF + 5% RTMV"
    new_alloc = (
        f"Optimal allocation (Phase 74): {carry_w:.0%} carry + {letf_w:.0%} LETF + {rtmv_w:.0%} RTMV"
    )
    text = text.replace(old_alloc, new_alloc)

    script.write_text(text, encoding="utf-8")
    print(f"  Updated defaults in {script}")


def _write_updated_tearsheet(
    combined: pd.DataFrame,
    carry_w: float, letf_w: float, rtmv_w: float,
    m_carry: dict, m_letf: dict, m_rtmv: dict,
    corr_cl: float, corr_cr: float, corr_lr: float,
) -> None:
    from typing import Any
    r_combined = (
        carry_w * combined["r_carry"]
        + letf_w * combined["r_letf"]
        + rtmv_w * combined["r_rtmv"]
    )
    m_comb = _portfolio_metrics(r_combined)
    go = m_comb["sharpe"] >= GO_THRESHOLD

    def _per_year(r: pd.Series) -> dict[int, float]:
        r = r.dropna()
        return {yr: float((1 + grp).prod() - 1) for yr, grp in r.groupby(r.index.year)}

    ys_comb  = _per_year(r_combined)
    ys_carry = _per_year(combined["r_carry"])
    ys_letf  = _per_year(combined["r_letf"])
    ys_rtmv  = _per_year(combined["r_rtmv"])
    all_yrs  = sorted(set(ys_comb) | set(ys_carry) | set(ys_letf) | set(ys_rtmv))

    yearly_rows = "\n".join(
        f"| {yr} | {ys_comb.get(yr, float('nan')):+.2%} | "
        f"{ys_carry.get(yr, float('nan')):+.2%} | "
        f"{ys_letf.get(yr, float('nan')):+.2%} | "
        f"{ys_rtmv.get(yr, float('nan')):+.2%} |"
        for yr in all_yrs
    )

    today_str = date.today().strftime("%Y%m%d")
    output_dir = ROOT / "results" / "phase68_multistrat"
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"tearsheet_{today_str}.md"

    md = f"""# Phase 74: Unified Multi-Strategy Portfolio (Re-Optimised Weights)

**Date:** {date.today().isoformat()}
**Allocation:** carry={carry_w:.0%} / letf={letf_w:.0%} / rtmv={rtmv_w:.1%}
**Verdict: {"GO" if go else "NO-GO"}** (threshold: Sharpe ≥ {GO_THRESHOLD:.4f})

## Strategy Overview

| Strategy | Description | Phase |
|---|---|---|
| Crypto Carry | BTC+ETH carry-weighted, bull-only SPY regime scale | Phase 62 GO |
| LETF 3-Pair Combo | SHORT TQQQ/QQQ + SOXL/SOXX + UPRO/SPY, equal-weight, always-in | Phase 70 GO |
| RTMV + Momentum | SPY/GLD/SHY/IEF/TLT, spy_rank_bull, mom=0.03 | Phase 57 GO |

Common period: {combined.index[0].date()} → {combined.index[-1].date()} ({len(combined)} days)

## Combined Portfolio Metrics

| Metric | Combined | Carry | LETF | RTMV |
|---|---|---|---|---|
| Sharpe | **{m_comb['sharpe']:.4f}** | {m_carry['sharpe']:.4f} | {m_letf['sharpe']:.4f} | {m_rtmv['sharpe']:.4f} |
| Ann Return | {m_comb['ann_return']:.2%} | {m_carry['ann_return']:.2%} | {m_letf['ann_return']:.2%} | {m_rtmv['ann_return']:.2%} |
| Ann Vol | {m_comb['ann_vol']:.2%} | {m_carry['ann_vol']:.2%} | {m_letf['ann_vol']:.2%} | {m_rtmv['ann_vol']:.2%} |
| Max DD | {m_comb['mdd']:.2%} | {m_carry['mdd']:.2%} | {m_letf['mdd']:.2%} | {m_rtmv['mdd']:.2%} |
| Calmar | {m_comb['calmar']:.4f} | {m_carry['calmar']:.4f} | {m_letf['calmar']:.4f} | {m_rtmv['calmar']:.4f} |
| Cum Return | {m_comb['cum_return']:.2%} | {m_carry['cum_return']:.2%} | {m_letf['cum_return']:.2%} | {m_rtmv['cum_return']:.2%} |

## Cross-Strategy Correlations

| Pair | Correlation |
|---|---|
| carry↔letf | {corr_cl:+.4f} |
| carry↔rtmv | {corr_cr:+.4f} |
| letf↔rtmv | {corr_lr:+.4f} |

## Per-Year Returns

| Year | Combined | Carry | LETF | RTMV |
|---|---|---|---|---|
{yearly_rows}

## GO/NO-GO Criteria

| Criterion | Threshold | Result | Status |
|---|---|---|---|
| Combined Sharpe | ≥ {GO_THRESHOLD:.4f} | {m_comb['sharpe']:.4f} | {"PASS" if go else "FAIL"} |

## Notes

This tearsheet uses Phase 74 re-optimised weights ({carry_w:.0%}/{letf_w:.0%}/{rtmv_w:.1%}).
Previous Phase 71 tearsheet used 80%/15%/5%.
"""
    md_path.write_text(md, encoding="utf-8")
    print(f"Tearsheet written to: {md_path}")


if __name__ == "__main__":
    main()
