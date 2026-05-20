"""Phase 68 / Phase 71 / Phase 77: Unified Multi-Strategy Portfolio Executor.

Combines three independent alpha strategies validated in Phases 52a/57, 60/62, and 65/70:
  1. Crypto funding carry (BTC+ETH, spot_spread_dynamic + momentum_filter) — Phase 77 upgrade
  2. LETF decay 3-pair combo (TQQQ/QQQ + SOXL/SOXX + UPRO/SPY, equal-weight) — Phase 70 GO
  3. RTMV + momentum (SPY/GLD/SHY/IEF/TLT, spy_rank_bull, mom=0.03) — Phase 57 GO

Phase 71 upgrade: replaced single TQQQ/QQQ pair with 3-pair equal-weight combo.
  - TQQQ/QQQ:  borrow 1.5%/yr  (Phase 68 baseline)
  - SOXL/SOXX: borrow 2.0%/yr  (semiconductor 3× ETF)
  - UPRO/SPY:  borrow 0.5%/yr  (S&P 500 3× ETF; SPY reused from RTMV download)

Phase 77 upgrade: carry leg now uses Phase 75 + Phase 76 improvements:
  - Phase 75 momentum filter: skip periods when previous combined rate ≤ 0 (18.13 → 18.57 Sharpe)
  - Phase 76 spot_spread_dynamic: ETH weight = clip(eth_rate / sum, 0.40, 0.80) (18.57 → 19.73)

Optimal allocation (Phase 66): 80% carry + 15% LETF + 5% RTMV → Sharpe ~9.8, MDD -0.55%.
Pairwise correlations (Phase 61): structurally near-zero across all pairs.

Usage
-----
Backtest (2020-01-01 to present):
    uv run python scripts/run_phase68_multistrat_live.py --mode backtest

Signal (today's combined status):
    uv run python scripts/run_phase68_multistrat_live.py --mode signal

Custom allocation:
    uv run python scripts/run_phase68_multistrat_live.py \\
        --mode backtest --carry-weight 0.70 --letf-weight 0.20 --rtmv-weight 0.10

GO threshold: Combined Sharpe >= 11.05 (Phase 68 baseline)
"""

from __future__ import annotations

import argparse
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
CARRY_PERIODS_PER_YEAR = 3 * 365      # 8-hourly periods
CARRY_COST_PER_PERIOD = 0.005 / CARRY_PERIODS_PER_YEAR  # ~0.5% annual

# Phase 71: 3-pair LETF combo costs (borrow + ER_3x + ER_1x + friction per pair)
# TQQQ/QQQ:  0.009 + 0.0086 + 0.0020 + 0.0004 = 0.0200/yr
# SOXL/SOXX: 0.020 + 0.0095 + 0.0035 + 0.0010 = 0.0340/yr
# UPRO/SPY:  0.005 + 0.0091 + 0.0009 + 0.0001 = 0.0151/yr
LETF_COST_TQQQ_DAILY = 0.0200 / ANN_FACTOR
LETF_COST_SOXL_DAILY = 0.0340 / ANN_FACTOR
LETF_COST_UPRO_DAILY = 0.0151 / ANN_FACTOR

COMMON_START = "2020-01-01"           # Carry data begins 2020; aligns all three strategies
GO_THRESHOLD = 11.05                  # Phase 68 baseline Sharpe to beat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Defensively flatten multi-level yfinance column index."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _portfolio_metrics(r: pd.Series) -> dict[str, float]:
    """Compute standard tearsheet metrics from a daily return series."""
    r = r.dropna()
    if len(r) < 20:
        return {"sharpe": 0.0, "ann_return": 0.0, "ann_vol": 0.0, "mdd": 0.0,
                "calmar": 0.0, "cum_return": 0.0, "n": len(r)}
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


def _per_year_returns(r: pd.Series) -> dict[int, float]:
    """Annual compound return by calendar year."""
    r = r.dropna()
    return {yr: float((1 + grp).prod() - 1) for yr, grp in r.groupby(r.index.year)}


# ---------------------------------------------------------------------------
# Leg 1: Crypto Funding Carry (Phase 62 carry_weighted_bull_only)
# ---------------------------------------------------------------------------

def build_carry_daily(start: str = COMMON_START) -> pd.Series:
    """Fetch BTC+ETH funding rates and compute carry-weighted daily returns.

    Phase 77 upgrade — integrates Phase 75 + Phase 76 carry improvements:
    - Phase 76 spot_spread_dynamic: ETH weight = clip(eth_rate / (eth_rate + btc_rate), 0.40, 0.80)
      Instantaneous proportional weighting (replaces Phase 60 rolling 90-period weights)
    - Phase 75 momentum filter: skip periods when previous combined rate ≤ 0 (hold cash)
    - Scale by SPY HMM rank: {0: 1.0, 1: 1.0, 2: 1.5} (bull-only, Phase 62)
    - ~0.5% annual friction applied per occupied period

    Carry leg Sharpe progression:
      Phase 60 (rolling carry weights):         18.13
      Phase 75 (+ momentum filter):             18.57
      Phase 76 (+ spot_spread_dynamic):         19.73  ← this implementation

    Returns
    -------
    pd.Series
        Daily log-return series, index = DatetimeIndex (tz-naive, date-normalized).
    """
    from funding_carry_backtest import fetch_funding_rates  # type: ignore[import-not-found]
    from rde.models.hmm import train_hmm

    print("  [carry] Fetching BTC funding rates from Binance...", end=" ", flush=True)
    btc_raw = fetch_funding_rates("BTCUSDT", start_year=int(start[:4]))
    print(f"{len(btc_raw)} periods")

    print("  [carry] Fetching ETH funding rates from Binance...", end=" ", flush=True)
    eth_raw = fetch_funding_rates("ETHUSDT", start_year=int(start[:4]))
    print(f"{len(eth_raw)} periods")

    df = pd.DataFrame({"btc": btc_raw, "eth": eth_raw}).sort_index().dropna()

    # Phase 76 GO: instantaneous spot-spread dynamic weights for BTC/ETH.
    # ETH weight = clip(eth_rate / (btc_rate+ + eth_rate+), 0.40, 0.80).
    # Clip negative rates to 0 before computing proportional weight.
    r_btc_pos = df["btc"].clip(lower=0.0)
    r_eth_pos = df["eth"].clip(lower=0.0)
    denom = (r_btc_pos + r_eth_pos).where(lambda x: x > 0, np.nan)
    raw_w_eth = r_eth_pos / denom
    w_eth = raw_w_eth.clip(lower=0.40, upper=0.80).fillna(0.5)
    w_btc = 1.0 - w_eth

    # Gate on whether any carry is positive (mirrors Phase 60 entry logic)
    any_positive = (df["btc"] > 0) | (df["eth"] > 0)
    w_btc_final = w_btc.where(any_positive, 0.0)
    w_eth_final = w_eth.where(any_positive, 0.0)
    in_mkt = any_positive

    # Phase 75 GO: lag-1 momentum filter — skip periods when previous combined rate ≤ 0.
    combined_rate = (df["btc"] + df["eth"]) * 0.5
    prev_positive = combined_rate.shift(1) > 0
    skip_mask = ~prev_positive.fillna(True)  # skip on first period and when prev ≤ 0
    in_mkt = in_mkt & ~skip_mask
    w_btc_final = w_btc_final.where(~skip_mask, 0.0)
    w_eth_final = w_eth_final.where(~skip_mask, 0.0)

    # SPY HMM rank for regime scaling (Phase 62 bull-only)
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
            [
                {0: 1.0, 1: 1.0, 2: 1.5}.get(ranks_by_date.get(ts.date(), -1), 1.0)
                for ts in df.index
            ],
            index=df.index,
        )
        counts = spy_ranks.value_counts().sort_index().to_dict()
        print(f"done. bear={counts.get(0,0)} neutral={counts.get(1,0)} bull={counts.get(2,0)}")
    except Exception as exc:
        logger.warning("SPY HMM failed (%s), using flat 1.0× scale", exc)
        scale_arr = pd.Series(1.0, index=df.index)

    # 8-hourly net return: spot_spread_dynamic weights * scale - friction
    raw_8h = w_btc_final * df["btc"] + w_eth_final * df["eth"]
    net_8h = raw_8h * scale_arr - in_mkt.astype(float) * CARRY_COST_PER_PERIOD

    # Aggregate to daily log-return (sum of log-returns ≈ product for small r)
    net_8h.index = net_8h.index.normalize().tz_localize(None)
    daily = (1 + net_8h).groupby(net_8h.index).prod() - 1
    daily.name = "r_carry"

    # Clip to start date and return
    daily = daily[daily.index >= pd.Timestamp(start).normalize()]
    print(f"  [carry] Daily series: {len(daily)} bars "
          f"({daily.index[0].date()} -> {daily.index[-1].date()})")
    return daily


# ---------------------------------------------------------------------------
# Leg 2: LETF Decay 3-Pair Combo (Phase 71 upgrade from single TQQQ/QQQ pair)
# ---------------------------------------------------------------------------

def _pair_returns(
    ticker_3x: str,
    ticker_1x: str,
    cost_daily: float,
    start: str,
    close_1x_cache: dict[str, pd.Series] | None = None,
) -> pd.Series:
    """Compute daily log-return for one delta-neutral LETF pair.

    r_pair = 3 * r_1x - r_3x - cost_daily

    Parameters
    ----------
    ticker_3x : str
        Ticker for the 3× leveraged ETF (short leg).
    ticker_1x : str
        Ticker for the 1× underlying ETF (long leg, 3 units).
    cost_daily : float
        Total daily cost (borrow + ER + friction), pre-divided by 252.
    start : str
        Earliest date for data download (ISO format).
    close_1x_cache : dict, optional
        Pre-downloaded Close series keyed by ticker to avoid re-downloading SPY.
    """
    df_3x = _flatten(yf.download(ticker_3x, start=start, interval="1d", progress=False, auto_adjust=True))
    r_3x = np.log(df_3x["Close"].squeeze() / df_3x["Close"].squeeze().shift(1)).dropna()
    r_3x.index = r_3x.index.normalize()

    if close_1x_cache is not None and ticker_1x in close_1x_cache:
        close_1x = close_1x_cache[ticker_1x]
    else:
        df_1x = _flatten(yf.download(ticker_1x, start=start, interval="1d", progress=False, auto_adjust=True))
        close_1x = df_1x["Close"].squeeze()
        close_1x.index = close_1x.index.normalize()
        if close_1x_cache is not None:
            close_1x_cache[ticker_1x] = close_1x

    r_1x = np.log(close_1x / close_1x.shift(1)).dropna()
    r_1x.index = r_1x.index.normalize()

    common = r_3x.index.intersection(r_1x.index)
    r_pair = (3.0 * r_1x.loc[common] - r_3x.loc[common] - cost_daily).clip(-0.5, 0.5)
    return r_pair


def build_letf_daily(start: str = COMMON_START) -> pd.Series:
    """Compute daily returns of the 3-pair equal-weight LETF combo (Phase 71).

    Pairs (always-in, delta-neutral):
      - SHORT TQQQ + LONG 3×QQQ  (Nasdaq 100)
      - SHORT SOXL + LONG 3×SOXX (Semiconductors)
      - SHORT UPRO + LONG 3×SPY  (S&P 500)

    Equal-weight average of all three pair returns.

    Returns
    -------
    pd.Series
        Daily log-return series, index = DatetimeIndex (tz-naive, date-normalized).
    """
    # Cache SPY closes so it is downloaded once (also used by RTMV leg)
    spy_cache: dict[str, pd.Series] = {}

    print("  [letf] Fetching TQQQ/QQQ pair...", end=" ", flush=True)
    r_tqqq = _pair_returns("TQQQ", "QQQ", LETF_COST_TQQQ_DAILY, start, spy_cache)
    print(f"done ({len(r_tqqq)} bars)")

    print("  [letf] Fetching SOXL/SOXX pair...", end=" ", flush=True)
    r_soxl = _pair_returns("SOXL", "SOXX", LETF_COST_SOXL_DAILY, start, spy_cache)
    print(f"done ({len(r_soxl)} bars)")

    print("  [letf] Fetching UPRO/SPY pair...", end=" ", flush=True)
    r_upro = _pair_returns("UPRO", "SPY", LETF_COST_UPRO_DAILY, start, spy_cache)
    print(f"done ({len(r_upro)} bars)")

    # Equal-weight on common dates
    combo = pd.DataFrame({
        "r_tqqq": r_tqqq,
        "r_soxl": r_soxl,
        "r_upro": r_upro,
    }).dropna()

    r_letf = combo.mean(axis=1)
    r_letf.name = "r_letf"
    print(
        f"  [letf] 3-pair combo: {len(r_letf)} bars "
        f"({r_letf.index[0].date()} -> {r_letf.index[-1].date()})"
    )
    return r_letf


# ---------------------------------------------------------------------------
# Leg 3: RTMV (Phase 52a + 57) — load from snapshots or run quick backtest
# ---------------------------------------------------------------------------

def build_rtmv_daily(start: str = COMMON_START, output_dir: Path | None = None) -> pd.Series:
    """Return daily RTMV portfolio returns.

    Loads from ``results/rtmv_live_5asset/snapshots.parquet`` if it exists and
    has data after *start*.  Otherwise runs a quick backtest (warns it is slower).

    Returns
    -------
    pd.Series
        Daily return series from equity pct_change, index = tz-naive dates.
    """
    snap_path = ROOT / "results" / "rtmv_live_5asset" / "snapshots.parquet"

    if snap_path.exists():
        print(f"  [rtmv] Loading from {snap_path}...", end=" ", flush=True)
        snaps = pd.read_parquet(snap_path)
        # Normalise date column (could be 'date' or datetime index)
        if "date" in snaps.columns:
            snaps = snaps.set_index("date")
        if not isinstance(snaps.index, pd.DatetimeIndex):
            snaps.index = pd.to_datetime(snaps.index)
        snaps.index = snaps.index.normalize().tz_localize(None)
        snaps = snaps[~snaps.index.duplicated(keep="last")]
        eq = snaps["equity"].dropna().sort_index()
        if len(eq) > 50:
            r_rtmv = eq.pct_change().dropna()
            r_rtmv = np.log1p(r_rtmv)  # convert to log returns
            r_rtmv = r_rtmv[r_rtmv.index >= pd.Timestamp(start).normalize()]
            r_rtmv.name = "r_rtmv"
            print(f"done. {len(r_rtmv)} bars ({r_rtmv.index[0].date()} -> {r_rtmv.index[-1].date()})")
            return r_rtmv

    # Fallback: run backtest
    print("  [rtmv] snapshots.parquet not found or insufficient — running backtest...", flush=True)
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
    asset_returns = pd.DataFrame({a: feat_dfs[a]["log_return"] for a in assets}, index=common_dates)

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
        output_dir=output_dir or ROOT / "results" / "rtmv_live_5asset",
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
# Tearsheet printer
# ---------------------------------------------------------------------------

def _print_tearsheet(
    label: str,
    r: pd.Series,
    indent: str = "  ",
) -> dict[str, float]:
    m = _portfolio_metrics(r)
    years = _per_year_returns(r)
    print(f"\n{indent}{label}")
    print(f"{indent}  Sharpe      : {m['sharpe']:.4f}")
    print(f"{indent}  Ann Return  : {m['ann_return']:.2%}")
    print(f"{indent}  Ann Vol     : {m['ann_vol']:.2%}")
    print(f"{indent}  Max DD      : {m['mdd']:.2%}")
    print(f"{indent}  Calmar      : {m['calmar']:.4f}")
    print(f"{indent}  Cum Return  : {m['cum_return']:.2%}")
    print(f"{indent}  Period      : {r.dropna().index[0].date()} -> {r.dropna().index[-1].date()}  ({m['n']} days)")
    print(f"{indent}  Per-year:")
    for yr, ret in sorted(years.items()):
        marker = " !" if ret < 0 else ""
        print(f"{indent}    {yr}  {ret:+.2%}{marker}")
    return m


# ---------------------------------------------------------------------------
# Mode: backtest
# ---------------------------------------------------------------------------

def run_backtest(
    carry_weight: float,
    letf_weight: float,
    rtmv_weight: float,
    output_dir: Path,
) -> None:
    """Run full backtest over common date range, print tearsheet, write outputs."""

    total_weight = carry_weight + letf_weight + rtmv_weight
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {total_weight:.4f}")

    print("=" * 70)
    print("PHASE 77: UNIFIED MULTI-STRATEGY PORTFOLIO BACKTEST (PHASE 75+76 CARRY UPGRADE)")
    print(f"  Allocation: carry={carry_weight:.0%}  letf={letf_weight:.0%}  rtmv={rtmv_weight:.0%}")
    print(f"  LETF leg: 3-pair equal-weight (TQQQ/QQQ + SOXL/SOXX + UPRO/SPY)")
    print(f"  Common start: {COMMON_START}")
    print("=" * 70)

    print("\n[1/3] Building carry P&L (BTC+ETH, carry-weighted, bull-only regime)...")
    r_carry = build_carry_daily(start=COMMON_START)

    print("\n[2/3] Building LETF 3-pair combo P&L (Phase 71: TQQQ/QQQ + SOXL/SOXX + UPRO/SPY)...")
    r_letf = build_letf_daily(start=COMMON_START)

    print("\n[3/3] Loading RTMV P&L (SPY/GLD/SHY/IEF/TLT, spy_rank_bull, mom=0.03)...")
    r_rtmv = build_rtmv_daily(start=COMMON_START)

    # Align all three to common index
    combined = pd.DataFrame({
        "r_carry": r_carry,
        "r_letf": r_letf,
        "r_rtmv": r_rtmv,
    }).dropna()

    print(f"\nCommon date range: {combined.index[0].date()} -> {combined.index[-1].date()}")
    print(f"Common bars: {len(combined)}")

    # Combined return
    combined["r_combined"] = (
        carry_weight * combined["r_carry"]
        + letf_weight * combined["r_letf"]
        + rtmv_weight * combined["r_rtmv"]
    )

    sep = "=" * 70
    print(f"\n{sep}")
    print("INDIVIDUAL STRATEGY TEARSHEETS")
    print(sep)

    m_carry  = _print_tearsheet("Carry (BTC+ETH carry-weighted + bull-only)", combined["r_carry"])
    m_letf   = _print_tearsheet("LETF 3-Pair Combo (TQQQ/QQQ + SOXL/SOXX + UPRO/SPY)", combined["r_letf"])
    m_rtmv   = _print_tearsheet("RTMV + Momentum (SPY/GLD/SHY/IEF/TLT)", combined["r_rtmv"])

    print(f"\n{sep}")
    print(f"COMBINED PORTFOLIO ({carry_weight:.0%}/{letf_weight:.0%}/{rtmv_weight:.0%})")
    print(sep)
    m_combined = _print_tearsheet(
        f"Combined Phase 71 (carry={carry_weight:.0%} letf={letf_weight:.0%} rtmv={rtmv_weight:.0%})",
        combined["r_combined"],
    )

    # Comparison benchmarks
    print(f"\n{sep}")
    print("COMPARISON: EQUAL-WEIGHT vs SINGLE-STRATEGY")
    print(sep)
    r_eq = (1/3) * combined["r_carry"] + (1/3) * combined["r_letf"] + (1/3) * combined["r_rtmv"]
    _print_tearsheet("Equal-weight (33%/33%/34%)", r_eq)
    _print_tearsheet("Carry only (100%)", combined["r_carry"])
    _print_tearsheet("LETF pair only (100%)", combined["r_letf"])
    _print_tearsheet("RTMV only (100%)", combined["r_rtmv"])

    # Cross-strategy correlations
    print(f"\n{sep}")
    print("CROSS-STRATEGY CORRELATIONS")
    print(sep)
    corr_carry_letf = float(combined["r_carry"].corr(combined["r_letf"]))
    corr_carry_rtmv = float(combined["r_carry"].corr(combined["r_rtmv"]))
    corr_letf_rtmv  = float(combined["r_letf"].corr(combined["r_rtmv"]))
    print(f"  carry↔letf : {corr_carry_letf:+.4f}")
    print(f"  carry↔rtmv : {corr_carry_rtmv:+.4f}")
    print(f"  letf↔rtmv  : {corr_letf_rtmv:+.4f}")

    # GO/NO-GO verdict
    sharpe_combined = m_combined["sharpe"]
    go = sharpe_combined >= GO_THRESHOLD
    verdict = "GO" if go else "NO-GO"
    print(f"\n{sep}")
    print(f"GO/NO-GO VERDICT")
    print(sep)
    print(f"  Combined Sharpe  : {sharpe_combined:.4f}")
    print(f"  GO threshold     : {GO_THRESHOLD:.1f}")
    print(f"  Max DD           : {m_combined['mdd']:.2%}")
    print(f"  Verdict          : {verdict}")
    print(sep)

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    today_str = date.today().strftime("%Y%m%d")

    # Write returns parquet
    parquet_path = output_dir / f"returns_{today_str}.parquet"
    combined[["r_carry", "r_letf", "r_rtmv", "r_combined"]].to_parquet(parquet_path)
    print(f"\nReturns written to: {parquet_path}")

    # Write tearsheet markdown
    years_combined = _per_year_returns(combined["r_combined"])
    years_carry    = _per_year_returns(combined["r_carry"])
    years_letf     = _per_year_returns(combined["r_letf"])
    years_rtmv     = _per_year_returns(combined["r_rtmv"])
    all_years = sorted(set(years_combined) | set(years_carry) | set(years_letf) | set(years_rtmv))

    yearly_rows = "\n".join(
        f"| {yr} | {years_combined.get(yr, float('nan')):+.2%} | "
        f"{years_carry.get(yr, float('nan')):+.2%} | "
        f"{years_letf.get(yr, float('nan')):+.2%} | "
        f"{years_rtmv.get(yr, float('nan')):+.2%} |"
        for yr in all_years
    )

    md_path = output_dir / f"tearsheet_{today_str}.md"
    md_content = f"""# Phase 77: Unified Multi-Strategy Portfolio (Phase 75+76 Carry Upgrade)

**Date:** {date.today().isoformat()}
**Allocation:** carry={carry_weight:.0%} / letf={letf_weight:.0%} / rtmv={rtmv_weight:.0%}
**Verdict: {verdict}** (threshold: Sharpe ≥ {GO_THRESHOLD:.2f})

## Strategy Overview

| Strategy | Description | Phase |
|---|---|---|
| Crypto Carry | BTC+ETH spot_spread_dynamic + momentum_filter + bull-only regime scale | Phase 77 (75+76 GO) |
| LETF 3-Pair Combo | SHORT TQQQ/QQQ + SOXL/SOXX + UPRO/SPY, equal-weight, always-in | Phase 70 GO |
| RTMV + Momentum | SPY/GLD/SHY/IEF/TLT, spy_rank_bull, mom=0.03 | Phase 57 GO |

Common period: {combined.index[0].date()} → {combined.index[-1].date()} ({len(combined)} days)

## Combined Portfolio Metrics

| Metric | Combined | Carry | LETF | RTMV |
|---|---|---|---|---|
| Sharpe | **{m_combined['sharpe']:.4f}** | {m_carry['sharpe']:.4f} | {m_letf['sharpe']:.4f} | {m_rtmv['sharpe']:.4f} |
| Ann Return | {m_combined['ann_return']:.2%} | {m_carry['ann_return']:.2%} | {m_letf['ann_return']:.2%} | {m_rtmv['ann_return']:.2%} |
| Ann Vol | {m_combined['ann_vol']:.2%} | {m_carry['ann_vol']:.2%} | {m_letf['ann_vol']:.2%} | {m_rtmv['ann_vol']:.2%} |
| Max DD | {m_combined['mdd']:.2%} | {m_carry['mdd']:.2%} | {m_letf['mdd']:.2%} | {m_rtmv['mdd']:.2%} |
| Calmar | {m_combined['calmar']:.4f} | {m_carry['calmar']:.4f} | {m_letf['calmar']:.4f} | {m_rtmv['calmar']:.4f} |
| Cum Return | {m_combined['cum_return']:.2%} | {m_carry['cum_return']:.2%} | {m_letf['cum_return']:.2%} | {m_rtmv['cum_return']:.2%} |

## Cross-Strategy Correlations

| Pair | Correlation |
|---|---|
| carry↔letf | {corr_carry_letf:+.4f} |
| carry↔rtmv | {corr_carry_rtmv:+.4f} |
| letf↔rtmv | {corr_letf_rtmv:+.4f} |

## Per-Year Returns

| Year | Combined | Carry | LETF | RTMV |
|---|---|---|---|---|
{yearly_rows}

## GO/NO-GO Criteria

| Criterion | Threshold | Result | Status |
|---|---|---|---|
| Combined Sharpe | ≥ {GO_THRESHOLD:.1f} | {sharpe_combined:.4f} | {"PASS" if go else "FAIL"} |

## Notes

- Carry: BTC+ETH 8-hourly funding rates aggregated to daily. Carry-weighted allocation,
  bull-only SPY HMM regime scale {'{0:1.0, 1:1.0, 2:1.5}'}. ~0.5%/yr friction.
- LETF (Phase 71 3-pair combo):
    TQQQ/QQQ:  r_pair = 3×r_QQQ - r_TQQQ - {LETF_COST_TQQQ_DAILY*ANN_FACTOR:.2%}/yr. Always in.
    SOXL/SOXX: r_pair = 3×r_SOXX - r_SOXL - {LETF_COST_SOXL_DAILY*ANN_FACTOR:.2%}/yr. Always in.
    UPRO/SPY:  r_pair = 3×r_SPY  - r_UPRO - {LETF_COST_UPRO_DAILY*ANN_FACTOR:.2%}/yr. Always in.
    Equal-weight combination (1/3 each). Returns clipped to [-50%, +50%] per day.
- RTMV: loaded from results/rtmv_live_5asset/snapshots.parquet (Phase 57 backtest output).
  Daily equity pct_change converted to log returns.
"""
    md_path.write_text(md_content, encoding="utf-8")
    print(f"Tearsheet written to: {md_path}")


# ---------------------------------------------------------------------------
# Mode: signal
# ---------------------------------------------------------------------------

def run_signal() -> None:
    """Print today's combined portfolio signal from all three strategies.

    Phase 79: updated to show Phase 75-78 carry filter status:
    - momentum filter active (lag-1 combined rate check)
    - spot-spread-weight ratio (instantaneous ETH/BTC carry split)
    """
    from rde.trading.carry_executor import _fetch_latest_funding_rate, _PERIODS_PER_YEAR

    print("\n=== PHASE 77 MULTISTRAT SIGNAL (Phase 79 hardened) ===\n")
    today = date.today()
    print(f"  Date      : {today}")
    print(f"  Config    : 80% carry (Phase 75+76+78) + 15% LETF 3-pair + 5% RTMV")

    # --- Carry signal (Phase 75+76+78) ---
    print("\n  [1] CARRY (BTC+ETH) — Phase 75 momentum filter + Phase 76 spot-spread-weight:")
    rates_live: dict[str, float] = {}
    for sym in ["BTCUSDT", "ETHUSDT"]:
        try:
            rate = _fetch_latest_funding_rate(sym)
            rates_live[sym] = rate
            ann = rate * _PERIODS_PER_YEAR
            signal = "IN (positive carry)" if ann > 0.05 else ("EXIT (negative carry)" if ann < -0.02 else "HOLD (low carry)")
            print(f"       {sym:10s}: rate={rate:+.5f}  ann={ann:+.1%}  -> {signal}")
        except Exception as exc:
            print(f"       {sym:10s}: ERROR — {exc}")

    # Show Phase 76 instantaneous ETH/BTC spot-spread weight
    if len(rates_live) == 2:
        r_btc = max(0.0, rates_live.get("BTCUSDT", 0.0))
        r_eth = max(0.0, rates_live.get("ETHUSDT", 0.0))
        denom = r_btc + r_eth
        if denom > 0:
            raw_w_eth = r_eth / denom
            w_eth = min(0.80, max(0.40, raw_w_eth))
            w_btc = 1.0 - w_eth
        else:
            w_eth = w_btc = 0.5
        print(f"       Spot-spread weight : ETH={w_eth:.1%}  BTC={w_btc:.1%}  (Phase 76)")
        combined_rate = (rates_live.get("BTCUSDT", 0.0) + rates_live.get("ETHUSDT", 0.0)) * 0.5
        print(f"       Combined rate now  : {combined_rate:+.6f}  ann={combined_rate * _PERIODS_PER_YEAR:+.1%}")
        print(f"       Momentum filter    : active (skip next period if this rate ≤ 0)  [Phase 75]")
        print(f"       Robustness         : Phase 78 ROBUST — both half-periods Sharpe >15, p=0.0000")

    # --- LETF signal (Phase 71: 3-pair combo) ---
    print("\n  [2] LETF 3-PAIR COMBO (Phase 71: TQQQ/QQQ + SOXL/SOXX + UPRO/SPY):")
    letf_pair_pnl: list[float] = []
    for ticker_3x, ticker_1x, cost_daily, label in [
        ("TQQQ", "QQQ",  LETF_COST_TQQQ_DAILY, "TQQQ/QQQ  "),
        ("SOXL", "SOXX", LETF_COST_SOXL_DAILY, "SOXL/SOXX "),
        ("UPRO", "SPY",  LETF_COST_UPRO_DAILY, "UPRO/SPY  "),
    ]:
        try:
            df_3x = _flatten(yf.download(ticker_3x, period="5d", interval="1d", progress=False, auto_adjust=True))
            df_1x = _flatten(yf.download(ticker_1x, period="5d", interval="1d", progress=False, auto_adjust=True))
            if len(df_3x) >= 2 and len(df_1x) >= 2:
                r_3x_today = float(np.log(df_3x["Close"].iloc[-1] / df_3x["Close"].iloc[-2]))
                r_1x_today = float(np.log(df_1x["Close"].iloc[-1] / df_1x["Close"].iloc[-2]))
                r_pair_today = 3.0 * r_1x_today - r_3x_today - cost_daily
                letf_pair_pnl.append(r_pair_today)
                print(f"       {label}: {ticker_3x} {r_3x_today:+.2%}, {ticker_1x} {r_1x_today:+.2%}  "
                      f"→ pair {r_pair_today:+.2%}")
            else:
                print(f"       {label}: Insufficient data (market may be closed)")
        except Exception as exc:
            print(f"       {label}: ERROR — {exc}")
    if letf_pair_pnl:
        combo_today = float(np.mean(letf_pair_pnl))
        print(f"       Equal-weight combo P&L : {combo_today:+.2%}")
        print(f"       Status                 : IN POSITION (always-in, all 3 pairs)")

    # --- RTMV signal ---
    print("\n  [3] RTMV (SPY/GLD/SHY/IEF/TLT):")
    snap_path = ROOT / "results" / "rtmv_live_5asset" / "snapshots.parquet"
    if snap_path.exists():
        try:
            snaps = pd.read_parquet(snap_path)
            if "date" in snaps.columns:
                snaps = snaps.set_index("date")
            if not isinstance(snaps.index, pd.DatetimeIndex):
                snaps.index = pd.to_datetime(snaps.index)
            snaps.index = snaps.index.normalize()
            last = snaps.iloc[-1]
            print(f"       Last date    : {snaps.index[-1].date()}")
            if "equity" in snaps.columns:
                print(f"       Equity       : {float(last['equity']):.2f}")
            if "current_weights" in snaps.columns:
                print(f"       Weights      : {last['current_weights']}")
            if "n_rebalances" in snaps.columns:
                print(f"       Rebalances   : {int(last['n_rebalances'])}")
        except Exception as exc:
            print(f"       ERROR reading snapshots: {exc}")
    else:
        print(f"       No snapshots found at {snap_path}")
        print("       Run: make backtest-rtmv-5asset to generate")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 68: Unified multi-strategy live executor (carry+LETF+RTMV)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["backtest", "signal"], default="signal",
        help="Execution mode",
    )
    parser.add_argument("--carry-weight", type=float, default=0.80,
                        help="Carry strategy weight (default: 0.80)")
    parser.add_argument("--letf-weight",  type=float, default=0.15,
                        help="LETF pair strategy weight (default: 0.15)")
    parser.add_argument("--rtmv-weight",  type=float, default=0.05,
                        help="RTMV strategy weight (default: 0.05)")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "phase68_multistrat",
                        help="Output directory for results")
    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest(
            carry_weight=args.carry_weight,
            letf_weight=args.letf_weight,
            rtmv_weight=args.rtmv_weight,
            output_dir=args.output_dir,
        )
    elif args.mode == "signal":
        run_signal()


if __name__ == "__main__":
    main()
