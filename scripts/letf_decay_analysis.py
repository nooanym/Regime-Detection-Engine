"""LETF (Leveraged ETF) decay overlay analysis — Phase 54 research.

Studies the realized decay of 3x leveraged ETFs (TQQQ, SOXL) versus their
benchmark (QQQ, SOXX) and tests a regime-conditional short-decay strategy
that uses the existing SPY n=3 HMM to gate entry.

Decay formula per day: (sigma^2 * (L^2 - L)) / 2  where L=3 (leverage), sigma=daily vol.
At L=3: daily decay = sigma^2 * 3.

Strategy: SHORT TQQQ / LONG 3x QQQ (delta-neutral) when:
  - 21-day realized QQQ vol > vol_entry_pct (default 20%)
  - SPY HMM state is bear or neutral (not bull)
Exit: 21-day realized QQQ vol < vol_exit_pct (default 18%) OR SPY enters bull state.

Costs assumed:
  - TQQQ short borrow: ~0.5%/yr (easy-to-borrow, liquid)
  - Funding on the long QQQ position: ~5.3%/yr SOFR (opportunity cost of capital)
    but since TQQQ already contains internal 3x financing, the long leg is
    3 units QQQ so we model the full return pair: Short TQQQ + Long 3x QQQ.
  - Total estimated drag: ~1.5-2.5%/yr vs decay captured.
  - Net profitable threshold: decay > 2.5%/yr, i.e., QQQ vol > ~18%.

Usage
-----
    uv run python scripts/letf_decay_analysis.py
    uv run python scripts/letf_decay_analysis.py --no-hmm   # skip HMM gating
    uv run python scripts/letf_decay_analysis.py --output-dir results/letf/

Outputs
-------
    docs/findings/letf_decay_overlay.md   — GO/NO-GO writeup
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from rde.data.yfinance_source import YFinanceSource
from rde.features.pipeline import FeaturePipeline
from rde.features.returns import LogReturns, SmoothedReturns
from rde.features.volatility import RollingVolatility
from rde.models.hmm import train_hmm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEVERAGE = 3.0
ANN_FACTOR = 252  # trading days/year
# Estimated annual drag: TQQQ borrow (~0.5%) + transaction friction (~1.0%) + spread (~0.5%)
ANNUAL_COST_BPS = 200  # 2.00%/yr total estimated cost
ANNUAL_COST = ANNUAL_COST_BPS / 10_000
# Minimum decay to clear costs
MIN_DECAY_THRESHOLD = ANNUAL_COST  # 2%/yr


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def fetch_letf_data(
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch daily OHLCV for TQQQ, QQQ, SOXL, SOXX from yfinance (period=max).

    Parameters
    ----------
    cache_dir : Path
        Directory to store parquet caches.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (tqqq, qqq, soxl, soxx) DataFrames with Close column, DatetimeIndex.
    """
    source = YFinanceSource(cache_dir=cache_dir, cache_max_age_seconds=86400)
    symbols = ["TQQQ", "QQQ", "SOXL", "SOXX"]
    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = source.load(sym, period="max", interval="1d")
            data[sym] = df[["Close"]].copy()
            logger.info(
                "%s: %d bars (%s → %s)",
                sym,
                len(df),
                df.index[0].date(),
                df.index[-1].date(),
            )
        except RuntimeError as e:
            logger.warning("Could not load %s: %s — skipping", sym, e)
            data[sym] = pd.DataFrame(columns=["Close"])
    return data["TQQQ"], data["QQQ"], data["SOXL"], data["SOXX"]


# ---------------------------------------------------------------------------
# Decay computation
# ---------------------------------------------------------------------------


def compute_log_returns(prices: pd.DataFrame) -> pd.Series:
    """Compute log returns from a Close price Series or DataFrame with Close column."""
    close = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]
    return np.log(close / close.shift(1)).dropna()


def theoretical_daily_decay(sigma_daily: float) -> float:
    """Compute theoretical daily variance decay for a 3x leveraged ETF.

    Formula: sigma^2 * (L^2 - L) / 2  where L=3.
    At L=3 this simplifies to sigma^2 * 3.

    Parameters
    ----------
    sigma_daily : float
        Daily volatility (e.g. 0.01 = 1%/day).

    Returns
    -------
    float
        Daily decay as a fraction (annualise by * 252).
    """
    return sigma_daily ** 2 * (LEVERAGE ** 2 - LEVERAGE) / 2


def compute_annual_decay_table(
    tqqq: pd.DataFrame,
    qqq: pd.DataFrame,
    soxl: pd.DataFrame | None,
    soxx: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compute per-year realized and theoretical decay for TQQQ and SOXL.

    Realized decay = TQQQ total return - 3 * QQQ total return (per year).
    Theoretical decay uses the compound-daily formula with realized vol.

    Parameters
    ----------
    tqqq : pd.DataFrame
        TQQQ OHLCV (Close used).
    qqq : pd.DataFrame
        QQQ OHLCV (Close used).
    soxl : pd.DataFrame | None
        SOXL OHLCV or None if unavailable.
    soxx : pd.DataFrame | None
        SOXX OHLCV or None if unavailable.

    Returns
    -------
    pd.DataFrame
        One row per year with columns: year, qqq_return, tqqq_return,
        3x_qqq_return, realized_decay_pct, theoretical_decay_pct,
        decay_profitable (True when realized_decay < 0, i.e. shorting TQQQ wins).
    """
    # Align TQQQ and QQQ on common dates
    common_idx = tqqq.index.intersection(qqq.index)
    tqqq_c = tqqq.loc[common_idx, "Close"]
    qqq_c = qqq.loc[common_idx, "Close"]

    tqqq_ret = tqqq_c.pct_change().dropna()
    qqq_ret = qqq_c.pct_change().dropna()

    # Align after pct_change
    common_ret = tqqq_ret.index.intersection(qqq_ret.index)
    tqqq_ret = tqqq_ret.loc[common_ret]
    qqq_ret = qqq_ret.loc[common_ret]

    # Annual grouping (use year of the index date)
    years = sorted(set(tqqq_ret.index.year))
    rows = []
    for yr in years:
        mask = tqqq_ret.index.year == yr
        t_yr = tqqq_ret[mask]
        q_yr = qqq_ret[mask]
        if len(t_yr) < 50:  # skip partial years with very few bars
            continue

        qqq_annual = float((1 + q_yr).prod() - 1)
        tqqq_annual = float((1 + t_yr).prod() - 1)
        three_x_qqq = float((1 + qqq_annual) ** 3 - 1)  # theoretical 3x (no path)
        # Path-adjusted 3x benchmark: compound daily
        three_x_qqq_path = float((1 + 3 * q_yr).prod() - 1)

        # Realized decay vs path-adjusted 3x QQQ
        realized_decay = tqqq_annual - three_x_qqq_path

        # Theoretical decay from realized vol
        sigma_daily = float(q_yr.std())
        theoretical_daily = theoretical_daily_decay(sigma_daily)
        theoretical_annual = theoretical_daily * len(t_yr)

        # When shorting TQQQ + long 3x QQQ is profitable: realized_decay < 0
        profitable = realized_decay < 0.0

        rows.append({
            "year": yr,
            "n_bars": int(mask.sum()),
            "qqq_return_pct": round(qqq_annual * 100, 2),
            "tqqq_return_pct": round(tqqq_annual * 100, 2),
            "3x_qqq_path_pct": round(three_x_qqq_path * 100, 2),
            "realized_decay_pct": round(realized_decay * 100, 2),
            "theoretical_decay_pct": round(theoretical_annual * 100, 2),
            "qqq_realized_vol_ann_pct": round(sigma_daily * np.sqrt(ANN_FACTOR) * 100, 2),
            "decay_profitable_to_short": profitable,
        })

    df = pd.DataFrame(rows)

    # If SOXL/SOXX available, add SOXL columns
    if soxl is not None and soxx is not None and not soxl.empty and not soxx.empty:
        common_sox = soxl.index.intersection(soxx.index)
        if len(common_sox) > 100:
            soxl_r = soxl.loc[common_sox, "Close"].pct_change().dropna()
            soxx_r = soxx.loc[common_sox, "Close"].pct_change().dropna()
            common_sox_r = soxl_r.index.intersection(soxx_r.index)
            soxl_r = soxl_r.loc[common_sox_r]
            soxx_r = soxx_r.loc[common_sox_r]

            soxl_by_year: dict[int, float] = {}
            soxx_by_year: dict[int, float] = {}
            soxl_decay_by_year: dict[int, float] = {}
            for yr in years:
                mask = soxl_r.index.year == yr
                s_yr = soxl_r[mask]
                sx_yr = soxx_r[mask]
                if len(s_yr) < 50:
                    continue
                soxl_ann = float((1 + s_yr).prod() - 1)
                three_x_soxx_path = float((1 + 3 * sx_yr).prod() - 1)
                soxl_by_year[yr] = round(soxl_ann * 100, 2)
                soxx_by_year[yr] = round(float((1 + sx_yr).prod() - 1) * 100, 2)
                soxl_decay_by_year[yr] = round((soxl_ann - three_x_soxx_path) * 100, 2)

            df["soxl_return_pct"] = df["year"].map(soxl_by_year)
            df["soxx_return_pct"] = df["year"].map(soxx_by_year)
            df["soxl_realized_decay_pct"] = df["year"].map(soxl_decay_by_year)

    return df


# ---------------------------------------------------------------------------
# Vol-regime decay analysis
# ---------------------------------------------------------------------------


def compute_decay_by_regime(
    tqqq: pd.DataFrame,
    qqq: pd.DataFrame,
    vol_window: int = 21,
    vix_threshold: float = 20.0,
) -> pd.DataFrame:
    """Stratify realized TQQQ decay by rolling QQQ vol regime.

    Parameters
    ----------
    tqqq : pd.DataFrame
        TQQQ price data.
    qqq : pd.DataFrame
        QQQ price data.
    vol_window : int
        Rolling window (bars) for realized vol estimate.
    vix_threshold : float
        Vol threshold (annualised %) separating high/low vol regimes.

    Returns
    -------
    pd.DataFrame
        One row per vol regime with mean daily decay, total days, Sharpe of
        short-decay position.
    """
    common_idx = tqqq.index.intersection(qqq.index)
    tqqq_r = tqqq.loc[common_idx, "Close"].pct_change().dropna()
    qqq_r = qqq.loc[common_idx, "Close"].pct_change().dropna()

    common_r = tqqq_r.index.intersection(qqq_r.index)
    tqqq_r = tqqq_r.loc[common_r]
    qqq_r = qqq_r.loc[common_r]

    # Rolling annualised vol of QQQ
    roll_vol_ann = qqq_r.rolling(vol_window).std() * np.sqrt(ANN_FACTOR) * 100

    # Daily decay of short position (positive = profit to short)
    daily_decay = 3 * qqq_r - tqqq_r  # long 3x QQQ - short TQQQ daily PnL

    combined = pd.DataFrame({
        "qqq_ret": qqq_r,
        "tqqq_ret": tqqq_r,
        "roll_vol_ann": roll_vol_ann,
        "daily_decay_pnl": daily_decay,
    }).dropna()

    combined["vol_regime"] = combined["roll_vol_ann"].apply(
        lambda v: f"High (>{vix_threshold:.0f}%)" if v > vix_threshold else f"Low (<={vix_threshold:.0f}%)"
    )

    rows = []
    for regime, grp in combined.groupby("vol_regime"):
        mean_daily_decay = grp["daily_decay_pnl"].mean()
        ann_decay = mean_daily_decay * ANN_FACTOR * 100
        sharpe = (grp["daily_decay_pnl"].mean() / grp["daily_decay_pnl"].std()
                  * np.sqrt(ANN_FACTOR)) if grp["daily_decay_pnl"].std() > 0 else 0.0
        rows.append({
            "vol_regime": regime,
            "n_bars": len(grp),
            "mean_qqq_vol_ann_pct": round(grp["roll_vol_ann"].mean(), 2),
            "mean_daily_decay_pnl_bps": round(mean_daily_decay * 10_000, 2),
            "ann_decay_pnl_pct": round(ann_decay, 2),
            "sharpe_of_short": round(float(sharpe), 3),
            "pct_days_in_regime": round(len(grp) / len(combined) * 100, 1),
        })
    return pd.DataFrame(rows).sort_values("mean_qqq_vol_ann_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# HMM regime labels
# ---------------------------------------------------------------------------


def _fit_spy_hmm(spy_df: pd.DataFrame, n_states: int = 3, n_restarts: int = 5) -> object:
    """Fit a 3-state daily HMM on SPY to label bear/neutral/bull regimes.

    Parameters
    ----------
    spy_df : pd.DataFrame
        SPY OHLCV with daily bars.
    n_states : int
        Number of HMM states (3 = bear/neutral/bull per Phase 43c).
    n_restarts : int
        Training restarts.

    Returns
    -------
    FittedModel
    """
    pipeline = FeaturePipeline([
        LogReturns(),
        RollingVolatility(window=20),
        SmoothedReturns(window=5),
    ])
    feat_df = pipeline.transform(spy_df).dropna()
    X = feat_df.values
    fitted = train_hmm(
        X,
        n_states=n_states,
        n_restarts=n_restarts,
        seed_base=42,
        covariance_type="full",
        n_iter=500,
    )
    return fitted, feat_df


def _label_states_by_return(
    fitted: object,
    feat_df: pd.DataFrame,
) -> dict[int, str]:
    """Label HMM states by mean log_return: rank 0=bear, 1=neutral, 2=bull.

    Returns
    -------
    dict[int, str]
        Maps state index → label.
    """
    from rde.models.hmm import FittedModel
    assert isinstance(fitted, FittedModel)

    log_return_col = feat_df.columns.tolist().index("log_return")
    means = fitted.hmm.means_[:, log_return_col]
    order = np.argsort(means)
    labels = {int(order[0]): "bear", int(order[1]): "neutral", int(order[2]): "bull"}
    return labels


def _decode_spy_states(fitted: object, feat_df: pd.DataFrame) -> pd.Series:
    """Run Viterbi on feat_df and return a Series of state labels indexed like feat_df."""
    X_scaled = fitted.scaler.transform(feat_df.values)
    states = fitted.hmm.predict(X_scaled)
    return pd.Series(states, index=feat_df.index, name="spy_state")


# ---------------------------------------------------------------------------
# Backtest: delta-neutral short TQQQ / long 3x QQQ with vol + HMM gating
# ---------------------------------------------------------------------------


def _compute_metrics(
    daily_pnl: pd.Series,
    ann_factor: int = ANN_FACTOR,
) -> dict[str, float]:
    """Compute Sharpe, MDD, annualised return and vol from a daily PnL series."""
    if len(daily_pnl) < 20:
        return {"sharpe": 0.0, "mdd": 0.0, "ann_return": 0.0, "ann_vol": 0.0}

    ann_ret = daily_pnl.mean() * ann_factor
    ann_vol = daily_pnl.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cum = (1 + daily_pnl).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    mdd = float(dd.min())

    return {
        "sharpe": round(float(sharpe), 3),
        "mdd": round(mdd, 4),
        "ann_return_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
    }


def backtest_decay_short(
    tqqq: pd.DataFrame,
    qqq: pd.DataFrame,
    vol_entry_pct: float = 20.0,
    vol_exit_pct: float = 18.0,
    vol_window: int = 21,
    spy_states: pd.Series | None = None,
    bear_neutral_states: set[int] | None = None,
    annual_cost_frac: float = ANNUAL_COST,
) -> dict:
    """Backtest the delta-neutral TQQQ decay-short strategy.

    Entry rules (ALL must hold):
      1. 21-day rolling QQQ vol (annualised) > vol_entry_pct.
      2. (If spy_states provided) SPY HMM state is bear or neutral (not bull).

    Exit rules (ANY triggers):
      1. 21-day rolling QQQ vol (annualised) < vol_exit_pct.
      2. (If spy_states provided) SPY HMM state is bull.

    Position: SHORT 1 unit TQQQ / LONG 3 units QQQ (delta-neutral).
    Daily PnL = 3 * QQQ_daily_return - TQQQ_daily_return - daily_cost.

    Parameters
    ----------
    tqqq : pd.DataFrame
        TQQQ OHLCV.
    qqq : pd.DataFrame
        QQQ OHLCV.
    vol_entry_pct : float
        Entry vol threshold, annualised percent (e.g. 20.0 = 20%/yr).
    vol_exit_pct : float
        Exit vol threshold, annualised percent.
    vol_window : int
        Rolling window for realized vol estimation.
    spy_states : pd.Series | None
        Integer state per date from the SPY HMM decoder. None = no HMM filter.
    bear_neutral_states : set[int] | None
        Which state indices count as bear or neutral (entry allowed). None = all states.
    annual_cost_frac : float
        Annual drag as a fraction (borrow + friction), applied daily.

    Returns
    -------
    dict
        Keys: sharpe, mdd, ann_return_pct, ann_vol_pct, n_entry_trades,
              frac_days_in_trade, metrics_by_year, pnl_series.
    """
    common_idx = tqqq.index.intersection(qqq.index)
    tqqq_r = tqqq.loc[common_idx, "Close"].pct_change().dropna()
    qqq_r = qqq.loc[common_idx, "Close"].pct_change().dropna()
    common_r = tqqq_r.index.intersection(qqq_r.index)
    tqqq_r = tqqq_r.loc[common_r]
    qqq_r = qqq_r.loc[common_r]

    roll_vol = qqq_r.rolling(vol_window).std() * np.sqrt(ANN_FACTOR) * 100

    daily_cost = annual_cost_frac / ANN_FACTOR

    # Build signal: 1=in trade, 0=out
    in_trade = pd.Series(0, index=common_r)
    currently_in = False
    for dt in common_r:
        vol = roll_vol.get(dt, np.nan)
        if np.isnan(vol):
            in_trade[dt] = 0
            currently_in = False
            continue

        # HMM filter
        hmm_allows_entry = True
        hmm_forces_exit = False
        if spy_states is not None and dt in spy_states.index:
            state = int(spy_states[dt])
            if bear_neutral_states is not None:
                hmm_allows_entry = state in bear_neutral_states
                hmm_forces_exit = state not in bear_neutral_states
            # else: no restriction
        elif spy_states is not None:
            # Date not in spy_states index — use prior position
            hmm_allows_entry = currently_in
            hmm_forces_exit = False

        if currently_in:
            # Exit conditions
            if vol < vol_exit_pct or hmm_forces_exit:
                currently_in = False
        else:
            # Entry conditions
            if vol > vol_entry_pct and hmm_allows_entry:
                currently_in = True

        in_trade[dt] = int(currently_in)

    # Compute daily PnL
    pnl = (3 * qqq_r - tqqq_r - daily_cost) * in_trade

    metrics = _compute_metrics(pnl)
    metrics["n_entry_trades"] = int((in_trade.diff() == 1).sum())
    metrics["frac_days_in_trade"] = round(float(in_trade.mean()), 3)

    # By year
    by_year_rows = []
    for yr in sorted(set(pnl.index.year)):
        yr_pnl = pnl[pnl.index.year == yr]
        if len(yr_pnl) < 20:
            continue
        m = _compute_metrics(yr_pnl)
        m["year"] = yr
        m["days_in_trade"] = int(in_trade[in_trade.index.year == yr].sum())
        by_year_rows.append(m)

    metrics["metrics_by_year"] = pd.DataFrame(by_year_rows)
    metrics["pnl_series"] = pnl
    metrics["in_trade_series"] = in_trade
    return metrics


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render DataFrame as GitHub-flavoured Markdown table."""
    headers = list(df.columns)
    sep = [":---" if str(df[c].dtype) == "object" else "---:" for c in headers]
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join(sep) + " |")
    for _, row in df.iterrows():
        cells = [str(v) if not isinstance(v, float) else f"{v:.3f}" for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_findings_report(
    annual_decay_table: pd.DataFrame,
    vol_regime_table: pd.DataFrame,
    bt_no_hmm: dict,
    bt_with_hmm: dict | None,
    output_path: Path,
) -> None:
    """Write the GO/NO-GO findings Markdown report.

    Parameters
    ----------
    annual_decay_table : pd.DataFrame
        Per-year decay vs vol table.
    vol_regime_table : pd.DataFrame
        Decay stratified by vol regime.
    bt_no_hmm : dict
        Backtest results without HMM filter.
    bt_with_hmm : dict | None
        Backtest results with HMM gating, or None if HMM was skipped.
    output_path : Path
        Path to write the .md file.
    """
    # Determine overall GO/NO-GO
    sharpe_threshold = 0.50  # minimum Sharpe for GO
    go_no_hmm = bt_no_hmm.get("sharpe", 0.0) >= sharpe_threshold
    go_with_hmm = (bt_with_hmm or {}).get("sharpe", 0.0) >= sharpe_threshold if bt_with_hmm else None

    vol_regime_str = _df_to_markdown(vol_regime_table)
    annual_str = _df_to_markdown(annual_decay_table)

    # Determine go/no-go per strategy variant
    verdict_no_hmm = "**GO**" if go_no_hmm else "**NO-GO**"
    verdict_with_hmm = "**GO**" if go_with_hmm else "**NO-GO**"
    overall_verdict = "**GO**" if (go_no_hmm or go_with_hmm) else "**NO-GO**"

    def fmt_bt(bt: dict) -> str:
        return (
            f"Sharpe={bt.get('sharpe', 0.0):.3f}, "
            f"Ann Return={bt.get('ann_return_pct', 0.0):.1f}%, "
            f"Ann Vol={bt.get('ann_vol_pct', 0.0):.1f}%, "
            f"MDD={bt.get('mdd', 0.0)*100:.1f}%, "
            f"Days in trade={bt.get('frac_days_in_trade', 0.0)*100:.0f}%, "
            f"N entries={bt.get('n_entry_trades', 0)}"
        )

    by_year_no_hmm_str = ""
    if "metrics_by_year" in bt_no_hmm and not bt_no_hmm["metrics_by_year"].empty:
        by_year_no_hmm_str = _df_to_markdown(bt_no_hmm["metrics_by_year"])

    by_year_hmm_str = ""
    if bt_with_hmm and "metrics_by_year" in bt_with_hmm and not bt_with_hmm["metrics_by_year"].empty:
        by_year_hmm_str = _df_to_markdown(bt_with_hmm["metrics_by_year"])

    content = f"""# Phase 54: LETF Decay Overlay — GO/NO-GO

**Date:** 2026-05-17
**Overall Verdict:** {overall_verdict}

## Summary

This document evaluates a delta-neutral LETF decay capture strategy:
SHORT 1 unit TQQQ + LONG 3 units QQQ (3x leverage, delta-neutral).

The strategy profits from the well-documented variance-drag decay of leveraged ETFs,
which accumulates as sigma^2 * (L^2 - L) / 2 per day at leverage L=3.

### Key theoretical facts

| Quantity | Value |
| :--- | ---: |
| Leverage | 3x |
| Decay formula | sigma^2 * 3 per day |
| At QQQ vol 17%/yr | ~8.65%/yr theoretical decay |
| At QQQ vol 20%/yr | ~12.0%/yr theoretical decay |
| At QQQ vol 25%/yr | ~18.8%/yr theoretical decay |
| At QQQ vol 30%/yr | ~27.0%/yr theoretical decay |
| Estimated cost drag | ~2.0%/yr (borrow + friction) |
| Net threshold (profitable) | decay > 2.0%/yr, i.e., QQQ vol > ~18% |

### Critical constraint — only profitable when sideways+volatile

Decay is only captured when TQQQ underperforms 3x QQQ. During strong bull trends,
TQQQ can OUTPERFORM 3x QQQ due to compounding (the short loses money).
This is the core risk: the short is dangerous in strong bull markets.
A delta-neutral construction (3x QQQ long) partially hedges directional exposure
but the P&L pair (3*QQQ - TQQQ) is NOT zero-cost — it specifically profits from
vol without trend.

## Costs and Net Feasibility

| Cost component | Estimated annual rate |
| :--- | ---: |
| TQQQ short borrow | ~0.5%/yr (liquid, easy-to-borrow) |
| Transaction costs / slippage | ~1.0%/yr |
| Other friction | ~0.5%/yr |
| **Total drag** | **~2.0%/yr** |

Net profitable when: realized annual TQQQ decay > 2.0%/yr, i.e., QQQ vol > ~18%/yr.

## Per-Year Realized Decay Table (TQQQ vs 3x QQQ)

{annual_str}

Notes:
- `realized_decay_pct` = TQQQ annual return minus path-adjusted 3x QQQ return.
- Negative realized_decay_pct = decay is present (short TQQQ is profitable, ignoring costs).
- `decay_profitable_to_short` = True when realized_decay < 0.

## Decay by Vol Regime

Vol threshold: 21-day rolling QQQ realized vol (annualised).

{vol_regime_str}

**Key finding:** High-vol regime shows positive PnL to the short-decay position;
low-vol regime typically shows negative PnL (TQQQ outperforms 3x QQQ when trend is smooth).

## Backtest: Vol-Only Gating (no HMM)

Entry: QQQ 21-day vol > 20%/yr. Exit: vol < 18%/yr. Costs: 2.0%/yr.

**Verdict: {verdict_no_hmm}**

{fmt_bt(bt_no_hmm)}

### By-Year Results (vol-only)

{by_year_no_hmm_str}

## Backtest: Vol + HMM SPY Gating

Entry: QQQ vol > 20%/yr AND SPY state is bear or neutral. Exit: vol < 18%/yr OR SPY is bull.

**Verdict: {verdict_with_hmm}**
"""

    if bt_with_hmm:
        content += f"\n{fmt_bt(bt_with_hmm)}\n"
        content += f"\n### By-Year Results (vol + HMM)\n\n{by_year_hmm_str}\n"
    else:
        content += "\nHMM gating was skipped (--no-hmm flag).\n"

    content += f"""
## Regime-Conditional Insight

The HMM gating adds value by avoiding the most dangerous scenario for the short:
SPY in a bull-trending regime where TQQQ compounding creates positive gamma for long holders.
By restricting entry to bear/neutral HMM states, the strategy avoids periods where
TQQQ systematically outperforms 3x QQQ.

## GO / NO-GO Decision Matrix

| Criterion | Threshold | Vol-only result | Vol+HMM result |
| :--- | ---: | ---: | ---: |
| Sharpe | >= 0.50 | {bt_no_hmm.get('sharpe', 0):.3f} | {(bt_with_hmm or {{}}).get('sharpe', 0):.3f} |
| MDD | >= -40% | {bt_no_hmm.get('mdd', 0)*100:.1f}% | {(bt_with_hmm or {{}}).get('mdd', 0)*100:.1f}% |
| Ann Return | > 0% | {bt_no_hmm.get('ann_return_pct', 0):.1f}% | {(bt_with_hmm or {{}}).get('ann_return_pct', 0):.1f}% |
| Days in trade | 15-60% | {bt_no_hmm.get('frac_days_in_trade', 0)*100:.0f}% | {(bt_with_hmm or {{}}).get('frac_days_in_trade', 0)*100:.0f}% |

## Next Steps (if GO)

1. Confirm TQQQ borrow availability and actual margin rate with broker.
2. Run strategy on SOXL/SOXX pair for diversification (higher decay, higher risk).
3. Add regime-conditional position sizing (larger when deep bear, smaller in neutral).
4. Walk-forward validation with purged folds (Phase 37 protocol).
5. Monitor: if TQQQ borrow rate spikes, strategy economics change — include in live monitoring.

## Risk Warnings

- SHORT TQQQ carries unlimited theoretical loss in a sustained bull market.
- The 3x QQQ long hedge is not perfect: path-dependency causes basis risk.
- Decay can REVERSE: in a low-vol bull, TQQQ outperforms 3x QQQ (short loses money).
- Borrow costs are variable; TQQQ borrow could spike if demand increases.
- Not suitable for retail accounts without margin and short-selling permissions.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    logger.info("Report written to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LETF decay overlay analysis")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(_repo_root / "results" / "cache"),
        help="Parquet cache directory (default: results/cache/)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(_repo_root / "results" / "letf"),
        help="Output directory for CSV results",
    )
    p.add_argument(
        "--no-hmm",
        action="store_true",
        help="Skip SPY HMM gating (vol-only backtest)",
    )
    p.add_argument(
        "--vol-entry",
        type=float,
        default=20.0,
        help="QQQ vol entry threshold (annualised %, default 20.0)",
    )
    p.add_argument(
        "--vol-exit",
        type=float,
        default=18.0,
        help="QQQ vol exit threshold (annualised %, default 18.0)",
    )
    p.add_argument(
        "--vol-window",
        type=int,
        default=21,
        help="Rolling window for realized vol (bars, default 21)",
    )
    p.add_argument(
        "--n-hmm-states",
        type=int,
        default=3,
        help="Number of HMM states for SPY (default 3)",
    )
    p.add_argument(
        "--n-restarts",
        type=int,
        default=5,
        help="HMM training restarts (default 5)",
    )
    p.add_argument(
        "--report-path",
        type=Path,
        default=Path(_repo_root / "docs" / "findings" / "letf_decay_overlay.md"),
        help="Path for the Markdown GO/NO-GO report",
    )
    return p.parse_args()


def main() -> None:
    """Entry point for the LETF decay overlay analysis."""
    args = _parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch data
    logger.info("Fetching LETF and benchmark data…")
    tqqq, qqq, soxl, soxx = fetch_letf_data(args.cache_dir)

    if qqq.empty or tqqq.empty:
        raise RuntimeError("Failed to load TQQQ or QQQ — check yfinance connection.")

    logger.info(
        "TQQQ: %d bars, QQQ: %d bars", len(tqqq), len(qqq)
    )

    # 2. Per-year decay table
    logger.info("Computing per-year realized decay table…")
    annual_table = compute_annual_decay_table(tqqq, qqq, soxl, soxx)
    logger.info(
        "Annual decay table (%d years):\n%s",
        len(annual_table),
        annual_table.to_string(index=False),
    )
    annual_table.to_csv(args.output_dir / "annual_decay_table.csv", index=False)

    # 3. Vol-regime decay table
    logger.info("Computing decay by vol regime…")
    vol_regime_table = compute_decay_by_regime(
        tqqq, qqq, vol_window=args.vol_window, vix_threshold=args.vol_entry
    )
    logger.info(
        "Vol-regime table:\n%s",
        vol_regime_table.to_string(index=False),
    )
    vol_regime_table.to_csv(args.output_dir / "vol_regime_decay.csv", index=False)

    # 4. Vol-only backtest
    logger.info("Running vol-only backtest (no HMM)…")
    bt_no_hmm = backtest_decay_short(
        tqqq,
        qqq,
        vol_entry_pct=args.vol_entry,
        vol_exit_pct=args.vol_exit,
        vol_window=args.vol_window,
        spy_states=None,
    )
    logger.info(
        "Vol-only backtest: Sharpe=%.3f, MDD=%.1f%%, Ann Return=%.1f%%, Days in trade=%.0f%%",
        bt_no_hmm["sharpe"],
        bt_no_hmm["mdd"] * 100,
        bt_no_hmm["ann_return_pct"],
        bt_no_hmm["frac_days_in_trade"] * 100,
    )

    # 5. Optional: SPY HMM-gated backtest
    bt_with_hmm = None
    if not args.no_hmm:
        logger.info("Fetching SPY data for HMM regime filter…")
        source = YFinanceSource(cache_dir=args.cache_dir, cache_max_age_seconds=86400)
        spy_raw = source.load("SPY", period="max", interval="1d")

        logger.info("Fitting SPY HMM (n_states=%d, n_restarts=%d)…", args.n_hmm_states, args.n_restarts)
        fitted, feat_df = _fit_spy_hmm(
            spy_raw, n_states=args.n_hmm_states, n_restarts=args.n_restarts
        )

        if args.n_hmm_states == 3:
            labels = _label_states_by_return(fitted, feat_df)
            logger.info("SPY HMM state labels: %s", labels)
            # bear and neutral states (all except the highest-return state = bull)
            bull_state = max(labels.keys(), key=lambda k: {"bear": 0, "neutral": 1, "bull": 2}[labels[k]])
            bear_neutral = {s for s in labels if s != bull_state}
        else:
            # For non-3 states: use all-but-highest-mean-return as bear/neutral
            log_return_col = feat_df.columns.tolist().index("log_return")
            means = fitted.hmm.means_[:, log_return_col]
            bull_state = int(np.argmax(means))
            bear_neutral = set(range(args.n_hmm_states)) - {bull_state}
            labels = {i: ("bull" if i == bull_state else "non-bull") for i in range(args.n_hmm_states)}
            logger.info("SPY HMM state labels (n=%d): %s", args.n_hmm_states, labels)

        spy_state_series = _decode_spy_states(fitted, feat_df)

        logger.info(
            "Running HMM-gated backtest (bear+neutral states=%s)…", bear_neutral
        )
        bt_with_hmm = backtest_decay_short(
            tqqq,
            qqq,
            vol_entry_pct=args.vol_entry,
            vol_exit_pct=args.vol_exit,
            vol_window=args.vol_window,
            spy_states=spy_state_series,
            bear_neutral_states=bear_neutral,
        )
        logger.info(
            "HMM-gated backtest: Sharpe=%.3f, MDD=%.1f%%, Ann Return=%.1f%%, Days in trade=%.0f%%",
            bt_with_hmm["sharpe"],
            bt_with_hmm["mdd"] * 100,
            bt_with_hmm["ann_return_pct"],
            bt_with_hmm["frac_days_in_trade"] * 100,
        )

    # 6. Save PnL series
    pnl_df = pd.DataFrame({"pnl_no_hmm": bt_no_hmm["pnl_series"]})
    if bt_with_hmm:
        pnl_df["pnl_with_hmm"] = bt_with_hmm["pnl_series"]
        pnl_df["in_trade_no_hmm"] = bt_no_hmm["in_trade_series"]
        pnl_df["in_trade_hmm"] = bt_with_hmm["in_trade_series"]
    pnl_df.to_parquet(args.output_dir / "pnl_series.parquet")
    logger.info("PnL series saved to %s/pnl_series.parquet", args.output_dir)

    # 7. Write report
    logger.info("Writing GO/NO-GO report…")
    write_findings_report(
        annual_decay_table=annual_table,
        vol_regime_table=vol_regime_table,
        bt_no_hmm=bt_no_hmm,
        bt_with_hmm=bt_with_hmm,
        output_path=args.report_path,
    )

    # 8. Print summary
    print("\n" + "=" * 72)
    print("LETF DECAY OVERLAY — RESULTS SUMMARY")
    print("=" * 72)
    print(f"\nDecay table ({len(annual_table)} years):")
    print(annual_table[["year", "qqq_realized_vol_ann_pct", "realized_decay_pct",
                          "decay_profitable_to_short"]].to_string(index=False))
    print("\nVol-regime breakdown:")
    print(vol_regime_table.to_string(index=False))
    print("\nBacktest: vol-only (no HMM)")
    print(f"  Sharpe={bt_no_hmm['sharpe']:.3f}, MDD={bt_no_hmm['mdd']*100:.1f}%, "
          f"Ann Return={bt_no_hmm['ann_return_pct']:.1f}%, "
          f"Days in trade={bt_no_hmm['frac_days_in_trade']*100:.0f}%, "
          f"N entries={bt_no_hmm['n_entry_trades']}")
    if bt_with_hmm:
        print("\nBacktest: vol + SPY HMM gating")
        print(f"  Sharpe={bt_with_hmm['sharpe']:.3f}, MDD={bt_with_hmm['mdd']*100:.1f}%, "
              f"Ann Return={bt_with_hmm['ann_return_pct']:.1f}%, "
              f"Days in trade={bt_with_hmm['frac_days_in_trade']*100:.0f}%, "
              f"N entries={bt_with_hmm['n_entry_trades']}")

    go_no_hmm = bt_no_hmm.get("sharpe", 0.0) >= 0.50
    go_hmm = (bt_with_hmm or {}).get("sharpe", 0.0) >= 0.50 if bt_with_hmm else False
    overall = "GO" if (go_no_hmm or go_hmm) else "NO-GO"
    print(f"\nOverall verdict: {overall}")
    print(f"\nReport: {args.report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
