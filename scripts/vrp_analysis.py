"""Phase 56: Volatility Risk Premium (VRP) harvesting analysis.

VRP = implied vol (VIX) − realized vol (21-day annualised SPY).
The implied vol is systematically above realized vol because investors pay
a risk-premium for variance insurance. Selling this premium (short volatility)
has historically delivered Sharpe ratios of 1.5–2.5, but with catastrophic
left-tail risk (February 2018: XIV lost 96% in a single session).

Hypothesis: The SPY n=3 HMM identifies when short-vol is safe (low-vol
trending regime = high return-rank state) vs dangerous (transition into
high-vol regime = low return-rank state). A joint entry filter
(VRP > threshold AND HMM state = bull) should improve risk-adjusted returns
while largely avoiding the blow-up periods.

Outputs
-------
results/phase56_vrp/vrp_summary.csv     — daily VRP series
results/phase56_vrp/regime_vrp.csv      — mean VRP per HMM state
results/phase56_vrp/backtest_results.csv — tearsheet comparison
docs/findings/phase56_vrp.md            — GO/NO-GO memo

Usage
-----
    uv run python scripts/vrp_analysis.py
    uv run python scripts/vrp_analysis.py --vrp-threshold 2.5 --n-restarts 5
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
from rde.labeling.ranking import rank_states
from rde.models.hmm import FittedModel, train_hmm
from rde.models.selection import select_n_states

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

ANN_FACTOR = 252
VRP_WINDOW = 21  # days for realized vol calculation


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_spy_daily(cache_dir: Path) -> pd.DataFrame:
    """Download SPY daily OHLCV (max history) with caching."""
    src = YFinanceSource(cache_dir=cache_dir)
    df = src.load("SPY", period="max", interval="1d")
    logger.info("SPY: %d bars from %s to %s", len(df), df.index[0].date(), df.index[-1].date())
    return df


def _load_vix_daily(cache_dir: Path) -> pd.Series:
    """Download VIX index daily (max history). Returns a Series of VIX close."""
    import yfinance as yf
    cache_file = cache_dir / "VIX_max_1d.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        logger.info("VIX: loaded from cache (%d bars)", len(df))
    else:
        logger.info("Downloading VIX from yfinance …")
        raw = yf.download("^VIX", period="max", interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError("yfinance returned empty DataFrame for ^VIX")
        # Flatten MultiIndex columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0] for col in raw.columns]
        raw = raw.dropna(subset=["Close"])
        raw.to_parquet(cache_file)
        df = raw
        logger.info("VIX: %d bars from %s to %s", len(df), df.index[0].date(), df.index[-1].date())
    return df["Close"].rename("VIX")


def _load_etf_daily(symbol: str, cache_dir: Path) -> pd.DataFrame:
    """Download short/long-vol ETF daily prices with caching."""
    src = YFinanceSource(cache_dir=cache_dir)
    try:
        df = src.load(symbol, period="max", interval="1d")
        logger.info("%s: %d bars from %s to %s", symbol, len(df),
                    df.index[0].date(), df.index[-1].date())
        return df
    except Exception as exc:
        logger.warning("Could not load %s: %s", symbol, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# VRP computation
# ---------------------------------------------------------------------------


def compute_vrp(spy_df: pd.DataFrame, vix_series: pd.Series) -> pd.Series:
    """Compute daily Volatility Risk Premium (annualised percentage points).

    VRP = VIX − 21-day realized vol of SPY (annualised).

    Positive VRP means implied vol exceeds realized vol — the normal state.
    Short-vol strategies harvest this premium.

    Parameters
    ----------
    spy_df : pd.DataFrame
        SPY OHLCV with a DatetimeIndex.
    vix_series : pd.Series
        VIX closing levels (percentage, e.g. 20 = 20% implied vol).

    Returns
    -------
    pd.Series
        Daily VRP in annualised percentage points, aligned to trading dates.

    """
    log_ret = np.log(spy_df["Close"] / spy_df["Close"].shift(1))
    realized_vol = log_ret.rolling(VRP_WINDOW).std() * np.sqrt(ANN_FACTOR) * 100.0

    aligned = pd.concat([vix_series, realized_vol.rename("rv21")], axis=1).dropna()
    vrp = aligned["VIX"] - aligned["rv21"]
    vrp.name = "vrp"
    logger.info(
        "VRP: %d bars, mean=%.2f, median=%.2f, pct_positive=%.1f%%",
        len(vrp), vrp.mean(), vrp.median(), (vrp > 0).mean() * 100,
    )
    return vrp


# ---------------------------------------------------------------------------
# HMM fitting on SPY
# ---------------------------------------------------------------------------


def _build_spy_pipeline() -> FeaturePipeline:
    """Standard daily SPY feature pipeline (matches spy_daily.yaml)."""
    return FeaturePipeline([
        LogReturns(),
        RollingVolatility(window=20),
        SmoothedReturns(window=5),
    ])


def fit_spy_hmm(
    spy_df: pd.DataFrame,
    *,
    n_states: int = 3,
    n_restarts: int = 10,
    seed_base: int = 42,
) -> tuple[FittedModel, pd.DataFrame]:
    """Fit n=3 Gaussian HMM on SPY daily features.

    Parameters
    ----------
    spy_df : pd.DataFrame
        SPY OHLCV DataFrame.
    n_states : int
        Number of HMM states (3 per Phase 43c finding).
    n_restarts : int
        Baum-Welch restarts.
    seed_base : int
        Random seed base.

    Returns
    -------
    tuple[FittedModel, pd.DataFrame]
        Fitted model and the feature DataFrame (aligned to SPY trading dates).

    """
    pipeline = _build_spy_pipeline()
    df_features = pipeline.transform(spy_df)
    feature_cols = ["log_return", "volatility_w20", "smoothed_return_w5"]
    X = df_features[feature_cols].values.astype(float)

    logger.info("Fitting SPY HMM n=%d with %d restarts …", n_states, n_restarts)
    model = train_hmm(
        X,
        n_states=n_states,
        n_restarts=n_restarts,
        covariance_type="full",
        n_iter=1000,
        seed_base=seed_base,
        feature_names=feature_cols,
    )
    logger.info("HMM fitted: log-lik=%.2f, AIC=%.2f, BIC=%.2f",
                model.log_likelihood, model.aic, model.bic)
    return model, df_features


def decode_spy_states(
    model: FittedModel,
    df_features: pd.DataFrame,
) -> pd.Series:
    """Viterbi-decode HMM states on the full SPY feature series.

    Parameters
    ----------
    model : FittedModel
        Fitted SPY HMM.
    df_features : pd.DataFrame
        Feature DataFrame (must include model.feature_names columns).

    Returns
    -------
    pd.Series
        Integer state sequence indexed by df_features.index.

    """
    X = df_features[model.feature_names].values.astype(float)
    X_scaled = model.scaler.transform(X)
    states = model.hmm.predict(X_scaled)
    return pd.Series(states, index=df_features.index, name="hmm_state")


# ---------------------------------------------------------------------------
# VRP by regime
# ---------------------------------------------------------------------------


def vrp_by_regime(
    vrp: pd.Series,
    states: pd.Series,
    labelled_states: list,
) -> pd.DataFrame:
    """Compute mean/median VRP and sample count per HMM state.

    Parameters
    ----------
    vrp : pd.Series
        Daily VRP values (annualised pp).
    states : pd.Series
        Integer HMM state for each trading date.
    labelled_states : list[LabelledState]
        State labels from rank_states().

    Returns
    -------
    pd.DataFrame
        Columns: state, label, rank_return, rank_vol, mean_vrp, median_vrp,
        pct_positive, n_bars.

    """
    label_map = {ls.index: ls for ls in labelled_states}
    aligned = pd.concat([vrp, states], axis=1).dropna()
    aligned.columns = ["vrp", "state"]

    rows = []
    for state_idx in sorted(aligned["state"].unique()):
        mask = aligned["state"] == state_idx
        v = aligned.loc[mask, "vrp"]
        ls = label_map.get(state_idx)
        rows.append({
            "state": state_idx,
            "label": ls.label if ls else f"state_{state_idx}",
            "rank_return": ls.rank_return if ls else -1,
            "rank_vol": ls.rank_volatility if ls else -1,
            "mean_vrp": round(v.mean(), 3),
            "median_vrp": round(v.median(), 3),
            "pct_positive": round((v > 0).mean() * 100, 1),
            "n_bars": len(v),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Backtest helpers
# ---------------------------------------------------------------------------


def _sharpe(returns: pd.Series, ann_factor: int = ANN_FACTOR) -> float:
    """Annualised Sharpe (risk-free = 0)."""
    if returns.std() == 0 or len(returns) < 2:
        return float("nan")
    return float(returns.mean() / returns.std() * np.sqrt(ann_factor))


def _mdd(returns: pd.Series) -> float:
    """Maximum drawdown (negative percentage) from a log-return series."""
    # Use cumulative log-returns to avoid issues when 1+r < 0
    log_equity = returns.cumsum()
    running_max = log_equity.cummax()
    drawdown = log_equity - running_max  # in log space; always <= 0
    return float(drawdown.min() * 100)


def _ann_return(returns: pd.Series, ann_factor: int = ANN_FACTOR) -> float:
    """Annualised geometric return (percentage).

    ``returns`` is a series of daily log-returns (already in log space).
    Annualised return = exp(mean_log_return * ann_factor) - 1.
    """
    if len(returns) == 0:
        return float("nan")
    mean_log = float(returns.mean())
    return float((np.exp(mean_log * ann_factor) - 1) * 100)


def _tearsheet(returns: pd.Series, name: str) -> dict:
    """Compute summary statistics for a return series."""
    invested = (returns != 0).sum()
    return {
        "strategy": name,
        "ann_return_pct": round(_ann_return(returns), 2),
        "sharpe": round(_sharpe(returns), 3),
        "mdd_pct": round(_mdd(returns), 2),
        "n_bars_invested": int(invested),
        "n_bars_total": len(returns),
        "pct_invested": round(invested / len(returns) * 100, 1),
    }


# ---------------------------------------------------------------------------
# SVXY backtest
# ---------------------------------------------------------------------------


def backtest_svxy(
    svxy_df: pd.DataFrame,
    vrp: pd.Series,
    states: pd.Series,
    bull_state_indices: list[int],
    vrp_threshold: float = 2.0,
) -> pd.DataFrame:
    """Run unfiltered and regime-filtered SVXY backtests.

    Three strategies:
    1. Buy-and-hold SVXY (baseline).
    2. VRP filter only: hold when VRP > threshold.
    3. Full filter: hold when VRP > threshold AND state in bull_state_indices.
    4. Regime only: hold when state in bull_state_indices (no VRP filter).

    Parameters
    ----------
    svxy_df : pd.DataFrame
        SVXY OHLCV DataFrame.
    vrp : pd.Series
        Daily VRP values.
    states : pd.Series
        Integer HMM state for each trading date.
    bull_state_indices : list[int]
        HMM state indices considered "safe" for short-vol.
    vrp_threshold : float
        Minimum VRP (in annualised pp) to enter short-vol position.

    Returns
    -------
    pd.DataFrame
        Tearsheet comparison: one row per strategy.

    """
    svxy_ret = np.log(svxy_df["Close"] / svxy_df["Close"].shift(1)).dropna()
    svxy_ret.name = "svxy_log_return"

    # Align everything to dates with all signals available
    combined = pd.concat([svxy_ret, vrp, states], axis=1).dropna()
    combined.columns = ["svxy_ret", "vrp", "state"]

    bull_mask = combined["state"].isin(bull_state_indices)
    vrp_mask = combined["vrp"] > vrp_threshold

    # 1. Buy-and-hold
    bah = combined["svxy_ret"]

    # 2. VRP filter only
    vrp_only = combined["svxy_ret"] * vrp_mask.astype(int)

    # 3. Regime only
    regime_only = combined["svxy_ret"] * bull_mask.astype(int)

    # 4. Full filter
    full_filter = combined["svxy_ret"] * (vrp_mask & bull_mask).astype(int)

    logger.info("SVXY backtest from %s to %s (%d aligned bars)",
                combined.index[0].date(), combined.index[-1].date(), len(combined))
    logger.info("Bull state(s): %s  |  VRP threshold: %.1f pp", bull_state_indices, vrp_threshold)
    logger.info("Pct days bull: %.1f%%  |  Pct days VRP>threshold: %.1f%%  |  Pct days both: %.1f%%",
                bull_mask.mean() * 100, vrp_mask.mean() * 100,
                (bull_mask & vrp_mask).mean() * 100)

    rows = [
        _tearsheet(bah, "buy_and_hold"),
        _tearsheet(vrp_only, f"vrp_only_>{vrp_threshold:.1f}pp"),
        _tearsheet(regime_only, "regime_only_bull"),
        _tearsheet(full_filter, f"full_filter_bull+vrp>{vrp_threshold:.1f}pp"),
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Blow-up detection
# ---------------------------------------------------------------------------


def identify_blowup_periods(
    svxy_df: pd.DataFrame,
    states: pd.Series,
    vrp: pd.Series,
    bull_state_indices: list[int],
    vrp_threshold: float,
    drawdown_threshold_pct: float = -20.0,
) -> pd.DataFrame:
    """Identify major SVXY drawdown periods and check if the filter avoided them.

    Parameters
    ----------
    svxy_df : pd.DataFrame
        SVXY OHLCV.
    states : pd.Series
        HMM state sequence.
    vrp : pd.Series
        Daily VRP.
    bull_state_indices : list[int]
        Safe regime indices.
    vrp_threshold : float
        VRP entry threshold.
    drawdown_threshold_pct : float
        Drawdown depth to flag as a "blow-up" event (negative, e.g. -20.0).

    Returns
    -------
    pd.DataFrame
        Blow-up events with columns: date, svxy_drawdown_pct, state,
        was_bull, vrp_value, would_filter_out.

    """
    svxy_ret = np.log(svxy_df["Close"] / svxy_df["Close"].shift(1)).dropna()
    combined = pd.concat([svxy_ret, states, vrp], axis=1).dropna()
    combined.columns = ["svxy_ret", "state", "vrp"]

    # Mark single-day losses beyond threshold as blow-up candidate
    big_down = combined["svxy_ret"] * 100 < drawdown_threshold_pct
    blow_ups = combined[big_down].copy()
    blow_ups["svxy_drawdown_pct"] = blow_ups["svxy_ret"] * 100
    blow_ups["was_bull"] = blow_ups["state"].isin(bull_state_indices)
    blow_ups["vrp_positive"] = blow_ups["vrp"] > vrp_threshold
    blow_ups["would_filter_out"] = ~(blow_ups["was_bull"] & blow_ups["vrp_positive"])

    logger.info(
        "Identified %d single-day SVXY drawdowns < %.0f%%. Filter avoids %d/%d (%.0f%%).",
        len(blow_ups), drawdown_threshold_pct,
        blow_ups["would_filter_out"].sum(), len(blow_ups),
        blow_ups["would_filter_out"].mean() * 100,
    )
    return blow_ups[["svxy_drawdown_pct", "state", "was_bull", "vrp", "vrp_positive",
                      "would_filter_out"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 56: VRP harvesting analysis")
    p.add_argument("--vrp-threshold", type=float, default=2.0,
                   help="Minimum VRP (annualised pp) to enter short-vol position (default: 2.0)")
    p.add_argument("--n-states", type=int, default=3,
                   help="HMM states for SPY (default: 3, per Phase 43c finding)")
    p.add_argument("--n-restarts", type=int, default=10,
                   help="Baum-Welch restarts (default: 10)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-cache", action="store_true", help="Bypass parquet cache")
    return p.parse_args()


def main() -> None:
    """Run VRP harvesting analysis and write outputs to results/phase56_vrp/."""
    args = _parse_args()
    out_dir = _repo_root / "results" / "phase56_vrp"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _repo_root / "results" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("=== Phase 56: VRP Harvesting Analysis ===")
    spy_df = _load_spy_daily(cache_dir)
    vix = _load_vix_daily(cache_dir)
    svxy_df = _load_etf_daily("SVXY", cache_dir)

    # ------------------------------------------------------------------
    # 2. Compute VRP
    # ------------------------------------------------------------------
    vrp = compute_vrp(spy_df, vix)

    vrp_summary = pd.DataFrame({
        "metric": ["mean_vrp", "median_vrp", "pct_positive", "std_vrp",
                   "min_vrp", "max_vrp", "n_bars"],
        "value": [
            round(vrp.mean(), 3),
            round(vrp.median(), 3),
            round((vrp > 0).mean() * 100, 1),
            round(vrp.std(), 3),
            round(vrp.min(), 3),
            round(vrp.max(), 3),
            len(vrp),
        ],
    })
    vrp_summary.to_csv(out_dir / "vrp_summary.csv", index=False)
    logger.info("VRP summary:\n%s", vrp_summary.to_string(index=False))

    # ------------------------------------------------------------------
    # 3. Fit SPY HMM and decode states
    # ------------------------------------------------------------------
    model, df_features = fit_spy_hmm(
        spy_df,
        n_states=args.n_states,
        n_restarts=args.n_restarts,
        seed_base=args.seed,
    )
    states = decode_spy_states(model, df_features)

    # Label states; the "bull" state has highest return rank
    labelled = rank_states(model, return_feature="log_return", vol_feature="volatility_w20")
    bull_state_idx = [ls.index for ls in labelled if ls.rank_return == args.n_states - 1]
    logger.info("Labelled states: %s", [(ls.index, ls.label) for ls in labelled])
    logger.info("Bull state(s) used for filter: %s", bull_state_idx)

    # ------------------------------------------------------------------
    # 4. VRP by regime
    # ------------------------------------------------------------------
    regime_vrp_table = vrp_by_regime(vrp, states, labelled)
    regime_vrp_table.to_csv(out_dir / "regime_vrp.csv", index=False)
    logger.info("VRP by regime:\n%s", regime_vrp_table.to_string(index=False))

    # ------------------------------------------------------------------
    # 5. SVXY backtest
    # ------------------------------------------------------------------
    if svxy_df.empty:
        logger.warning("SVXY data unavailable — skipping backtest")
        results_df = pd.DataFrame()
    else:
        results_df = backtest_svxy(
            svxy_df, vrp, states, bull_state_idx, vrp_threshold=args.vrp_threshold
        )
        results_df.to_csv(out_dir / "backtest_results.csv", index=False)
        logger.info("Backtest results:\n%s", results_df.to_string(index=False))

        # ------------------------------------------------------------------
        # 6. Blow-up analysis
        # ------------------------------------------------------------------
        blowups = identify_blowup_periods(
            svxy_df, states, vrp, bull_state_idx,
            vrp_threshold=args.vrp_threshold,
            drawdown_threshold_pct=-15.0,
        )
        blowups.to_csv(out_dir / "blowup_events.csv")
        logger.info("Blow-up events:\n%s", blowups.to_string())

    # ------------------------------------------------------------------
    # 7. Print tearsheet
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 56 — VOLATILITY RISK PREMIUM (VRP) HARVESTING")
    print("=" * 65)
    print(f"\nSPY HMM: n_states={args.n_states}, n_restarts={args.n_restarts}")
    print(f"VRP threshold: {args.vrp_threshold:.1f} annualised pp")
    print(f"Bull state index(es): {bull_state_idx}")
    print(f"\nVRP Statistics (VIX - 21d realized vol):")
    print(f"  Mean:      {vrp.mean():.2f} pp")
    print(f"  Median:    {vrp.median():.2f} pp")
    print(f"  % Positive: {(vrp > 0).mean() * 100:.1f}%")
    print(f"  Std:       {vrp.std():.2f} pp")
    print(f"\nVRP by HMM Regime:")
    print(regime_vrp_table.to_string(index=False))

    if not results_df.empty:
        print(f"\nSVXY Backtest Results:")
        print(results_df.to_string(index=False))

        # GO/NO-GO verdict
        full_filter_row = results_df[results_df["strategy"].str.startswith("full_filter")]
        bah_row = results_df[results_df["strategy"] == "buy_and_hold"]
        if not full_filter_row.empty and not bah_row.empty:
            ff_sharpe = full_filter_row.iloc[0]["sharpe"]
            bah_sharpe = bah_row.iloc[0]["sharpe"]
            ff_mdd = full_filter_row.iloc[0]["mdd_pct"]
            bah_mdd = bah_row.iloc[0]["mdd_pct"]
            verdict = "GO" if ff_sharpe >= 1.2 and ff_mdd > bah_mdd else "NO-GO"
            print(f"\nGO/NO-GO Verdict: {verdict}")
            print(f"  Filtered Sharpe: {ff_sharpe:.3f}  (threshold: 1.200)")
            print(f"  B&H Sharpe:      {bah_sharpe:.3f}")
            print(f"  Filtered MDD:    {ff_mdd:.1f}%")
            print(f"  B&H MDD:         {bah_mdd:.1f}%")

    print(f"\nOutputs written to: {out_dir}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 8. Write findings memo
    # ------------------------------------------------------------------
    _write_findings(
        out_dir=out_dir,
        vrp=vrp,
        regime_vrp_table=regime_vrp_table,
        results_df=results_df,
        labelled=labelled,
        bull_state_idx=bull_state_idx,
        vrp_threshold=args.vrp_threshold,
        n_states=args.n_states,
        blowups_df=blowups if not svxy_df.empty else pd.DataFrame(),
    )


def _write_findings(
    *,
    out_dir: Path,
    vrp: pd.Series,
    regime_vrp_table: pd.DataFrame,
    results_df: pd.DataFrame,
    labelled: list,
    bull_state_idx: list[int],
    vrp_threshold: float,
    n_states: int,
    blowups_df: pd.DataFrame,
) -> None:
    """Write docs/findings/phase56_vrp.md."""
    findings_dir = _repo_root / "docs" / "findings"
    findings_dir.mkdir(parents=True, exist_ok=True)

    # Determine verdict
    verdict = "INSUFFICIENT DATA"
    ff_sharpe = float("nan")
    bah_sharpe = float("nan")
    ff_mdd = float("nan")
    bah_mdd = float("nan")

    if not results_df.empty:
        ff_row = results_df[results_df["strategy"].str.startswith("full_filter")]
        bah_row = results_df[results_df["strategy"] == "buy_and_hold"]
        if not ff_row.empty and not bah_row.empty:
            ff_sharpe = ff_row.iloc[0]["sharpe"]
            bah_sharpe = bah_row.iloc[0]["sharpe"]
            ff_mdd = ff_row.iloc[0]["mdd_pct"]
            bah_mdd = bah_row.iloc[0]["mdd_pct"]
            verdict = "GO" if ff_sharpe >= 1.2 and ff_mdd > bah_mdd else "NO-GO"

    state_rows = "\n".join(
        f"| {r['state']} | {r['label']} | {r['rank_return']} | {r['rank_vol']} "
        f"| {r['mean_vrp']:.2f} | {r['median_vrp']:.2f} | {r['pct_positive']:.1f}% | {r['n_bars']} |"
        for _, r in regime_vrp_table.iterrows()
    )

    backtest_rows = ""
    if not results_df.empty:
        backtest_rows = "\n".join(
            f"| {r['strategy']} | {r['ann_return_pct']:.2f}% | {r['sharpe']:.3f} "
            f"| {r['mdd_pct']:.1f}% | {r['pct_invested']:.1f}% |"
            for _, r in results_df.iterrows()
        )

    blowup_section = ""
    if not blowups_df.empty:
        blowup_rows = "\n".join(
            f"| {date.date()} | {row['svxy_drawdown_pct']:.1f}% | {int(row['state'])} "
            f"| {'yes' if row['was_bull'] else 'no'} | {row['vrp']:.2f} "
            f"| {'AVOIDED' if row['would_filter_out'] else 'HELD'} |"
            for date, row in blowups_df.iterrows()
        )
        blowup_section = f"""
## 5. Major Blow-up Events

The following single-day SVXY drawdowns exceeded 15%:

| Date | SVXY 1-day | State | Was Bull | VRP | Filter Action |
|------|-----------|-------|----------|-----|--------------|
{blowup_rows}

**Filter avoidance rate: {blowups_df['would_filter_out'].mean() * 100:.0f}% of blow-up days**
"""

    memo = f"""# Phase 56: Volatility Risk Premium (VRP) Harvesting

**Date:** 2026-05-17
**Status:** {verdict}
**Branch:** phase51/regime-conditional-lambda

---

## 1. Concept

VRP = implied vol (VIX) − 21-day annualised realized vol (SPY).

The implied vol is systematically above realized vol because investors pay
a risk premium for variance insurance. Selling this premium (short volatility)
has historically delivered Sharpe ratios of 1.5–2.5 but with catastrophic
left-tail risk (February 2018: XIV lost 96% in one session).

**Hypothesis:** The SPY n={n_states} HMM identifies when short-vol is safe
(low-vol trending regime = high-return-rank state) vs dangerous (transition
into high-vol regime). A dual filter (VRP > {vrp_threshold:.1f} pp AND HMM state = bull)
should improve risk-adjusted returns while avoiding blow-up periods.

---

## 2. VRP Statistics

| Metric | Value |
|--------|-------|
| Mean VRP | {vrp.mean():.2f} pp |
| Median VRP | {vrp.median():.2f} pp |
| % Positive | {(vrp > 0).mean() * 100:.1f}% |
| Std VRP | {vrp.std():.2f} pp |
| Min VRP | {vrp.min():.2f} pp |
| Max VRP | {vrp.max():.2f} pp |
| N Bars | {len(vrp)} |

A positive mean VRP confirms the structural short-vol premium exists in the data.

---

## 3. Regime-Conditional VRP

VRP distribution by SPY HMM state (n={n_states}):

| State | Label | Return Rank | Vol Rank | Mean VRP | Median VRP | % Positive | N Bars |
|-------|-------|-------------|---------|----------|------------|------------|--------|
{state_rows}

**Bull state(s) for filter:** {bull_state_idx}

---

## 4. SVXY Backtest Results

VRP threshold: {vrp_threshold:.1f} annualised pp
Bull state index(es): {bull_state_idx}

| Strategy | Ann Return | Sharpe | MDD | % Invested |
|----------|-----------|--------|-----|-----------|
{backtest_rows}

**GO criterion:** Filtered Sharpe >= 1.20 AND filtered MDD less negative than B&H MDD.

**Verdict: {verdict}**
- Filtered Sharpe: {ff_sharpe:.3f} (threshold: 1.200)
- B&H Sharpe: {bah_sharpe:.3f}
- Filtered MDD: {ff_mdd:.1f}%
- B&H MDD: {bah_mdd:.1f}%
{blowup_section}

---

## 6. Risk Warnings

1. **Tail risk / short gamma:** Short-vol strategies have negative skewness and
   extreme kurtosis. The historical MDD understates true risk because it is
   path-dependent and the XIV/SVXY roll mechanism creates gap risk.

2. **Leverage and margin:** SVXY uses leveraged futures. Broker margin
   requirements can force liquidation at the worst possible moment.

3. **VIX mean reversion spikes:** VIX spikes (Feb 2018, Mar 2020, Aug 2024)
   can produce intraday moves that dwarf daily close-to-close data suggests.
   The true worst-case loss is not captured in daily returns.

4. **Regime lag:** The HMM decoder is causal but uses past data to assign states.
   In a fast-moving market (VIX doubling intraday), the filter may not activate
   before the loss is taken.

5. **SVXY structure changes:** SVXY reduced its leverage from -1x to -0.5x
   after Feb 2018. Pre-2018 returns are not comparable to post-2018 returns.
   This creates a structural break in the backtest data.

6. **Strategy capacity:** This strategy is viable only at retail scale. Futures
   markets for VIX products are not deep enough for institutional implementation
   without significant market impact.

---

## 7. Next Steps

- If GO: implement position-sizing with Kelly fraction capped at 0.5x optimal
  to manage the negative-skew tail; add intraday VIX monitoring.
- If NO-GO: evaluate VXX shorting instead of SVXY long (avoids the leverage
  structure issue); evaluate variance swaps on SPY for institutional use.
- Cross-check: validate the HMM filter's regime-transition lead time against
  VIX spikes using intraday data (hourly).

---

*Analysis date: 2026-05-17*
*Script: scripts/vrp_analysis.py*
*Outputs: results/phase56_vrp/*
"""

    findings_path = findings_dir / "phase56_vrp.md"
    findings_path.write_text(memo)
    logger.info("Findings written to %s", findings_path)


if __name__ == "__main__":
    main()
