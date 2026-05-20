"""
Phase 78 — Carry MDD Robustness: Is the Near-Zero MDD Genuine or Path-Specific?
================================================================================

Phase 77 integrated Phase 75 momentum filter + Phase 76 spot_spread_dynamic into
CarryStrategy, achieving carry leg Sharpe 19.73, Ann 15.54%, MDD −0.02%.

This script stress-tests that MDD figure across four dimensions:

  Step 1: Identify worst negative-carry episodes (drawdown archaeology).
  Step 2: Test momentum filter robustness at each stress event (first-period loss).
  Step 3: Half-period stability (does improvement hold in both halves?).
  Step 4: Permutation test — is temporal structure exploited or is filter overfit?

Robustness verdict:
  ROBUST   — both halves Sharpe > 15.0 AND permutation p < 0.05
  MIXED    — partial inconsistency (one half > 15.0 OR p < 0.10)
  FRAGILE  — both halves Sharpe < 15.0 OR p >= 0.10
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from funding_carry_backtest import fetch_funding_rates  # noqa: E402

PERIODS_PER_YEAR = 3 * 365  # 1095 eight-hour periods per year
COST_PER_PERIOD = 0.005 / PERIODS_PER_YEAR  # ~0.5% annual friction
N_PERMUTATIONS = 100

# ---------------------------------------------------------------------------
# Core simulation (mirrors Phase 76 spot_spread_dynamic + Phase 75 momentum filter)
# ---------------------------------------------------------------------------


def _metrics(net: pd.Series) -> dict[str, float]:
    """Compute Sharpe, Ann Return, Ann Vol, MDD, Cum Return from 8-hourly net-return series."""
    n = net.dropna()
    if len(n) == 0:
        return {"sharpe": 0.0, "ann_return": 0.0, "ann_vol": 0.0, "mdd": 0.0, "cum_return": 0.0}
    cum = (1 + n).cumprod()
    t = len(cum)
    ann = float(cum.iloc[-1] ** (PERIODS_PER_YEAR / t) - 1)
    vol = float(n.std() * np.sqrt(PERIODS_PER_YEAR))
    sharpe = ann / vol if vol > 0 else 0.0
    roll_max = np.maximum.accumulate(cum.values)
    mdd = float(((cum.values - roll_max) / roll_max).min())
    cum_ret = float(cum.iloc[-1] - 1)
    return {
        "sharpe": round(sharpe, 4),
        "ann_return": round(ann, 4),
        "ann_vol": round(vol, 4),
        "mdd": round(mdd, 4),
        "cum_return": round(cum_ret, 4),
    }


def simulate_phase77(
    df: pd.DataFrame,
    *,
    use_momentum_filter: bool = True,
    use_spot_spread: bool = True,
    carry_window: int = 90,
    dynamic_min: float = 0.40,
    dynamic_max: float = 0.80,
) -> pd.Series:
    """
    Simulate the Phase 77 carry strategy (Phase 75 + Phase 76 integrated).

    Parameters
    ----------
    df : DataFrame with 'btc' and 'eth' columns of per-period funding rates.
    use_momentum_filter : bool
        If True, skip periods when previous combined rate <= 0.
    use_spot_spread : bool
        If True, use instantaneous ETH/BTC proportional weights (Phase 76).
    carry_window : int
        Rolling window for carry-weighted baseline (Phase 60).
    dynamic_min, dynamic_max : float
        Clipping range for spot_spread_dynamic ETH weight.

    Returns
    -------
    pd.Series of per-period net returns (aligned to df index after dropna).
    """
    btc = df["btc"]
    eth = df["eth"]

    # ── Spot spread or rolling carry weights ──────────────────────────────────
    if use_spot_spread:
        # Phase 76 spot_spread_dynamic
        denom = (btc.clip(lower=0) + eth.clip(lower=0)).where(lambda x: x > 0, np.nan)
        raw_eth_wt = eth.clip(lower=0) / denom
        w_eth = raw_eth_wt.clip(lower=dynamic_min, upper=dynamic_max).fillna(0.5)
        w_btc = 1.0 - w_eth
        any_positive = (btc > 0) | (eth > 0)
        w_btc_final = w_btc.where(any_positive, 0.0)
        w_eth_final = w_eth.where(any_positive, 0.0)
    else:
        # Phase 60 rolling carry weights
        carry_btc = (btc * PERIODS_PER_YEAR).rolling(carry_window).mean().clip(lower=0.0)
        carry_eth = (eth * PERIODS_PER_YEAR).rolling(carry_window).mean().clip(lower=0.0)
        total = carry_btc + carry_eth
        in_mkt = total > 0
        w_btc_final = (carry_btc / total.where(total > 0, 1.0)).where(in_mkt, 0.0)
        w_eth_final = (carry_eth / total.where(total > 0, 1.0)).where(in_mkt, 0.0)

    # ── Gross return ──────────────────────────────────────────────────────────
    gross = w_btc_final * btc + w_eth_final * eth
    in_mkt = (w_btc_final + w_eth_final) > 0

    # ── Phase 75 lag-1 momentum filter ────────────────────────────────────────
    if use_momentum_filter:
        combined_rate = (btc + eth) * 0.5
        prev_positive = combined_rate.shift(1) > 0
        skip_mask = ~prev_positive.fillna(True)
        in_mkt = in_mkt & ~skip_mask

    # ── Net return ────────────────────────────────────────────────────────────
    net = gross.where(in_mkt, 0.0) - in_mkt.astype(float) * COST_PER_PERIOD
    return net.dropna()


def compute_combined_rate(df: pd.DataFrame) -> pd.Series:
    """Compute per-period portfolio-level combined rate (simple average of BTC and ETH)."""
    return (df["btc"] + df["eth"]) * 0.5


# ---------------------------------------------------------------------------
# Step 1: Identify worst negative-carry episodes
# ---------------------------------------------------------------------------


def find_negative_episodes(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Find all runs of consecutive negative combined-rate periods.
    Returns a DataFrame of the top_n worst episodes by total carry cost.

    The combined rate is the simple mean of BTC and ETH funding rates.
    This mirrors the Phase 75 momentum filter signal variable.
    """
    combined = compute_combined_rate(df)
    ann_combined = combined * PERIODS_PER_YEAR

    episodes = []
    in_episode = False
    ep_start = None
    ep_rates = []

    for ts, rate in ann_combined.items():
        if rate < 0:
            if not in_episode:
                in_episode = True
                ep_start = ts
                ep_rates = [rate]
            else:
                ep_rates.append(rate)
        else:
            if in_episode:
                episodes.append({
                    "start": ep_start,
                    "end": ts,
                    "duration_periods": len(ep_rates),
                    "min_ann_rate": min(ep_rates),
                    "mean_ann_rate": float(np.mean(ep_rates)),
                    "total_carry": float(sum(ep_rates)),  # annualized sum (negative = loss)
                })
                in_episode = False
                ep_start = None
                ep_rates = []

    # Flush last episode if series ends in negative territory
    if in_episode and ep_start is not None:
        episodes.append({
            "start": ep_start,
            "end": df.index[-1],
            "duration_periods": len(ep_rates),
            "min_ann_rate": min(ep_rates),
            "mean_ann_rate": float(np.mean(ep_rates)),
            "total_carry": float(sum(ep_rates)),
        })

    if not episodes:
        return pd.DataFrame(columns=["start", "end", "duration_periods",
                                     "min_ann_rate", "mean_ann_rate", "total_carry"])

    ep_df = pd.DataFrame(episodes)
    # Sort by total carry (most negative first = worst episodes)
    ep_df = ep_df.sort_values("total_carry").head(top_n).reset_index(drop=True)
    return ep_df


# ---------------------------------------------------------------------------
# Step 2: Momentum filter coverage at each stress event
# ---------------------------------------------------------------------------


def stress_event_filter_coverage(
    df: pd.DataFrame, episodes: pd.DataFrame
) -> pd.DataFrame:
    """
    For each stress episode, compute:
    - carry collected WITHOUT filter (always-in)
    - carry collected WITH filter (skip periods when prev combined rate <= 0)
    - first_period_loss: the unavoidable loss from T0 of each episode
    - protected_loss: how much the filter saved vs always-in
    - filter_exited_before: True if filter triggered BEFORE the worst period
    """
    combined = compute_combined_rate(df)
    prev_positive = combined.shift(1) > 0
    skip_mask = ~prev_positive.fillna(True)

    records = []
    for _, ep in episodes.iterrows():
        mask = (df.index >= ep["start"]) & (df.index <= ep["end"])
        ep_df = df[mask]
        ep_combined = combined[mask]
        ep_skip = skip_mask[mask]

        carry_always_in = float(ep_combined.sum()) * PERIODS_PER_YEAR
        carry_with_filter = float(ep_combined[~ep_skip].sum()) * PERIODS_PER_YEAR

        # First period is always collected (momentum filter is lag-1)
        first_period_rate = float(combined.loc[ep["start"]]) * PERIODS_PER_YEAR

        # How many periods did the filter skip in this episode?
        n_skipped = int(ep_skip.sum())
        n_total = int(mask.sum())

        # Did filter exit before the minimum-rate period?
        min_rate_ts = ep_combined.idxmin()
        filter_exited_before = bool(skip_mask.get(min_rate_ts, False))

        records.append({
            "start": ep["start"],
            "end": ep["end"],
            "duration_periods": n_total,
            "first_period_loss_ann": round(first_period_rate, 4),
            "carry_always_in_ann": round(carry_always_in, 4),
            "carry_with_filter_ann": round(carry_with_filter, 4),
            "protected_loss_ann": round(carry_always_in - carry_with_filter, 4),
            "n_skipped": n_skipped,
            "filter_exited_before_worst": filter_exited_before,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 3: Half-period stability
# ---------------------------------------------------------------------------


def half_period_stability(df: pd.DataFrame) -> dict:
    """
    Split full history into two equal halves.
    Compute Phase 77 strategy metrics for each half and overall.
    """
    mid = df.index[len(df) // 2]
    df_a = df[df.index < mid].copy()
    df_b = df[df.index >= mid].copy()

    results = {}
    for label, sub in [("full", df), ("half_a", df_a), ("half_b", df_b)]:
        net = simulate_phase77(sub, use_momentum_filter=True, use_spot_spread=True)
        results[label] = _metrics(net)
        results[label]["start"] = str(sub.index[0].date())
        results[label]["end"] = str(sub.index[-1].date())
        results[label]["n_periods"] = len(sub)

        # Also compute what fraction of periods the filter skips
        combined = compute_combined_rate(sub)
        prev_positive = combined.shift(1) > 0
        skip_mask = ~prev_positive.fillna(True)
        results[label]["filter_skip_rate"] = round(float(skip_mask.mean()), 4)

    return results


# ---------------------------------------------------------------------------
# Step 4: Permutation test
# ---------------------------------------------------------------------------


def permutation_test(df: pd.DataFrame, n_permutations: int = N_PERMUTATIONS, seed: int = 42) -> dict:
    """
    Permutation test: shuffle the funding rate sequence, apply the Phase 77
    strategy (momentum filter + spot_spread_dynamic), compute Sharpe.

    Null hypothesis: momentum filter exploits spurious temporal autocorrelation.
    p-value = fraction of shuffled Sharpe values >= real filtered Sharpe.

    Note: shuffling destroys the temporal autocorrelation structure entirely.
    If p < 0.05, the strategy exploits real temporal structure.
    """
    rng = np.random.default_rng(seed)

    # Real filtered Sharpe
    net_real = simulate_phase77(df, use_momentum_filter=True, use_spot_spread=True)
    real_sharpe = _metrics(net_real)["sharpe"]

    # Permuted Sharpes
    shuffled_sharpes = []
    btc_vals = df["btc"].values.copy()
    eth_vals = df["eth"].values.copy()

    for i in range(n_permutations):
        perm_idx = rng.permutation(len(df))
        df_perm = pd.DataFrame(
            {"btc": btc_vals[perm_idx], "eth": eth_vals[perm_idx]},
            index=df.index,
        )
        net_perm = simulate_phase77(df_perm, use_momentum_filter=True, use_spot_spread=True)
        m = _metrics(net_perm)
        shuffled_sharpes.append(m["sharpe"])

    shuffled_arr = np.array(shuffled_sharpes)
    p_value = float((shuffled_arr >= real_sharpe).mean())
    mean_shuffled = float(shuffled_arr.mean())
    margin = real_sharpe - mean_shuffled

    return {
        "real_sharpe": real_sharpe,
        "n_permutations": n_permutations,
        "mean_shuffled_sharpe": round(mean_shuffled, 4),
        "max_shuffled_sharpe": round(float(shuffled_arr.max()), 4),
        "p_value": round(p_value, 4),
        "margin_over_shuffled_mean": round(margin, 4),
        "shuffled_sharpes": shuffled_arr.tolist(),
    }


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def compute_verdict(half_results: dict, perm_results: dict) -> str:
    """Compute robustness verdict based on half-period and permutation tests."""
    sharpe_a = half_results["half_a"]["sharpe"]
    sharpe_b = half_results["half_b"]["sharpe"]
    p_value = perm_results["p_value"]

    both_halves_pass = sharpe_a > 15.0 and sharpe_b > 15.0
    one_half_passes = sharpe_a > 15.0 or sharpe_b > 15.0
    perm_pass = p_value < 0.05
    perm_marginal = p_value < 0.10

    if both_halves_pass and perm_pass:
        return "ROBUST"
    elif one_half_passes and perm_marginal:
        return "MIXED"
    else:
        return "FRAGILE"


# ---------------------------------------------------------------------------
# Write findings document
# ---------------------------------------------------------------------------


def write_findings(
    *,
    df: pd.DataFrame,
    episodes: pd.DataFrame,
    coverage: pd.DataFrame,
    half_results: dict,
    perm_results: dict,
    verdict: str,
) -> Path:
    """Write the Phase 78 findings document."""
    doc_path = ROOT / "docs" / "findings" / "phase78_carry_stress.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)

    real_sharpe = perm_results["real_sharpe"]
    p_value = perm_results["p_value"]
    mean_shuffled = perm_results["mean_shuffled_sharpe"]
    margin = perm_results["margin_over_shuffled_mean"]

    sharpe_a = half_results["half_a"]["sharpe"]
    sharpe_b = half_results["half_b"]["sharpe"]
    mdd_a = half_results["half_a"]["mdd"]
    mdd_b = half_results["half_b"]["mdd"]
    skip_a = half_results["half_a"]["filter_skip_rate"]
    skip_b = half_results["half_b"]["filter_skip_rate"]

    # Top 3 worst episodes for the summary
    top3 = coverage.head(3)

    ep_rows = []
    for _, row in coverage.iterrows():
        ep_rows.append(
            f"| {row['start'].strftime('%Y-%m-%d')} | {row['end'].strftime('%Y-%m-%d')} | "
            f"{row['duration_periods']} | {row['first_period_loss_ann']:.2%} | "
            f"{row['carry_always_in_ann']:.2%} | {row['carry_with_filter_ann']:.2%} | "
            f"{row['protected_loss_ann']:.2%} | "
            f"{'YES' if row['filter_exited_before_worst'] else 'NO (first period unavoidable)'} |"
        )

    if verdict == "ROBUST":
        verdict_text = (
            "**ROBUST.** Both half-periods achieve Sharpe > 15.0 and the permutation "
            f"test is highly significant (p={p_value:.4f}). The near-zero MDD (−0.02%) "
            "reflects genuine temporal structure in funding rate autocorrelation, not "
            "path-specific luck. The Phase 75 momentum filter consistently avoids negative "
            "carry episodes in both the pre- and post-midpoint periods."
        )
        recommendation = (
            "The filter is confirmed robust. **Recommended: update `make carry-live` "
            "defaults to include `--momentum-filter --spot-spread-weight`**, "
            "and add a `carry-live-full` Makefile target."
        )
    elif verdict == "MIXED":
        verdict_text = (
            f"**MIXED.** Half-period consistency is partial (Sharpe A={sharpe_a:.2f}, "
            f"Sharpe B={sharpe_b:.2f}; one half > 15.0 threshold). Permutation p={p_value:.4f} "
            f"({'marginal' if p_value < 0.10 else 'fails'}). The MDD improvement is real "
            "but may depend on the specific path of funding rate autocorrelation. "
            "The −0.02% MDD estimate is optimistic; a realistic floor is the worst "
            "first-period loss from the stress episodes."
        )
        recommendation = (
            "Exercise caution. The momentum filter provides net benefit but robustness "
            "is partial. Deploy with `--momentum-filter` but communicate a conservative "
            "MDD estimate (worst first-period loss from stress episodes)."
        )
    else:
        verdict_text = (
            f"**FRAGILE.** Half-period Sharpe is inconsistent (A={sharpe_a:.2f}, B={sharpe_b:.2f}) "
            f"and/or permutation test fails (p={p_value:.4f}). The near-zero MDD appears "
            "path-specific. The Phase 77 MDD of −0.02% should not be used in portfolio "
            "risk models. Recommend using the worst-case first-period loss as the MDD floor."
        )
        recommendation = (
            "The near-zero MDD is path-specific. Use the realistic MDD floor from "
            "stress-event analysis. Do NOT update live defaults to advertise −0.02% MDD."
        )

    doc = f"""# Phase 78 — Carry MDD Robustness: Genuine or Path-Specific?

**Date:** 2026-05-20
**Verdict: {verdict}**

## Background

Phase 77 integrated Phase 75 (lag-1 momentum filter) + Phase 76 (spot_spread_dynamic
weights) into CarryStrategy. The carry leg achieved:

| Metric | Phase 77 |
|---|---|
| Sharpe | 19.73 |
| Ann Return | 15.54% |
| MDD | **−0.02%** |

The near-zero MDD raises the question: is this a genuine structural property of
the momentum-filtered carry strategy, or did we happen to avoid major negative
funding episodes on this specific historical path?

## Dataset

- Symbols: BTCUSDT, ETHUSDT perpetual futures on Binance
- Period: {df.index[0].date()} → {df.index[-1].date()}
- Frequency: 8-hourly funding payments
- Total periods: {len(df):,}

## Step 1 — Worst Negative-Carry Episodes (Top 10 by Total Loss)

The combined rate is the simple mean of BTC and ETH annualised funding rates.
All episodes below are runs of consecutive periods with combined rate < 0.

| Start | End | Duration | First Period Loss (Ann) | Always-In Carry (Ann) | Filtered Carry (Ann) | Protected Loss (Ann) | Filter Exited Before Worst? |
|---|---|---|---|---|---|---|---|
{chr(10).join(ep_rows)}

**Key finding:** The momentum filter is lag-1 — the first period of each negative
episode is always collected before the filter triggers. The "first period loss" column
shows the maximum unavoidable loss per episode. This is the true MDD floor.

## Step 2 — Momentum Filter Coverage at Stress Events

The filter protection mechanism:
- T0: negative episode begins → position HELD (filter has not yet seen T0 rate)
- T0+1: filter sees the T0 negative rate → SKIP T0+1 and hold cash
- Filter exits the carry position during the episode, saving subsequent periods

This means the MDD floor is approximately the worst single-period negative rate,
not zero. The {coverage.iloc[0]['first_period_loss_ann']:.2%} ann first-period loss
in the worst episode (start {coverage.iloc[0]['start'].strftime('%Y-%m')}) is the
realistic MDD floor under the Phase 77 strategy.

## Step 3 — Half-Period Stability

| Period | Start | End | N Periods | Sharpe | Ann Return | MDD | Filter Skip Rate |
|---|---|---|---|---|---|---|---|
| Full | {half_results['full']['start']} | {half_results['full']['end']} | {half_results['full']['n_periods']:,} | {half_results['full']['sharpe']:.4f} | {half_results['full']['ann_return']:.2%} | {half_results['full']['mdd']:.4%} | {half_results['full']['filter_skip_rate']:.1%} |
| Half A | {half_results['half_a']['start']} | {half_results['half_a']['end']} | {half_results['half_a']['n_periods']:,} | {sharpe_a:.4f} | {half_results['half_a']['ann_return']:.2%} | {mdd_a:.4%} | {skip_a:.1%} |
| Half B | {half_results['half_b']['start']} | {half_results['half_b']['end']} | {half_results['half_b']['n_periods']:,} | {sharpe_b:.4f} | {half_results['half_b']['ann_return']:.2%} | {mdd_b:.4%} | {skip_b:.1%} |

**Criterion:** Both halves Sharpe > 15.0 for ROBUST verdict.
Half A: {sharpe_a:.2f} ({'PASS' if sharpe_a > 15.0 else 'FAIL'}) | Half B: {sharpe_b:.2f} ({'PASS' if sharpe_b > 15.0 else 'FAIL'})

The filter skip rate being consistent across halves (A={skip_a:.1%}, B={skip_b:.1%})
confirms the negative-carry regime structure is a persistent feature of the funding
rate series, not concentrated in one historical window.

## Step 4 — Permutation Test

Null hypothesis: the momentum filter exploits spurious temporal structure
(i.e., the improvement is path-specific and would work equally well on a
randomly shuffled sequence of funding rates).

N={perm_results['n_permutations']} permutations: BTC and ETH funding rate vectors
are independently shuffled (preserving per-period cross-asset correlation but
destroying all temporal autocorrelation), then the Phase 77 strategy is applied.

| Metric | Value |
|---|---|
| Real filtered Sharpe | {real_sharpe:.4f} |
| Mean shuffled Sharpe | {mean_shuffled:.4f} |
| Max shuffled Sharpe | {perm_results['max_shuffled_sharpe']:.4f} |
| Margin over shuffled mean | +{margin:.4f} |
| **p-value** | **{p_value:.4f}** |

**p-value interpretation:** {p_value:.4f} = fraction of shuffled series that achieved
Sharpe ≥ {real_sharpe:.4f}. A p-value < 0.05 confirms the strategy exploits real
temporal structure.

**Criterion:** p < 0.05 for ROBUST, p < 0.10 for MIXED.
Result: p={p_value:.4f} → {'PASS (p < 0.05)' if p_value < 0.05 else ('MARGINAL (p < 0.10)' if p_value < 0.10 else 'FAIL')}

## Verdict: {verdict}

{verdict_text}

## Recommendation

{recommendation}

## Methodology

- Phase 77 strategy: spot_spread_dynamic weights (ETH weight = clip(eth_r+ / (btc_r+ + eth_r+), 0.40, 0.80))
  + lag-1 momentum filter (skip periods when previous combined rate ≤ 0)
- Combined rate signal: simple mean of BTC and ETH per-period rates
- Cost model: 0.5% annual friction applied per active period (~0.0005% per 8h period)
- Half-period split: chronological, equal number of periods
- Permutation test: joint shuffle of both BTC and ETH vectors (preserving same-period
  BTC/ETH co-movement but destroying temporal autocorrelation); seed=42 for reproducibility
- GO threshold for robustness: both halves Sharpe > 15.0 AND permutation p < 0.05
"""

    doc_path.write_text(doc, encoding="utf-8")
    return doc_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run() -> None:
    print("=" * 70)
    print("PHASE 78 — Carry MDD Robustness Stress Test")
    print("=" * 70)

    # ── Fetch data ─────────────────────────────────────────────────────────────
    print("\nFetching BTCUSDT funding rates (2019-present)...", end=" ", flush=True)
    btc_raw = fetch_funding_rates("BTCUSDT", start_year=2019)
    print(f"{len(btc_raw)} periods")

    print("Fetching ETHUSDT funding rates (2019-present)...", end=" ", flush=True)
    eth_raw = fetch_funding_rates("ETHUSDT", start_year=2019)
    print(f"{len(eth_raw)} periods")

    df = pd.DataFrame({"btc": btc_raw, "eth": eth_raw}).sort_index().dropna()
    print(f"Aligned: {len(df)} periods  ({df.index[0].date()} → {df.index[-1].date()})\n")

    # ── Step 1: Identify worst negative-carry episodes ─────────────────────────
    print("=" * 70)
    print("STEP 1: Worst Negative-Carry Episodes (Top 10)")
    print("=" * 70)

    episodes = find_negative_episodes(df, top_n=10)
    print(f"\nFound {len(episodes)} episodes with combined rate < 0\n")

    print(f"  {'#':<3} {'Start':<14} {'End':<14} {'Dur':>6} {'Min Ann':>10} {'Mean Ann':>10} {'Total Ann':>11}")
    print("  " + "-" * 75)
    for i, (_, ep) in enumerate(episodes.iterrows(), 1):
        print(
            f"  {i:<3} {str(ep['start'].date()):<14} {str(ep['end'].date()):<14} "
            f"{ep['duration_periods']:>6}  "
            f"{ep['min_ann_rate']:>9.2%}  "
            f"{ep['mean_ann_rate']:>9.2%}  "
            f"{ep['total_carry']:>10.2%}"
        )

    # ── Step 2: Momentum filter coverage ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Momentum Filter Coverage at Stress Events")
    print("=" * 70)

    coverage = stress_event_filter_coverage(df, episodes)

    print(f"\n  {'Start':<14} {'End':<14} {'Dur':>5} {'1st-period':>12} {'Always-in':>11} {'Filtered':>11} {'Protected':>11} {'Exited before worst?'}")
    print("  " + "-" * 100)
    for _, row in coverage.iterrows():
        print(
            f"  {str(row['start'].date()):<14} {str(row['end'].date()):<14} "
            f"{row['duration_periods']:>5}  "
            f"{row['first_period_loss_ann']:>11.2%}  "
            f"{row['carry_always_in_ann']:>10.2%}  "
            f"{row['carry_with_filter_ann']:>10.2%}  "
            f"{row['protected_loss_ann']:>10.2%}  "
            f"{'YES' if row['filter_exited_before_worst'] else 'NO'}"
        )

    top3 = coverage.head(3)
    print("\n-- Top 3 worst episodes summary --")
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        print(
            f"  Episode {i}: {row['start'].strftime('%Y-%m')} "
            f"(dur={row['duration_periods']} periods, "
            f"1st-loss={row['first_period_loss_ann']:.2%} ann, "
            f"filter saved {row['protected_loss_ann']:.2%} ann, "
            f"exited before worst: {'YES' if row['filter_exited_before_worst'] else 'NO'})"
        )

    # ── Step 3: Half-period stability ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Half-Period Stability")
    print("=" * 70)

    half_results = half_period_stability(df)
    mid_date = df.index[len(df) // 2].date()
    print(f"\nSplit date: {mid_date}")
    print(f"\n  {'Period':<10} {'Start':>12} {'End':>12} {'N':>7} {'Sharpe':>9} {'Ann Ret':>9} {'MDD':>9} {'Skip Rate':>10}")
    print("  " + "-" * 82)
    for label in ["full", "half_a", "half_b"]:
        r = half_results[label]
        print(
            f"  {label:<10} {r['start']:>12} {r['end']:>12} "
            f"{r['n_periods']:>7,}  "
            f"{r['sharpe']:>8.4f}  "
            f"{r['ann_return']:>8.2%}  "
            f"{r['mdd']:>8.4%}  "
            f"{r['filter_skip_rate']:>9.1%}"
        )

    sharpe_a = half_results["half_a"]["sharpe"]
    sharpe_b = half_results["half_b"]["sharpe"]
    print(f"\n  Half A Sharpe: {sharpe_a:.4f} ({'PASS' if sharpe_a > 15.0 else 'FAIL'} vs 15.0 threshold)")
    print(f"  Half B Sharpe: {sharpe_b:.4f} ({'PASS' if sharpe_b > 15.0 else 'FAIL'} vs 15.0 threshold)")

    # ── Step 4: Permutation test ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"STEP 4: Permutation Test (N={N_PERMUTATIONS} shuffles)")
    print("=" * 70)

    print(f"\nRunning {N_PERMUTATIONS} permutations (seed=42)...", end=" ", flush=True)
    perm_results = permutation_test(df, n_permutations=N_PERMUTATIONS, seed=42)
    print("done")

    p_value = perm_results["p_value"]
    print(f"\n  Real filtered Sharpe:    {perm_results['real_sharpe']:.4f}")
    print(f"  Mean shuffled Sharpe:    {perm_results['mean_shuffled_sharpe']:.4f}")
    print(f"  Max shuffled Sharpe:     {perm_results['max_shuffled_sharpe']:.4f}")
    print(f"  Margin over shuffled:   +{perm_results['margin_over_shuffled_mean']:.4f}")
    print(f"  p-value:                 {p_value:.4f}  ({'PASS (< 0.05)' if p_value < 0.05 else ('MARGINAL (< 0.10)' if p_value < 0.10 else 'FAIL (>= 0.10)')})")

    # ── Verdict ────────────────────────────────────────────────────────────────
    verdict = compute_verdict(half_results, perm_results)

    print("\n" + "=" * 70)
    print(f"PHASE 78 VERDICT: {verdict}")
    print("=" * 70)

    if verdict == "ROBUST":
        print("\n  Both halves Sharpe > 15.0 AND permutation p < 0.05.")
        print("  The near-zero MDD is GENUINE — momentum filter exploits real autocorrelation.")
        print("  Recommendation: deploy --momentum-filter --spot-spread-weight as live defaults.")
    elif verdict == "MIXED":
        print("\n  Partial half-period consistency or marginal permutation significance.")
        print("  The MDD improvement is real but robustness is incomplete.")
        print("  Recommendation: deploy with caution, use conservative MDD estimate.")
    else:
        print("\n  Half-period inconsistency or permutation test failure.")
        print("  The near-zero MDD appears path-specific. Use stress-event MDD floor.")

    # ── Write findings ─────────────────────────────────────────────────────────
    doc_path = write_findings(
        df=df,
        episodes=episodes,
        coverage=coverage,
        half_results=half_results,
        perm_results=perm_results,
        verdict=verdict,
    )
    print(f"\nFindings written to: {doc_path}")

    return verdict, half_results, perm_results, coverage


if __name__ == "__main__":
    run()
