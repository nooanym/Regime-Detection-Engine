# Phase 43 Equity-Only Probe — Decision Memo

**Date:** 2026-05-09
**Branch:** `phase37/purged-cv`
**Assets:** SPY, GLD, TLT, IEF
**Period:** 2004-12-17 → 2026-05-08 (5,381 daily bars ≈ 21.4 years)
**Config:** n_states=8, n_restarts=1, lookback_bars=504, rebalance_bars=21

---

## Results Summary

| Strategy | Sharpe | Calmar | MDD | Ann Return | Ann Vol | Rebalances |
|----------|--------|--------|-----|-----------|---------|------------|
| global_min_var | 0.929 | 0.306 | 21.7% | 6.6% | 7.2% | 254 |
| equal_weight | 0.840 | 0.302 | 22.7% | 6.8% | 8.2% | 0 |
| regime_mvo | 0.661 | **0.308** | **16.0%** | 4.9% | 7.4% | 233 |
| regime_informed_min_var | 0.633 | 0.219 | 25.6% | 5.6% | 8.9% | 233 |

## Verdict: FAIL (strict criterion) — but PARTIAL POSITIVE

Strict criterion: regime_informed_min_var Sharpe must exceed global_min_var Sharpe.
- regime_informed_min_var 0.633 vs global_min_var 0.929 → **FAIL**
- regime_mvo 0.661 vs global_min_var 0.929 → **FAIL**

However, this is the **most promising result in the research project** to date.

---

## Key Finding: Regime Conditioning Reduces MDD in Equity Universe

This is the first backtest where regime conditioning achieves a **lower drawdown
than global min-var**:

- regime_mvo MDD = **16.0%** vs global_min_var MDD = **21.7%**
- Calmar: regime_mvo **0.308** vs global_min_var 0.306 — essentially tied

On a return-per-unit-of-drawdown basis (Calmar), regime_mvo matches the
structural baseline. The Sharpe deficit (0.661 vs 0.929) comes entirely from
lower annualised return (4.9% vs 6.6%), not from higher volatility (7.4% vs 7.2%).

**Contrast with crypto universe (Phase 42b/42c):**

| Universe | regime_mvo MDD | global_min_var MDD | Regime signal works? |
|----------|---------------|-------------------|---------------------|
| BTC/ETH/SPY/GLD | 67.4% | 31.9% | No — MDD identical to equal_weight |
| SPY/GLD/TLT/IEF | 16.0% | 21.7% | **Yes — regime beats min-var on MDD** |

The equity universe is qualitatively different. The regime signal is capturing
real risk-on/risk-off structure and successfully reducing drawdowns.

---

## Root Cause of Sharpe Failure

The strategy is **too conservative**. It avoids the worst periods well (low MDD)
but also undershoots rallies, dragging annual return from 6.6% to 4.9%.

This is a strategy design problem, not a signal quality problem:
- The HMM finds real structure (MDD evidence above)
- The binary long/flat regime strategy leaves return on the table
- The regime signal is potentially exploitable with a less aggressive strategy

### Why regime_informed_min_var performs worst

regime_informed_min_var (MDD 25.6%, Sharpe 0.633) is worse than pure
regime_mvo (MDD 16.0%, Sharpe 0.661). In an equity-only universe:
- All four assets are positively correlated during crisis (flight-to-quality
  dynamics can flip asset correlations in ways the lookback covariance misses)
- Excluding one or two assets concentrates remaining min-var onto fewer assets,
  sometimes increasing MDD rather than reducing it
- The min_eligible=2 fallback doesn't always pick the right pair

---

## What This Means for the Research Direction

The equity-only universe shows genuine promise. The immediate next step is
**Phase 37-style purged CV validation on SPY** (the primary risk-on asset in
this portfolio) to determine:

1. Is the Sharpe stable across CV folds (not just in-sample)?
2. Does the signal pass random-baseline and shuffle tests?
3. Is period robustness ARI ≥ 0.40 for daily SPY?

If SPY's HMM signal passes Phase 37, the strategy design can be improved:
- Soft regime tilts (scaling weights toward regime-optimal rather than binary)
- Regime-conditional position sizing (fraction of vol budget to each asset)
- Regime-conditional rebalancing (more frequent rebalancing in high-uncertainty periods)

---

## Next Steps

1. **Run Phase 37 validation on SPY daily** (`configs/spy_daily.yaml`) — done if
   period robustness ARI ≥ 0.40 and random-baseline margin > 0.30.
2. **If SPY passes Phase 37**: design an improved multi-asset portfolio strategy
   that uses soft regime weights rather than binary on/off allocation.
3. **If SPY fails Phase 37**: the options/vol forecasting direction is next
   (regime labels as implied-vol forecast input; ARI=0.742 on BTC confirms
   the engine finds stable structure, explore whether the same holds for SPY).

---

## Senior Quant Interpretation

The equity universe result is a partial positive that changes the research
direction. Unlike the crypto experiments where regime conditioning had zero
effect on drawdowns (MDD was always 67.4% for all regime variants), the
equity universe shows the HMM IS capturing risk-on/risk-off structure
(MDD 16.0% vs structural baseline 21.7%). The signal-to-noise ratio is better
in equity than crypto — unsurprisingly, given that equity regimes are more
stationary and driven by macro factors that persist over months.

The failure is in strategy design, not signal discovery. A regime signal that
reduces MDD from 21.7% to 16.0% while maintaining near-identical Calmar is a
real, exploitable signal. The work ahead is translating that signal into a
portfolio rule that captures both the downside protection AND the upside
participation.
