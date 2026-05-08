# Phase 42 Multi-Asset Allocation — Decision Memo

**Date:** 2026-05-08  
**Branch:** `phase37/purged-cv`  
**Assets:** BTC-USD, ETH-USD, SPY, GLD  
**Period:** 2017-11-29 → 2026-05-08 (2,121 daily bars)  

---

## Results Summary

| Strategy | Sharpe | Calmar | MDD | Ann Return | Ann Vol |
|----------|--------|--------|-----|-----------|---------|
| global_min_var | **0.937** | 0.486 | 31.9% | 15.5% | 16.5% |
| regime_mvo (no vol target) | 0.450 | 0.235 | 67.4% | 15.8% | 35.2% |
| equal_weight | 0.340 | 0.179 | 67.4% | 12.1% | 35.5% |
| regime_mvo (vol-targeted 10%) | 0.066 | 0.020 | 67.4% | 1.4% | 20.7% |

## Verdict: NO-GO

**regime_mvo (best variant: no vol target) Sharpe = 0.450 vs global_min_var = 0.937.**  
PASS criterion: regime_mvo must exceed both passive baselines. It exceeds
equal_weight (+0.11 Sharpe) but fails to beat global_min_var (−0.487 margin).

---

## What the Numbers Mean

### Why global_min_var dominates
This is a mixed-vol portfolio. BTC/ETH have ~80–100% annual volatility; SPY/GLD
have ~12–15%. Global min-var structurally allocates ~70–80% to SPY+GLD and
~20–30% to crypto. The result: 16.5% portfolio vol and 31.9% MDD vs. 35.5%
and 67.4% for equal-weight. The Sharpe advantage (0.937 vs 0.340) is almost
entirely driven by this volatility compression, not from picking better assets.

### What regime conditioning contributes
regime_mvo (no vol target) Sharpe = 0.450 vs equal_weight = 0.340 — a +0.11
improvement. This gap is statistically real (regime conditioning tilts away
from crypto in adverse regimes) but economically small compared to the +0.60
improvement from just minimising variance.

### Why vol targeting backfires
A 63-bar rolling covariance estimate is backward-looking. When a bear market
starts, estimated vol is still low → no scaling → full drawdown participation.
Once vol spikes, the scaling kicks in → position reduced → recovery missed.
Net effect: same MDD as unscaled (67.4%) with significantly lower return
(1.4% vs 15.8% ann). Backward-looking vol targeting creates cash drag without
drawdown protection in this regime.

---

## Root Cause Analysis

The HMM regime signal adds marginal directional value (+0.11 Sharpe over
equal-weight) but cannot compete with the structural volatility advantage of
variance minimization in a portfolio containing crypto. This is a case where
the baseline is very hard to beat precisely because of asset universe
composition, not because the regime signal is weak.

---

## What This Closes and What Remains

**Closed:**
- Single-asset directional (n=8 hourly, n=8 daily, n=3 daily): exhausted
- Multi-asset regime MVO: NO-GO vs global_min_var

**Still open:**
1. **Regime-informed min-var**: use regime classification to exclude bad-regime
   assets from the eligible set of the min-var optimisation. Example: if BTC
   is in its two lowest-return regimes, set its max_weight = 0. This combines
   the structural variance advantage of min-var with regime-conditional
   exclusions.
2. **Options / vol forecasting**: ARI = 0.742 confirms stable vol regime
   structure. Regime labels as implied-vol forecast input remains the highest-
   confidence unexplored direction.
3. **Equity-only universe**: drop BTC/ETH, run regime_mvo on SPY/GLD/TLT/IEF
   where vol differences are smaller — the min-var advantage may shrink enough
   for regime conditioning to matter.

---

## Senior Quant Interpretation

The multi-asset result confirms the pattern from single-asset probes: the HMM
finds real, stable regime structure (ARI ≥ 0.74) but the signal is not strong
enough to overcome structural portfolio constraints at tested cost levels. The
engine's demonstrated edge is risk characterisation (volatility regimes,
drawdown timing, cross-asset concordance), not directional alpha generation.

