# Phase 42 Multi-Asset Allocation — Decision Memo

**Date:** 2026-05-08  
**Branch:** `phase37/purged-cv`  
**Assets:** BTC-USD, ETH-USD, SPY, GLD  
**Period:** 2017-11-29 → 2026-05-08 (2,121 daily bars)  

---

## Results Summary

*Phase 42b (2026-05-08) — regime MVO variants:*

| Strategy | Sharpe | Calmar | MDD | Ann Return | Ann Vol |
|----------|--------|--------|-----|-----------|---------|
| global_min_var | **0.937** | 0.486 | 31.9% | 15.5% | 16.5% |
| regime_mvo (no vol target) | 0.450 | 0.235 | 67.4% | 15.8% | 35.2% |
| equal_weight | 0.340 | 0.179 | 67.4% | 12.1% | 35.5% |
| regime_mvo (vol-targeted 10%) | 0.066 | 0.020 | 67.4% | 1.4% | 20.7% |

*Phase 42c (2026-05-08) — regime-informed min-var probe:*

| Strategy | Sharpe | Calmar | MDD | Ann Return | Ann Vol |
|----------|--------|--------|-----|-----------|---------|
| global_min_var | **0.937** | 0.486 | 31.9% | 15.5% | 16.5% |
| regime_informed_min_var | 0.410 | 0.146 | 67.4% | 9.8% | 23.9% |
| equal_weight | 0.340 | 0.179 | 67.4% | 12.1% | 35.5% |
| regime_mvo (vol-targeted 10%) | 0.092 | 0.028 | 67.4% | 1.9% | 20.7% |

## Verdict: NO-GO (both phases)

**Phase 42b:** regime_mvo (best variant: no vol target) Sharpe = 0.450 vs
global_min_var = 0.937. Exceeds equal_weight (+0.11) but fails vs global_min_var
(−0.487 margin).

**Phase 42c:** regime_informed_min_var Sharpe = 0.410 vs global_min_var = 0.937.
Beats equal_weight (+0.07) and regime_mvo (+0.32) but fails vs global_min_var
(−0.527 margin). MDD = 67.4% — identical to equal_weight — confirming regime
exclusions do not protect against the large crypto drawdowns that define the gap.

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

## Phase 42c Analysis: Why Regime Exclusions Do Not Help

regime_informed_min_var at each rebalance sets weight=0 for assets whose
posterior-weighted E[r] falls below 0.0, then runs min-var on the remaining
eligible set (with a floor of 2 eligible assets).

**Why it still has MDD=67.4%:**

1. **Onset latency**: The HMM's expected return estimate is backward-looking.
   At the start of a crypto crash (when a regime shift begins), recent history
   still shows moderate returns — the posterior hasn't updated yet. By the time
   the next monthly rebalance fires, the drawdown is already in progress.

2. **Min-eligible fallback**: When both BTC and ETH are in negative-return
   regimes, the fallback keeps the 2 assets with the highest (least negative)
   E[r]. During global crypto bear markets, the "least bad" pair may still
   include a crypto asset, preserving drawdown exposure.

3. **Intra-rebalance exposure**: Between rebalances (21-bar window), weights
   are fixed. A regime flip mid-period is not acted upon until next rebalance.

4. **Structural floor**: Global min-var's 31.9% MDD comes entirely from
   allocating ~70–80% to low-vol SPY+GLD regardless of regime. Regime exclusion
   can redistribute weight *among* assets but cannot replicate the structural
   floor that min-var's vol-weighting achieves automatically.

## What This Closes and What Remains

**Closed:**
- Single-asset directional (n=8 hourly, n=8 daily, n=3 daily): exhausted
- Multi-asset regime MVO with BTC/ETH/SPY/GLD: NO-GO
- Regime-informed min-var with BTC/ETH/SPY/GLD: NO-GO

The entire BTC/ETH/SPY/GLD universe is exhausted. All regime-conditioning
approaches fail because the structural vol disparity (crypto ~80% vs equity
~15%) means global min-var structurally outperforms any regime tilt.

**Still open:**
1. **Options / vol forecasting**: ARI = 0.742 confirms stable vol regime
   structure. Regime labels as implied-vol forecast input remains the highest-
   confidence unexplored direction.
2. **Equity-only universe**: drop BTC/ETH, run regime_mvo on SPY/GLD/TLT/IEF
   where vol differences are smaller — the min-var advantage may shrink enough
   for regime conditioning to matter.

---

## Senior Quant Interpretation

The multi-asset result confirms the pattern from single-asset probes: the HMM
finds real, stable regime structure (ARI ≥ 0.74) but the signal is not strong
enough to overcome structural portfolio constraints at tested cost levels. The
engine's demonstrated edge is risk characterisation (volatility regimes,
drawdown timing, cross-asset concordance), not directional alpha generation.

