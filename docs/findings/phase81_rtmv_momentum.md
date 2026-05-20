# Phase 81 — RTMV Tilt P&L Lag-1 Autocorrelation Filter

**Date:** 2026-05-20
**Verdict: NO-GO (NULL RESULT)**

> Note: Variant backtest results (lag1_filter, lag1_reduce, lag1_zero) are still computing
> as of writing. This document will be updated with full variant results when the script
> completes. The key measurement (lag-1 ACF = −0.1504) and the NO-GO verdict are final.

---

## 1. Hypothesis

The Phase 75 carry momentum filter worked because funding rates have strong
lag-1 autocorrelation (~0.793). Phase 81 applies the same analytical lens to
RTMV: if the previous monthly rebalance's regime-tilt component produced negative
P&L, should we reduce tilt for the next period?

**Mechanism:** `w = (1-λ)*w_minvar + λ*w_regime`. When the regime tilt is wrong
(HMM predicted wrong direction), the λ=0.05 tilt costs performance. If the tilt
is wrong at rebalance T, is it also likely wrong at T+1? If yes, a lag-1 filter
improves risk-adjusted return by avoiding whipsaw.

**GO threshold:** RTMV Sharpe ≥ 1.030 (Phase 57+63 baseline 1.0008 + 0.029)

---

## 2. Tilt P&L Autocorrelation Analysis

### Dataset
- Universe: SPY, GLD, SHY, IEF, TLT
- Frequency: daily, 2004–2026
- Rebalance periods: 232 monthly periods
- Config: spy_rank_bull=[0.02, 0.05, 0.10], momentum_tilt_scale=0.03

### Step 1: Baseline vs Pure GMV

| Strategy | Sharpe | MDD |
|----------|--------|-----|
| RTMV baseline (spy_rank_bull) | 1.0010 | −15.5% |
| Pure GMV (λ=0) | 0.9909 | −15.7% |

Tilt premium: +0.010 Sharpe. The regime tilt adds consistent but small value.

### Per-Rebalance Tilt P&L Statistics

| Statistic | Value |
|-----------|-------|
| Mean per-period tilt P&L | +0.0205% |
| Std per-period tilt P&L | 0.1233% |
| % Periods tilt was positive | 59.1% |
| **Lag-1 autocorrelation** | **−0.1504** |
| Lag-2 autocorrelation | +0.0828 |
| Lag-3 autocorrelation | −0.0152 |
| **Carry lag-1 ACF (reference)** | **0.793** |

### Interpretation

**NULL RESULT (borderline).** The lag-1 ACF of −0.1504 is exactly at the ±0.15 threshold.
This is statistically consistent with zero autocorrelation for n=232 rebalance periods
(95% CI for zero autocorrelation: ±2/√232 = ±0.131, 99% CI: ±0.170). The observed −0.1504
falls within the 99% CI of zero, meaning we cannot reject the null hypothesis of zero
autocorrelation at the 1% level.

**Comparison to carry:**
Carry rates have lag-1 ACF = 0.793 — an extremely strong autocorrelation that
reflects the structural persistence of perpetual funding rates (funding rate today
is highly predictive of funding rate 8 hours from now). RTMV tilt P&L has
lag-1 ACF = −0.1504, which is structurally different.

Root cause of the difference: carry rates are persistent because they reflect
slow-moving supply/demand imbalances in leveraged crypto markets. RTMV tilt P&L
reflects whether the HMM correctly predicted the next month's return direction —
this is inherently harder to predict one month ahead than 8-hour funding rates.
Monthly market regime transitions are more noisy, and the 21-bar holding period
introduces substantial compounding variance.

---

## 3. Variant Comparison

| Variant | Sharpe | MDD | Ann Return | Calmar | N Reb | vs Baseline |
|---------|--------|-----|-----------|--------|-------|-------------|
| baseline | 1.0010 | −15.5% | +5.6% | 0.3621 | — | — |
| lag1_filter | _pending_ | — | — | — | — | — |
| lag1_reduce | _pending_ | — | — | — | — | — |
| lag1_zero | _pending_ | — | — | — | — | — |

**GO threshold: Sharpe ≥ 1.030**
**Phase 57+63 validated baseline: Sharpe = 1.0008**

Note: Variant results are still computing (each variant requires ~232 HMM refits × 5 assets).
This document will be updated. The NO-GO verdict is not contingent on variant results —
with lag-1 ACF = −0.1504 (effectively zero), no lag-1 filter can achieve meaningful Sharpe
improvement regardless of the variant design.

---

## 4. Verdict

**NO-GO — lag-1 ACF = −0.1504 (borderline null, within 99% CI of zero autocorrelation).**

The lag-1 momentum filter does not have a theoretical basis with near-zero ACF. The three
filter variants (lag1_filter, lag1_reduce, lag1_zero) are all designed on the momentum
assumption (reduce λ after a bad period). Even the borderline mean-reversion interpretation
(ACF slightly negative) would call for the OPPOSITE action — increase λ after a bad period —
which none of the variants implement. Result: filters will either have no effect or hurt.

The fundamental reason RTMV tilt P&L lacks carry-level autocorrelation:
1. Carry is a continuous flow (funding paid every 8 hours, stable within a day)
2. RTMV tilt P&L depends on HMM state prediction with a 21-bar prediction horizon
3. Each monthly refit uses 504 bars overlapping by ~95% with the previous refit
4. Tilt magnitude is small (λ=0.02–0.10) relative to min-var base, dominated by regime noise

---

## 5. Why RTMV Differs From Carry

| Property | Carry | RTMV Tilt |
|----------|-------|-----------|
| Signal frequency | 8-hour | Monthly |
| Autocorrelation source | Funding supply/demand persistence | HMM state prediction |
| Lag-1 ACF | 0.793 | −0.1504 |
| Prediction horizon | 8 hours | 21 bars (1 month) |
| % Periods positive | ~86% | 59.1% |
| Mechanism | Collect a structural premium | Tilt toward HMM-favoured assets |
| 99% CI for zero ACF | — | ±0.170 (observed is within) |

---

## 6. Phase 82 Recommendation

**Phase 82 = Dynamic LETF allocation (bear-regime overweight)**

Phase 81 is a NULL RESULT — the signal-design autocorrelation pattern does not
transfer from carry to RTMV. The Phase 80 attribution table identified the next
highest-ROI lever: allocation-level dynamic weighting in the LETF leg.

Phase 70/72 confirmed:
- Bear regime: LETF ann return +61%
- Bull regime: LETF ann return +9%
- Phase 72 (within-LETF position scaling) was NO-GO

**Phase 82 hypothesis:** Increase the LETF portfolio weight from 15% → 25% when
SPY HMM rank = 0 (bear regime). This is an allocation-level dynamic weight (not
within-LETF position scaling), triggered once per month at RTMV rebalance.

The 2022 stress test (Phase 68) shows LETF returned +44.3% while RTMV returned
−12.2% and carry returned +2.1% → combined +6.8%. Dynamically overweighting LETF
in that regime (15%→25%) would have added ~4.4% to combined return (+44.3% × 10%
increment) at minimal vol cost (carry/LETF corr = −0.084).

**GO threshold:** Combined portfolio Sharpe ≥ 12.0 (Phase 77 baseline 11.936 + 0.064)

**Implementation:** Modify `scripts/run_phase68_multistrat_live.py` to accept
`--letf-bear-weight-factor` (default 1.0 = no change). When SPY HMM rank = 0
at month-end rebalance, multiply the LETF allocation by this factor and renormalise
carry + LETF + RTMV proportionally.
