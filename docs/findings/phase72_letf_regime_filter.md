# Phase 72 — SPY HMM Regime Filter on 3-Pair LETF Combo

**Date:** 2026-05-18
**Verdict: NO-GO** (GO threshold: bull_only_15x Sharpe ≥ 7.06)

## Objective

Test whether applying the SPY HMM bull-only regime filter (1×/1×/1.5×) improves
the 3-pair equal-weight LETF combo validated in Phase 71 (Sharpe 7.06).

Analogous to Phase 55 which found bull-only carry scaling was a STRONG GO (+3.42% ann).

**Prior hypothesis (going in):** likely NULL or NEGATIVE — LETF decay earns from
VOLATILITY, not direction. SPY rank=2 (bull) is typically low-vol, which means
LESS vol-drag decay premium. Bear/neutral regimes (e.g. 2022 crypto winter +48.94%
LETF return) historically produce the most premium via VIX spikes.

## Regime-Conditional LETF 3-Pair Returns

| Regime | Label | N days | Mean daily | Ann return | Ann vol | % positive days |
|---|---|---|---|---|---|---|
| 0 | bear | 738 | +0.00189 | +61.02% | 7.52% | 75.2% |
| 1 | neutral | 1455 | +0.00073 | +20.26% | 2.62% | 68.6% |
| 2 | bull | 1878 | +0.00034 | +8.91% | 1.70% | 63.7% |

**Key finding:** BEAR regimes produce the most LETF decay premium.

BEAR/NEUTRAL regimes produce HIGHER LETF decay premium (bear ann=61.02%, neutral=20.26%, bull=8.91%). This confirms the hypothesis: SPY rank=2 (bull) is low-vol, producing LESS vol-drag premium. Bear/neutral periods (high VIX, e.g. 2022) earn MORE from the delta-neutral pair strategy. Scaling UP in bull regimes (when premium is lower) reduces the expected return per unit of notional.

## Scaling Variant Results

| Variant | Sharpe | vs flat | Ann Ret | Ann Vol | MDD | Calmar | GO? |
|---|---|---|---|---|---|---|---|
| flat | 5.4672 | +0.0000 | 21.05% | 3.85% | -1.74% | 12.08 | NO |
| bull_only_15x | 5.7997 | +0.3325 | 23.44% | 4.04% | -2.61% | 8.99 | NO |
| bear_reduce_05x | 6.4264 | +0.9592 | 18.27% | 2.84% | -2.61% | 7.01 | NO |
| bear_reduce_075x | 6.0307 | +0.5635 | 19.66% | 3.26% | -2.18% | 9.04 | NO |

## Per-Year Returns

| Year | flat | bull_only_15x | bear_reduce_05x | bear_reduce_075x |
|---|---|---|---|---|
| 2010 | +12.56% | +12.57% | +9.77% | +11.16% |
| 2011 | +22.86% | +23.90% | +15.13% | +18.94% |
| 2012 | +7.88% | +8.92% | +8.92% | +8.40% |
| 2013 | +5.16% | +6.39% | +6.39% | +5.78% |
| 2014 | +6.05% | +7.28% | +7.28% | +6.66% |
| 2015 | +10.58% | +11.50% | +9.43% | +10.00% |
| 2016 | +9.39% | +11.02% | +9.36% | +9.38% |
| 2017 | +5.91% | +8.99% | +8.99% | +7.44% |
| 2018 | +21.73% | +24.40% | +16.58% | +19.12% |
| 2019 | +15.57% | +18.72% | +16.59% | +16.08% |
| 2020 | +65.15% | +66.41% | +32.43% | +47.93% |
| 2021 | +15.67% | +18.39% | +18.39% | +17.02% |
| 2022 | +48.95% | +48.95% | +26.26% | +37.14% |
| 2023 | +25.52% | +29.78% | +29.78% | +27.63% |
| 2024 | +31.63% | +40.47% | +38.72% | +35.13% |
| 2025 | +39.38% | +43.40% | +33.78% | +36.56% |
| 2026 | +11.39% | +13.32% | +13.32% | +12.35% |

## GO/NO-GO Verdict

| Criterion | Threshold | Result | Status |
|---|---|---|---|
| bull_only_15x Sharpe | ≥ 7.06 | 5.7997 | FAIL |

**Verdict: NO-GO.**

Estimated combined portfolio impact: +0.0499 Sharpe (15% LETF weight × +0.3325 LETF delta)

## Combined Portfolio Impact (if deployed)

Phase 71 combined baseline (80% carry + 15% LETF + 5% RTMV): Sharpe = 11.5636.

If bull_only_15x were integrated: estimated combined Sharpe ≈ 11.6135.

Note: This is a linear approximation; actual impact depends on how regime-scaled LETF
correlations with carry and RTMV change. The estimate may overstate the benefit if
bull-regime periods coincide with carry bull-regime (correlated regime entry).

## Phase 73 Recommendation (NO-GO path — two options)

**Phase 73 Recommendation (NO-GO path — two options):**

**Option A (Recommended): Carry universe expansion — SOL perpetuals.**
Phase 59 (2023) found SOL/AVAX/DOGE diluted BTC+ETH Sharpe (16.0 threshold).
SOL perpetual funding has been consistently positive through 2024–2026. Check:
if 90-day trailing SOL carry > 5%/yr AND individual_sharpe > 10, test
BTC+ETH+SOL basket. GO threshold: combined Sharpe > 15.86 (current BTC+ETH baseline).

**Option B: Carry-LETF correlation deep dive in bear regimes.**
Phase 71 shows carry↔letf correlation = −0.097 on average.
Test whether this correlation strengthens in bear regimes (LETF earns more from vol,
carry may turn negative from liquidation cascades). If bear-regime correlation < −0.20,
the portfolio allocation should OVER-weight LETF in bear periods, not under-weight.

## Technical Notes

- Global HMM fit: SPY n=3, 3 restarts, seed_base=42. Features: [log_return, 20-day rolling vol].
  Acceptable for hypothesis testing (pairs are always-in; rank only scales an open position).
- LETF pairs: delta-neutral, always-in. r_pair = 3×r_1x - r_3x - cost_daily. Clipped [-50%, +50%].
- Costs: TQQQ/QQQ 2.00%/yr, SOXL/SOXX 3.40%/yr, UPRO/SPY 1.51%/yr (borrow + ER + friction).
- Equal-weight combo: simple mean of three pair daily returns on common dates.
- Data start: 2010-01-01. SOXL inception ~2010-03-12 sets the effective start.
- Sharpe discrepancy vs Phase 71 (5.47 here vs 7.06): Phase 71 ran over 2020–2026 (post-COVID
  vol-spike era). The full 2010–2026 period includes lower-vol years (2012–2016) that dilute the
  Sharpe. Both figures are correct for their respective windows. Phase 72 uses the longer window
  to improve statistical power for the regime-conditional comparison.
- Regime-conditional analysis spans the full 2010–2026 period:
    Bear (738 days, 18%): ann=61.02% — dominated by 2020, 2022, 2011 high-vol periods
    Neutral (1455 days, 36%): ann=20.26% — transition periods, moderate vol
    Bull (1878 days, 46%): ann=8.91% — low-vol trending periods, less decay premium
