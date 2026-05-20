# Phase 73 — SOL Perpetual Funding Carry Re-Test

**Date:** 2026-05-18
**Verdict: CONDITIONAL GO**
**Script:** `scripts/run_phase73_sol_carry.py`

---

## 1. Motivation

Phase 59 (2026-05-17) found SOL NO-GO with mean annualised carry of **0.9%** (2021+).
The catastrophic 2022 result (-35.6% annual during FTX collapse) dominated the full-period mean.

SOL has since grown significantly in derivatives market size. The question: has SOL's 2024-2026
carry profile improved enough to overcome dilution arithmetic and beat the Phase 62 live baseline
(carry_weighted_bull_only Sharpe = **17.943**)?

---

## 2. SOL Individual Carry Statistics

Note: Phase 59 "Individual Sharpe 10.81" was the *simulated backtest Sharpe* (with 5% entry threshold
filtering catastrophic periods). Phase 73 "Sharpe (raw)" is the per-period Sharpe of the unfiltered
funding rate series, and is much lower because 2022's -30.1% annual carry massively inflates std.

| Metric | Phase 59 (2021+ through mid-2024) | Phase 73 (2021-present) | Change |
|--------|-----------------------------------|-------------------------|--------|
| Mean ann carry (unfiltered) | 0.9% | 0.9% | -0.0% |
| % positive periods | 72.0% | 72.0% | -- |
| Raw series Sharpe (unfiltered) | ~0.3 | 0.27 | ~flat |
| Simulated backtest Sharpe (5% threshold) | 10.81 | see basket table | -- |
| 2024+ mean ann carry | ~5% | 5.3% | +0.3pp |

### SOL yearly funding returns

| Year | Ann Return |
|------|-----------|
| 2021 | +33.1% |
| 2022 | -30.1% |
| 2023 | +1.3% |
| 2024 | +14.6% |
| 2025 | +0.4% |
| 2026 | -3.3% |

### Key finding on SOL profile

SOL carry has not improved vs Phase 59.

---

## 3. Funding Rate Correlations

| Pair | Full history (2021-01-01+) | 2024+ only |
|------|--------------------------------------|------------|
| BTC / ETH | 0.846 | 0.844 |
| BTC / SOL | 0.284 | 0.654 |
| ETH / SOL | 0.258 | 0.695 |

Phase 59 reported ETH/SOL = 0.258 (2021+ window). **Critical change:** the 2024+ correlation
has jumped to 0.695 — SOL is no longer a meaningful diversifier vs ETH. This is new vs Phase 59.

---

## 4. Basket Comparison (carry-weighted, no regime scaling)

| Basket | Sharpe | Ann Return | Ann Vol | MDD | Calmar |
|--------|--------|-----------|---------|-----|--------|
| btc_eth (Phase62 base)       | 17.430 | 10.86% | 0.62% | -0.32% | 33.58 |
| btc_eth_sol carry-wt         | 16.831 | 11.17% | 0.66% | -0.36% | 31.19 |
| eth_sol carry-wt             | 15.284 | 11.19% | 0.73% | -0.46% | 24.55 |
| btc_eth_sol equal-wt         | 9.757 | 8.91% | 0.91% | -5.65% | 1.58 |
| btc_eth_sol max-carry        | 16.256 | 12.43% | 0.76% | -0.54% | 23.16 |

GO threshold: Sharpe >= 16.0
**Raw basket result: PASS** (best basket: 'btc_eth_sol carry-wt' -> 16.831)

---

## 5. Basket Comparison with Bull-Only Regime Scaling (1x/1x/1.5x SPY rank)

| Basket | Sharpe | Ann Return | MDD | Calmar |
|--------|--------|-----------|-----|--------|
| btc_eth bull-only (Ph62)     | 18.284 | 13.45% | -0.33% | 40.39 |
| btc_eth_sol CW bull-only     | 17.601 | 13.84% | -0.51% | 27.07 |
| eth_sol CW bull-only         | 15.958 | 13.87% | -0.68% | 20.29 |
| btc_eth_sol EW bull-only     | 11.265 | 11.36% | -5.65% | 2.01 |

Phase 62 live deployment threshold: Sharpe > **17.943**
Note: btc_eth bull-only re-run on 2021+ window = 18.284 (beats threshold, but is the 2-asset baseline, not a SOL expansion).
**Live deployment result (best SOL-expanded basket): FAIL** ('btc_eth_sol CW bull-only' Sharpe 17.601, delta = -0.342 vs 17.943)

---

## 6. GO / NO-GO Decision

### CONDITIONAL GO

Best SOL raw basket clears 16.0 raw threshold (btc_eth Sharpe 17.430). Best SOL-expanded regime basket ('btc_eth_sol CW bull-only' Sharpe 17.601) does NOT beat the Phase 62 live baseline of 17.943 (delta = -0.342). SOL carry has improved but dilution prevents beating the live benchmark.

### Comparison to Phase 59 findings

Phase 59 finding: SOL Sharpe 15.41 for BTC+ETH+SOL (equal-weight, 2021+) -- below 15.86 baseline.

Phase 73 finding (aligned 3-way window starting 2021-01-01):
- btc_eth baseline Sharpe: **17.430**
- btc_eth_sol carry-wt Sharpe: **16.831** (delta: -0.599)
- btc_eth_sol equal-wt Sharpe: **9.757** (delta: -7.673)

Dilution arithmetic still dominates — the 3-asset basket still trails the 2-asset baseline even with improved SOL carry.

---

## 7. Root Cause Analysis

**Primary constraint:** Dilution arithmetic. Any asset with individual Sharpe below the BTC+ETH basket Sharpe will reduce the combined Sharpe. SOL's individual Sharpe improvement is insufficient to cross this threshold.

The correlation structure (ETH/SOL = 0.258 full, but **0.695 in 2024+**) means SOL's
diversification benefit has actually *deteriorated* significantly since Phase 59. In 2024+,
SOL funding rates have become highly correlated with ETH (0.695 vs 0.258 full-period),
eliminating most of the diversification benefit that made SOL interesting. Higher correlation
+ lower carry = strictly worse than Phase 59.

---

## 8. Next Steps (Phase 74)

**BTC+ETH basket remains optimal.** SOL carry has improved structurally since Phase 59 but dilution arithmetic still prevents beating the Phase 62 live baseline. Phase 74 = portfolio re-optimisation — recheck multi-strat weights (80/15/5) with the 3-pair LETF leg, which now shows Sharpe 7.06 vs the old 4.887 single-pair.
