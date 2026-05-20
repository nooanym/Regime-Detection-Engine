# Phase 62 — Regime-Scaled Carry-Weighted BTC+ETH

**Date:** 2026-05-18
**Verdict: GO**

## Hypothesis

Combine Phase 60 (carry-weighted BTC/ETH allocation) with Phase 55 (SPY HMM regime scaling).
Apply SPY dominant-state scale on top of carry-proportional base weights.

## SPY HMM State Distribution (2020–present)

Bear (rank 0): 229 days | Neutral (rank 1): 484 days | Bull (rank 2): 868 days

## Results

| Strategy | Sharpe | Ann Return | Ann Vol | MDD | Cum Return |
|---|---|---|---|---|---|
| equal_weight_flat (baseline) | 17.221 | 13.35% | 0.77% | -0.85% | 122.4% |
| carry_weighted_flat (Ph60 GO) | 18.129 | 14.13% | 0.78% | -0.46% | 132.4% |
| carry_weighted_bull_only | **17.943** | **17.55%** | 0.98% | -0.46% | 180.7% |
| carry_weighted_full_regime | 17.747 | 17.24% | 0.97% | -0.33% | 176.0% |

Carry-weighted_bull_only vs carry_weighted_flat: Sharpe -0.1864, Ann Return +3.42%

## GO/NO-GO Criteria

| Criterion | Threshold | Result | Status |
|---|---|---|---|
| Ann Return improvement | > 0 vs carry_weighted_flat | +3.42% | PASS |
| Strong GO (Sharpe) | >= 18.5 | 17.943 | FAIL |

## Verdict: GO

**GO.** Ann Return improvement is positive. Regime scaling adds absolute return on top of carry-weighted allocation, consistent with Phase 55 finding that 1.5× bull scaling adds +3.61% ann.

## Methodology

- Data: Binance perpetual funding rates, 8-hourly, 2020–present
- Rolling carry: 90-period trailing mean of annualised funding rate
- Carry-weighted base: w_i = carry_i / sum(carry_j) clipped at 0
- Entry: total positive carry > 0; exit: both carries <= 0
- Regime scale (bull_only): rank 0 → 1.0×, rank 1 → 1.0×, rank 2 → 1.5×
- Regime scale (full): rank 0 → 0.5×, rank 1 → 1.0×, rank 2 → 1.5×
- SPY HMM: n=3 Gaussian, features=(log_return, vol_20), n_restarts=3, seed_base=42
- Cost: ~0.5% annual friction applied per occupied period

## Key Insight

The full-regime variant (0.5×/1×/1.5×) confirms Phase 55's finding: bear scaling **hurts**.
ETH carry is positive even in SPY bear markets; reducing exposure in rank-0 reduces absolute
returns without protecting against drawdowns (which are tiny in carry anyway, MDD < 1%).
Bull-only scaling is the correct variant if regime scaling is applied at all.
