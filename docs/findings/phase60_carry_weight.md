# Phase 60 — Dynamic Carry-Weighted BTC+ETH Allocation

**Date:** 2026-05-18
**Verdict: STRONG GO**

## Hypothesis

Weight BTC vs ETH proportionally to their trailing 90-period annualised carry.

## Results

| Strategy | Sharpe | Ann Return | Ann Vol | MDD | Invested |
|---|---|---|---|---|---|
| equal_weight (baseline) | 17.499 | 13.62% | 0.78% | -0.85% | 100.0% |
| carry_weighted | **18.129** | 14.13% | 0.78% | -0.46% | 94.2% |
| max_carry | 17.725 | 15.60% | 0.88% | -0.34% | 94.2% |

Sharpe improvement (carry_weighted vs equal_weight): +0.630

Avg ETH weight when in market: 51.2%  |  Avg BTC weight: 48.8%

Yearly avg ETH in-market weight:
- 2020: 65.3%
- 2021: 58.8%
- 2022: 30.0%
- 2023: 51.1%
- 2024: 55.5%
- 2025: 49.3%
- 2026: 32.3%

## Methodology

- Data: Binance perpetual funding rates, 2020-present
- Rolling carry estimate: 90-period trailing mean of annualised funding rate
- Carry-weighted: w_i = carry_i / sum(carry_j) for carry_i > 0, else 0
- Entry: either carry > 0; exit: both carries <= 0 (~5.8% of periods skipped)
- Cost: ~0.5% annual friction per 8-hour period

## Verdict

**STRONG GO.** Sharpe improvement = +0.630.
Dynamic carry weighting adds value over static 50/50. ETH dominates in bull years (2020 65%, 2021 59%) and correctly recedes when BTC carry leads (2022 30%). MDD improves from -0.85% to -0.46% by skipping negative carry periods.
Sharpe 18.129 >= 16.5 qualifies as Strong GO.

## Next Steps

- Phase 61: Regime-scaled carry overlay — use SPY HMM state rank to scale carry notional (rank-0 bear: 0.5x, rank-1 neutral: 1.0x, rank-2 bull: 1.5x). Layer on top of carry_weighted base weights.
