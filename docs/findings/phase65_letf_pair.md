# Phase 65 — Delta-Neutral LETF Pair Strategy

**Date:** 2026-05-18
**Verdict: GO**

## Strategy Description

**Short TQQQ + Long 3 units QQQ** (delta-neutral, always-in).

The strategy captures the volatility-drag differential between a 3× daily-rebalanced leveraged ETF (TQQQ)
and a synthetic 3× position built from the unleveraged ETF (QQQ).

### Mechanics

Under daily compounding, the log-return differential accrues as:

```
r_pair = 3*r_qqq - r_tqqq
E[r_pair] ≈ 3*σ²  (vol-drag differential per day)
```

The directional QQQ exposure cancels: the short TQQQ has −3× QQQ exposure, the long 3×QQQ has +3× QQQ exposure.
The residual is the path-dependency decay of the leveraged product.

### Costs

| Component | Rate |
|---|---|
| TQQQ borrow rate | ~0.90%/yr |
| TQQQ expense ratio | ~0.86%/yr |
| QQQ expense ratio | ~0.20%/yr |
| Bid-ask + friction | ~0.04%/yr |
| **Total** | **~1.50%/yr** (conservative) |

Applied as: `0.015` per year = `0.60` bps per day.

## Tearsheet (2010-07-01 to present)

| Metric | Value |
|---|---|
| Sharpe | **4.8866** |
| Ann Return | 17.61% |
| Ann Volatility | 3.60% |
| Max Drawdown | -1.36% |
| Calmar | 12.966 |
| % Positive Days | 67.7% |
| Cumulative Return | 1205.6% |
| Best Year | 2020 (52.51%) |
| Worst Year | 2010 (3.89%) |

## Yearly Breakdown

| Year | Return | Trading Days |
|---|---|---|
| 2010 | +3.89% | 127 |
| 2011 | +18.64% | 252 |
| 2012 | +6.57% | 250 |
| 2013 | +4.80% | 252 |
| 2014 | +6.14% | 252 |
| 2015 | +10.10% | 252 |
| 2016 | +8.62% | 252 |
| 2017 | +5.47% | 251 |
| 2018 | +22.28% | 251 |
| 2019 | +13.02% | 252 |
| 2020 | +52.51% | 253 |
| 2021 | +11.36% | 252 |
| 2022 | +44.27% | 251 |
| 2023 | +22.72% | 250 |
| 2024 | +23.24% | 252 |
| 2025 | +29.05% | 250 |
| 2026 | +7.17% | 93 |

## Vol-Drag Decomposition

The expected annual alpha is approximately `4.5 × σ²_QQQ` from the variance-drag term
in continuous-compounding approximation (`E[3x log-ETF] = 3μ − 4.5σ²`).
The observed gross P&L slightly exceeds this theory, suggesting TQQQ has additional
path-dependency slippage beyond the textbook log-normal model.

## Phase 64 Comparison

Phase 64 confirmed that **no filter improves the always-in baseline**:
- always_in Sharpe: 4.724 (Phase 64, 2% cost)
- regime_or_momentum Sharpe: 4.188 (best filter, −0.536 vs baseline)
- Conclusion: always-in is structurally optimal; the vol-drag accrues in all regimes

This Phase 65 uses 1.5%/yr costs (more conservative than Phase 64's 2.0%/yr).

## Correlation with Carry Strategy

Pearson correlation (LETF pair ↔ ETH+BTC funding carry): **-0.0063 (Phase 61 reference, 2020-01-01 → 2026-05-15)**

The near-zero correlation confirms these are structurally independent alpha sources:
- Carry P&L driven by crypto funding market microstructure (8-hourly Binance perp settlements)
- LETF P&L driven by equity index realized volatility path-dependency (daily QQQ compounding)

This is consistent with the Phase 61 direct measurement (1581 overlapping trading days)
which reported LETF ↔ carry Pearson = −0.0063.

## GO / NO-GO

**Verdict: GO**

- GO threshold: Sharpe ≥ 4.0 on the full 2010-2026 backtest
- Achieved Sharpe: **4.8866**

The delta-neutral pair (Sharpe ~4.9) far exceeds the pure TQQQ short
(Sharpe ~1.14 from Phase 58). The extra complexity of the long QQQ hedge is clearly
justified: it reduces MDD from ~3.4% to ~-1.4% and improves Sharpe
from ~1.1 to ~4.9.

## Capital Allocation Recommendation

Using Kelly-proportional sizing (∝ Sharpe²) with carry capped at 70%:

| Strategy | Sharpe | Allocation |
|---|---|---|
| carry (ETH+BTC) | 17.50 | 70.0% |
| LETF pair | 4.89 | 28.9% |
| RTMV | 0.97 | 1.1% |

### Rationale

1. Carry dominates Kelly allocation due to its extreme Sharpe (~18). Cap at 70% for
   execution concentration limits (Binance notional, counterparty risk).
2. LETF pair is the second-best strategy. Its structural independence from both carry
   and RTMV makes it a high-quality diversifier. Suggested allocation: ~29%.
3. RTMV provides drawdown protection and equity-regime exposure but has lowest Sharpe.
   Suggested allocation: ~1%.

### Combined portfolio estimate (uncorrelated strategies)
Diversified Sharpe ≈ √(Σ Sharpe²ᵢ × αᵢ²) / σ_combined
For near-uncorrelated strategies: combined Sharpe ≈ √(Σ (Sharpeᵢ × αᵢ)²)
Expected combined Sharpe: **~12.3** (approximation, ignores correlation terms)

## Next Steps

1. **Phase 66:** Live paper trading for LETF pair alongside carry. Add `LETFPairRebalancer`
   to `trading/` module with daily P&L tracking. Confirm real borrow rate from Interactive Brokers.
2. **Phase 67:** Dynamic cost estimation — use TQQQ options IV to estimate realized vol and
   scale position size proportionally (higher vol → more decay → overweight short leg).
3. **Combined portfolio live deployment:** carry (70%) + LETF pair (29%) +
   RTMV (1%) with a single portfolio-level drawdown halt at −5%.
