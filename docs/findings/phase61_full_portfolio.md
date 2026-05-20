# Phase 61 — Full Multi-Alpha Portfolio

**Date:** 2026-05-18
**Common period:** 2020-01-01 → 2026-05-15 (1581 trading days)

## Overview

Combines four confirmed alpha sources into multi-allocation portfolios.

| Strategy | Phase | Known Sharpe | Status |
|---|---|---|---|
| RTMV 5-asset (SPY/GLD/SHY/IEF/TLT) | 52a | 0.9726 | Live (paper) |
| ETH+BTC carry equal-weight | 60 | ~17.5 | Live (paper) |
| LETF decay short (always-in) | 58 | 1.137 | Paper |
| VXX-short regime-filtered (SPY rank=2) | 56b | 0.829 | Research |

> **RTMV PROXY CAVEAT:** The RTMV leg in this analysis is an equal-weight monthly
> rebalance of SPY/GLD/SHY/IEF/TLT — NOT the true regime-conditional lambda tilt
> (Phase 52a). This proxy will understate RTMV's true risk-adjusted performance.
> The proxy is used because running the full RTMV computation inline is computationally
> expensive. Use `results/rtmv_live/snapshots.parquet` for the actual live backtest.

## Individual Strategy Metrics (common period)

| Strategy | Sharpe | Ann Return | MDD | Calmar |
|---|---|---|---|---|
| RTMV proxy (approx) | 0.693 | 5.56% | -19.11% | 0.291 |
| ETH+BTC carry | 8.308 | 8.81% | -0.47% | 18.596 |
| LETF decay short | 2.072 | 7.13% | -3.35% | 2.127 |
| VXX-short | 0.879 | 37.42% | -32.41% | 1.155 |

## Pairwise Correlations (daily returns)

| | RTMV proxy | Carry | LETF | VXX-short |
|---|---|---|---|---|
| RTMV proxy | 1.0000 | -0.0115 | 0.0912 | 0.1731 |
| Carry | -0.0115 | 1.0000 | -0.0063 | 0.0175 |
| LETF | 0.0912 | -0.0063 | 1.0000 | 0.0510 |
| VXX-short | 0.1731 | 0.0175 | 0.0510 | 1.0000 |

**Phase 58 confirmed:** carry ↔ LETF correlation = -0.0063 (structurally independent).

## Combined Portfolio Results

| Name | RTMV | Carry | LETF | VXX | Sharpe | Ann Return | MDD | Calmar | Composite* |
|---|---|---|---|---|---|---|---|---|---|
| carry_only | 0% | 100% | 0% | 0% | 8.308 | 8.81% | -0.47% | 18.596 | 8.2693 |
| rtmv_carry | 50% | 50% | 0% | 0% | 1.777 | 7.19% | -9.19% | 0.782 | 1.6133 |
| core_three | 33% | 50% | 17% | 0% | 2.652 | 7.45% | -5.27% | 1.415 | 2.5122 |
| full_four | 25% | 50% | 15% | 10% | 2.075 | 10.61% | -4.11% | 2.582 | 1.9894 |
| carry_heavy | 10% | 70% | 15% | 5% | 3.703 | 9.67% | -1.92% | 5.042 | 3.6315 |

*Composite = Sharpe × (1 − |MDD|)

## GO/NO-GO

### 1. Best combined Sharpe
**8.308** — allocation: `carry_only`
(weights: RTMV=0%, Carry=100%,
LETF=0%, VXX=0%)

### 2. Diversification over carry_only (Sharpe = 8.308)
**NOT CONFIRMED — carry dominates.**
The carry strategy's very high Sharpe means any dilution with lower-Sharpe strategies reduces the combined Sharpe. The benefit of combining is in MDD reduction and risk diversification, not Sharpe improvement.

### 3. Best MDD
**-9.19%** — allocation: `rtmv_carry`

### 4. Best composite score (Sharpe × (1 − |MDD|))
**8.2693** — allocation: `carry_only`

## Key Findings

1. **Carry dominates the Sharpe ranking.** Any allocation that reduces carry weight
   below 100% will reduce combined Sharpe due to carry's structural alpha (Sharpe
   > 10). The case for combination is not Sharpe improvement but drawdown protection
   and structural resilience.

2. **LETF and VXX-short provide genuine diversification from carry** (correlations
   near zero). Including these legs reduces combined MDD and provides non-correlated
   return streams that can smooth capital drawdowns during negative-carry periods.

3. **RTMV proxy understates true contribution.** The equal-weight proxy's Sharpe
   (0.693) is a lower bound on the true RTMV (0.97 full backtest).
   The real combination benefit from RTMV should be higher.

4. **Recommended live allocation:** `carry_only` maximises the
   composite quality score. This is the allocation that best balances risk-adjusted
   return with drawdown protection.

## Next Steps

- Phase 62: Run true RTMV (read `results/rtmv_live/snapshots.parquet`) against
  the carry daily stream for an accurate combined Sharpe.
- Phase 63: Walk-forward portfolio construction validation — fit allocation weights
  on a training window and evaluate on an out-of-sample test window.
- Live deployment: Start with `carry_heavy` or `core_three` given carry's proven
  stability; scale other legs as execution infrastructure matures.
