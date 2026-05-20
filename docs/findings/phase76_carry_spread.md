# Phase 76 — ETH-BTC Carry Spread Signal for Relative Weighting

**Date:** 2026-05-18
**Verdict: GO**

## Part A: Momentum Filter Deployment

The Phase 75 lag-1 momentum filter has been deployed to `CarryStrategy` in
`src/rde/trading/carry_executor.py`.

### Changes made

- `CarryStrategy.__init__()`: added `momentum_filter: bool = False` parameter
  (default False for backward compatibility with existing callers).
- `CarryStrategy._last_combined_rate: float | None = None`: tracks previous
  period's portfolio-level carry rate.
- `CarryStrategy.step()`: when `momentum_filter=True` and previous combined
  rate was ≤ 0, exits all open positions and holds cash for that period. The
  `_last_combined_rate` is updated *before* the skip logic so the filter state
  is maintained correctly regardless of exit paths.
- `scripts/run_carry_live.py`: added `--momentum-filter` CLI flag.
- `Makefile`: added `carry-live-filtered` target (live mode with filter enabled)
  and `carry-phase76` target.

### Verification

| Variant | Sharpe | Ann Ret | MDD |
|---|---|---|---|
| baseline (no momentum filter) | 18.1215 | 14.12% | -0.46% |
| baseline + momentum filter (Phase 75 GO) | 18.5646 | 14.18% | -0.22% |
| Sharpe improvement from filter | +0.4431 | — | — |

The momentum-filtered baseline Sharpe of **18.5646** is consistent with the
Phase 75 GO result (+0.4431 improvement vs unfiltered).

## Part B: Instantaneous ETH-BTC Carry Spread Signal

### Hypothesis

When the spot ETH carry rate is > 1.5× the BTC carry rate right now (not rolling average),
overweight ETH to 75%+. This is a faster signal than the Phase 60 rolling 90-period weights.

### Dataset

- Symbols: BTCUSDT, ETHUSDT perpetual futures on Binance
- Period: 2020-present (6995 periods)
- Frequency: 8-hourly

### ETH/BTC Instantaneous Ratio Distribution

| Statistic | Value |
|---|---|
| Mean ratio | 0.039 |
| Median ratio | 1.000 |
| Periods where ratio > 1.5× | 21.2% |
| Periods where ratio < 1.0× | 41.2% |

### Variant Comparison (all with Phase 75 momentum filter applied)

| Variant | Sharpe | Ann Ret | Ann Vol | MDD | vs Baseline | GO? |
|---|---|---|---|---|---|---|
| baseline_no_filter | 18.1215 | 14.12% | 0.78% | -0.46% | -0.4431 |  |
| baseline_carry_wt | 18.5646 | 14.18% | 0.76% | -0.22% | +0.0000 |  |
| spot_spread_15x | 19.0785 | 15.16% | 0.79% | -0.03% | +0.5139 | GO |
| spot_spread_dynamic | 19.7252 | 15.53% | 0.79% | -0.02% | +1.1606 | GO |

**GO threshold: Sharpe >= 18.6**

### Verdict: GO

**GO.** The `spot_spread_dynamic` variant achieves Sharpe 19.7252, clearing the GO threshold of 18.6. The instantaneous ETH-BTC carry spread provides incremental signal for relative weighting beyond the Phase 60 rolling 90-period baseline.

## Phase 77 Recommendation

Phase 77: Integrate instantaneous spread weighting into `CarryStrategy` as a new `spot_spread_weight` parameter. Update the combined multi-strategy tearsheet (Phase 68 / `make multistrat-backtest`) with the momentum-filtered carry leg deployed.

## Methodology

- Carry weights (baseline): 90-period trailing rolling mean of annualised rate,
  proportional allocation clipped at 0 (Phase 60)
- spot_spread_15x: ETH weight = 0.75 when ETH_rate/BTC_rate > 1.5, else 0.50
- spot_spread_dynamic: ETH weight = min(0.80, max(0.40,
    ETH_rate / (ETH_rate + BTC_rate))), using clipped instantaneous rates
- All variants: entry requires any positive carry; Phase 75 momentum filter applied
- Cost: ~0.5% annual friction per active 8h period
- Backtest window: 2020-present
