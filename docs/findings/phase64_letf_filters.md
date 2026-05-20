# Phase 64 — LETF Decay Short: Regime + Momentum Filter

**Date:** 2026-05-18
**Verdict: NO-GO**

## Strategy

Delta-neutral decay capture: SHORT 1 unit TQQQ + LONG 3 units QQQ (daily).
Net costs: ~2.0%/yr (borrow + friction + spread).
Baseline (always-in) Sharpe: **4.724**.

## Hypothesis

A combined SPY-regime (rank=2 bull) + QQQ 12m-1m momentum filter avoids periods
where TQQQ can strongly recover (momentum reversal after bear regime), preserving
the carry while shedding the worst drawdown episodes.

## Results

| Strategy | Sharpe | Ann Return | Ann Vol | MDD | Invested |
|---|---|---|---|---|---|
| always_in                      | 4.724 | 17.02% | 3.60% | -1.36% | 100.0% |
| regime_only                    | 2.890 | 3.14% | 1.09% | -0.68% | 46.0% |
| momentum_only                  | 4.152 | 13.88% | 3.34% | -1.32% | 83.8% |
| regime_and_momentum            | 2.863 | 2.99% | 1.04% | -0.68% | 42.0% |
| regime_or_momentum             | 4.188 | 14.06% | 3.36% | -1.32% | 87.8% |

## Parameters

- TQQQ/QQQ data from 2010-07-01 (TQQQ inception) to present
- Decay: -r_tqqq + 3 * r_qqq gross, net = gross - 2.0%/yr
- SPY HMM: n=3 states, 3 restarts, seed=42; rank 2 = bull (highest mean log-return)
- QQQ momentum: 12-month minus 1-month cumulative log-return > 0

## Verdict: NO-GO

- Baseline Sharpe: **4.724**
- Best filter: **regime_or_momentum** → Sharpe **4.188**
- Improvement: **-0.536**

No filter improves Sharpe vs always-in. Phase 55 finding confirmed: the delta-neutral decay harvest is not improved by regime or momentum gating. The gross alpha (~5–6%/yr) is a structural vol-path artefact that accrues in all regimes.

## Next Steps

Always-in remains the recommended approach for the LETF leg. No further filter testing warranted.
