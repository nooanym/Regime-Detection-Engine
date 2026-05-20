# Phase 69 — Regime-Adaptive Carry Entry Thresholds

**Date:** 2026-05-18
**Verdict: GO**

## Hypothesis

In SPY bear regime (rank=0), crypto funding carry is still positive on average,
but there are more negative-carry spikes. Raising the entry threshold in bear
markets (8% ann vs baseline 5%) would filter out marginal entries, reducing
drawdown at the cost of some missed carry periods.

**Phase 62 baseline:** `entry_threshold=0.05` flat regardless of SPY rank.
`CarryStrategy(carry_weighted=True, regime_scale={0:1.0, 1:1.0, 2:1.5})`.

## SPY HMM State Distribution (2020–present, n=3)

| State | Days |
|-------|------|
| Bear (rank 0) | 229 |
| Neutral (rank 1) | 484 |
| Bull (rank 2) | 869 |

## Results

| Strategy | Sharpe (Δ) | Ann Return | MDD (Δ) | % Active | Skipped periods |
|---|---|---|---|---|---|
| baseline (flat 5%) | 18.567 (+0.000) | 19.04% | -0.41% (+0.0000) | 93.3% | 0 |
| bear_entry=6% | 18.567 (-0.001) | 19.04% | -0.41% (+0.0000) | 93.3% | +3 |
| bear_entry=8% | 18.565 (-0.002) | 19.04% | -0.41% (+0.0000) | 93.3% | +4 |
| bear_entry=10% | 18.565 (-0.002) | 19.04% | -0.41% (+0.0000) | 93.3% | +4 |
| bull_entry=3% | 18.571 (+0.004) | 19.05% | -0.41% (+0.0000) | 93.3% | 0 |

Notes:
- Skipped = net extra periods excluded by the adaptive threshold vs baseline 5%
- Negative skipped = extra entry periods gained (bull_entry=3% variant)

## GO/NO-GO Criteria

| Criterion | Threshold | Result |
|---|---|---|
| GO criterion A | Sharpe >= 17.0 | 18.571 — PASS |
| GO criterion B | MDD improvement >= 0.05pp AND Sharpe >= 16.5 | FAIL |

## Verdict: MARGINAL GO (effectively NULL RESULT)

**Technically GO** on Criterion A (all variants clear Sharpe >= 17.0 because the
baseline itself is 18.57). However the _practical_ result is a **null finding**:

- All bear-entry raising variants: Sharpe change = −0.001 to −0.002, MDD change = 0.0000pp
- `bull_entry=3%`: Sharpe +0.004, MDD change = 0.0000pp — the "winner" is noise
- Total skipped/gained periods across 6 years: 3–4 periods out of 6,989

**Root cause — why adaptive thresholds are nearly inert:**

The 90-period rolling carry window is a heavily smoothed signal. Even in SPY bear
regimes where 63% of individual 8-hour periods have raw carry below 5% ann, the
trailing 90-period average remains above 5% for most of the bear period because
prior high-carry periods anchor the rolling mean. The entry threshold fires only
when the trailing 90-period mean itself dips below the threshold — which takes
months of sustained low carry, at which point the baseline strategy's −2% exit
threshold has already long closed the position. The adaptive threshold operates
on the same smoothed signal as the entry; it provides no incremental filtering.

**Key empirical finding from carry stats by regime:**

In SPY bear (229 days, ~19% of the sample):
- BTC rolling ann carry mean = +5.35% (barely above the 5% baseline)
- BTC % periods below 5% = 63.2% (but these are raw 8-hour rates, not the 90-period mean)

This confirms Phase 55's finding: ETH/BTC carry is structurally positive even in
bear SPY regimes. The carry strategy's MDD of −0.41% is already essentially zero;
there is no drawdown to reduce.

## Methodology

- Data: Binance perpetual funding rates, 8-hourly, 2020–present
- Rolling carry: 90-period trailing mean of annualised funding rate
- Carry-weighted base: w_i = carry_i / sum(carry_j) clipped at 0
- Entry logic: per-symbol check of rolling ann carry vs regime-specific threshold
- Exit threshold: -2% annualised (universal, not regime-adaptive)
- Regime scale (bull-only): rank 0 → 1.0×, rank 1 → 1.0×, rank 2 → 1.5×
- SPY HMM: n=3 Gaussian, features=(log_return, vol_20), n_restarts=3, seed_base=42
- Cost: ~0.5% annual friction applied per occupied period

## Implications

**Adaptive entry threshold tuning is exhausted.** The 90-period rolling carry
entry filter already acts as a sufficient smoothing mechanism. There is no
additional information in the SPY regime label that would improve the entry filter
beyond what the trailing carry estimate already captures.

**Recommended next direction:** The carry strategy's MDD of −0.41% is essentially
optimal for a pure carry strategy. Future alpha comes from universe expansion
(new assets with uncorrelated carry cycles) or basis-risk reduction, not from
threshold tuning.

**The Phase 62 config (`flat 5% entry, bull-only scale, carry_weighted=True`)
remains the definitive carry deployment. No parameter changes recommended.**
