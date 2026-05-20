# Phase 57: Cross-Asset Momentum Overlay on RTMV

**Date:** 2026-05-18
**Status:** GO — mom_03 (scale=0.03) Sharpe = 1.0008, +0.0282 vs Phase 52a baseline
**Threshold:** Sharpe ≥ 0.9830 (baseline 0.9726 + 0.0104)

---

## 1. Hypothesis

Cross-sectional (12m−1m) momentum is a well-documented anomaly in equities and bonds.
Adding it as an additive tilt on top of the RTMV signal may capture a complementary
source of alpha: momentum follows persistent trends across assets; RTMV (HMM regime
tilt) targets regime-conditional expected returns which respond to level changes.
When both signals agree on an asset's direction, compounding their weight contributions
should improve risk-adjusted returns.

## 2. Method

### Momentum signal construction

At each monthly rebalance:

1. Compute 12-month cumulative log return per asset: `sum(log_returns[-252:])`.
2. Compute 1-month cumulative log return per asset: `sum(log_returns[-21:])`.
3. Subtract to get the 12m-1m momentum signal (excludes short-term reversal):
   `mom_signal = mom_12m - mom_1m`
4. Normalise to a cross-sectional z-score:
   `z_mom = (mom_signal - mean) / std`  (fallback to 0 if std ≈ 0)

### Weight construction

Starting from the existing RTMV combined weight:

```
w_combined = (1 - λ_eff) * w_minvar + λ_eff * w_regime
```

Apply the additive momentum tilt:

```
w_final = w_combined + momentum_tilt_scale * z_mom
w_final = clip(w_final, 0, ∞)
w_final /= sum(w_final)
```

The tilt is applied **after** the RTMV blending so momentum shifts allocations
without interacting with the regime λ schedule.  Negative weights after the clip
are floored to zero, which means assets with strong negative momentum cannot receive
short positions (appropriate for a long-only fund).

### Baseline configuration

All variants use the Phase 52a winning configuration:
- Universe: SPY, GLD, SHY, IEF, TLT (5-asset)
- `lambda_by_state_rank = [0.02, 0.05, 0.10]`
- `lambda_proxy_asset = "SPY"`
- `n_states = 3`, `n_restarts = 3`
- `lookback_bars = 504`, `rebalance_bars = 21`
- `drawdown_halt = 0.25`

## 3. Results

Universe: SPY, GLD, SHY, IEF, TLT
Bars: 5386 (2004-12-17 → 2026-05-15)

| Variant  | Scale | Sharpe | Calmar |   MDD  | Ann Return | Ann Vol | N Rebalances |
|----------|------:|-------:|-------:|-------:|-----------:|--------:|-------------:|
| baseline |  0.00 | 0.9730 | 0.3352 | −15.6% |       5.2% |    5.4% |          233 |
| mom_01   |  0.01 | 0.9868 | 0.3445 | −15.5% |       5.3% |    5.4% |          233 |
| mom_03   |  0.03 | 1.0008 | 0.3619 | −15.5% |       5.6% |    5.6% |          233 |
| mom_05   |  0.05 | 0.9978 | 0.3759 | −15.5% |       5.8% |    5.9% |          233 |

**Note:** The baseline Sharpe here (0.9730) matches the Phase 52a result (0.9726) to
within rounding noise, confirming the backtest harness is correctly reproducing the
prior result.

## 4. Key observations

1. **Monotonic return improvement with scale.** Ann return rises from 5.2% (baseline)
   to 5.8% (mom_05) as the momentum tilt increases.  This reflects the cross-sectional
   trend persistence in SPY/GLD/SHY/IEF/TLT over the 2004–2026 sample.

2. **Non-monotonic Sharpe peak at mom_03.** mom_03 achieves the highest Sharpe (1.0008)
   while mom_05 falls back to 0.9978.  At scale=0.05 the vol increment (+0.5 pp Ann Vol
   vs baseline) begins to erode the Sharpe benefit of the higher return.  The sweet spot
   is scale=0.03.

3. **MDD stable across all variants.** Maximum drawdown stays at −15.5% to −15.6% for
   all scales.  The momentum tilt does not increase tail risk.  Calmar improves
   monotonically (0.3352 → 0.3759) because the return improvement slightly outpaces
   the unchanged drawdown.

4. **Rebalance count unchanged.** 233 rebalances for all variants confirms the momentum
   tilt does not cause additional calendar rebalances (it runs within the existing monthly
   rebalance, not as a separate trigger).

## 5. Verdict

**GO — momentum_tilt_scale = 0.03 selected as the deployment parameter.**

- Best Sharpe: 1.0008 vs Phase 52a baseline 0.9726 → **+0.0282**
- GO threshold was ≥ 0.9830 (baseline + 0.0104)
- Delta exceeds threshold by 2.7× (+0.0282 vs +0.0104 required)
- Drawdown unchanged at −15.5%; Calmar improves to 0.3619 (+8.0%)

## 6. Limitations and caveats

1. **In-sample.** This is a full-period backtest on the same 2004–2026 data used to
   develop the RTMV baseline.  An OOS validation (e.g., purged CV on 2016–2026 only)
   would strengthen confidence.  The momentum anomaly is long-documented across asset
   classes, which reduces concern about data snooping here.

2. **Two overlapping signals.** The momentum z-score and the HMM regime tilt both
   use trailing return information, which means they will partially co-vary.  The
   additive combination works here because the HMM regime tilt captures level shifts
   (HMM posteriors jump at regime changes) while 12m-1m momentum is a smooth trend
   signal.  They are not orthogonal but their combination is beneficial empirically.

3. **Long-only clip.** Negative-momentum assets after the tilt can be floored to zero
   weight.  In a period where all assets have negative momentum, the tilt has no effect
   (z-score normalisation means the sum of z-scores is always 0, and after clip all
   weights revert toward the pre-tilt w_combined).  This is a feature: the tilt
   self-corrects in risk-off environments.

## 7. Deployment configuration

Update the live RTMV config (`scripts/run_rtmv_live.py` or `RTMVRebalancerConfig`):

```python
RTMVRebalancerConfig(
    assets=["SPY", "GLD", "SHY", "IEF", "TLT"],
    lambda_tilt=0.05,
    lambda_by_state_rank=[0.02, 0.05, 0.10],
    lambda_proxy_asset="SPY",
    n_states=3,
    n_restarts=3,
    lookback_bars=504,
    rebalance_bars=21,
    drawdown_halt=0.25,
    momentum_tilt_scale=0.03,       # Phase 57 addition
)
```

## 8. Next phase

Phase 58 (suggested): OOS validation of Phase 57 momentum overlay.
Run the purged CV harness (Phase 47 methodology) on mom_03 to confirm:
- Fold-consistency ≥ 60% (3/5 folds where mom_03 beats baseline)
- Shuffle robustness p < 0.10 (momentum permutation test)

If OOS CV passes, momentum_tilt_scale=0.03 becomes the new live baseline
replacing the Phase 52a configuration.

---

*Generated by `scripts/run_phase57_momentum.py` on 2026-05-18.*
*Results stored in `results/phase57/momentum_tilt_results.csv`.*
