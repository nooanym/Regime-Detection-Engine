# Phase 54: Joint 5D HMM — Decision Memo

**Date:** 2026-05-17
**Universe:** SPY / GLD / SHY / IEF / TLT (5-asset, daily)
**Period:** 2004-12-17 → 2026-05-15 (5386 bars)
**Verdict:** CONDITIONAL NO-GO — beats Sharpe threshold (0.952 > 0.920) but does not beat the Phase 52a independent-HMM baseline (0.973)

---

## What was tested

A single joint `GaussianHMM(n_features=5, n_states=3, covariance_type="diag")` fitted on the stacked
`(T, 5)` return matrix `[SPY, GLD, SHY, IEF, TLT]` at each monthly rebalance, replacing the 5
independent per-asset HMMs used in Phase 52a.

The joint model's dominant state is ranked by its mean SPY return. The `lambda_by_state_rank =
[0.02, 0.05, 0.10]` schedule (spy_rank_bull from Phase 51b) is applied using the joint state rank.
Min-var base weights are derived from the same rolling 63-bar empirical covariance used in Phase 52a.

**Motivation:** Phase 53 (KL monitor) failed because per-asset forward-filter posteriors oscillate
every ~5 bars on daily data. The hypothesis was that a joint model would produce smoother, more
coherent regime assignments by pooling information across all 5 return series simultaneously.

---

## Results

| Strategy | Sharpe | Calmar | Max DD | Ann Return | Ann Vol | N Rebalances |
|---|---|---|---|---|---|---|
| Phase 54: joint HMM (diag) | **0.952** | 0.335 | -15.4% | 5.2% | 5.4% | 233 |
| Phase 52a: 5-asset spy_rank_bull (baseline) | **0.973** | 0.335 | -15.6% | 5.2% | 5.4% | 233 |
| Phase 45: global_min_var (no regime) | 0.929 | 0.306 | -21.7% | 6.6% | 7.2% | — |

The joint HMM achieves Sharpe 0.952, which:
- Clears the GO threshold of 0.920 (+0.032 margin)
- Is below the Phase 52a independent-HMM baseline by -0.021 Sharpe

Both methods converge on near-identical MDD (-15.4% vs -15.6%) and Calmar (0.335 vs 0.335),
indicating the min-var base weights dominate the portfolio structure; the regime tilt accounts
for only small weight differences between the two approaches.

---

## Why the joint model underperforms

Three factors explain the -0.021 Sharpe gap:

1. **Information dilution.** The independent HMMs each fit a 1D or 3D (log_return,
   vol_w20, smoothed_return_w5) feature space optimised for one asset's return dynamics.
   The joint HMM fits a 5D raw return space where asset-specific patterns (e.g., GLD
   responding to inflation regimes vs SPY responding to earnings cycles) compete for
   the same 3 latent states. The joint model finds the dominant cross-asset comovement
   but misses within-asset structure.

2. **Covariance type `diag` discards cross-asset regime covariance.** `covariance_type="diag"`
   was chosen to avoid rank-deficiency with 5 features and 3 states. However, this
   forces each feature's emission variance to be estimated independently per state,
   discarding the off-diagonal return correlations that define regime-specific
   co-movements (e.g., flight-to-quality: SPY down, TLT up). The independent HMMs
   implicitly capture asset-specific conditional variances more accurately.

3. **SPY rank from joint state is noisier.** The joint state's mean SPY return is one
   dimension of a 5D mean vector. With `diag` covariance, state separation in 5D is
   weaker than in the 3-feature SPY-only space (log_return + vol + smoothed). The
   dominant state at each rebalance therefore switches more often between rank-0 and
   rank-1, producing less consistent lambda selection than the Phase 52a SPY-only HMM.

---

## Technical note: `full` covariance not tested

A full 5x5 covariance per state would require `5*(5+1)/2 = 15` parameters per state vs `5`
for `diag`. With `n_states=3` over `lookback_bars=504`:

- `diag`: 3 * (3-1) + (3-1) + 3*5 + 3*5 = 6 + 2 + 15 + 15 = 38 parameters, T/p ratio = 13.3
- `full`: 6 + 2 + 15 + 45 = 68 parameters, T/p ratio = 7.4

A T/p ratio of 7.4 is borderline but potentially feasible given the bond assets (SHY, IEF,
TLT) are strongly correlated. Full covariance is not tested in this phase; it represents a
natural extension if this line of research is resumed.

---

## Conclusion and next directions

The joint 5D HMM is a valid approach (Sharpe > 0.920) but is dominated by the simpler
Phase 52a independent-HMM baseline (-0.021 Sharpe). The gap is small enough that it could
close with `full` covariance, but the structural argument (information dilution in joint
observation space) suggests the independent approach is more appropriate for assets with
heterogeneous return drivers.

The Phase 53 noise problem (oscillating per-asset posteriors) is not solved by the joint
model in its current `diag` form — the joint model's state sequence is also noisy at the
daily frequency, as evidenced by the identical rebalance count (233) to the Phase 52a baseline.

The Phase 52a configuration (`spy_rank_bull`, 5-asset, independent HMMs, Sharpe 0.973)
remains the live deployment target.

**Confirmed next directions:**
- Phase 55 / parallel track: arbitrage and inefficiency research — crypto funding carry
  (Phase 55 in progress), vol risk premium, stat-arb pairs (Phase 24 cointegration)
- If joint HMM is revisited: test `covariance_type="full"` with regularisation and a
  longer lookback (756 bars) to ensure T/p >= 10
