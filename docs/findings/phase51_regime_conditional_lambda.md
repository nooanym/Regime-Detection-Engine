# Phase 51 — Regime-Conditional λ

**Date:** 2026-05-13  
**Universe:** SPY / GLD / TLT / IEF (daily, 2004-12-17 → 2026-05-13, 5,384 bars)  
**n_states=3, n_restarts=3, lookback=504, halt=25%**  
**Verdict: NO-GO**

---

## Hypothesis

Fixed λ=0.05 (Phase 50 baseline) applies the same regime tilt regardless of which state
is dominant. The hypothesis: applying a higher λ when the portfolio is in a "bull" state
(amplify the directional signal) and lower λ in a "bear" state (conserve toward min-var)
should recover the residual gap between the live backtest (Sharpe=0.891) and the research
target (0.934).

Two schedules tested:
- **rank_bull**: `[λ_bear=0.02, λ_neutral=0.05, λ_bull=0.10]` — high tilt when equity is trending
- **rank_bear**: `[λ_bear=0.10, λ_neutral=0.05, λ_bull=0.02]` — high tilt (defensive) when equity is falling

---

## Results

| Variant    | Sharpe | Calmar | MDD    | Ann Return | Rebalances |
|-----------|--------|--------|--------|------------|------------|
| fixed_l05  | 0.8910 | 0.3025 | −21.5% | 6.5%       | 233        |
| fixed_l10  | 0.8941 | 0.3118 | −21.2% | 6.6%       | 233        |
| rank_bear  | 0.8935 | 0.3054 | −21.4% | 6.5%       | 233        |
| **rank_bull** | **0.8884** | 0.3008 | −21.5% | 6.5% | 233 |

---

## Key Findings

1. **rank_bull HURTS** (−0.003 Sharpe vs baseline). Amplifying tilt in "bull" states is counterproductive. The hypothesis was wrong.

2. **rank_bear is marginally better** (+0.003 Sharpe vs baseline) but well below the +0.010 GO threshold.

3. **fixed_l10 is the best** (+0.003 Sharpe, +0.009 Calmar, MDD −0.3pp). Simply raising λ from 0.05 → 0.10 outperforms any regime-conditional schedule.

4. **No variant crosses the +0.010 Sharpe threshold required for GO.**

---

## Root Cause: Incoherent Portfolio-Level Rank Signal

The regime-conditional λ was implemented by:
1. Fitting per-asset independent HMMs (SPY, GLD, TLT, IEF each get their own n=3 model)
2. For each asset, finding the dominant state's return rank (rank 0 = lowest return state)
3. Averaging ranks across assets → portfolio-level rank → select λ

**The problem:** per-asset state ranks are not commensurate.

- SPY rank-2 (high-return state) ≠ TLT rank-2 (high-return state for bonds = low-vol crash protection)
- In a bear equity environment, SPY is in rank-0 but TLT is often in rank-2 (flight to quality)
- The mean rank across SPY/GLD/TLT/IEF cancels out: equity bear + bond bull → mean rank ≈ 1 (neutral)
- The resulting λ stays near λ_neutral=0.05 most of the time, not capturing the intended signal

Why does rank_bull hurt? When the mean rank tips to "bull" (rank ≈ 2), the portfolio is often in
a late-cycle state where all four assets have positive expected returns — the extra tilt amplifies
whichever asset happens to have the highest posterior × mean return, which is often TLT/IEF in the
low-vol state rather than SPY.

---

## What Would Work

A coherent portfolio-level regime signal requires either:

1. **Joint HMM on portfolio returns** — fit a single HMM on the 4-asset return vector jointly.
   One dominant state directly characterizes the portfolio regime (no averaging needed).
   Trade-off: requires a joint covariance HMM (n_features=4), harder to fit stably with n=3.

2. **Equity-proxy state only** — use only SPY's dominant rank to set λ for the whole portfolio.
   Rationale: SPY is the risk-on/off signal; other assets' state ranks are confounders.
   This is a targeted fix that avoids the averaging problem.

3. **Macro-state indicator** — use an external regime indicator (e.g., VIX level, yield curve slope)
   rather than per-asset HMM state ranks. More robust to the multi-asset state incoherence.

---

## Recommendation

**Do not deploy regime-conditional λ with the current per-asset averaging design.**
Keep `lambda_by_state_rank = []` (disabled) in production.

Consider running a follow-up probe with **equity-proxy state only** (SPY rank → λ) as it avoids
the averaging problem and is a one-line config change. This is queued as Phase 51b.

---

## Files Changed

- `src/rde/analysis/multi_asset_allocation.py` — `lambda_by_state_rank` param in `compute_rtmv_weights_now`
- `src/rde/trading/rtmv_rebalancer.py` — `lambda_by_state_rank` in config + wired through step()
- `scripts/compare_lambda_strategies.py` — comparison runner
- `results/phase51/lambda_comparison.csv` — full results table
- `tests/test_rtmv_rebalancer.py` — `TestRegimeConditionalLambda` (7 tests)

## Reproduce

```bash
uv run python scripts/compare_lambda_strategies.py
```
