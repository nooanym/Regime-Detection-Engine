# Phase 43b: SPY Daily Phase 37 Validation — Findings

**Date:** 2026-05-09
**Config:** n_states=8, n_restarts=3, train_bars=1000, test_bars=126
**Data:** SPY daily 1993–2026, 58 purged CV folds
**File:** `results/SPY-daily/honest_tearsheet.md`

---

## GO/NO-GO Verdict: NO-GO

| Criterion | Threshold | Value | Result |
|-----------|-----------|-------|--------|
| Period robustness ARI | ≥ 0.40 | 0.311 | FAIL |
| Combo CV Sharpe | ≥ 0.30, std < mean | 0.386 ± 1.376 | FAIL |
| Cost break-even | ≥ 10 bps | ∞ bps | PASS |
| Annual turnover | < 30/yr | 28.4 | PASS |
| Beats all 5 baselines | 5/5 | 0/5 | FAIL |

---

## The Critical Signal: n=8 is Overfit for SPY

The two_state_hmm baseline achieves **Sharpe=1.016** vs the 8-state model's
0.386. This is the largest baseline gap in the entire research project (−0.630
margin). It tells us directly: n=8 states is too many for SPY daily. The model
is finding statistical artefacts in the training data that don't generalise.

BTC needed 8 states because it has genuinely distinct micro-regimes
(moon/crash/recovery/accumulation at different vol levels). SPY's structure
is simpler: roughly 3 states (bull market, crisis/bear, recovery). More states
fragment these into unstable sub-regimes that flip label ordering across
different training windows → low ARI.

### Comparison with BTC daily at n=8

| Metric | BTC daily n=8 | SPY daily n=8 |
|--------|---------------|---------------|
| Period robustness ARI | 0.368 | 0.311 |
| Random baseline margin | −0.008 | +0.092 |
| Shuffle test margin | +0.023 | +0.093 |
| Beats baselines | 1/5 | 0/5 |
| Two-state HMM margin | −0.059 | **−0.630** |

SPY's random/shuffle margins are actually better than BTC's (0.09 vs 0.02),
suggesting the signal IS real — the problem is purely n_states, not signal
absence.

---

## What the Cost and Turnover Numbers Mean

- Cost break-even = ∞ bps: the strategy is profitable at any tested cost level
- 28.4 trades/year: roughly one trade every 9 trading days — manageable
- Mean fold MDD = 8.4%: the strategy protects drawdowns well in most folds

These infrastructure numbers PASS. The only failures are statistical (ARI,
baseline comparisons) driven by model complexity mismatch.

---

## Root Cause: n_states Mismatch

Hypothesis: BIC-optimal n for SPY daily is 3–4 (not 8). If correct:
- Fewer states → more stable regime labels across time windows → higher ARI
- Simpler model → better generalisation → better baseline comparisons
- The 2-state baseline's Sharpe=1.016 is strong evidence that less complexity works

## Next Step: BIC Selection on SPY Daily

Running `scripts/run_bic_audit.py --config configs/spy_daily.yaml` to find
the BIC-optimal n_states for SPY daily. Expected outcome: n=3 or n=4.

Once BIC optimal n is known, re-run Phase 37 validation with that n_states.
If ARI ≥ 0.40 and random/shuffle margins > 0.30, we have a working equity signal.
