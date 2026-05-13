# Phase 50b — KL-Divergence Posterior-Triggered Rebalancing

**Date:** 2026-05-13
**Universe:** SPY / GLD / TLT / IEF (daily, 2004-12-17 → 2026-05-11, 5,382 bars)
**n_states=3, n_restarts=2, lookback=504, halt disabled for clean comparison**
**Verdict: NO-GO**

---

## Hypothesis

Monthly calendar rebalancing misses regime transitions mid-month and rebalances pointlessly
when regimes are stable. A posterior KL-divergence trigger — rebalance when the current
posterior shifts far enough from the last-rebalance posterior — should reduce unnecessary
turnover while capturing transitions faster.

---

## What was built

`RTMVRebalancerConfig.rebalance_kl_threshold: float = 0.0` (disabled by default).

When enabled, at each bar the rebalancer computes `compute_rtmv_weights_now(...,
return_posteriors=True)` and measures mean KL(current || stored) across assets. If
`mean_kl > threshold` AND the calendar has not fired, an early rebalance is triggered
and `posterior_at_last_rebalance` is updated.

New state field: `RTMVRebalancerState.posterior_at_last_rebalance: dict[str, np.ndarray]`.
New helper: `RTMVRebalancer._mean_posterior_kl()`.

---

## Results

| Config                  | Sharpe | MDD    | Rebalances | KL triggers |
|------------------------|--------|--------|------------|-------------|
| calendar 21-bar         | 0.901  | −21.5% | 233        | 0           |
| kl=0.30, cal=42-bar    | 0.884  | −21.0% | 4,688      | 4,687       |
| kl=0.15, cal=63-bar    | 0.884  | −21.0% | 4,698      | 4,697       |

---

## Root cause: structural mismatch

**The KL trigger fires on nearly every bar** (4,687 out of 5,382 ≈ 87% daily rate).
At `kl=0.30` with a 42-bar calendar fallback, almost every step is KL-triggered — the
strategy effectively rebalances daily instead of monthly.

This happens because of an architectural conflict: `compute_rtmv_weights_now()` **refits
the HMM from scratch** on each call, using a rolling window of the last 504 bars. Each
new bar shifts the window by one observation, which changes the fitted model and its
posteriors. The KL between consecutive refit posteriors is dominated by:

1. **Window drift** — the oldest bar is dropped and a new one added on every step; this
   alone produces ≥0.2 KL in natural log units across assets.
2. **HMM noise** — multi-restart Baum-Welch produces slightly different converged models
   each time; the state alignment is approximate.
3. **Genuine regime transitions** — the actual signal; swamped by 1 and 2.

The result: a threshold low enough to catch real transitions (0.15–0.30) is also low
enough to fire on window drift almost every day. A threshold high enough to avoid window
drift (e.g. KL > 2.0) would almost never fire, defeating the purpose.

---

## Alternative direction

The correct implementation would decouple the triggering signal from the refitting cost:

- Maintain a **live posterior** via `OnlineDecoder.step()` (no refitting, just Bayesian
  filtering on the existing model) and trigger calendar rebalance + refit when the
  online posterior diverges from the stored rebalance-time posterior by > threshold.
- This separates the cheap "does the posterior suggest a regime shift?" check (online
  filter) from the expensive "refit the model and compute new weights" action.

This is a non-trivial refactor and is out of scope for Phase 50. The calendar schedule
remains the correct approach; see Phase 50c for why 21-bar calendar with halt=25% is
optimal.

---

## Recommendation

**Do not enable `rebalance_kl_threshold` in production.** Keep it at 0.0 (disabled).

The feature is architecturally correct in isolation but the trigger signal is dominated
by refitting noise. A future phase could implement an online-posterior KL monitor that
avoids per-step refitting.

---

## Files changed

- `src/rde/trading/rtmv_rebalancer.py` — `rebalance_kl_threshold`, `posterior_at_last_rebalance`,
  `_mean_posterior_kl()`, `return_posteriors` in `compute_rtmv_weights_now` call path.
- `src/rde/analysis/multi_asset_allocation.py` — `return_posteriors` kwarg in
  `compute_rtmv_weights_now()`.
- `tests/test_rtmv_rebalancer.py` — `TestKLTriggeredRebalance` (4 tests).
- `scripts/compare_kl_rebalance.py` — comparison runner.

## Reproduce

```bash
uv run python scripts/compare_kl_rebalance.py
```
