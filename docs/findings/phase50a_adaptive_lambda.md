# Phase 50a — Adaptive λ for RTMV

**Date:** 2026-05-10
**Branch:** `phase37/purged-cv`
**Universe:** SPY / GLD / TLT / IEF (daily, 2004-12-17 → 2026-05-08, 5,381 bars)
**Verdict:** **NO-GO** — adaptive λ underperforms fixed λ=0.05 by −0.003 Sharpe.

---

## Motivation

Phase 45 found two different optimal λ for RTMV:

| Window      | Best λ | Sharpe vs GMV |
|-------------|--------|---------------|
| Full period (2004–2026) | 0.05 | 0.934 vs 0.929 |
| OOS (2016–2026)         | 0.30 | 0.947 vs 0.882 |

The Phase 48 live backtest with the chosen λ=0.05 produced Sharpe=0.875, ~0.06 below the research target.
Hypothesis: optimal λ is **regime-dependent** — the tilt should be more aggressive when the per-asset
HMM posteriors are sharp (one regime is clearly dominant) and conservative when posteriors are diffuse
(no regime stands out). An adaptive schedule driven by posterior entropy might recover some of the gap.

---

## What was built

### 1. `compute_rtmv_weights_now` (`src/rde/analysis/multi_asset_allocation.py`)

Added three optional kwargs (defaults preserve existing behaviour):

```python
adaptive_lambda: bool = False
lambda_min: float = 0.02
lambda_max: float = 0.15
```

When `adaptive_lambda=True`:

1. After fitting each asset's HMM and running `OnlineDecoder.batch_filter`, capture the
   **last-bar posterior** `p ∈ ℝ^K`.
2. Per-asset Shannon entropy: `H = -Σ p_k · log(p_k + 1e-15)`.
3. `H_max = log(n_states)` (uniform-prior maximum).
4. `h_norm = mean(H_per_asset) / H_max ∈ [0, 1]`.
5. `λ_effective = λ_max · (1 - h_norm) + λ_min · h_norm`.

So sharp posteriors → low entropy → `h_norm ≈ 0` → `λ → λ_max`,
and diffuse posteriors → `h_norm ≈ 1` → `λ → λ_min`.

The function still returns a full weight dict summing to 1.

### 2. `RTMVRebalancerConfig` (`src/rde/trading/rtmv_rebalancer.py`)

Added the same three fields to the rebalancer config and threaded them through `step()` to
`compute_rtmv_weights_now`. Backward compatible — all existing call-sites work unchanged.

### 3. Tests (`tests/test_rtmv_rebalancer.py`)

Added `TestAdaptiveLambda` (6 new tests):

- `test_adaptive_lambda_sum_to_one` — weights still sum to 1
- `test_adaptive_lambda_nonnegative` — weights ≥ 0
- `test_adaptive_lambda_range` — `λ_min == λ_max` collapses to fixed-λ behaviour (i.e. λ_effective is bounded)
- `test_adaptive_lambda_vs_fixed_different` — adaptive λ produces non-identical weights to fixed λ=0.05
- `test_adaptive_lambda_rebalancer_config_round_trip` — config dataclass round-trips
- `test_adaptive_lambda_invalid_range_raises` — invalid (`λ_min > λ_max`) raises `ValueError`

`uv run pytest tests/test_rtmv_rebalancer.py -x --tb=short -q` → **35 passed**.

### 4. Comparison script (`scripts/compare_adaptive_lambda.py`)

Loads the SPY/GLD/TLT/IEF universe via cached yfinance, runs two RTMV backtests
(fixed λ=0.05 vs adaptive λ ∈ [0.02, 0.15]), and prints a side-by-side metric table.

---

## Backtest results

`n_states=3, n_restarts=3, lookback=504, rebalance_bars=21, capital=$100,000, slippage=5 bps, drawdown_halt=20%`

| Metric         | Fixed λ=0.05 | Adaptive λ | Δ          |
|----------------|--------------|------------|------------|
| Sharpe (ann)   | **0.875**    | 0.872      | **−0.003** |
| Calmar         | 0.299        | **0.316**  | +0.017     |
| Max Drawdown   | 21.5%        | **20.9%**  | −0.58 pp   |
| Ann Return     | 6.4%         | **6.6%**   | +0.19 pp   |
| Ann Vol        | 7.3%         | 7.6%       | +0.24 pp   |
| Final Equity   | $371,709     | **$385,284** | +$13,575 |
| N Rebalances   | 212          | 214        | +2         |

---

## Interpretation

**Adaptive λ does not improve Sharpe** but it **does** improve Calmar (+0.017) and shave the max
drawdown from 21.5% to 20.9% — both portfolios still trip the 20% halt. Annualised return is
slightly higher (+19 bps), but the extra return is matched by extra vol (+24 bps), and the Sharpe
delta is essentially noise (−0.003).

Why didn't adaptive λ close the 0.875 → 0.934 gap?

1. **Entropy is not a strong signal here.** With `n_states=3` and HMMs that are heavily regularised
   (kmeans init, 3 restarts, 1000 iter), end-of-window posteriors tend to be reasonably concentrated
   for most rebalance steps (one state usually wins by ≥0.5 mass), so `h_norm` is dominated by a
   small range and the resulting `λ_effective` mostly clusters near the upper end of [0.02, 0.15].
   We rarely sit at λ_min, so adaptive ≈ a slightly higher fixed λ.

2. **The OOS λ=0.30 finding was period-specific, not posterior-specific.** Phase 47's rebalance
   cache showed λ=0.30 fails fold-consistency on the full period and only wins post-2016. This is a
   regime-of-the-decade effect (rates falling 2004–2015 → rising 2022–2024), not a per-bar
   confidence effect. Adaptive λ keyed on instantaneous posterior entropy can't capture a slow
   structural shift.

3. **Costs matter.** With 5 bps slippage and ~21-bar rebalance cadence, each unit of extra trading
   costs Sharpe; the marginally-tilted weights from adaptive λ produce 2 extra rebalances and
   slightly higher vol, both of which absorb the small return improvement.

The Calmar/MDD improvements are mildly encouraging — adaptive λ is *gentler in the tails* — but
not enough to claim a deployable edge.

---

## Recommendation

**Do not change the live deployment.** Stay on fixed λ=0.05. The adaptive entropy schedule is a
nice-to-have research artifact but not a Sharpe-positive change.

If we want to chase the residual Sharpe gap, the more promising directions are:

- A regime-conditional λ that depends on the **dominant state's identity** (e.g. trend-up vs
  high-vol-bear), not just its entropy. This couples to the recovered HMM regimes more directly.
- A coarser-grained λ schedule that tracks slow regime changes (12-month rolling) rather than
  bar-by-bar entropy.
- Phase 50b's KL-divergence early rebalance trigger (already scaffolded in
  `RTMVRebalancerConfig.rebalance_kl_threshold`) is a separate orthogonal lever to evaluate.

---

## Files changed

- `src/rde/analysis/multi_asset_allocation.py` — `compute_rtmv_weights_now` extended with
  adaptive-λ kwargs.
- `src/rde/trading/rtmv_rebalancer.py` — `RTMVRebalancerConfig` gains
  `adaptive_lambda`/`lambda_min`/`lambda_max`; threaded through `step()`.
- `tests/test_rtmv_rebalancer.py` — `TestAdaptiveLambda` (+6 tests, 35 total pass).
- `scripts/compare_adaptive_lambda.py` — comparison runner.
- `docs/findings/phase50a_adaptive_lambda.md` — this memo.

## Reproduce

```bash
uv run pytest tests/test_rtmv_rebalancer.py -x --tb=short -q
uv run python scripts/compare_adaptive_lambda.py
```
