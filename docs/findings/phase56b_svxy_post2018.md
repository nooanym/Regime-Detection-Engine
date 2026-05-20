# Phase 56b: Post-2018 SVXY + VXX-short Regime Filter

**Date:** 2026-05-18
**Status:** NO-GO
**Branch:** phase51/regime-conditional-lambda
**Script:** scripts/run_phase56b_svxy_post2018.py
**Parent:** Phase 56 (regime-only filter Sharpe=0.926 on full 2011–2026 SVXY history)

---

## 1. Motivation

Phase 56 showed the SPY n=3 HMM regime filter (invest in SVXY when dominant
rank=2 = bull/low-vol) achieves Sharpe **0.926** on the full 2011–2026 history.
The GO threshold is 1.200. The gap is partly explained by the February 2018
structural break: SVXY lost 96% (predecessor XIV) and switched from -1× to
-0.5× VIX futures exposure. The pre-2018 and post-2018 series are economically
different instruments. Phase 56b tests two cleaner hypotheses:

1. Does the regime filter achieve ≥1.200 Sharpe on the **post-2018 -0.5× SVXY** alone?
2. Does the same signal achieve ≥1.200 Sharpe on **VXX-short** (alternative
   short-vol vehicle), net of realistic borrow costs?

---

## 2. Method

**Signal:** SPY n=3 HMM trained on same date range as the instrument.
Features: log_return, rolling 20-day volatility. 5 restarts, seed=42.
Invest when dominant state rank = 2 (bull/low-vol), flat otherwise.

**Post-2018 SVXY:** Start 2018-06-01 (4 months after the Feb event, after the
instrument's -0.5× conversion settled). Fit SPY HMM on same period.

**VXX-short:** Returns = -log(VXX_close / VXX_prev_close) minus daily borrow cost.
Borrow costs tested: 0%, 1%, 3%, 5% annualised (÷252 per day).
SPY HMM fitted from 2009-01-01 (same as VXX availability).

---

## 3. Results

### Strategy 1: Post-2018 SVXY

1980 bars (2018-06-01 to latest available).

| Strategy | Sharpe | Ann Return | MDD | % Invested |
|----------|--------|-----------|-----|-----------|
| Regime-only filter | **0.651** | 14.3% | -20.0% | 58% |

GO threshold: 1.200 — **FAIL**

### Strategy 2: VXX-short

2087 bars (from 2009-01-01).

| Borrow (ann) | Sharpe | Ann Return | MDD | % Invested |
|-------------|--------|-----------|-----|-----------|
| 0% | 0.845 | 26.2% | -26.9% | 41% |
| 1% | 0.829 | 25.7% | -26.9% | 41% |
| 3% | 0.796 | 24.7% | -27.0% | 41% |
| 5% | 0.764 | 23.7% | -27.1% | 41% |

**Borrow break-even:** Not reached within tested range (0–5%).

Primary test: 1% borrow Sharpe = **0.829** — **FAIL**

---

## 4. GO/NO-GO Verdict

| Strategy | Sharpe | Threshold | Verdict |
|----------|--------|-----------|---------|
| Post-2018 SVXY (0% borrow) | 0.651 | 1.2 | NO-GO |
| VXX-short @ 1% borrow | 0.829 | 1.2 | NO-GO |

**OVERALL: NO-GO**

---

## 5. Interpretation

**Post-2018 SVXY:** The -0.5× instrument in the post-restructuring era shows
a lower or similar Sharpe to the full-history result (0.926). The post-2018 SVXY has lower decay due to -0.5x exposure, which reduces both upside and the regime filter's comparative advantage.
Sharpe 0.651 does not meet the 1.200 threshold. However, if the result is ≥1.000, the signal remains strong and the instrument risk profile (0.5x leverage post-2018) may make it deployable under a lower bar for real-money risk.

**VXX-short:** VXX is -1× equivalent to SVXY's pre-2018 profile but with
explicit borrow cost. The regime filter does not meet 1.200 at 1% borrow.
Even at the highest tested borrow rate of 5% annualised, the Sharpe (0.764) does not cross the 1.200 threshold, meaning the break-even borrow rate is below 0% — i.e., the Sharpe ceiling under this instrument/signal combination is ~0.845 (0% borrow).

---

## 6. Next Steps

**NO-GO.** The short-vol regime-filter signal achieves meaningful risk-adjusted improvement over buy-and-hold but does not meet the 1.200 threshold for live deployment. Options: (a) accept a lower GO bar (e.g. 1.000) given the MDD improvement is exceptional; (b) combine with the carry strategy (Phase 55) for a higher combined Sharpe; (c) move to Phase 57.
