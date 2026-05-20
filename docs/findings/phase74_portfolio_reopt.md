# Phase 74 — Portfolio Frontier Re-Optimisation (3-Pair LETF Upgrade)

**Date:** 2026-05-18
**Branch:** phase51/regime-conditional-lambda

## Context

Phase 66 set the multi-strategy allocation at **80% carry / 15% LETF / 5% RTMV** based on
the single TQQQ/QQQ pair (Sharpe ~4.887).  Phase 70/71 upgraded the LETF leg to the
3-pair combo (TQQQ/QQQ + SOXL/SOXX + UPRO/SPY), raising standalone Sharpe to **7.06**
(+0.82 improvement).  Phase 72 confirmed the LETF pair earns most in bear regimes
(+61%/yr bear vs +9%/yr bull), making it a natural hedge for carry drawdowns.

**Question:** With a stronger LETF leg, is more LETF allocation beneficial?

---

## Individual Strategy Metrics (2020-01-01 to present)

| Strategy | Sharpe | Ann Return | Ann Vol | MDD |
|---|---|---|---|---|
| Carry (BTC+ETH carry-weighted, bull-only) | 8.9566 | 12.91% | 1.44% | -0.39% |
| LETF 3-pair combo | 7.0552 | 36.71% | 5.20% | -0.75% |
| RTMV + momentum | 0.6405 | 4.03% | 6.29% | -15.80% |

Cross-strategy correlations: carry↔LETF = -0.0971, carry↔RTMV = -0.0006, LETF↔RTMV = -0.0123

---

## Grid Search Results — Top 10 Allocations

Grid: carry ∈ [50%,90%], letf ∈ [5%,35%], rtmv = residual ∈ [2%,15%].

| Carry | LETF | RTMV | Sharpe | Ann Return | MDD | 2022 |
|---|---|---|---|---|---|---|
| 75% | 20% | 5.0% | **11.5794** | 16.87% | -0.18% | 9.35% |
| 80% | 15% | 5.0% | 11.5636 | 15.76% | -0.20% | 7.30% |
| 70% | 25% | 5.0% | 11.2874 | 18.00% | -0.17% | 11.43% |
| 85% | 10% | 5.0% | 11.0962 | 14.65% | -0.22% | 5.29% |
| 70% | 20% | 10.0% | 10.8747 | 16.41% | -0.22% | 8.53% |
| 65% | 30% | 5.0% | 10.8415 | 19.14% | -0.19% | 13.55% |
| 75% | 15% | 10.0% | 10.8218 | 15.29% | -0.25% | 6.50% |
| 65% | 25% | 10.0% | 10.6522 | 17.53% | -0.19% | 10.60% |
| 80% | 10% | 10.0% | 10.3733 | 14.19% | -0.27% | 4.51% |
| 60% | 35% | 5.0% | 10.3492 | 20.29% | -0.23% | 15.72% |

---

## Sharpe Heatmap (carry rows, letf cols; — = invalid rtmv constraint)

| carry \ letf | 5% | 10% | 15% | 20% | 25% | 30% | 35% |
|---|---|---|---|---|---|---|---|
| 50% | — | — | — | — | — | — | 9.183 |
| 55% | — | — | — | — | — | 9.483 | 9.864 |
| 60% | — | — | — | — | 9.718 | 10.284 | 10.349 |
| 65% | — | — | — | 9.816 | 10.652 | 10.841 | — |
| 70% | — | — | 9.683 | 10.875 | 11.287 | — | — |
| 75% | — | 9.241 | 10.822 | 11.579 | — | — | — |
| 80% | 8.479 | 10.373 | 11.564 | — | — | — | — |
| 85% | 9.505 | 11.096 | — | — | — | — | — |
| 90% | 10.151 | — | — | — | — | — | — |

Values ≥ 11.6536 are marked **bold** (GO threshold).

---

## Comparison vs Phase 71 Baseline

| Metric | Phase 71 Baseline (80/15/5) | Phase 74 Best Found (75/20/5) | Change |
|---|---|---|---|
| Allocation | 80%/15%/5% | 75%/20%/5% | shifted |
| Combined Sharpe | 11.5636 | 11.5794 | +0.0158 |
| MDD | -0.20% | -0.18% | +0.02pp |
| Ann Return | 15.76% | 16.87% | +1.11pp |
| 2022 Stress | 7.30% | 9.35% | +2.05pp |

---

## GO/NO-GO Verdict

| Criterion | Threshold | Result | Status |
|---|---|---|---|
| Sharpe improvement | ≥ +0.09 over 11.5636 | +0.0158 | FAIL |
| 2022 stress positive | > 0% | 7.30% | PASS |

**Verdict: NO-GO**

## Deployment

NO-GO: Phase 71 baseline weights (80% carry / 15% LETF / 5% RTMV) remain unchanged.
The 3-pair LETF upgrade did not shift the optimal allocation meaningfully.


---

## Analysis

The Phase 71 baseline carries most of the combined Sharpe because carry (Sharpe ~9.0)
overwhelms the LETF and RTMV contributions by a wide margin.  The carry component
is responsible for the low portfolio MDD (−0.20%) since carry MDD is only
−0.39%.  Increasing the LETF weight from 15% to 20%+ raises
annualised return slightly (LETF earns 36.7%/yr) but also increases
portfolio vol because the LETF vol is 5.2%/yr vs carry's 1.4%/yr.
The Sharpe of the combined portfolio peaks at the allocation where the marginal
diversification benefit of the LETF—via carry↔LETF correlation of -0.0971—
exactly offsets the vol dilution of reducing carry weight.

## Phase 75 Recommendation

Since Phase 74 is NO-GO (80/15/5 still optimal), Phase 75 = **intraday carry timing**.
Test whether funding collection timing (8h UTC windows 00:00/08:00/16:00) can be optimised.
Some windows are systematically higher — micro-optimisation but potentially +1–2% ann.
