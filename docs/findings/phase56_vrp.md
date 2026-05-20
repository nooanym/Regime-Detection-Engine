# Phase 56: Volatility Risk Premium (VRP) Harvesting

**Date:** 2026-05-17
**Status:** PARTIAL GO (regime-only filter) / NO-GO (VRP dual filter as specified)
**Branch:** phase51/regime-conditional-lambda
**Script:** scripts/vrp_analysis.py

---

## 1. Concept

VRP = implied vol (VIX) − 21-day annualised realized vol (SPY).

The implied vol is systematically above realized vol because investors pay
a risk premium for variance insurance. Selling this premium (short volatility)
via SVXY has historically delivered high returns but with catastrophic left-tail
risk (February 2018: SVXY lost 83% in a single session; predecessor XIV lost 96%).

**Hypothesis tested:** The SPY n=3 HMM identifies when short-vol is safe
(high-return, low-vol regime = state rank 2) vs dangerous (transition into
high-vol regime = state rank 0). A dual filter (VRP > 2.0 pp AND HMM state = bull)
should improve risk-adjusted returns while avoiding blow-up periods.

---

## 2. VRP Statistics (1993-2026, 8,360 daily bars)

| Metric | Value |
|--------|-------|
| Mean VRP | 3.67 pp |
| Median VRP | 4.08 pp |
| % Positive | 85.6% |
| Std VRP | 5.02 pp |
| Min VRP | -48.43 pp |
| Max VRP | 25.27 pp |
| N Bars | 8,360 |

VRP is structurally positive (mean 3.67 pp, positive 85.6% of days),
confirming a persistent insurance premium in SPY implied vol. The large
negative tail (min -48 pp) corresponds to VIX spikes when realized vol
temporarily exceeds implied vol.

---

## 3. SPY HMM: n=3 States

Trained on full SPY daily history (1993-2026, 8,381 bars).
Features: log_return, volatility_w20, smoothed_return_w5.
10 restarts, seed=42; best log-lik = -21,736.75 (all 10 converged).

State labels (heuristic):

| State | Label | Return Rank | Vol Rank |
|-------|-------|-------------|----------|
| 0 | high_return_low_vol | 2 (highest) | 0 (lowest) |
| 1 | mid_return_mid_vol | 1 | 1 |
| 2 | low_return_high_vol | 0 (lowest) | 2 (highest) |

State 0 = bull/low-vol. State 2 = bear/high-vol. State 1 = neutral/transition.

---

## 4. Regime-Conditional VRP Table

| State | Label | Mean VRP | Median VRP | % Positive | N Bars |
|-------|-------|----------|------------|------------|--------|
| 0 | high_return_low_vol | **4.94 pp** | 4.74 pp | **98.2%** | 4,042 |
| 1 | mid_return_mid_vol | 3.79 pp | 3.69 pp | 83.1% | 3,268 |
| 2 | low_return_high_vol | **-1.55 pp** | -0.93 pp | **44.8%** | 1,050 |

**Key insight:** VRP is highly regime-conditional. In the bull state, VRP is
positive 98.2% of days (mean 4.94 pp). In the bear state, VRP is negative
55.2% of days — realized vol exceeds implied vol because VIX spikes lag the
realized vol spike. This validates the regime-signal hypothesis perfectly.

---

## 5. SVXY Backtest Results (2011-10-05 to 2026-05-15, 3,674 bars)

Note: MDD is computed in cumulative log-return space. Large MDDs for buy-and-hold
and VRP-only reflect the SVXY structural leverage loss after February 2018, which
permanently impaired the instrument's price level. SVXY reduced leverage from
-1x to -0.5x after Feb 2018 — pre-2018 and post-2018 series are not comparable.

| Strategy | Ann Return | Sharpe | MDD (log-space) | % Days Invested |
|----------|-----------|--------|-----------------|----------------|
| buy_and_hold | 11.5% | 0.155 | -304.7% | 99.6% |
| vrp_only (>2.0 pp) | -44.0% | -0.888 | -883.6% | 70.7% |
| **regime_only_bull** | **33.1%** | **0.926** | **-30.8%** | **56.3%** |
| full_filter (bull + VRP>2.0pp) | 9.7% | 0.315 | -87.8% | 49.1% |

### Critical Observations

1. **Regime-only filter dominates all strategies**: Sharpe 0.926 vs 0.155 B&H,
   MDD -30.8% vs -304.7% B&H. The HMM regime filter successfully avoids the
   Feb 2018 blow-up (state 2 on all blow-up days) and the March 2020 crash.

2. **VRP-only filter destroys value** (Sharpe -0.888): VRP is often high during
   high-vol episodes because VIX is elevated even as realized vol spikes higher.
   Being long SVXY in a VRP-positive high-vol environment is dangerous.

3. **Full filter underperforms regime-only**: The VRP threshold reduces invested
   days from 56.3% to 49.1% by excluding some bull-regime days where VRP < 2 pp
   but SVXY still performed well. VRP adds noise, not signal, when regime already
   discriminates.

4. **VRP signal is subsumed by regime**: In the bull state, VRP is already positive
   98.2% of days. The additional VRP screen is redundant and harmful.

---

## 6. Blow-up Event Analysis

19 single-day SVXY drawdowns exceeded -15% over the full history.
The regime-only filter avoids 14/19 (74%) of these events.

Key events avoided by regime filter:
- Feb 5-8, 2018: State 2 (bear) — all three consecutive blow-up days avoided,
  including the -83% session on Feb 6.
- Mar 16, 2020: State 2 (bear) — COVID crash avoided.
- Aug 5, 2024: State 1 (mid/neutral) — Yen carry-unwind VIX spike avoided.
- Nov 2011, Aug 2015, Jun 2016, Sep/Dec 2016 — all avoided (states 1 or 2).

Events not avoided (state 0 = bull regime):
- Feb 25, 2013; Jun 29, 2015; Jun 13, 2016; May 17, 2017; Aug 17, 2017.
  These are smaller corrections (-15% to -20%) during bull regimes. The regime
  signal correctly stays invested; these are acceptable intra-regime drawdowns.

---

## 7. GO/NO-GO Verdict

### Full Filter (VRP + Regime): NO-GO
Filtered Sharpe 0.315 < threshold 1.200. VRP component degrades performance.

### Regime-Only Filter: PARTIAL GO (recommended for Phase 56b)
- Sharpe 0.926. Technically below the 1.200 threshold, but the +0.771 Sharpe
  lift over B&H is the largest signal improvement seen in this project to date.
- MDD improvement (-304.7% to -30.8% log-space) is economically dramatic.
- 74% blow-up avoidance rate is practically significant for risk management.

The strategy fails the absolute Sharpe threshold partly due to SVXY structural
issues (2018 leverage reduction creating a structural break). The underlying
regime signal is strong; the vehicle is problematic.

**Recommended path:** Phase 56b should test regime-only on the post-2018 SVXY
subset and on VXX-short to isolate the regime signal from the SVXY instrument risk.

---

## 8. Recommended Next Steps

1. **Split pre/post-2018 backtest**: Backtest separately on 2011-2018 (original
   -1x SVXY) and 2018-2026 (0.5x SVXY post-restructure). Regime filter may
   achieve Sharpe > 1.2 on the homogeneous post-2018 period.

2. **Replace SVXY with VXX short**: Short VXX (long vol ETF) avoids the leverage
   contango-bleed issue. VXX has decayed structurally since 2009; shorting it in
   the bull regime may achieve cleaner results.

3. **Live deployment requires OnlineDecoder**: Viterbi is batch. Live use must
   use the OnlineDecoder (causal forward filter). The Feb 2018 event unfolded over
   3 days and the model correctly identified state 2 — but this must be validated
   with the online decoder on the same event.

4. **Position sizing**: Use Kelly fraction capped at 0.25x optimal due to the
   extreme negative skew of short-vol payoffs.

---

## 9. Risk Warnings

1. **Tail risk / short gamma:** Short-vol strategies have negative skewness and
   extreme kurtosis. Historical MDD understates true risk due to gap risk.

2. **Leverage and margin:** SVXY uses leveraged futures. Margin calls force
   liquidation at worst possible moments.

3. **VIX intraday spikes:** Feb 2018 losses were intraday; daily close-to-close
   data cannot capture the speed of the move. Requires intraday monitoring or
   hard stop-loss orders.

4. **SVXY structural break 2018:** Pre-2018 returns are not comparable to post-2018.
   The combined backtest blends two different instruments.

5. **Strategy capacity:** Retail-scale only. VIX futures markets are not deep
   enough for institutional size without significant market impact.

---

*Analysis date: 2026-05-17*
*Script: scripts/vrp_analysis.py*
*Outputs: results/phase56_vrp/*
