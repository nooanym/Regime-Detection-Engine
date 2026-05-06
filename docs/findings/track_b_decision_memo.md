# Track B Decision Memo — Vol-Target Overlay

**Date:** 2026-05-05  
**Asset:** BTC-USD, 730d hourly (≈ 17 520 bars)  
**Branch:** `phase37/purged-cv`  
**Status:** FAIL

---

## What was tried

A volatility-targeting exposure overlay on a passive BTC long position, driven by the
regime model's forward-filtered posteriors (`OnlineDecoder.batch_filter`).

**Design rationale:** Phase 37.1-37.5 purged-CV showed mean Sharpe of 0.608 and the
shuffle / random-baseline skeptic tests passed cleanly (margins > 1.6), confirming the
HMM is capturing real temporal structure. The problem was execution cost: the binary
long/flat directional signal required too many trades and had a cost break-even of only
1 bps. Rather than a hard signal, an exposure overlay was hypothesised to produce fewer
trades while harvesting the same edge.

The 1.2 half-dataset stability diagnostic confirmed regime structure is stable
(inter-half ARI = 0.742), ruling out non-stationarity as the cause of failure and
making Track A (NHHMM) unnecessary.

**Implementation:** `src/rde/research/strategies/vol_target_overlay.py`

```
OverlayConfig:
  high_conf_threshold = 0.70   # above this → exposure = 1.0
  low_conf_threshold  = 0.50   # below this → exposure = min_exposure (0.30)
  min_exposure        = 0.30
  rebalance_bars      = 24     # 24-bar window average, ~daily
  transaction_cost    = 0.0001 # 1 bp one-way
```

Position logic: every 24 bars, compute the average posterior vector over the look-back
window, take its maximum component (`max_post`), linearly interpolate exposure between
`(low_conf_threshold, min_exposure)` and `(high_conf_threshold, 1.0)`.

A parameter sweep was run over `rebalance_bars ∈ {12, 24, 48}` and
`high_conf_threshold ∈ {0.65, 0.70, 0.75}`.

---

## Metrics obtained

| Metric | Default config | Best config | Pass threshold |
|--------|---------------|-------------|---------------|
| Overlay Sharpe | 0.118 | ~0.12 | — |
| Passive Sharpe | 0.273 | 0.273 | — |
| **Sharpe improvement** | **-0.155** | **-0.136** | **≥ 0.20** |
| **Trades / year** | **293.6** | **311.8** | **< 20** |
| **Cost break-even** | **4.5 bps** | **~5.0 bps** | **≥ 5 bps** |
| Period robustness ARI | 0.505 | — | > 0.30 |
| Random baseline margin | +1.687 | — | > 0 (PASS) |
| Shuffle test margin | +1.665 | — | > 0 (PASS) |

---

## Root cause of failure

**The turnover problem:** With `rebalance_bars=24`, the overlay evaluates 730 times per
year. 40 % of those evaluations result in a position change — giving 293 trades/year —
because the `max_post` of the **averaged** 24-bar posterior window oscillates across the
thresholds.

**Why averaging collapses confidence:** At the bar level, mean max posterior is 0.945 —
one state dominates almost always. But with 8 states, *which* state dominates rotates
within a 24-bar window. When state 0 dominates for 12 bars and state 1 for the next 12,
the averaged posterior vector becomes approximately uniform (0.125 per state), and the
maximum component drops well below the 0.50 low-confidence threshold. This triggers
exposure scaling, and when the next window shifts back, a reverse rebalance occurs.

This is not a regime change — it is posterior label permutation within a window. An
8-state model without label continuity guarantees will exhibit this at any window length
shorter than a typical dwell time (which for BTC at n=8 is 15–40 bars per state).

**Why the overlay also hurts Sharpe:** By scaling exposure down when the window
contains mixed-state bars, the overlay systematically reduces position during normal
high-confidence periods and incurs transaction costs on every oscillation. The net effect
is negative alpha (-0.155 Sharpe improvement) rather than the intended risk-adjusted
improvement.

**Parameter sweep finding:** Increasing `rebalance_bars` to 48 or `high_conf_threshold`
to 0.75 does not fix the problem. The window averaging issue is structural, not
tuneable with these parameters alone.

---

## VERDICT

**FAIL.** All three critical thresholds are unmet. Track B does not provide a deployable
edge on Trading212-class venues (≥ 5 bps round-trip cost).

Proceed to **Track C — negative result writeup and archive.**
