# Negative Result — BTC Regime Detection as a Trading Signal

**Date:** 2026-05-05  
**Asset:** BTC-USD, 730d hourly  
**Research phases:** 37.1–37.5 (edge validation) + 37b (stability diagnostic, vol-target overlay)  
**Final tag:** `v2.1-final-research`

---

## Summary

The BTC-USD Hidden Markov Model regime engine reliably identifies stable, economically
interpretable market regimes. The shuffle and random-baseline tests confirm the model
captures real temporal structure that random labelling cannot replicate. Despite this,
no viable trading strategy could be constructed that clears the minimum edge threshold
required for deployment on a venue with ≥ 5 bps round-trip execution costs. This
document records what the model can do, what it cannot, and why.

---

## What the model CAN do

**1. Stable regime identification.**  
The 1.2 half-dataset stability diagnostic (inter-half ARI = 0.742) confirms that BTC
exhibits persistent, detectable regime structure over the 730-day horizon. Fitting
independent HMMs on the first and second halves of the dataset and comparing their
decodings of the same observations yields strong agreement. The 30-day rolling inter-half
ARI ranges 0.56–0.91, well above the 0.4 noise floor.

**2. Economically interpretable regimes.**  
At n=8 states, the HMM identifies distinct regimes differentiated by mean return,
realised volatility, and autocorrelation structure. Regime-conditional risk metrics
(VaR, CVaR, Sortino, Calmar) show significant dispersion across states, confirming they
are not artefacts of the training procedure.

**3. Regime-conditional risk management.**  
Regime labels provide a meaningful signal for dynamic position sizing, volatility
targeting, and drawdown control. A portfolio risk engine conditioned on regime
probabilities can reduce tail exposure during high-vol states without requiring any
directional edge.

**4. Cross-asset concordance analysis.**  
The regime engine extends cleanly to equities (SPY), commodities (GLD), and other crypto
assets (ETH, SOL). Cross-asset concordance metrics identify leading/lagging regime
transitions and macro alignment, which is useful for portfolio construction even without
a trading signal.

**5. Real temporal structure (not data-mining artefact).**  
Random-baseline margin: +1.687 (p = 0.000). Shuffle-test margin: +1.665 (p = 0.000).
The model's CV Sharpe of 0.608 (mean over purged folds) is not achievable with random
or shuffled regime assignments. The information content is real.

---

## What the model CANNOT do

**1. Generate a deployable directional trading signal after execution costs.**  
The binary long/flat directional strategy (Phase 37.1–37.5) has a cost break-even of
1.0 bps — far below the 5 bps floor. At 1 bp per side on Trading212 or comparable
venues, the strategy loses money. This is not because the signal is weak in gross terms
(Sharpe 0.608 before costs) but because the required trade frequency exceeds 100
trades/year on hourly data, and each trade erodes a disproportionate fraction of the
per-trade alpha.

**2. Generate a deployable exposure overlay after execution costs.**  
The vol-target overlay (Phase 37b Track B) was designed to reduce turnover by scaling
exposure continuously rather than switching binary positions. It fails because the
8-state posterior label permutation within the 24-bar averaging window causes exposure
oscillation at the same frequency as the original binary signal. Measured turnover:
294 trades/year; Sharpe improvement vs passive: -0.155. See
[track_b_decision_memo.md](track_b_decision_memo.md) for the detailed root cause.

**3. Produce stable label alignment across time windows at n=8 states.**  
The period robustness metric (exposure series ARI across rolling windows: 0.505) is
borderline and reflects a real limitation: with 8 states, the Viterbi path is sensitive
to initialisation and window boundaries. State labels are not canonical identifiers —
they are local optima. This makes regime-conditional strategy rules brittle across
re-fitting cycles, which is a deployment requirement for any live system.

---

## Why

**Execution cost threshold.** The minimum viable edge for a daily-rebalanced strategy
on BTC at hourly data is approximately 5 bps per round trip. Achieving this requires
either: (a) fewer than ~20 trades/year with Sharpe ≥ 0.3 above a passive benchmark, or
(b) gross alpha significantly larger than what an 8-state HMM on log returns and rolling
vol produces at hourly resolution.

**Turnover mechanics.** The HMM on BTC hourly data transitions frequently between states
(mean dwell time 15–40 bars at n=8). Any strategy that reacts to individual state
changes will trade > 100 times/year. The only way to reduce turnover is to either: (i)
use much lower temporal resolution (daily bars, at the cost of losing the model's
statistical power), or (ii) add strong hysteresis with a minimum position change
threshold — but this requires the gross per-trade alpha to be materially larger to
absorb the hysteresis dead band.

**Insufficient edge margin at hourly resolution.** The model's conditional Sharpe
advantage over a passive position is approximately 0.35 on an annualised basis (gross,
before costs). At the trade frequencies implied by the model's state-switching rate, this
margin is entirely consumed by execution costs at 5 bps/side.

---

## What was NOT tried (scope boundaries)

The following approaches were considered but fall outside the Phase 37b scope and would
require new research phases to evaluate properly:

- **Daily-bar resampling** with a 2–3 state model for reduced frequency.
- **Regime-conditional options strategies** (using regime labels as a volatility
  forecast input rather than a directional signal).
- **Regime-conditional portfolio allocation** across multiple assets (where the edge
  is in diversification rather than directional BTC exposure).
- **Alternative venues with lower execution costs** (DeFi AMMs, perps with rebates).

The first three represent legitimate alternative uses of the regime engine. They are
appropriate topics for a Phase 38 research scoping, if undertaken.

---

## Archival

```
Branch:   phase37/purged-cv
Tag:      v2.1-final-research
Key files:
  results/BTC-USD/honest_tearsheet.md
  results/BTC-USD/skeptics_report.md
  docs/findings/track_b_decision_memo.md
  docs/findings/negative_result_writeup.md   ← this file
```

The Notion workspace holds the full activity diary (Phases 31–37b) and decision log.

---

*This is a true negative result. The model is not broken — it reliably finds regimes —
but regimes alone do not constitute a trading edge on a cost-constrained venue.*
