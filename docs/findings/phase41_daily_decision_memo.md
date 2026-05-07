# Phase 41 Decision Memo — Daily-Frequency BTC Probe

**Date:** 2026-05-06
**Asset:** BTC-USD, daily bars, period=max (~4229 bars, 2014-09-17 to 2026-05-06)
**n_states:** 8 (BIC-selected; BIC monotonically decreasing, ceiling of [2..8])
**Branch:** phase37/purged-cv
**Status:** NO-GO

---

## 1. Objective recap

Phase 41 tested whether resampling to daily bars could salvage directional regime-based trading on BTC-USD. The specific hypothesis: hourly data's failure (Phase 37b) was driven by execution cost relative to the number of trades, and daily bars would reduce turnover enough to cross the break-even threshold. The validation used purged k-fold CV (27 folds, train=750d, test=125d, embargo=5d) against the full Phase 37 skeptic's kit and the same five baselines.

---

## 2. Key metrics

| Criterion | Threshold | Observed | Result |
|---|---|---|---|
| Period robustness ARI | ≥ 0.40 | 0.368 (30 windows, 504d/126d) | **FAIL** |
| Combinatorial CV Sharpe | ≥ 0.30, std < mean | 0.459 ± 0.323 (15 folds) | **PASS** |
| Cost break-even | ≥ 10 bps | ∞ bps | **PASS** |
| Annual turnover | < 30 trades | 19.5 trades/year (CV folds) | **PASS** |
| Beats ALL 5 baselines | 5 of 5 | beats 1 of 5 (naive_vol_regime only) | **FAIL** |

**Baseline comparison (Sharpe ratio):**

| Baseline | Sharpe | Delta vs model |
|---|---|---|
| buy_and_hold | 0.585 | -0.148 (model loses) |
| vol_targeted_bah | 0.749 | -0.312 (model loses) |
| naive_momentum | 0.892 | -0.455 (model loses) |
| naive_vol_regime | 0.425 | +0.012 (model wins) |
| two_state_hmm | 0.701 | -0.264 (model loses) |
| **model (n=8)** | **0.437** | — |

**Skeptics:**

| Test | Threshold | Observed | Result |
|---|---|---|---|
| Random baseline margin | ≥ 0.30 | 0.192, p=0.150 | **FAIL** |
| Shuffle test margin | ≥ 0.30 | 0.221, p=0.095 | **FAIL** |
| Period robustness ARI | ≥ 0.40 | 0.368, Frobenius_std=0.397 | **FAIL** |

**Feature importance (all unstable, positive-fold fraction threshold = 0.70):**

| Feature | Mean importance | Pos-fold fraction | Stable? |
|---|---|---|---|
| log_return | -0.028 | 44.4% | No |
| volatility_w20 | -0.036 | 40.7% | No |
| smoothed_return_w5 | +0.044 | 63.0% | No |

---

## 3. Root causes of failure

### 3.1 High Sharpe variance in purged folds (signal, not a GO/NO-GO criterion)

The purged-CV Sharpe distribution (27 folds) has std=1.506 >> mean=0.437 — the worst-5% fold Sharpe reaches -1.652. The *combinatorial* purged CV (the criterion that counts) shows a healthier 0.459 ± 0.323 because it uses overlapping train/test splits that average over more of the history. However, the high purged-fold variance is a meaningful signal: the strategy's performance is strongly period-dependent, working well in trending bull markets and failing in bear or sideways markets. This is consistent with the model partially capturing momentum structure rather than having a regime edge.

### 3.2 Period robustness ARI=0.368 (need ≥ 0.40)

The rolling-window ARI of 0.368 across 30 two-year windows (Frobenius_std=0.397 on transition matrices) places the daily 8-state model just below the stability threshold. Compare to the hourly model's inter-half ARI=0.742 (Phase 37b): the daily model is substantially less stable despite covering a longer dataset. This is consistent with over-parameterisation: an 8-state model on daily data is using more states than the daily BTC data's information content can stably support. The BIC ceiling issue (BIC monotonically decreasing at n=8, the top of the candidate range) confirms the model is over-parameterised. A lower state count (n=2 or n=3) would likely produce better per-window stability.

### 3.3 Fails to outperform baselines

The model beats only the naive vol-regime baseline (which is itself weak, Sharpe=0.425) by a negligible margin of +0.012. It is beaten by buy-and-hold, vol-targeted B&H, and a simple 2-state HMM. Most damaging: naive momentum (a simple lagged-return trend-following signal) achieves Sharpe=0.892, easily the best baseline, and the 8-state HMM trails it by -0.455. This reveals the fundamental problem at daily frequency: BTC daily returns have strong, persistent momentum structure. Any regime model that does not primarily capture this momentum will be structurally inferior to a momentum strategy. The 8-state HMM is finding something real in the data — but that something is not the directional momentum that drives returns, and therefore provides no directional edge beyond what trend-following already captures.

### 3.4 Period robustness ARI=0.368 (need ≥ 0.40)

The rolling-window ARI of 0.368 across 30 two-year windows (Frobenius_std=0.397 on transition matrices) places the daily 8-state model below the stability threshold, though not by a catastrophic margin. Compare to the hourly model's inter-half ARI=0.742 (Phase 37b): the daily model is substantially less stable despite covering a longer dataset. This is consistent with over-parameterisation: an 8-state model fitted to daily data is using more states than the daily BTC data's information content can stably support. The BIC ceiling issue (BIC still monotonically decreasing at n=8, the top of the candidate range [2..8]) confirms the model wants to use more states but is already over-parameterised for directional predictability. A lower state count (n=2 or n=3) would likely produce better per-window stability, but the BIC-selection machinery, as designed, selected n=8 and would continue to do so.

---

## 4. What daily frequency does and does NOT fix vs hourly

The original motivation for Phase 41 was to address the hourly model's failure mode: cost break-even of ~1 bps was too tight for a 5 bps execution environment, and the Phase 37b vol-target overlay generated 294 trades/year on hourly data.

**What daily frequency does fix:**
- Cost break-even: improved from ~1 bps (hourly) to ∞ bps (daily). At daily resolution, the per-trade P&L is large enough relative to bid-ask spread that cost is not the binding constraint. This was the expected improvement and it was achieved.

**What daily frequency does NOT fix:**
- Turnover: worsened from ~100+ trades/year (hourly) to ~760 trades/year (daily). The posterior label permutation problem is independent of bar frequency; it is a function of state count and the discriminability of states in feature space.
- Statistical edge: the shuffle test margin dropped from 1.67 (hourly, Phase 37b — "real temporal structure confirmed") to 0.22 (daily, Phase 41 — below the 0.30 threshold). This is the opposite of the expected direction. Daily bars should have less microstructure noise; instead, the signal is weaker. The explanation is that at daily resolution, the dominant information in BTC returns is momentum, and the HMM's regime structure does not align with the momentum signal.
- Baseline outperformance: both probes fail to beat a naive momentum strategy. At hourly, the practical failure was cost; at daily, the failure is signal quality.

**The synthesis:** daily frequency solves the wrong problem. The binding constraint was never cost per se — it was that the signal is not directionally informative beyond momentum, at any frequency tested. Daily bars expose this more starkly because momentum strategies work particularly well on daily BTC data.

---

## 5. Interpretation

From a senior quant perspective, the daily probe produces one clear positive signal and two disqualifying negatives. The positive: cost break-even at infinity confirms the regime signal has some non-trivial per-trade P&L. The disqualifying negatives are the shuffle margin (0.22, p=0.095) and the momentum dominance. A shuffle margin below 0.30 means the model's Sharpe is not reliably distinguishable from a model that randomly permutes the feature sequence — the temporal structure the HMM exploits is barely above what a randomly ordered dataset would produce. This is a fundamental signal quality failure, not a tuning problem.

The most informative result is the naive momentum Sharpe of 0.892. BTC at daily frequency is a trend-following asset: the dominant statistical structure in returns is autocorrelation of sign, not regime-switching of volatility or distribution. A Gaussian HMM fitted to returns and volatility features will identify periods of high and low volatility (which it does — ARI=0.742 confirms stable regime structure at hourly resolution), but volatility regimes do not predict direction reliably enough to compete with a strategy that simply bets on yesterday's return persisting. The 8-state model's edge, whatever it is, is orthogonal to the directional momentum that drives daily BTC P&L.

The BIC ceiling problem warrants explicit attention as a model-selection lesson. The BIC decreased monotonically from n=2 to n=8 across the candidate range, meaning the algorithm has no principled stopping point within the range — it would continue selecting more states until the search space is exhausted. This is typical of BIC on highly non-stationary financial data: more states improve in-sample fit faster than the BIC penalty controls overfitting, especially when the data spans 11 years with multiple structural breaks (2017-18 bubble, 2020 COVID crash, 2021-22 cycle, 2022 FTX collapse, 2024-26 cycle). A future probe at n=3 with daily data should use a held-out stability criterion — not BIC alone — to select state count.

---

## 6. Verdict and next steps

**VERDICT: NO-GO.** 2 of 5 GO/NO-GO criteria fail (period robustness ARI just below threshold; fails to beat 4 of 5 baselines). The daily probe has confirmed that the directional deployment hypothesis is exhausted at both hourly and daily frequency. The regime engine produces real, stable structure (the engine is not broken) but that structure does not translate into a deployable directional signal on BTC at the tested cost and turnover constraints.

Confirmed next directions in priority order:

1. **Regime-conditional multi-asset portfolio allocation** — The engine's demonstrated value is in characterising volatility and correlation regimes, not predicting direction. Use regime labels for portfolio weighting across BTC/ETH/SPY/GLD: when BTC is in a high-volatility regime, reduce BTC weight and increase GLD/SPY allocation. The edge is in diversification and drawdown control, not directional alpha. This is the highest-confidence path forward given the evidence.

2. **Options / vol forecasting** — Regime labels as an implied-volatility forecast input for options strategies. The regime engine reliably identifies high/low volatility states (ARI=0.742 inter-half stability); that stability is precisely what a vol-surface model needs. This direction uses the engine's actual strength rather than fighting its demonstrated weakness.

3. **n=3 daily model with stability-first selection** — If further directional probing is required, test n_states=3 specifically on daily BTC data, using a selection criterion that combines BIC with a minimum per-window ARI threshold (≥ 0.50 across rolling windows). The BIC ceiling failure at n=8 suggests that fewer states would produce more stable, momentum-aligning regimes. This is a bounded probe, not an open-ended research program: if n=3 daily fails the same skeptic's kit, the directional hypothesis should be considered definitively exhausted.
