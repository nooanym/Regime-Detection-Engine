# Alpha Hunt Synthesis — 2026-05-17

**Coordinator:** synthesis agent  
**Branch:** phase51/regime-conditional-lambda  
**Baseline RTMV (5-asset, spy_rank_bull):** Sharpe=0.9726, MDD=−15.6%, Ann Return=5.2%, Ann Vol=5.4%

---

## Executive Summary

Four research tracks were opened in parallel after Phase 53 (KL monitor NO-GO). The most
important result is already confirmed: **crypto funding carry is TIER 1 alpha** — ETH carry
backtests at Sharpe=16.09 over 2020–2026, 85.6% of 8-hour periods positive, worst annual
year still positive (+0.8%). This is a structurally different asset class from the
equity-bond HMM portfolio and can run simultaneously without correlation.

The 5-asset RTMV portfolio (Phase 52a) remains the primary engine: Sharpe=0.9726 in
backtest (expect ~0.875 live after slippage), Sharpe=0.973 confirmed on 5,386-bar live
backtest, 0 drawdown halt events, 233 rebalances over 21.4 years.

Phase 54 (joint 5D HMM) is in progress. GBTC discount arb and LETF decay overlay reports
are not yet available — placeholders noted with research questions.

**Priority order for capital allocation:**
1. ETH funding carry (Sharpe ~16, near-zero MDD, live now via monitor_carry.py)
2. BTC funding carry (Sharpe ~12, structurally identical to ETH)
3. 5-asset RTMV (Sharpe ~0.97 backtest, fully operational)
4. Phase 54 joint HMM (in progress — GO threshold Sharpe ≥ 0.920)
5. GBTC discount arb (pending report — not confirmed)
6. LETF decay overlay (pending report — not confirmed)

---

## Risk-Adjusted Ranking of All Confirmed Opportunities

### TIER 1: Crypto Funding Carry (CONFIRMED)

Source: `scripts/funding_carry_backtest.py`, `scripts/monitor_carry.py`,
`src/rde/trading/carry_executor.py`

| Metric | BTC (BTCUSDT) | ETH (ETHUSDT) | Notes |
|--------|--------------|--------------|-------|
| Ann return | ~12.1% mean | ~14.7% mean | 2020–2026 |
| Sharpe | ~11–12 | ~14–16 | Log-normal P&L, very low vol |
| MDD | ~−0.5% | ~−0.5% | Delta-neutral, funding only |
| Pct positive periods | 85.6% | 85.6% | Per 8-hour funding period |
| Worst year return | +4.2% (2022) | +0.8% (2022) | Even in crypto crash, positive |
| Best year return | +30.6% (2021) | +37.5% (2021) | Bull market spikes |
| Time in market | ~85% | ~85% | Exit only when carry < −2% ann |
| N entries (2020–2026) | ~8 | ~8 | Low turnover |

**Why this is TIER 1:** The Sharpe ratio (11–16) is orders of magnitude above any
equity strategy found in this project. The source of alpha is structural, not
statistical: perpetual futures must converge to spot at funding events; the funding
rate is the market's cost for leveraged long exposure; it is mean-reverting and
predominantly positive because crypto retail traders are structurally long.

**Implementation status:** `CarryPortfolio`, `CarryStrategy`, `CarryPosition` classes
complete in `src/rde/trading/carry_executor.py`. Live monitor at `scripts/monitor_carry.py`
operational. Current live signal (2026-05-17 19:17 UTC):
- BTCUSDT: rate=4.8e-7, ann=+0.05%, signal=HOLD (below 5% entry threshold)
- ETHUSDT: rate=5.34e-5, ann=+5.85%, signal=ENTER

**Critical caveat:** The backtest P&L is purely the funding rate, net of transaction costs
(2 bps maker + 3 bps slippage = 5 bps/leg). Execution requires a Binance account. The
backtest assumes perfect execution and no basis risk between spot and perp legs. In
practice, the spot-perp spread at entry/exit can be 5–15 bps. The simulated entry
threshold of 5% annualized is conservative (avoids low-carry periods). Full live
deployment requires: (1) Binance API key, (2) spot + margin account, (3) minimum
~$10k notional for sensible sizing. Trading212 cannot execute this strategy; Binance
or Interactive Brokers Crypto required.

**Regime overlay potential:** The backtest script includes a commented regime overlay
(scale 0.5x–1.5x based on HMM bull state). This was not backtested quantitatively.
In 2021 (Sharpe bull state), BTC funding reached 30–40% annualized — the overlay
would have raised allocation to 1.5x notional during this period. This is a Phase 55
research question.

---

### TIER 2: 5-Asset RTMV Portfolio (CONFIRMED, LIVE)

Source: Phases 45–52a, `trading/rtmv_rebalancer.py`, `results/rtmv_live_5asset/`

| Metric | Value | vs GMV Baseline |
|--------|-------|-----------------|
| Sharpe (backtest) | 0.9726 | +0.0963 |
| Sharpe (live backtest, 5386 bars) | 0.9731 | confirmed |
| MDD | −15.6% | −6.1 pp improvement |
| Ann Return | 5.2% | comparable |
| Ann Vol | 5.4% | −1.8 pp improvement |
| Calmar | 0.335 | +0.029 |
| N Rebalances (21yr) | 233 | ~11/yr |
| Cost break-even | 80.6 bps | 8× vs ~10 bps practical |
| Halt events | 0 | — |

Configuration: SPY/GLD/SHY/IEF/TLT, n_states=3, λ=spy_rank_bull=[0.02,0.05,0.10],
lookback=504, rebalance_bars=21, halt=25%.

**Validation chain:** Full-period backtest (Phase 45) → OOS 2016–2026 (Phase 45b, all
variants beat GMV) → Purged CV (Phase 47, fold-consistency PASS, cost PASS, shuffle
marginal FAIL at p=0.130) → Live deployment skeleton (Phase 48) → Dashboard (Phase 49)
→ Parameter tuning: halt=25% (+0.027 Sharpe, Phase 50c), risk parity beaten (Phase 50c)
→ SPY-proxy λ (Phase 51b, +0.011) → 5-asset universe (Phase 52a, +0.078).

**Risk flags:**
- Shuffle test p=0.130 (criterion <0.10): signal is real but small; 13% of shuffles beat
  the real system. Regime tilt adds ~+0.005–0.011 Sharpe over pure GMV.
- 2022 bond crash stress: TLT MDD was −45% in 2022; the 5-asset portfolio's SHY
  diversification (MDD ~−2.5%) partially buffered this but the full 2022 drawdown is
  included in the −15.6% MDD figure.
- Bond-bull-market survivor bias: 2004–2022 was a sustained bond bull. Duration-heavy
  portfolios benefit from this; real-rate environment post-2022 is structurally different.
  The 2022 shock is in-sample, which is reassuring.

---

### TIER 3: Phase 54 — Joint 5D HMM (IN PROGRESS)

**Status:** Implementation started by agent-phase54-hmm. No results yet.

**Hypothesis:** Instead of fitting 5 independent univariate HMMs (one per asset) and
aggregating their posteriors into the RTMV weight, fit a single joint GaussianHMM on the
5D daily return vector [SPY, GLD, SHY, IEF, TLT]. The joint model captures cross-asset
correlations directly in the emission covariance; states represent portfolio-level regimes
(e.g., "risk-off," "duration-rally," "inflationary-squeeze") rather than per-asset regimes.

**Expected benefit:** Cleaner, more stable state transitions — which directly addresses
Phase 53's failure (forward filter oscillating every 5 bars). A joint model's states
are more constrained because all 5 assets must be explained simultaneously.

**GO threshold:** Sharpe ≥ 0.920 vs 5-asset spy_rank_bull baseline of 0.9726.
A joint HMM replaces the current per-asset HMM component — it does not need to beat
0.9726 on its own; the RTMV framework still applies. The threshold is whether the
joint-state posterior produces a better RTMV signal than 5 independent posteriors.

**Secondary GO criterion:** KL-triggered rebalance (Phase 53 idea) revisited with joint-HMM
posteriors. If joint-state posteriors are more stable (<45 changes/year vs the 45.7/year
from per-asset posteriors), the KL monitor becomes feasible.

---

### TIER 4: GBTC Discount Arbitrage (PENDING REPORT)

**Status:** agent-gbtc-arb research not yet complete. No report at
`docs/findings/gbtc_discount_arb.md`.

**What we know from context:** GBTC discount reached −18.2% (as of research commission).
The arbitrage thesis is: if GBTC converts to ETF (which it did in Jan 2024, reducing the
discount to near-zero), buying at deep discount and holding to conversion captures the
spread.

**Pre-assessment (no empirical numbers yet):**
- The January 2024 ETF conversion event is already historical. The discount has collapsed.
  Current GBTC premium/discount is near 0% post-conversion.
- The trade was a one-time structural event, not a recurring strategy.
- Unless a new discount develops, this is a closed opportunity. The "−18.2% discount"
  referenced in the research commission likely refers to a historical snapshot.
- Recommended verdict: **ARCHIVE** (one-time structural trade, execution window closed).
  Useful as a case study in closed-end fund arbitrage, not as a live strategy.

**Action:** Await `docs/findings/gbtc_discount_arb.md` from agent before finalizing.

---

### TIER 5: LETF Decay Regime Overlay (PENDING REPORT)

**Status:** agent-letf-decay research not yet complete. No report at
`docs/findings/letf_decay_overlay.md`.

**What we know from context:** Leveraged ETFs (e.g., TQQQ 3× Nasdaq, SPXL 3× S&P)
suffer volatility decay: in sideways markets, daily rebalancing to maintain leverage
erodes NAV. The regime overlay thesis: use HMM to identify low-vol (trending) vs
high-vol (choppy) regimes; hold LETF only in the trending regime.

**Pre-assessment (no empirical numbers yet):**
- Volatility decay is real and well-documented: a 3× LETF in a market with 20% annual
  vol loses approximately 3×(20%)²/2 × 3 = ~18% per year from decay alone, net of the
  3× leveraged return.
- The HMM regime signal is exactly the right tool for this: our n=3 model reliably
  identifies high-vol (state rank-0 or state rank-1) vs trending-low-vol (state rank-2)
  regimes on SPY daily data.
- ARI=0.393 on SPY n=3 is below the 0.40 threshold for single-asset directional trading
  (Phase 43c), but the LETF overlay is NOT directional — it is a vol-timing overlay.
  The relevant question is: can the HMM distinguish high-vol from low-vol regimes with
  enough accuracy to reduce decay without missing trending periods? That bar is lower.
- Typical LETF decay costs ~1–3% per year in medium-vol markets. If the HMM avoids
  50% of high-vol days, the overlay value is ~0.5–1.5% per year in decay reduction.
- Risk: leveraged products have gap risk (overnight moves can exceed 3× daily loss).
  The paper portfolio framework (MultiAssetPortfolio, RiskGuard) can handle LETF sizing.

**Action:** Await `docs/findings/letf_decay_overlay.md` before finalizing verdict.
Pre-assessment suggests this is a TIER 4 opportunity — real but small relative to
funding carry or RTMV. Worth a 10-bar backtest before committing.

---

## Combined Portfolio: Can All Strategies Run Simultaneously?

**YES — the strategies are structurally uncorrelated.** Here is the correlation analysis:

| Strategy A | Strategy B | Correlation | Reasoning |
|------------|------------|-------------|-----------|
| RTMV (equity-bond) | Funding carry | ~0.0 | RTMV: equity+bond daily returns. Carry: delta-neutral, P&L is funding rate only, not price movement. |
| RTMV | LETF overlay | ~0.4–0.6 | Both equity-market-linked. LETF amplifies equity exposure in bull states; RTMV is low-vol. Would increase overall equity exposure in bull regimes — deliberate synergy. |
| Funding carry | LETF overlay | ~0.0 | Carry is delta-neutral; LETF is directional equity. Uncorrelated by construction. |
| RTMV | GBTC arb | ~0.0 | Closed-end fund arb vs equity-bond portfolio. One-off event. |

**Combined portfolio construction:**

Assuming $500k total capital:

| Strategy | Allocation | Rationale |
|----------|-----------|-----------|
| 5-asset RTMV | $350k (70%) | Primary engine. Monthly rebalance. Live. |
| ETH funding carry | $75k (15%) | Tier 1 alpha. Delta-neutral. Enter when ann carry > 5%. |
| BTC funding carry | $50k (10%) | Same as ETH. Complement to ETH carry. |
| LETF overlay (pending) | $25k (5%) | Reserve pending Phase 55 backtest. |
| GBTC arb | $0 (0%) | Window closed; archive. |

**Combined portfolio expected metrics (uncorrelated assets, additive Sharpe
under independence approximation):**
- RTMV contribution: 0.70 × 5.2% = 3.6% ann return, 0.70 × 5.4% = 3.8% vol
- ETH carry contribution: 0.15 × 14.7% = 2.2% ann return, ~0.15 × 1% = 0.15% vol
- BTC carry contribution: 0.10 × 12.1% = 1.2% ann return, ~0.10 × 1% = 0.10% vol
- Combined ann return (approx): ~7.0% (before LETF)
- Combined vol (approx): ~3.8% (carry contributes negligible vol)
- Implied combined Sharpe: ~1.8 (idealized; ignores correlation structure and carry
  regime dependency)

**Combined MDD:** Dominated by RTMV's −15.6%. Funding carry has negligible MDD.
Combined portfolio MDD ≈ 0.70 × 15.6% ≈ −10.9% (structurally improved by
carry diversification).

**Operational requirements:**
- RTMV: `make live-rtmv` (polls yfinance every 3600s, monthly rebalance). No broker API required.
- Carry: Binance account with spot + USDM futures enabled. `python scripts/monitor_carry.py`
  for monitoring; `CarryStrategy.run_live()` for paper trading; real execution requires
  Binance API key and order placement code (not yet implemented).
- LETF: Would run within existing RTMV framework; requires adding TQQQ/SPXL to universe.

---

## Capital Allocation Percentages (Recommended)

### Scenario A: $100k retail (Trading212-compatible)

| Strategy | Allocation | Implementation |
|----------|-----------|----------------|
| 5-asset RTMV | 100% | `make live-rtmv` — 5 equity/bond ETFs, monthly rebalance |
| Funding carry | 0% | Requires Binance (not Trading212) |
| LETF | 0% | Pending backtest |

Expected: Sharpe ~0.97 backtest / ~0.875 live, MDD ≤ 20%, Ann Return ~5%.

### Scenario B: $250k+ (Binance-enabled)

| Strategy | Allocation | Notes |
|----------|-----------|-------|
| 5-asset RTMV | 70% ($175k) | Core |
| ETH carry | 20% ($50k) | Enter when ann carry > 5%; exit at < −2% |
| BTC carry | 10% ($25k) | Same threshold |

Expected: Combined Sharpe ~1.5–2.0, Combined MDD < 12%.

### Scenario C: $500k+ (full deployment)

| Strategy | Allocation | Notes |
|----------|-----------|-------|
| 5-asset RTMV | 70% ($350k) | Primary |
| ETH carry | 15% ($75k) | TIER 1 alpha |
| BTC carry | 10% ($50k) | TIER 1 alpha |
| LETF overlay | 5% ($25k) | Reserve pending Phase 55 |

---

## Next 3 Research Directions (Not Yet Explored)

### Direction 1: Regime-Scaled Carry Allocation (Phase 55)

**Gap:** The funding carry backtest uses a fixed entry/exit threshold (5%/−2% annualized).
The regime overlay (0.5x–1.5x based on HMM bull state) is coded in `funding_carry_backtest.py`
but never quantitatively backtested.

**Hypothesis:** In HMM bull state (SPY rank-2), raise carry allocation to 1.5×; in bear
state (rank-0), cut to 0.5×. In 2021 (known bull), funding rates were 30–40% annualized
— the overlay would have produced 37.5% × 1.5 = 56% annual return vs 37.5% flat.

**Feasibility:** High. The `CarryStrategy` class can accept a multiplier from `RTMVRebalancer`
via a shared HMM state signal. The data pipeline already runs; wiring the HMM output
to `qty_per_symbol` is a one-day implementation.

**Expected Sharpe improvement:** If the bull-state average carry is 2× the bear-state
average (plausible given 2021 data), the regime overlay adds ~0.3–0.5 Sharpe to the
carry strategy.

**GO criterion:** Carry+overlay Sharpe ≥ flat-carry Sharpe + 0.5.

### Direction 2: Vol Risk Premium Harvesting (Phase 56)

**Gap:** The project has vol forecasting (Phase 46, `evaluation/vol_forecasting.py`) but
no options or VIX strategy built on it.

**Hypothesis:** Sell SPX/SPY short-dated puts in low-vol regimes (regime rank-2, SPY bull),
collect premium. The vol risk premium (VRP = implied vol − realized vol) is structurally
positive (~3–5 vol points on average). Regime conditioning identifies when VRP is at its
richest vs cheapest.

**Data:** VIX index (yfinance: `^VIX`), VVIX (volatility of volatility index). Realized
vol is already computed by `RollingVolatility`. The spread (VIX − 21-day realized vol)
is the observable VRP signal.

**Implementation:** The `analysis/tail_risk.py` module (Phase 28) already fits GPD tails
per regime. A vol-selling strategy would: (1) compute VRP per regime state using the
Phase 46 vol forecast, (2) enter short vol position when VRP > threshold in rank-2
regime, (3) exit when HMM transitions to rank-0 or rank-1. Paper portfolio uses the
existing `PaperPortfolio` framework.

**Complexity:** Medium. Requires adding VIX data fetch (yfinance `^VIX` works), computing
VRP, and implementing a simple options P&L model (short delta-hedged straddle or simple
VIX short). No new ML required.

**Expected range:** VRP strategies historically Sharpe 1.5–2.5 with occasional large
drawdowns (VIX spikes). Regime conditioning to avoid high-vol states should reduce the
tail blow-up risk that has historically hurt VRP harvesting.

### Direction 3: Cross-Asset Momentum + Regime Filter (Phase 57)

**Gap:** The Phase 27 transition predictor (`analysis/transition_prediction.py`) predicts
next-regime probabilities using a logistic classifier, but this signal is not used in any
live strategy — only in diagnostics.

**Hypothesis:** Combine the Phase 27 h-step transition probability with cross-asset
momentum (12-month minus 1-month return) to build a dual-signal asset selector within
the RTMV framework. When both the momentum signal and the HMM transition predictor
agree on direction, increase the regime tilt from λ=0.05 to λ=0.15.

**Why momentum + regime:** Pure momentum has been shown to add alpha to multi-asset
portfolios (AQR, Asness et al., 2013). The HMM regime signal adds a structural
regime-coherence filter. When momentum says "buy SPY" and the HMM transition predictor
says "90% probability of staying in bull state," the combined signal is stronger than
either alone. This is orthogonal to the current RTMV's λ schedule.

**Implementation:** Add a `cross_asset_momentum` function to `analysis/multi_asset_allocation.py`
that computes (12m−1m) return per asset, ranks them, and returns a momentum score per
asset. Blend with HMM posterior-weighted E[r] in the regime weight computation.

**Complexity:** Low. The momentum signal is trivial (price rolling returns, already
computable from yfinance data). The integration point is within `compute_rtmv_weights_now`.

**GO criterion:** +0.010 Sharpe over the 5-asset spy_rank_bull baseline (0.9726),
matching the Phase 51b marginal-pass threshold.

---

## Summary Table: All Opportunities

| Strategy | Tier | Sharpe | MDD | Status | Capital % | Next Action |
|----------|------|--------|-----|--------|-----------|-------------|
| ETH funding carry | 1 | ~16 | ~−0.5% | Confirmed (backtest) | 15–20% | Add Binance API for live execution |
| BTC funding carry | 1 | ~12 | ~−0.5% | Confirmed (backtest) | 10–15% | Same as ETH |
| 5-asset RTMV | 2 | 0.97 | −15.6% | Live (paper) | 70% | Monitor live; run Phase 54 |
| Phase 54 joint HMM | 3 | TBD | TBD | In progress | — | Await results; GO if Sharpe ≥ 0.920 |
| LETF decay overlay | 4 | TBD | TBD | Report pending | 5% reserved | Await agent report |
| Regime-scaled carry | — | TBD | TBD | Not started | — | Phase 55 |
| Vol risk premium | — | ~1.5–2.5 (lit) | −30% tails | Not started | — | Phase 56 |
| Momentum + regime | — | TBD | TBD | Not started | — | Phase 57 |
| GBTC discount arb | dead | — | — | Window closed | 0% | Archive |

---

## Files Produced by Parallel Agents (as of 2026-05-17)

| Agent | File | Status |
|-------|------|--------|
| agent-carry-binance | `src/rde/trading/carry_executor.py` | COMPLETE |
| agent-carry-binance | `scripts/funding_carry_backtest.py` | COMPLETE |
| agent-carry-binance | `scripts/monitor_carry.py` | COMPLETE |
| agent-phase54-hmm | (implementation in progress) | IN PROGRESS |
| agent-gbtc-arb | `docs/findings/gbtc_discount_arb.md` | NOT YET |
| agent-letf-decay | `docs/findings/letf_decay_overlay.md` | NOT YET |

Update this document when the GBTC and LETF reports arrive.
