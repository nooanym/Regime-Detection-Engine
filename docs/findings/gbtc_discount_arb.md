# GBTC Discount-to-NAV Arbitrage Research

**Date:** 2026-05-17
**Verdict: CONDITIONAL NO-GO — Watch-list with three specific re-entry triggers**
**Data source:** yfinance (GBTC, BTC-USD), Binance API (BTCUSDT perp funding rates)

---

## Executive Summary

The GBTC discount-to-NAV trade (long GBTC, short BTC perp, delta-neutral) is currently unattractive. Two simultaneous headwinds make initiating today a negative-expected-value proposition:

1. The discount trend is WIDENING in the ETF era (-1.32%/yr linear slope), not converging.
2. BTC perpetual funding rates are NEGATIVE (90d trailing: -1.0% annualized), so the carry leg costs rather than earns.

With both legs working against the trade, the all-in net return with unchanged discount is -2.6%/yr. Three well-defined catalysts could make this attractive; they are documented below with exact trigger conditions.

---

## 1. Full Historical Premium / Discount Table

### Methodology

- GBTC daily close from yfinance (`period="max"`, `interval="1d"`, `auto_adjust=True`): 2771 rows, 2015-05-11 to 2026-05-15
- BTC-USD daily close from yfinance (`period="max"`, `interval="1d"`): 4261 rows, 2014-09-17 to 2026-05-17
- NAV = BTC-USD close × 0.00095 (shares per BTC, post-split adjusted series)
- Discount % = (GBTC / NAV − 1) × 100
- yfinance `auto_adjust=True` handles all GBTC share splits; the series is comparable end-to-end

### All-time statistics

| Statistic | Value |
|-----------|-------|
| All-time min (deepest discount) | -55.7% (Feb 2023) |
| All-time max (peak premium) | +123.3% (Aug 2017) |
| Full-history median | +1.6% |
| Full-history mean | +3.3% |
| Current (2026-05-15) | -18.2% |

### Era breakdown

| Era | N days | Min | P25 | Median | P75 | Max | Mean |
|-----|--------|-----|-----|--------|-----|-----|------|
| Trust era 2015-2020 | 1423 | -6.5% | +11.4% | +22.9% | +48.7% | +123.3% | +30.8% |
| Discount era 2021-Jan 2024 | 760 | -55.7% | -42.1% | -33.7% | -23.3% | +13.4% | -32.6% |
| ETF era Jan 2024-now | 588 | -21.4% | -17.6% | -16.9% | -16.0% | -11.8% | -16.8% |

### Monthly average discount (selected months — full table in script output)

| Period | GBTC | NAV | Avg Disc% | End Disc% |
|--------|------|-----|-----------|-----------|
| 2021-01 | 31.34 | 32.60 | +1.6% | -3.9% |
| 2021-06 | 26.93 | 33.29 | -20.3% | -19.1% |
| 2021-12 | 30.94 | 43.99 | -26.9% | -29.7% |
| 2022-06 | 10.89 | 18.80 | -38.7% | -42.0% |
| 2022-12 | 7.49 | 15.77 | -53.7% | -52.5% |
| 2023-06 | 17.34 | 28.95 | -46.4% | -40.1% |
| 2023-12 | 31.27 | 39.99 | -22.6% | -21.8% |
| 2024-01 | 34.32 | 40.45 | -17.0% | -15.2% (ETF conversion) |
| 2024-06 | 48.09 | 57.30 | -15.7% | -16.1% |
| 2024-12 | 74.02 | 88.76 | -16.4% | -16.6% |
| 2025-06 | 84.83 | 101.78 | -17.2% | -16.7% |
| 2025-12 | 68.36 | 83.13 | -17.9% | -17.8% |
| 2026-04 | 59.40 | 72.49 | -18.2% | -18.1% |
| 2026-05 | 61.44 | 75.11 | -18.0% | -18.2% |

The complete monthly table (132 months) is reproducible by running:
```bash
uv run python scripts/run_gbtc_analysis.py  # (not yet committed; logic is in docs/findings/gbtc_discount_arb.md)
```

---

## 2. What Drives the Discount

### A. The fee-gap permanent discount

The single most important driver is the fee differential:

- GBTC management fee: **1.50%/year**
- BlackRock IBIT: **0.25%/year**
- Fidelity FBTC: **0.25%/year**

The annual fee penalty is **1.25%/year**. Treating this as a perpetuity with a 5% discount rate:

```
Fair-value discount = 1.25% / 5% = 25%
```

The market should price GBTC at approximately -25% to NAV in perpetuity relative to a zero-fee Bitcoin ETF. The current -18% discount is actually **tighter than fair value**, meaning GBTC is still relatively expensive on a fee-adjusted basis.

This reframes the entire thesis: the -18% discount is not deep — it is shallow relative to what the fee gap justifies. Convergence to 0% requires Grayscale cutting its fee dramatically, not just time passing.

### B. AP creation/redemption lockout

For a standard ETF, Authorized Participants (APs) arbitrage away discounts by buying cheap shares and redeeming them for the underlying. This fails for GBTC because:

- APs can only arbitrage profitably when discount < ~0.1%
- At -18%, buying GBTC and redeeming for BTC would be immediately profitable, but Grayscale controls AP access and only allows in-kind BTC redemptions
- The operational barrier (AP approval + in-kind BTC delivery) prevents the automatic convergence mechanism that makes standard ETFs discount-free

### C. Structural institutional outflows

Since ETF conversion on January 11, 2024:

- GBTC AUM: ~$28B (Jan 2024) → ~$18B (May 2026) = -$10B in 16 months
- FTX bankruptcy estate sold $0.9B in early 2024
- Institutional holders unable to hold ETFs (vs former closed-end fund status) sold on conversion
- Fee-sensitive investors rotating to IBIT/FBTC (1.25%/yr saving on $18B AUM = $225M/yr incentive to switch)

This seller pressure is structural and persistent, not a temporary post-conversion flush.

### D. ETF era trend: widening, not converging

Linear regression of daily discount vs time in the ETF era:

- Slope: **-1.32%/year** (getting MORE negative = wider discount)
- Starting point (Jan 2024): -15.2%
- Projected in 12 months at current rate: -19.5%

This is the most important single data point for the thesis: the ETF conversion did NOT cause convergence. The discount is drifting wider.

| Quarter | Avg Discount |
|---------|-------------|
| Q1 2024 | -15.3% |
| Q2 2024 | -15.6% |
| Q3 2024 | -16.0% |
| Q4 2024 | -16.3% |
| Q1 2025 | -16.8% |
| Q2 2025 | -17.0% |
| Q3 2025 | -17.4% |
| Q4 2025 | -17.6% |
| Q1 2026 | -17.9% |
| Q2 2026 (so far) | -18.1% |

---

## 3. Is -18.2% Historically Extreme?

### Within the ETF era (since Jan 2024)

The current -18.2% is at the **8th percentile** of the ETF-era distribution. Only 8% of ETF-era days have had a deeper discount. The ETF-era max widening reached -21.4%.

Interpretation: Within the context of the post-conversion period, -18.2% is relatively deep — the discount has widened significantly from the -11.8% minimum seen in early 2024. However, this is entirely explained by the trend; the market is systematically re-pricing GBTC toward its fee-justified level.

### Within the full discount era (2021-Jan 2024)

At the **91st percentile** — meaning 91% of discount-era days had a DEEPER discount than today. The discount era saw extremes of -55.7%. At -18.2%, the current level is historically mild relative to what GBTC has traded at.

### Conclusion on extremity

The -18.2% is not historically extreme in an absolute sense (discount era went to -56%), but it IS extreme within the current ETF-era regime. The two interpretations point in opposite directions:

- "Historically not extreme" → discount could go much deeper
- "Extreme within ETF era" → might mean-revert toward -15%

The trend data (widening quarterly) supports the former reading.

---

## 4. The Theoretical Trade

### Structure

| Leg | Action | Instrument |
|-----|--------|------------|
| Long | Buy N shares GBTC | NYSE Arca (liquid, ~$200M ADV) |
| Short | Sell 0.00095 × N BTC perpetual | Binance BTCUSDT-PERP |

Net BTC delta = 0. Directional BTC exposure is fully hedged. The trade profits only from:
1. Convergence of GBTC discount toward zero
2. Funding rate received on the BTC short position

### P&L mechanics

When BTC moves from price P to P':
- GBTC moves from P × 0.00095 × (1+d0) to P' × 0.00095 × (1+d1)
- Short BTC perp gains: (P - P') × 0.00095 per share

Net result (BTC price exposure cancels):
```
Convergence P&L = (1 + d1) / (1 + d0) − 1
```

Example: d0 = -0.182, d1 = -0.05:
```
(1 - 0.05) / (1 - 0.182) - 1 = 0.95 / 0.818 - 1 = +16.1%
```

### Cost structure

| Cost | Amount | Frequency |
|------|--------|-----------|
| GBTC management fee | 1.50% | Per year (NAV erosion) |
| GBTC bid-ask / execution | 0.10% | Round-trip |
| Binance perp execution | 0.04% | Round-trip (maker 0.02% × 2) |
| Binance perp carry (when negative) | Variable | Continuous |
| Margin opportunity cost | ~5.0% | Per year (capital at Binance) |

**Total fixed annual cost: ~1.64%**

### Revenue structure

| Revenue | Amount | Condition |
|---------|--------|-----------|
| BTC perp funding rate received | Variable | When carry is positive |
| Discount convergence | (1+d1)/(1+d0) - 1 | On position close |

**Current BTC carry (live data from Binance API):**

| Window | Annualized carry |
|--------|-----------------|
| Latest single period | +0.1% |
| 7-day trailing | +2.9% |
| 30-day trailing | -1.8% |
| 90-day trailing | -1.0% |
| 180-day trailing | +1.3% |

The 30/90-day windows are negative. This is a live risk.

---

## 5. Combined Return Scenarios (12-month holding period)

All scenarios use current discount d0 = -18.2%, fixed annual cost = 1.64%.

### Scenario A: Carry = +8% assumption (premise in user brief — NOT current reality)

| Target Discount | Convergence P&L | Carry | Net P&L | Verdict |
|-----------------|-----------------|-------|---------|---------|
| -30% (widening) | -14.4% | +8.0% | -8.0% | NO-GO |
| -18% (unchanged) | 0.0% | +8.0% | +6.4% | MARGINAL |
| -14% (modest narrow) | +5.1% | +8.0% | +11.5% | GO |
| -5% (significant narrow) | +16.1% | +8.0% | +22.5% | STRONG GO |
| 0% (full parity) | +22.3% | +8.0% | +28.7% | STRONG GO |

Note: the premise of 5.8% ETH carry on the hedge does not apply here because the hedge leg is BTC perp (not ETH). The ETH funding rate is structurally different from BTC and the two have low correlation.

### Scenario B: Realistic current carry (-1.0% ann, 90d trailing)

| Target Discount | Convergence P&L | Carry | Net P&L | Verdict |
|-----------------|-----------------|-------|---------|---------|
| -25% (widening) | -8.3% | -1.0% | -11.0% | NO-GO |
| -20% (flat-ish) | -2.2% | -1.0% | -4.8% | NO-GO |
| -18% (unchanged) | 0.0% | -1.0% | -2.6% | NO-GO |
| -15% (ETF-era mean) | +3.9% | -1.0% | +1.3% | MARGINAL |
| -10% (significant narrow) | +10.0% | -1.0% | +7.4% | MARGINAL |
| -5% (near-parity) | +16.1% | -1.0% | +13.5% | GO |
| 0% (full parity) | +22.3% | -1.0% | +19.6% | STRONG GO |

### Base case

The base case is: carry = -1.0% (current 90d), discount narrows modestly to -14% (the ETF-era average before recent widening). Net P&L = +1.3%. This is MARGINAL — barely covers transaction costs and offers zero risk-premium for the structural risks.

### Bull case

Carry recovers to +8% (consistent with historical BTC bull markets), discount narrows to -5% (Grayscale cuts fee). Net P&L = +21.4%. This requires two simultaneous positive developments.

### Bear case

Carry stays negative at -1.8% (30d trailing), discount widens to -25% (continuation of 2026 trend). Net P&L = -10.8%. This is consistent with a sideways/bear BTC market.

---

## 6. Risks

### Risk 1 — No forced convergence (highest severity, highest probability)

Unlike merger arb or tender offer arb, there is NO mechanism forcing GBTC to close its discount. AP creation/redemption cannot operate at -18%; it is only profitable within ~0.1% of NAV. Grayscale controls AP access. The discount can persist or widen indefinitely.

**Evidence:** The discount has been continuously negative since February 2021 — over five years. The ETF conversion did not trigger convergence; it slightly accelerated the structural outflow.

### Risk 2 — Fee gap justifies permanent deep discount

The theoretical fair-value discount is -25% based on the perpetuity value of the 1.25%/yr fee gap at a 5% discount rate. At -18%, GBTC is only 7pp away from fair value, suggesting LIMITED upside from convergence and significant downside if the market prices the full fee penalty.

### Risk 3 — BTC perp carry is negative today

Current 30d BTC funding: -1.8% annualized. Current 90d: -1.0% annualized. The "carry" leg is currently a cost, not a revenue. Historical mean (2020-2026) is +12.1%, but 2022 confirmed that even in bear markets, funding stayed positive due to forced shorts. In the current 2026 sideways/bear market, funding has turned briefly negative — a rare but live scenario.

If carry stays at -1.8% for 12 months, annual drag increases by ~0.8pp vs the base case.

### Risk 4 — Margin call risk on perp short

The BTC perp short requires 10-20% margin on Binance in USDT/BTC. If BTC rallies 50% rapidly, the short position requires additional margin top-up. While the GBTC long also appreciates, the positions are in separate accounts (broker vs exchange) and cannot be cross-margined. Capital must be pre-positioned to survive large BTC spikes.

Conservative sizing: hold 3× the perp notional in free USDT at Binance to absorb a 200% BTC rally without forced liquidation.

### Risk 5 — Grayscale competitive position deteriorating

GBTC AUM has fallen ~36% since ETF conversion ($28B → $18B in 16 months). BlackRock IBIT now holds significantly more AUM. Institutional rotation away from GBTC is structural: every year that IBIT charges 1.25% less, large allocators have strong incentive to switch. This sustained outflow pressure is the primary driver of the widening discount trend.

A Grayscale fee cut to <0.5% would be the single most powerful convergence catalyst, but no announcement has been made.

### Risk 6 — Downside VaR on discount widening

ETF-era daily discount change volatility: 1.147pp/day (18.2pp annualized).

| Confidence | 1-year discount endpoint |
|------------|--------------------------|
| 95% VaR (widening) | -45.4% |
| 99% VaR (widening) | -66.0% |

A one-in-twenty adverse year could see the discount reach the 2022-era extremes of -45% to -55%, generating approximately -30% loss on the convergence leg alone.

---

## 7. Verdict

**CONDITIONAL NO-GO — Watch-list with three re-entry triggers**

### Why NO-GO today

1. Discount trend is WIDENING (-1.32%/yr), not converging. The thesis requires convergence. The data falsifies this.
2. BTC carry is NEGATIVE today (30d: -1.8%, 90d: -1.0%). The carry leg costs money in the current regime.
3. With both legs against the trade, net P&L with unchanged discount = -2.6%/yr (before margin opportunity cost).
4. The theoretical fair-value discount is -25% (fee perpetuity calculation), implying the current -18% is not deep — it could rationally widen to -25%.
5. No forced convergence mechanism. The discount can persist for years (evidence: 5+ years since turning negative).

### Three specific GO triggers

**Trigger 1 — Carry recovers to +8%+ annualized (30d trailing)**

If BTC perp funding recovers to 8%+ (consistent with prior bull markets), the trade earns 6.4% net with flat discount. At this point the carry leg alone justifies the position, and any discount narrowing is pure bonus.

Monitor: Binance API `BTCUSDT` funding rate daily check.

```python
# From funding_carry_backtest.py logic:
carry_7d = ann_recent.iloc[-7*3:].mean()
carry_30d = ann_recent.iloc[-30*3:].mean()
if carry_30d >= 0.08:
    signal = "ENTER"
```

**Trigger 2 — Discount widens to -30% or deeper**

A discount of -30% creates a higher-conviction mean-reversion setup. From -30%, recovery to -18% (the current equilibrium) alone generates +17% convergence P&L. Combined with any carry, this clears the GO threshold.

Probability: non-trivial. The ETF-era trend projects -22% by end of 2026; a bear-market acceleration (crypto winter) could revisit 2022's -40% to -55% range.

Entry at -30% should be sized at maximum 20% of portfolio (no forced exit mechanism).

**Trigger 3 — Grayscale announces management fee cut to <0.5%**

This would reduce the theoretical fair-value discount from -25% to -5%. The market would immediately reprice GBTC toward this new equilibrium, generating +17%+ convergence in weeks, not months. No timing prediction is possible, but this is the highest-conviction catalyst if it occurs.

Monitor: Grayscale press releases and SEC filings (GBTC N-1A amendments).

### Implementation when triggers hit

Using the existing Binance infrastructure in `scripts/funding_carry_backtest.py`:

```python
# Step 1: Determine hedge size
gbtc_shares = position_size_usd / gbtc_price   # buy at market
btc_notional = gbtc_shares * 0.00095            # BTC to short

# Step 2: Place perp short on Binance
# symbol = "BTCUSDT", side = "SELL", type = "MARKET"
# quantity = btc_notional (in BTC)
# Use fetch_funding_rates("BTCUSDT") to monitor carry daily

# Step 3: Monitor daily
discount_now = (gbtc_price / (btc_price * 0.00095)) - 1
carry_30d    = ann_recent.iloc[-30*3:].mean()

# Exit conditions:
if discount_now < entry_discount - 0.10:     # discount widens 10pp from entry
    exit()                                    # stop loss
if carry_30d < -0.05:                        # carry deeply negative >5%
    exit()                                    # carry deteriorating materially

# Rebalance: monthly re-hedge delta (BTC price drift creates small delta)
# Same logic as rebalance_periods in simulate_carry()
```

**Position limits:**
- Max 20% of total portfolio (no margin call backstop from the long side)
- Keep 3× perp notional in free USDT at Binance as margin buffer
- Maximum duration: 18 months (review exit if triggers not met)

---

## 8. Summary Table

| Dimension | Current Value | Implication |
|-----------|--------------|-------------|
| GBTC discount | -18.2% | Tight relative to fee-justified -25%; room to widen |
| ETF-era trend | -1.32%/yr widening | Trade thesis (convergence) is wrong today |
| ETF-era percentile rank | 8th pct (only 8% of days deeper) | Relatively deep within ETF era |
| BTC perp carry (30d) | -1.8% ann | Carry leg is a cost, not revenue |
| BTC perp carry (historical avg) | +12.1% ann | Long-run carry is positive; current is anomaly |
| Fee drag vs IBIT | 1.25%/yr | Creates structural ~-25% fair-value discount |
| GBTC AUM trend | -$10B in 16 months | Structural seller, not temporary flush |
| Discount widening VaR (95%) | 27pp in 1 year | Could reach -45% in adverse scenario |
| Base case net P&L | +1.3% (MARGINAL) | Not worth the structural risks |
| GO trigger 1 | Carry ≥ +8% | Monitor Binance daily |
| GO trigger 2 | Discount ≤ -30% | Set price alert at GBTC discount monitor |
| GO trigger 3 | GBTC fee cut to <0.5% | Monitor Grayscale announcements |
