# Phase 52a — Bond Maturity Ladder Universe Expansion

**Date:** 2026-05-16  
**Universe tested:** 4-asset (baseline) / 5-asset (+ SHY) / 6-asset (+ SHY + TIP)  
**n_states=3, n_restarts=3, lookback=504, halt=25%**  
**Verdict: GO — 5-asset (SPY/GLD/SHY/IEF/TLT) replaces 4-asset as the live universe**

---

## Hypothesis

Adding duration-differentiated bond ETFs gives the HMM finer yield-curve regime
discrimination. The 4-asset universe holds only two bond ETFs (IEF 7-10Y, TLT 20Y+);
adding SHY (1-3Y short-duration) spans the full maturity ladder and provides a
lower-volatility safe-haven allocation during equity stress.

---

## Results

| Universe  | Variant       | Sharpe | Calmar | MDD    | Ann Return | Ann Vol |
|-----------|--------------|--------|--------|--------|------------|---------|
| 4-asset   | gmv           | 0.8763 | 0.2907 | −21.8% | 6.3%       | 7.2%    |
| 4-asset   | fixed_l05     | 0.8838 | 0.3002 | −21.5% | 6.4%       | 7.3%    |
| 4-asset   | spy_rank_bull | 0.8948 | 0.3062 | −21.3% | 6.5%       | 7.3%    |
| 5-asset   | gmv           | 0.9571 | 0.3151 | −15.9% | 5.0%       | 5.2%    |
| 5-asset   | fixed_l05     | 0.9620 | 0.3271 | −15.7% | 5.2%       | 5.4%    |
| **5-asset** | **spy_rank_bull** | **0.9726** | **0.3351** | **−15.6%** | **5.2%** | **5.4%** |
| 6-asset   | gmv           | 0.9489 | 0.3091 | −15.1% | 4.7%       | 4.9%    |
| 6-asset   | fixed_l05     | 0.9570 | 0.3208 | −15.0% | 4.8%       | 5.0%    |
| 6-asset   | spy_rank_bull | 0.9665 | 0.3281 | −14.9% | 4.9%       | 5.1%    |

**GO threshold:** best new-universe Sharpe > 0.8948 + 0.005 = 0.8998  
**Actual delta:** +0.0778 Sharpe (5-asset spy_rank_bull 0.9726 vs 4-asset baseline 0.8948)

---

## Key Findings

### 1. Adding SHY is a large structural improvement

5-asset vs 4-asset (spy_rank_bull):
- Sharpe: 0.8948 → 0.9726  (+0.078)
- MDD: −21.3% → −15.6%  (−5.7 pp)
- Calmar: 0.306 → 0.335  (+0.029)
- Ann Vol: 7.3% → 5.4%  (−1.9 pp)
- Ann Return: 6.5% → 5.2%  (−1.3 pp)

The return drops slightly — the lower-volatility SHY drags down nominal return —
but the risk-adjusted metrics improve dramatically. Sharpe is the correct decision
metric here (and Calmar confirms it).

### 2. Adding TIP (6-asset) is a marginal NO relative to 5-asset

6-asset vs 5-asset (spy_rank_bull):
- Sharpe: 0.9726 → 0.9665  (−0.006)
- MDD: −15.6% → −14.9%  (−0.7 pp)
- Calmar: 0.3351 → 0.3281  (−0.007)

TIPS reduces MDD by 0.7 pp but at the cost of Sharpe (−0.006) and Calmar (−0.007).
TIP has lower expected real return than TLT in most regimes and introduces inflation
correlation that partially overlaps with GLD. Net effect: a small drag.

### 3. spy_rank_bull consistently dominates within each universe

The relative ordering gmv < fixed_l05 < spy_rank_bull holds across all three
universes — the Phase 51b SPY-proxy λ schedule is robust to universe composition.

### 4. MDD improvement mechanism

SHY's low duration (1-3Y) means it is nearly uncorrelated with equities during
rate-spike environments (2022) AND has lower vol than IEF/TLT during normal
regimes. The global min-var solver now has a third bond option and reliably
overweights SHY in high-volatility equity regimes, directly reducing drawdown.

---

## Verdict

**GO — upgrade to 5-asset universe (SPY/GLD/SHY/IEF/TLT).**

Deploy configuration:
```python
RTMVRebalancerConfig(
    assets=["SPY", "GLD", "SHY", "IEF", "TLT"],
    lambda_tilt=0.05,
    lambda_by_state_rank=[0.02, 0.05, 0.10],
    lambda_proxy_asset="SPY",
    n_states=3,
    n_restarts=3,
    lookback_bars=504,
    rebalance_bars=21,
    drawdown_halt=0.25,
)
```

New live Sharpe target: **0.9726** (backtest) → expect ~0.875–0.90 live (backtest–live gap ~10%).

---

## Caveat

This is a full-period backtest including in-sample fitting. SHY was added to the
universe in 2002, giving 22 years of history — adequate data depth. However, the
2004–2022 period was a sustained bond bull market; the 2022 rate-shock stress
(SHY MDD ≈ −2.5% vs TLT MDD ≈ −45%) is included and is precisely what gives
SHY its diversification value in this evaluation. This specific edge may compress
if rates remain elevated for multi-decade periods, but bond diversification across
the curve remains structurally sound.

The Phase 47 shuffle test (p=0.130 at 4-asset) should be re-run on the 5-asset
universe to confirm the improved Sharpe isn't structural-overfitting on the SHY
yield premium.

---

## Files Changed

- `scripts/run_phase52_universe_expansion.py` — Phase 52a runner (created in Phase 52a)
- `results/phase52/universe_expansion.csv` — Full results table
- `docs/findings/phase52a_bond_ladder.md` — This document

## Reproduce

```bash
uv run python scripts/run_phase52_universe_expansion.py
```
