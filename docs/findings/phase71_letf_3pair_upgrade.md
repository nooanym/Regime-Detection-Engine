# Phase 71 — LETF 3-Pair Upgrade to Multi-Strategy Portfolio

**Date:** 2026-05-18

**Verdict: GO** (Combined Sharpe 11.5636 > 11.05 threshold)

## Objective

Replace the single TQQQ/QQQ LETF pair in the Phase 68 multi-strategy portfolio with the
3-pair equal-weight combo validated in Phase 70 (TQQQ/QQQ + SOXL/SOXX + UPRO/SPY).

## Result Summary

| Metric | Phase 68 (single TQQQ/QQQ) | Phase 71 (3-pair combo) | Change |
|---|---|---|---|
| **Combined Sharpe** | **11.0483** | **11.5636** | **+0.515** |
| Ann Return | 14.79% | 15.76% | +0.97pp |
| Ann Vol | 1.34% | 1.36% | +0.02pp |
| Max DD | -0.21% | -0.20% | +0.01pp (improvement) |
| Calmar | 69.35 | 78.74 | +9.39 |
| Cum Return | 140.09% | 153.18% | +13.09pp |

## LETF Leg Comparison

### Phase 68: Single TQQQ/QQQ Pair

| Metric | Value |
|---|---|
| Sharpe | 6.2401 |
| Ann Return | 29.33% |
| Ann Vol | 4.70% |
| Max DD | -1.32% |
| Calmar | 22.27 |

### Phase 71: 3-Pair Equal-Weight Combo

| Metric | TQQQ/QQQ | SOXL/SOXX | UPRO/SPY | **Combo** |
|---|---|---|---|---|
| Borrow + costs/yr | 2.00% | 3.40% | 1.51% | (blended) |
| Sharpe (combo) | — | — | — | **7.0552** |
| Ann Return | — | — | — | 36.71% |
| Ann Vol | — | — | — | 5.20% |
| Max DD | — | — | — | -0.75% |
| Calmar | — | — | — | 49.01 |

The 3-pair combo vs single TQQQ/QQQ pair over the 2020–2026 window:
- Ann Return: 36.71% vs 29.33% (+7.38pp) — SOXL/SOXX and UPRO/SPY add substantial gross premium
- Sharpe: 7.06 vs 6.24 (+0.82) — diversification reduces volatility relative to return
- MDD: −0.75% vs −1.32% (MDD improved despite higher ann return)

The improvement is consistent with Phase 70's finding that the 3-pair combo Sharpe (5.47) dominates the
single TQQQ/QQQ pair Sharpe (4.58) over the full 2010–2026 period.

## Per-Year Returns

| Year | Combined P71 | Combined P68 | Carry | LETF 3-pair | RTMV |
|---|---|---|---|---|---|
| 2020 | +26.79% | +25.28% | +21.99% | +64.97% | +4.97% |
| 2021 | +30.44% | +29.70% | +35.43% | +15.67% | +2.33% |
| 2022 | +7.30% | +6.79% | +2.14% | +48.94% | -12.24% |
| 2023 | +10.07% | +9.70% | +7.41% | +25.52% | +9.58% |
| 2024 | +16.10% | +14.96% | +13.85% | +31.63% | +8.74% |
| 2025 | +9.56% | +8.29% | +4.48% | +39.38% | +13.13% |
| 2026 | +1.89% | +1.33% | +0.27% | +11.24% | +1.12% |

Every calendar year is positive for the combined portfolio. The 2022 stress year
(Nasdaq -33%, crypto winter) shows combined +7.30%: LETF contributed +48.94%
(vol spike = more decay premium), carry contributed +2.14%, RTMV -12.24% (equity
drawdown partially offsets).

## Cross-Strategy Correlations

| Pair | Phase 71 | Phase 68 |
|---|---|---|
| carry↔letf | -0.0971 | -0.0765 |
| carry↔rtmv | -0.0006 | -0.0006 |
| letf↔rtmv | -0.0123 | -0.0189 |

Adding SOXL/SOXX (semiconductor exposure, uncorrelated with broad equity RTMV) slightly
increases the carry↔letf negative correlation (from -0.077 to -0.097). All three remain
structurally near-zero — the diversification thesis holds.

## GO/NO-GO

| Criterion | Threshold | Result | Status |
|---|---|---|---|
| Combined Sharpe | ≥ 11.05 (Phase 68 baseline) | 11.5636 | **PASS** |

**Verdict: GO.** The 3-pair LETF combo improves combined Sharpe by +0.52 over the Phase 68
baseline. Every year is positive. MDD improves marginally (-0.20% vs -0.21%). `make multistrat-backtest`
now uses the 3-pair LETF leg.

## Technical Notes

- SOXL/SOXX data start: 2010-03-12 (SOXL inception). Over the 2020–2026 common window,
  SOXX showed annualised vol ~40% (vs ~26% for QQQ), providing a higher gross decay premium.
- UPRO/SPY: cheapest borrow (0.5%/yr), lowest gross premium (SPY vol ~18%), but low cost and
  near-zero correlation with SOXL provides genuine diversification.
- Returns clipped to [-50%, +50%] per day per pair to handle any stale-price gaps in SOXX data.
  In practice, no day hit this ceiling over the backtest period.
- SPY is downloaded once (reused across UPRO/SPY pair and RTMV leg).

## Phase 72 Recommendation

Two candidates, ranked by expected information content:

**Option A (Recommended): Regime-filtered LETF 3-pair**
Test whether the SPY HMM bull-only filter (1× neutral/bear, 1.5× bull) improves the
3-pair LETF combo. Analogous to Phase 55 carry regime scaling. Quick to implement since
SPY rank is already computed. Hypothesis: vol is higher in bear/neutral regimes (2022) so
the pair already earns more during those periods — bear-regime scaling down might actually
hurt (similar to Phase 55's finding). Expected result: likely null, but worth 1 run to confirm.

**Option B: Carry universe expansion (SOL/AVAX perpetuals)**
Phase 59 checked in 2023 and found BNB/SOL/AVAX diluted the BTC+ETH Sharpe. Conditions
may have changed: SOL perpetual funding has been consistently positive throughout 2024–2026.
If SOL 90-day trailing carry > 5%/yr AND Sharpe (BTC+ETH+SOL) > 15.86 (current baseline),
this is a GO. Check current carry levels before committing.

**Recommendation:** Start with Option A (1–2 hours of work, definitive answer on regime-scaled LETF).
If null result, proceed to Option B (requires Binance API access for SOL funding data).
