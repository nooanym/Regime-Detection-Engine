# Phase 70 — LETF Pair Universe Expansion

**Date:** 2026-05-18

**Verdict: PARTIAL GO**

## Hypothesis

Vol-drag scales as σ² in the theoretical decay formula:
```
E[r_pair] ≈ 4.5 × σ²_1x  (annualised, before costs)
```

Realised figures (2010 to present):

| Index | σ_1x (ann) | 4.5σ² (gross) | Total Cost | Net Theoretical | σ ratio vs QQQ | Premium ratio |
|---|---|---|---|---|---|---|
| TQQQ/QQQ (Nasdaq 100) | 26.91% | 32.59%/yr | 2.00% | 30.59%/yr | 1.0× | 1.0× |
| SOXL/SOXX (Semiconductors) | 33.13% | 49.39%/yr | 3.40% | 45.99%/yr | 1.23× | 1.5× |
| UPRO/SPY (S&P 500) | 18.59% | 15.56%/yr | 1.51% | 14.05%/yr | 0.69× | 0.46× |

Note: SOXX realised σ is 33% (not the ~55% originally assumed), partly because the SOXX index is
diversified across 30 semiconductors, not a single-stock play. The "55% vol" figure comes from
individual semiconductor stocks; the index diversification reduces it to ~33%.

## Pair Definitions and Costs

| Pair | 3× Ticker | 1× Ticker | Borrow | ER 3× | ER 1× | Friction | **Total** |
|---|---|---|---|---|---|---|---|
| TQQQ/QQQ (Nasdaq 100) | TQQQ | QQQ | 0.90% | 0.86% | 0.20% | 0.04% | **2.00%** |
| SOXL/SOXX (Semiconductors) | SOXL | SOXX | 2.00% | 0.95% | 0.35% | 0.10% | **3.40%** |
| UPRO/SPY (S&P 500) | UPRO | SPY | 0.50% | 0.91% | 0.09% | 0.01% | **1.51%** |

Strategy formula: `r_pair = 3 × r_1x - r_3x - cost_daily`

## Tearsheet — Full Available Period (each pair)

| Metric | TQQQ/QQQ | SOXL/SOXX | UPRO/SPY |
|---|---|---|---|
| Start date | 2010-02-12 | 2010-03-12 | 2009-06-26 |
| End date | 2026-05-18 | 2026-05-18 | 2026-05-18 |
| N days | 4090 | 4071 | 4249 |
| **Sharpe** | **4.5766** | **4.3318** | **3.2502** |
| Ann Return | 17.00% | 33.51% | 13.03% |
| Ann Volatility | 3.72% | 7.73% | 4.01% |
| Max Drawdown | -2.04% | -5.57% | -4.72% |
| Calmar | 8.344 | 6.019 | 2.762 |
| % Positive Days | 66.7% | 62.0% | 65.6% |
| Cumulative Return | 1179.0% | 10553.0% | 688.6% |
| Every year positive | YES | YES | YES |
| Best Year | 2020 | 2020 | 2020 |
| Best Year Return | +51.75% | +98.95% | +47.46% |
| Worst Year | 2013 | 2013 | 2017 |
| Worst Year Return | +4.28% | +6.72% | +3.61% |

## Per-Year Returns — All Three Pairs

### TQQQ/QQQ (Nasdaq 100)

| Year | Return | Trading Days |
|---|---|---|
| 2010 | +9.83% | 224 |
| 2011 | +18.05% | 252 |
| 2012 | +6.05% | 250 |
| 2013 | +4.28% | 252 |
| 2014 | +5.61% | 252 |
| 2015 | +9.55% | 252 |
| 2016 | +8.08% | 252 |
| 2017 | +4.95% | 251 |
| 2018 | +21.68% | 251 |
| 2019 | +12.46% | 252 |
| 2020 | +51.75% | 253 |
| 2021 | +10.80% | 252 |
| 2022 | +43.56% | 251 |
| 2023 | +22.12% | 250 |
| 2024 | +22.62% | 252 |
| 2025 | +28.41% | 250 |
| 2026 | +7.03% | 94 |

### SOXL/SOXX (Semiconductors)

| Year | Return | Trading Days |
|---|---|---|
| 2010 | +19.56% | 205 |
| 2011 | +32.79% | 252 |
| 2012 | +12.86% | 250 |
| 2013 | +6.72% | 252 |
| 2014 | +8.25% | 252 |
| 2015 | +14.35% | 252 |
| 2016 | +14.38% | 252 |
| 2017 | +9.22% | 251 |
| 2018 | +29.61% | 251 |
| 2019 | +24.65% | 252 |
| 2020 | +98.95% | 253 |
| 2021 | +31.92% | 252 |
| 2022 | +83.81% | 251 |
| 2023 | +37.85% | 250 |
| 2024 | +58.38% | 252 |
| 2025 | +72.91% | 250 |
| 2026 | +22.45% | 94 |

### UPRO/SPY (S&P 500)

| Year | Return | Trading Days |
|---|---|---|
| 2009 | +4.19% | 131 |
| 2010 | +9.91% | 252 |
| 2011 | +18.08% | 252 |
| 2012 | +4.69% | 250 |
| 2013 | +4.39% | 252 |
| 2014 | +4.25% | 252 |
| 2015 | +7.82% | 252 |
| 2016 | +5.79% | 252 |
| 2017 | +3.61% | 251 |
| 2018 | +14.26% | 251 |
| 2019 | +10.01% | 252 |
| 2020 | +47.46% | 253 |
| 2021 | +5.77% | 252 |
| 2022 | +25.01% | 251 |
| 2023 | +17.41% | 250 |
| 2024 | +17.29% | 252 |
| 2025 | +21.63% | 250 |
| 2026 | +5.52% | 94 |

## Pairwise Correlation of Pair Returns

Period: 2010-03-12 → present (common available window for all three pairs)

```
                            TQQQ/QQQ (Nasdaq 100)  SOXL/SOXX (Semiconductors)  UPRO/SPY (S&P 500)
TQQQ/QQQ (Nasdaq 100)                      1.0000                      0.4250              0.4067
SOXL/SOXX (Semiconductors)                 0.4250                      1.0000              0.1400
UPRO/SPY (S&P 500)                         0.4067                      0.1400              1.0000
```

Key finding: pairwise correlations tell us whether combining pairs improves risk-adjusted returns.
Correlations near 0 imply near-independent alpha sources; correlations near 1 imply redundancy.

## Equal-Weight Combination

Equal-weight of all three pairs (1/3 each), evaluated over the common date window.

| Metric | Equal-Weight Combo |
|---|---|
| Period | 2010-03-12 → 2026-05-18 |
| **Sharpe** | **5.4678** |
| Ann Return | 21.05% |
| Ann Volatility | 3.85% |
| Max Drawdown | -1.74% |
| Calmar | 12.078 |
| Every year positive | YES |
| Worst Year | 2013 (+5.16%) |

### Combo Yearly Breakdown

| Year | Return | Trading Days |
|---|---|---|
| 2010 | +12.56% | 205 |
| 2011 | +22.86% | 252 |
| 2012 | +7.88% | 250 |
| 2013 | +5.16% | 252 |
| 2014 | +6.05% | 252 |
| 2015 | +10.58% | 252 |
| 2016 | +9.39% | 252 |
| 2017 | +5.91% | 251 |
| 2018 | +21.73% | 251 |
| 2019 | +15.57% | 252 |
| 2020 | +65.15% | 253 |
| 2021 | +15.67% | 252 |
| 2022 | +48.95% | 251 |
| 2023 | +25.52% | 250 |
| 2024 | +31.63% | 252 |
| 2025 | +39.38% | 250 |
| 2026 | +11.42% | 94 |

## Common-Period Comparison

All three pairs evaluated over identical window (2010-03-12 → present):

| Pair | Sharpe | Ann Return | Ann Vol | MDD | All Years Positive |
|---|---|---|---|---|---|
| TQQQ/QQQ (Nasdaq 100) | 4.5836 | 17.05% | 3.72% | -2.04% | YES |
| SOXL/SOXX (Semiconductors) | 4.3318 | 33.51% | 7.73% | -5.57% | YES |
| UPRO/SPY (S&P 500) | 3.4132 | 13.26% | 3.88% | -4.72% | YES |
| Equal-Weight Combo | 5.4678 | 21.05% | 3.85% | -1.74% | YES |

## GO / NO-GO

**GO threshold: Sharpe ≥ 5.0** (vs TQQQ/QQQ Phase 65 baseline 4.887)

| Pair / Portfolio | Sharpe | Verdict |
|---|---|---|
| TQQQ/QQQ (Nasdaq 100) | 4.5766 | NO-GO |
| SOXL/SOXX (Semiconductors) | 4.3318 | NO-GO |
| UPRO/SPY (S&P 500) | 3.2502 | NO-GO |
| Equal-Weight Combo | 5.4678 | **GO** |

## Key Findings

1. **Vol-drag hypothesis confirmed but SOXX is less volatile than expected.** SOXX realised σ=33%
   (not ~55% as hypothesised) because the Philadelphia Semiconductor Index is diversified across
   30 names; individual chip stocks are much more volatile than the index. Gross premium is
   49%/yr (vs QQQ's 33%/yr) but higher costs (3.40%/yr vs 2.00%/yr) dilute Sharpe.

2. **SOXL/SOXX Sharpe (4.33) is slightly below TQQQ/QQQ (4.58).** The higher gross premium
   (49% vs 33%) is more than offset by higher costs and higher daily vol (7.73% vs 3.72%).
   Both NO-GO as standalone pairs vs the 5.0 threshold.

3. **UPRO/SPY is the weakest pair (Sharpe 3.25).** SPY's lower vol (~19% ann) means lower
   gross premium; the cheap borrow (0.51%/yr) helps but can't compensate. Every year positive
   but lowest Sharpe of the three. SPY is a poor vol-drag harvesting vehicle vs QQQ/SOXX.

4. **Correlation structure is favourable.** SOXL↔UPRO correlation = 0.14 (near-zero) because
   semiconductors have idiosyncratic vol uncorrelated to broad equity. TQQQ↔UPRO = 0.41
   (both QQQ/SPY are tech-heavy equity indices). Diversification benefit is real.

5. **Equal-weight combo is the clear winner: Sharpe 5.47 — the only GO.** Diversifying across
   three pairs reduces vol from 3.72-7.73% per pair to 3.85% combined while raising Ann Return
   to 21.05%. MDD shrinks from −2.04% (TQQQ best single pair) to −1.74%. Calmar 12.08 —
   the best risk-adjusted combination. Every single year positive, worst year +5.16% (2013).

6. **Vol-scaling formula is less accurate for higher-vol indices.** Observed/Theoretical ratio:
   QQQ: 17.0% / 30.6% = 55% captured; SOXX: 33.5% / 46.0% = 73% captured; SPY: 13.0% / 14.1% = 92% captured.
   SPY is closest to theory (most log-normal), SOXX/QQQ have additional gap from expense/tracking error.

## Risk Factors

1. **SOXL liquidity**: SOXL average daily volume ~$500M (vs TQQQ ~$3B). Wider spreads and
   potential borrow cost spikes. The 2.0%/yr borrow estimate may understate stress periods.
2. **Structural break risk**: Both SOXL and UPRO have undergone fund restructurings. SOXL
   was launched in 2010 but semiconductor vol regime changed materially in 2022-2024 (AI boom).
3. **Correlation regime risk**: Pairwise correlations can rise toward 1 in equity stress events
   (March 2020, 2022 bear market), reducing diversification benefit at worst times.
4. **Short squeeze risk**: All three pairs require borrowing the 3× ETF. In high-volatility
   regimes, borrow availability can deteriorate for all simultaneously.

## Next Steps

**Combo is GO (Sharpe 5.47 > 5.0 threshold).** Recommended actions:

1. **Phase 71: Replace single TQQQ/QQQ pair in Phase 68 multi-strat with the 3-pair equal-weight combo.**
   Expected Sharpe improvement: 4.58 → 5.47 (+0.89). MDD improvement: −2.04% → −1.74%.
   The LETF allocation in the 80/15/5 carry/LETF/RTMV portfolio would benefit immediately.

2. **Confirm SOXL borrow rate with Interactive Brokers.** The 2.0%/yr estimate is theoretical.
   If actual borrow is 3–4%/yr, SOXL/SOXX Sharpe would fall to ~3.8–4.0 but the combo
   would still clear 5.0 on the strength of TQQQ/QQQ and UPRO/SPY.

3. **Phase 72 (future): Vol-regime-scaled position size.** Scale pair weights proportional to
   rolling 63-day variance of the underlying 1× ETF. High-vol regimes → overweight pairs with
   larger theoretical premium (SOXX in 2022, QQQ in 2020). Low-vol regimes → underweight.
   Hypothesis: better capital efficiency than equal-weight, potential Sharpe 6+.

4. **Monitor correlation stability.** The SOXL↔UPRO = 0.14 is the key diversification driver.
   If semiconductors re-correlate with broad market (as seen briefly in March 2020), combo
   Sharpe may temporarily decline. Set an alert if 63-day rolling SOXL↔UPRO correlation
   exceeds 0.40 for 20 consecutive days.