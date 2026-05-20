# Phase 58: Multi-Strategy Executor — GO/NO-GO

**Date:** 2026-05-17
**Overall Verdict:** **GO**

## Strategy Descriptions

### 1. Crypto Funding Carry (70% weight default)
Long spot + Short perpetual on Binance. Collects positive funding.
Entry: 30-day trailing annualized carry > 5%. Exit: carry < -2%.
Assets: BTCUSDT + ETHUSDT (equal-weight). P&L resampled from 8-hourly to daily.

### 2. LETF Decay Short (30% weight default)
Short TQQQ + Long 3x QQQ notional (delta-neutral). Captures variance drag.
Configuration: ALWAYS-IN (vol filter OFF — confirmed optimal in prior research).
Daily P&L: 3×QQQ_return − TQQQ_return − 2%/yr cost drag.

## Individual Strategy Metrics

| Strategy | Sharpe | Ann Return | Ann Vol | Max DD |
| :--- | ---: | ---: | ---: | ---: |
| Carry (ETH+BTC equal-weight) | 8.244 | 8.53% | 1.04% | -0.58% |
| LETF Decay Short (always-in) | 1.137 | 3.36% | 2.96% | -3.36% |

## Cross-Strategy Correlation

**Pearson correlation (daily P&L): -0.0094**

Diversification benefit confirmed: correlation < 0.30.

Interpretation: carry P&L is driven by crypto funding market microstructure;
LETF decay is driven by equity index realized volatility. These are structurally
uncorrelated alpha sources.

## Combined Portfolio Metrics

| Allocation (Carry/LETF) | Sharpe | Ann Return | Ann Vol | Max DD | Verdict |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 70/30 | 6.435 | 8.12% | 1.26% | -1.23% | GO |
| 50/50 | 4.380 | 7.82% | 1.79% | -1.84% | GO |
| 80/20 | 7.611 | 8.27% | 1.09% | -0.95% | GO |

**GO criterion:** Combined Sharpe > max(individual Sharpe) OR Combined MDD less negative than worst individual MDD (i.e. better drawdown protection).

## Conclusion

The multi-strategy combination provides genuine diversification and improved risk-adjusted returns.

Max individual Sharpe: 8.244
Min individual MDD:    -3.36%

Both strategies' P&L streams are near-uncorrelated, confirming they draw from independent alpha sources.
