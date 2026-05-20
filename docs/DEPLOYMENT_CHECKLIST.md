# Live Deployment Checklist

**Last validated: 2026-05-20**
**Phase: 79 (deployment hardening)**

---

## Current production configuration

- **Multi-strategy**: 80% carry + 15% LETF (3-pair) + 5% RTMV
- **Combined Sharpe**: 11.94 (Phase 77 tearsheet)
- **Combined MDD**: -0.10%
- **Individual Sharpe**: carry ~19.73 / LETF ~5.47 / RTMV ~1.001

---

## Carry leg (80%)

| Parameter | Value | Phase |
|---|---|---|
| Symbols | BTCUSDT + ETHUSDT | Phase 54 |
| Carry-weighted | YES (spot_spread_dynamic) | Phase 76 GO |
| ETH weight | clip(eth_rate / (eth_rate + btc_rate), 0.40, 0.80) | Phase 76 GO |
| Regime scale | bull-only 1x/1x/1.5x (SPY HMM rank) | Phase 55/62 GO |
| Momentum filter | YES (skip if previous combined rate <= 0) | Phase 75 GO |
| Entry threshold | 5% annualized | Phase 54 |
| Exit threshold | -2% annualized | Phase 54 |
| Friction | ~0.5%/yr per occupied period | Phase 54 |
| Robustness | permutation p=0.0000, both half-periods Sharpe >15 | Phase 78 ROBUST |

**Phase history**: 54 (base) → 55 (regime scale) → 60 (carry-weighted) → 62 (bull-only scale) → 75 (momentum filter +0.44 Sharpe) → 76 (spot-spread-weight +1.16 Sharpe) → 78 (validated robust)

**Live command**:
```bash
make carry-live-full
# = uv run python scripts/run_carry_live.py --mode live --momentum-filter --spot-spread-weight --regime-scale bull-only
```

---

## LETF leg (15%)

| Parameter | Value | Phase |
|---|---|---|
| Pairs | TQQQ/QQQ + SOXL/SOXX + UPRO/SPY (equal-weight) | Phase 70 GO |
| Filter | Always-in (regime filters confirmed NO-GO) | Phase 72 NO-GO |
| TQQQ/QQQ cost | 2.00%/yr (borrow + ER + friction) | Phase 65 |
| SOXL/SOXX cost | 3.40%/yr | Phase 70 |
| UPRO/SPY cost | 1.51%/yr | Phase 70 |
| Mechanism | delta-neutral: 3*r_1x - r_3x - cost_daily | Phase 65 |

**Phase history**: 65 (single TQQQ/QQQ pair, Sharpe 4.887) → 70 (3-pair combo, Sharpe 5.47) → 72 (regime filter NO-GO — always-in confirmed optimal)

**Live command**: No separate live mode — always-in, no polling required. Current P&L via:
```bash
make letf-pair-signal
```

---

## RTMV leg (5%)

| Parameter | Value | Phase |
|---|---|---|
| Assets | SPY / GLD / SHY / IEF / TLT | Phase 52a GO |
| Lambda | spy_rank_bull = [0.02, 0.05, 0.10] | Phase 51b GO |
| Proxy asset | SPY | Phase 51b GO |
| Momentum tilt scale | 0.03 | Phase 57 GO |
| Rebalance frequency | monthly (21-bar) | Phase 48 |
| Drawdown halt | 25% | Phase 50c GO |
| n_states | 3 | Phase 43c |
| n_restarts | 3 | Phase 48 |

**Phase history**: 45 (RTMV baseline) → 47 (CV validated) → 50 (halt=25%) → 51b (spy_rank_bull) → 52a (5-asset universe) → 57 (momentum tilt +0.028 Sharpe) → 63 (CV STRONG GO: shuffle p=0.0000)

**Live command**:
```bash
make live-rtmv
# = uv run python scripts/run_rtmv_live.py --assets SPY,GLD,SHY,IEF,TLT \
#     --lambda-tilt 0.05 --lambda-by-state-rank 0.02,0.05,0.10 \
#     --proxy-asset SPY --n-states 3 --n-restarts 3 \
#     --momentum-tilt-scale 0.03 --output-dir results/rtmv_live \
#     --mode live --poll-interval 3600
```

---

## Start/restart procedure

```bash
# 1. Start carry polling (every 8 hours via Binance funding rate schedule)
make carry-live-full

# 2. Start RTMV polling (monthly rebalance, checks every 3600s)
make live-rtmv

# 3. Check status after start
make multistrat-signal
```

---

## Health check commands

| Command | Purpose |
|---|---|
| `make multistrat-signal` | Current signals for all 3 legs (Phase 79: shows momentum filter + spot-spread status) |
| `make carry-signal` | Current BTC+ETH funding rates + carry signal |
| `make letf-pair-signal` | Current LETF pair today's P&L + YTD return |
| `make monitor` | Live RTMV paper-portfolio status (Sharpe, DD, fills) |
| `make multistrat-backtest` | Full backtest from 2020 to today (Phase 75+76+78 carry included) |

---

## Backtest benchmarks (all verified 2020-present)

| Strategy | Sharpe | Ann Return | MDD | Phase |
|---|---|---|---|---|
| Combined (80/15/5) | 11.94 | ~14.8% | -0.10% | 77 |
| Carry only | ~19.73 | ~17% | ~-0.02% | 76 |
| LETF 3-pair | ~5.47 | ~21% | -1.74% | 70 |
| RTMV | ~1.001 | ~7.3% | -15.5% | 57 |

---

## Decision log (GO verdicts feeding current config)

| Phase | Decision | Key metric |
|---|---|---|
| 57 | RTMV momentum tilt scale=0.03 | +0.028 Sharpe |
| 63 | Phase 57 CV STRONG GO | shuffle p=0.0000 |
| 60 | Carry-weighted BTC+ETH | +0.630 Sharpe |
| 62 | Bull-only regime scale | +3.4% ann return |
| 65 | LETF pair STRONG GO | Sharpe 4.887, every year positive |
| 70 | 3-pair LETF combo GO | Sharpe 5.47, +3-pair diversification |
| 75 | Lag-1 momentum filter | +0.44 Sharpe, MDD halved |
| 76 | Spot-spread-weight | +1.16 Sharpe, MDD near-zero |
| 78 | Phase 75+76 ROBUST | permutation p=0.0000, both halves >15 |

---

## Known limitations

- Carry data begins 2020 — RTMV/LETF data available 2010+ but combined backtest uses 2020 start
- RTMV equity curve loaded from `results/rtmv_live_5asset/snapshots.parquet` — stale if live-rtmv not running
- SPY HMM refitted from scratch each carry-live restart — consistent since seed=42 is fixed
- `make letf-pair-signal` fetches last 5 days from yfinance; will show "Insufficient data" on weekends

---

## Phase history summary

```
Phase 68: Unified multi-strategy (80% carry + 15% LETF + 5% RTMV) — Sharpe 11.0483
Phase 70: 3-pair LETF combo — Sharpe 11.5636
Phase 71: 3-pair LETF integrated into multistrat
Phase 75: Momentum filter on carry — carry Sharpe 18.57
Phase 76: Spot-spread-weight on carry — carry Sharpe 19.73
Phase 77: Phase 75+76 integrated into multistrat — combined Sharpe 11.9359, MDD -0.10%
Phase 78: Phase 75+76 validated ROBUST (permutation p=0.0000, half-period A=22.60/B=28.86)
Phase 79: Deployment hardening — all live targets verified, deployment checklist written
```
