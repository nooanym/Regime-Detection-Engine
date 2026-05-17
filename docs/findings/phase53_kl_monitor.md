# Phase 53: Online-Posterior KL Monitor — NO-GO

**Date:** 2026-05-17  
**Verdict: NO-GO (all variants)**

## Setup

Tested reference-posterior KL trigger at thresholds [0.0, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0] on the 5-asset spy_rank_bull baseline (SPY/GLD/SHY/IEF/TLT, n_states=3, lookback=504, rebalance=21).

Phase 53 fixed Phase 50b's core flaw: reference posterior now anchored at refit time, online forward filter updates using the SAME fitted model's `OnlineDecoder.step()`. KL then measures regime drift only, not model drift.

Confidence gate added mid-experiment: `kl_min_dominant_confidence=0.50` — only count a dominant-state change if the winning state's posterior ≥ 0.50.

## Results

| Threshold | Sharpe | MDD | Reb/yr | KL53/yr | Delta vs baseline |
|-----------|--------|-----|--------|---------|-------------------|
| calendar-only | 0.9724 | −15.6% | 10.9 | 0.0 | — |
| kl53=0.05 | 0.9538 | −15.5% | 45.7 | 45.7 | −0.0192 |
| kl53=0.10 | 0.9538 | −15.5% | 45.7 | 45.7 | −0.0192 |
| kl53=0.20 | 0.9538 | −15.5% | 45.7 | 45.7 | −0.0192 |
| kl53=0.50 | 0.9538 | −15.5% | 45.7 | 45.7 | −0.0192 |
| kl53=1.0  | 0.9538 | −15.5% | 45.7 | 45.7 | −0.0192 |
| kl53=2.0  | 0.9686 | −15.6% | 45.2 | 45.2 | −0.0044 |

## Root Cause

The `kl_min_bars_between_triggers=5` floor is the binding constraint — not the KL threshold. Every threshold from 0.05 to 2.0 produces the same trigger frequency (45.7/yr ≈ 252/5.5) because:

1. With n_states=3 on daily data, the forward filter posterior is near-uniform (~[0.40, 0.35, 0.25])
2. Even with confidence gate (≥0.50 dominant posterior), the dominant state genuinely exceeds 0.50 frequently on daily data — it just oscillates rapidly between states
3. The KL between the current forward-filter posterior and the refit-time reference posterior is ALWAYS large — the forward filter naturally drifts from any fixed reference within 1–2 weeks

The confidence gate confirmed the real issue: it's not that posteriors are uncertain — it's that the forward filter dynamics on a 3-state daily HMM produce genuine state oscillations at ~5-bar frequency that cannot be distinguished from noise by KL threshold alone.

## Why This Cannot Be Fixed

Any threshold-based KL trigger on the forward filter will either:
- Fire too frequently (threshold too low → hurts Sharpe from overtrading)
- Never fire (threshold too high → degenerates to calendar-only)

There is no threshold that produces 1–4 triggers/year with positive Sharpe improvement.

## What Would Actually Work

1. **Smoothed posteriors** (forward-backward algorithm) — but these require look-ahead, destroying causality
2. **Longer cooldown** (`kl_min_bars_between_triggers` >> 21) — but this just becomes a variant of calendar rebalancing
3. **Joint 5D HMM** (Phase 54) — a single portfolio-level HMM with cleaner state transitions may produce more stable posteriors

## Next Steps

→ Phase 54: Joint HMM on 5D return vector (SPY/GLD/SHY/IEF/TLT simultaneously)  
→ Parallel track: arbitrage/inefficiency research (cross-asset stat-arb, vol premium, crypto carry)
