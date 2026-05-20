"""Phase 64 — LETF Decay Short: Regime + Momentum Filter.

Strategy: SHORT TQQQ + LONG 3x QQQ (delta-neutral decay capture).
Baseline always-in Sharpe ~ 1.137 from prior analysis (Phase 54/55).
GO if best filter Sharpe > always_in. Strong GO if >= 1.500.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path.cwd() / "src"))
from rde.models.hmm import train_hmm

BORROW_DAILY = 0.009 / 252
ANNUAL_COST  = 0.020          # 2%/yr total estimated costs
DAILY_COST   = ANNUAL_COST / 252

# --- 1. Download ---
print("Downloading data...")
tqqq    = yf.download("TQQQ", start="2010-07-01", interval="1d", progress=False, auto_adjust=True)
qqq     = yf.download("QQQ",  start="2010-07-01", interval="1d", progress=False, auto_adjust=True)
spy_raw = yf.download("SPY",  start="2010-07-01", interval="1d", progress=False, auto_adjust=True)
for df in (tqqq, qqq, spy_raw):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
print(f"  TQQQ: {len(tqqq)} bars  QQQ: {len(qqq)} bars  SPY: {len(spy_raw)} bars")

# --- 2. Delta-neutral decay returns: SHORT TQQQ + LONG 3x QQQ ---
r_tqqq = np.log(tqqq["Close"] / tqqq["Close"].shift(1)).dropna()
r_qqq  = np.log(qqq["Close"]  / qqq["Close"].shift(1)).dropna()
common_idx   = r_tqqq.index.intersection(r_qqq.index)
r_pair_net   = (-r_tqqq.loc[common_idx] + 3.0 * r_qqq.loc[common_idx] - DAILY_COST)
r_pair_net.index = r_pair_net.index.normalize()

# --- 3. SPY HMM ranks (n=3; rank 2 = bull) ---
print("Fitting SPY HMM (n=3, 3 restarts)...")
lr_spy  = np.log(spy_raw["Close"] / spy_raw["Close"].shift(1)).dropna()
vol_spy = lr_spy.rolling(20).std().dropna()
feats   = pd.concat([lr_spy, vol_spy], axis=1).dropna()
feats.columns = ["log_return", "volatility"]
m         = train_hmm(feats.values, n_states=3, n_restarts=3, seed_base=42)
states    = m.hmm.predict(m.scaler.transform(feats.values))
rank_map  = {int(s): int(r) for r, s in enumerate(np.argsort(m.hmm.means_[:, 0]))}
spy_ranks = pd.Series([rank_map[s] for s in states], index=feats.index.normalize())
print(f"  SPY state counts: { {r: int((spy_ranks == r).sum()) for r in range(3)} }")

# --- 4. QQQ 12m-1m momentum signal ---
qqq_lr     = np.log(qqq["Close"] / qqq["Close"].shift(1)).dropna()
qqq_lr.index = qqq_lr.index.normalize()
mom_signal = (qqq_lr.rolling(252).sum() - qqq_lr.rolling(21).sum()) > 0

# --- 5. Align signals ---
base_index        = r_pair_net.index
spy_ranks_aligned = spy_ranks.reindex(base_index).ffill()
mom_aligned       = mom_signal.reindex(base_index).ffill().fillna(False).astype(bool)

mask_regime   = (spy_ranks_aligned == 2)
mask_momentum = mom_aligned
mask_and      = mask_regime & mask_momentum
mask_or       = mask_regime | mask_momentum

strategies = {
    "always_in":            pd.Series(True,  index=base_index),
    "regime_only":          mask_regime,
    "momentum_only":        mask_momentum,
    "regime_and_momentum":  mask_and,
    "regime_or_momentum":   mask_or,
}

# --- 6. Metrics ---
def metrics(r_series: pd.Series, label: str) -> dict:
    cum = (1 + r_series.dropna()).cumprod()
    n   = len(cum)
    ann = float(cum.iloc[-1] ** (252 / n) - 1)
    vol = float(r_series.dropna().std() * np.sqrt(252))
    sharpe   = ann / vol if vol > 0 else 0.0
    rmax     = np.maximum.accumulate(cum.values)
    mdd      = float(((cum.values - rmax) / rmax).min())
    invested = float((r_series.dropna() != 0.0).mean())
    print(f"  {label:<30} Sharpe={sharpe:.3f}  ann={ann:.2%}  mdd={mdd:.2%}  invested≈{invested:.1%}")
    return {"sharpe": sharpe, "ann_return": ann, "vol": vol, "mdd": mdd, "invested": invested}

print("\n--- Phase 64 Results ---")
results = {}
for label, mask in strategies.items():
    r = r_pair_net.copy()
    r[~mask.reindex(base_index).fillna(False)] = 0.0
    results[label] = metrics(r, label)

# --- 7. Verdict ---
baseline_sharpe = results["always_in"]["sharpe"]
best_label  = max((k for k in results if k != "always_in"), key=lambda k: results[k]["sharpe"])
best_sharpe = results[best_label]["sharpe"]
improvement = best_sharpe - baseline_sharpe

print(f"\nBaseline (always_in) Sharpe : {baseline_sharpe:.3f}")
print(f"Best filter ({best_label}) Sharpe: {best_sharpe:.3f}")
print(f"Improvement                 : {improvement:+.3f}")

if best_sharpe <= baseline_sharpe:
    verdict = "NO-GO"
elif best_sharpe >= 1.500:
    verdict = "STRONG GO"
else:
    verdict = "GO"
print(f"\nVerdict: {verdict}")

# --- 8. Write findings ---
rows = "\n".join(
    f"| {k:<30} | {v['sharpe']:.3f} | {v['ann_return']:.2%} | {v['vol']:.2%} | {v['mdd']:.2%} | {v['invested']:.1%} |"
    for k, v in results.items()
)
if best_sharpe > baseline_sharpe:
    interp = (
        f"Filtering improves Sharpe vs always-in. Best: **{best_label}** "
        f"(Sharpe {best_sharpe:.3f} vs {baseline_sharpe:.3f}, improvement {improvement:+.3f})."
    )
    next_steps = (
        f"Consider deploying **{best_label}** as the LETF leg gate in the full portfolio. "
        "Validate on OOS sub-period (2020–present) and confirm MDD improvement."
    )
else:
    interp = (
        "No filter improves Sharpe vs always-in. Phase 55 finding confirmed: the delta-neutral "
        "decay harvest is not improved by regime or momentum gating. The gross alpha (~5–6%/yr) "
        "is a structural vol-path artefact that accrues in all regimes."
    )
    next_steps = "Always-in remains the recommended approach for the LETF leg. No further filter testing warranted."

md = f"""# Phase 64 — LETF Decay Short: Regime + Momentum Filter

**Date:** 2026-05-18
**Verdict: {verdict}**

## Strategy

Delta-neutral decay capture: SHORT 1 unit TQQQ + LONG 3 units QQQ (daily).
Net costs: ~{ANNUAL_COST:.1%}/yr (borrow + friction + spread).
Baseline (always-in) Sharpe: **{baseline_sharpe:.3f}**.

## Hypothesis

A combined SPY-regime (rank=2 bull) + QQQ 12m-1m momentum filter avoids periods
where TQQQ can strongly recover (momentum reversal after bear regime), preserving
the carry while shedding the worst drawdown episodes.

## Results

| Strategy | Sharpe | Ann Return | Ann Vol | MDD | Invested |
|---|---|---|---|---|---|
{rows}

## Parameters

- TQQQ/QQQ data from 2010-07-01 (TQQQ inception) to present
- Decay: -r_tqqq + 3 * r_qqq gross, net = gross - {ANNUAL_COST:.1%}/yr
- SPY HMM: n=3 states, 3 restarts, seed=42; rank 2 = bull (highest mean log-return)
- QQQ momentum: 12-month minus 1-month cumulative log-return > 0

## Verdict: {verdict}

- Baseline Sharpe: **{baseline_sharpe:.3f}**
- Best filter: **{best_label}** → Sharpe **{best_sharpe:.3f}**
- Improvement: **{improvement:+.3f}**

{interp}

## Next Steps

{next_steps}
"""
md_path = Path("docs/findings/phase64_letf_filters.md")
md_path.parent.mkdir(parents=True, exist_ok=True)
md_path.write_text(md)
print(f"\nFindings written to {md_path}")
