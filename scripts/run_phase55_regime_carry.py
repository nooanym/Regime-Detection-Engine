"""
Phase 55 — Regime-Scaled Funding Carry Backtest
================================================
Wire SPY HMM dominant-state rank into CarryStrategy position sizing:
  rank 0 (bear):       0.5× qty  — funding often negative or low
  rank 1 (neutral):    1.0× qty  — baseline
  rank 2 (bull/lo-vol): 1.5× qty  — funding often 30–40% ann

Hypothesis: scaling up in bull regimes and down in bear regimes raises
the net carry collected and reduces exposure during negative-funding periods.

GO threshold: Sharpe ≥ 16.5 (vs flat-qty baseline 16.09 for ETH)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rde.models.hmm import train_hmm
from rde.trading.carry_executor import CarryStrategy


PERIODS_PER_YEAR = 3 * 365


def fetch_funding(symbol: str, start_year: int = 2020) -> pd.Series:
    """Fetch Binance perpetual funding history (reuse pattern from backtest script)."""
    import json
    from datetime import datetime, timezone

    import requests

    base = "https://fapi.binance.com/fapi/v1/fundingRate"
    records: list = []
    t = int(datetime(start_year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    while t < end_ts:
        url = f"{base}?symbol={symbol}&limit=1000&startTime={t}"
        try:
            r = requests.get(url, timeout=15, verify=True)
        except requests.exceptions.SSLError:
            r = requests.get(url, timeout=15, verify=False)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        records.extend(batch)
        t = max(d["fundingTime"] for d in batch) + 1

    seen: set = set()
    unique = []
    for d in records:
        if d["fundingTime"] not in seen:
            seen.add(d["fundingTime"])
            from datetime import timezone as tz_mod
            dt = datetime.fromtimestamp(d["fundingTime"] / 1000, tz=tz_mod.utc)
            unique.append((pd.Timestamp(dt), float(d["fundingRate"])))

    unique.sort(key=lambda x: x[0])
    idx, vals = zip(*unique)
    return pd.Series(list(vals), index=pd.DatetimeIndex(list(idx)), name=symbol)


def compute_spy_hmm_ranks(start: str = "2020-01-01") -> pd.Series:
    """Fit SPY n=3 HMM and return daily dominant-state rank series.

    Rank = 0 (lowest mean log-return state) … 2 (highest).
    """
    spy = yf.download("SPY", start=start, interval="1d", progress=False, auto_adjust=True)
    log_ret = np.log(spy["Close"] / spy["Close"].shift(1)).dropna()
    vol = log_ret.rolling(20).std().dropna()
    features = pd.concat([log_ret, vol], axis=1).dropna()
    features.columns = ["log_return", "volatility"]

    X = features.values
    model = train_hmm(X, n_states=3, n_restarts=3, seed_base=42)

    # Viterbi decode
    X_scaled = model.scaler.transform(X)
    states = model.hmm.predict(X_scaled)

    # Rank states by mean log-return (0 = lowest = bear)
    means = model.hmm.means_[:, 0]  # first feature is log_return
    rank_map = {int(s): int(r) for r, s in enumerate(np.argsort(means))}
    ranks = pd.Series(
        [rank_map[s] for s in states],
        index=features.index.normalize(),  # date only
        name="hmm_rank",
    )
    return ranks


def simulate_direct(
    funding: pd.Series,
    hmm_ranks: pd.Series | None,
    regime_scale: dict[int, float] | None = None,
    entry_threshold: float = 0.05,
    exit_threshold: float = -0.02,
) -> dict:
    """Direct vectorised simulation of regime-scaled carry.

    Computes net return per 8-hour period as:
        r_t = funding_rate_t × scale_t − cost_per_period
    where scale_t comes from the HMM rank at date t.

    This correctly captures the dollar-weighted benefit of scaling position
    size in bull vs bear regimes.
    """
    COST_PER_PERIOD = 0.005 / PERIODS_PER_YEAR  # ~0.5% annual friction
    ann_carry = funding * PERIODS_PER_YEAR
    rolling_carry = ann_carry.rolling(90).mean()

    # Build position flag (same entry/exit logic as CarryStrategy.step)
    position = pd.Series(0.0, index=funding.index)
    in_pos = False
    for i in range(len(funding)):
        carry_est = rolling_carry.iloc[i]
        if not in_pos and carry_est > entry_threshold:
            in_pos = True
        elif in_pos and carry_est < exit_threshold:
            in_pos = False
        position.iloc[i] = 1.0 if in_pos else 0.0

    # Resolve regime scale per funding period
    if hmm_ranks is not None and regime_scale is not None:
        # Re-index ranks to daily dates (Python date objects) for fast lookup
        ranks_by_date = {ts.date(): int(v) for ts, v in hmm_ranks.items() if not pd.isna(v)}
        scale_arr = np.array([
            regime_scale.get(ranks_by_date.get(ts.date(), -1), 1.0)
            for ts in funding.index
        ])
    else:
        scale_arr = np.ones(len(funding))

    net = position.values * scale_arr * funding.values - (position.values * COST_PER_PERIOD)
    cum = (1 + net).cumprod()
    ann_return = float(cum[-1] ** (PERIODS_PER_YEAR / len(cum)) - 1)
    ann_vol = float(net.std() * np.sqrt(PERIODS_PER_YEAR))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    roll_max = np.maximum.accumulate(cum)
    max_dd = float(((cum - roll_max) / roll_max).min())

    return {
        "sharpe": round(sharpe, 3),
        "ann_return": round(ann_return, 4),
        "max_dd": round(max_dd, 4),
        "n_periods": int((position > 0).sum()),
        "cum_return": round(float(cum[-1] - 1), 4),
        "avg_scale": round(float((position.values * scale_arr)[position.values > 0].mean()), 3) if (position > 0).any() else 1.0,
    }


def main() -> None:
    print("Phase 55 — Regime-Scaled Carry Backtest")
    print("Fetching ETH funding rates...", end=" ", flush=True)
    eth = fetch_funding("ETHUSDT", start_year=2020)
    print(f"{len(eth)} periods")

    print("Fitting SPY HMM (n=3, 3 restarts)...", end=" ", flush=True)
    ranks = compute_spy_hmm_ranks(start="2020-01-01")
    print(f"done. State rank counts: {ranks.value_counts().sort_index().to_dict()}")

    SCALE = {0: 0.5, 1: 1.0, 2: 1.5}

    print("\nRunning baseline (flat 1.0× qty)...")
    base = simulate_direct(eth, hmm_ranks=None, regime_scale=None)

    print("Running Phase 55 (0.5×/1.0×/1.5× regime-scaled)...")
    p55 = simulate_direct(eth, hmm_ranks=ranks, regime_scale=SCALE)

    # Bull-only scaling sanity check: what if we only scale up in rank-2?
    print("Running bull-only scaling (1.0×/1.0×/1.5×)...")
    bull_only = simulate_direct(eth, hmm_ranks=ranks, regime_scale={0: 1.0, 1: 1.0, 2: 1.5})

    print(f"\n{'='*60}")
    print("PHASE 55 — REGIME-SCALED CARRY RESULTS (ETH 2020–present)")
    print(f"{'='*60}")
    header = f"{'Metric':<18} {'Flat':>10} {'Regime-scaled':>14} {'Delta':>8}"
    print(header)
    print("-" * 60)
    for k, fmt in [("sharpe", ".3f"), ("ann_return", ".2%"), ("max_dd", ".2%"), ("cum_return", ".1%")]:
        bv = base[k]; pv = p55[k]; delta = pv - bv
        sign = "+" if delta >= 0 else ""
        bfmt = f"{bv:{fmt}}"; pfmt = f"{pv:{fmt}}"; dfmt = f"{sign}{delta:{fmt}}"
        print(f"  {k:<16} {bfmt:>10} {pfmt:>14} {dfmt:>8}")
    print(f"  {'avg_scale':<16} {'1.000':>10} {p55['avg_scale']:>14.3f}")
    print()
    print(f"  Bull-only (1×/1×/1.5×): Sharpe={bull_only['sharpe']:.3f}  ann={bull_only['ann_return']:.2%}")

    verdict = "GO" if p55["sharpe"] >= 16.5 else "MARGINAL" if p55["sharpe"] >= base["sharpe"] else "NO-GO"
    print(f"\nVERDICT: {verdict}  (GO threshold Sharpe ≥ 16.5)")
    print(f"Regime-scaled Sharpe: {p55['sharpe']:.3f}  vs baseline: {base['sharpe']:.3f}")

    # Save findings
    findings_dir = ROOT / "docs" / "findings"
    findings_dir.mkdir(parents=True, exist_ok=True)
    report = findings_dir / "phase55_regime_carry.md"
    report.write_text(f"""# Phase 55 — Regime-Scaled Funding Carry

## Hypothesis

Wire SPY HMM dominant-state rank into `CarryStrategy` position sizing.
In bull regimes, ETH funding rates spike to 30–40% ann; scaling up captures more.
In bear regimes, funding is low or negative; scaling down reduces exposure.

## Implementation

- Rank 0 (bear):        0.5× qty  (reduce exposure)
- Rank 1 (neutral):     1.0× qty  (baseline)
- Rank 2 (bull/lo-vol): 1.5× qty  (scale up, high carry expected)

SPY HMM state counts (2020–present): bear={ranks.value_counts().get(0,0)} neutral={ranks.value_counts().get(1,0)} bull={ranks.value_counts().get(2,0)}

## Results (ETH, 2020–present)

| Metric | Flat baseline | Regime-scaled | Delta |
|---|---|---|---|
| Sharpe | {base['sharpe']} | {p55['sharpe']} | {p55['sharpe']-base['sharpe']:+.3f} |
| Ann return | {base['ann_return']:.2%} | {p55['ann_return']:.2%} | {p55['ann_return']-base['ann_return']:+.2%} |
| Max DD | {base['max_dd']:.2%} | {p55['max_dd']:.2%} | {p55['max_dd']-base['max_dd']:+.2%} |
| Cumulative | {base['cum_return']:.2%} | {p55['cum_return']:.2%} | {p55['cum_return']-base['cum_return']:+.2%} |
| Avg scale when in market | 1.000 | {p55['avg_scale']:.3f} | — |

Bull-only variant (1×/1×/1.5×): Sharpe={bull_only['sharpe']:.3f}, ann={bull_only['ann_return']:.2%}

## Verdict: {verdict}

GO threshold: Sharpe ≥ 16.5 vs baseline {base['sharpe']:.3f}

## Risk note

The regime scaling amplifies drawdowns in bear regimes where position is reduced but
funding turns negative. The 0.5× floor prevents large losses but does not eliminate
them. Monitor BTC perp carry (currently negative) as a warning signal.
""")
    print(f"\nReport → {report}")


if __name__ == "__main__":
    main()
