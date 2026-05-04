"""BIC ceiling audit — runs select_n_states for n ∈ [2..10] on BTC, ETH, SPY.

Produces per-asset selection_table.csv and bic_curve.png, then writes
results/n_states_audit.md summarising findings.

Usage::

    uv run python scripts/run_bic_audit.py

Requires cached parquet files in results/cache/.  Run `rde run` once per
asset before this script if the cache is cold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repo root on the path
# ---------------------------------------------------------------------------
repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo / "src"))

from rde.data.cache import cache_path, read_cache
from rde.features.pipeline import FeaturePipeline
from rde.features.returns import LogReturns, SmoothedReturns
from rde.features.volatility import RollingVolatility
from rde.models.hmm import train_hmm
from rde.evaluation.persistence import empirical_dwell_times
from rde.inference.viterbi import viterbi_decode

CACHE_DIR = repo / "results" / "cache"
RESULTS_DIR = repo / "results"
ASSETS = ["BTC-USD", "ETH-USD", "SPY"]
CANDIDATE_STATES = list(range(2, 11))  # [2..10]
N_RESTARTS = 5
N_ITER = 200
SEED_BASE = 42

PIPELINE = FeaturePipeline([
    LogReturns(),
    RollingVolatility(window=24),
    SmoothedReturns(window=12),
])


def _load_features(asset: str) -> np.ndarray:
    path = cache_path(CACHE_DIR, asset, "730d", "1h")
    if not path.exists():
        raise FileNotFoundError(
            f"Cache missing for {asset}: {path}\n"
            "Run `uv run rde run --config configs/{asset_lower}.yaml` first."
        )
    raw = read_cache(path)
    df = PIPELINE.transform(raw).dropna()
    X = df[["log_return", "volatility_w24", "smoothed_return_w12"]].values
    return X, df


def _median_dwell(X: np.ndarray, fitted) -> dict[int, float]:
    from sklearn.preprocessing import StandardScaler
    Xs = fitted.scaler.transform(X)
    states = viterbi_decode(fitted.hmm, Xs)
    dwells = empirical_dwell_times(states)
    return {k: float(np.median(v)) for k, v in dwells.items()}


def run_asset(asset: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  {asset}  —  n_states ∈ {CANDIDATE_STATES}")
    print(f"{'='*60}")

    X, df = _load_features(asset)
    print(f"  Observations: {len(X)}")

    rows = []
    for n in CANDIDATE_STATES:
        print(f"  n={n:2d}  ", end="", flush=True)
        try:
            fitted = train_hmm(
                X,
                n_states=n,
                n_restarts=N_RESTARTS,
                n_iter=N_ITER,
                seed_base=SEED_BASE,
            )
            print(
                f"LL={fitted.log_likelihood:>12.2f}  "
                f"AIC={fitted.aic:>12.2f}  "
                f"BIC={fitted.bic:>12.2f}  "
                f"seed={fitted.seed}"
            )
            rows.append({
                "n_states": n,
                "log_likelihood": fitted.log_likelihood,
                "aic": fitted.aic,
                "bic": fitted.bic,
                "n_restarts": N_RESTARTS,
                "best_seed": fitted.seed,
                "_fitted": fitted,
            })
        except Exception as exc:
            print(f"FAILED: {exc}")
            rows.append({
                "n_states": n,
                "log_likelihood": float("nan"),
                "aic": float("nan"),
                "bic": float("nan"),
                "n_restarts": N_RESTARTS,
                "best_seed": -1,
                "_fitted": None,
            })

    table = pd.DataFrame(rows)
    return table, X


def _bic_minimum(table: pd.DataFrame) -> int:
    valid = table.dropna(subset=["bic"])
    return int(valid.loc[valid["bic"].idxmin(), "n_states"])


def _bic_still_declining(table: pd.DataFrame) -> bool:
    valid = table.dropna(subset=["bic"])
    bics = valid["bic"].values
    return bics[-1] < bics[-2]


def save_curve(table: pd.DataFrame, asset: str, out_dir: Path) -> Path:
    valid = table.dropna(subset=["bic"])
    n_vals = valid["n_states"].values
    bic_vals = valid["bic"].values
    aic_vals = valid["aic"].values

    best_n = _bic_minimum(table)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(n_vals, bic_vals, "o-", color="#1f77b4", linewidth=2, label="BIC")
    ax.plot(n_vals, aic_vals, "s--", color="#ff7f0e", linewidth=1.5, alpha=0.7, label="AIC")
    ax.axvline(best_n, color="#2ca02c", linestyle=":", linewidth=1.5,
               label=f"BIC min = {best_n}")
    ax.set_xlabel("n_states")
    ax.set_ylabel("Information Criterion")
    ax.set_title(f"{asset} — BIC / AIC vs n_states")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png = out_dir / "bic_curve.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


def compute_dwell_summary(table: pd.DataFrame, X: np.ndarray, n_vals: list[int]) -> dict:
    summary = {}
    for n in n_vals:
        row = table[table["n_states"] == n]
        if row.empty or row["_fitted"].values[0] is None:
            continue
        fitted = row["_fitted"].values[0]
        try:
            med_dwells = _median_dwell(X, fitted)
            summary[n] = med_dwells
        except Exception:
            pass
    return summary


def main() -> None:
    all_results: dict[str, dict] = {}

    for asset in ASSETS:
        out_dir = RESULTS_DIR / asset
        out_dir.mkdir(parents=True, exist_ok=True)

        table, X = run_asset(asset)

        # Save CSV (drop internal _fitted column)
        csv_cols = ["n_states", "log_likelihood", "aic", "bic", "n_restarts", "best_seed"]
        csv_path = out_dir / "selection_table.csv"
        table[csv_cols].to_csv(csv_path, index=False)
        print(f"\n  Saved {csv_path}")

        # BIC curve PNG
        png_path = save_curve(table, asset, out_dir)
        print(f"  Saved {png_path}")

        best_n = _bic_minimum(table)
        max_n = CANDIDATE_STATES[-1]
        still_declining = _bic_still_declining(table)

        # Dwell times at key n values
        key_ns = sorted(set([best_n, 6, 8, max_n]) & set(CANDIDATE_STATES))
        dwell_summary = compute_dwell_summary(table, X, key_ns)

        all_results[asset] = {
            "table": table[csv_cols],
            "best_n": best_n,
            "still_declining": still_declining,
            "max_n": max_n,
            "dwell_summary": dwell_summary,
            "n_obs": len(X),
        }

    # Write audit markdown
    write_audit_md(all_results)
    print(f"\n  Saved {RESULTS_DIR / 'n_states_audit.md'}")


def write_audit_md(results: dict) -> None:
    lines = [
        "# n_states BIC Audit",
        "",
        f"Run date: 2026-05-04  ",
        f"n_restarts per state: {N_RESTARTS}  ",
        f"n_iter per restart: {N_ITER}  ",
        f"Candidate range: {CANDIDATE_STATES}",
        "",
    ]

    for asset, r in results.items():
        best_n = r["best_n"]
        still = r["still_declining"]
        obs = r["n_obs"]
        table = r["table"]
        dwells = r["dwell_summary"]

        lines += [
            f"## {asset}",
            "",
            f"Observations: {obs}  ",
            f"**BIC-optimal n_states: {best_n}**  ",
            f"BIC still declining at n={r['max_n']}: {'YES — ceiling still hit, consider extending' if still else 'NO — minimum confirmed'}",
            "",
            "### Selection table",
            "",
            "| n_states | log_likelihood | AIC | BIC | best_seed |",
            "|----------|---------------|-----|-----|-----------|",
        ]
        for _, row in table.iterrows():
            marker = " ← BIC min" if int(row["n_states"]) == best_n else ""
            lines.append(
                f"| {int(row['n_states'])} | {row['log_likelihood']:.2f} | "
                f"{row['aic']:.2f} | {row['bic']:.2f} | {int(row['best_seed'])}{marker} |"
            )
        lines.append("")

        if dwells:
            lines += [
                "### Median dwell times (bars) at key n values",
                "",
                "States with median dwell < 2 bars are an overfitting signal.",
                "",
            ]
            for n, state_dwells in sorted(dwells.items()):
                min_dwell = min(state_dwells.values())
                flag = " ⚠️ min dwell < 2 bars" if min_dwell < 2.0 else ""
                dwell_str = ", ".join(
                    f"S{k}={v:.1f}" for k, v in sorted(state_dwells.items())
                )
                lines.append(f"- **n={n}**: {dwell_str}{flag}")
            lines.append("")

    lines += [
        "## Summary and production recommendations",
        "",
    ]
    for asset, r in results.items():
        best_n = r["best_n"]
        still = r["still_declining"]
        if still:
            lines.append(
                f"- **{asset}**: BIC minimum not confirmed within [2..{r['max_n']}]. "
                "Extend candidate range to [2..12] and re-run."
            )
        else:
            lines.append(
                f"- **{asset}**: BIC minimum confirmed at n={best_n}. "
                f"Set `candidate_states: [2..{max(best_n + 2, r['max_n'])}]` "
                "in production config to allow selection to find this minimum reproducibly."
            )
    lines.append("")

    (RESULTS_DIR / "n_states_audit.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
