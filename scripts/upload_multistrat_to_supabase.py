"""Upload Phase 68+ multi-strategy backtest results to Supabase.

Reads the most recent (or specified) results/phase68_multistrat/returns_*.parquet,
derives equity curve, drawdown, rolling Sharpe, and cumulative return columns,
then upserts to ``multistrat_runs`` and ``multistrat_snapshots``.

Usage
-----
# Dry-run (no network calls) — verify data before uploading:
uv run python scripts/upload_multistrat_to_supabase.py --dry-run

# Upload with service-role key (required for writes):
uv run python scripts/upload_multistrat_to_supabase.py \\
    --service-key <YOUR_SERVICE_ROLE_KEY>

# Or set env var:
SUPABASE_SERVICE_KEY=<key> uv run python scripts/upload_multistrat_to_supabase.py

# Point to a specific parquet file:
uv run python scripts/upload_multistrat_to_supabase.py \\
    --parquet results/phase68_multistrat/returns_20260520.parquet \\
    --service-key <key>

Notes
-----
- The anon key (hard-coded below) has SELECT-only access via RLS.
  Writes require the service-role key from
  Supabase dashboard → Settings → API → service_role secret.
- Rows are upserted on (run_id, bar_date) so re-running is idempotent.
- Batch size is 500 rows per POST to stay within Supabase request limits.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supabase connection constants
# ---------------------------------------------------------------------------

SUPABASE_URL = "https://asegrralbtwjbuurtbvo.supabase.co"
SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFzZWdycmFsYnR3amJ1dXJ0YnZvIiwicm9sZSI"
    "6ImFub24iLCJpYXQiOjE3Nzg2MDY1NjIsImV4cCI6MjA5NDE4MjU2Mn0"
    ".NzWgh6LjkZigwyPToxhCCW0mRWc6km9lEdNpYySJAYo"
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "phase68_multistrat"
INITIAL_EQUITY = 100_000.0
BATCH_SIZE = 500

# Portfolio weights for Phase 77 baseline run
CARRY_WEIGHT = 0.80
LETF_WEIGHT = 0.15
RTMV_WEIGHT = 0.05

# ---------------------------------------------------------------------------
# Supabase client — prefer supabase-py, fall back to requests
# ---------------------------------------------------------------------------

try:
    from supabase import create_client as _create_client

    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False

try:
    import requests as _requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _find_latest_parquet(results_dir: Path) -> Path:
    """Return the most recently modified returns_*.parquet in results_dir."""
    candidates = sorted(results_dir.glob("returns_*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"No returns_*.parquet files found in {results_dir}. "
            "Run: make multistrat-backtest"
        )
    return candidates[-1]


def _build_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all dashboard columns from the raw returns DataFrame.

    Parameters
    ----------
    df:
        Raw parquet with columns [r_carry, r_letf, r_rtmv, r_combined]
        and a DatetimeIndex (daily).

    Returns
    -------
    DataFrame with all multistrat_snapshots columns (except run_id and
    created_at, which are added at upload time).
    """
    required = {"r_carry", "r_letf", "r_rtmv", "r_combined"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Returns parquet missing columns: {missing}")

    out = df[list(required)].copy()

    # ---- equity curve (compound from $100k) --------------------------------
    equity = INITIAL_EQUITY * (1.0 + out["r_combined"]).cumprod()
    out["equity"] = equity.round(4)

    # ---- drawdown (positive fraction from rolling peak) -------------------
    peak = equity.cummax()
    out["drawdown"] = ((peak - equity) / peak).round(6)

    # ---- rolling 252-bar Sharpe (annualised, std of daily returns) ---------
    r = out["r_combined"]
    rolling_mean = r.rolling(252, min_periods=63).mean()
    rolling_std = r.rolling(252, min_periods=63).std()
    # annualise: sqrt(252) * mean/std
    with np.errstate(divide="ignore", invalid="ignore"):
        rolling_sharpe = np.where(
            rolling_std > 0,
            rolling_mean / rolling_std * math.sqrt(252),
            np.nan,
        )
    out["sharpe_rolling_252"] = np.round(rolling_sharpe, 4)

    # ---- cumulative returns per leg ----------------------------------------
    for col in ("r_carry", "r_letf", "r_rtmv", "r_combined"):
        cum_col = col.replace("r_", "cum_return_")
        out[cum_col] = ((1.0 + df[col]).cumprod()).round(6)

    # ---- constant weight columns -------------------------------------------
    out["carry_weight"] = CARRY_WEIGHT
    out["letf_weight"] = LETF_WEIGHT
    out["rtmv_weight"] = RTMV_WEIGHT

    return out


def _compute_run_stats(snapshots: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """Compute aggregate stats for the multistrat_runs row."""
    r = returns["r_combined"].dropna()
    sharpe = float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else float("nan")

    final_equity = float(snapshots["equity"].iloc[-1])
    n_bars = len(r)
    ann_return = float((final_equity / INITIAL_EQUITY) ** (252 / n_bars) - 1) if n_bars > 0 else float("nan")

    peak = snapshots["equity"].cummax()
    mdd = float(((peak - snapshots["equity"]) / peak).max())

    return {
        "final_sharpe": round(sharpe, 4),
        "final_ann_return": round(ann_return, 6),
        "final_mdd": round(mdd, 6),
        "n_bars": n_bars,
    }


def _make_run_id(parquet_path: Path, mode: str = "backtest") -> str:
    """Derive a run_id string from the parquet filename, e.g. multistrat_20260520_backtest."""
    stem = parquet_path.stem  # e.g. returns_20260520
    date_part = stem.replace("returns_", "")
    return f"multistrat_{date_part}_{mode}"


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


def _rows_for_upload(run_id: str, snapshots: pd.DataFrame) -> list[dict]:
    """Convert snapshots DataFrame to list of dicts for Supabase upsert."""
    rows = []
    for bar_date, row in snapshots.iterrows():
        d: dict = {"run_id": run_id}
        # bar_date may be a Timestamp or date
        if hasattr(bar_date, "date"):
            d["bar_date"] = bar_date.date().isoformat()
        else:
            d["bar_date"] = str(bar_date)

        for col in [
            "equity", "drawdown",
            "r_carry", "r_letf", "r_rtmv", "r_combined",
            "carry_weight", "letf_weight", "rtmv_weight",
            "sharpe_rolling_252",
            "cum_return_carry", "cum_return_letf",
            "cum_return_rtmv", "cum_return_combined",
        ]:
            val = row.get(col, None)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                d[col] = None
            else:
                d[col] = float(val)
        rows.append(d)
    return rows


def _upload_via_supabase_py(
    service_key: str,
    run_id: str,
    run_row: dict,
    snapshot_rows: list[dict],
) -> None:
    """Upload using the supabase-py client (preferred)."""
    client = _create_client(SUPABASE_URL, service_key)

    log.info("Upserting run row: %s", run_id)
    client.table("multistrat_runs").upsert(run_row).execute()

    log.info("Upserting %d snapshot rows in batches of %d…", len(snapshot_rows), BATCH_SIZE)
    for i in range(0, len(snapshot_rows), BATCH_SIZE):
        batch = snapshot_rows[i : i + BATCH_SIZE]
        client.table("multistrat_snapshots").upsert(
            batch,
            on_conflict="run_id,bar_date",
        ).execute()
        log.info("  uploaded batch %d/%d", i // BATCH_SIZE + 1, math.ceil(len(snapshot_rows) / BATCH_SIZE))

    log.info("Upload complete via supabase-py.")


def _upload_via_requests(
    service_key: str,
    run_id: str,
    run_row: dict,
    snapshot_rows: list[dict],
) -> None:
    """Upload using raw HTTP requests (fallback when supabase-py unavailable)."""
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }

    # Upsert run row
    log.info("Upserting run row: %s", run_id)
    resp = _requests.post(
        f"{SUPABASE_URL}/rest/v1/multistrat_runs",
        headers=headers,
        data=json.dumps(run_row),
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"multistrat_runs upsert failed [{resp.status_code}]: {resp.text}"
        )

    # Upsert snapshots in batches
    log.info("Upserting %d snapshot rows in batches of %d…", len(snapshot_rows), BATCH_SIZE)
    n_batches = math.ceil(len(snapshot_rows) / BATCH_SIZE)
    for i in range(0, len(snapshot_rows), BATCH_SIZE):
        batch = snapshot_rows[i : i + BATCH_SIZE]
        resp = _requests.post(
            f"{SUPABASE_URL}/rest/v1/multistrat_snapshots",
            headers=headers,
            data=json.dumps(batch),
            timeout=60,
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"multistrat_snapshots batch {i // BATCH_SIZE + 1}/{n_batches} "
                f"failed [{resp.status_code}]: {resp.text}"
            )
        log.info("  uploaded batch %d/%d", i // BATCH_SIZE + 1, n_batches)

    log.info("Upload complete via requests.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Upload Phase 68+ multistrat backtest results to Supabase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Path to returns_*.parquet. Defaults to most recent file in results/phase68_multistrat/.",
    )
    parser.add_argument(
        "--service-key",
        default=os.getenv("SUPABASE_SERVICE_KEY", ""),
        help="Supabase service-role key. Required for writes. "
             "Alternatively set SUPABASE_SERVICE_KEY env var.",
    )
    parser.add_argument(
        "--mode",
        default="backtest",
        choices=["backtest", "live"],
        help="Run mode label embedded in the run_id (default: backtest).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first 3 rows and run stats without uploading.",
    )
    args = parser.parse_args(argv)

    # ---- locate parquet ----------------------------------------------------
    parquet_path: Path = args.parquet or _find_latest_parquet(RESULTS_DIR)
    log.info("Reading returns from: %s", parquet_path)
    returns = pd.read_parquet(parquet_path)
    log.info("Loaded %d bars, columns: %s", len(returns), returns.columns.tolist())

    # ---- build derived columns ---------------------------------------------
    snapshots = _build_snapshots(returns)
    run_stats = _compute_run_stats(snapshots, returns)
    run_id = _make_run_id(parquet_path, args.mode)

    log.info("Run ID: %s", run_id)
    log.info(
        "Stats — Sharpe: %.4f  AnnRet: %.2f%%  MDD: %.2f%%  Bars: %d",
        run_stats["final_sharpe"],
        run_stats["final_ann_return"] * 100,
        run_stats["final_mdd"] * 100,
        run_stats["n_bars"],
    )

    run_row: dict = {
        "id": run_id,
        "started_at": returns.index[0].isoformat() if hasattr(returns.index[0], "isoformat") else str(returns.index[0]),
        "ended_at": returns.index[-1].isoformat() if hasattr(returns.index[-1], "isoformat") else str(returns.index[-1]),
        "carry_weight": CARRY_WEIGHT,
        "letf_weight": LETF_WEIGHT,
        "rtmv_weight": RTMV_WEIGHT,
        "carry_symbols": ["BTCUSDT", "ETHUSDT"],
        "letf_pairs": ["TQQQ/QQQ", "SOXL/SOXX", "UPRO/SPY"],
        "rtmv_assets": ["SPY", "GLD", "SHY", "IEF", "TLT"],
        "notes": (
            "Phase 77: BTC+ETH carry (spot_spread_dynamic+momentum_filter+bull-only scale) | "
            "LETF 3-pair combo (TQQQ/QQQ + SOXL/SOXX + UPRO/SPY, equal-weight, always-in) | "
            "RTMV spy_rank_bull mom=0.03 (SPY/GLD/SHY/IEF/TLT)"
        ),
        **run_stats,
    }

    snapshot_rows = _rows_for_upload(run_id, snapshots)

    # ---- dry-run -----------------------------------------------------------
    if args.dry_run:
        print("\n--- DRY RUN (no upload) ---\n")
        print(f"Run row:\n{json.dumps(run_row, indent=2, default=str)}\n")
        print("First 3 snapshot rows:")
        for row in snapshot_rows[:3]:
            print(json.dumps(row, indent=2, default=str))
        print(f"\nTotal snapshot rows to upload: {len(snapshot_rows)}")
        print(f"supabase-py available: {HAS_SUPABASE}")
        print(f"requests available:    {HAS_REQUESTS}")
        print("\nTo upload, run again without --dry-run and provide --service-key <key>")
        return 0

    # ---- validate service key before attempting upload ---------------------
    if not args.service_key:
        log.error(
            "No service-role key provided.\n"
            "  The anon key only has SELECT access (RLS policy).\n"
            "  Provide --service-key <key>  OR  set SUPABASE_SERVICE_KEY env var.\n"
            "  Get your service_role key from:\n"
            "  https://app.supabase.com/project/asegrralbtwjbuurtbvo/settings/api"
        )
        return 1

    # ---- upload ------------------------------------------------------------
    if HAS_SUPABASE:
        log.info("Using supabase-py client.")
        _upload_via_supabase_py(args.service_key, run_id, run_row, snapshot_rows)
    elif HAS_REQUESTS:
        log.info("supabase-py not available, falling back to requests.")
        _upload_via_requests(args.service_key, run_id, run_row, snapshot_rows)
    else:
        log.error(
            "Neither supabase-py nor requests is available. "
            "Install one: uv add supabase  OR  uv add requests"
        )
        return 1

    log.info("Done. run_id=%s  bars=%d", run_id, len(snapshot_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
