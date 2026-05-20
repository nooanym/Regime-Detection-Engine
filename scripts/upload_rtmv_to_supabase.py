"""Upload RTMV 5-asset backtest results to Supabase.

Reads results/rtmv_live_5asset/snapshots.parquet (and optionally
fills.parquet) and upserts to ``rtmv_runs``, ``rtmv_snapshots``,
and ``rtmv_fills``.

Usage
-----
# Dry-run:
uv run python scripts/upload_rtmv_to_supabase.py --dry-run

# Upload with service-role key:
uv run python scripts/upload_rtmv_to_supabase.py --service-key <KEY>

# Or via env var:
SUPABASE_SERVICE_KEY=<key> uv run python scripts/upload_rtmv_to_supabase.py

Notes
-----
- The anon key is read-only (RLS: SELECT only).
  Writes require the service-role key from
  Supabase dashboard → Settings → API → service_role secret.
- All upserts are idempotent (merge on primary key / unique constraint).
- RTMV fills are large (~1167 rows for the 5-asset backtest); they are
  batched in BATCH_SIZE chunks.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPABASE_URL = "https://asegrralbtwjbuurtbvo.supabase.co"
SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFzZWdycmFsYnR3amJ1dXJ0YnZvIiwicm9sZSI"
    "6ImFub24iLCJpYXQiOjE3Nzg2MDY1NjIsImV4cCI6MjA5NDE4MjU2Mn0"
    ".NzWgh6LjkZigwyPToxhCCW0mRWc6km9lEdNpYySJAYo"
)

RESULTS_DIR = Path(__file__).parent.parent / "results" / "rtmv_live_5asset"
BATCH_SIZE = 500

RTMV_ASSETS = ["SPY", "GLD", "SHY", "IEF", "TLT"]

# ---------------------------------------------------------------------------
# Optional imports
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


def _build_run_row(snapshots: pd.DataFrame, run_id: str, mode: str) -> dict:
    """Compute run-level aggregate stats."""
    equity = snapshots["equity"].dropna()
    n_bars = len(equity)

    # Sharpe from daily equity pct-change
    daily_r = equity.pct_change().dropna()
    sharpe = float(daily_r.mean() / daily_r.std() * math.sqrt(252)) if daily_r.std() > 0 else float("nan")

    initial = float(equity.iloc[0])
    final = float(equity.iloc[-1])
    ann_return = float((final / initial) ** (252 / n_bars) - 1) if n_bars > 0 else float("nan")

    peak = equity.cummax()
    mdd = float(((peak - equity) / peak).max())

    started_at = snapshots.index[0]
    ended_at = snapshots.index[-1]

    return {
        "id": run_id,
        "mode": mode,
        "assets": RTMV_ASSETS,
        "lambda_tilt": 0.05,
        "n_states": 3,
        "lookback_bars": 504,
        "rebalance_bars": 21,
        "drawdown_halt": 0.25,
        "started_at": started_at.isoformat() if hasattr(started_at, "isoformat") else str(started_at),
        "ended_at": ended_at.isoformat() if hasattr(ended_at, "isoformat") else str(ended_at),
        "final_sharpe": round(sharpe, 4),
        "final_mdd": round(mdd, 6),
        "n_bars": n_bars,
        "notes": (
            "Phase 57 5-asset backtest: spy_rank_bull lambda, "
            "momentum_tilt_scale=0.03, halt=25%, SPY/GLD/SHY/IEF/TLT"
        ),
    }


def _snapshot_rows(run_id: str, snapshots: pd.DataFrame) -> list[dict]:
    """Convert snapshots DataFrame to list of dicts for Supabase upsert."""
    weight_cols = [f"weight_{a}" for a in RTMV_ASSETS if f"weight_{a}" in snapshots.columns]
    target_cols = [f"target_{a}" for a in RTMV_ASSETS if f"target_{a}" in snapshots.columns]

    rows = []
    for ts, row in snapshots.iterrows():
        bar_date = ts.date().isoformat() if hasattr(ts, "date") else str(ts)

        weights = {a: (round(float(row[f"weight_{a}"]), 6) if not math.isnan(float(row[f"weight_{a}"])) else None)
                   for a in RTMV_ASSETS if f"weight_{a}" in row.index}
        targets = {a: (round(float(row[f"target_{a}"]), 6) if not math.isnan(float(row[f"target_{a}"])) else None)
                   for a in RTMV_ASSETS if f"target_{a}" in row.index}

        def _safe(v) -> float | None:
            if v is None:
                return None
            fv = float(v)
            return None if math.isnan(fv) else round(fv, 4)

        rows.append({
            "run_id": run_id,
            "bar_date": bar_date,
            "equity": _safe(row.get("equity")),
            "cash": _safe(row.get("cash")),
            "drawdown": _safe(row.get("drawdown")),
            "peak_equity": _safe(snapshots.loc[:ts, "equity"].max()),
            "is_halted": bool(row.get("is_halted", False)),
            "rebalanced": bool(row.get("rebalance", False)),
            "n_rebalances": int(row.get("n_rebalances", 0)),
            "weights": weights,
            "target_weights": targets,
        })
    return rows


def _fill_rows(run_id: str, fills: pd.DataFrame) -> list[dict]:
    """Convert fills DataFrame to list of dicts."""
    rows = []
    for _, row in fills.iterrows():
        ts = row["timestamp"]
        fill_at = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        rows.append({
            "run_id": run_id,
            "fill_at": fill_at,
            "asset": str(row["asset"]),
            "side": str(row["side"]),
            "quantity": round(float(row["quantity"]), 6),
            "price": round(float(row["price"]), 4),
            "slippage": round(float(row["slippage"]), 6),
            "notional": round(float(row["notional"]), 4),
        })
    return rows


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


def _upsert_supabase_py(service_key: str, run_row: dict, snapshot_rows: list[dict], fill_rows: list[dict]) -> None:
    client = _create_client(SUPABASE_URL, service_key)

    log.info("Upserting run row: %s", run_row["id"])
    client.table("rtmv_runs").upsert(run_row).execute()

    log.info("Upserting %d snapshot rows…", len(snapshot_rows))
    for i in range(0, len(snapshot_rows), BATCH_SIZE):
        batch = snapshot_rows[i : i + BATCH_SIZE]
        client.table("rtmv_snapshots").upsert(batch).execute()
        log.info("  snapshots batch %d/%d", i // BATCH_SIZE + 1, math.ceil(len(snapshot_rows) / BATCH_SIZE))

    if fill_rows:
        log.info("Inserting %d fill rows…", len(fill_rows))
        for i in range(0, len(fill_rows), BATCH_SIZE):
            batch = fill_rows[i : i + BATCH_SIZE]
            client.table("rtmv_fills").upsert(batch).execute()
            log.info("  fills batch %d/%d", i // BATCH_SIZE + 1, math.ceil(len(fill_rows) / BATCH_SIZE))

    log.info("Upload complete via supabase-py.")


def _upsert_requests(service_key: str, run_row: dict, snapshot_rows: list[dict], fill_rows: list[dict]) -> None:
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }

    def _post(table: str, data: list[dict] | dict, label: str) -> None:
        resp = _requests.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=headers,
            data=json.dumps(data),
            timeout=60,
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"{label} [{resp.status_code}]: {resp.text}")

    log.info("Upserting run row: %s", run_row["id"])
    _post("rtmv_runs", run_row, "rtmv_runs upsert")

    log.info("Upserting %d snapshot rows…", len(snapshot_rows))
    n_batches = math.ceil(len(snapshot_rows) / BATCH_SIZE)
    for i in range(0, len(snapshot_rows), BATCH_SIZE):
        batch = snapshot_rows[i : i + BATCH_SIZE]
        _post("rtmv_snapshots", batch, f"rtmv_snapshots batch {i // BATCH_SIZE + 1}/{n_batches}")
        log.info("  snapshots batch %d/%d", i // BATCH_SIZE + 1, n_batches)

    if fill_rows:
        log.info("Inserting %d fill rows…", len(fill_rows))
        n_fb = math.ceil(len(fill_rows) / BATCH_SIZE)
        for i in range(0, len(fill_rows), BATCH_SIZE):
            batch = fill_rows[i : i + BATCH_SIZE]
            _post("rtmv_fills", batch, f"rtmv_fills batch {i // BATCH_SIZE + 1}/{n_fb}")
            log.info("  fills batch %d/%d", i // BATCH_SIZE + 1, n_fb)

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
        description="Upload RTMV 5-asset backtest results to Supabase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=RESULTS_DIR / "snapshots.parquet",
        help="Path to RTMV snapshots.parquet.",
    )
    parser.add_argument(
        "--fills",
        type=Path,
        default=RESULTS_DIR / "fills.parquet",
        help="Path to RTMV fills.parquet (optional — skipped if absent).",
    )
    parser.add_argument(
        "--service-key",
        default=os.getenv("SUPABASE_SERVICE_KEY", ""),
        help="Supabase service-role key (required for writes). "
             "Alternatively set SUPABASE_SERVICE_KEY env var.",
    )
    parser.add_argument(
        "--mode",
        default="backtest",
        choices=["backtest", "live"],
        help="Run mode label (default: backtest).",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Override run_id (default: rtmv_5asset_backtest_YYYYMMDD).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print first 3 rows without uploading.",
    )
    args = parser.parse_args(argv)

    # ---- load snapshots ----------------------------------------------------
    if not args.snapshots.exists():
        log.error("Snapshots file not found: %s\nRun: make backtest-rtmv-5asset", args.snapshots)
        return 1

    log.info("Reading RTMV snapshots from: %s", args.snapshots)
    snapshots = pd.read_parquet(args.snapshots)
    log.info("Loaded %d bars", len(snapshots))

    # ---- load fills (optional) ---------------------------------------------
    fills_df: pd.DataFrame | None = None
    if args.fills.exists():
        fills_df = pd.read_parquet(args.fills)
        log.info("Loaded %d fill rows from: %s", len(fills_df), args.fills)
    else:
        log.info("Fills file not found (%s) — skipping fills upload.", args.fills)

    # ---- build upload data -------------------------------------------------
    from datetime import date as _date
    today_str = _date.today().strftime("%Y%m%d")
    run_id = args.run_id or f"rtmv_5asset_{args.mode}_{today_str}"

    run_row = _build_run_row(snapshots, run_id, args.mode)
    snap_rows = _snapshot_rows(run_id, snapshots)
    fill_rows_list = _fill_rows(run_id, fills_df) if fills_df is not None else []

    log.info("Run ID: %s  Sharpe: %.4f  Bars: %d  Fills: %d",
             run_id, run_row.get("final_sharpe", float("nan")),
             run_row.get("n_bars", 0), len(fill_rows_list))

    # ---- dry-run -----------------------------------------------------------
    if args.dry_run:
        print("\n--- DRY RUN (no upload) ---\n")
        print(f"Run row:\n{json.dumps(run_row, indent=2, default=str)}\n")
        print("First 3 snapshot rows:")
        for row in snap_rows[:3]:
            print(json.dumps(row, indent=2, default=str))
        if fill_rows_list:
            print("\nFirst 3 fill rows:")
            for row in fill_rows_list[:3]:
                print(json.dumps(row, indent=2, default=str))
        print(f"\nTotal snapshot rows: {len(snap_rows)},  fill rows: {len(fill_rows_list)}")
        print(f"supabase-py available: {HAS_SUPABASE}")
        print(f"requests available:    {HAS_REQUESTS}")
        return 0

    # ---- validate service key ----------------------------------------------
    if not args.service_key:
        log.error(
            "No service-role key provided.\n"
            "  Provide --service-key <key>  OR  set SUPABASE_SERVICE_KEY env var.\n"
            "  Get your key from:\n"
            "  https://app.supabase.com/project/asegrralbtwjbuurtbvo/settings/api"
        )
        return 1

    # ---- upload ------------------------------------------------------------
    if HAS_SUPABASE:
        log.info("Using supabase-py client.")
        _upsert_supabase_py(args.service_key, run_row, snap_rows, fill_rows_list)
    elif HAS_REQUESTS:
        log.info("supabase-py not available, falling back to requests.")
        _upsert_requests(args.service_key, run_row, snap_rows, fill_rows_list)
    else:
        log.error("Neither supabase-py nor requests is available. Run: uv add supabase")
        return 1

    log.info("Done. run_id=%s", run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
