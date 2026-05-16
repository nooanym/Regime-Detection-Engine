"""Supabase persistence layer for RTMVRebalancer (Phase 51).

Writes portfolio snapshots and trade fills to Supabase in real-time.
Falls back to a no-op gracefully if the library is not installed or
credentials are absent — the rebalancer continues in parquet-only mode.

Environment variables
---------------------
SUPABASE_URL          https://<ref>.supabase.co
SUPABASE_SERVICE_KEY  service-role key from Supabase dashboard → Settings → API

The service-role key bypasses RLS so the Python process can write
without an authenticated session.  The Lovable dashboard reads via the
anon/publishable key with the SELECT-only RLS policies applied.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

try:
    from supabase import create_client, Client as _Client

    _SUPABASE_AVAILABLE = True
except ImportError:
    _SUPABASE_AVAILABLE = False
    _Client = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SupabaseWriterConfig:
    """Configuration for the Supabase writer.

    All fields have sensible defaults.  ``url`` and ``service_key`` fall
    back to the ``SUPABASE_URL`` / ``SUPABASE_SERVICE_KEY`` environment
    variables when left empty.
    """

    url: str = ""
    service_key: str = ""
    run_id: str = ""
    mode: str = "backtest"
    assets: list[str] = field(default_factory=list)
    lambda_tilt: float = 0.05
    n_states: int = 3
    lookback_bars: int = 504
    rebalance_bars: int = 21
    drawdown_halt: float = 0.25
    notes: str = ""


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class SupabaseWriter:
    """Writes RTMV snapshots and fills to Supabase.

    Usage::

        cfg = SupabaseWriterConfig(run_id="live_20260513", mode="live", ...)
        writer = SupabaseWriter(cfg)
        # inside the rebalancer loop:
        writer.write_snapshot(bar_date, equity, ...)
        writer.write_fills([{...}, ...])
        writer.finalize_run(sharpe, mdd, final_equity)

    If ``supabase-py`` is not installed or credentials are missing,
    all methods become silent no-ops.
    """

    def __init__(self, config: SupabaseWriterConfig) -> None:
        self._enabled = False
        self._client: _Client | None = None
        self._run_id = config.run_id

        if not _SUPABASE_AVAILABLE:
            log.warning(
                "supabase-py not installed — Supabase writes disabled. "
                "Run: uv add supabase"
            )
            return

        url = config.url or os.getenv("SUPABASE_URL", "")
        key = config.service_key or os.getenv("SUPABASE_SERVICE_KEY", "")

        if not url or not key:
            log.warning(
                "SUPABASE_URL or SUPABASE_SERVICE_KEY not set — "
                "Supabase writes disabled."
            )
            return

        try:
            self._client = create_client(url, key)
            self._register_run(config)
            self._enabled = True
            log.info("SupabaseWriter connected: run_id=%s", self._run_id)
        except Exception as exc:
            log.warning("SupabaseWriter init failed (%s) — continuing without Supabase.", exc)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def run_id(self) -> str:
        return self._run_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_run(self, cfg: SupabaseWriterConfig) -> None:
        assert self._client is not None
        self._client.table("rtmv_runs").upsert({
            "id": self._run_id,
            "mode": cfg.mode,
            "assets": cfg.assets,
            "lambda_tilt": cfg.lambda_tilt,
            "n_states": cfg.n_states,
            "lookback_bars": cfg.lookback_bars,
            "rebalance_bars": cfg.rebalance_bars,
            "drawdown_halt": cfg.drawdown_halt,
            "notes": cfg.notes,
        }).execute()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_snapshot(
        self,
        bar_date: date,
        equity: float,
        cash: float,
        drawdown: float,
        peak_equity: float,
        is_halted: bool,
        rebalanced: bool,
        n_rebalances: int,
        weights: dict[str, float],
        target_weights: dict[str, float] | None = None,
    ) -> None:
        """Upsert one daily snapshot row."""
        if not self._enabled:
            return
        try:
            assert self._client is not None
            self._client.table("rtmv_snapshots").upsert({
                "run_id": self._run_id,
                "bar_date": bar_date.isoformat(),
                "equity": round(equity, 4),
                "cash": round(cash, 4),
                "drawdown": round(drawdown, 6),
                "peak_equity": round(peak_equity, 4),
                "is_halted": is_halted,
                "rebalanced": rebalanced,
                "n_rebalances": n_rebalances,
                "weights": {k: round(v, 6) for k, v in weights.items()},
                "target_weights": {k: round(v, 6) for k, v in (target_weights or {}).items()},
            }).execute()
        except Exception as exc:
            log.warning("write_snapshot failed: %s", exc)

    def write_fills(self, fills: list[dict[str, Any]]) -> None:
        """Insert a batch of fill rows.

        Each dict must contain: fill_at, asset, side, quantity, price,
        slippage, notional.  ``run_id`` is injected automatically.
        """
        if not self._enabled or not fills:
            return
        try:
            assert self._client is not None
            rows = [{"run_id": self._run_id, **f} for f in fills]
            self._client.table("rtmv_fills").insert(rows).execute()
        except Exception as exc:
            log.warning("write_fills failed: %s", exc)

    def finalize_run(self, sharpe: float, mdd: float, equity: float) -> None:
        """Stamp the run row with final metrics and ended_at timestamp."""
        if not self._enabled:
            return
        try:
            assert self._client is not None
            self._client.table("rtmv_runs").update({
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "final_sharpe": round(sharpe, 4),
                "final_mdd": round(mdd, 6),
                "final_equity": round(equity, 2),
            }).eq("id", self._run_id).execute()
            log.info("SupabaseWriter: run %s finalized (Sharpe=%.3f)", self._run_id, sharpe)
        except Exception as exc:
            log.warning("finalize_run failed: %s", exc)
