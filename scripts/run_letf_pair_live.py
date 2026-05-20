"""LETF pair paper-trading monitor.

Tracks the delta-neutral SHORT TQQQ + LONG 3×QQQ position.
Phase 65 confirmed: Sharpe 4.887, Ann 17.61%, MDD -1.36% (always-in).

Usage
-----
Signal (today's P&L + YTD stats):
    uv run python scripts/run_letf_pair_live.py --mode signal

Backtest (full 2010-07-01 → present tearsheet):
    uv run python scripts/run_letf_pair_live.py --mode backtest

Monitor (poll daily, log to parquet):
    uv run python scripts/run_letf_pair_live.py --mode monitor
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKTEST_START = "2010-07-01"   # TQQQ inception 2010-02-11; buffer to 2010-07
COST_ANNUAL    = 0.015           # 1.5%/yr (borrow + ER + friction)
COST_DAILY     = COST_ANNUAL / 252
OUTPUT_DIR     = ROOT / "results" / "letf_pair_live"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Defensively flatten multi-level yfinance column index."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _download(ticker: str, period: str | None = None, start: str | None = None) -> pd.Series:
    """Return daily Close series for *ticker*."""
    kwargs: dict = dict(interval="1d", progress=False, auto_adjust=True)
    if start:
        kwargs["start"] = start
    elif period:
        kwargs["period"] = period
    df = yf.download(ticker, **kwargs)
    df = _flatten(df)
    return df["Close"].squeeze()


def _pair_returns(tqqq_close: pd.Series, qqq_close: pd.Series) -> pd.Series:
    """Compute daily log-return of SHORT TQQQ + LONG 3×QQQ."""
    r_tqqq = np.log(tqqq_close / tqqq_close.shift(1)).dropna()
    r_qqq  = np.log(qqq_close  / qqq_close.shift(1)).dropna()
    idx    = r_tqqq.index.intersection(r_qqq.index)
    r_tqqq = r_tqqq.loc[idx]
    r_qqq  = r_qqq.loc[idx]
    r_tqqq.index = r_tqqq.index.normalize()
    r_qqq.index  = r_qqq.index.normalize()
    return (3.0 * r_qqq - r_tqqq - COST_DAILY).rename("r_pair")


def _tearsheet_dict(r: pd.Series) -> dict[str, float]:
    r = r.dropna()
    cum    = (1 + r).cumprod()
    n      = len(r)
    ann_r  = float(cum.iloc[-1] ** (252 / n) - 1)
    ann_v  = float(r.std() * np.sqrt(252))
    sharpe = ann_r / ann_v if ann_v > 0 else 0.0
    roll   = np.maximum.accumulate(cum.values)
    mdd    = float(((cum.values - roll) / roll).min())
    return {"ann_return": ann_r, "ann_vol": ann_v, "sharpe": sharpe, "mdd": mdd, "n": n}


def _ytd_stats(r: pd.Series) -> dict[str, float]:
    today = pd.Timestamp.now(tz="UTC").normalize()
    ytd   = r[r.index >= f"{today.year}-01-01"]
    if ytd.empty:
        return {"cum_ytd": 0.0, "mdd_ytd": 0.0, "n_ytd": 0}
    cum   = (1 + ytd).cumprod()
    roll  = np.maximum.accumulate(cum.values)
    mdd   = float(((cum.values - roll) / roll).min())
    return {"cum_ytd": float(cum.iloc[-1] - 1), "mdd_ytd": mdd, "n_ytd": len(ytd)}


# ---------------------------------------------------------------------------
# Mode: backtest
# ---------------------------------------------------------------------------

def run_backtest() -> None:
    """Full backtest from TQQQ inception → today. Prints tearsheet."""
    logger.info("Fetching TQQQ and QQQ from %s...", BACKTEST_START)
    tqqq = _download("TQQQ", start=BACKTEST_START)
    qqq  = _download("QQQ",  start=BACKTEST_START)
    logger.info("TQQQ: %d bars  QQQ: %d bars", len(tqqq), len(qqq))

    r = _pair_returns(tqqq, qqq)
    m = _tearsheet_dict(r)

    sep = "=" * 56
    print(f"\n{sep}")
    print("  LETF PAIR BACKTEST  (SHORT TQQQ + LONG 3×QQQ, always-in)")
    print(sep)
    print(f"  Period      : {r.index[0].date()} → {r.index[-1].date()}  ({m['n']} days)")
    print(f"  Cost        : {COST_ANNUAL:.1%}/yr ({COST_DAILY*1e4:.2f} bps/day)")
    print(f"  Sharpe      : {m['sharpe']:.4f}")
    print(f"  Ann Return  : {m['ann_return']:.2%}")
    print(f"  Ann Vol     : {m['ann_vol']:.2%}")
    print(f"  Max DD      : {m['mdd']:.2%}")

    print("\n  Yearly breakdown:")
    annual = (1 + r).groupby(r.index.year).prod() - 1
    for yr, ret in annual.items():
        print(f"    {yr}  {ret:+.2%}")
    print(sep)


# ---------------------------------------------------------------------------
# Mode: signal
# ---------------------------------------------------------------------------

def run_signal() -> None:
    """Print today's pair P&L and YTD stats."""
    logger.info("Fetching last 5 days of TQQQ and QQQ...")
    tqqq_5d = _download("TQQQ", period="5d")
    qqq_5d  = _download("QQQ",  period="5d")

    if len(tqqq_5d) < 2 or len(qqq_5d) < 2:
        print("ERROR: insufficient data from yfinance (market may be closed).")
        return

    # Today's single-day return (most recent bar)
    r_tqqq_today = float(np.log(tqqq_5d.iloc[-1] / tqqq_5d.iloc[-2]))
    r_qqq_today  = float(np.log(qqq_5d.iloc[-1]  / qqq_5d.iloc[-2]))
    r_pair_today = 3.0 * r_qqq_today - r_tqqq_today - COST_DAILY

    # YTD: fetch from start of year
    today = pd.Timestamp.now(tz="UTC").normalize()
    ytd_start = f"{today.year}-01-01"
    tqqq_ytd = _download("TQQQ", start=ytd_start)
    qqq_ytd  = _download("QQQ",  start=ytd_start)
    r_ytd    = _pair_returns(tqqq_ytd, qqq_ytd)
    stats    = _ytd_stats(r_ytd)

    print()
    print("=== LETF PAIR SIGNAL ===")
    print(f"  Date:              {today.date()}")
    print(f"  TQQQ return today: {r_tqqq_today:+.2%}")
    print(f"  QQQ  return today: {r_qqq_today:+.2%}")
    print(f"  Pair P&L today:    {r_pair_today:+.2%}")
    print(f"  Cum return YTD:    {stats['cum_ytd']:+.2%}  ({stats['n_ytd']} days)")
    print(f"  Max DD (YTD):      {stats['mdd_ytd']:.2%}")
    print(f"  Status:            IN POSITION (always)")
    print()


# ---------------------------------------------------------------------------
# Mode: monitor
# ---------------------------------------------------------------------------

def run_monitor(output_dir: Path, poll_interval: float = 86400.0) -> None:
    """Poll daily, compute r_pair, append to parquet log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "daily_log.parquet"

    # Load existing log or start fresh
    existing: pd.DataFrame
    if log_path.exists():
        existing = pd.read_parquet(log_path)
        logger.info("Loaded %d existing rows from %s", len(existing), log_path)
    else:
        existing = pd.DataFrame(
            columns=["date", "r_tqqq", "r_qqq", "r_pair", "cum_return", "drawdown"]
        )

    logger.info("Monitor mode started. Poll interval: %ds. Ctrl+C to stop.", int(poll_interval))

    while True:
        try:
            ts = pd.Timestamp.now(tz="UTC")
            tqqq_5d = _download("TQQQ", period="5d")
            qqq_5d  = _download("QQQ",  period="5d")

            if len(tqqq_5d) < 2 or len(qqq_5d) < 2:
                logger.warning("Insufficient data — skipping tick.")
            else:
                r_tqqq = float(np.log(tqqq_5d.iloc[-1] / tqqq_5d.iloc[-2]))
                r_qqq  = float(np.log(qqq_5d.iloc[-1]  / qqq_5d.iloc[-2]))
                r_pair = 3.0 * r_qqq - r_tqqq - COST_DAILY
                today  = ts.normalize()

                # Skip if we already logged today
                if not existing.empty and pd.Timestamp(existing["date"].iloc[-1]) >= today:
                    logger.info("Already logged for %s — skipping.", today.date())
                else:
                    # Append row and recompute cumulative stats
                    new_row = pd.DataFrame([{
                        "date": today,
                        "r_tqqq": r_tqqq,
                        "r_qqq": r_qqq,
                        "r_pair": r_pair,
                        "cum_return": float("nan"),
                        "drawdown": float("nan"),
                    }])
                    existing = pd.concat([existing, new_row], ignore_index=True)

                    # Recompute cum_return and drawdown
                    r_series = existing["r_pair"].astype(float)
                    cum = (1 + r_series).cumprod()
                    roll = np.maximum.accumulate(cum.values)
                    existing["cum_return"] = (cum - 1).values
                    existing["drawdown"]   = ((cum.values - roll) / roll)

                    existing.to_parquet(log_path, index=False)
                    logger.info(
                        "Logged %s: r_pair=%+.4f  cum=%+.2%%  dd=%.2%%",
                        today.date(), r_pair,
                        float(existing["cum_return"].iloc[-1]) * 100,
                        float(existing["drawdown"].iloc[-1]) * 100,
                    )

        except KeyboardInterrupt:
            logger.info("Monitor stopped by user.")
            break
        except Exception as exc:
            logger.error("Error in monitor tick: %s", exc)

        logger.info("Next poll in %ds...", int(poll_interval))
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LETF pair paper-trading monitor (SHORT TQQQ + LONG 3×QQQ)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "signal", "monitor"],
        default="signal",
        help="Execution mode",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Output directory for parquet log (monitor mode)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=86400.0,
        help="Poll interval in seconds (monitor mode)",
    )
    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest()
    elif args.mode == "signal":
        run_signal()
    elif args.mode == "monitor":
        run_monitor(Path(args.output_dir), poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
