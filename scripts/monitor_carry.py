"""
Live funding carry monitor — polls Binance every hour and logs to parquet.

Run in a persistent terminal:
    uv run python scripts/monitor_carry.py

Or as a background tmux session:
    tmux new -s carry && uv run python scripts/monitor_carry.py

Output file: results/carry_live/carry_log.parquet
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


SYMBOLS = ["BTCUSDT", "ETHUSDT"]
OUTPUT_DIR = Path("results/carry_live")
OUTPUT_FILE = OUTPUT_DIR / "carry_log.parquet"
POLL_INTERVAL = 3600  # 1 hour
PERIODS_PER_YEAR = 3 * 365
ENTRY_THRESHOLD = 0.05   # 5% ann carry to enter
EXIT_THRESHOLD = -0.02   # -2% ann carry to exit


def _get(url: str) -> list:
    try:
        r = requests.get(url, timeout=15, verify=True)
    except requests.exceptions.SSLError:
        r = requests.get(url, timeout=15, verify=False)
    r.raise_for_status()
    return r.json()


def fetch_latest(symbol: str) -> dict:
    rates = _get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=3")
    rates.sort(key=lambda x: x["fundingTime"])
    latest = rates[-1]
    rate = float(latest["fundingRate"])
    ann = rate * PERIODS_PER_YEAR
    next_ts = latest["fundingTime"] + 8 * 3600 * 1000
    return {
        "symbol": symbol,
        "timestamp": datetime.now(tz=timezone.utc),
        "funding_rate": rate,
        "ann_carry": ann,
        "next_payment_utc": datetime.fromtimestamp(next_ts / 1000, tz=timezone.utc),
        "signal": "ENTER" if ann > ENTRY_THRESHOLD else "EXIT" if ann < EXIT_THRESHOLD else "HOLD",
    }


def load_log() -> pd.DataFrame:
    if OUTPUT_FILE.exists():
        return pd.read_parquet(OUTPUT_FILE)
    return pd.DataFrame()


def append_log(df: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(rows)
    combined = pd.concat([df, new], ignore_index=True) if not df.empty else new
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_FILE, index=False)
    return combined


def print_status(rows: list[dict], log: pd.DataFrame) -> None:
    now = datetime.now(tz=timezone.utc)
    print(f"\n{'='*60}")
    print(f"  FUNDING CARRY MONITOR  —  {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")

    for r in rows:
        ann = r["ann_carry"]
        sig = r["signal"]
        trend = ""
        if not log.empty:
            hist = log[log["symbol"] == r["symbol"]].tail(8)
            if len(hist) > 1:
                recent_ann = hist["ann_carry"].mean() * PERIODS_PER_YEAR
                trend = f"  7d avg: {recent_ann:+.1%}"
        print(f"  {r['symbol']:10s}  rate={r['funding_rate']:+.5f}  ann={ann:+7.1%}  [{sig:5s}]{trend}")
        print(f"             next payment: {r['next_payment_utc'].strftime('%H:%M UTC')}")

    if not log.empty:
        print(f"\n  History: {len(log)} readings, first: {log['timestamp'].min()}")
        for sym in SYMBOLS:
            sub = log[log["symbol"] == sym]
            if len(sub) > 0:
                avg_ann = sub["ann_carry"].mean() * PERIODS_PER_YEAR
                pos_pct = (sub["ann_carry"] > 0).mean()
                print(f"  {sym}: mean={avg_ann:+.1%} ann | {pos_pct:.0%} positive")

    print(f"\n  Next poll: {now.strftime('%H:%M')} + {POLL_INTERVAL//60}m")
    print(f"  Log: {OUTPUT_FILE}")


def main() -> None:
    print("Funding carry monitor starting. Ctrl-C to stop.")
    print(f"Polling every {POLL_INTERVAL//60} minutes. Log → {OUTPUT_FILE}")

    while True:
        try:
            rows = [fetch_latest(sym) for sym in SYMBOLS]
            log = load_log()
            log = append_log(log, rows)
            print_status(rows, log)
        except Exception as e:
            print(f"[{datetime.now(tz=timezone.utc).strftime('%H:%M')}] Error: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
