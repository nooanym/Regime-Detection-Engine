"""Entry point for the crypto funding carry paper-trading strategy.

Usage
-----
Live polling (polls Binance every hour):
    uv run python scripts/run_carry_live.py --mode live

Backtest replay (uses historical Binance funding data):
    uv run python scripts/run_carry_live.py --mode backtest --start-year 2020

Signal check only (print current rates, no positions):
    uv run python scripts/run_carry_live.py --mode signal
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root without installing the package
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rde.trading.carry_executor import CarryStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)


def _compute_spy_hmm_ranks(start: str = "2020-01-01") -> "pd.Series":
    """Fit SPY n=3 HMM and return daily dominant-state rank series (0=bear, 2=bull)."""
    import pandas as pd
    import yfinance as yf
    from rde.models.hmm import train_hmm

    spy = yf.download("SPY", start=start, interval="1d", progress=False, auto_adjust=True)
    log_ret = np.log(spy["Close"] / spy["Close"].shift(1)).dropna()
    vol = log_ret.rolling(20).std().dropna()
    features = pd.concat([log_ret, vol], axis=1).dropna()
    features.columns = ["log_return", "volatility"]
    X = features.values
    model = train_hmm(X, n_states=3, n_restarts=3, seed_base=42)
    X_scaled = model.scaler.transform(X)
    states = model.hmm.predict(X_scaled)
    means = model.hmm.means_[:, 0]
    rank_map = {int(s): int(r) for r, s in enumerate(np.argsort(means))}
    return pd.Series(
        [rank_map[s] for s in states],
        index=features.index.normalize(),
        name="hmm_rank",
    )


def _fetch_backtest_data(symbols: list[str], start_year: int) -> dict:
    """Fetch historical funding rates from Binance for backtest replay.

    Parameters
    ----------
    symbols : list[str]
    start_year : int

    Returns
    -------
    dict with key "funding_series" -> dict[str, pd.Series]
    """
    # Import here so the module can be imported without requests installed
    from funding_carry_backtest import fetch_funding_rates  # type: ignore[import-not-found]

    funding_series = {}
    for sym in symbols:
        logger.info("Fetching %s funding rates from %d...", sym, start_year)
        funding_series[sym] = fetch_funding_rates(sym, start_year=start_year)
        logger.info("  -> %d periods", len(funding_series[sym]))
    return funding_series


def _print_live_signal(symbols: list[str]) -> None:
    """Print current funding rates and entry/exit signals."""
    from rde.trading.carry_executor import _fetch_latest_funding_rate, _PERIODS_PER_YEAR

    print("=== LIVE CARRY SIGNAL ===")
    for sym in symbols:
        try:
            rate = _fetch_latest_funding_rate(sym)
            ann = rate * _PERIODS_PER_YEAR
            signal = "ENTER/HOLD" if ann > 0.05 else ("EXIT" if ann < -0.02 else "HOLD (low carry)")
            print(f"  {sym:10s}: rate={rate:+.5f}  ann={ann:+.1%}  -> {signal}")
        except Exception as exc:
            print(f"  {sym:10s}: ERROR — {exc}")
    print()


def main() -> None:
    """Parse CLI arguments and run the carry strategy."""
    parser = argparse.ArgumentParser(
        description="Crypto funding carry paper-trading strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["live", "backtest", "signal"],
        default="live",
        help="Execution mode",
    )
    parser.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT",
        help="Comma-separated Binance futures symbols",
    )
    parser.add_argument("--start-year", type=int, default=2020,
                        help="Backtest start year (backtest mode only)")
    parser.add_argument("--entry-threshold", type=float, default=0.05,
                        help="Minimum annualised carry to enter (e.g. 0.05 = 5%%)")
    parser.add_argument("--exit-threshold", type=float, default=-0.02,
                        help="Annualised carry below which to exit (e.g. -0.02 = -2%%)")
    parser.add_argument("--qty", type=float, default=1.0,
                        help="Units to trade per symbol (paper only)")
    parser.add_argument("--poll-interval", type=float, default=3600.0,
                        help="Poll interval in seconds (live mode)")
    parser.add_argument("--output-dir", default="results/carry_live",
                        help="Output directory for parquet files")
    parser.add_argument("--initial-capital", type=float, default=100_000.0,
                        help="Notional capital for tracking")
    parser.add_argument(
        "--no-carry-weighted",
        action="store_true",
        default=False,
        help="Disable carry-weighted allocation (Phase 60 STRONG GO). Default: enabled.",
    )
    parser.add_argument(
        "--regime-scale",
        choices=["none", "bull-only", "full"],
        default="bull-only",
        help=(
            "SPY HMM regime scaling preset: "
            "none=flat 1×, bull-only=1×/1×/1.5× (Phase 55 GO), "
            "full=0.5×/1×/1.5× (Phase 55 NO-GO)"
        ),
    )
    parser.add_argument(
        "--momentum-filter",
        action="store_true",
        default=False,
        help=(
            "Skip periods when previous combined carry rate was ≤ 0 "
            "(Phase 75 GO: +0.44 Sharpe, MDD halved from -0.46%% to -0.22%%)."
        ),
    )
    parser.add_argument(
        "--spot-spread-weight",
        action="store_true",
        default=False,
        help=(
            "Use instantaneous ETH/BTC carry ratio for per-symbol weighting instead of "
            "rolling 90-period carry weights. ETH weight = clip(eth_rate / (eth_rate + btc_rate), "
            "0.40, 0.80). Phase 76 GO: Sharpe ~19.73, MDD ~-0.02%% (requires --momentum-filter)."
        ),
    )
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    if args.mode == "signal":
        _print_live_signal(symbols)
        return

    _SCALE_PRESETS = {
        "none": None,
        "bull-only": {0: 1.0, 1: 1.0, 2: 1.5},
        "full": {0: 0.5, 1: 1.0, 2: 1.5},
    }
    regime_scale = _SCALE_PRESETS[args.regime_scale]

    strategy = CarryStrategy(
        symbols=symbols,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        qty_per_symbol=args.qty,
        output_dir=Path(args.output_dir),
        initial_capital=args.initial_capital,
        regime_scale=regime_scale,
        carry_weighted=not args.no_carry_weighted,
        momentum_filter=args.momentum_filter,
        spot_spread_weight=args.spot_spread_weight,
    )

    if args.mode == "backtest":
        try:
            funding_series = _fetch_backtest_data(symbols, args.start_year)
        except Exception as exc:
            logger.error("Failed to fetch backtest data: %s", exc)
            sys.exit(1)
        hmm_ranks = None
        if regime_scale is not None:
            logger.info("Fitting SPY HMM (n=3) for regime scaling...")
            try:
                hmm_ranks = _compute_spy_hmm_ranks(start=f"{args.start_year}-01-01")
                logger.info("SPY ranks computed: %s", hmm_ranks.value_counts().sort_index().to_dict())
            except Exception as exc:
                logger.warning("SPY HMM failed, falling back to flat scaling: %s", exc)
        log_df = strategy.run_backtest(funding_series, hmm_ranks=hmm_ranks)
        print(f"\nBacktest complete: {len(log_df)} steps")
        if not log_df.empty:
            snap = strategy._portfolio.snapshot()
            print(f"  Open positions : {snap['open_positions']}")
            print(f"  Total funding  : {snap['total_funding_collected']:.4f}")
            print(f"  Realized P&L   : {snap['realized_pnl_usd']:.2f} USD")
        print(f"\nOutputs written to: {args.output_dir}/")

    elif args.mode == "live":
        _print_live_signal(symbols)
        if regime_scale is not None:
            logger.info("Regime scaling enabled (%s). Fitting initial SPY HMM...", args.regime_scale)
            try:
                spy_ranks = _compute_spy_hmm_ranks()
                current_rank = int(spy_ranks.iloc[-1]) if len(spy_ranks) > 0 else None
                logger.info("SPY current dominant rank: %s", current_rank)
            except Exception as exc:
                logger.warning("SPY HMM failed, live mode will use rank=None (flat 1×): %s", exc)
                current_rank = None
        else:
            current_rank = None
        logger.info("Starting live paper trading. Ctrl+C to stop.")
        try:
            strategy.run_live(poll_interval=args.poll_interval, initial_hmm_rank=current_rank)
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            snap = strategy._portfolio.snapshot()
            print(f"\nFinal snapshot: {snap}")


if __name__ == "__main__":
    main()
