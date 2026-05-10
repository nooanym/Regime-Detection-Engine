"""Phase 46: Vol forecasting quality test.

Compares HMM posterior-weighted vol forecast vs EWMA and historical vol
baselines using purged k-fold CV. Tests whether the engine adds vol
forecasting value beyond simple baselines.

Usage
-----
    uv run python scripts/run_vol_forecast_test.py --config configs/spy_daily.yaml
    uv run python scripts/run_vol_forecast_test.py --config configs/btc_daily.yaml \\
        --n-states 3 --horizons 5,10,21 --ann-factor 252
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from rde.config.loader import load_config
from rde.data.yfinance_source import YFinanceSource
from rde.evaluation.vol_forecasting import compare_vol_forecasters, run_vol_forecast_cv
from rde.features.pipeline import FeaturePipeline
from rde.features.returns import LogReturns, SmoothedReturns
from rde.features.volatility import RollingVolatility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 46: Vol forecasting quality test (HMM vs baselines)"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--train-bars", type=int, default=504)
    parser.add_argument("--test-bars", type=int, default=63)
    parser.add_argument("--embargo-bars", type=int, default=10)
    parser.add_argument("--ann-factor", type=int, default=252)
    parser.add_argument(
        "--horizons",
        type=str,
        default="5,10,21",
        help="Comma-separated forecast horizons in bars (default: 5,10,21)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: results/{asset}/)",
    )
    args = parser.parse_args()

    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    cfg = load_config(args.config)
    symbol = cfg.asset.symbol
    output_dir = args.output_dir or (_repo_root / "results" / f"{symbol}-vol-forecast")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = _repo_root / "results" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Phase 46 Vol Forecast Test ===")
    logger.info("Asset: %s  n_states=%d  horizons=%s", symbol, args.n_states, horizons)

    # Load data
    source = YFinanceSource(cache_dir=cache_dir)
    df_raw = source.load(cfg.asset.symbol, cfg.asset.period, cfg.asset.interval)

    pipeline = FeaturePipeline([
        LogReturns(),
        RollingVolatility(window=20),
        SmoothedReturns(window=5),
    ])
    df = pipeline.transform(df_raw)
    feature_cols = pipeline.output_columns
    returns_col = "log_return"

    logger.info("Data shape after features: %s", df.shape)

    results = run_vol_forecast_cv(
        df,
        feature_cols=feature_cols,
        returns_col=returns_col,
        n_states=args.n_states,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        embargo_bars=args.embargo_bars,
        horizons=horizons,
        ann_factor=args.ann_factor,
        train_kwargs={"n_restarts": args.n_restarts},
    )

    comparison = compare_vol_forecasters(results)

    print("\n" + "=" * 70)
    print(f"VOL FORECASTING TEST COMPLETE — {symbol}")
    print("=" * 70)
    print(comparison.to_string(index=False))
    print("=" * 70)

    # Check if HMM beats EWMA on MSE at any horizon
    for h in horizons:
        h_df = comparison[comparison["horizon_bars"] == h].set_index("method")
        if "hmm" in h_df.index and "ewma" in h_df.index:
            hmm_mse = h_df.loc["hmm", "mse"]
            ewma_mse = h_df.loc["ewma", "mse"]
            hist_mse = h_df.loc["hist_vol", "mse"] if "hist_vol" in h_df.index else float("inf")
            if hmm_mse < ewma_mse:
                logger.info(
                    "h=%d: HMM MSE=%.6f < EWMA MSE=%.6f — HMM BETTER",
                    h, hmm_mse, ewma_mse,
                )
            else:
                logger.info(
                    "h=%d: HMM MSE=%.6f >= EWMA MSE=%.6f — HMM NOT better",
                    h, hmm_mse, ewma_mse,
                )
            if hmm_mse < hist_mse:
                logger.info(
                    "h=%d: HMM MSE=%.6f < HistVol MSE=%.6f — HMM BETTER than hist",
                    h, hmm_mse, hist_mse,
                )

    # Write report
    run_date = pd.Timestamp.now().strftime("%Y%m%d")
    report_path = output_dir / f"vol_forecast_report_{run_date}.md"
    lines = [
        f"# Phase 46: Vol Forecasting Quality — {symbol}",
        "",
        f"**Date:** {run_date}",
        f"**Asset:** {symbol}",
        f"**n_states:** {args.n_states}  **train_bars:** {args.train_bars}",
        f"**Horizons:** {horizons}  **ann_factor:** {args.ann_factor}",
        "",
        "## Results",
        "",
        "> Sorted by horizon and MSE (lower MSE = better forecast).",
        "",
    ]

    # Format table
    headers = list(comparison.columns)
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in comparison.iterrows():
        cells = []
        for col in headers:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## GO/NO-GO (HMM vs EWMA)",
        "",
    ]
    for h in horizons:
        h_df = comparison[comparison["horizon_bars"] == h].set_index("method")
        if "hmm" in h_df.index and "ewma" in h_df.index:
            hmm_mse = h_df.loc["hmm", "mse"]
            ewma_mse = h_df.loc["ewma", "mse"]
            verdict = "PASS" if hmm_mse < ewma_mse else "FAIL"
            lines.append(
                f"- h={h}: HMM MSE={hmm_mse:.6f} vs EWMA={ewma_mse:.6f} → **{verdict}**"
            )

    lines += ["", "---", f"*Generated by `scripts/run_vol_forecast_test.py` on {run_date}.*", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written → %s", report_path)
    print(f"\nReport:  {report_path}")


if __name__ == "__main__":
    main()
