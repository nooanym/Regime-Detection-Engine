"""Multi-asset regime-conditional allocation backtest script.

Downloads daily OHLCV for multiple assets (default: BTC-USD, ETH-USD, SPY, GLD),
builds per-asset feature DataFrames, aligns on common trading dates, runs the
regime-conditional MVO allocation and two baselines, then writes a comparison
table and Markdown report.

Usage
-----
    uv run python scripts/run_multi_asset_backtest.py
    uv run python scripts/run_multi_asset_backtest.py \\
        --assets BTC-USD,ETH-USD,SPY,GLD \\
        --n-states 8 --n-restarts 3 --lookback-bars 504 --target-vol 0.10

Outputs
-------
    results/multi_asset/backtest_{YYYYMMDD}.parquet   — weights + aligned returns
    results/multi_asset/report_{YYYYMMDD}.md          — Markdown report with GO/NO-GO
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src/ to path when run as a script outside the installed package.
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from rde.analysis.multi_asset_allocation import (
    MultiAssetConfig,
    MultiAssetResult,
    compare_allocations,
    equal_weight_baseline,
    global_min_var_baseline,
    run_multi_asset_allocation,
)
from rde.data.yfinance_source import YFinanceSource
from rde.features.pipeline import FeaturePipeline
from rde.features.returns import LogReturns, SmoothedReturns
from rde.features.volatility import RollingVolatility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature pipeline factory
# ---------------------------------------------------------------------------


def _build_pipeline() -> FeaturePipeline:
    """Return the standard daily feature pipeline.

    Columns produced: ``log_return``, ``volatility_w20``,
    ``smoothed_return_w5``.
    """
    return FeaturePipeline([
        LogReturns(),
        RollingVolatility(window=20),
        SmoothedReturns(window=5),
    ])


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_asset_features(
    symbol: str,
    source: YFinanceSource,
    pipeline: FeaturePipeline,
) -> pd.DataFrame:
    """Load daily OHLCV for *symbol* and apply the feature pipeline.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. ``"BTC-USD"``.
    source : YFinanceSource
        Configured data source with caching.
    pipeline : FeaturePipeline
        Feature pipeline to apply after loading.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame with NaN rows dropped.  Index is a DatetimeIndex.

    Raises
    ------
    RuntimeError
        If yfinance returns empty data for the symbol.
    """
    logger.info("Loading %s (period=max, interval=1d)…", symbol)
    raw = source.load(symbol, period="max", interval="1d")
    logger.info("  %s: %d raw bars (%s → %s)", symbol, len(raw),
                raw.index[0].date(), raw.index[-1].date())
    feat_df = pipeline.transform(raw).dropna()
    logger.info("  %s: %d bars after feature pipeline + dropna", symbol, len(feat_df))
    return feat_df


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def _align_features(
    feat_dfs: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.core.indexes.datetimes.DatetimeIndex]:
    """Intersect all feature DataFrames to a common trading-date index.

    Parameters
    ----------
    feat_dfs : dict[str, pd.DataFrame]
        Keyed by asset symbol.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], DatetimeIndex]
        Aligned feature DataFrames and the common index.
    """
    common_idx = None
    for feat_df in feat_dfs.values():
        if common_idx is None:
            common_idx = feat_df.index
        else:
            common_idx = common_idx.intersection(feat_df.index)

    if common_idx is None or len(common_idx) == 0:
        raise RuntimeError(
            "No common trading dates found across assets. "
            "Check that all symbols return overlapping date ranges."
        )

    aligned = {symbol: df.loc[common_idx] for symbol, df in feat_dfs.items()}
    logger.info(
        "Common index: %d bars (%s → %s)",
        len(common_idx),
        common_idx[0].date(),
        common_idx[-1].date(),
    )
    return aligned, common_idx


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table string."""
    headers = list(df.columns)
    sep = [":---" if df[c].dtype == object else "---:" for c in headers]
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join(sep) + " |")
    for _, row in df.iterrows():
        cells = []
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_report(
    report_path: Path,
    assets: list[str],
    config: MultiAssetConfig,
    feat_dfs_aligned: dict[str, pd.DataFrame],
    common_idx: pd.DatetimeIndex,
    comparison_df: pd.DataFrame,
    run_date: str,
) -> None:
    """Write a Markdown report to *report_path*.

    Parameters
    ----------
    report_path : Path
    assets : list[str]
    config : MultiAssetConfig
    feat_dfs_aligned : dict[str, pd.DataFrame]
    common_idx : pd.DatetimeIndex
    comparison_df : pd.DataFrame
        Output of :func:`compare_allocations`.
    run_date : str
        ``YYYYMMDD`` string.
    """
    # GO/NO-GO verdict: regime_mvo Sharpe must exceed both baselines.
    regime_rows = comparison_df[comparison_df["name"] == "regime_mvo"]
    ew_rows = comparison_df[comparison_df["name"] == "equal_weight"]
    mv_rows = comparison_df[comparison_df["name"] == "global_min_var"]

    regime_sharpe = float(regime_rows["sharpe"].iloc[0]) if len(regime_rows) else float("-inf")
    ew_sharpe = float(ew_rows["sharpe"].iloc[0]) if len(ew_rows) else float("-inf")
    mv_sharpe = float(mv_rows["sharpe"].iloc[0]) if len(mv_rows) else float("-inf")
    baseline_max = max(ew_sharpe, mv_sharpe)

    verdict = "PASS" if regime_sharpe > baseline_max else "FAIL"

    if verdict == "PASS":
        next_step = (
            "Regime-conditional MVO clears the baseline bar. "
            "Proceed to Phase 37-style purged CV validation on the multi-asset portfolio "
            "to stress-test regime stability and cost sensitivity before live deployment."
        )
    else:
        next_step = (
            "Regime-conditional MVO does not beat the best passive baseline. "
            "Investigate: (1) whether n_states or lookback_bars is misspecified; "
            "(2) whether any single asset is dragging performance; "
            "(3) whether the vol-target overlay is distorting weights. "
            "If no fix is found after targeted probes, file this as a negative result "
            "and consider the Options/vol-forecasting direction (CLAUDE.md §9)."
        )

    # Data summary table.
    data_rows = []
    for symbol in assets:
        df = feat_dfs_aligned[symbol]
        data_rows.append({
            "asset": symbol,
            "n_bars": len(df),
            "date_from": str(df.index[0].date()),
            "date_to": str(df.index[-1].date()),
        })
    data_df = pd.DataFrame(data_rows)

    lines = [
        f"# Multi-Asset Regime Backtest — {run_date}",
        "",
        "## 1. Run metadata",
        "",
        f"| Parameter | Value |",
        f"|:---|---:|",
        f"| run_date | {run_date} |",
        f"| assets | {', '.join(assets)} |",
        f"| n_bars_common | {len(common_idx)} |",
        f"| date_from | {common_idx[0].date()} |",
        f"| date_to | {common_idx[-1].date()} |",
        f"| ann_factor | {config.ann_factor} |",
        f"| rebalance_bars | {config.rebalance_bars} |",
        f"| lookback_bars | {config.lookback_bars} |",
        f"| cov_window_bars | {config.cov_window_bars} |",
        f"| n_states | {config.n_states} |",
        f"| n_restarts | {config.n_restarts} |",
        f"| transaction_cost | {config.transaction_cost} |",
        f"| target_vol | {config.target_vol} |",
        f"| mvo_kind | {config.mvo_kind} |",
        f"| min_weight | {config.min_weight} |",
        f"| max_weight | {config.max_weight} |",
        "",
        "## 2. Data summary",
        "",
        _df_to_markdown_table(data_df),
        "",
        "## 3. Strategy comparison",
        "",
        "> Sorted by Sharpe ratio (descending).",
        "",
        _df_to_markdown_table(comparison_df),
        "",
        "## 4. GO/NO-GO verdict",
        "",
        f"**Verdict: {verdict}**",
        "",
        (
            f"regime_mvo Sharpe = {regime_sharpe:.3f} "
            f"vs equal_weight = {ew_sharpe:.3f}, "
            f"global_min_var = {mv_sharpe:.3f}."
        ),
        (
            "PASS criterion: regime_mvo Sharpe must exceed "
            "max(equal_weight Sharpe, global_min_var Sharpe)."
        ),
        "",
        "## 5. Next steps",
        "",
        next_step,
        "",
        "---",
        f"*Generated by `scripts/run_multi_asset_backtest.py` on {run_date}.*",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written → %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the multi-asset allocation backtest."""
    parser = argparse.ArgumentParser(
        description="Walk-forward regime-conditional multi-asset allocation backtest"
    )
    parser.add_argument(
        "--assets",
        type=str,
        default="BTC-USD,ETH-USD,SPY,GLD",
        help="Comma-separated list of asset symbols (default: BTC-USD,ETH-USD,SPY,GLD)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/multi_asset"),
        help="Output directory (default: results/multi_asset)",
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=8,
        help="Number of HMM states per asset (default: 8)",
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=3,
        help="HMM training restarts per asset (default: 3)",
    )
    parser.add_argument(
        "--lookback-bars",
        type=int,
        default=504,
        help="HMM training window in bars (default: 504 ≈ 2yr daily)",
    )
    parser.add_argument(
        "--target-vol",
        type=float,
        default=0.10,
        help="Annualised portfolio vol target; 0 = disable (default: 0.10)",
    )
    args = parser.parse_args()

    assets = [s.strip() for s in args.assets.split(",") if s.strip()]
    if len(assets) < 2:
        parser.error("--assets must list at least 2 symbols.")

    output_dir: Path = _repo_root / args.output_dir if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _repo_root / "results" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    run_date = pd.Timestamp.now().strftime("%Y%m%d")

    logger.info("=== Multi-Asset Backtest starting ===")
    logger.info("Assets: %s", assets)
    logger.info("n_states=%d  n_restarts=%d  lookback_bars=%d  target_vol=%s",
                args.n_states, args.n_restarts, args.lookback_bars, args.target_vol)

    # --- 1. Load data and build features ---
    source = YFinanceSource(cache_dir=cache_dir)
    pipeline = _build_pipeline()

    feat_dfs: dict[str, pd.DataFrame] = {}
    for symbol in assets:
        feat_dfs[symbol] = _load_asset_features(symbol, source, pipeline)

    # --- 2. Align on common trading dates ---
    feat_dfs_aligned, common_idx = _align_features(feat_dfs)

    # --- 3. Build the aligned returns DataFrame ---
    asset_returns = pd.DataFrame(
        {symbol: feat_dfs_aligned[symbol]["log_return"] for symbol in assets},
        index=common_idx,
    )

    # --- 4. Build MultiAssetConfig ---
    target_vol_param: float | None = args.target_vol if args.target_vol > 0.0 else None
    config = MultiAssetConfig(
        ann_factor=252,
        rebalance_bars=21,
        lookback_bars=args.lookback_bars,
        cov_window_bars=63,
        n_states=args.n_states,
        n_restarts=args.n_restarts,
        transaction_cost=0.001,
        target_vol=target_vol_param,
        mvo_kind="max_sharpe",
        min_weight=0.0,
        max_weight=0.60,
    )

    # --- 5. Run regime-conditional MVO ---
    logger.info(
        "Running regime-conditional MVO (%d assets, %d bars)…",
        len(assets), len(common_idx),
    )
    result = run_multi_asset_allocation(
        asset_returns,
        feat_dfs_aligned,
        config=config,
    )
    logger.info(
        "regime_mvo: Sharpe=%.3f  ann_return=%.3f  ann_vol=%.3f  MDD=%.3f  rebalances=%d",
        result.sharpe, result.ann_return, result.ann_vol,
        result.max_drawdown, len(result.rebalance_dates),
    )

    # --- 6. Equal-weight baseline ---
    logger.info("Running equal-weight baseline…")
    ew = equal_weight_baseline(asset_returns, ann_factor=252, transaction_cost=0.001)
    logger.info(
        "equal_weight: Sharpe=%.3f  ann_return=%.3f  ann_vol=%.3f  MDD=%.3f",
        ew.sharpe, ew.ann_return, ew.ann_vol, ew.max_drawdown,
    )

    # --- 7. Global min-var baseline ---
    logger.info("Running global min-var baseline…")
    mv = global_min_var_baseline(
        asset_returns,
        ann_factor=252,
        rebalance_bars=21,
        cov_window_bars=63,
        transaction_cost=0.001,
    )
    logger.info(
        "global_min_var: Sharpe=%.3f  ann_return=%.3f  ann_vol=%.3f  MDD=%.3f",
        mv.sharpe, mv.ann_return, mv.ann_vol, mv.max_drawdown,
    )

    # --- 8. Comparison table ---
    comparison_df = compare_allocations({
        "regime_mvo": result,
        "equal_weight": ew,
        "global_min_var": mv,
    })

    # Print to stdout.
    print("\n" + "=" * 70)
    print(f"MULTI-ASSET BACKTEST COMPLETE — {run_date}")
    print(f"Assets: {', '.join(assets)}")
    print(f"Common bars: {len(common_idx)}  ({common_idx[0].date()} → {common_idx[-1].date()})")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print("=" * 70)

    # --- 9. Save weights + returns parquet ---
    backtest_path = output_dir / f"backtest_{run_date}.parquet"
    weights_out = result.weights.copy()
    weights_out.columns = [f"weight_{c}" for c in weights_out.columns]
    returns_out = asset_returns.copy()
    returns_out.columns = [f"return_{c}" for c in returns_out.columns]
    port_ret_series = result.portfolio_returns.rename("portfolio_return")
    ew_ret_series = ew.portfolio_returns.rename("ew_portfolio_return")
    mv_ret_series = mv.portfolio_returns.rename("mv_portfolio_return")

    combined = pd.concat(
        [weights_out, returns_out, port_ret_series, ew_ret_series, mv_ret_series],
        axis=1,
    )
    combined.to_parquet(backtest_path)
    logger.info("Weights + returns saved → %s", backtest_path)

    # --- 10. Write Markdown report ---
    report_path = output_dir / f"report_{run_date}.md"
    _write_report(
        report_path=report_path,
        assets=assets,
        config=config,
        feat_dfs_aligned=feat_dfs_aligned,
        common_idx=common_idx,
        comparison_df=comparison_df,
        run_date=run_date,
    )

    print(f"\nParquet:  {backtest_path}")
    print(f"Report:   {report_path}")
    print()


if __name__ == "__main__":
    main()
