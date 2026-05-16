"""Phase 48: RTMV live deployment skeleton.

Runs the Regime-Tilted Min-Var portfolio (λ=0.05) on the SPY/GLD/TLT/IEF
universe in either backtest mode (default) or live polling mode.

Backtest mode feeds pre-downloaded historical data bar-by-bar through the
rebalancer and writes equity, weights, drawdown, and fill records.

Live mode polls yfinance every ``--poll-interval`` seconds and rebalances
on the configured cadence.

Usage
-----
    # Backtest (default): full history via yfinance
    uv run python scripts/run_rtmv_live.py

    # Backtest with custom parameters
    uv run python scripts/run_rtmv_live.py \\
        --assets SPY,GLD,TLT,IEF \\
        --lambda-tilt 0.05 \\
        --n-states 3 \\
        --n-restarts 3 \\
        --lookback-bars 504 \\
        --rebalance-bars 21 \\
        --capital 100000 \\
        --output-dir results/rtmv_live \\
        --mode backtest

    # Live polling (paper trades against real-time yfinance quotes)
    uv run python scripts/run_rtmv_live.py --mode live --poll-interval 3600

Outputs (backtest mode)
-----------------------
    results/rtmv_live/snapshots.parquet  — per-bar equity, weights, drawdown
    results/rtmv_live/fills.parquet      — every rebalance fill
    results/rtmv_live/report_{date}.md   — summary tearsheet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

# Load .env from repo root if present (allows credentials without shell exports).
_env_file = _repo_root / ".env"
if _env_file.exists():
    import os
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from rde.analysis.multi_asset_allocation import MultiAssetConfig
from rde.data.yfinance_source import YFinanceSource
from rde.features.pipeline import FeaturePipeline
from rde.features.returns import LogReturns, SmoothedReturns
from rde.features.volatility import RollingVolatility
from rde.trading.rtmv_rebalancer import RTMVRebalancer, RTMVRebalancerConfig
from rde.trading.supabase_writer import SupabaseWriter, SupabaseWriterConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------


def _build_pipeline() -> FeaturePipeline:
    """Daily feature pipeline: log_return, volatility_w20, smoothed_return_w5."""
    return FeaturePipeline([
        LogReturns(),
        RollingVolatility(window=20),
        SmoothedReturns(window=5),
    ])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_aligned_data(
    assets: list[str],
    source: YFinanceSource,
    pipeline: FeaturePipeline,
    start_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Download and align returns + features for all assets.

    Returns
    -------
    asset_returns : pd.DataFrame
        Shape (T, N), log returns on common dates.
    asset_features : dict[str, pd.DataFrame]
        Per-asset feature DataFrames on the same common dates.
    """
    feat_dfs: dict[str, pd.DataFrame] = {}
    for asset in assets:
        logger.info("Loading %s…", asset)
        raw = source.load(asset, period="max", interval="1d")
        feat_df = pipeline.transform(raw).dropna()
        feat_dfs[asset] = feat_df
        logger.info("  %s: %d bars (%s → %s)", asset, len(feat_df),
                    feat_df.index[0].date(), feat_df.index[-1].date())

    # Align to common trading dates.
    common_dates = feat_dfs[assets[0]].index
    for a in assets[1:]:
        common_dates = common_dates.intersection(feat_dfs[a].index)
    common_dates = common_dates.sort_values()

    if start_date:
        common_dates = common_dates[common_dates >= pd.Timestamp(start_date)]

    for a in assets:
        feat_dfs[a] = feat_dfs[a].loc[common_dates]

    asset_returns = pd.DataFrame(
        {a: feat_dfs[a]["log_return"] for a in assets},
        index=common_dates,
    )
    logger.info("Aligned: %d bars  (%s → %s)", len(common_dates),
                common_dates[0].date(), common_dates[-1].date())
    return asset_returns, feat_dfs


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _write_report(
    snaps: pd.DataFrame,
    rebalancer: RTMVRebalancer,
    output_dir: Path,
    date_str: str,
) -> Path:
    """Write a Markdown summary of the backtest."""
    cfg = rebalancer.config
    eq = snaps["equity"].dropna()
    ret = eq.pct_change().dropna()
    ann = 252
    mu = float(ret.mean()) * ann
    sig = float(ret.std()) * (ann ** 0.5) + 1e-15
    sharpe = mu / sig
    peak = eq.cummax()
    mdd = float(((eq - peak) / (peak + 1e-15)).min())
    calmar = mu / (abs(mdd) + 1e-15)

    rebalance_rows = snaps[snaps.get("rebalance", False)] if "rebalance" in snaps.columns else pd.DataFrame()
    n_reb = int(snaps["n_rebalances"].iloc[-1]) if "n_rebalances" in snaps.columns else 0
    halted_bars = int(snaps["is_halted"].sum()) if "is_halted" in snaps.columns else 0

    # Current weights (last rebalance).
    weight_cols = [c for c in snaps.columns if c.startswith("target_")]
    last_weights = {}
    if weight_cols and not rebalance_rows.empty:
        last_reb = rebalance_rows.iloc[-1]
        last_weights = {c.replace("target_", ""): float(last_reb[c]) for c in weight_cols}

    lines = [
        f"# Phase 48: RTMV Live Deployment — Backtest Report",
        f"",
        f"**Date:** {date_str}  ",
        f"**Assets:** {', '.join(cfg.assets)}  ",
        f"**Period:** {eq.index[0].date()} → {eq.index[-1].date()}  ",
        f"**λ:** {cfg.lambda_tilt}  n_states={cfg.n_states}  "
        f"lookback={cfg.lookback_bars}  rebalance_bars={cfg.rebalance_bars}  ",
        f"",
        f"---",
        f"",
        f"## Performance",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Sharpe (ann) | {sharpe:.3f} |",
        f"| Calmar | {calmar:.3f} |",
        f"| Max Drawdown | {mdd:.1%} |",
        f"| Ann Return | {mu:.1%} |",
        f"| Ann Vol | {sig:.1%} |",
        f"| N Rebalances | {n_reb} |",
        f"| Halted Bars | {halted_bars} |",
        f"| Final Equity | ${eq.iloc[-1]:,.2f} |",
        f"",
        f"---",
        f"",
        f"## Current Target Weights (last rebalance)",
        f"",
    ]
    if last_weights:
        lines += [f"| Asset | Weight |", f"|-------|--------|"]
        for a, w in sorted(last_weights.items(), key=lambda x: -x[1]):
            lines.append(f"| {a} | {w:.3f} |")
    else:
        lines.append("*No rebalances completed (lookback not satisfied).*")

    lines += [
        f"",
        f"---",
        f"",
        f"## Risk Flags",
        f"",
        (
            f"- Drawdown halt triggered {halted_bars} bars. "
            f"Threshold: {cfg.drawdown_halt:.0%}."
            if halted_bars > 0
            else "- No drawdown halt events."
        ),
        f"",
        f"---",
        f"*Generated by `scripts/run_rtmv_live.py`*",
    ]

    report_path = output_dir / f"report_{date_str}.md"
    report_path.write_text("\n".join(lines))
    logger.info("Report written → %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 48: RTMV live/backtest rebalancer")
    p.add_argument("--assets", default="SPY,GLD,TLT,IEF",
                   help="Comma-separated ticker list")
    p.add_argument("--lambda-tilt", type=float, default=0.05)
    p.add_argument("--n-states", type=int, default=3)
    p.add_argument("--n-restarts", type=int, default=3)
    p.add_argument("--lookback-bars", type=int, default=504)
    p.add_argument("--rebalance-bars", type=int, default=21)
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--drawdown-halt", type=float, default=0.25)
    p.add_argument("--output-dir", default="results/rtmv_live")
    p.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    p.add_argument("--poll-interval", type=float, default=3600.0,
                   help="Poll interval in seconds (live mode only)")
    p.add_argument("--start-date", default=None,
                   help="Backtest start date YYYY-MM-DD (default: full history)")
    p.add_argument("--lambda-by-state-rank", default=None,
                   help="Comma-separated λ schedule by SPY state rank, e.g. 0.02,0.05,0.10")
    p.add_argument("--proxy-asset", default=None,
                   help="Asset whose state rank drives λ selection (e.g. SPY)")
    p.add_argument("--kl-trigger-threshold", type=float, default=0.0,
                   help="Phase 53 KL threshold for out-of-cycle rebalance (0=disabled)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    assets = [a.strip() for a in args.assets.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lambda_by_state_rank = (
        [float(x) for x in args.lambda_by_state_rank.split(",")]
        if args.lambda_by_state_rank else []
    )
    cfg = RTMVRebalancerConfig(
        assets=assets,
        lambda_tilt=args.lambda_tilt,
        lambda_by_state_rank=lambda_by_state_rank,
        lambda_proxy_asset=args.proxy_asset,
        kl_trigger_threshold=args.kl_trigger_threshold,
        rebalance_bars=args.rebalance_bars,
        lookback_bars=args.lookback_bars,
        n_states=args.n_states,
        n_restarts=args.n_restarts,
        initial_capital=args.capital,
        slippage_bps=args.slippage_bps,
        drawdown_halt=args.drawdown_halt,
        poll_interval_s=args.poll_interval,
        output_dir=output_dir,
    )
    ma_cfg = MultiAssetConfig(
        ann_factor=252,
        rebalance_bars=args.rebalance_bars,
        lookback_bars=args.lookback_bars,
        n_states=args.n_states,
        n_restarts=args.n_restarts,
    )
    # Wire Supabase if credentials are present (graceful no-op otherwise).
    import os
    run_id = f"rtmv_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{args.mode}"
    sb_writer = SupabaseWriter(
        SupabaseWriterConfig(
            run_id=run_id,
            mode=args.mode,
            assets=assets,
            lambda_tilt=args.lambda_tilt,
            n_states=args.n_states,
            lookback_bars=args.lookback_bars,
            rebalance_bars=args.rebalance_bars,
            drawdown_halt=args.drawdown_halt,
        )
    )
    if sb_writer.enabled:
        logger.info("Supabase connected — run_id=%s", run_id)
    else:
        logger.info("Supabase not configured — parquet-only mode.")

    rebalancer = RTMVRebalancer(cfg, ma_config=ma_cfg, supabase_writer=sb_writer)

    if args.mode == "live":
        rebalancer.run_forever()
        return

    # Backtest mode.
    source = YFinanceSource(cache_dir=Path("results/cache"))
    pipeline = _build_pipeline()
    asset_returns, asset_features = _load_aligned_data(
        assets, source, pipeline, start_date=args.start_date
    )

    logger.info(
        "Running backtest: %d bars  λ=%.2f  n_states=%d  "
        "lookback=%d  rebalance=%d",
        len(asset_returns),
        args.lambda_tilt,
        args.n_states,
        args.lookback_bars,
        args.rebalance_bars,
    )

    snaps = rebalancer.run_backtest(asset_returns, asset_features)
    rebalancer.save_state()

    date_str = pd.Timestamp.now().strftime("%Y%m%d")
    report_path = _write_report(snaps, rebalancer, output_dir, date_str)

    # Print summary.
    eq = snaps["equity"].dropna()
    ret = eq.pct_change().dropna()
    ann = 252
    mu = float(ret.mean()) * ann
    sig = float(ret.std()) * (ann ** 0.5) + 1e-15
    sharpe = mu / sig
    peak = eq.cummax()
    mdd = float(((eq - peak) / (peak + 1e-15)).min())

    print("\n" + "=" * 60)
    print("RTMV BACKTEST COMPLETE")
    print("=" * 60)
    print(f"Assets:      {', '.join(assets)}")
    print(f"Period:      {eq.index[0].date()} → {eq.index[-1].date()}")
    print(f"λ:           {args.lambda_tilt}")
    print(f"Sharpe:      {sharpe:.3f}")
    print(f"Max DD:      {mdd:.1%}")
    print(f"Ann Return:  {mu:.1%}")
    print(f"Rebalances:  {rebalancer.state.n_rebalances}")
    print(f"Final eq:    ${eq.iloc[-1]:,.2f}")
    print(f"Report:      {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
