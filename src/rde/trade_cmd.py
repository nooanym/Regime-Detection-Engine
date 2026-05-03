"""CLI command: ``rde trade`` — paper-trading loop (Phase 35)."""
from __future__ import annotations

from pathlib import Path

import click


@click.command("trade")
@click.option(
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a model.pkl saved with ``rde run --save-model``.",
)
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Asset config YAML (same one used to train the model).",
)
@click.option(
    "--capital",
    default=100_000.0,
    show_default=True,
    type=float,
    help="Initial paper-trading capital in quote currency.",
)
@click.option(
    "--output",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Directory for trade_log.parquet and portfolio_snapshots.parquet. "
         "Defaults to results/<symbol>/live/.",
)
@click.option(
    "--slippage-bps",
    default=5.0,
    show_default=True,
    type=float,
    help="Paper-trading slippage in basis points.",
)
@click.option(
    "--poll-interval",
    default=60.0,
    show_default=True,
    type=float,
    help="Seconds between yfinance polls in live mode.",
)
@click.option(
    "--warmup-bars",
    default=200,
    show_default=True,
    type=int,
    help="Historical bars fed to OnlineDecoder before trading begins.",
)
@click.option(
    "--backtest",
    is_flag=True,
    default=False,
    help="Run on historical bars from regimes.parquet instead of live polling.",
)
@click.option(
    "--regimes-path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to regimes.parquet for --backtest mode.",
)
def trade_cmd(
    model_path: str,
    config_path: str,
    capital: float,
    output_dir: str | None,
    slippage_bps: float,
    poll_interval: float,
    warmup_bars: int,
    backtest: bool,
    regimes_path: str | None,
) -> None:
    """Paper-trading loop: regime → signal → paper order → trade log.

    Two modes:

    \b
      Live mode (default)
        Polls yfinance at --poll-interval seconds, decodes the regime with
        OnlineDecoder, and executes paper trades via PaperPortfolio.

    \b
      Backtest mode (--backtest)
        Replays regimes.parquet from disk at full speed — useful for
        verifying strategy P&L on historical data.

    Writes trade_log.parquet and portfolio_snapshots.parquet to --output.
    """
    import logging

    import pandas as pd

    from rde.config.loader import load_config
    from rde.models.persistence import load_model
    from rde.trading.loop import TradingLoop, TradingLoopConfig

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load config + model ────────────────────────────────────────────────────
    cfg = load_config(Path(config_path))
    symbol = cfg.asset.symbol

    click.echo(f"\n  Loading model from {model_path} …")
    fitted = load_model(Path(model_path))
    click.echo(f"  Model: {fitted.n_states} states  features: {fitted.feature_names}")

    # ── Resolve output dir ────────────────────────────────────────────────────
    out = Path(output_dir) if output_dir else Path(cfg.run.output_dir.format(symbol=symbol)) / "live"

    # ── Strategy: best-return state = long, worst = flat ─────────────────────
    # Read regime analytics if available to rank states
    ana_path = Path(cfg.run.output_dir.format(symbol=symbol)) / "regime_analytics.parquet"
    strategy_rules: list[tuple[int, float]] = []
    if ana_path.exists():
        ana = pd.read_parquet(ana_path)
        sharpe_col = ana["sharpe_ann"] if "sharpe_ann" in ana.columns else None
        if sharpe_col is not None:
            best_state = int(sharpe_col.idxmax())
            worst_state = int(sharpe_col.idxmin())
            strategy_rules = [
                (best_state, 1.0),   # long the best-Sharpe regime
                (worst_state, 0.0),  # flat the worst-Sharpe regime
            ]
            click.echo(f"  Strategy: long state {best_state}, flat state {worst_state}")
    if not strategy_rules:
        strategy_rules = [(0, 1.0)]  # fallback: long state 0
        click.echo("  Strategy: long state 0 (fallback — run rde run first for analytics).")

    # ── Build loop ───────────────────────────────────────────────────────────
    loop_cfg = TradingLoopConfig(
        symbol=symbol,
        interval=cfg.asset.interval,
        poll_interval_s=poll_interval,
        warmup_bars=warmup_bars,
        output_dir=out,
        slippage_bps=slippage_bps,
        initial_capital=capital,
    )
    loop = TradingLoop(fitted=fitted, config=loop_cfg, strategy_rules=strategy_rules)

    sep = "═" * 60
    click.echo(f"\n{sep}")
    click.echo(f"  Paper Trading — {symbol}")
    click.echo(f"  Capital: ${capital:,.0f}  Slippage: {slippage_bps}bps")
    click.echo(f"  Output:  {out}")
    click.echo(sep)

    # ── Execute ────────────────────────────────────────────────────────────────
    if backtest:
        reg_path = Path(regimes_path) if regimes_path else (
            Path(cfg.run.output_dir.format(symbol=symbol)) / "regimes.parquet"
        )
        if not reg_path.exists():
            click.echo(f"  ERROR: regimes.parquet not found at {reg_path}.", err=True)
            raise SystemExit(1)

        click.echo(f"  Backtesting on {reg_path} …")
        bars_df = pd.read_parquet(reg_path)
        final_equity = loop.run_once(bars_df)
        loop.save_state()

        n_trades = loop.state.n_trades
        ret = (final_equity - capital) / capital * 100
        click.echo(f"\n  Backtest complete")
        click.echo(f"  Final equity: ${final_equity:,.2f}  ({ret:+.2f}%)")
        click.echo(f"  Trades: {n_trades}")
        click.echo(f"  Output written to {out}")
    else:
        click.echo(f"  Starting live loop (poll every {poll_interval:.0f}s)  Ctrl-C to stop.\n")
        loop.run_forever()
