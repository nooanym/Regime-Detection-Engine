"""Live paper-trading loop (Phase 35).

Connects the fitted HMM model with the OnlineDecoder, RegimeSignalStrategy,
PaperPortfolio, and RegimeChangeMonitor into a single streaming loop that
polls yfinance for new bars and executes paper trades.

The loop is designed to run as a foreground process (CLI) and logs all
activity to the standard Python logging system.  The trade log and portfolio
snapshots are written to disk at a configurable interval.

Typical usage (via CLI)::

    uv run rde trade \\
        --model  results/BTC-USD/model.pkl \\
        --config configs/btc.yaml \\
        --capital 100000 \\
        --output  results/BTC-USD/live/

Public API
----------
TradingLoopConfig
TradingLoopState
TradingLoop
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from rde.models.hmm import FittedModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TradingLoopConfig:
    """Configuration for the paper-trading loop.

    Attributes
    ----------
    poll_interval_s : float
        Seconds between yfinance polls.  Default: 60 (1 minute).  For hourly
        data, the bar only changes once per hour; polling more frequently is
        safe (yfinance returns cached data) but wastes I/O.
    symbol : str
        Yahoo Finance ticker, e.g. ``"BTC-USD"``.
    interval : str
        Bar interval, e.g. ``"1h"``.
    warmup_bars : int
        Number of historical bars to feed through OnlineDecoder before
        treating the posterior as warmed-up.  Default: 200.
    save_interval_bars : int
        Write the trade log and portfolio snapshot to disk every N new bars.
        Default: 10.
    output_dir : Path
        Directory for trade_log.parquet and portfolio_snapshots.parquet.
    slippage_bps : float
        Paper-trading slippage in basis points.
    initial_capital : float
        Starting cash balance in quote currency.
    """

    symbol: str
    interval: str = "1h"
    poll_interval_s: float = 60.0
    warmup_bars: int = 200
    save_interval_bars: int = 10
    output_dir: Path = Path("results/live")
    slippage_bps: float = 5.0
    initial_capital: float = 100_000.0


# ---------------------------------------------------------------------------
# State snapshot
# ---------------------------------------------------------------------------


@dataclass
class TradingLoopState:
    """Mutable runtime state of a running :class:`TradingLoop`.

    Attributes
    ----------
    last_bar_ts : pd.Timestamp or None
        Timestamp of the last bar processed by the decoder.
    current_regime : int or None
        Most recently decoded regime index.
    equity : float
        Mark-to-market equity at last bar.
    n_bars_processed : int
        Total bars processed since loop start.
    n_trades : int
        Total fills executed.
    is_warmed_up : bool
        True once ``warmup_bars`` bars have been processed.
    snapshots : list[dict]
        Per-bar portfolio snapshots (timestamp, regime, equity, position).
    """

    last_bar_ts: pd.Timestamp | None = None
    current_regime: int | None = None
    equity: float = 0.0
    n_bars_processed: int = 0
    n_trades: int = 0
    is_warmed_up: bool = False
    snapshots: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trading Loop
# ---------------------------------------------------------------------------


class TradingLoop:
    """Paper-trading loop: decoder → strategy → portfolio → alerts.

    Parameters
    ----------
    fitted : FittedModel
        The trained HMM loaded from disk.
    config : TradingLoopConfig
    strategy_rules : list[tuple[int, float]]
        Each element is ``(regime_index, target_weight)``.  Target weight
        of ``1.0`` = 100% long, ``0.0`` = flat, ``-1.0`` = 100% short.
    alert_config : AlertConfig | None
        Alerting configuration.  Defaults to LOG-only.
    regime_labels : dict[int, str] | None
        Human-readable label per regime, used in alerts.

    Methods
    -------
    run_once(bars_df)
        Process a batch of new OHLCV bars.  Updates decoder and portfolio.
        Returns the new portfolio equity.
    run_forever()
        Block and poll yfinance at ``config.poll_interval_s`` intervals.
        Ctrl-C to stop.
    step(timestamp, feature_vector, price)
        Process a single pre-computed feature bar (useful for backtesting).
    save_state()
        Flush trade log and snapshots to parquet in output_dir.
    """

    def __init__(
        self,
        fitted: "FittedModel",
        config: TradingLoopConfig,
        strategy_rules: list[tuple[int, float]] | None = None,
        alert_config=None,
        regime_labels: dict[int, str] | None = None,
        risk_guard_config=None,
    ) -> None:
        from rde.inference.online import OnlineDecoder
        from rde.trading.alerts import AlertConfig, AlertChannel, RegimeChangeMonitor
        from rde.trading.paper_portfolio import PaperPortfolio, PortfolioConfig
        from rde.trading.signal_strategy import (
            RegimeRule,
            RegimeSignalStrategy,
            SignalStrategyConfig,
        )

        self.fitted = fitted
        self.config = config
        self.state = TradingLoopState(equity=config.initial_capital)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.decoder = OnlineDecoder(fitted)

        # ── Strategy ──────────────────────────────────────────────────────────
        rules = [
            RegimeRule(regime=r, target_weight=w)
            for r, w in (strategy_rules or [])
        ]
        self.strategy = RegimeSignalStrategy(
            SignalStrategyConfig(rules=rules, default_weight=0.0)
        )

        # ── Portfolio ─────────────────────────────────────────────────────────
        quote = config.symbol.split("-")[-1] if "-" in config.symbol else "USD"
        self.portfolio = PaperPortfolio(
            PortfolioConfig(
                initial_capital=config.initial_capital,
                slippage_bps=config.slippage_bps,
            )
        )
        self._quote = quote

        # ── Alerts ────────────────────────────────────────────────────────────
        if alert_config is None:
            alert_config = AlertConfig(channels=[AlertChannel.LOG])
        self.monitor = RegimeChangeMonitor(
            symbol=config.symbol,
            config=alert_config,
            regime_labels=regime_labels,
        )

        # ── Trade log ─────────────────────────────────────────────────────────
        from rde.trading.trade_log import TradeLog
        self.trade_log = TradeLog()

        # ── Risk guard ────────────────────────────────────────────────────────
        from rde.trading.risk_guard import RiskGuard, RiskGuardConfig
        _rg_cfg = risk_guard_config if risk_guard_config is not None else RiskGuardConfig()
        self.risk_guard = RiskGuard(_rg_cfg, initial_equity=config.initial_capital)

        # ── State ─────────────────────────────────────────────────────────────
        self._prev_regime: int | None = None
        self._bars_since_save: int = 0

        config.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Core step ─────────────────────────────────────────────────────────────

    def step(
        self,
        timestamp: pd.Timestamp,
        feature_vector: np.ndarray,
        price: float,
    ) -> float:
        """Process one bar: decode regime, trade if needed, record snapshot.

        Parameters
        ----------
        timestamp : pd.Timestamp
        feature_vector : np.ndarray
            Raw (unscaled) feature vector for this bar — same columns as
            ``fitted.feature_names``.
        price : float
            Close price used for mark-to-market and order execution.

        Returns
        -------
        float
            Current equity after processing this bar.
        """
        # Decode
        posterior = self.decoder.step(feature_vector)
        regime = int(np.argmax(posterior))
        confidence = float(posterior[regime])

        self.state.n_bars_processed += 1
        self.state.last_bar_ts = timestamp
        self.state.current_regime = regime

        # Warm-up gate
        if self.state.n_bars_processed < self.config.warmup_bars:
            self.state.equity = self.portfolio.mark_to_market(price)
            return self.state.equity
        if not self.state.is_warmed_up:
            self.state.is_warmed_up = True
            logger.info("Decoder warmed up after %d bars.", self.state.n_bars_processed)

        # Regime change alert
        self.monitor.update(timestamp, regime, posterior)

        # Trading — gated by risk guard
        equity = self.portfolio.mark_to_market(price)
        trading_permitted = self.risk_guard.update(timestamp, equity)

        if trading_permitted:
            target_qty = self.strategy.target_quantity(regime, equity, price)
            if self.strategy.signal_changed(self._prev_regime or regime, regime) or self._prev_regime is None:
                fill = self.portfolio.execute(timestamp, self.config.symbol, target_qty, price)
                if fill is not None:
                    self.trade_log.append(fill)
                    self.state.n_trades += 1
                    logger.info(
                        "%s  regime=%d  %s %.4f @ %.2f  equity=%.2f",
                        timestamp, regime, fill.side, fill.quantity, fill.price, equity,
                    )
        else:
            logger.warning(
                "%s  RiskGuard HALT — skipping trade  reason=%s",
                timestamp, self.risk_guard.state.halt_reason,
            )

        self._prev_regime = regime
        equity = self.portfolio.mark_to_market(price)
        self.state.equity = equity

        self.state.snapshots.append({
            "timestamp": timestamp,
            "regime": regime,
            "confidence": confidence,
            "price": price,
            "equity": equity,
            "position": self.portfolio.position,
            "cash": self.portfolio.cash,
            "risk_halted": not trading_permitted,
        })

        self._bars_since_save += 1
        if self._bars_since_save >= self.config.save_interval_bars:
            self.save_state()
            self._bars_since_save = 0

        return equity

    # ── Batch processing ──────────────────────────────────────────────────────

    def run_once(self, bars_df: pd.DataFrame) -> float:
        """Process a DataFrame of new OHLCV bars with pre-computed features.

        Parameters
        ----------
        bars_df : pd.DataFrame
            Must have a DatetimeIndex and columns matching
            ``fitted.feature_names`` plus ``"Close"``.

        Returns
        -------
        float
            Portfolio equity after processing all rows.
        """
        feature_names = self.fitted.feature_names
        missing = [f for f in feature_names if f not in bars_df.columns]
        if missing:
            raise ValueError(f"bars_df missing feature columns: {missing}")
        if "Close" not in bars_df.columns:
            raise ValueError("bars_df must have a 'Close' column.")

        equity = self.state.equity
        for ts, row in bars_df.iterrows():
            x = row[feature_names].values.astype(float)
            price = float(row["Close"])
            equity = self.step(ts, x, price)
        return equity

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_state(self) -> None:
        """Write trade log and portfolio snapshots to parquet."""
        out = self.config.output_dir
        if self.trade_log:
            self.trade_log.save_parquet(out / "trade_log.parquet")

        if self.state.snapshots:
            snap_df = pd.DataFrame(self.state.snapshots).set_index("timestamp")
            snap_df.to_parquet(out / "portfolio_snapshots.parquet")

        logger.debug(
            "State saved: %d fills, %d snapshots",
            len(self.trade_log),
            len(self.state.snapshots),
        )

    # ── Live polling ──────────────────────────────────────────────────────────

    def run_forever(self) -> None:
        """Poll yfinance at ``config.poll_interval_s`` and process new bars.

        Runs until interrupted by Ctrl-C.
        """
        import yfinance as yf
        from rde.features.pipeline import FeaturePipeline
        from rde.features.returns import LogReturns, SmoothedReturns
        from rde.features.volatility import RollingVolatility

        logger.info(
            "Starting live loop for %s  (poll every %.0fs)  Ctrl-C to stop.",
            self.config.symbol, self.config.poll_interval_s,
        )

        feature_pipeline = FeaturePipeline([
            LogReturns(),
            RollingVolatility(window=24),
            SmoothedReturns(window=12),
        ])

        last_processed_ts: pd.Timestamp | None = None

        while True:
            try:
                raw = yf.download(
                    self.config.symbol,
                    period="5d",
                    interval=self.config.interval,
                    auto_adjust=True,
                    progress=False,
                )
                if raw.empty:
                    logger.warning("yfinance returned empty data — retrying in %.0fs", self.config.poll_interval_s)
                    time.sleep(self.config.poll_interval_s)
                    continue

                # Flatten multi-level columns if present
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                df = feature_pipeline.transform(raw).dropna()
                if df.empty:
                    time.sleep(self.config.poll_interval_s)
                    continue

                # Find bars we haven't processed yet
                if last_processed_ts is not None:
                    new_bars = df[df.index > last_processed_ts]
                else:
                    # First run: use warmup + any new bars
                    new_bars = df.iloc[-(self.config.warmup_bars + 5):]

                for ts, row in new_bars.iterrows():
                    x = row[self.fitted.feature_names].values.astype(float)
                    price = float(row["Close"])
                    equity = self.step(ts, x, price)
                    last_processed_ts = ts

                logger.info(
                    "Polled at %s — regime=%s  equity=%.2f  trades=%d",
                    pd.Timestamp.now(tz="UTC").isoformat()[:19],
                    self.state.current_regime,
                    self.state.equity,
                    self.state.n_trades,
                )

            except KeyboardInterrupt:
                logger.info("Loop stopped by user.")
                self.save_state()
                break
            except Exception as exc:
                logger.error("Loop error: %s — retrying in %.0fs", exc, self.config.poll_interval_s)

            time.sleep(self.config.poll_interval_s)
