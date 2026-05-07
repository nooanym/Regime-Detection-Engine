"""Live trading utilities (Phases 34–36).

Public API
----------
Paper portfolio
  - :class:`PaperPortfolio` — simulated portfolio tracking cash, position, P&L.
  - :class:`PortfolioConfig` — configuration for :class:`PaperPortfolio`.
  - :class:`Fill` — record of a single simulated fill.

Signal strategy
  - :class:`RegimeSignalStrategy` — maps HMM regime to target weight / quantity.
  - :class:`SignalStrategyConfig` — configuration for :class:`RegimeSignalStrategy`.
  - :class:`RegimeRule` — per-regime weight specification.

Trade log
  - :class:`TradeLog` — append-only fill log with parquet/CSV serialisation.

Exchange abstraction
  - :class:`ExchangeABC` — abstract interface.
  - :class:`MockExchange` — in-memory exchange for paper trading and tests.
  - :class:`BinanceTestnetExchange` — ccxt Binance testnet wrapper.
  - :class:`OrderRequest`, :class:`OrderResult`, :class:`Balance` — data classes.

Alerting
  - :class:`RegimeChangeMonitor` — streaming regime-change detector.
  - :class:`AlertConfig` — configuration for :class:`RegimeChangeMonitor`.
  - :class:`AlertChannel` — dispatch targets (LOG, WEBHOOK, CALLBACK).
  - :class:`RegimeChangeAlert` — alert payload.

Risk guard
  - :class:`RiskGuard` — drawdown and daily-loss monitor; halts trading on breach.
  - :class:`RiskGuardConfig` — thresholds and cooldown configuration.
  - :class:`RiskGuardState` — mutable runtime state exposed for inspection.
"""

from rde.trading.paper_portfolio import Fill, PaperPortfolio, PortfolioConfig
from rde.trading.signal_strategy import RegimeRule, RegimeSignalStrategy, SignalStrategyConfig
from rde.trading.trade_log import TradeLog
from rde.trading.exchange import (
    Balance,
    BinanceTestnetExchange,
    ExchangeABC,
    MockExchange,
    OrderRequest,
    OrderResult,
)
from rde.trading.alerts import (
    AlertChannel,
    AlertConfig,
    RegimeChangeAlert,
    RegimeChangeMonitor,
)
from rde.trading.risk_guard import RiskGuard, RiskGuardConfig, RiskGuardState

__all__ = [
    # paper portfolio
    "Fill",
    "PaperPortfolio",
    "PortfolioConfig",
    # signal strategy
    "RegimeRule",
    "RegimeSignalStrategy",
    "SignalStrategyConfig",
    # trade log
    "TradeLog",
    # exchange
    "Balance",
    "BinanceTestnetExchange",
    "ExchangeABC",
    "MockExchange",
    "OrderRequest",
    "OrderResult",
    # alerting
    "AlertChannel",
    "AlertConfig",
    "RegimeChangeAlert",
    "RegimeChangeMonitor",
    # risk guard
    "RiskGuard",
    "RiskGuardConfig",
    "RiskGuardState",
]
