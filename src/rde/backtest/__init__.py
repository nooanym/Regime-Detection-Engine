"""Vectorized backtesting for regime-based trading signals."""

from rde.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from rde.backtest.metrics import compute_metrics
from rde.backtest.walk_forward_backtest import (
    WalkForwardBacktestConfig,
    WalkForwardBacktestResult,
    walk_forward_backtest,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "compute_metrics",
    "WalkForwardBacktestConfig",
    "WalkForwardBacktestResult",
    "walk_forward_backtest",
]
