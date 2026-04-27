"""Vectorized backtesting for regime-based trading signals."""

from rde.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from rde.backtest.metrics import compute_metrics

__all__ = ["BacktestConfig", "BacktestResult", "run_backtest", "compute_metrics"]
