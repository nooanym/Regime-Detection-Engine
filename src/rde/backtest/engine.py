"""Vectorized backtest engine for regime-based trading signals.

Translates a continuous signal in [-1, 1] into a position series and computes
a portfolio equity curve net of transaction costs. Strictly causal: the
position at bar *t* is derived from the signal at bar *t* but applied to the
return at bar *t+1* via a one-bar position shift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rde.backtest.metrics import compute_metrics

logger = logging.getLogger(__name__)

_VALID_MODES = frozenset({"proportional", "threshold", "long_only"})


@dataclass
class BacktestConfig:
    """Configuration for the vectorized backtest engine.

    Attributes
    ----------
    mode : str
        Position sizing mode: ``"proportional"`` clips signal to [-1, 1];
        ``"threshold"`` goes long/short/flat based on a threshold;
        ``"long_only"`` clips signal to [0, 1].
    threshold : float
        Threshold for ``"threshold"`` mode. Long when signal > +threshold,
        short when signal < -threshold, flat otherwise.
    transaction_cost_bps : float
        Transaction cost in basis points per unit of position change.
        Applied as ``abs(pos_t - pos_{t-1}) * cost_bps / 10_000`` per bar.
    initial_capital : float
        Starting portfolio value (used to denominate the equity curve).
    bars_per_year : float
        Number of bars per calendar year, used for annualisation in metrics.
        Default 8760 assumes hourly bars.
    """

    mode: str = "proportional"
    threshold: float = 0.25
    transaction_cost_bps: float = 1.0
    initial_capital: float = 10_000.0
    bars_per_year: float = 8760.0


@dataclass
class BacktestResult:
    """Output container for a completed backtest run.

    Attributes
    ----------
    equity_curve : pd.Series
        Portfolio value at each bar, indexed like the input data.
    positions : pd.Series
        Position size at each bar in [-1, 1].
    strategy_returns : pd.Series
        Gross per-bar returns before transaction costs:
        ``position.shift(1) * returns``.
    net_returns : pd.Series
        Per-bar returns after transaction costs.
    metrics : dict
        Performance metrics from :func:`compute_metrics`.
    config : BacktestConfig
        The configuration used for this backtest.
    """

    equity_curve: pd.Series
    positions: pd.Series
    strategy_returns: pd.Series
    net_returns: pd.Series
    metrics: dict
    config: BacktestConfig


def run_backtest(
    returns: pd.Series,
    signal: pd.Series,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a vectorized backtest on a regime signal.

    Parameters
    ----------
    returns : pd.Series
        Log returns aligned with the signal. Must have the same length as
        ``signal``.
    signal : pd.Series
        Causal signal (use filtered/online-decoded output). Values should be
        in approximately [-1, 1]; clamped by the position mode.
    config : BacktestConfig, optional
        Backtest parameters. Defaults to ``BacktestConfig()`` if ``None``.

    Returns
    -------
    BacktestResult
        Equity curve, positions, returns, and performance metrics.

    Raises
    ------
    ValueError
        If ``config.mode`` is not one of ``{"proportional", "threshold",
        "long_only"}`` or if ``returns`` and ``signal`` have different lengths.

    Notes
    -----
    **No lookahead guarantee**: positions are shifted forward by one bar so
    that the return at bar *t+1* is earned using only information available at
    bar *t*. The first bar's position is always 0 (flat).
    """
    if config is None:
        config = BacktestConfig()

    if config.mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid mode {config.mode!r}. Must be one of {sorted(_VALID_MODES)}."
        )

    if len(returns) != len(signal):
        raise ValueError(
            f"returns (len={len(returns)}) and signal (len={len(signal)}) "
            "must have the same length."
        )

    # ── Position sizing ───────────────────────────────────────────────────────
    sig_vals = signal.values.astype(float)

    if config.mode == "proportional":
        pos_vals = np.clip(sig_vals, -1.0, 1.0)
    elif config.mode == "threshold":
        thresh = config.threshold
        pos_vals = np.where(
            sig_vals > thresh, 1.0,
            np.where(sig_vals < -thresh, -1.0, 0.0),
        )
    else:  # long_only
        pos_vals = np.clip(sig_vals, 0.0, 1.0)

    positions = pd.Series(pos_vals, index=signal.index, name="position")

    # ── No-lookahead: shift positions forward by 1 ───────────────────────────
    # position.shift(1) gives NaN at t=0; fill with 0 (flat at start).
    lagged_pos = positions.shift(1).fillna(0.0)

    # ── Gross strategy returns ────────────────────────────────────────────────
    ret_vals = returns.values.astype(float)
    strategy_ret_vals = lagged_pos.values * ret_vals
    strategy_returns = pd.Series(
        strategy_ret_vals, index=returns.index, name="strategy_return"
    )

    # ── Transaction costs ─────────────────────────────────────────────────────
    cost_per_unit = config.transaction_cost_bps / 10_000.0
    pos_diff = np.abs(np.diff(pos_vals, prepend=0.0))  # position change at each bar
    pos_diff[0] = 0.0  # no startup cost; bar 0 earns no return (lagged_pos=0)
    costs = pos_diff * cost_per_unit

    net_ret_vals = strategy_ret_vals - costs
    net_returns = pd.Series(net_ret_vals, index=returns.index, name="net_return")

    # ── Equity curve ──────────────────────────────────────────────────────────
    equity_vals = config.initial_capital * np.cumprod(1.0 + np.nan_to_num(net_ret_vals))
    equity_curve = pd.Series(equity_vals, index=returns.index, name="equity_curve")

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(
        equity_curve=equity_curve,
        strategy_returns=net_returns,
        positions=positions,
        bars_per_year=config.bars_per_year,
    )

    logger.info(
        "Backtest complete: mode=%s  cost_bps=%.2f  total_return=%.4f  "
        "sharpe=%.4f  max_dd=%.4f",
        config.mode,
        config.transaction_cost_bps,
        metrics["total_return"],
        metrics["sharpe_ann"],
        metrics["max_drawdown"],
    )

    return BacktestResult(
        equity_curve=equity_curve,
        positions=positions,
        strategy_returns=strategy_returns,
        net_returns=net_returns,
        metrics=metrics,
        config=config,
    )


def backtest_from_parquet(
    regimes_path: str | Path,
    signals_path: str | Path,
    config: BacktestConfig | None = None,
    return_col: str = "log_return",
    signal_col: str = "signal",
) -> BacktestResult:
    """Load regime and signal parquets and run a backtest.

    Parameters
    ----------
    regimes_path : str or Path
        Path to ``regimes.parquet`` produced by the CLI (must contain
        ``return_col``).
    signals_path : str or Path
        Path to ``signals.parquet`` produced by the CLI with ``--signal``
        (must contain ``signal_col``).
    config : BacktestConfig, optional
        Backtest configuration. Defaults to ``BacktestConfig()`` if ``None``.
    return_col : str, optional
        Column name for log returns in the regimes parquet. Default
        ``"log_return"``.
    signal_col : str, optional
        Column name for the signal in the signals parquet. Default
        ``"signal"``.

    Returns
    -------
    BacktestResult
        The backtest result aligned on the intersection of the two parquet
        indices.

    Raises
    ------
    KeyError
        If the expected columns are not found in the parquets.
    ValueError
        If the aligned index is empty.
    """
    regimes_path = Path(regimes_path)
    signals_path = Path(signals_path)

    logger.info("Loading regimes from %s", regimes_path)
    df_regimes = pd.read_parquet(regimes_path)

    logger.info("Loading signals from %s", signals_path)
    df_signals = pd.read_parquet(signals_path)

    if return_col not in df_regimes.columns:
        raise KeyError(
            f"Column {return_col!r} not found in regimes parquet. "
            f"Available: {list(df_regimes.columns)}"
        )
    if signal_col not in df_signals.columns:
        raise KeyError(
            f"Column {signal_col!r} not found in signals parquet. "
            f"Available: {list(df_signals.columns)}"
        )

    # Align on common index
    common_idx = df_regimes.index.intersection(df_signals.index)
    if len(common_idx) == 0:
        raise ValueError(
            "No overlapping index entries between regimes and signals parquets."
        )

    returns = df_regimes.loc[common_idx, return_col]
    signal = df_signals.loc[common_idx, signal_col]

    logger.info(
        "Aligned %d bars for backtest (regimes=%d, signals=%d)",
        len(common_idx),
        len(df_regimes),
        len(df_signals),
    )

    return run_backtest(returns=returns, signal=signal, config=config)
