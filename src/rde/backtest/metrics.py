"""Performance metrics for vectorized backtests.

Computes standard return, risk-adjusted, and trade-level statistics from an
equity curve, a returns series, and a positions series.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_metrics(
    equity_curve: pd.Series,
    strategy_returns: pd.Series,
    positions: pd.Series,
    bars_per_year: float = 8760.0,
) -> dict:
    """Compute a standard set of backtest performance metrics.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio value over time (initial_capital * cumprod(1 + net_returns)).
    strategy_returns : pd.Series
        Per-bar strategy returns (net of costs), aligned with equity_curve.
    positions : pd.Series
        Position size at each bar (range [-1, 1] for long/short, 0 for flat).
    bars_per_year : float, optional
        Number of bars per calendar year used for annualisation. Default 8760
        assumes hourly bars (24 * 365).

    Returns
    -------
    dict
        Keys: ``total_return``, ``cagr``, ``sharpe_ann``, ``max_drawdown``,
        ``calmar``, ``hit_rate``, ``profit_factor``, ``n_trades``,
        ``avg_return_per_trade``.

    Notes
    -----
    - Risk-free rate is assumed to be zero for Sharpe computation.
    - ``max_drawdown`` is expressed in [0, 1] (positive = loss).
    - ``calmar`` is ``nan`` when ``max_drawdown == 0`` to avoid division-by-zero.
    - ``profit_factor`` is ``nan`` when there are no negative-return bars.
    - Returns are treated as arithmetic (not log) for hit_rate / profit_factor.
    """
    # ── Total return ──────────────────────────────────────────────────────────
    initial = float(equity_curve.iloc[0])
    final = float(equity_curve.iloc[-1])
    total_return = (final / initial) - 1.0 if initial != 0.0 else float("nan")

    # ── CAGR ─────────────────────────────────────────────────────────────────
    n_bars = len(equity_curve)
    years = n_bars / bars_per_year
    if years > 0 and initial > 0 and final > 0:
        cagr = (final / initial) ** (1.0 / years) - 1.0
    else:
        cagr = float("nan")

    # ── Annualised Sharpe (risk-free = 0) ─────────────────────────────────────
    ret_vals = strategy_returns.dropna().values
    if len(ret_vals) >= 2:
        mean_r = float(np.mean(ret_vals))
        std_r = float(np.std(ret_vals, ddof=1))
        sharpe_ann = (mean_r / std_r * np.sqrt(bars_per_year)) if std_r > 0.0 else float("nan")
    else:
        sharpe_ann = float("nan")

    # ── Max drawdown ──────────────────────────────────────────────────────────
    ec_vals = equity_curve.values.astype(float)
    running_max = np.maximum.accumulate(ec_vals)
    drawdowns = (running_max - ec_vals) / running_max
    max_drawdown = float(np.nanmax(drawdowns)) if len(drawdowns) > 0 else float("nan")

    # ── Calmar ────────────────────────────────────────────────────────────────
    calmar = (cagr / max_drawdown) if (np.isfinite(max_drawdown) and max_drawdown > 0.0) else float("nan")

    # ── Hit rate ─────────────────────────────────────────────────────────────
    nonzero_returns = ret_vals[ret_vals != 0.0]
    hit_rate = float(np.mean(nonzero_returns > 0.0)) if len(nonzero_returns) > 0 else float("nan")

    # ── Profit factor ─────────────────────────────────────────────────────────
    positive_sum = float(np.sum(ret_vals[ret_vals > 0.0]))
    negative_sum = float(np.sum(ret_vals[ret_vals < 0.0]))  # negative number
    if negative_sum < 0.0:
        profit_factor = positive_sum / abs(negative_sum)
    else:
        profit_factor = float("nan")

    # ── Number of trades (position changes) ───────────────────────────────────
    pos_vals = positions.fillna(0.0).values.astype(float)
    position_changes = np.diff(pos_vals)
    n_trades = int(np.sum(position_changes != 0.0))

    # ── Average return per trade ──────────────────────────────────────────────
    avg_return_per_trade = (total_return / n_trades) if (n_trades > 0 and np.isfinite(total_return)) else float("nan")

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe_ann": sharpe_ann,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "hit_rate": hit_rate,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
        "avg_return_per_trade": avg_return_per_trade,
    }
