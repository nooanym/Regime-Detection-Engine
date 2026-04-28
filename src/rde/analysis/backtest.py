"""Regime-aware backtesting framework (Phase 29).

Implements:

1. **RegimeStrategyConfig** — per-regime strategy parameters (target
   position, scaling, stop-loss, take-profit).

2. **BacktestResult** — full equity curve, trade log, and regime-
   stratified performance statistics.

3. **run_backtest** — simulate a regime-conditional strategy over a
   historical return series, applying the regime-specific rules
   using the decoded regime sequence.

4. **regime_performance_attribution** — break down total PnL and
   Sharpe by regime.

5. **backtest_tearsheet** — aggregate statistics in a single DataFrame
   (annualised return, vol, Sharpe, Sortino, max drawdown, Calmar,
   win rate, average trade, regime breakdown).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------


@dataclass
class RegimeRule:
    """Position rule for a single regime.

    Attributes
    ----------
    target_position : float
        Signed target position (e.g. 1.0 = full long, -1.0 = full short,
        0.0 = flat).
    stop_loss : float or None
        Absolute loss (as fraction of equity) that triggers liquidation
        within the regime.
    take_profit : float or None
        Absolute gain that triggers profit-taking within the regime.
    """

    target_position: float = 1.0
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass
class RegimeStrategyConfig:
    """Strategy configuration mapping regime index to position rules.

    Attributes
    ----------
    rules : list[RegimeRule]
        One entry per regime (indexed by regime label 0..K-1).
    transaction_cost : float
        One-way transaction cost as a fraction of position change.
    ann_factor : int
        Bars per year for annualisation (e.g. 252, 8760).
    """

    rules: list[RegimeRule] = field(default_factory=list)
    transaction_cost: float = 0.001
    ann_factor: int = 252


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Output of :func:`run_backtest`.

    Attributes
    ----------
    equity : pd.Series
        Portfolio equity curve (starts at 1.0).
    returns : pd.Series
        Strategy return per bar.
    positions : pd.Series
        Position at each bar.
    trades : pd.DataFrame
        Columns: bar, direction (+1/-1), size, cost.
    regime_labels : np.ndarray
        Regime at each bar.
    """

    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    regime_labels: np.ndarray


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


def run_backtest(
    returns: pd.Series | np.ndarray,
    regimes: np.ndarray,
    config: RegimeStrategyConfig | None = None,
) -> BacktestResult:
    """Simulate a regime-conditional strategy.

    At each bar:
    - Look up target position for current regime.
    - If position changes, incur transaction cost on the |delta|.
    - Apply stop-loss / take-profit checks within the bar.
    - Compute bar PnL = position * return - transaction cost.

    Parameters
    ----------
    returns : array-like, length T
        Per-bar asset returns (e.g. log returns).
    regimes : np.ndarray, shape (T,)
        Regime label at each bar.
    config : RegimeStrategyConfig, optional

    Returns
    -------
    BacktestResult
    """
    cfg = config or RegimeStrategyConfig()
    r = np.asarray(returns, dtype=float)
    k_labels = np.asarray(regimes, dtype=int)
    T = len(r)
    idx = returns.index if isinstance(returns, pd.Series) else pd.RangeIndex(T)

    # Ensure we have rules for all regime labels.
    max_regime = int(k_labels.max()) if T > 0 else 0
    while len(cfg.rules) <= max_regime:
        cfg.rules.append(RegimeRule(target_position=0.0))

    positions = np.zeros(T)
    strat_returns = np.zeros(T)
    equity = np.zeros(T + 1)
    equity[0] = 1.0
    trade_rows: list[dict] = []

    current_pos = 0.0
    entry_equity = 1.0

    for t in range(T):
        regime = k_labels[t]
        rule = cfg.rules[regime]
        target = rule.target_position

        # Check stop / take-profit from prior bar's entry.
        if current_pos != 0.0:
            drawdown = (equity[t] - entry_equity) / (entry_equity + 1e-15)
            if rule.stop_loss is not None and drawdown <= -rule.stop_loss:
                target = 0.0
            elif rule.take_profit is not None and drawdown >= rule.take_profit:
                target = 0.0

        # Trade.
        delta = target - current_pos
        cost = abs(delta) * cfg.transaction_cost
        if abs(delta) > 1e-10:
            trade_rows.append({
                "bar": t,
                "direction": int(np.sign(delta)),
                "size": float(abs(delta)),
                "cost": float(cost),
            })
            if target != 0.0:
                entry_equity = equity[t]

        bar_ret = current_pos * r[t] - cost
        strat_returns[t] = bar_ret
        equity[t + 1] = equity[t] * (1.0 + bar_ret)
        positions[t] = current_pos
        current_pos = target

    equity_series = pd.Series(equity[1:], index=idx)
    return BacktestResult(
        equity=equity_series,
        returns=pd.Series(strat_returns, index=idx),
        positions=pd.Series(positions, index=idx),
        trades=pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(
            columns=["bar", "direction", "size", "cost"]
        ),
        regime_labels=k_labels,
    )


# ---------------------------------------------------------------------------
# Performance metrics helpers
# ---------------------------------------------------------------------------


def _sharpe(returns: np.ndarray, ann_factor: int, rf: float = 0.0) -> float:
    mu = float(returns.mean())
    sig = float(returns.std(ddof=1)) + 1e-15
    return float((mu - rf / ann_factor) / sig * np.sqrt(ann_factor))


def _sortino(returns: np.ndarray, ann_factor: int, mar: float = 0.0) -> float:
    downside = returns[returns < mar]
    if len(downside) == 0:
        return np.inf
    mu = float(returns.mean())
    dd_std = float(downside.std(ddof=1)) + 1e-15
    return float((mu - mar) / dd_std * np.sqrt(ann_factor))


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-15)
    return float(-dd.min())


def _calmar(total_return: float, max_dd: float, ann_factor: int, T: int) -> float:
    ann_ret = float((1.0 + total_return) ** (ann_factor / max(T, 1)) - 1.0)
    return ann_ret / (max_dd + 1e-15)


# ---------------------------------------------------------------------------
# Regime performance attribution
# ---------------------------------------------------------------------------


def regime_performance_attribution(
    result: BacktestResult,
    ann_factor: int = 252,
) -> pd.DataFrame:
    """Break down performance statistics by regime.

    Parameters
    ----------
    result : BacktestResult
    ann_factor : int

    Returns
    -------
    pd.DataFrame
        One row per regime + "all", columns: n_bars, total_return,
        ann_vol, sharpe, win_rate.
    """
    r = result.returns.values
    regimes = result.regime_labels
    K = int(regimes.max()) + 1

    rows = []
    for k in range(K):
        mask = regimes == k
        r_k = r[mask]
        if len(r_k) == 0:
            rows.append({"regime": k, "n_bars": 0, "total_return": 0.0,
                         "ann_vol": 0.0, "sharpe": np.nan, "win_rate": np.nan})
            continue
        total_ret = float(np.exp(r_k.sum()) - 1.0)
        ann_vol = float(r_k.std(ddof=1)) * np.sqrt(ann_factor)
        sharpe = _sharpe(r_k, ann_factor)
        win_rate = float((r_k > 0).mean())
        rows.append({
            "regime": k,
            "n_bars": int(mask.sum()),
            "total_return": total_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "win_rate": win_rate,
        })

    # All regimes combined.
    total_ret_all = float(np.exp(r.sum()) - 1.0)
    rows.append({
        "regime": "all",
        "n_bars": len(r),
        "total_return": total_ret_all,
        "ann_vol": float(r.std(ddof=1)) * np.sqrt(ann_factor),
        "sharpe": _sharpe(r, ann_factor),
        "win_rate": float((r > 0).mean()),
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tearsheet
# ---------------------------------------------------------------------------


def backtest_tearsheet(
    result: BacktestResult,
    ann_factor: int = 252,
) -> pd.Series:
    """Compute aggregate backtest statistics.

    Returns
    -------
    pd.Series
        Index: metric name, values: floats.
    """
    r = result.returns.values
    eq = result.equity.values
    T = len(r)

    total_ret = float(eq[-1] - 1.0)
    ann_ret = float((eq[-1]) ** (ann_factor / max(T, 1)) - 1.0)
    ann_vol = float(r.std(ddof=1)) * np.sqrt(ann_factor)
    sharpe = _sharpe(r, ann_factor)
    sortino = _sortino(r, ann_factor)
    mdd = _max_drawdown(eq)
    calmar = _calmar(total_ret, mdd, ann_factor, T)
    win_rate = float((r > 0).mean())
    avg_trade = float(result.trades["size"].mean()) if len(result.trades) > 0 else 0.0
    n_trades = int(len(result.trades))

    metrics = {
        "total_return": total_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "win_rate": win_rate,
        "n_trades": float(n_trades),
        "avg_trade_size": avg_trade,
        "n_bars": float(T),
    }
    return pd.Series(metrics)
