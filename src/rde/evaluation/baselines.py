"""Proper strategy baselines for honest edge comparison (Phase 37.3).

Five baselines implemented without look-ahead:

1. **Buy-and-hold** — always long, no transaction costs.
2. **Vol-targeted buy-and-hold** — scales position by realized-vol target.
3. **Naive momentum** — long when rolling mean return is positive.
4. **Naive vol-regime** — long when realized vol is below expanding median.
5. **Two-state HMM** — minimal complexity HMM with same default strategy.

All baselines use causal signals only (no future data).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rde.analysis.backtest import (
    BacktestResult,
    RegimeRule,
    RegimeStrategyConfig,
    backtest_tearsheet,
    run_backtest,
)
from rde.evaluation.purged_cv import _default_strategy_config
from rde.inference.online import OnlineDecoder
from rde.models.hmm import train_hmm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BaselineResult:
    """Performance metrics for a single baseline strategy.

    Attributes
    ----------
    name : str
        Baseline identifier.
    sharpe : float
    calmar : float
    max_drawdown : float
    ann_return : float
    ann_vol : float
    n_trades : int
    equity : pd.Series
        Full equity curve (starts at 1.0).
    """

    name: str
    sharpe: float
    calmar: float
    max_drawdown: float
    ann_return: float
    ann_vol: float
    n_trades: int
    equity: pd.Series


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tearsheet_to_baseline(name: str, bt: BacktestResult, ann_factor: int) -> BaselineResult:
    ts = backtest_tearsheet(bt, ann_factor=ann_factor)
    return BaselineResult(
        name=name,
        sharpe=float(ts["sharpe"]),
        calmar=float(ts["calmar"]),
        max_drawdown=float(ts["max_drawdown"]),
        ann_return=float(ts["ann_return"]),
        ann_vol=float(ts["ann_vol"]),
        n_trades=int(ts["n_trades"]),
        equity=bt.equity,
    )


def _constant_regime_backtest(
    returns: pd.Series,
    position: float,
    n_states: int,
    ann_factor: int,
    transaction_cost: float,
) -> BacktestResult:
    """Backtest a single constant-position strategy (mapped to regime 0)."""
    rules = [RegimeRule(target_position=position)] + [
        RegimeRule(target_position=0.0) for _ in range(n_states - 1)
    ]
    cfg = RegimeStrategyConfig(rules=rules, transaction_cost=transaction_cost, ann_factor=ann_factor)
    regimes = np.zeros(len(returns), dtype=int)
    return run_backtest(returns, regimes, cfg)


# ---------------------------------------------------------------------------
# Public baselines
# ---------------------------------------------------------------------------


def buy_and_hold(
    returns: pd.Series,
    ann_factor: int = 8760,
) -> BaselineResult:
    """Always long (+1.0). No transaction costs (position never changes).

    Parameters
    ----------
    returns : pd.Series
    ann_factor : int

    Returns
    -------
    BaselineResult
    """
    bt = _constant_regime_backtest(returns, 1.0, 1, ann_factor, 0.0)
    return _tearsheet_to_baseline("buy_and_hold", bt, ann_factor)


def vol_targeted_buy_and_hold(
    returns: pd.Series,
    ann_factor: int = 8760,
    target_vol: float = 0.15,
    vol_window: int = 24,
    transaction_cost: float = 0.0001,
) -> BaselineResult:
    """Scale position by realized-vol target; cap at 1.0.

    ``position_t = min(target_vol / (realized_vol_t * sqrt(ann_factor)), 1.0)``

    Uses an expanding window for the first ``vol_window`` bars so the position
    is always computable without look-ahead.

    Parameters
    ----------
    returns : pd.Series
    ann_factor : int
    target_vol : float
        Annualised volatility target (default 15%).
    vol_window : int
        Rolling window for realized vol estimation (default 24 bars).
    transaction_cost : float

    Returns
    -------
    BaselineResult
    """
    r = returns.values
    T = len(r)
    positions = np.zeros(T)
    for t in range(T):
        window = r[max(0, t - vol_window + 1): t + 1]
        real_vol = float(np.std(window, ddof=1)) if len(window) > 1 else 1e-9
        ann_vol = real_vol * np.sqrt(ann_factor)
        positions[t] = min(target_vol / (ann_vol + 1e-15), 1.0)

    # Simulate manually for variable positions
    equity = np.ones(T + 1)
    strat_ret = np.zeros(T)
    trades_count = 0
    cur_pos = 0.0
    for t in range(T):
        target = positions[t]
        delta = abs(target - cur_pos)
        cost = delta * transaction_cost
        if delta > 1e-10:
            trades_count += 1
        bar_ret = cur_pos * r[t] - cost
        strat_ret[t] = bar_ret
        equity[t + 1] = equity[t] * (1.0 + bar_ret)
        cur_pos = target

    idx = returns.index
    equity_series = pd.Series(equity[1:], index=idx)
    strat_series = pd.Series(strat_ret, index=idx)
    bt = BacktestResult(
        equity=equity_series,
        returns=strat_series,
        positions=pd.Series(positions, index=idx),
        trades=pd.DataFrame({"bar": [], "direction": [], "size": [], "cost": []}),
        regime_labels=np.zeros(T, dtype=int),
    )
    # Override n_trades in tearsheet manually — tearsheet uses len(trades) df
    ts = backtest_tearsheet(bt, ann_factor=ann_factor)
    return BaselineResult(
        name="vol_targeted_bah",
        sharpe=float(ts["sharpe"]),
        calmar=float(ts["calmar"]),
        max_drawdown=float(ts["max_drawdown"]),
        ann_return=float(ts["ann_return"]),
        ann_vol=float(ts["ann_vol"]),
        n_trades=trades_count,
        equity=equity_series,
    )


def naive_momentum(
    returns: pd.Series,
    lookback_bars: int = 24,
    ann_factor: int = 8760,
    transaction_cost: float = 0.0001,
) -> BaselineResult:
    """Long when rolling mean return is positive; flat otherwise.

    Uses a causal rolling window: at bar ``t`` only data from
    ``[t - lookback_bars + 1, t]`` is used.

    Parameters
    ----------
    returns : pd.Series
    lookback_bars : int
    ann_factor : int
    transaction_cost : float

    Returns
    -------
    BaselineResult
    """
    r = returns.values
    T = len(r)
    regimes = np.zeros(T, dtype=int)
    for t in range(T):
        window = r[max(0, t - lookback_bars + 1): t + 1]
        regimes[t] = 1 if float(window.mean()) > 0 else 0

    # regime 0 = flat, regime 1 = long (matches the assignment above)
    rules = [RegimeRule(target_position=0.0), RegimeRule(target_position=1.0)]
    cfg = RegimeStrategyConfig(rules=rules, transaction_cost=transaction_cost, ann_factor=ann_factor)
    bt = run_backtest(returns, regimes, cfg)
    return _tearsheet_to_baseline("naive_momentum", bt, ann_factor)


def naive_vol_regime(
    returns: pd.Series,
    vol_window: int = 24,
    ann_factor: int = 8760,
    transaction_cost: float = 0.0001,
) -> BaselineResult:
    """Long when realized vol is below expanding historical median; flat otherwise.

    The expanding median is computed only from data seen so far (causal).

    Parameters
    ----------
    returns : pd.Series
    vol_window : int
        Window for realized vol computation.
    ann_factor : int
    transaction_cost : float

    Returns
    -------
    BaselineResult
    """
    r = returns.values
    T = len(r)
    realized_vols = np.zeros(T)
    for t in range(T):
        window = r[max(0, t - vol_window + 1): t + 1]
        realized_vols[t] = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0

    regimes = np.zeros(T, dtype=int)
    for t in range(T):
        if t == 0:
            regimes[t] = 1  # no history, default to long
        else:
            expanding_median = float(np.median(realized_vols[:t]))
            regimes[t] = 1 if realized_vols[t] <= expanding_median else 0

    rules = [RegimeRule(target_position=0.0), RegimeRule(target_position=1.0)]
    cfg = RegimeStrategyConfig(rules=rules, transaction_cost=transaction_cost, ann_factor=ann_factor)
    bt = run_backtest(returns, regimes, cfg)
    return _tearsheet_to_baseline("naive_vol_regime", bt, ann_factor)


def two_state_hmm_baseline(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    returns_col: str,
    ann_factor: int = 8760,
    transaction_cost: float = 0.0001,
    train_frac: float = 0.7,
    train_kwargs: dict | None = None,
) -> BaselineResult:
    """Minimal 2-state Gaussian HMM with the same default strategy.

    Shows whether adding more states (beyond 2) actually helps. Uses
    causal :class:`OnlineDecoder` decoding on the test window.

    Parameters
    ----------
    df_features : pd.DataFrame
    feature_cols : list[str]
    returns_col : str
    ann_factor : int
    transaction_cost : float
    train_frac : float
        Fraction of data used for training (default 0.7).
    train_kwargs : dict | None

    Returns
    -------
    BaselineResult
    """
    kw = dict(train_kwargs or {})
    kw["feature_names"] = feature_cols

    n = len(df_features)
    split = int(n * train_frac)
    X_all = df_features[feature_cols].values
    X_train = X_all[:split]
    X_test = X_all[split:]
    ret_test = pd.Series(df_features[returns_col].values[split:])

    fitted = train_hmm(X_train, n_states=2, **kw)
    strategy = _default_strategy_config(fitted, ann_factor, transaction_cost)
    decoder = OnlineDecoder(fitted)
    posteriors = decoder.batch_filter(X_test)
    regimes = posteriors.argmax(axis=1)

    bt = run_backtest(ret_test, regimes, strategy)
    return _tearsheet_to_baseline("two_state_hmm", bt, ann_factor)


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------


def run_all_baselines(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    returns_col: str,
    ann_factor: int = 8760,
    transaction_cost: float = 0.0001,
    train_kwargs: dict | None = None,
) -> dict[str, BaselineResult]:
    """Run all five baselines and return a dict keyed by name.

    Parameters
    ----------
    df_features : pd.DataFrame
    feature_cols : list[str]
    returns_col : str
    ann_factor : int
    transaction_cost : float
    train_kwargs : dict | None

    Returns
    -------
    dict[str, BaselineResult]
    """
    returns = df_features[returns_col]
    results: dict[str, BaselineResult] = {}

    logger.info("Running buy-and-hold baseline.")
    results["buy_and_hold"] = buy_and_hold(returns, ann_factor=ann_factor)

    logger.info("Running vol-targeted buy-and-hold baseline.")
    results["vol_targeted_bah"] = vol_targeted_buy_and_hold(
        returns, ann_factor=ann_factor, transaction_cost=transaction_cost,
    )

    logger.info("Running naive momentum baseline.")
    results["naive_momentum"] = naive_momentum(
        returns, ann_factor=ann_factor, transaction_cost=transaction_cost,
    )

    logger.info("Running naive vol-regime baseline.")
    results["naive_vol_regime"] = naive_vol_regime(
        returns, ann_factor=ann_factor, transaction_cost=transaction_cost,
    )

    logger.info("Running 2-state HMM baseline.")
    results["two_state_hmm"] = two_state_hmm_baseline(
        df_features, feature_cols, returns_col,
        ann_factor=ann_factor, transaction_cost=transaction_cost,
        train_kwargs=train_kwargs,
    )

    return results


def compare_model_to_baselines(
    model_sharpe: float,
    baselines: dict[str, BaselineResult],
    model_se: float = 0.0,
) -> pd.DataFrame:
    """Compare model Sharpe against all baselines.

    Parameters
    ----------
    model_sharpe : float
        Mean Sharpe across CV folds.
    baselines : dict[str, BaselineResult]
    model_se : float
        Standard error of the model Sharpe across folds (0 = no SE info).

    Returns
    -------
    pd.DataFrame
        Columns: name, baseline_sharpe, model_sharpe, beats, margin.
        ``beats`` is ``True`` when ``model_sharpe > baseline_sharpe``.
    """
    rows = []
    for name, br in baselines.items():
        rows.append({
            "name": name,
            "baseline_sharpe": br.sharpe,
            "model_sharpe": model_sharpe,
            "model_se": model_se,
            "beats": model_sharpe > br.sharpe,
            "margin": model_sharpe - br.sharpe,
        })
    return pd.DataFrame(rows)
