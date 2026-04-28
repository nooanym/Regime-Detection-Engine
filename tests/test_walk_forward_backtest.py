"""Tests for rde.backtest.walk_forward_backtest (Phase 12).

Uses short synthetic data so no yfinance calls are made.
The walk-forward harness here uses a 7-day train window and daily
data so tests complete in seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.backtest.engine import BacktestConfig, BacktestResult
from rde.backtest.walk_forward_backtest import (
    WalkForwardBacktestConfig,
    WalkForwardBacktestResult,
    walk_forward_backtest,
)


# ── helpers ───────────────────────────────────────────────────────────────────

_N_STATES = 2
_TRAIN_DAYS = 30
_TOTAL_DAYS = 120  # gives ~3 monthly folds after warm-up


def _make_df(n: int = _TOTAL_DAYS, seed: int = 0, freq: str = "D") -> pd.DataFrame:
    """Synthetic daily feature + return DataFrame."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq=freq, tz="UTC")
    log_ret = rng.normal(0.0, 0.01, size=n)
    f1 = rng.standard_normal(n)
    f2 = np.abs(rng.standard_normal(n))
    return pd.DataFrame(
        {"log_return": log_ret, "feat1": f1, "feat2": f2},
        index=idx,
    )


def _cfg(**kwargs) -> WalkForwardBacktestConfig:
    return WalkForwardBacktestConfig(
        train_window=f"{_TRAIN_DAYS}d",
        n_restarts=1,
        **kwargs,
    )


def _run(df=None, **kwargs):
    if df is None:
        df = _make_df()
    cfg = _cfg(**kwargs)
    return walk_forward_backtest(
        df_features=df,
        n_states=_N_STATES,
        feature_cols=["feat1", "feat2"],
        config=cfg,
        seed_base=0,
    )


# ── return type ───────────────────────────────────────────────────────────────


def test_returns_wfbt_result():
    result = _run()
    assert isinstance(result, WalkForwardBacktestResult)


def test_backtest_is_backtest_result():
    result = _run()
    assert isinstance(result.backtest, BacktestResult)


# ── OOS coverage ─────────────────────────────────────────────────────────────


def test_oos_returns_is_series():
    result = _run()
    assert isinstance(result.oos_returns, pd.Series)


def test_oos_signal_is_series():
    result = _run()
    assert isinstance(result.oos_signal, pd.Series)


def test_oos_returns_and_signal_same_length():
    result = _run()
    assert len(result.oos_returns) == len(result.oos_signal)


def test_oos_returns_and_signal_same_index():
    result = _run()
    pd.testing.assert_index_equal(result.oos_returns.index, result.oos_signal.index)


def test_oos_bars_less_than_total():
    """OOS bars < total because first train_window is warm-up."""
    df = _make_df()
    result = _run(df=df)
    assert len(result.oos_returns) < len(df)


def test_no_oos_bar_in_warmup_window():
    """OOS bars must all be after the first recalibration boundary."""
    df = _make_df()
    result = _run(df=df)
    first_oos = result.oos_returns.index[0]
    expected_min = df.index[0] + pd.Timedelta(days=_TRAIN_DAYS)
    assert first_oos >= expected_min


# ── no-lookahead (equity curve) ───────────────────────────────────────────────


def test_equity_curve_starts_at_initial_capital():
    cfg = _cfg()
    cfg.backtest_config = BacktestConfig(initial_capital=10_000.0)
    result = walk_forward_backtest(
        df_features=_make_df(),
        n_states=_N_STATES,
        feature_cols=["feat1", "feat2"],
        config=cfg,
        seed_base=0,
    )
    assert result.backtest.equity_curve.iloc[0] == pytest.approx(10_000.0)


# ── fold summary ──────────────────────────────────────────────────────────────


def test_fold_summary_is_dataframe():
    result = _run()
    assert isinstance(result.fold_summary, pd.DataFrame)


def test_fold_summary_has_required_columns():
    result = _run()
    required = {"train_bars", "oos_bars", "train_log_lik"}
    assert required.issubset(result.fold_summary.columns)


def test_n_folds_matches_summary_rows():
    result = _run()
    assert result.n_folds == len(result.fold_summary)


def test_at_least_one_fold():
    result = _run()
    assert result.n_folds >= 1


def test_fold_train_bars_positive():
    result = _run()
    assert (result.fold_summary["train_bars"] > 0).all()


def test_fold_oos_bars_positive():
    result = _run()
    assert (result.fold_summary["oos_bars"] > 0).all()


# ── metrics ───────────────────────────────────────────────────────────────────

_REQUIRED_KEYS = {
    "total_return", "cagr", "sharpe_ann", "max_drawdown",
    "calmar", "hit_rate", "profit_factor", "n_trades", "avg_return_per_trade",
}


def test_metrics_keys_present():
    result = _run()
    assert _REQUIRED_KEYS.issubset(result.backtest.metrics.keys())


def test_max_drawdown_in_range():
    result = _run()
    mdd = result.backtest.metrics["max_drawdown"]
    if not np.isnan(mdd):
        assert 0.0 <= mdd <= 1.0


def test_hit_rate_in_range():
    result = _run()
    hr = result.backtest.metrics["hit_rate"]
    if not np.isnan(hr):
        assert 0.0 <= hr <= 1.0


# ── signal range ─────────────────────────────────────────────────────────────


def test_oos_signal_in_minus_one_plus_one():
    result = _run()
    sig = result.oos_signal.values
    assert np.all(sig >= -1.0 - 1e-9)
    assert np.all(sig <= 1.0 + 1e-9)


# ── error handling ────────────────────────────────────────────────────────────


def test_missing_return_col_raises():
    df = _make_df().drop(columns=["log_return"])
    with pytest.raises(ValueError, match="return_col"):
        walk_forward_backtest(
            df_features=df,
            n_states=_N_STATES,
            feature_cols=["feat1", "feat2"],
            config=_cfg(),
            seed_base=0,
        )


def test_missing_feature_col_raises():
    df = _make_df()
    with pytest.raises(ValueError, match="feature_col"):
        walk_forward_backtest(
            df_features=df,
            n_states=_N_STATES,
            feature_cols=["feat1", "nonexistent"],
            config=_cfg(),
            seed_base=0,
        )


def test_train_window_too_long_raises():
    df = _make_df(n=10)  # only 10 days of data
    cfg = WalkForwardBacktestConfig(train_window="365d", n_restarts=1)
    with pytest.raises(ValueError):
        walk_forward_backtest(
            df_features=df,
            n_states=_N_STATES,
            feature_cols=["feat1", "feat2"],
            config=cfg,
            seed_base=0,
        )


def test_non_datetime_index_raises():
    df = _make_df().reset_index(drop=True)  # integer index
    with pytest.raises(ValueError, match="DatetimeIndex"):
        walk_forward_backtest(
            df_features=df,
            n_states=_N_STATES,
            feature_cols=["feat1", "feat2"],
            config=_cfg(),
            seed_base=0,
        )
