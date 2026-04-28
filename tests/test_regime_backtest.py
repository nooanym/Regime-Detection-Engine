"""Tests for rde.analysis.backtest — regime-aware engine (Phase 29)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.analysis.backtest import (
    BacktestResult,
    RegimeRule,
    RegimeStrategyConfig,
    backtest_tearsheet,
    regime_performance_attribution,
    run_backtest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _data(T: int = 200, K: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    r = pd.Series(rng.standard_normal(T) * 0.01)
    regimes = rng.integers(0, K, T)
    return r, regimes


def _long_flat_config() -> RegimeStrategyConfig:
    return RegimeStrategyConfig(
        rules=[RegimeRule(target_position=1.0), RegimeRule(target_position=0.0)],
        transaction_cost=0.0,
    )


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------


class TestRunBacktest:
    def test_returns_backtest_result(self) -> None:
        r, reg = _data()
        result = run_backtest(r, reg, _long_flat_config())
        assert isinstance(result, BacktestResult)

    def test_equity_starts_at_one(self) -> None:
        r, reg = _data()
        result = run_backtest(r, reg)
        assert result.equity.iloc[0] == pytest.approx(1.0, rel=0.01)

    def test_equity_length(self) -> None:
        T = 150
        r, reg = _data(T)
        result = run_backtest(r, reg)
        assert len(result.equity) == T

    def test_returns_length(self) -> None:
        T = 150
        r, reg = _data(T)
        result = run_backtest(r, reg)
        assert len(result.returns) == T

    def test_positions_length(self) -> None:
        T = 150
        r, reg = _data(T)
        result = run_backtest(r, reg)
        assert len(result.positions) == T

    def test_trades_dataframe(self) -> None:
        r, reg = _data()
        result = run_backtest(r, reg)
        assert isinstance(result.trades, pd.DataFrame)

    def test_flat_strategy_zero_returns(self) -> None:
        r, reg = _data()
        cfg = RegimeStrategyConfig(
            rules=[RegimeRule(target_position=0.0), RegimeRule(target_position=0.0)],
            transaction_cost=0.0,
        )
        result = run_backtest(r, reg, cfg)
        np.testing.assert_allclose(result.returns.values, 0.0, atol=1e-12)

    def test_always_long_returns_match_asset(self) -> None:
        T = 100
        rng = np.random.default_rng(0)
        r = pd.Series(rng.standard_normal(T) * 0.01)
        reg = np.zeros(T, dtype=int)
        cfg = RegimeStrategyConfig(
            rules=[RegimeRule(target_position=1.0)],
            transaction_cost=0.0,
        )
        result = run_backtest(r, reg, cfg)
        # Position set after each bar, so bar 0 position=0; bars 1+ match asset
        np.testing.assert_allclose(result.returns.values[1:], r.values[1:], atol=1e-10)

    def test_transaction_costs_reduce_returns(self) -> None:
        r, reg = _data()
        cfg_no = RegimeStrategyConfig(rules=[RegimeRule(1.0), RegimeRule(0.0)],
                                      transaction_cost=0.0)
        cfg_yes = RegimeStrategyConfig(rules=[RegimeRule(1.0), RegimeRule(0.0)],
                                       transaction_cost=0.01)
        r_no = run_backtest(r, reg, cfg_no).returns.sum()
        r_yes = run_backtest(r, reg, cfg_yes).returns.sum()
        assert r_no >= r_yes

    def test_stop_loss_limits_drawdown(self) -> None:
        T = 200
        rng = np.random.default_rng(5)
        r = pd.Series(np.concatenate([
            np.full(50, -0.005),
            rng.standard_normal(150) * 0.01,
        ]))
        reg = np.zeros(T, dtype=int)
        cfg_no = RegimeStrategyConfig(rules=[RegimeRule(1.0, stop_loss=None)],
                                      transaction_cost=0.0)
        cfg_stop = RegimeStrategyConfig(rules=[RegimeRule(1.0, stop_loss=0.1)],
                                        transaction_cost=0.0)
        from rde.analysis.backtest import _max_drawdown
        mdd_no = _max_drawdown(run_backtest(r, reg, cfg_no).equity.values)
        mdd_stop = _max_drawdown(run_backtest(r, reg, cfg_stop).equity.values)
        assert mdd_stop <= mdd_no + 1e-6

    def test_regime_labels_stored(self) -> None:
        r, reg = _data()
        result = run_backtest(r, reg)
        np.testing.assert_array_equal(result.regime_labels, reg)

    def test_auto_extends_rules_for_unknown_regimes(self) -> None:
        r, _ = _data()
        reg = np.ones(len(r), dtype=int) * 5
        cfg = RegimeStrategyConfig(rules=[RegimeRule(1.0)])
        result = run_backtest(r, reg, cfg)
        assert len(result.equity) == len(r)

    def test_no_nan_in_equity(self) -> None:
        r, reg = _data()
        result = run_backtest(r, reg)
        assert not result.equity.isnull().any()

    def test_preserves_index(self) -> None:
        T = 50
        idx = pd.date_range("2024-01-01", periods=T, freq="D")
        r = pd.Series(np.zeros(T), index=idx)
        reg = np.zeros(T, dtype=int)
        result = run_backtest(r, reg)
        assert (result.equity.index == idx).all()


# ---------------------------------------------------------------------------
# regime_performance_attribution
# ---------------------------------------------------------------------------


class TestRegimePerformanceAttribution:
    def test_returns_dataframe(self) -> None:
        r, reg = _data()
        result = run_backtest(r, reg)
        df = regime_performance_attribution(result)
        assert isinstance(df, pd.DataFrame)

    def test_has_all_row(self) -> None:
        r, reg = _data()
        result = run_backtest(r, reg)
        df = regime_performance_attribution(result)
        assert "all" in df["regime"].values

    def test_n_rows(self) -> None:
        K = 3
        r, reg = _data(K=K)
        result = run_backtest(r, reg)
        df = regime_performance_attribution(result)
        assert len(df) == K + 1

    def test_n_bars_sum_to_total(self) -> None:
        T = 200
        r, reg = _data(T=T, K=2)
        result = run_backtest(r, reg)
        df = regime_performance_attribution(result)
        regime_rows = df[df["regime"] != "all"]
        assert regime_rows["n_bars"].sum() == T

    def test_columns_present(self) -> None:
        r, reg = _data()
        result = run_backtest(r, reg)
        df = regime_performance_attribution(result)
        for col in ["regime", "n_bars", "total_return", "ann_vol", "sharpe", "win_rate"]:
            assert col in df.columns

    def test_win_rate_in_range(self) -> None:
        r, reg = _data()
        result = run_backtest(r, reg)
        df = regime_performance_attribution(result)
        numeric = df[df["win_rate"].notna()]["win_rate"].astype(float)
        assert (numeric >= 0).all() and (numeric <= 1).all()


# ---------------------------------------------------------------------------
# backtest_tearsheet
# ---------------------------------------------------------------------------


class TestBacktestTearsheet:
    def test_returns_series(self) -> None:
        r, reg = _data()
        result = run_backtest(r, reg)
        ts = backtest_tearsheet(result)
        assert isinstance(ts, pd.Series)

    def test_metrics_present(self) -> None:
        r, reg = _data()
        result = run_backtest(r, reg)
        ts = backtest_tearsheet(result)
        for m in ["total_return", "ann_return", "ann_vol", "sharpe",
                  "sortino", "max_drawdown", "calmar", "win_rate"]:
            assert m in ts.index

    def test_max_drawdown_non_negative(self) -> None:
        r, reg = _data()
        result = run_backtest(r, reg)
        ts = backtest_tearsheet(result)
        assert ts["max_drawdown"] >= 0.0

    def test_win_rate_in_range(self) -> None:
        r, reg = _data()
        result = run_backtest(r, reg)
        ts = backtest_tearsheet(result)
        assert 0.0 <= ts["win_rate"] <= 1.0

    def test_flat_strategy_zero_ann_vol(self) -> None:
        r, _ = _data()
        reg = np.zeros(len(r), dtype=int)
        cfg = RegimeStrategyConfig(rules=[RegimeRule(target_position=0.0)],
                                   transaction_cost=0.0)
        result = run_backtest(r, reg, cfg)
        ts = backtest_tearsheet(result)
        assert ts["ann_vol"] == pytest.approx(0.0, abs=1e-10)

    def test_n_bars_correct(self) -> None:
        T = 180
        r, reg = _data(T)
        result = run_backtest(r, reg)
        ts = backtest_tearsheet(result)
        assert ts["n_bars"] == float(T)
