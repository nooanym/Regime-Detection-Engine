"""Tests for rde.trading (Phase 34 — paper trading core).

Covers:
- PortfolioConfig defaults and custom values
- PaperPortfolio.execute: buy from flat, sell from long, no-op at target
- Slippage calculation for buys and sells
- mark_to_market correctness
- realized_pnl across round-trip trades
- equity_curve produces a Series indexed by fill timestamps
- TradeLog.append / to_dataframe shape
- TradeLog save/load round-trip (parquet + CSV)
- RegimeSignalStrategy.target_weight: rule lookup and default fallback
- RegimeSignalStrategy.target_quantity: equity * weight / price
- RegimeSignalStrategy.signal_changed: True/False cases
- Full integration: strategy → portfolio → trade log over 5 synthetic bars
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from rde.trading import (
    Fill,
    PaperPortfolio,
    PortfolioConfig,
    RegimeRule,
    RegimeSignalStrategy,
    SignalStrategyConfig,
    TradeLog,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TS = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
TS1 = pd.Timestamp("2024-01-01 01:00:00", tz="UTC")
TS2 = pd.Timestamp("2024-01-01 02:00:00", tz="UTC")
TS3 = pd.Timestamp("2024-01-01 03:00:00", tz="UTC")
TS4 = pd.Timestamp("2024-01-01 04:00:00", tz="UTC")
TS5 = pd.Timestamp("2024-01-01 05:00:00", tz="UTC")


# ---------------------------------------------------------------------------
# PortfolioConfig
# ---------------------------------------------------------------------------


def test_portfolio_config_defaults():
    cfg = PortfolioConfig()
    assert cfg.initial_capital == 100_000.0
    assert cfg.slippage_bps == 5.0
    assert cfg.commission_flat == 0.0


def test_portfolio_config_custom():
    cfg = PortfolioConfig(initial_capital=50_000.0, slippage_bps=10.0, commission_flat=1.5)
    assert cfg.initial_capital == 50_000.0
    assert cfg.slippage_bps == 10.0
    assert cfg.commission_flat == 1.5


# ---------------------------------------------------------------------------
# PaperPortfolio — execute
# ---------------------------------------------------------------------------


def test_execute_buy_from_flat():
    """Buying from a flat position creates a long position and reduces cash."""
    cfg = PortfolioConfig(initial_capital=10_000.0, slippage_bps=0.0, commission_flat=0.0)
    portfolio = PaperPortfolio(cfg, symbol="BTC")
    fill = portfolio.execute(TS, "BTC", target_quantity=1.0, price=1000.0)

    assert fill is not None
    assert fill.side == "buy"
    assert fill.quantity == pytest.approx(1.0)
    assert portfolio.position == pytest.approx(1.0)
    assert portfolio.cash == pytest.approx(9_000.0)


def test_execute_sell_from_long():
    """Selling from a long position reduces units and increases cash."""
    cfg = PortfolioConfig(initial_capital=10_000.0, slippage_bps=0.0, commission_flat=0.0)
    portfolio = PaperPortfolio(cfg, symbol="BTC")
    portfolio.execute(TS, "BTC", target_quantity=2.0, price=1000.0)
    fill = portfolio.execute(TS1, "BTC", target_quantity=0.0, price=1100.0)

    assert fill is not None
    assert fill.side == "sell"
    assert fill.quantity == pytest.approx(2.0)
    assert portfolio.position == pytest.approx(0.0)


def test_execute_noop_when_at_target():
    """No fill is generated when the position is already at the target."""
    cfg = PortfolioConfig(initial_capital=10_000.0, slippage_bps=0.0, commission_flat=0.0)
    portfolio = PaperPortfolio(cfg, symbol="BTC")
    portfolio.execute(TS, "BTC", target_quantity=1.0, price=1000.0)
    fill = portfolio.execute(TS1, "BTC", target_quantity=1.0, price=1100.0)

    assert fill is None
    assert portfolio.position == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Slippage calculation
# ---------------------------------------------------------------------------


def test_slippage_buy_raises_fill_price():
    """Buy slippage increases the effective fill price above mid."""
    cfg = PortfolioConfig(initial_capital=100_000.0, slippage_bps=10.0, commission_flat=0.0)
    portfolio = PaperPortfolio(cfg, symbol="BTC")
    price = 2000.0
    fill = portfolio.execute(TS, "BTC", target_quantity=1.0, price=price)

    assert fill is not None
    expected_slip_dollar = 1.0 * price * 10.0 / 10_000.0
    assert fill.slippage == pytest.approx(expected_slip_dollar)
    # Effective fill price = price * (1 + bps/10000)
    assert fill.fill_price == pytest.approx(price * (1 + 10.0 / 10_000.0))


def test_slippage_sell_lowers_fill_price():
    """Sell slippage decreases the effective fill price below mid."""
    cfg = PortfolioConfig(initial_capital=100_000.0, slippage_bps=10.0, commission_flat=0.0)
    portfolio = PaperPortfolio(cfg, symbol="BTC")
    # First buy (no slippage concern for this assertion)
    portfolio.execute(TS, "BTC", target_quantity=1.0, price=2000.0)
    price = 2500.0
    fill = portfolio.execute(TS1, "BTC", target_quantity=0.0, price=price)

    assert fill is not None
    expected_slip_dollar = 1.0 * price * 10.0 / 10_000.0
    assert fill.slippage == pytest.approx(expected_slip_dollar)
    # Effective fill price = price * (1 - bps/10000)
    assert fill.fill_price == pytest.approx(price * (1 - 10.0 / 10_000.0))


def test_slippage_zero_bps():
    """With slippage_bps=0, fill price equals mid price."""
    cfg = PortfolioConfig(initial_capital=10_000.0, slippage_bps=0.0, commission_flat=0.0)
    portfolio = PaperPortfolio(cfg, symbol="BTC")
    fill = portfolio.execute(TS, "BTC", target_quantity=1.0, price=500.0)

    assert fill is not None
    assert fill.slippage == pytest.approx(0.0)
    assert fill.fill_price == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# mark_to_market
# ---------------------------------------------------------------------------


def test_mark_to_market_flat():
    """MTM with no position equals cash."""
    cfg = PortfolioConfig(initial_capital=10_000.0, slippage_bps=0.0)
    portfolio = PaperPortfolio(cfg)
    assert portfolio.mark_to_market(1000.0) == pytest.approx(10_000.0)


def test_mark_to_market_with_position():
    """MTM = cash + position * price."""
    cfg = PortfolioConfig(initial_capital=10_000.0, slippage_bps=0.0, commission_flat=0.0)
    portfolio = PaperPortfolio(cfg, symbol="BTC")
    portfolio.execute(TS, "BTC", target_quantity=2.0, price=1000.0)
    # cash = 10000 - 2*1000 = 8000; position = 2 units at price 1200
    mtm = portfolio.mark_to_market(1200.0)
    assert mtm == pytest.approx(8_000.0 + 2.0 * 1_200.0)


# ---------------------------------------------------------------------------
# realized_pnl
# ---------------------------------------------------------------------------


def test_realized_pnl_round_trip():
    """Round-trip trade produces correct realized P&L."""
    cfg = PortfolioConfig(initial_capital=10_000.0, slippage_bps=0.0, commission_flat=0.0)
    portfolio = PaperPortfolio(cfg, symbol="BTC")
    buy_price = 1000.0
    sell_price = 1200.0
    qty = 1.0

    portfolio.execute(TS, "BTC", target_quantity=qty, price=buy_price)
    portfolio.execute(TS1, "BTC", target_quantity=0.0, price=sell_price)

    expected_pnl = qty * (sell_price - buy_price)
    assert portfolio.realized_pnl() == pytest.approx(expected_pnl)


def test_realized_pnl_accumulates():
    """Multiple round trips accumulate realized P&L."""
    cfg = PortfolioConfig(initial_capital=50_000.0, slippage_bps=0.0, commission_flat=0.0)
    portfolio = PaperPortfolio(cfg, symbol="BTC")

    portfolio.execute(TS, "BTC", target_quantity=1.0, price=1000.0)
    portfolio.execute(TS1, "BTC", target_quantity=0.0, price=1100.0)  # +100
    portfolio.execute(TS2, "BTC", target_quantity=2.0, price=1100.0)
    portfolio.execute(TS3, "BTC", target_quantity=0.0, price=900.0)   # -200

    expected = 1.0 * (1100.0 - 1000.0) + 2.0 * (900.0 - 1100.0)
    assert portfolio.realized_pnl() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# equity_curve
# ---------------------------------------------------------------------------


def test_equity_curve_returns_series():
    """equity_curve returns a Series indexed by fill timestamps."""
    cfg = PortfolioConfig(initial_capital=10_000.0, slippage_bps=0.0, commission_flat=0.0)
    portfolio = PaperPortfolio(cfg, symbol="BTC")
    portfolio.execute(TS, "BTC", target_quantity=1.0, price=1000.0)
    portfolio.execute(TS1, "BTC", target_quantity=0.0, price=1100.0)

    prices = pd.Series({TS: 1000.0, TS1: 1100.0})
    curve = portfolio.equity_curve(prices)

    assert isinstance(curve, pd.Series)
    assert len(curve) == 2
    assert TS in curve.index
    assert TS1 in curve.index


def test_equity_curve_empty_portfolio():
    """equity_curve returns empty Series when no fills exist."""
    cfg = PortfolioConfig(initial_capital=10_000.0)
    portfolio = PaperPortfolio(cfg)
    prices = pd.Series(dtype=float)
    curve = portfolio.equity_curve(prices)
    assert isinstance(curve, pd.Series)
    assert len(curve) == 0


# ---------------------------------------------------------------------------
# TradeLog
# ---------------------------------------------------------------------------


def _make_fill(ts: pd.Timestamp = TS) -> Fill:
    return Fill(
        timestamp=ts,
        symbol="BTC",
        side="buy",
        quantity=1.0,
        price=1000.0,
        slippage=0.5,
        commission=0.0,
    )


def test_trade_log_append_and_len():
    log = TradeLog()
    assert len(log) == 0
    log.append(_make_fill())
    assert len(log) == 1
    log.append(_make_fill(TS1))
    assert len(log) == 2


def test_trade_log_to_dataframe_shape():
    log = TradeLog()
    log.append(_make_fill(TS))
    log.append(_make_fill(TS1))
    df = log.to_dataframe()
    assert df.shape == (2, 7)
    assert list(df.columns) == ["timestamp", "symbol", "side", "quantity", "price", "slippage", "commission"]


def test_trade_log_to_dataframe_empty():
    log = TradeLog()
    df = log.to_dataframe()
    assert df.empty
    assert list(df.columns) == ["timestamp", "symbol", "side", "quantity", "price", "slippage", "commission"]


def test_trade_log_iter():
    log = TradeLog()
    f1 = _make_fill(TS)
    f2 = _make_fill(TS1)
    log.append(f1)
    log.append(f2)
    fills = list(log)
    assert fills[0] is f1
    assert fills[1] is f2


def test_trade_log_parquet_round_trip(tmp_path: Path):
    log = TradeLog()
    log.append(_make_fill(TS))
    log.append(Fill(TS1, "ETH", "sell", 0.5, 2000.0, 1.0, 0.0))
    path = tmp_path / "fills.parquet"
    log.save_parquet(path)

    loaded = TradeLog.load_parquet(path)
    assert len(loaded) == 2
    fills = list(loaded)
    assert fills[0].symbol == "BTC"
    assert fills[0].side == "buy"
    assert fills[1].symbol == "ETH"
    assert fills[1].side == "sell"
    assert fills[1].quantity == pytest.approx(0.5)


def test_trade_log_csv_round_trip(tmp_path: Path):
    log = TradeLog()
    log.append(_make_fill(TS))
    path = tmp_path / "fills.csv"
    log.save_csv(path)
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "BTC"


# ---------------------------------------------------------------------------
# RegimeSignalStrategy — target_weight
# ---------------------------------------------------------------------------


def test_target_weight_rule_lookup():
    rules = [RegimeRule(regime=0, target_weight=1.0), RegimeRule(regime=1, target_weight=-0.5)]
    cfg = SignalStrategyConfig(rules=rules, default_weight=0.0)
    strategy = RegimeSignalStrategy(cfg)

    assert strategy.target_weight(0) == pytest.approx(1.0)
    assert strategy.target_weight(1) == pytest.approx(-0.5)


def test_target_weight_default_fallback():
    rules = [RegimeRule(regime=0, target_weight=1.0)]
    cfg = SignalStrategyConfig(rules=rules, default_weight=0.25)
    strategy = RegimeSignalStrategy(cfg)

    # Regime 2 not in rules → default
    assert strategy.target_weight(2) == pytest.approx(0.25)


def test_target_weight_empty_rules_uses_default():
    cfg = SignalStrategyConfig(rules=[], default_weight=-0.1)
    strategy = RegimeSignalStrategy(cfg)
    assert strategy.target_weight(0) == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# RegimeSignalStrategy — target_quantity
# ---------------------------------------------------------------------------


def test_target_quantity_math():
    """target_quantity = equity * weight / price."""
    rules = [RegimeRule(regime=0, target_weight=0.5)]
    cfg = SignalStrategyConfig(rules=rules, default_weight=0.0)
    strategy = RegimeSignalStrategy(cfg)

    equity = 10_000.0
    price = 500.0
    qty = strategy.target_quantity(regime=0, equity=equity, price=price)
    assert qty == pytest.approx(equity * 0.5 / price)


def test_target_quantity_flat_regime():
    rules = [RegimeRule(regime=1, target_weight=0.0)]
    cfg = SignalStrategyConfig(rules=rules)
    strategy = RegimeSignalStrategy(cfg)
    qty = strategy.target_quantity(regime=1, equity=50_000.0, price=100.0)
    assert qty == pytest.approx(0.0)


def test_target_quantity_invalid_price():
    cfg = SignalStrategyConfig(rules=[], default_weight=1.0)
    strategy = RegimeSignalStrategy(cfg)
    with pytest.raises(ValueError):
        strategy.target_quantity(regime=0, equity=10_000.0, price=0.0)


# ---------------------------------------------------------------------------
# RegimeSignalStrategy — signal_changed
# ---------------------------------------------------------------------------


def test_signal_changed_returns_true():
    rules = [
        RegimeRule(regime=0, target_weight=1.0),
        RegimeRule(regime=1, target_weight=0.0),
    ]
    cfg = SignalStrategyConfig(rules=rules, min_weight_change=0.01)
    strategy = RegimeSignalStrategy(cfg)

    assert strategy.signal_changed(prev_regime=0, new_regime=1) is True


def test_signal_changed_returns_false_same_regime():
    rules = [RegimeRule(regime=0, target_weight=0.5)]
    cfg = SignalStrategyConfig(rules=rules, min_weight_change=0.01)
    strategy = RegimeSignalStrategy(cfg)

    # Same regime → no change
    assert strategy.signal_changed(prev_regime=0, new_regime=0) is False


def test_signal_changed_below_threshold():
    """Weight change below min_weight_change is ignored."""
    rules = [
        RegimeRule(regime=0, target_weight=0.5000),
        RegimeRule(regime=1, target_weight=0.5005),  # diff = 0.0005 < 0.01
    ]
    cfg = SignalStrategyConfig(rules=rules, min_weight_change=0.01)
    strategy = RegimeSignalStrategy(cfg)

    assert strategy.signal_changed(prev_regime=0, new_regime=1) is False


# ---------------------------------------------------------------------------
# Full integration: strategy → portfolio → trade log over 5 bars
# ---------------------------------------------------------------------------


def test_integration_5_bars():
    """Simulate 5 bars with regime changes driving portfolio rebalances."""
    # Regimes: 0=long, 1=flat, 2=short
    rules = [
        RegimeRule(regime=0, target_weight=1.0),
        RegimeRule(regime=1, target_weight=0.0),
        RegimeRule(regime=2, target_weight=-1.0),
    ]
    strategy_cfg = SignalStrategyConfig(rules=rules, default_weight=0.0, min_weight_change=0.01)
    portfolio_cfg = PortfolioConfig(initial_capital=10_000.0, slippage_bps=5.0, commission_flat=0.0)

    strategy = RegimeSignalStrategy(strategy_cfg)
    portfolio = PaperPortfolio(portfolio_cfg, symbol="TEST")
    log = TradeLog()

    timestamps = [TS, TS1, TS2, TS3, TS4]
    prices = [100.0, 105.0, 110.0, 108.0, 112.0]
    regimes = [0, 0, 1, 2, 1]  # long, long, flat, short, flat

    prev_regime = None
    for ts, price, regime in zip(timestamps, prices, regimes):
        if prev_regime is None or strategy.signal_changed(prev_regime, regime):
            equity = portfolio.mark_to_market(price)
            target_qty = strategy.target_quantity(regime, equity, price)
            fill = portfolio.execute(ts, "TEST", target_quantity=target_qty, price=price)
            if fill is not None:
                log.append(fill)
        prev_regime = regime

    # Should have fills at bars 0 (go long), 2 (go flat), 3 (go short), 4 (go flat)
    assert len(log) >= 3

    df = log.to_dataframe()
    assert "timestamp" in df.columns
    assert "side" in df.columns

    # Portfolio should be approximately flat at end (regime=1, weight=0)
    assert portfolio.position == pytest.approx(0.0, abs=1e-6)

    # Equity should be positive
    final_equity = portfolio.mark_to_market(prices[-1])
    assert final_equity > 0
