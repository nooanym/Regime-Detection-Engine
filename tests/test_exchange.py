"""Tests for rde.trading.exchange (Phase 34)."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from rde.trading.exchange import (
    Balance,
    BinanceTestnetExchange,
    ExchangeABC,
    MockExchange,
    OrderRequest,
    OrderResult,
)


# ---------------------------------------------------------------------------
# OrderRequest / OrderResult / Balance
# ---------------------------------------------------------------------------

class TestDataClasses:
    def test_order_request_defaults(self):
        req = OrderRequest(symbol="BTC/USDT", side="buy", quantity=0.5)
        assert req.order_type == "market"
        assert req.limit_price is None

    def test_order_request_limit(self):
        req = OrderRequest(
            symbol="ETH/USDT", side="sell", quantity=1.0,
            order_type="limit", limit_price=3000.0,
        )
        assert req.limit_price == 3000.0

    def test_balance_total_computed(self):
        b = Balance(asset="BTC", free=1.5, locked=0.5)
        assert b.total == pytest.approx(2.0)

    def test_order_result_fields(self):
        ts = pd.Timestamp.now(tz="UTC")
        r = OrderResult(
            order_id="abc", symbol="BTC/USDT", side="buy",
            quantity=1.0, filled_quantity=1.0,
            average_price=50_000.0, status="filled", timestamp=ts,
        )
        assert r.fee == 0.0
        assert r.status == "filled"


# ---------------------------------------------------------------------------
# MockExchange
# ---------------------------------------------------------------------------

class TestMockExchangeInit:
    def test_default_empty_balances(self):
        ex = MockExchange()
        b = ex.get_balance("USDT")
        assert b.free == 0.0

    def test_custom_initial_balances(self):
        ex = MockExchange(initial_balances={"USDT": 10_000.0, "BTC": 0.5})
        assert ex.get_balance("USDT").free == pytest.approx(10_000.0)
        assert ex.get_balance("BTC").free == pytest.approx(0.5)

    def test_unknown_asset_balance_zero(self):
        ex = MockExchange()
        assert ex.get_balance("ETH").free == 0.0


class TestMockExchangePlaceOrder:
    def test_buy_deducts_quote_adds_base(self):
        ex = MockExchange(initial_balances={"USDT": 10_000.0, "BTC": 0.0})
        ex.set_price("BTC/USDT", 50_000.0)
        result = ex.place_order(OrderRequest("BTC/USDT", "buy", 0.1))
        assert result.status == "filled"
        assert result.filled_quantity == pytest.approx(0.1)
        assert ex.get_balance("BTC").free == pytest.approx(0.1)
        assert ex.get_balance("USDT").free == pytest.approx(5_000.0)

    def test_sell_deducts_base_adds_quote(self):
        ex = MockExchange(initial_balances={"USDT": 0.0, "BTC": 1.0})
        ex.set_price("BTC/USDT", 50_000.0)
        result = ex.place_order(OrderRequest("BTC/USDT", "sell", 0.5))
        assert result.status == "filled"
        assert ex.get_balance("BTC").free == pytest.approx(0.5)
        assert ex.get_balance("USDT").free == pytest.approx(25_000.0)

    def test_slippage_buy_increases_price(self):
        ex = MockExchange(
            initial_balances={"USDT": 100_000.0},
            slippage_bps=10,
        )
        ex.set_price("BTC/USDT", 50_000.0)
        result = ex.place_order(OrderRequest("BTC/USDT", "buy", 1.0))
        expected = 50_000.0 * (1 + 10 / 10_000)
        assert result.average_price == pytest.approx(expected)

    def test_slippage_sell_decreases_price(self):
        ex = MockExchange(
            initial_balances={"BTC": 1.0},
            slippage_bps=10,
        )
        ex.set_price("BTC/USDT", 50_000.0)
        result = ex.place_order(OrderRequest("BTC/USDT", "sell", 1.0))
        expected = 50_000.0 * (1 - 10 / 10_000)
        assert result.average_price == pytest.approx(expected)

    def test_insufficient_quote_raises(self):
        ex = MockExchange(initial_balances={"USDT": 100.0})
        ex.set_price("BTC/USDT", 50_000.0)
        with pytest.raises(ValueError, match="Insufficient USDT"):
            ex.place_order(OrderRequest("BTC/USDT", "buy", 1.0))

    def test_insufficient_base_raises(self):
        ex = MockExchange(initial_balances={"BTC": 0.001})
        ex.set_price("BTC/USDT", 50_000.0)
        with pytest.raises(ValueError, match="Insufficient BTC"):
            ex.place_order(OrderRequest("BTC/USDT", "sell", 1.0))

    def test_fill_price_override(self):
        ex = MockExchange(
            initial_balances={"USDT": 100_000.0},
            fill_price_override=99_999.0,
        )
        ex.set_price("BTC/USDT", 50_000.0)  # should be ignored
        result = ex.place_order(OrderRequest("BTC/USDT", "buy", 0.1))
        assert result.average_price == pytest.approx(99_999.0)

    def test_limit_order_uses_limit_price(self):
        ex = MockExchange(initial_balances={"USDT": 100_000.0})
        result = ex.place_order(
            OrderRequest("BTC/USDT", "buy", 0.1, order_type="limit", limit_price=48_000.0)
        )
        assert result.average_price == pytest.approx(48_000.0)

    def test_limit_order_missing_price_raises(self):
        ex = MockExchange(initial_balances={"USDT": 100_000.0})
        with pytest.raises(ValueError, match="limit_price must be set"):
            ex.place_order(
                OrderRequest("BTC/USDT", "buy", 0.1, order_type="limit")
            )

    def test_no_price_set_raises(self):
        ex = MockExchange(initial_balances={"USDT": 100_000.0})
        with pytest.raises(ValueError, match="No price set"):
            ex.place_order(OrderRequest("BTC/USDT", "buy", 0.1))


class TestMockExchangeHistory:
    def test_order_history_accumulates(self):
        ex = MockExchange(initial_balances={"USDT": 200_000.0})
        ex.set_price("BTC/USDT", 50_000.0)
        ex.place_order(OrderRequest("BTC/USDT", "buy", 0.1))
        ex.place_order(OrderRequest("BTC/USDT", "buy", 0.1))
        assert len(ex.order_history) == 2

    def test_reset_clears_orders_and_balances(self):
        ex = MockExchange(initial_balances={"USDT": 10_000.0, "BTC": 0.0})
        ex.set_price("BTC/USDT", 5_000.0)
        ex.place_order(OrderRequest("BTC/USDT", "buy", 0.5))
        ex.reset()
        assert len(ex.order_history) == 0
        assert ex.get_balance("USDT").free == pytest.approx(10_000.0)
        assert ex.get_balance("BTC").free == pytest.approx(0.0)

    def test_cancel_order_returns_false(self):
        ex = MockExchange(initial_balances={"USDT": 100_000.0})
        ex.set_price("BTC/USDT", 50_000.0)
        result = ex.place_order(OrderRequest("BTC/USDT", "buy", 0.1))
        assert ex.cancel_order(result.order_id, "BTC/USDT") is False

    def test_get_last_price(self):
        ex = MockExchange()
        ex.set_price("BTC/USDT", 55_000.0)
        assert ex.get_last_price("BTC/USDT") == pytest.approx(55_000.0)

    def test_get_last_price_unknown_raises(self):
        ex = MockExchange()
        with pytest.raises(KeyError):
            ex.get_last_price("BTC/USDT")

    def test_round_trip_balance_correct(self):
        ex = MockExchange(initial_balances={"USDT": 10_000.0, "BTC": 0.0})
        ex.set_price("BTC/USDT", 50_000.0)
        ex.place_order(OrderRequest("BTC/USDT", "buy", 0.1))
        ex.set_price("BTC/USDT", 60_000.0)
        ex.place_order(OrderRequest("BTC/USDT", "sell", 0.1))
        assert ex.get_balance("BTC").free == pytest.approx(0.0)
        assert ex.get_balance("USDT").free == pytest.approx(11_000.0)


# ---------------------------------------------------------------------------
# BinanceTestnetExchange — requires env vars + ccxt (mock both)
# ---------------------------------------------------------------------------

class TestBinanceTestnetExchange:
    def test_raises_if_ccxt_missing(self):
        with patch("rde.trading.exchange._ccxt", None):
            with pytest.raises(ImportError, match="ccxt is required"):
                BinanceTestnetExchange()

    def test_raises_if_env_vars_missing(self):
        mock_ccxt = MagicMock()
        with patch("rde.trading.exchange._ccxt", mock_ccxt):
            with patch.dict(os.environ, {}, clear=True):
                # Ensure the keys are absent
                os.environ.pop("BINANCE_TESTNET_API_KEY", None)
                os.environ.pop("BINANCE_TESTNET_SECRET", None)
                with pytest.raises(ValueError, match="BINANCE_TESTNET_API_KEY"):
                    BinanceTestnetExchange()

    def test_initialises_with_valid_env(self):
        mock_ccxt = MagicMock()
        mock_ccxt.binance.return_value = MagicMock()
        with patch("rde.trading.exchange._ccxt", mock_ccxt):
            with patch.dict(os.environ, {
                "BINANCE_TESTNET_API_KEY": "test_key",
                "BINANCE_TESTNET_SECRET": "test_secret",
            }):
                ex = BinanceTestnetExchange()
                assert isinstance(ex, ExchangeABC)
