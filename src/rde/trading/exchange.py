"""Exchange abstraction layer (Phase 34).

Provides a unified interface for order placement and balance queries across
paper trading (MockExchange) and live exchange connections (BinanceTestnetExchange).

Public API
----------
OrderRequest
OrderResult
Balance
ExchangeABC
MockExchange
BinanceTestnetExchange
"""

from __future__ import annotations

import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

# Soft import — ccxt is an optional dependency for live trading.
try:
    import ccxt as _ccxt  # type: ignore[import-not-found]
except ImportError:
    _ccxt = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class OrderRequest:
    """A request to place an order on an exchange.

    Attributes
    ----------
    symbol : str
        Trading pair, e.g. ``"BTC/USDT"``.
    side : {"buy", "sell"}
        Order direction.
    quantity : float
        Amount of base asset to trade.
    order_type : {"market", "limit"}
        Execution type.  Defaults to ``"market"``.
    limit_price : float or None
        Required for ``"limit"`` orders; ignored for ``"market"`` orders.
    """

    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    order_type: Literal["market", "limit"] = "market"
    limit_price: float | None = None


@dataclass
class OrderResult:
    """The outcome of a placed order.

    Attributes
    ----------
    order_id : str
        Exchange-assigned identifier (UUID for MockExchange).
    symbol : str
    side : {"buy", "sell"}
    quantity : float
        Requested quantity.
    filled_quantity : float
        Actually filled quantity (may differ for partial fills).
    average_price : float
        Volume-weighted average fill price.
    status : {"filled", "partial", "rejected", "open"}
    timestamp : pd.Timestamp
    fee : float
        Total fee in quote currency.
    """

    order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    filled_quantity: float
    average_price: float
    status: Literal["filled", "partial", "rejected", "open"]
    timestamp: pd.Timestamp
    fee: float = 0.0


@dataclass
class Balance:
    """Asset balance on an exchange.

    Attributes
    ----------
    asset : str
        Asset ticker, e.g. ``"USDT"``.
    free : float
        Available for trading.
    locked : float
        Reserved in open orders.
    total : float
        ``free + locked`` — computed automatically.
    """

    asset: str
    free: float
    locked: float
    total: float = field(init=False)

    def __post_init__(self) -> None:
        self.total = self.free + self.locked


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ExchangeABC(ABC):
    """Abstract interface for order placement and balance querying."""

    @abstractmethod
    def place_order(self, request: OrderRequest) -> OrderResult:
        """Place an order and return the fill result."""

    @abstractmethod
    def get_balance(self, asset: str) -> Balance:
        """Return the current balance for *asset*."""

    @abstractmethod
    def get_last_price(self, symbol: str) -> float:
        """Return the latest price for *symbol*."""

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order.  Returns True if cancelled, False otherwise."""


# ---------------------------------------------------------------------------
# Mock exchange (in-memory, for tests and paper trading)
# ---------------------------------------------------------------------------


class MockExchange(ExchangeABC):
    """In-memory exchange for paper trading and testing.

    Market orders fill immediately at the current price adjusted for
    slippage.  Balances are checked before execution; insufficient balance
    raises :class:`ValueError`.

    Parameters
    ----------
    initial_balances : dict[str, float]
        Starting balance per asset, e.g. ``{"USDT": 10_000.0, "BTC": 0.0}``.
    slippage_bps : float
        Basis points of slippage applied to market orders.  Buy orders fill
        at ``price * (1 + bps / 10_000)``; sell orders at
        ``price * (1 - bps / 10_000)``.
    fill_price_override : float or None
        When set, all market orders fill at exactly this price (ignores the
        price set via :meth:`set_price`).  Useful for deterministic tests.
    """

    def __init__(
        self,
        initial_balances: dict[str, float] | None = None,
        slippage_bps: float = 0.0,
        fill_price_override: float | None = None,
    ) -> None:
        self._initial_balances: dict[str, float] = dict(initial_balances or {})
        self._balances: dict[str, float] = dict(self._initial_balances)
        self._slippage_bps = slippage_bps
        self._fill_price_override = fill_price_override
        self._prices: dict[str, float] = {}
        self._orders: list[OrderResult] = []

    # ── Price management ─────────────────────────────────────────────────────

    def set_price(self, symbol: str, price: float) -> None:
        """Set the current mock price for *symbol*."""
        self._prices[symbol] = price

    # ── ExchangeABC implementation ───────────────────────────────────────────

    def place_order(self, request: OrderRequest) -> OrderResult:
        """Execute a market or limit order immediately at the mock price.

        Parameters
        ----------
        request : OrderRequest

        Returns
        -------
        OrderResult

        Raises
        ------
        ValueError
            If the account has insufficient balance to cover the order.
        ValueError
            If the symbol has no price set and no ``fill_price_override`` is
            configured.
        """
        if request.order_type == "market":
            raw_price = (
                self._fill_price_override
                if self._fill_price_override is not None
                else self._prices.get(request.symbol)
            )
            if raw_price is None:
                raise ValueError(
                    f"No price set for {request.symbol}. "
                    "Call set_price() before placing a market order, "
                    "or pass fill_price_override to MockExchange."
                )
            bps = self._slippage_bps
            if request.side == "buy":
                fill_price = raw_price * (1.0 + bps / 10_000.0)
            else:
                fill_price = raw_price * (1.0 - bps / 10_000.0)
        else:
            # Limit order: fill at limit_price (no slippage simulation)
            if request.limit_price is None:
                raise ValueError("limit_price must be set for limit orders.")
            fill_price = request.limit_price

        cost = fill_price * request.quantity

        # ── Balance check ─────────────────────────────────────────────────────
        base, quote = self._parse_symbol(request.symbol)
        if request.side == "buy":
            available = self._balances.get(quote, 0.0)
            if available < cost:
                raise ValueError(
                    f"Insufficient {quote} balance: need {cost:.6f}, "
                    f"have {available:.6f}."
                )
            self._balances[quote] = available - cost
            self._balances[base] = self._balances.get(base, 0.0) + request.quantity
        else:
            available = self._balances.get(base, 0.0)
            if available < request.quantity:
                raise ValueError(
                    f"Insufficient {base} balance: need {request.quantity:.6f}, "
                    f"have {available:.6f}."
                )
            self._balances[base] = available - request.quantity
            self._balances[quote] = self._balances.get(quote, 0.0) + cost

        result = OrderResult(
            order_id=str(uuid.uuid4()),
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            filled_quantity=request.quantity,
            average_price=fill_price,
            status="filled",
            timestamp=pd.Timestamp.now(tz="UTC"),
        )
        self._orders.append(result)
        logger.debug(
            "MockExchange fill: %s %s %.6f @ %.4f",
            result.side, result.symbol, result.filled_quantity, result.average_price,
        )
        return result

    def get_balance(self, asset: str) -> Balance:
        """Return balance for *asset*.  Returns zero if asset is unknown."""
        free = self._balances.get(asset, 0.0)
        return Balance(asset=asset, free=free, locked=0.0)

    def get_last_price(self, symbol: str) -> float:
        """Return the price set by :meth:`set_price`.

        Raises
        ------
        KeyError
            If no price has been set for *symbol*.
        """
        if symbol not in self._prices:
            raise KeyError(f"No price set for {symbol}. Call set_price() first.")
        return self._prices[symbol]

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Market orders fill immediately; cancellation always returns False."""
        return False

    # ── Additional helpers ───────────────────────────────────────────────────

    @property
    def order_history(self) -> list[OrderResult]:
        """All orders placed since initialisation or last :meth:`reset`."""
        return list(self._orders)

    def reset(self) -> None:
        """Clear order history and restore initial balances."""
        self._balances = dict(self._initial_balances)
        self._orders.clear()
        self._prices.clear()

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_symbol(symbol: str) -> tuple[str, str]:
        """Split ``"BTC/USDT"`` → ``("BTC", "USDT")``."""
        if "/" not in symbol:
            raise ValueError(
                f"Symbol must be in 'BASE/QUOTE' format, got '{symbol}'."
            )
        base, quote = symbol.split("/", 1)
        return base.strip(), quote.strip()


# ---------------------------------------------------------------------------
# Binance testnet (requires ccxt + env vars)
# ---------------------------------------------------------------------------


class BinanceTestnetExchange(ExchangeABC):
    """Exchange wrapper using ccxt pointed at Binance testnet.

    Requires:
    - ``ccxt`` installed: ``uv add ccxt``
    - Environment variables ``BINANCE_TESTNET_API_KEY`` and
      ``BINANCE_TESTNET_SECRET``

    Parameters
    ----------
    symbol_map : dict[str, str] or None
        Optional mapping from internal symbol notation (e.g. ``"BTC-USD"``)
        to ccxt notation (e.g. ``"BTC/USDT"``).  When absent, ``"-"`` is
        replaced with ``"/"`` and ``"USD"`` is replaced with ``"USDT"`` as a
        heuristic fallback.

    Raises
    ------
    ImportError
        If ccxt is not installed.
    ValueError
        If ``BINANCE_TESTNET_API_KEY`` or ``BINANCE_TESTNET_SECRET`` are not
        set in the environment.
    """

    _TESTNET_URLS = {
        "api": "https://testnet.binance.vision/api",
        "fapiPublic": "https://testnet.binancefuture.com/fapi/v1",
        "fapiPrivate": "https://testnet.binancefuture.com/fapi/v1",
    }

    def __init__(self, symbol_map: dict[str, str] | None = None) -> None:
        if _ccxt is None:
            raise ImportError(
                "ccxt is required for BinanceTestnetExchange. "
                "Install with: uv add ccxt"
            )

        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        secret = os.getenv("BINANCE_TESTNET_SECRET")

        if not api_key or not secret:
            raise ValueError(
                "Environment variables BINANCE_TESTNET_API_KEY and "
                "BINANCE_TESTNET_SECRET must be set to use BinanceTestnetExchange."
            )

        self._exchange = _ccxt.binance(
            {
                "apiKey": api_key,
                "secret": secret,
                "options": {"defaultType": "spot"},
                "urls": {"api": self._TESTNET_URLS},
            }
        )
        self._exchange.set_sandbox_mode(True)
        self._symbol_map: dict[str, str] = symbol_map or {}
        logger.info("BinanceTestnetExchange initialised (sandbox mode).")

    def _translate_symbol(self, symbol: str) -> str:
        """Convert internal symbol to ccxt format."""
        if symbol in self._symbol_map:
            return self._symbol_map[symbol]
        return symbol.replace("-", "/").replace("USD", "USDT")

    def place_order(self, request: OrderRequest) -> OrderResult:
        """Place a spot order on Binance testnet via ccxt."""
        ccxt_symbol = self._translate_symbol(request.symbol)
        if request.order_type == "market":
            raw = self._exchange.create_market_order(
                ccxt_symbol, request.side, request.quantity
            )
        else:
            if request.limit_price is None:
                raise ValueError("limit_price must be set for limit orders.")
            raw = self._exchange.create_limit_order(
                ccxt_symbol, request.side, request.quantity, request.limit_price
            )

        filled_qty = float(raw.get("filled", 0.0))
        avg_price = float(raw.get("average") or raw.get("price") or 0.0)
        fee_info = raw.get("fee") or {}
        fee = float(fee_info.get("cost", 0.0))

        return OrderResult(
            order_id=str(raw["id"]),
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            filled_quantity=filled_qty,
            average_price=avg_price,
            status=raw.get("status", "open"),
            timestamp=pd.Timestamp.now(tz="UTC"),
            fee=fee,
        )

    def get_balance(self, asset: str) -> Balance:
        """Fetch spot balance for *asset* from testnet."""
        raw = self._exchange.fetch_balance()
        info = raw.get(asset, {})
        return Balance(
            asset=asset,
            free=float(info.get("free", 0.0)),
            locked=float(info.get("used", 0.0)),
        )

    def get_last_price(self, symbol: str) -> float:
        """Fetch last traded price for *symbol*."""
        ccxt_symbol = self._translate_symbol(symbol)
        ticker = self._exchange.fetch_ticker(ccxt_symbol)
        return float(ticker["last"])

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order.  Returns True on success, False if already filled."""
        ccxt_symbol = self._translate_symbol(symbol)
        try:
            self._exchange.cancel_order(order_id, ccxt_symbol)
            return True
        except Exception as exc:
            logger.warning("cancel_order failed for %s: %s", order_id, exc)
            return False
