"""Paper trading portfolio simulator (Phase 34).

Simulates fill execution against synthetic prices, tracking cash, position,
and P&L without connecting to any live exchange.

Slippage convention
-------------------
``slippage_bps`` is expressed in basis points (1 bps = 0.0001).
For a **buy** the fill price is raised:  ``fill_price = price * (1 + slippage_bps / 10_000)``
For a **sell** the fill price is lowered: ``fill_price = price * (1 - slippage_bps / 10_000)``
The dollar slippage cost is ``|quantity| * price * slippage_bps / 10_000``.
This matches the convention used in :mod:`rde.analysis.execution` where slippage
is expressed as a fraction of price.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Configuration for :class:`PaperPortfolio`.

    Attributes:
        initial_capital: Starting cash balance in currency units.
        slippage_bps: One-way slippage in basis points (1 bps = 0.0001).
            Applied to every fill: buys pay ``price * (1 + bps/10000)``,
            sells receive ``price * (1 - bps/10000)``.
        commission_flat: Flat dollar commission per fill (both sides).
    """

    initial_capital: float = 100_000.0
    slippage_bps: float = 5.0
    commission_flat: float = 0.0


@dataclass
class Fill:
    """Record of a single executed order.

    Attributes:
        timestamp: Bar timestamp at which the fill occurred.
        symbol: Ticker symbol.
        side: ``"buy"`` or ``"sell"``.
        quantity: Absolute units traded (always positive).
        price: Mid-price at fill time (before slippage).
        slippage: Dollar slippage cost (always non-negative).
        commission: Dollar commission cost (always non-negative).
    """

    timestamp: pd.Timestamp
    symbol: str
    side: str
    quantity: float
    price: float
    slippage: float
    commission: float

    @property
    def fill_price(self) -> float:
        """Effective execution price (mid ± slippage per unit)."""
        slip_per_unit = self.slippage / self.quantity if self.quantity else 0.0
        if self.side == "buy":
            return self.price + slip_per_unit
        return self.price - slip_per_unit

    @property
    def gross_cost(self) -> float:
        """Signed cash impact of the fill (negative = cash outflow for buys).

        Does not include commission.
        """
        if self.side == "buy":
            return -(self.quantity * self.fill_price)
        return self.quantity * self.fill_price


class PaperPortfolio:
    """Simulated portfolio tracking cash, position, and P&L.

    All fills are executed immediately at the supplied price adjusted for
    slippage.  No exchange connection is required.

    Args:
        config: Portfolio configuration.
        symbol: Ticker symbol for this portfolio (used in fills).
    """

    def __init__(self, config: PortfolioConfig | None = None, symbol: str = "UNKNOWN") -> None:
        self._config: PortfolioConfig = config or PortfolioConfig()
        self._symbol: str = symbol
        self._cash: float = self._config.initial_capital
        self._position: float = 0.0
        self._realized_pnl: float = 0.0
        self._fills: list[Fill] = []
        # Average cost basis of current position (used for realized P&L).
        self._avg_cost: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position(self) -> float:
        """Current units held (positive = long, negative = short)."""
        return self._position

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._cash

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def execute(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        target_quantity: float,
        price: float,
    ) -> Fill | None:
        """Move from the current position to *target_quantity* at *price*.

        If the current position already equals *target_quantity* (within
        floating-point tolerance), no trade is generated and ``None`` is
        returned.

        Slippage is applied as follows:

        - **Buy** (target > current): fill at ``price * (1 + bps/10_000)``.
        - **Sell** (target < current): fill at ``price * (1 - bps/10_000)``.

        Realized P&L is updated whenever a position is wholly or partially
        closed.

        Args:
            timestamp: Bar timestamp.
            symbol: Asset symbol.
            target_quantity: Desired holding in units after the trade.
            price: Reference mid-price for the bar.

        Returns:
            A :class:`Fill` if a trade was executed, otherwise ``None``.
        """
        delta = target_quantity - self._position
        if abs(delta) < 1e-10:
            return None

        side = "buy" if delta > 0 else "sell"
        quantity = abs(delta)

        slip_fraction = self._config.slippage_bps / 10_000.0
        slippage_dollar = quantity * price * slip_fraction
        commission = self._config.commission_flat

        fill = Fill(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            slippage=slippage_dollar,
            commission=commission,
        )

        # Update cash
        self._cash += fill.gross_cost - commission

        # Update realized P&L and average cost
        if side == "buy":
            # Acquiring more units
            total_units = self._position + quantity
            if total_units != 0.0:
                self._avg_cost = (
                    (self._position * self._avg_cost + quantity * fill.fill_price)
                    / total_units
                )
            else:
                self._avg_cost = fill.fill_price
        else:
            # Selling / reducing position
            # Realized P&L on the closing portion
            closing_qty = min(quantity, abs(self._position)) if self._position > 0 else quantity
            if self._position > 0:
                # Long position being reduced
                realized = closing_qty * (fill.fill_price - self._avg_cost)
                self._realized_pnl += realized
            elif self._position < 0:
                # Short position being covered
                realized = closing_qty * (self._avg_cost - fill.fill_price)
                self._realized_pnl += realized
            # If position crosses zero, the excess quantity establishes a new position
            if abs(delta) > abs(self._position) and self._position != 0.0:
                excess = quantity - abs(self._position)
                self._avg_cost = fill.fill_price  # new position at current fill price
                _ = excess  # already handled by avg_cost reset

        self._position = target_quantity
        self._fills.append(fill)

        logger.debug(
            "Fill: %s %s %.6f @ %.4f (slip=%.4f, comm=%.4f) | cash=%.2f pos=%.6f",
            side,
            symbol,
            quantity,
            price,
            slippage_dollar,
            commission,
            self._cash,
            self._position,
        )
        return fill

    def mark_to_market(self, price: float) -> float:
        """Return total portfolio equity at *price*.

        Args:
            price: Current mid-price for the held asset.

        Returns:
            Cash + position * price.
        """
        return self._cash + self._position * price

    def unrealized_pnl(self, price: float) -> float:
        """Unrealized P&L at *price*.

        Args:
            price: Current mid-price.

        Returns:
            ``position * (price - avg_cost)`` for longs;
            ``position * (avg_cost - price)`` for shorts.
        """
        if self._position > 0:
            return self._position * (price - self._avg_cost)
        elif self._position < 0:
            return abs(self._position) * (self._avg_cost - price)
        return 0.0

    def realized_pnl(self) -> float:
        """Cumulative realized P&L across all closed trades.

        Returns:
            Sum of realized P&L from all partial and full position closes.
        """
        return self._realized_pnl

    def equity_curve(self, prices: pd.Series) -> pd.Series:
        """Replay fills against a price series and return equity over time.

        Only fill timestamps are included in the output; the equity is
        computed as cash (after all fills up to that point) plus
        ``position * price`` at each fill timestamp.

        Args:
            prices: Price series whose index must cover all fill timestamps.
                The index should be a :class:`pandas.DatetimeIndex` or
                otherwise contain the fill timestamps.

        Returns:
            :class:`pandas.Series` indexed by fill timestamps with equity values.
        """
        if not self._fills:
            return pd.Series(dtype=float)

        # Replay fills in chronological order
        replay_cash = self._config.initial_capital
        replay_pos = 0.0
        replay_avg_cost = 0.0
        equity_values: dict[pd.Timestamp, float] = {}

        for fill in self._fills:
            delta = fill.quantity if fill.side == "buy" else -fill.quantity
            replay_cash += fill.gross_cost - fill.commission

            if fill.side == "buy":
                new_pos = replay_pos + fill.quantity
                if new_pos != 0.0:
                    replay_avg_cost = (
                        replay_pos * replay_avg_cost + fill.quantity * fill.fill_price
                    ) / new_pos
                else:
                    replay_avg_cost = fill.fill_price
            else:
                pass  # avg_cost unchanged on sell for long positions

            replay_pos += delta

            # Look up price at fill timestamp
            if fill.timestamp in prices.index:
                px = float(prices.loc[fill.timestamp])
            else:
                px = fill.price  # fallback to fill price
            equity_values[fill.timestamp] = replay_cash + replay_pos * px

        return pd.Series(equity_values)
