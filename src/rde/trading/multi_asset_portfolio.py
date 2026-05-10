"""Multi-asset paper portfolio for RTMV rebalancer (Phase 48).

Tracks N positions, executes weight-based rebalances with slippage, and
records per-fill history.  Designed for daily-rebalanced ETF portfolios
where each rebalance moves from current weights to a new target vector.

Public API
----------
MultiAssetPortfolioConfig
MultiAssetFill
MultiAssetPortfolio
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MultiAssetPortfolioConfig:
    """Configuration for :class:`MultiAssetPortfolio`.

    Attributes
    ----------
    initial_capital : float
        Starting cash balance.
    slippage_bps : float
        One-way slippage in basis points applied to every fill.
        Buys pay ``price * (1 + bps/10_000)``; sells receive the
        opposite.
    min_trade_usd : float
        Minimum notional trade size.  Rebalance deltas smaller than
        this are skipped to avoid micro-fills.
    """

    initial_capital: float = 100_000.0
    slippage_bps: float = 5.0
    min_trade_usd: float = 10.0


@dataclass
class MultiAssetFill:
    """Record of a single executed fill.

    Attributes
    ----------
    timestamp : pd.Timestamp
    asset : str
    side : str
        ``"buy"`` or ``"sell"``.
    quantity : float
        Units traded (always positive).
    price : float
        Mid-price at execution time.
    slippage : float
        Dollar slippage cost (always non-negative).
    """

    timestamp: pd.Timestamp
    asset: str
    side: str
    quantity: float
    price: float
    slippage: float

    @property
    def fill_price(self) -> float:
        """Effective execution price including slippage."""
        slip_per_unit = self.slippage / self.quantity if self.quantity else 0.0
        return self.price + slip_per_unit if self.side == "buy" else self.price - slip_per_unit

    @property
    def notional(self) -> float:
        """Absolute notional value of the fill at mid-price."""
        return self.quantity * self.price


class MultiAssetPortfolio:
    """Simulated multi-asset portfolio with N positions.

    Parameters
    ----------
    config : MultiAssetPortfolioConfig
    assets : list[str]
        Ordered list of asset symbols held in this portfolio.
    """

    def __init__(
        self,
        config: MultiAssetPortfolioConfig,
        assets: list[str],
    ) -> None:
        self._config = config
        self._assets = list(assets)
        self._positions: dict[str, float] = {a: 0.0 for a in assets}
        self._cash: float = config.initial_capital
        self._fills: list[MultiAssetFill] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._cash

    @property
    def positions(self) -> dict[str, float]:
        """Current holdings in units per asset (copy)."""
        return dict(self._positions)

    @property
    def fills(self) -> list[MultiAssetFill]:
        """All fills recorded since construction (copy)."""
        return list(self._fills)

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------

    def equity(self, prices: dict[str, float]) -> float:
        """Total portfolio value: cash + sum(position_i * price_i).

        Parameters
        ----------
        prices : dict[str, float]
            Current price per asset.

        Returns
        -------
        float
        """
        asset_value = sum(self._positions[a] * prices.get(a, 0.0) for a in self._assets)
        return self._cash + asset_value

    def current_weights(self, prices: dict[str, float]) -> dict[str, float]:
        """Actual weight per asset as a fraction of total equity.

        Parameters
        ----------
        prices : dict[str, float]

        Returns
        -------
        dict[str, float]
            Zero-weight for all assets if equity is non-positive.
        """
        eq = self.equity(prices)
        if eq <= 1e-10:
            return {a: 0.0 for a in self._assets}
        return {a: self._positions[a] * prices.get(a, 0.0) / eq for a in self._assets}

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def set_target_weights(
        self,
        target_weights: dict[str, float],
        prices: dict[str, float],
        timestamp: pd.Timestamp,
    ) -> list[MultiAssetFill]:
        """Rebalance to target weights at the given prices.

        Sells are executed before buys so that proceeds fund purchases.
        Fills smaller than ``config.min_trade_usd`` in notional are skipped.

        Parameters
        ----------
        target_weights : dict[str, float]
            Desired weight per asset.  Values should sum to approximately 1.0.
        prices : dict[str, float]
            Current price per asset.
        timestamp : pd.Timestamp

        Returns
        -------
        list[MultiAssetFill]
            All fills executed during this rebalance.
        """
        eq = self.equity(prices)
        if eq <= 1e-10:
            logger.warning("Portfolio equity ≤ 0 at %s — skipping rebalance.", timestamp)
            return []

        slip_frac = self._config.slippage_bps / 10_000.0

        # Compute signed delta values (positive = buy, negative = sell).
        deltas: list[tuple[str, float]] = []
        for asset in self._assets:
            price = prices.get(asset, 0.0)
            if price <= 0.0:
                continue
            target_value = target_weights.get(asset, 0.0) * eq
            current_value = self._positions[asset] * price
            delta_value = target_value - current_value
            if abs(delta_value) >= self._config.min_trade_usd:
                deltas.append((asset, delta_value))

        # Sells first to free cash, then buys.
        deltas.sort(key=lambda x: x[1])

        fills: list[MultiAssetFill] = []
        for asset, delta_value in deltas:
            price = prices[asset]
            if delta_value < 0:  # sell
                exec_price = price * (1.0 - slip_frac)
                qty = -delta_value / price
                slip = qty * price * slip_frac
                self._cash += qty * exec_price
                self._positions[asset] = max(0.0, self._positions[asset] - qty)
                fill = MultiAssetFill(timestamp, asset, "sell", qty, price, slip)
            else:  # buy
                exec_price = price * (1.0 + slip_frac)
                qty = delta_value / exec_price
                slip = qty * price * slip_frac
                self._cash -= qty * exec_price
                self._positions[asset] += qty
                fill = MultiAssetFill(timestamp, asset, "buy", qty, price, slip)

            fills.append(fill)
            logger.debug(
                "%s  %s %s  qty=%.4f  price=%.4f  slip=%.4f  cash=%.2f",
                timestamp,
                fill.side.upper(),
                asset,
                qty,
                price,
                slip,
                self._cash,
            )

        self._fills.extend(fills)
        return fills

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(
        self,
        timestamp: pd.Timestamp,
        prices: dict[str, float],
        target_weights: dict[str, float] | None = None,
    ) -> dict:
        """Return a flat dict of the current portfolio state.

        Parameters
        ----------
        timestamp, prices, target_weights

        Returns
        -------
        dict
            Keys: ``timestamp``, ``equity``, ``cash``, plus per-asset
            ``pos_{a}``, ``price_{a}``, ``weight_{a}`` (and optionally
            ``target_{a}`` if ``target_weights`` is given).
        """
        eq = self.equity(prices)
        actual_w = self.current_weights(prices)
        row: dict = {"timestamp": timestamp, "equity": eq, "cash": self._cash}
        for a in self._assets:
            row[f"pos_{a}"] = self._positions[a]
            row[f"price_{a}"] = prices.get(a, float("nan"))
            row[f"weight_{a}"] = actual_w.get(a, 0.0)
            if target_weights is not None:
                row[f"target_{a}"] = target_weights.get(a, 0.0)
        return row

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_fills(self, path: Path) -> None:
        """Write fills to parquet.  No-op if there are no fills."""
        if not self._fills:
            return
        rows = [
            {
                "timestamp": f.timestamp,
                "asset": f.asset,
                "side": f.side,
                "quantity": f.quantity,
                "price": f.price,
                "slippage": f.slippage,
                "notional": f.notional,
            }
            for f in self._fills
        ]
        pd.DataFrame(rows).to_parquet(path, index=False)
        logger.debug("Fills saved to %s (%d records).", path, len(rows))
