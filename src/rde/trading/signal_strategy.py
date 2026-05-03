"""Regime-driven signal strategy (Phase 34).

Maps HMM state indices to target portfolio weights and computes the
corresponding target quantity to hold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RegimeRule:
    """Target weight specification for a single HMM state.

    Attributes:
        regime: HMM state index (0-based).
        target_weight: Desired portfolio weight.
            ``-1.0`` = full short, ``0.0`` = flat, ``1.0`` = full long.
            Intermediate values represent partial exposure.
    """

    regime: int
    target_weight: float


@dataclass
class SignalStrategyConfig:
    """Configuration for :class:`RegimeSignalStrategy`.

    Attributes:
        rules: Mapping from regime indices to target weights.
        default_weight: Weight applied to regimes not covered by *rules*.
        min_weight_change: Minimum absolute weight change required to trigger
            a rebalance.  Changes smaller than this are ignored to avoid
            excessive turnover.
    """

    rules: list[RegimeRule] = field(default_factory=list)
    default_weight: float = 0.0
    min_weight_change: float = 0.01


class RegimeSignalStrategy:
    """Map HMM regime → target portfolio weight → target quantity.

    Args:
        config: Strategy configuration containing per-regime rules.
    """

    def __init__(self, config: SignalStrategyConfig) -> None:
        self._config = config
        self._weight_map: dict[int, float] = {r.regime: r.target_weight for r in config.rules}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def target_weight(self, regime: int) -> float:
        """Return the target portfolio weight for *regime*.

        If *regime* is not covered by any rule, the configured
        ``default_weight`` is returned.

        Args:
            regime: HMM state index.

        Returns:
            Target weight in ``[-1.0, 1.0]``.
        """
        weight = self._weight_map.get(regime, self._config.default_weight)
        logger.debug("Regime %d → target_weight=%.4f", regime, weight)
        return weight

    def target_quantity(self, regime: int, equity: float, price: float) -> float:
        """Compute the target units to hold given current equity and price.

        Formula::

            target_units = equity * target_weight / price

        Args:
            regime: HMM state index.
            equity: Current total portfolio equity in currency units.
            price: Current asset price.

        Returns:
            Target holding in units (positive = long, negative = short,
            zero = flat).

        Raises:
            ValueError: If *price* is not positive.
        """
        if price <= 0.0:
            raise ValueError(f"price must be positive, got {price}")
        weight = self.target_weight(regime)
        qty = equity * weight / price
        logger.debug(
            "Regime %d equity=%.2f price=%.4f → target_quantity=%.6f",
            regime,
            equity,
            price,
            qty,
        )
        return qty

    def signal_changed(self, prev_regime: int, new_regime: int) -> bool:
        """Return True if switching from *prev_regime* to *new_regime* triggers a rebalance.

        A rebalance is triggered when the absolute change in target weight
        exceeds ``min_weight_change``.

        Args:
            prev_regime: Previous HMM state index.
            new_regime: New HMM state index.

        Returns:
            ``True`` if ``|new_weight - prev_weight| > min_weight_change``.
        """
        prev_w = self.target_weight(prev_regime)
        new_w = self.target_weight(new_regime)
        changed = abs(new_w - prev_w) > self._config.min_weight_change
        logger.debug(
            "signal_changed: regime %d→%d weight %.4f→%.4f changed=%s",
            prev_regime,
            new_regime,
            prev_w,
            new_w,
            changed,
        )
        return changed
