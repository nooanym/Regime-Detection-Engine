"""Risk guard for the paper-trading loop (Phase 36).

Monitors running drawdown and halts trading when a configurable threshold
is breached. Integrates with :class:`~rde.trading.loop.TradingLoop` via
its :meth:`step` pipeline.

Public API
----------
RiskGuardConfig
RiskGuardState
RiskGuard
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskGuardConfig:
    """Configuration for :class:`RiskGuard`.

    Attributes
    ----------
    max_drawdown_pct : float
        Maximum allowable peak-to-trough drawdown expressed as a positive
        percentage (e.g. ``10.0`` = halt when equity drops 10% from peak).
    cooldown_bars : int
        After a halt, the guard remains engaged for this many bars even if
        equity recovers above the threshold.  Prevents rapid re-entry.
    daily_loss_limit_pct : float or None
        Optional daily loss limit as a percentage of start-of-day equity.
        When breached the guard halts until the next calendar day.
    """

    max_drawdown_pct: float = 10.0
    cooldown_bars: int = 24
    daily_loss_limit_pct: float | None = None


@dataclass
class RiskGuardState:
    """Mutable state tracked by :class:`RiskGuard`.

    Attributes
    ----------
    peak_equity : float
        Highest equity seen since initialisation or last reset.
    current_drawdown_pct : float
        Current drawdown from peak as a positive percentage.
    is_halted : bool
        True when trading is suspended.
    halt_reason : str or None
        Human-readable reason for the most recent halt, or None.
    cooldown_bars_remaining : int
        Bars remaining in the post-halt cooldown period.
    n_halts : int
        Total number of halts triggered since initialisation.
    day_start_equity : float
        Equity at the start of the current calendar day (UTC).
    last_day : int or None
        ``pandas.Timestamp.day_of_year`` of the last processed bar.
    """

    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    is_halted: bool = False
    halt_reason: str | None = None
    cooldown_bars_remaining: int = 0
    n_halts: int = 0
    day_start_equity: float = 0.0
    last_day: int | None = None


class RiskGuard:
    """Monitors running drawdown and suspends trading when limits are breached.

    Call :meth:`update` on each bar.  It returns ``True`` when trading is
    permitted and ``False`` when the loop should skip order generation.

    Parameters
    ----------
    config : RiskGuardConfig
    initial_equity : float
        Starting equity — used to initialise the peak and day-start values.

    Examples
    --------
    >>> guard = RiskGuard(RiskGuardConfig(max_drawdown_pct=5.0), initial_equity=100_000.0)
    >>> guard.update(pd.Timestamp.now(tz="UTC"), 95_001.0)
    True
    >>> guard.update(pd.Timestamp.now(tz="UTC"), 94_900.0)
    False   # drawdown > 5% → halted
    """

    def __init__(self, config: RiskGuardConfig, initial_equity: float) -> None:
        self.config = config
        self.state = RiskGuardState(
            peak_equity=initial_equity,
            day_start_equity=initial_equity,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, timestamp: pd.Timestamp, equity: float) -> bool:
        """Update risk state for the current bar and return trading permission.

        Parameters
        ----------
        timestamp : pd.Timestamp
            Current bar timestamp (tz-aware UTC recommended).
        equity : float
            Current mark-to-market equity.

        Returns
        -------
        bool
            ``True`` → trading is permitted.
            ``False`` → trading is suspended (drawdown or daily-loss limit).
        """
        self._refresh_day(timestamp, equity)
        self._update_peak(equity)
        self._compute_drawdown(equity)

        if self.state.is_halted:
            if self.state.cooldown_bars_remaining > 0:
                self.state.cooldown_bars_remaining -= 1
                return False
            else:
                # cooldown expired (0 bars remaining) — clear halt this bar
                logger.info("RiskGuard cooldown expired — trading resumed.")
                self.state.is_halted = False
                self.state.halt_reason = None

        if not self.state.is_halted:
            if self._drawdown_breached():
                self._trigger_halt(
                    f"Max drawdown {self.state.current_drawdown_pct:.2f}% "
                    f"≥ {self.config.max_drawdown_pct}%"
                )
            elif self._daily_loss_breached(equity):
                self._trigger_halt(
                    f"Daily loss limit breached — equity {equity:.2f} vs "
                    f"day-start {self.state.day_start_equity:.2f}"
                )

        return not self.state.is_halted

    def reset(self, equity: float) -> None:
        """Reset the guard — clears halt, cooldown, and sets new peak.

        Parameters
        ----------
        equity : float
            Current equity, used as new peak and day-start baseline.
        """
        prev_n_halts = self.state.n_halts
        self.state = RiskGuardState(
            peak_equity=equity,
            day_start_equity=equity,
            n_halts=prev_n_halts,
        )
        logger.info("RiskGuard reset — new peak %.2f", equity)

    @property
    def is_halted(self) -> bool:
        """True when trading is currently suspended."""
        return self.state.is_halted

    @property
    def current_drawdown_pct(self) -> float:
        """Current drawdown from peak as a positive percentage."""
        return self.state.current_drawdown_pct

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _update_peak(self, equity: float) -> None:
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity

    def _compute_drawdown(self, equity: float) -> None:
        if self.state.peak_equity > 0:
            dd = (self.state.peak_equity - equity) / self.state.peak_equity * 100.0
            self.state.current_drawdown_pct = max(dd, 0.0)

    def _drawdown_breached(self) -> bool:
        return self.state.current_drawdown_pct >= self.config.max_drawdown_pct

    def _daily_loss_breached(self, equity: float) -> bool:
        limit = self.config.daily_loss_limit_pct
        if limit is None or self.state.day_start_equity <= 0:
            return False
        daily_loss_pct = (
            (self.state.day_start_equity - equity) / self.state.day_start_equity * 100.0
        )
        return daily_loss_pct >= limit

    def _trigger_halt(self, reason: str) -> None:
        self.state.is_halted = True
        self.state.halt_reason = reason
        self.state.cooldown_bars_remaining = self.config.cooldown_bars
        self.state.n_halts += 1
        logger.warning("RiskGuard HALT: %s", reason)

    def _refresh_day(self, timestamp: pd.Timestamp, equity: float) -> None:
        day = timestamp.day_of_year
        if self.state.last_day is None:
            # First bar — day_start_equity is already set to initial_equity in __init__
            self.state.last_day = day
        elif day != self.state.last_day:
            self.state.last_day = day
            self.state.day_start_equity = equity
            logger.debug("RiskGuard: new trading day — day_start_equity=%.2f", equity)
