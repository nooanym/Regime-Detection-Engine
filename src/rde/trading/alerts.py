"""Regime change alerting module (Phase 34).

Detects HMM regime transitions in a streaming context and dispatches alerts
via configurable channels: Python logging, HTTP webhooks, and user callbacks.

Typical usage::

    config = AlertConfig(
        channels=[AlertChannel.LOG, AlertChannel.CALLBACK],
        min_confidence=0.7,
        cooldown_bars=5,
    )
    monitor = RegimeChangeMonitor(
        symbol="BTC-USD",
        config=config,
        regime_labels={0: "Bear", 1: "Bull"},
        callback=my_handler,
    )

    for ts, regime, posterior in bar_stream:
        alert = monitor.update(ts, regime, posterior)
        if alert is not None:
            print(f"Regime changed to {alert.new_label} at {alert.timestamp}")
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
import pandas as pd

# Soft import — requests is not a mandatory project dependency.
try:
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public enums and data classes
# ---------------------------------------------------------------------------


class AlertChannel(Enum):
    """Supported dispatch targets for regime-change alerts."""

    LOG = "log"
    """Emit a ``logging.warning`` message."""

    WEBHOOK = "webhook"
    """HTTP POST a JSON payload to a URL."""

    CALLBACK = "callback"
    """Invoke a user-supplied Python callable."""


@dataclass
class AlertConfig:
    """Configuration for :class:`RegimeChangeMonitor`.

    Attributes
    ----------
    channels : list[AlertChannel]
        One or more output channels. Defaults to ``[AlertChannel.LOG]``.
    webhook_url : str | None
        Target URL for HTTP POST requests. Required when
        ``AlertChannel.WEBHOOK`` is in *channels*.
    min_confidence : float
        Minimum posterior probability of the new regime required to fire an
        alert. Transitions where the posterior of the new regime is below this
        threshold are silently ignored. Must be in ``[0, 1]``.
    cooldown_bars : int
        Number of :meth:`~RegimeChangeMonitor.update` calls after an alert
        during which subsequent change detections are suppressed. Set to ``0``
        to disable cooldown.
    """

    channels: list[AlertChannel] = field(
        default_factory=lambda: [AlertChannel.LOG]
    )
    webhook_url: str | None = None
    min_confidence: float = 0.6
    cooldown_bars: int = 3


@dataclass
class RegimeChangeAlert:
    """Represents a single detected regime transition.

    Attributes
    ----------
    timestamp : pd.Timestamp
        Wall-clock time of the bar at which the change was detected.
    symbol : str
        Asset identifier (e.g. ``"BTC-USD"``).
    prev_regime : int
        State index of the regime before the transition.
    new_regime : int
        State index of the regime after the transition.
    prev_label : str
        Human-readable label for *prev_regime*.
    new_label : str
        Human-readable label for *new_regime*.
    confidence : float
        Filtered posterior probability of *new_regime* at the transition bar.
    alert_id : str
        UUID4 string uniquely identifying this alert instance.
    """

    timestamp: pd.Timestamp
    symbol: str
    prev_regime: int
    new_regime: int
    prev_label: str
    new_label: str
    confidence: float
    alert_id: str


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


class RegimeChangeMonitor:
    """Detect regime changes and dispatch alerts in a streaming context.

    Each call to :meth:`update` is considered one "bar". The monitor keeps
    track of the previously seen regime and triggers an alert whenever the
    regime index changes, subject to confidence gating and a cooldown window.

    Parameters
    ----------
    symbol : str
        Asset identifier forwarded into every :class:`RegimeChangeAlert`.
    config : AlertConfig
        Channel configuration, confidence threshold, and cooldown settings.
    regime_labels : dict[int, str] | None
        Optional mapping from state index to human-readable label. If a state
        index is missing from the mapping (or the mapping is ``None``), the
        label falls back to ``"Regime {k}"``.
    callback : Callable[[RegimeChangeAlert], None] | None
        Invoked synchronously when ``AlertChannel.CALLBACK`` is active and an
        alert fires. Ignored if the channel is not configured.
    """

    def __init__(
        self,
        symbol: str,
        config: AlertConfig,
        regime_labels: dict[int, str] | None = None,
        callback: Callable[[RegimeChangeAlert], None] | None = None,
    ) -> None:
        self._symbol = symbol
        self._config = config
        self._regime_labels: dict[int, str] = regime_labels or {}
        self._callback = callback

        self._prev_regime: int | None = None
        self._bars_since_last_alert: int | None = None  # None = no alert yet
        self._history: list[RegimeChangeAlert] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[RegimeChangeAlert]:
        """All alerts dispatched since construction or last :meth:`reset`.

        Returns
        -------
        list[RegimeChangeAlert]
            Read-only view (copy) of the alert history list.
        """
        return list(self._history)

    def reset(self) -> None:
        """Clear all state: history, previous regime, and cooldown counter.

        After calling this method the monitor behaves as if freshly
        constructed — the next :meth:`update` call will not fire an alert
        (no previous regime to compare against).
        """
        self._prev_regime = None
        self._bars_since_last_alert = None
        self._history.clear()

    def update(
        self,
        timestamp: pd.Timestamp,
        regime: int,
        posterior: np.ndarray,
    ) -> RegimeChangeAlert | None:
        """Feed one decoded bar into the monitor.

        Parameters
        ----------
        timestamp : pd.Timestamp
            Timestamp of the current bar.
        regime : int
            Argmax-decoded regime index for this bar (e.g. from Viterbi or
            the online filtered posterior).
        posterior : np.ndarray
            Shape ``(K,)``. Full filtered posterior probability vector. The
            confidence for the transition is taken as ``posterior[regime]``.

        Returns
        -------
        RegimeChangeAlert | None
            A populated :class:`RegimeChangeAlert` if:

            * This is not the first bar (there is a previous regime),
            * ``regime != prev_regime``,
            * ``posterior[regime] >= config.min_confidence``,
            * The cooldown window has expired (at least *cooldown_bars* bars
              have elapsed since the last dispatched alert).

            Returns ``None`` in all other cases. When an alert is returned it
            has already been dispatched to all configured channels.

        Notes
        -----
        The cooldown counter is incremented on every call regardless of
        whether an alert fires, so the window shrinks naturally as bars arrive.
        """
        # Advance cooldown counter on every bar.
        if self._bars_since_last_alert is not None:
            self._bars_since_last_alert += 1

        prev = self._prev_regime
        self._prev_regime = regime

        # First bar — nothing to compare against.
        if prev is None:
            return None

        # No regime change.
        if regime == prev:
            return None

        confidence = float(posterior[regime])

        # Confidence gate.
        if confidence < self._config.min_confidence:
            logger.debug(
                "Regime change %d → %d suppressed: confidence %.3f < threshold %.3f",
                prev,
                regime,
                confidence,
                self._config.min_confidence,
            )
            return None

        # Cooldown gate.
        if (
            self._bars_since_last_alert is not None
            and self._bars_since_last_alert <= self._config.cooldown_bars
        ):
            logger.debug(
                "Regime change %d → %d suppressed: cooldown (%d/%d bars)",
                prev,
                regime,
                self._bars_since_last_alert,
                self._config.cooldown_bars,
            )
            return None

        alert = RegimeChangeAlert(
            timestamp=timestamp,
            symbol=self._symbol,
            prev_regime=prev,
            new_regime=regime,
            prev_label=self._regime_labels.get(prev, f"Regime {prev}"),
            new_label=self._regime_labels.get(regime, f"Regime {regime}"),
            confidence=confidence,
            alert_id=str(uuid.uuid4()),
        )

        self._history.append(alert)
        self._bars_since_last_alert = 0
        self._dispatch(alert)
        return alert

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch(self, alert: RegimeChangeAlert) -> None:
        """Dispatch *alert* to every configured channel.

        Parameters
        ----------
        alert : RegimeChangeAlert
            The alert to dispatch.
        """
        for channel in self._config.channels:
            if channel is AlertChannel.LOG:
                self._dispatch_log(alert)
            elif channel is AlertChannel.WEBHOOK:
                self._dispatch_webhook(alert)
            elif channel is AlertChannel.CALLBACK:
                self._dispatch_callback(alert)

    def _dispatch_log(self, alert: RegimeChangeAlert) -> None:
        """Emit a ``logging.warning`` for *alert*.

        Parameters
        ----------
        alert : RegimeChangeAlert
        """
        logger.warning(
            "Regime change detected | symbol=%s | %s → %s | confidence=%.4f | id=%s | ts=%s",
            alert.symbol,
            alert.prev_label,
            alert.new_label,
            alert.confidence,
            alert.alert_id,
            alert.timestamp.isoformat(),
        )

    def _dispatch_webhook(self, alert: RegimeChangeAlert) -> None:
        """HTTP POST *alert* as JSON to the configured webhook URL.

        Parameters
        ----------
        alert : RegimeChangeAlert

        Notes
        -----
        If ``requests`` is not installed, or the POST raises any exception,
        a warning is logged and execution continues — this method never raises.
        """
        if _requests is None:
            logger.warning(
                "requests not installed; webhook skipped. "
                "Install with: uv add requests"
            )
            return

        if not self._config.webhook_url:
            logger.warning(
                "AlertChannel.WEBHOOK is configured but webhook_url is None; skipping."
            )
            return

        payload = {
            "symbol": alert.symbol,
            "timestamp": alert.timestamp.isoformat(),
            "prev_regime": alert.prev_regime,
            "new_regime": alert.new_regime,
            "prev_label": alert.prev_label,
            "new_label": alert.new_label,
            "confidence": alert.confidence,
            "alert_id": alert.alert_id,
        }
        try:
            _requests.post(self._config.webhook_url, json=payload, timeout=5)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Webhook POST to %s failed: %s", self._config.webhook_url, exc
            )

    def _dispatch_callback(self, alert: RegimeChangeAlert) -> None:
        """Invoke the user-supplied callback with *alert*.

        Parameters
        ----------
        alert : RegimeChangeAlert

        Notes
        -----
        If no callback was registered, this is a no-op. If the callback raises,
        the exception propagates (caller is responsible for its own callback).
        """
        if self._callback is not None:
            self._callback(alert)
