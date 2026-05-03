"""Tests for rde.trading.alerts (Phase 34)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from rde.trading.alerts import (
    AlertChannel,
    AlertConfig,
    RegimeChangeAlert,
    RegimeChangeMonitor,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ts(offset: int = 0) -> pd.Timestamp:
    """Return a deterministic timestamp offset by *offset* hours."""
    return pd.Timestamp("2024-01-01 00:00:00", tz="UTC") + pd.Timedelta(hours=offset)


def _posterior(n_states: int = 2, high_idx: int = 0, high_val: float = 0.9) -> np.ndarray:
    """Return a posterior array that places *high_val* on *high_idx*."""
    post = np.full(n_states, (1.0 - high_val) / (n_states - 1))
    post[high_idx] = high_val
    return post


def _monitor(
    *,
    channels: list[AlertChannel] | None = None,
    min_confidence: float = 0.6,
    cooldown_bars: int = 3,
    labels: dict[int, str] | None = None,
    callback=None,
    webhook_url: str | None = None,
) -> RegimeChangeMonitor:
    cfg = AlertConfig(
        channels=channels or [AlertChannel.LOG],
        min_confidence=min_confidence,
        cooldown_bars=cooldown_bars,
        webhook_url=webhook_url,
    )
    return RegimeChangeMonitor(
        symbol="BTC-USD",
        config=cfg,
        regime_labels=labels,
        callback=callback,
    )


# ---------------------------------------------------------------------------
# 1. AlertConfig defaults
# ---------------------------------------------------------------------------


def test_alert_config_defaults() -> None:
    cfg = AlertConfig()
    assert cfg.channels == [AlertChannel.LOG]
    assert cfg.webhook_url is None
    assert cfg.min_confidence == pytest.approx(0.6)
    assert cfg.cooldown_bars == 3


# ---------------------------------------------------------------------------
# 2. First bar returns None
# ---------------------------------------------------------------------------


def test_update_first_bar_returns_none() -> None:
    mon = _monitor()
    result = mon.update(_ts(0), regime=0, posterior=_posterior(high_idx=0))
    assert result is None


# ---------------------------------------------------------------------------
# 3. No regime change returns None
# ---------------------------------------------------------------------------


def test_update_no_regime_change_returns_none() -> None:
    mon = _monitor()
    mon.update(_ts(0), regime=0, posterior=_posterior(high_idx=0))
    result = mon.update(_ts(1), regime=0, posterior=_posterior(high_idx=0))
    assert result is None


# ---------------------------------------------------------------------------
# 4. Regime change above threshold fires alert
# ---------------------------------------------------------------------------


def test_update_regime_change_fires_alert() -> None:
    mon = _monitor(min_confidence=0.6)
    mon.update(_ts(0), regime=0, posterior=_posterior(high_idx=0))
    alert = mon.update(_ts(1), regime=1, posterior=_posterior(high_idx=1, high_val=0.85))
    assert alert is not None
    assert isinstance(alert, RegimeChangeAlert)
    assert alert.prev_regime == 0
    assert alert.new_regime == 1


# ---------------------------------------------------------------------------
# 5. Confidence below threshold suppresses alert
# ---------------------------------------------------------------------------


def test_update_below_confidence_threshold_returns_none() -> None:
    mon = _monitor(min_confidence=0.8)
    mon.update(_ts(0), regime=0, posterior=_posterior(high_idx=0))
    # New regime has only 0.65 posterior, below 0.80 threshold
    result = mon.update(_ts(1), regime=1, posterior=_posterior(high_idx=1, high_val=0.65))
    assert result is None


# ---------------------------------------------------------------------------
# 6. Cooldown suppresses immediate repeated alerts
# ---------------------------------------------------------------------------


def test_cooldown_suppresses_alert_within_window() -> None:
    mon = _monitor(cooldown_bars=3)
    mon.update(_ts(0), regime=0, posterior=_posterior(high_idx=0))

    # First change fires.
    alert1 = mon.update(_ts(1), regime=1, posterior=_posterior(high_idx=1, high_val=0.9))
    assert alert1 is not None

    # Second change immediately after — within cooldown.
    alert2 = mon.update(_ts(2), regime=0, posterior=_posterior(high_idx=0, high_val=0.9))
    assert alert2 is None

    # Third bar, still within cooldown window (bar 2, 3 counted after alert).
    alert3 = mon.update(_ts(3), regime=1, posterior=_posterior(high_idx=1, high_val=0.9))
    assert alert3 is None


# ---------------------------------------------------------------------------
# 7. Cooldown reset: alert fires again after window expires
# ---------------------------------------------------------------------------


def test_cooldown_reset_fires_after_window() -> None:
    mon = _monitor(cooldown_bars=2)
    mon.update(_ts(0), regime=0, posterior=_posterior(high_idx=0))

    # First change.
    alert1 = mon.update(_ts(1), regime=1, posterior=_posterior(high_idx=1, high_val=0.9))
    assert alert1 is not None

    # Bars 2, 3 within cooldown (bars_since = 1, 2).
    mon.update(_ts(2), regime=0, posterior=_posterior(high_idx=0, high_val=0.9))
    mon.update(_ts(3), regime=1, posterior=_posterior(high_idx=1, high_val=0.9))

    # Bar 4: bars_since = 3 > cooldown_bars=2, should fire.
    alert4 = mon.update(_ts(4), regime=0, posterior=_posterior(high_idx=0, high_val=0.9))
    assert alert4 is not None


# ---------------------------------------------------------------------------
# 8. History accumulates correctly
# ---------------------------------------------------------------------------


def test_history_accumulates() -> None:
    mon = _monitor(cooldown_bars=0)
    mon.update(_ts(0), regime=0, posterior=_posterior(high_idx=0))
    mon.update(_ts(1), regime=1, posterior=_posterior(high_idx=1, high_val=0.9))
    mon.update(_ts(2), regime=0, posterior=_posterior(high_idx=0, high_val=0.9))
    mon.update(_ts(3), regime=1, posterior=_posterior(high_idx=1, high_val=0.9))

    assert len(mon.history) == 3


# ---------------------------------------------------------------------------
# 9. reset() clears history and state
# ---------------------------------------------------------------------------


def test_reset_clears_state() -> None:
    mon = _monitor(cooldown_bars=0)
    mon.update(_ts(0), regime=0, posterior=_posterior(high_idx=0))
    mon.update(_ts(1), regime=1, posterior=_posterior(high_idx=1, high_val=0.9))
    assert len(mon.history) == 1

    mon.reset()
    assert len(mon.history) == 0

    # After reset, first bar produces None again (no prior regime).
    result = mon.update(_ts(2), regime=0, posterior=_posterior(high_idx=0))
    assert result is None


# ---------------------------------------------------------------------------
# 10. LOG channel invokes logging.warning
# ---------------------------------------------------------------------------


def test_log_channel_emits_warning(caplog: pytest.LogCaptureFixture) -> None:
    mon = _monitor(channels=[AlertChannel.LOG])
    mon.update(_ts(0), regime=0, posterior=_posterior(high_idx=0))

    with caplog.at_level(logging.WARNING, logger="rde.trading.alerts"):
        alert = mon.update(_ts(1), regime=1, posterior=_posterior(high_idx=1, high_val=0.9))

    assert alert is not None
    assert len(caplog.records) >= 1
    # Verify key info appears in the log message.
    messages = " ".join(r.message for r in caplog.records)
    assert "Regime change" in messages or "regime" in messages.lower()


# ---------------------------------------------------------------------------
# 11. WEBHOOK channel calls requests.post with correct payload
# ---------------------------------------------------------------------------


def test_webhook_channel_calls_requests_post() -> None:
    mon = _monitor(
        channels=[AlertChannel.WEBHOOK],
        webhook_url="http://example.com/hook",
    )
    mon.update(_ts(0), regime=0, posterior=_posterior(high_idx=0))

    with patch("rde.trading.alerts._requests") as mock_req:
        alert = mon.update(_ts(1), regime=1, posterior=_posterior(high_idx=1, high_val=0.9))

    assert alert is not None
    mock_req.post.assert_called_once()
    _, kwargs = mock_req.post.call_args
    payload = kwargs["json"]
    assert payload["symbol"] == "BTC-USD"
    assert payload["prev_regime"] == 0
    assert payload["new_regime"] == 1
    assert payload["alert_id"] == alert.alert_id
    assert kwargs["timeout"] == 5


# ---------------------------------------------------------------------------
# 12. WEBHOOK channel: request failure is caught and logged; monitor continues
# ---------------------------------------------------------------------------


def test_webhook_failure_is_caught_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    mon = _monitor(
        channels=[AlertChannel.WEBHOOK],
        webhook_url="http://example.com/hook",
    )
    mon.update(_ts(0), regime=0, posterior=_posterior(high_idx=0))

    with caplog.at_level(logging.WARNING, logger="rde.trading.alerts"):
        with patch("rde.trading.alerts._requests") as mock_req:
            mock_req.post.side_effect = Exception("connection timeout")
            alert = mon.update(
                _ts(1), regime=1, posterior=_posterior(high_idx=1, high_val=0.9)
            )

    # Alert object should still be returned (dispatch failed, but detection succeeded).
    assert alert is not None
    # A warning should have been logged about the failure.
    warning_messages = " ".join(r.message for r in caplog.records if r.levelno == logging.WARNING)
    assert "failed" in warning_messages.lower() or "webhook" in warning_messages.lower()


# ---------------------------------------------------------------------------
# 13. CALLBACK channel invokes user callback
# ---------------------------------------------------------------------------


def test_callback_channel_invokes_callback() -> None:
    received: list[RegimeChangeAlert] = []

    def my_callback(alert: RegimeChangeAlert) -> None:
        received.append(alert)

    mon = _monitor(channels=[AlertChannel.CALLBACK], callback=my_callback)
    mon.update(_ts(0), regime=0, posterior=_posterior(high_idx=0))
    alert = mon.update(_ts(1), regime=1, posterior=_posterior(high_idx=1, high_val=0.9))

    assert alert is not None
    assert len(received) == 1
    assert received[0] is alert


# ---------------------------------------------------------------------------
# 14. RegimeChangeAlert fields are populated correctly
# ---------------------------------------------------------------------------


def test_alert_fields_populated_correctly() -> None:
    labels = {0: "Bear", 1: "Bull"}
    mon = _monitor(
        channels=[AlertChannel.LOG],
        labels=labels,
        min_confidence=0.5,
        cooldown_bars=0,
    )
    ts0 = _ts(0)
    ts1 = _ts(1)
    post = _posterior(high_idx=1, high_val=0.75)

    mon.update(ts0, regime=0, posterior=_posterior(high_idx=0))
    alert = mon.update(ts1, regime=1, posterior=post)

    assert alert is not None
    assert alert.timestamp == ts1
    assert alert.symbol == "BTC-USD"
    assert alert.prev_regime == 0
    assert alert.new_regime == 1
    assert alert.prev_label == "Bear"
    assert alert.new_label == "Bull"
    assert alert.confidence == pytest.approx(0.75)
    # alert_id should be a non-empty UUID-like string
    assert len(alert.alert_id) == 36  # UUID4 canonical form


# ---------------------------------------------------------------------------
# 15. Fallback labels when regime_labels is None or missing key
# ---------------------------------------------------------------------------


def test_fallback_labels_when_no_mapping() -> None:
    mon = _monitor(channels=[AlertChannel.LOG], labels=None, cooldown_bars=0)
    mon.update(_ts(0), regime=0, posterior=_posterior(high_idx=0))
    alert = mon.update(_ts(1), regime=1, posterior=_posterior(high_idx=1, high_val=0.9))
    assert alert is not None
    assert alert.prev_label == "Regime 0"
    assert alert.new_label == "Regime 1"


# ---------------------------------------------------------------------------
# 16. history property returns a copy (mutation does not affect monitor)
# ---------------------------------------------------------------------------


def test_history_returns_copy() -> None:
    mon = _monitor(cooldown_bars=0)
    mon.update(_ts(0), regime=0, posterior=_posterior(high_idx=0))
    mon.update(_ts(1), regime=1, posterior=_posterior(high_idx=1, high_val=0.9))

    h = mon.history
    h.clear()  # mutate the copy

    assert len(mon.history) == 1  # internal list unaffected
