"""Tests for rde.trading.risk_guard (Phase 36)."""
from __future__ import annotations

import pytest
import pandas as pd

from rde.trading.risk_guard import RiskGuard, RiskGuardConfig, RiskGuardState


def _ts(offset_h: int = 0) -> pd.Timestamp:
    return pd.Timestamp("2024-01-15 10:00", tz="UTC") + pd.Timedelta(hours=offset_h)


# ---------------------------------------------------------------------------
# RiskGuardConfig
# ---------------------------------------------------------------------------

class TestRiskGuardConfig:
    def test_defaults(self):
        cfg = RiskGuardConfig()
        assert cfg.max_drawdown_pct == 10.0
        assert cfg.cooldown_bars == 24
        assert cfg.daily_loss_limit_pct is None

    def test_custom(self):
        cfg = RiskGuardConfig(max_drawdown_pct=5.0, cooldown_bars=10, daily_loss_limit_pct=2.0)
        assert cfg.max_drawdown_pct == 5.0
        assert cfg.cooldown_bars == 10
        assert cfg.daily_loss_limit_pct == 2.0


# ---------------------------------------------------------------------------
# RiskGuardState
# ---------------------------------------------------------------------------

class TestRiskGuardState:
    def test_defaults(self):
        s = RiskGuardState()
        assert s.peak_equity == 0.0
        assert not s.is_halted
        assert s.n_halts == 0
        assert s.halt_reason is None


# ---------------------------------------------------------------------------
# RiskGuard — construction
# ---------------------------------------------------------------------------

class TestRiskGuardInit:
    def test_initial_state_not_halted(self):
        guard = RiskGuard(RiskGuardConfig(), initial_equity=100_000.0)
        assert not guard.is_halted

    def test_initial_peak_equity(self):
        guard = RiskGuard(RiskGuardConfig(), initial_equity=50_000.0)
        assert guard.state.peak_equity == pytest.approx(50_000.0)

    def test_initial_drawdown_zero(self):
        guard = RiskGuard(RiskGuardConfig(), initial_equity=100_000.0)
        assert guard.current_drawdown_pct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RiskGuard.update — drawdown halt
# ---------------------------------------------------------------------------

class TestRiskGuardDrawdown:
    def test_no_halt_below_threshold(self):
        guard = RiskGuard(RiskGuardConfig(max_drawdown_pct=10.0), 100_000.0)
        result = guard.update(_ts(), 95_001.0)   # 4.999% drawdown
        assert result is True
        assert not guard.is_halted

    def test_halts_at_threshold(self):
        guard = RiskGuard(RiskGuardConfig(max_drawdown_pct=10.0), 100_000.0)
        result = guard.update(_ts(), 90_000.0)   # exactly 10%
        assert result is False
        assert guard.is_halted

    def test_halts_above_threshold(self):
        guard = RiskGuard(RiskGuardConfig(max_drawdown_pct=5.0), 100_000.0)
        guard.update(_ts(), 100_500.0)     # new peak
        guard.update(_ts(1), 93_000.0)    # 7.46% drawdown from peak
        assert guard.is_halted

    def test_peak_updates_on_new_high(self):
        guard = RiskGuard(RiskGuardConfig(max_drawdown_pct=10.0), 100_000.0)
        guard.update(_ts(), 110_000.0)
        assert guard.state.peak_equity == pytest.approx(110_000.0)

    def test_drawdown_computed_from_running_peak(self):
        guard = RiskGuard(RiskGuardConfig(max_drawdown_pct=20.0), 100_000.0)
        guard.update(_ts(), 120_000.0)    # peak = 120k
        guard.update(_ts(1), 108_000.0)  # dd = (120-108)/120 = 10%
        assert guard.current_drawdown_pct == pytest.approx(10.0, abs=0.01)

    def test_halt_reason_set(self):
        guard = RiskGuard(RiskGuardConfig(max_drawdown_pct=5.0), 100_000.0)
        guard.update(_ts(), 94_000.0)
        assert guard.state.halt_reason is not None
        assert "drawdown" in guard.state.halt_reason.lower()

    def test_n_halts_increments(self):
        guard = RiskGuard(RiskGuardConfig(max_drawdown_pct=5.0, cooldown_bars=0), 100_000.0)
        guard.update(_ts(), 94_000.0)   # halt 1
        guard.reset(100_000.0)
        guard.update(_ts(1), 94_000.0) # halt 2
        assert guard.state.n_halts == 2


# ---------------------------------------------------------------------------
# RiskGuard.update — cooldown
# ---------------------------------------------------------------------------

class TestRiskGuardCooldown:
    def test_stays_halted_during_cooldown(self):
        cfg = RiskGuardConfig(max_drawdown_pct=5.0, cooldown_bars=3)
        guard = RiskGuard(cfg, 100_000.0)
        guard.update(_ts(), 94_000.0)    # triggers halt, cooldown=3
        # Equity recovers but cooldown still active
        assert guard.update(_ts(1), 100_000.0) is False
        assert guard.update(_ts(2), 100_000.0) is False
        assert guard.update(_ts(3), 100_000.0) is False
        # Cooldown expired
        assert guard.update(_ts(4), 100_000.0) is True

    def test_cooldown_zero_resumes_immediately(self):
        cfg = RiskGuardConfig(max_drawdown_pct=5.0, cooldown_bars=0)
        guard = RiskGuard(cfg, 100_000.0)
        guard.update(_ts(), 94_000.0)
        # With cooldown=0, the decrement on the next bar should release
        result = guard.update(_ts(1), 100_000.0)
        assert result is True


# ---------------------------------------------------------------------------
# RiskGuard — daily loss limit
# ---------------------------------------------------------------------------

class TestRiskGuardDailyLoss:
    def test_no_halt_within_daily_limit(self):
        cfg = RiskGuardConfig(max_drawdown_pct=50.0, daily_loss_limit_pct=5.0)
        guard = RiskGuard(cfg, 100_000.0)
        result = guard.update(_ts(), 96_000.0)   # 4% daily loss
        assert result is True

    def test_halts_at_daily_limit(self):
        cfg = RiskGuardConfig(max_drawdown_pct=50.0, daily_loss_limit_pct=5.0)
        guard = RiskGuard(cfg, 100_000.0)
        guard.update(_ts(), 94_500.0)   # 5.5% daily loss
        assert guard.is_halted

    def test_daily_loss_resets_on_new_day(self):
        cfg = RiskGuardConfig(max_drawdown_pct=50.0, daily_loss_limit_pct=5.0, cooldown_bars=0)
        guard = RiskGuard(cfg, 100_000.0)
        # Day 1: breach limit
        guard.update(pd.Timestamp("2024-01-15 10:00", tz="UTC"), 94_000.0)
        assert guard.is_halted

        # Day 2: new day resets daily start equity; guard resumes after cooldown=0
        result = guard.update(pd.Timestamp("2024-01-16 10:00", tz="UTC"), 94_000.0)
        # Day-start equity is now 94_000 — small positive equity so not breached
        assert result is True


# ---------------------------------------------------------------------------
# RiskGuard.reset
# ---------------------------------------------------------------------------

class TestRiskGuardReset:
    def test_reset_clears_halt(self):
        cfg = RiskGuardConfig(max_drawdown_pct=5.0)
        guard = RiskGuard(cfg, 100_000.0)
        guard.update(_ts(), 94_000.0)
        assert guard.is_halted
        guard.reset(94_000.0)
        assert not guard.is_halted

    def test_reset_sets_new_peak(self):
        guard = RiskGuard(RiskGuardConfig(), 100_000.0)
        guard.reset(80_000.0)
        assert guard.state.peak_equity == pytest.approx(80_000.0)

    def test_reset_clears_cooldown(self):
        cfg = RiskGuardConfig(max_drawdown_pct=5.0, cooldown_bars=100)
        guard = RiskGuard(cfg, 100_000.0)
        guard.update(_ts(), 94_000.0)
        guard.reset(100_000.0)
        assert guard.state.cooldown_bars_remaining == 0
