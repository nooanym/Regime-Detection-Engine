"""Tests for rde.trading.loop (Phase 35)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from rde.trading.loop import TradingLoop, TradingLoopConfig, TradingLoopState


# ---------------------------------------------------------------------------
# Helpers — build minimal synthetic objects
# ---------------------------------------------------------------------------

def _make_fitted(n_states: int = 3, n_features: int = 3, seed: int = 0) -> MagicMock:
    """Return a FittedModel-like mock that OnlineDecoder can accept."""
    from sklearn.preprocessing import StandardScaler
    from hmmlearn.hmm import GaussianHMM

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((200, n_features))

    scaler = StandardScaler().fit(X)
    hmm = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=10, random_state=seed)
    hmm.fit(scaler.transform(X))

    fitted = MagicMock()
    fitted.hmm = hmm
    fitted.scaler = scaler
    fitted.n_states = n_states
    fitted.feature_names = ["log_return", "volatility_w24", "smoothed_return_w12"]
    return fitted


def _make_loop_config(tmp_path: Path, warmup_bars: int = 5) -> TradingLoopConfig:
    return TradingLoopConfig(
        symbol="BTC-USD",
        interval="1h",
        warmup_bars=warmup_bars,
        save_interval_bars=1000,   # don't save mid-test
        output_dir=tmp_path / "live",
        slippage_bps=0.0,
        initial_capital=100_000.0,
    )


def _make_bars_df(n: int, K: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "log_return": rng.normal(0, 0.01, size=n),
            "volatility_w24": rng.uniform(0.005, 0.02, size=n),
            "smoothed_return_w12": rng.normal(0, 0.005, size=n),
            "Close": rng.uniform(40_000, 70_000, size=n),
            "regime": rng.integers(0, K, size=n),
            "regime_label": ["Bull"] * n,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# TradingLoopConfig
# ---------------------------------------------------------------------------

class TestTradingLoopConfig:
    def test_defaults(self, tmp_path):
        cfg = TradingLoopConfig(symbol="BTC-USD", output_dir=tmp_path)
        assert cfg.interval == "1h"
        assert cfg.poll_interval_s == 60.0
        assert cfg.warmup_bars == 200
        assert cfg.initial_capital == 100_000.0
        assert cfg.slippage_bps == 5.0

    def test_custom_values(self, tmp_path):
        cfg = TradingLoopConfig(
            symbol="ETH-USD",
            output_dir=tmp_path,
            warmup_bars=50,
            initial_capital=5_000.0,
        )
        assert cfg.symbol == "ETH-USD"
        assert cfg.warmup_bars == 50
        assert cfg.initial_capital == 5_000.0


# ---------------------------------------------------------------------------
# TradingLoopState
# ---------------------------------------------------------------------------

class TestTradingLoopState:
    def test_initial_state(self):
        s = TradingLoopState()
        assert s.last_bar_ts is None
        assert s.current_regime is None
        assert s.n_bars_processed == 0
        assert s.n_trades == 0
        assert s.is_warmed_up is False
        assert s.snapshots == []


# ---------------------------------------------------------------------------
# TradingLoop construction
# ---------------------------------------------------------------------------

class TestTradingLoopInit:
    def test_creates_without_error(self, tmp_path):
        fitted = _make_fitted()
        cfg = _make_loop_config(tmp_path)
        loop = TradingLoop(fitted=fitted, config=cfg)
        assert loop.state.n_bars_processed == 0

    def test_output_dir_created(self, tmp_path):
        fitted = _make_fitted()
        cfg = _make_loop_config(tmp_path)
        TradingLoop(fitted=fitted, config=cfg)
        assert (tmp_path / "live").is_dir()

    def test_initial_equity(self, tmp_path):
        fitted = _make_fitted()
        cfg = _make_loop_config(tmp_path)
        loop = TradingLoop(fitted=fitted, config=cfg)
        assert loop.state.equity == pytest.approx(100_000.0)

    def test_with_strategy_rules(self, tmp_path):
        fitted = _make_fitted(n_states=3)
        cfg = _make_loop_config(tmp_path)
        loop = TradingLoop(
            fitted=fitted,
            config=cfg,
            strategy_rules=[(0, 1.0), (1, 0.0), (2, -1.0)],
        )
        assert loop.strategy.target_weight(0) == pytest.approx(1.0)
        assert loop.strategy.target_weight(1) == pytest.approx(0.0)
        assert loop.strategy.target_weight(2) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# TradingLoop.step
# ---------------------------------------------------------------------------

class TestTradingLoopStep:
    def test_step_increments_bar_count(self, tmp_path):
        fitted = _make_fitted()
        cfg = _make_loop_config(tmp_path, warmup_bars=2)
        loop = TradingLoop(fitted=fitted, config=cfg)
        ts = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        x = np.zeros(3)
        loop.step(ts, x, 50_000.0)
        assert loop.state.n_bars_processed == 1

    def test_not_warmed_up_during_warmup(self, tmp_path):
        fitted = _make_fitted()
        cfg = _make_loop_config(tmp_path, warmup_bars=10)
        loop = TradingLoop(fitted=fitted, config=cfg)
        ts = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        loop.step(ts, np.zeros(3), 50_000.0)
        assert not loop.state.is_warmed_up

    def test_warmed_up_after_warmup_bars(self, tmp_path):
        fitted = _make_fitted()
        cfg = _make_loop_config(tmp_path, warmup_bars=3)
        loop = TradingLoop(fitted=fitted, config=cfg)
        for i in range(4):
            ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=i)
            loop.step(ts, np.zeros(3), 50_000.0)
        assert loop.state.is_warmed_up

    def test_last_bar_ts_updated(self, tmp_path):
        fitted = _make_fitted()
        cfg = _make_loop_config(tmp_path, warmup_bars=1)
        loop = TradingLoop(fitted=fitted, config=cfg)
        ts = pd.Timestamp("2024-06-01 12:00", tz="UTC")
        loop.step(ts, np.zeros(3), 55_000.0)
        assert loop.state.last_bar_ts == ts

    def test_returns_float_equity(self, tmp_path):
        fitted = _make_fitted()
        cfg = _make_loop_config(tmp_path, warmup_bars=1)
        loop = TradingLoop(fitted=fitted, config=cfg)
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        result = loop.step(ts, np.zeros(3), 50_000.0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# TradingLoop.run_once
# ---------------------------------------------------------------------------

class TestTradingLoopRunOnce:
    def test_processes_all_bars(self, tmp_path):
        fitted = _make_fitted()
        cfg = _make_loop_config(tmp_path, warmup_bars=5)
        loop = TradingLoop(fitted=fitted, config=cfg)
        bars = _make_bars_df(20)
        loop.run_once(bars)
        assert loop.state.n_bars_processed == 20

    def test_returns_float(self, tmp_path):
        fitted = _make_fitted()
        cfg = _make_loop_config(tmp_path, warmup_bars=5)
        loop = TradingLoop(fitted=fitted, config=cfg)
        bars = _make_bars_df(10)
        result = loop.run_once(bars)
        assert isinstance(result, float)

    def test_raises_missing_features(self, tmp_path):
        fitted = _make_fitted()
        cfg = _make_loop_config(tmp_path, warmup_bars=1)
        loop = TradingLoop(fitted=fitted, config=cfg)
        bars = pd.DataFrame({"Close": [50_000.0]}, index=[pd.Timestamp("2024-01-01", tz="UTC")])
        with pytest.raises(ValueError, match="missing feature"):
            loop.run_once(bars)

    def test_raises_missing_close(self, tmp_path):
        fitted = _make_fitted()
        cfg = _make_loop_config(tmp_path, warmup_bars=1)
        loop = TradingLoop(fitted=fitted, config=cfg)
        bars = _make_bars_df(5).drop(columns=["Close"])
        with pytest.raises(ValueError, match="Close"):
            loop.run_once(bars)

    def test_snapshots_populated_after_warmup(self, tmp_path):
        fitted = _make_fitted()
        cfg = _make_loop_config(tmp_path, warmup_bars=3)
        loop = TradingLoop(fitted=fitted, config=cfg)
        bars = _make_bars_df(10)
        loop.run_once(bars)
        # Warmup condition: n_bars_processed < warmup_bars skips bars 1 and 2
        # bars 3-10 (8 bars) get snapshots
        assert len(loop.state.snapshots) == 10 - (cfg.warmup_bars - 1)


# ---------------------------------------------------------------------------
# TradingLoop.save_state
# ---------------------------------------------------------------------------

class TestTradingLoopSaveState:
    def test_writes_snapshot_parquet(self, tmp_path):
        fitted = _make_fitted()
        cfg = _make_loop_config(tmp_path, warmup_bars=2)
        loop = TradingLoop(fitted=fitted, config=cfg)
        bars = _make_bars_df(10)
        loop.run_once(bars)
        loop.save_state()
        snap_path = tmp_path / "live" / "portfolio_snapshots.parquet"
        assert snap_path.exists()
        snap_df = pd.read_parquet(snap_path)
        assert "equity" in snap_df.columns
        assert "regime" in snap_df.columns

    def test_snapshot_equity_is_positive(self, tmp_path):
        fitted = _make_fitted()
        cfg = _make_loop_config(tmp_path, warmup_bars=2)
        loop = TradingLoop(fitted=fitted, config=cfg)
        bars = _make_bars_df(15)
        loop.run_once(bars)
        loop.save_state()
        snap_df = pd.read_parquet(tmp_path / "live" / "portfolio_snapshots.parquet")
        assert (snap_df["equity"] > 0).all()
