"""Tests for the vectorized backtest engine (Phase 9).

Covers:
- BacktestConfig defaults
- run_backtest output shapes
- Equity curve initial value
- No-lookahead guarantee (flat first bar)
- All three position modes (proportional, threshold, long_only)
- Threshold mode neutral band
- Long-only mode: no short positions
- Zero vs non-zero transaction costs
- compute_metrics keys and value ranges
- Metrics correctness (hit_rate, max_drawdown, sharpe)
- All-positive signal on rising market -> positive total_return
- calmar=nan when max_drawdown=0
- backtest_from_parquet integration
- ValueError on bad mode
- ValueError on length mismatch
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rde.backtest.engine import BacktestConfig, BacktestResult, backtest_from_parquet, run_backtest
from rde.backtest.metrics import compute_metrics

# ── Fixtures ──────────────────────────────────────────────────────────────────

N = 100  # bars for synthetic tests


def make_returns(n: int = N, seed: int = 0) -> pd.Series:
    """Small synthetic log-return series."""
    rng = np.random.default_rng(seed)
    vals = rng.normal(0.0001, 0.01, size=n)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.Series(vals, index=idx, name="log_return")


def make_signal(n: int = N, seed: int = 1) -> pd.Series:
    """Synthetic signal in [-1, 1]."""
    rng = np.random.default_rng(seed)
    vals = np.clip(rng.normal(0.0, 0.6, size=n), -1.0, 1.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.Series(vals, index=idx, name="signal")


def make_rising_returns(n: int = N) -> pd.Series:
    """Deterministic positive returns for bull-market tests."""
    vals = np.full(n, 0.001)  # +0.1% every bar
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.Series(vals, index=idx, name="log_return")


def make_positive_signal(n: int = N) -> pd.Series:
    """Signal that is always +1 (fully long)."""
    vals = np.ones(n)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.Series(vals, index=idx, name="signal")


# ── BacktestConfig tests ──────────────────────────────────────────────────────


def test_backtest_config_defaults():
    cfg = BacktestConfig()
    assert cfg.mode == "proportional", "Default mode should be 'proportional'"
    assert cfg.threshold == 0.25, "Default threshold should be 0.25"
    assert cfg.transaction_cost_bps == 1.0, "Default cost should be 1.0 bps"
    assert cfg.initial_capital == 10_000.0, "Default initial capital should be 10_000"
    assert cfg.bars_per_year == 8760.0, "Default bars_per_year should be 8760"


def test_backtest_config_custom():
    cfg = BacktestConfig(mode="threshold", threshold=0.1, transaction_cost_bps=5.0,
                         initial_capital=1_000.0, bars_per_year=252.0)
    assert cfg.mode == "threshold"
    assert cfg.threshold == 0.1
    assert cfg.transaction_cost_bps == 5.0
    assert cfg.initial_capital == 1_000.0
    assert cfg.bars_per_year == 252.0


# ── run_backtest shape tests ──────────────────────────────────────────────────


def test_run_backtest_output_length():
    rets = make_returns()
    sig = make_signal()
    result = run_backtest(rets, sig)
    n = len(rets)
    assert len(result.equity_curve) == n, "equity_curve length must match input"
    assert len(result.positions) == n, "positions length must match input"
    assert len(result.strategy_returns) == n, "strategy_returns length must match input"
    assert len(result.net_returns) == n, "net_returns length must match input"


def test_run_backtest_returns_backtest_result():
    rets = make_returns()
    sig = make_signal()
    result = run_backtest(rets, sig)
    assert isinstance(result, BacktestResult), "run_backtest must return a BacktestResult"


def test_run_backtest_index_preserved():
    rets = make_returns()
    sig = make_signal()
    result = run_backtest(rets, sig)
    pd.testing.assert_index_equal(
        result.equity_curve.index, rets.index,
        check_names=False,
        obj="equity_curve index must match returns index",
    )


# ── Equity curve initial value ────────────────────────────────────────────────


def test_equity_curve_starts_at_initial_capital():
    cfg = BacktestConfig(initial_capital=10_000.0, transaction_cost_bps=0.0)
    rets = make_returns()
    sig = make_positive_signal()
    result = run_backtest(rets, sig, config=cfg)
    # First bar: position.shift(1) = 0, so net_return[0] = 0 - cost[0]
    # cost[0] = |pos[0] - 0| * bps_per_unit = 1.0 * 0 = 0
    # equity[0] = 10_000 * (1 + 0) = 10_000
    assert result.equity_curve.iloc[0] == pytest.approx(10_000.0), (
        "Equity curve first bar should equal initial_capital when first bar is flat"
    )


def test_equity_curve_initial_capital_respected():
    cfg = BacktestConfig(initial_capital=50_000.0, transaction_cost_bps=0.0)
    rets = make_returns()
    sig = pd.Series(np.zeros(N), index=rets.index)  # always flat
    result = run_backtest(rets, sig, config=cfg)
    # Flat signal → all strategy returns = 0 → equity is flat at 50_000
    assert result.equity_curve.iloc[0] == pytest.approx(50_000.0)
    assert result.equity_curve.iloc[-1] == pytest.approx(50_000.0)


# ── No-lookahead tests ────────────────────────────────────────────────────────


def test_no_lookahead_first_bar_flat():
    """Position shift must produce 0 at t=0 (flat at start)."""
    rets = make_returns()
    sig = make_positive_signal()  # signal always +1
    result = run_backtest(rets, sig, config=BacktestConfig(transaction_cost_bps=0.0))
    # At t=0 the lagged position is 0, so strategy_return[0] = 0 * ret[0] = 0
    assert result.strategy_returns.iloc[0] == pytest.approx(0.0), (
        "First bar strategy return must be 0 (flat before data starts)"
    )


def test_no_lookahead_second_bar_uses_previous_signal():
    """strategy_return[1] must equal position[0] * returns[1]."""
    rets = make_returns()
    sig = make_signal()
    cfg = BacktestConfig(mode="proportional", transaction_cost_bps=0.0)
    result = run_backtest(rets, sig, config=cfg)
    expected = float(result.positions.iloc[0]) * float(rets.iloc[1])
    actual = float(result.strategy_returns.iloc[1])
    assert actual == pytest.approx(expected, rel=1e-9), (
        "strategy_return[1] should equal position[0] * return[1]"
    )


# ── Position mode tests ───────────────────────────────────────────────────────


def test_proportional_clips_to_minus_one_plus_one():
    n = 10
    sig_vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.3, 0.7, 1.0, 1.5, 2.0])
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    sig = pd.Series(sig_vals, index=idx)
    rets = pd.Series(np.zeros(n), index=idx)
    cfg = BacktestConfig(mode="proportional", transaction_cost_bps=0.0)
    result = run_backtest(rets, sig, config=cfg)
    assert (result.positions.values >= -1.0).all(), "Proportional positions must be >= -1"
    assert (result.positions.values <= 1.0).all(), "Proportional positions must be <= +1"


def test_threshold_mode_above_threshold_is_long():
    sig_vals = np.array([0.5, 0.8, 1.0])
    idx = pd.date_range("2024-01-01", periods=3, freq="h")
    sig = pd.Series(sig_vals, index=idx)
    rets = pd.Series(np.zeros(3), index=idx)
    cfg = BacktestConfig(mode="threshold", threshold=0.25, transaction_cost_bps=0.0)
    result = run_backtest(rets, sig, config=cfg)
    assert (result.positions.values == 1.0).all(), (
        "Signals above threshold should produce position=+1"
    )


def test_threshold_mode_below_neg_threshold_is_short():
    sig_vals = np.array([-0.5, -0.8, -1.0])
    idx = pd.date_range("2024-01-01", periods=3, freq="h")
    sig = pd.Series(sig_vals, index=idx)
    rets = pd.Series(np.zeros(3), index=idx)
    cfg = BacktestConfig(mode="threshold", threshold=0.25, transaction_cost_bps=0.0)
    result = run_backtest(rets, sig, config=cfg)
    assert (result.positions.values == -1.0).all(), (
        "Signals below -threshold should produce position=-1"
    )


def test_threshold_mode_neutral_band():
    """Signal = 0.0 with threshold=0.25 should give position=0."""
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    sig = pd.Series(np.zeros(5), index=idx)
    rets = pd.Series(np.zeros(5), index=idx)
    cfg = BacktestConfig(mode="threshold", threshold=0.25, transaction_cost_bps=0.0)
    result = run_backtest(rets, sig, config=cfg)
    assert (result.positions.values == 0.0).all(), (
        "Signal=0 must produce neutral position in threshold mode"
    )


def test_long_only_no_short_positions():
    rets = make_returns()
    sig = make_signal()  # has both positive and negative values
    cfg = BacktestConfig(mode="long_only", transaction_cost_bps=0.0)
    result = run_backtest(rets, sig, config=cfg)
    assert (result.positions.values >= 0.0).all(), (
        "long_only mode must never produce short positions"
    )
    assert (result.positions.values <= 1.0).all(), (
        "long_only mode must not exceed +1"
    )


# ── Transaction cost tests ────────────────────────────────────────────────────


def test_zero_cost_strategy_returns_equal_net_returns():
    rets = make_returns()
    sig = make_signal()
    cfg = BacktestConfig(transaction_cost_bps=0.0)
    result = run_backtest(rets, sig, config=cfg)
    np.testing.assert_allclose(
        result.strategy_returns.values,
        result.net_returns.values,
        rtol=1e-12,
        err_msg="With zero cost, strategy_returns must equal net_returns",
    )


def test_nonzero_cost_net_returns_le_strategy_returns():
    rets = make_returns()
    sig = make_signal()
    cfg = BacktestConfig(transaction_cost_bps=5.0)
    result = run_backtest(rets, sig, config=cfg)
    assert (result.net_returns.values <= result.strategy_returns.values + 1e-12).all(), (
        "With positive costs, net_returns must be <= strategy_returns bar-by-bar"
    )


# ── compute_metrics key checks ────────────────────────────────────────────────

_REQUIRED_KEYS = {
    "total_return", "cagr", "sharpe_ann", "max_drawdown",
    "calmar", "hit_rate", "profit_factor", "n_trades", "avg_return_per_trade",
}


def test_compute_metrics_has_all_keys():
    rets = make_returns()
    sig = make_signal()
    result = run_backtest(rets, sig)
    missing = _REQUIRED_KEYS - set(result.metrics.keys())
    assert not missing, f"Metrics dict is missing keys: {missing}"


def test_compute_metrics_standalone():
    """compute_metrics can be called directly."""
    n = 50
    eq = pd.Series(np.cumprod(1 + np.random.default_rng(7).normal(0.001, 0.01, n)) * 10_000)
    strat_rets = pd.Series(np.random.default_rng(7).normal(0.001, 0.01, n))
    pos = pd.Series(np.where(strat_rets > 0, 1.0, -1.0))
    m = compute_metrics(eq, strat_rets, pos)
    assert set(m.keys()) == _REQUIRED_KEYS, "All metric keys must be present"


# ── Metric value range tests ──────────────────────────────────────────────────


def test_hit_rate_in_zero_one():
    rets = make_returns(seed=42)
    sig = make_signal(seed=42)
    result = run_backtest(rets, sig)
    hr = result.metrics["hit_rate"]
    if not np.isnan(hr):
        assert 0.0 <= hr <= 1.0, f"hit_rate must be in [0, 1], got {hr}"


def test_max_drawdown_in_zero_one():
    rets = make_returns()
    sig = make_signal()
    result = run_backtest(rets, sig)
    mdd = result.metrics["max_drawdown"]
    if not np.isnan(mdd):
        assert 0.0 <= mdd <= 1.0, f"max_drawdown must be in [0, 1], got {mdd}"


def test_sharpe_finite_for_nonconstant_returns():
    rets = make_returns(n=500, seed=10)
    sig = make_signal(n=500, seed=10)
    result = run_backtest(rets, sig)
    sharpe = result.metrics["sharpe_ann"]
    assert np.isfinite(sharpe), (
        f"Sharpe should be finite for non-constant returns, got {sharpe}"
    )


# ── Directional correctness ───────────────────────────────────────────────────


def test_all_positive_signal_rising_market_positive_total_return():
    """Fully long signal on a rising market must produce a positive total return."""
    rets = make_rising_returns()
    sig = make_positive_signal()
    cfg = BacktestConfig(transaction_cost_bps=0.0)
    result = run_backtest(rets, sig, config=cfg)
    tr = result.metrics["total_return"]
    assert tr > 0.0, (
        f"All-positive signal on rising market should give positive total_return, got {tr}"
    )


# ── calmar=nan when max_drawdown=0 ─────────────────────────────────────────────


def test_calmar_nan_when_max_drawdown_zero():
    """Monotonically increasing equity -> max_drawdown=0 -> calmar=nan."""
    # Construct a scenario where equity only ever goes up
    n = 50
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    equity = pd.Series(np.arange(1, n + 1, dtype=float) * 100.0, index=idx)
    strat_rets = pd.Series(np.full(n, 0.01), index=idx)
    pos = pd.Series(np.ones(n), index=idx)
    m = compute_metrics(equity, strat_rets, pos)
    assert m["max_drawdown"] == pytest.approx(0.0), "Expected zero drawdown"
    assert np.isnan(m["calmar"]), "calmar must be nan when max_drawdown=0"


# ── ValueError tests ──────────────────────────────────────────────────────────


def test_run_backtest_raises_on_bad_mode():
    rets = make_returns()
    sig = make_signal()
    with pytest.raises(ValueError, match="Invalid mode"):
        run_backtest(rets, sig, config=BacktestConfig(mode="binary"))


def test_run_backtest_raises_on_length_mismatch():
    rets = make_returns(n=100)
    sig = make_signal(n=50)  # different length
    with pytest.raises(ValueError, match="same length"):
        run_backtest(rets, sig)


# ── backtest_from_parquet integration ────────────────────────────────────────

_RESULTS_DIR = Path(__file__).parent.parent / "results" / "BTC-USD"
_REGIMES_PATH = _RESULTS_DIR / "regimes.parquet"
_SIGNALS_PATH = _RESULTS_DIR / "signals.parquet"


@pytest.mark.skipif(
    not (_REGIMES_PATH.exists() and _SIGNALS_PATH.exists()),
    reason="BTC-USD result parquets not present; run `rde run --config configs/btc.yaml --signal` first.",
)
def test_backtest_from_parquet_integration():
    """Integration test using the real BTC-USD results."""
    cfg = BacktestConfig(transaction_cost_bps=1.0)
    result = backtest_from_parquet(
        regimes_path=_REGIMES_PATH,
        signals_path=_SIGNALS_PATH,
        config=cfg,
    )
    assert isinstance(result, BacktestResult), "backtest_from_parquet must return BacktestResult"
    assert len(result.equity_curve) > 0, "Equity curve should be non-empty"
    assert set(result.metrics.keys()) == _REQUIRED_KEYS, "All metric keys must be present"
    assert result.equity_curve.iloc[0] == pytest.approx(cfg.initial_capital), (
        "First equity value should equal initial_capital"
    )


@pytest.mark.skipif(
    not (_REGIMES_PATH.exists() and _SIGNALS_PATH.exists()),
    reason="BTC-USD result parquets not present.",
)
def test_backtest_from_parquet_metrics_in_range():
    """Sanity-check metric ranges on real data."""
    result = backtest_from_parquet(
        regimes_path=_REGIMES_PATH,
        signals_path=_SIGNALS_PATH,
    )
    mdd = result.metrics["max_drawdown"]
    hr = result.metrics["hit_rate"]
    if not np.isnan(mdd):
        assert 0.0 <= mdd <= 1.0, f"max_drawdown out of range: {mdd}"
    if not np.isnan(hr):
        assert 0.0 <= hr <= 1.0, f"hit_rate out of range: {hr}"


@pytest.mark.skipif(
    not (_SIGNALS_PATH.exists()),
    reason="BTC-USD signals.parquet not present.",
)
def test_backtest_from_parquet_missing_return_col_raises():
    """backtest_from_parquet raises KeyError on missing return column."""
    with pytest.raises(KeyError, match="not found in regimes parquet"):
        backtest_from_parquet(
            regimes_path=_REGIMES_PATH,
            signals_path=_SIGNALS_PATH,
            return_col="nonexistent_column",
        )
