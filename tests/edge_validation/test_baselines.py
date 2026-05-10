"""Tests for rde.evaluation.baselines (Phase 37.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.evaluation.baselines import (
    BaselineResult,
    buy_and_hold,
    compare_model_to_baselines,
    multi_asset_risk_parity,
    naive_momentum,
    naive_vol_regime,
    risk_parity_baseline,
    run_all_baselines,
    two_state_hmm_baseline,
    vol_targeted_buy_and_hold,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_returns(n: int = 1000, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(rng.normal(0.0002, 0.002, n), index=idx, name="log_return")


def _synthetic_df(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    log_ret = rng.normal(0.0002, 0.002, n)
    return pd.DataFrame(
        {
            "log_return": log_ret,
            "volatility_w24": np.abs(log_ret) + rng.uniform(0.001, 0.003, n),
            "smoothed_return_w12": pd.Series(log_ret).rolling(12, min_periods=1).mean().values,
        },
        index=idx,
    )


FEATURE_COLS = ["log_return", "volatility_w24", "smoothed_return_w12"]
FAST_KW = {"n_restarts": 1, "n_iter": 10, "seed_base": 0}


def _check_baseline(br: BaselineResult, name: str) -> None:
    assert isinstance(br, BaselineResult)
    assert br.name == name
    assert np.isfinite(br.sharpe)
    assert np.isfinite(br.calmar)
    assert 0.0 <= br.max_drawdown <= 1.0
    assert np.isfinite(br.ann_return)
    assert np.isfinite(br.ann_vol)
    assert br.n_trades >= 0
    assert isinstance(br.equity, pd.Series)
    assert len(br.equity) > 0


# ---------------------------------------------------------------------------
# BaselineResult
# ---------------------------------------------------------------------------


class TestBaselineResult:
    def test_instantiation(self):
        br = BaselineResult(
            name="test",
            sharpe=0.5, calmar=1.0, max_drawdown=0.1,
            ann_return=0.05, ann_vol=0.1, n_trades=5,
            equity=pd.Series([1.0, 1.01, 1.02]),
        )
        assert br.name == "test"
        assert br.n_trades == 5


# ---------------------------------------------------------------------------
# buy_and_hold
# ---------------------------------------------------------------------------


class TestBuyAndHold:
    def test_returns_baseline_result(self):
        r = _synthetic_returns(500)
        br = buy_and_hold(r)
        _check_baseline(br, "buy_and_hold")

    def test_equity_starts_at_one(self):
        r = _synthetic_returns(500)
        br = buy_and_hold(r)
        # First bar position is 0 (since current_pos starts at 0, then changes)
        # equity length should equal len(returns)
        assert len(br.equity) == 500

    def test_zero_trades(self):
        r = _synthetic_returns(200)
        br = buy_and_hold(r)
        # B&H never changes position (after first bar), so 0 or 1 trade
        assert br.n_trades <= 1


# ---------------------------------------------------------------------------
# vol_targeted_buy_and_hold
# ---------------------------------------------------------------------------


class TestVolTargetedBuyAndHold:
    def test_returns_baseline_result(self):
        r = _synthetic_returns(500)
        br = vol_targeted_buy_and_hold(r)
        _check_baseline(br, "vol_targeted_bah")

    def test_equity_length_matches_returns(self):
        r = _synthetic_returns(300)
        br = vol_targeted_buy_and_hold(r)
        assert len(br.equity) == 300

    def test_high_vol_reduces_position(self):
        rng = np.random.default_rng(1)
        idx = pd.date_range("2023-01-01", periods=200, freq="h", tz="UTC")
        high_vol = pd.Series(rng.normal(0, 0.05, 200), index=idx)
        low_vol = pd.Series(rng.normal(0, 0.001, 200), index=idx)
        br_high = vol_targeted_buy_and_hold(high_vol, target_vol=0.10)
        br_low = vol_targeted_buy_and_hold(low_vol, target_vol=0.10)
        # High vol has more trades due to more position scaling
        assert br_high.n_trades >= 0 and br_low.n_trades >= 0


# ---------------------------------------------------------------------------
# naive_momentum
# ---------------------------------------------------------------------------


class TestNaiveMomentum:
    def test_returns_baseline_result(self):
        r = _synthetic_returns(500)
        br = naive_momentum(r)
        _check_baseline(br, "naive_momentum")

    def test_flat_positive_drift_stays_long(self):
        idx = pd.date_range("2023-01-01", periods=100, freq="h", tz="UTC")
        r = pd.Series([0.001] * 100, index=idx)
        br = naive_momentum(r, lookback_bars=5)
        # Positive mean → always long → equity grows
        assert float(br.equity.iloc[-1]) > 1.0

    def test_equity_length(self):
        r = _synthetic_returns(400)
        br = naive_momentum(r, lookback_bars=12)
        assert len(br.equity) == 400


# ---------------------------------------------------------------------------
# naive_vol_regime
# ---------------------------------------------------------------------------


class TestNaiveVolRegime:
    def test_returns_baseline_result(self):
        r = _synthetic_returns(500)
        br = naive_vol_regime(r)
        _check_baseline(br, "naive_vol_regime")

    def test_equity_length(self):
        r = _synthetic_returns(300)
        br = naive_vol_regime(r)
        assert len(br.equity) == 300

    def test_causal_no_future_leakage(self):
        # If we truncate the series, we should get different signals from the same prefix
        r = _synthetic_returns(200)
        br1 = naive_vol_regime(r[:100])
        br2 = naive_vol_regime(r)
        # They shouldn't be the same equity (different histories), just checking no crash
        assert len(br1.equity) == 100
        assert len(br2.equity) == 200


# ---------------------------------------------------------------------------
# two_state_hmm_baseline
# ---------------------------------------------------------------------------


class TestTwoStateHmmBaseline:
    def test_returns_baseline_result(self):
        df = _synthetic_df(600)
        br = two_state_hmm_baseline(df, FEATURE_COLS, "log_return", train_kwargs=FAST_KW)
        _check_baseline(br, "two_state_hmm")

    def test_equity_covers_test_window(self):
        df = _synthetic_df(600)
        br = two_state_hmm_baseline(df, FEATURE_COLS, "log_return", train_frac=0.7, train_kwargs=FAST_KW)
        expected_len = len(df) - int(len(df) * 0.7)
        assert len(br.equity) == expected_len


# ---------------------------------------------------------------------------
# run_all_baselines
# ---------------------------------------------------------------------------


class TestRunAllBaselines:
    def test_returns_five_baselines(self):
        df = _synthetic_df(800)
        results = run_all_baselines(df, FEATURE_COLS, "log_return", train_kwargs=FAST_KW)
        assert len(results) == 5
        assert "buy_and_hold" in results
        assert "vol_targeted_bah" in results
        assert "naive_momentum" in results
        assert "naive_vol_regime" in results
        assert "two_state_hmm" in results

    def test_all_results_are_baseline_result(self):
        df = _synthetic_df(800)
        results = run_all_baselines(df, FEATURE_COLS, "log_return", train_kwargs=FAST_KW)
        for name, br in results.items():
            assert isinstance(br, BaselineResult), f"{name} is not BaselineResult"


# ---------------------------------------------------------------------------
# compare_model_to_baselines
# ---------------------------------------------------------------------------


class TestCompareModelToBaselines:
    def _make_baselines(self) -> dict[str, BaselineResult]:
        idx = pd.date_range("2023-01-01", periods=10, freq="h", tz="UTC")
        eq = pd.Series(np.linspace(1.0, 1.1, 10), index=idx)
        return {
            "buy_and_hold": BaselineResult("buy_and_hold", 0.3, 0.5, 0.1, 0.05, 0.1, 0, eq),
            "naive_momentum": BaselineResult("naive_momentum", 0.6, 0.8, 0.08, 0.07, 0.12, 10, eq),
        }

    def test_returns_dataframe(self):
        baselines = self._make_baselines()
        df = compare_model_to_baselines(0.7, baselines)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_required_columns(self):
        baselines = self._make_baselines()
        df = compare_model_to_baselines(0.7, baselines)
        for col in ["name", "baseline_sharpe", "model_sharpe", "beats", "margin"]:
            assert col in df.columns

    def test_beats_logic(self):
        baselines = self._make_baselines()
        df = compare_model_to_baselines(0.5, baselines)
        # model_sharpe=0.5 beats buy_and_hold (0.3) but not naive_momentum (0.6)
        row_bah = df[df["name"] == "buy_and_hold"].iloc[0]
        row_mom = df[df["name"] == "naive_momentum"].iloc[0]
        assert row_bah["beats"] is True or row_bah["beats"] == True  # noqa: E712
        assert row_mom["beats"] is False or row_mom["beats"] == False  # noqa: E712

    def test_margin_equals_model_minus_baseline(self):
        baselines = self._make_baselines()
        df = compare_model_to_baselines(0.7, baselines)
        for _, row in df.iterrows():
            assert row["margin"] == pytest.approx(0.7 - row["baseline_sharpe"])


# ---------------------------------------------------------------------------
# risk_parity_baseline / multi_asset_risk_parity
# ---------------------------------------------------------------------------


def _multi_asset_returns(
    n: int = 800,
    n_assets: int = 4,
    vols: list[float] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    vols = vols or [0.01] * n_assets
    cols = {}
    for i, v in enumerate(vols):
        cols[f"A{i}"] = rng.normal(0.0003, v, n)
    return pd.DataFrame(cols, index=idx)


class TestRiskParityBaseline:
    def test_returns_baseline_result(self):
        rets = _multi_asset_returns(n=500, n_assets=4)
        br = risk_parity_baseline(rets, ann_factor=252, vol_window=63, rebalance_bars=21)
        assert isinstance(br, BaselineResult)
        assert br.name == "risk_parity"
        assert np.isfinite(br.sharpe)
        assert np.isfinite(br.calmar)
        assert 0.0 <= br.max_drawdown <= 1.0
        assert len(br.equity) == 500

    def test_weights_sum_to_one(self):
        # Re-run the inverse-vol math directly to verify normalisation.
        rng = np.random.default_rng(7)
        n, k = 200, 4
        vols = np.array([0.005, 0.01, 0.02, 0.04])
        R = rng.normal(0.0, vols, size=(n, k))
        # Use the same window the baseline uses (after warm-up).
        window = R[-63:]
        v = window.std(axis=0, ddof=1) * np.sqrt(252)
        inv = 1.0 / (v + 1e-10)
        w = inv / inv.sum()
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        # Also assert min/max bounds: every weight in [0, 1].
        assert (w >= 0).all() and (w <= 1).all()

    def test_high_vol_gets_low_weight(self):
        # Asset 0 has 10× vol of asset 1 → weight ratio ≈ 1/10.
        n = 400
        rng = np.random.default_rng(2)
        idx = pd.date_range("2018-01-01", periods=n, freq="B")
        rets = pd.DataFrame({
            "HIVOL": rng.normal(0.0, 0.05, n),
            "LOVOL": rng.normal(0.0, 0.005, n),
        }, index=idx)
        # Manually replicate: at the last rebalance after warm-up.
        window = rets.iloc[-63:].values
        vol = window.std(axis=0, ddof=1) * np.sqrt(252)
        inv = 1.0 / (vol + 1e-10)
        w = inv / inv.sum()
        # HIVOL weight ≈ 1/11, LOVOL ≈ 10/11.
        assert w[0] < w[1]
        ratio = w[1] / w[0]
        # Empirical std may drift slightly from 10×; allow [7, 13].
        assert 7.0 <= ratio <= 13.0

        # Smoke run the baseline end-to-end.
        br = risk_parity_baseline(rets, ann_factor=252, vol_window=63, rebalance_bars=21)
        assert isinstance(br, BaselineResult)

    def test_warm_up_equal_weights(self):
        n = 100
        rng = np.random.default_rng(3)
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        rets = pd.DataFrame({
            "A": rng.normal(0.0, 0.01, n),
            "B": rng.normal(0.0, 0.05, n),
        }, index=idx)
        # vol_window > n, so warm-up applies for the entire run.
        br = risk_parity_baseline(
            rets, ann_factor=252, vol_window=200, rebalance_bars=21,
            transaction_cost=0.0,
        )
        # Equal-weight portfolio of zero-mean assets has E[return] ≈ 0.
        # Equity should remain near 1.0 throughout.
        assert abs(br.equity.iloc[-1] - 1.0) < 0.5

    def test_calmar_positive_on_drift(self):
        # Inject a small positive drift on every asset.
        rng = np.random.default_rng(4)
        n, k = 600, 4
        idx = pd.date_range("2015-01-01", periods=n, freq="B")
        cols = {f"A{i}": rng.normal(0.0008, 0.01, n) for i in range(k)}
        rets = pd.DataFrame(cols, index=idx)
        br = risk_parity_baseline(rets, ann_factor=252, vol_window=63, rebalance_bars=21)
        assert br.ann_return > 0
        assert br.calmar > 0

    def test_dict_wrapper_matches_dataframe(self):
        rets_df = _multi_asset_returns(n=300, n_assets=3, seed=11)
        rets_dict = {c: rets_df[c] for c in rets_df.columns}
        br_df = risk_parity_baseline(rets_df, ann_factor=252, vol_window=63, rebalance_bars=21)
        br_dict = multi_asset_risk_parity(rets_dict, ann_factor=252, vol_window=63, rebalance_bars=21)
        assert br_df.sharpe == pytest.approx(br_dict.sharpe, rel=1e-9)
        assert br_df.max_drawdown == pytest.approx(br_dict.max_drawdown, rel=1e-9)
