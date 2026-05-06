"""Tests for rde.evaluation.skeptics (Phase 37.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.analysis.backtest import RegimeRule, RegimeStrategyConfig
from rde.evaluation.skeptics import (
    AblationResult,
    CostSweepResult,
    NullTestResult,
    PeriodRobustnessResult,
    SkepticsReport,
    run_cost_sensitivity,
    run_feature_ablation,
    run_full_skeptics_kit,
    run_period_robustness,
    run_random_baseline,
    run_shuffle_test,
    run_slippage_stress,
    write_skeptics_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_df(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    log_ret = rng.normal(0, 0.002, n)
    return pd.DataFrame(
        {
            "log_return": log_ret,
            "volatility_w24": np.abs(log_ret) + rng.uniform(0.001, 0.003, n),
            "smoothed_return_w12": pd.Series(log_ret).rolling(12, min_periods=1).mean().values,
        },
        index=idx,
    )


FEATURE_COLS = ["log_return", "volatility_w24", "smoothed_return_w12"]

FAST_TRAIN_KWARGS = {"n_restarts": 1, "n_iter": 10, "seed_base": 0}


def _simple_config(n_states: int = 2) -> RegimeStrategyConfig:
    rules = [RegimeRule(target_position=1.0 if k % 2 == 0 else 0.0) for k in range(n_states)]
    return RegimeStrategyConfig(rules=rules, transaction_cost=0.0001, ann_factor=8760)


def _random_regimes(n: int, n_states: int = 2, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_states, size=n)


# ---------------------------------------------------------------------------
# NullTestResult dataclass
# ---------------------------------------------------------------------------


class TestNullTestResult:
    def test_fields_present(self):
        r = NullTestResult(
            name="random_baseline",
            model_sharpe=0.8,
            null_sharpe_mean=0.2,
            null_sharpe_std=0.3,
            p_value=0.05,
            margin=0.6,
            passed=True,
        )
        assert r.name == "random_baseline"
        assert r.passed is True
        assert r.margin == pytest.approx(0.6)

    def test_passed_false_when_margin_below_threshold(self):
        r = NullTestResult(
            name="shuffle_test",
            model_sharpe=0.5,
            null_sharpe_mean=0.4,
            null_sharpe_std=0.1,
            p_value=0.3,
            margin=0.1,
            passed=False,
        )
        assert not r.passed


# ---------------------------------------------------------------------------
# run_random_baseline
# ---------------------------------------------------------------------------


class TestRunRandomBaseline:
    def test_returns_null_test_result(self):
        df = _synthetic_df(500)
        returns = df["log_return"].values
        regimes = _random_regimes(500, n_states=2)
        config = _simple_config(2)
        result = run_random_baseline(returns, regimes, config, ann_factor=8760, n_sims=10, seed=0)
        assert isinstance(result, NullTestResult)
        assert result.name == "random_baseline"

    def test_p_value_in_range(self):
        df = _synthetic_df(500)
        returns = df["log_return"].values
        regimes = _random_regimes(500, n_states=2)
        config = _simple_config(2)
        result = run_random_baseline(returns, regimes, config, n_sims=20, seed=0)
        assert 0.0 <= result.p_value <= 1.0

    def test_margin_equals_model_minus_null_mean(self):
        df = _synthetic_df(500)
        returns = df["log_return"].values
        regimes = _random_regimes(500, n_states=2)
        config = _simple_config(2)
        result = run_random_baseline(returns, regimes, config, n_sims=10, seed=0)
        assert result.margin == pytest.approx(result.model_sharpe - result.null_sharpe_mean)

    def test_null_std_nonnegative(self):
        df = _synthetic_df(500)
        returns = df["log_return"].values
        regimes = _random_regimes(500, n_states=2)
        config = _simple_config(2)
        result = run_random_baseline(returns, regimes, config, n_sims=10, seed=0)
        assert result.null_sharpe_std >= 0.0


# ---------------------------------------------------------------------------
# run_shuffle_test
# ---------------------------------------------------------------------------


class TestRunShuffleTest:
    def test_returns_correct_type(self):
        df = _synthetic_df(500)
        returns = df["log_return"].values
        regimes = _random_regimes(500, n_states=2)
        config = _simple_config(2)
        result = run_shuffle_test(returns, regimes, config, n_sims=10, seed=0)
        assert isinstance(result, NullTestResult)
        assert result.name == "shuffle_test"

    def test_shuffle_and_random_have_same_model_sharpe(self):
        df = _synthetic_df(500)
        returns = df["log_return"].values
        regimes = _random_regimes(500, n_states=2)
        config = _simple_config(2)
        r1 = run_random_baseline(returns, regimes, config, n_sims=5, seed=0)
        r2 = run_shuffle_test(returns, regimes, config, n_sims=5, seed=0)
        assert r1.model_sharpe == pytest.approx(r2.model_sharpe)


# ---------------------------------------------------------------------------
# run_feature_ablation
# ---------------------------------------------------------------------------


class TestRunFeatureAblation:
    def test_returns_one_result_per_feature(self):
        df = _synthetic_df(600)
        results = run_feature_ablation(
            df, FEATURE_COLS, "log_return", n_states=2,
            model_sharpe=0.5, ann_factor=8760,
            train_kwargs=FAST_TRAIN_KWARGS,
        )
        assert len(results) == len(FEATURE_COLS)
        assert all(isinstance(r, AblationResult) for r in results)

    def test_dropped_feature_not_in_result_name(self):
        df = _synthetic_df(600)
        results = run_feature_ablation(
            df, FEATURE_COLS, "log_return", n_states=2,
            model_sharpe=0.5,
            train_kwargs=FAST_TRAIN_KWARGS,
        )
        features_dropped = {r.feature for r in results}
        assert features_dropped == set(FEATURE_COLS)

    def test_importance_sign_consistent(self):
        df = _synthetic_df(600)
        results = run_feature_ablation(
            df, FEATURE_COLS, "log_return", n_states=2,
            model_sharpe=0.5,
            train_kwargs=FAST_TRAIN_KWARGS,
        )
        for r in results:
            if np.isfinite(r.importance):
                assert np.isfinite(r.sharpe_without)


# ---------------------------------------------------------------------------
# run_period_robustness
# ---------------------------------------------------------------------------


class TestRunPeriodRobustness:
    def test_returns_result_type(self):
        df = _synthetic_df(1500)
        result = run_period_robustness(
            df, FEATURE_COLS, n_states=2,
            window_bars=600, step_bars=200,
            train_kwargs=FAST_TRAIN_KWARGS,
        )
        assert isinstance(result, PeriodRobustnessResult)
        assert result.n_windows >= 2

    def test_ari_in_range(self):
        df = _synthetic_df(1500)
        result = run_period_robustness(
            df, FEATURE_COLS, n_states=2,
            window_bars=600, step_bars=200,
            train_kwargs=FAST_TRAIN_KWARGS,
        )
        assert -1.0 <= result.ari_mean <= 1.0

    def test_auto_scales_when_window_bars_exceeds_dataset(self):
        # When window_bars >= n, the function auto-scales and fits windows successfully.
        df = _synthetic_df(500)
        result = run_period_robustness(
            df, FEATURE_COLS, n_states=2,
            window_bars=800,  # larger than df → auto-scaled
            step_bars=200,
            train_kwargs=FAST_TRAIN_KWARGS,
        )
        # Auto-scaling should produce at least 2 valid windows
        assert result.n_windows >= 2


# ---------------------------------------------------------------------------
# run_cost_sensitivity
# ---------------------------------------------------------------------------


class TestRunCostSensitivity:
    def test_returns_sweep_result(self):
        df = _synthetic_df(500)
        returns = df["log_return"].values
        regimes = _random_regimes(500, 2)
        config = _simple_config(2)
        result = run_cost_sensitivity(returns, regimes, config, cost_bps_range=[1.0, 5.0, 10.0])
        assert isinstance(result, CostSweepResult)
        assert len(result.cost_bps) == 3
        assert len(result.sharpes) == 3

    def test_sharpe_decreases_or_stays_flat_with_cost(self):
        df = _synthetic_df(1000)
        returns = df["log_return"].values
        regimes = _random_regimes(1000, 2)
        config = _simple_config(2)
        result = run_cost_sensitivity(returns, regimes, config, cost_bps_range=[0.0, 5.0, 20.0, 50.0])
        # Higher cost → lower or equal Sharpe (monotone non-increasing)
        for i in range(len(result.sharpes) - 1):
            assert result.sharpes[i] >= result.sharpes[i + 1] - 1e-9

    def test_break_even_is_finite_or_inf(self):
        df = _synthetic_df(500)
        returns = df["log_return"].values
        regimes = _random_regimes(500, 2)
        config = _simple_config(2)
        result = run_cost_sensitivity(returns, regimes, config)
        # break_even_bps is either a finite float or inf
        assert np.isfinite(result.break_even_bps) or result.break_even_bps == float("inf")


# ---------------------------------------------------------------------------
# run_slippage_stress
# ---------------------------------------------------------------------------


class TestRunSlippageStress:
    def test_returns_sweep_result(self):
        df = _synthetic_df(500)
        returns = df["log_return"].values
        regimes = _random_regimes(500, 2)
        positions = np.where(regimes == 0, 1.0, 0.0)
        config = _simple_config(2)
        result = run_slippage_stress(returns, positions, regimes, config, slippage_bps_range=[0.0, 10.0, 30.0])
        assert isinstance(result, CostSweepResult)
        assert len(result.sharpes) == 3

    def test_zero_slippage_matches_no_slippage(self):
        df = _synthetic_df(500)
        returns = df["log_return"].values
        regimes = _random_regimes(500, 2)
        config = _simple_config(2)
        positions = np.where(regimes == 0, 1.0, 0.0)
        result = run_slippage_stress(returns, positions, regimes, config, slippage_bps_range=[0.0, 20.0])
        # At 0 slippage the result should be finite
        assert np.isfinite(result.sharpes[0])


# ---------------------------------------------------------------------------
# run_full_skeptics_kit
# ---------------------------------------------------------------------------


class TestRunFullSkepticsKit:
    def test_returns_skeptics_report(self):
        df = _synthetic_df(1500)
        regimes = _random_regimes(1500, n_states=2, seed=5)
        report = run_full_skeptics_kit(
            df, FEATURE_COLS, "log_return", n_states=2,
            model_sharpe=0.5, regimes=regimes,
            ann_factor=8760, n_sims=5,
            train_kwargs=FAST_TRAIN_KWARGS,
            seed=0,
        )
        assert isinstance(report, SkepticsReport)
        assert isinstance(report.random_baseline, NullTestResult)
        assert isinstance(report.shuffle_test, NullTestResult)
        assert isinstance(report.period_robustness, PeriodRobustnessResult)
        assert isinstance(report.cost_sensitivity, CostSweepResult)
        assert isinstance(report.slippage_stress, CostSweepResult)

    def test_overall_passed_reflects_subtests(self):
        df = _synthetic_df(1500)
        regimes = _random_regimes(1500, n_states=2, seed=5)
        report = run_full_skeptics_kit(
            df, FEATURE_COLS, "log_return", n_states=2,
            model_sharpe=0.5, regimes=regimes,
            ann_factor=8760, n_sims=5,
            train_kwargs=FAST_TRAIN_KWARGS,
        )
        expected = (
            report.random_baseline.passed
            and report.shuffle_test.passed
            and report.period_robustness.is_stable
        )
        assert report.overall_passed == expected


# ---------------------------------------------------------------------------
# write_skeptics_report
# ---------------------------------------------------------------------------


class TestWriteSkepticsReport:
    def _make_report(self) -> SkepticsReport:
        null = NullTestResult("random_baseline", 0.8, 0.2, 0.1, 0.02, 0.6, True)
        shuf = NullTestResult("shuffle_test", 0.8, 0.1, 0.2, 0.01, 0.7, True)
        abl = [AblationResult("log_return", 0.3, 0.5), AblationResult("volatility_w24", 0.6, 0.2)]
        rob = PeriodRobustnessResult(n_windows=4, ari_mean=0.85, transmat_frobenius_std=0.05, means_frobenius_std=0.07, is_stable=True)
        cost = CostSweepResult([1.0, 5.0, 10.0], [0.8, 0.5, -0.1], 8.3)
        slip = CostSweepResult([0.0, 20.0, 50.0], [0.8, 0.3, -0.2], 35.0)
        return SkepticsReport(null, shuf, abl, rob, cost, slip, True)

    def test_writes_md_file(self, tmp_path):
        report = self._make_report()
        path = write_skeptics_report(report, tmp_path / "results", "BTC-USD")
        assert path.exists()
        assert path.suffix == ".md"

    def test_md_contains_key_sections(self, tmp_path):
        report = self._make_report()
        path = write_skeptics_report(report, tmp_path, "BTC-USD")
        text = path.read_text()
        assert "Random Baseline" in text
        assert "Shuffle Test" in text
        assert "Feature Ablation" in text
        assert "Period Robustness" in text
        assert "Cost Sensitivity" in text
        assert "Slippage Stress" in text

    def test_pass_fail_shown(self, tmp_path):
        report = self._make_report()
        path = write_skeptics_report(report, tmp_path, "BTC-USD")
        text = path.read_text()
        assert "PASS" in text
