"""Tests for src/rde/analysis/pipeline.py (Phase 31).

These tests are written against the public interface spec for AnalysisConfig,
AnalysisReport, and AnalysisPipeline. pipeline.py may not exist yet when this
file is authored; tests are designed to be collected once it is in place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.analysis.pipeline import AnalysisConfig, AnalysisPipeline, AnalysisReport


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _make_inputs(T: int = 120, d: int = 3, K: int = 2, seed: int = 0):
    """Return synthetic (X, posteriors, transmat, returns, regimes).

    Parameters
    ----------
    T:
        Number of time-steps.
    d:
        Feature dimensionality.
    K:
        Number of hidden states.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    tuple
        (X, posteriors, transmat, returns, regimes)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, d)) * 0.01
    raw = np.abs(rng.standard_normal((T, K)))
    posteriors = raw / raw.sum(axis=1, keepdims=True)
    # Simple random-walk transition matrix
    transmat = np.full((K, K), 1.0 / K)
    np.fill_diagonal(transmat, 0.8)
    transmat /= transmat.sum(axis=1, keepdims=True)
    regimes = rng.integers(0, K, T)
    returns = pd.Series(X[:, 0])
    return X, posteriors, transmat, returns, regimes


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestAnalysisPipelineBasic:
    def test_returns_report_instance(self):
        X, p, A, r, reg = _make_inputs()
        ap = AnalysisPipeline()
        report = ap.run(X, p, A, r, reg)
        assert isinstance(report, AnalysisReport)

    def test_report_metadata(self):
        T, d, K = 120, 3, 2
        X, p, A, r, reg = _make_inputs(T, d, K)
        ap = AnalysisPipeline(AnalysisConfig(symbol="TEST"))
        report = ap.run(X, p, A, r, reg, feature_names=["a", "b", "c"])
        assert report.symbol == "TEST"
        assert report.n_obs == T
        assert report.n_states == K
        assert report.n_features == d
        assert report.feature_names == ["a", "b", "c"]

    def test_run_timestamp_is_string(self):
        X, p, A, r, reg = _make_inputs()
        report = AnalysisPipeline().run(X, p, A, r, reg)
        assert isinstance(report.run_timestamp, str)
        assert len(report.run_timestamp) > 0

    def test_default_feature_names(self):
        d = 4
        X, p, A, r, reg = _make_inputs(d=d)
        report = AnalysisPipeline().run(X, p, A, r, reg)
        assert report.feature_names == [f"feat_{i}" for i in range(d)]

    def test_all_modules_run_by_default(self):
        X, p, A, r, reg = _make_inputs()
        report = AnalysisPipeline().run(X, p, A, r, reg)
        # All fields should be populated (not None) with d=3 >= 2
        for field in [
            "execution",
            "factor_result",
            "cointegration",
            "portfolio",
            "signal_filtering",
            "transition",
            "tail_risk",
            "backtest",
            "correlation",
        ]:
            val = getattr(report, field)
            assert val is not None, f"{field} should not be None"

    def test_execution_has_expected_keys(self):
        X, p, A, r, reg = _make_inputs()
        report = AnalysisPipeline().run(X, p, A, r, reg)
        assert "slippage_by_regime" in report.execution
        assert "n_regimes" in report.execution

    def test_transition_has_expected_keys(self):
        X, p, A, r, reg = _make_inputs()
        report = AnalysisPipeline().run(X, p, A, r, reg)
        assert "expected_durations" in report.transition
        assert "5step_transmat" in report.transition
        assert "mean_transition_entropy" in report.transition

    def test_backtest_has_tearsheet(self):
        X, p, A, r, reg = _make_inputs()
        report = AnalysisPipeline().run(X, p, A, r, reg)
        assert "tearsheet" in report.backtest
        assert "attribution" in report.backtest
        ts = report.backtest["tearsheet"]
        for key in ["total_return", "sharpe", "max_drawdown"]:
            assert key in ts

    def test_correlation_has_stability(self):
        X, p, A, r, reg = _make_inputs()
        report = AnalysisPipeline().run(X, p, A, r, reg)
        assert "stability" in report.correlation
        assert isinstance(report.correlation["stability"], float)

    def test_tail_risk_has_regime_tails(self):
        X, p, A, r, reg = _make_inputs()
        report = AnalysisPipeline().run(X, p, A, r, reg)
        assert "regime_tails" in report.tail_risk
        assert isinstance(report.tail_risk["regime_tails"], list)


# ---------------------------------------------------------------------------
# Module-disable tests
# ---------------------------------------------------------------------------


class TestAnalysisPipelineDisableModules:
    def test_disable_execution(self):
        X, p, A, r, reg = _make_inputs()
        cfg = AnalysisConfig(run_execution=False)
        report = AnalysisPipeline(cfg).run(X, p, A, r, reg)
        assert report.execution is None

    def test_disable_factors(self):
        X, p, A, r, reg = _make_inputs()
        cfg = AnalysisConfig(run_factors=False)
        report = AnalysisPipeline(cfg).run(X, p, A, r, reg)
        assert report.factor_result is None

    def test_disable_tail_risk(self):
        X, p, A, r, reg = _make_inputs()
        cfg = AnalysisConfig(run_tail_risk=False)
        report = AnalysisPipeline(cfg).run(X, p, A, r, reg)
        assert report.tail_risk is None

    def test_disable_backtest(self):
        X, p, A, r, reg = _make_inputs()
        cfg = AnalysisConfig(run_backtest=False)
        report = AnalysisPipeline(cfg).run(X, p, A, r, reg)
        assert report.backtest is None

    def test_disable_correlation(self):
        X, p, A, r, reg = _make_inputs()
        cfg = AnalysisConfig(run_correlation=False)
        report = AnalysisPipeline(cfg).run(X, p, A, r, reg)
        assert report.correlation is None

    def test_cointegration_skipped_for_univariate(self):
        # d=1, so cointegration and portfolio should be auto-skipped
        T, d, K = 120, 1, 2
        X, p, A, r, reg = _make_inputs(T, d, K)
        report = AnalysisPipeline().run(X, p, A, r, reg)
        assert report.cointegration is None
        assert report.portfolio is None

    def test_disable_all(self):
        X, p, A, r, reg = _make_inputs()
        cfg = AnalysisConfig(
            run_execution=False,
            run_factors=False,
            run_cointegration=False,
            run_portfolio=False,
            run_signal_filtering=False,
            run_transition=False,
            run_tail_risk=False,
            run_backtest=False,
            run_correlation=False,
        )
        report = AnalysisPipeline(cfg).run(X, p, A, r, reg)
        for field in [
            "execution",
            "factor_result",
            "cointegration",
            "portfolio",
            "signal_filtering",
            "transition",
            "tail_risk",
            "backtest",
            "correlation",
        ]:
            assert getattr(report, field) is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestAnalysisPipelineEdgeCases:
    def test_single_regime(self):
        T, d, K = 100, 2, 1
        X, p, A, r, reg = _make_inputs(T, d, K)
        report = AnalysisPipeline().run(X, p, A, r, reg)
        assert isinstance(report, AnalysisReport)

    def test_many_regimes(self):
        T, d, K = 200, 3, 5
        X, p, A, r, reg = _make_inputs(T, d, K)
        report = AnalysisPipeline().run(X, p, A, r, reg)
        assert report.n_states == K

    def test_returns_as_numpy(self):
        X, p, A, r, reg = _make_inputs()
        report = AnalysisPipeline().run(X, p, A, r.values, reg)
        assert isinstance(report, AnalysisReport)

    def test_report_fields_are_dicts_not_dataclasses(self):
        X, p, A, r, reg = _make_inputs()
        report = AnalysisPipeline().run(X, p, A, r, reg)
        for field in [
            "execution",
            "factor_result",
            "signal_filtering",
            "transition",
            "tail_risk",
            "backtest",
            "correlation",
        ]:
            val = getattr(report, field)
            if val is not None:
                assert isinstance(val, dict), (
                    f"{field} should be a dict, got {type(val)}"
                )
