"""Tests for rde.evaluation.vol_forecasting (Phase 46)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.evaluation.vol_forecasting import (
    VolForecastResult,
    _ewma_vol_forecast,
    _hist_vol_forecast,
    _realised_vol,
    compare_vol_forecasters,
    run_vol_forecast_cv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(T: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=T, freq="B")
    ret = rng.standard_normal(T) * 0.01
    vol = pd.Series(ret).rolling(20).std().fillna(0.01).values
    smooth = pd.Series(ret).rolling(5).mean().fillna(0.0).values
    return pd.DataFrame(
        {"log_return": ret, "volatility_w20": vol, "smoothed_return_w5": smooth},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Unit tests for low-level helpers
# ---------------------------------------------------------------------------


class TestEWMAVolForecast:
    def test_returns_positive(self) -> None:
        r = np.random.default_rng(0).standard_normal(100) * 0.01
        v = _ewma_vol_forecast(r, 50, ann_factor=252)
        assert v > 0

    def test_annualised_scale(self) -> None:
        r = np.ones(100) * 0.01
        v = _ewma_vol_forecast(r, 50, ann_factor=252)
        assert 0.0 < v < 5.0  # sane annualised vol range

    def test_short_history(self) -> None:
        r = np.array([0.01, -0.01, 0.005])
        v = _ewma_vol_forecast(r, 2, ann_factor=252)
        assert np.isfinite(v) and v >= 0


class TestHistVolForecast:
    def test_returns_positive(self) -> None:
        r = np.random.default_rng(0).standard_normal(100) * 0.01
        v = _hist_vol_forecast(r, 50, ann_factor=252)
        assert v > 0

    def test_window_respected(self) -> None:
        # Constant return → zero vol
        r = np.ones(100) * 0.01
        v = _hist_vol_forecast(r, 50, window=10, ann_factor=252)
        assert v < 1e-6

    def test_short_history(self) -> None:
        r = np.array([0.01])
        v = _hist_vol_forecast(r, 1, ann_factor=252)
        assert np.isfinite(v) and v >= 0


class TestRealisedVol:
    def test_returns_value(self) -> None:
        r = np.random.default_rng(0).standard_normal(100) * 0.01
        rv = _realised_vol(r, 50, horizon=5, ann_factor=252)
        assert rv is not None and rv > 0

    def test_returns_none_at_end(self) -> None:
        r = np.ones(50) * 0.01
        rv = _realised_vol(r, 48, horizon=5, ann_factor=252)
        assert rv is None

    def test_annualised_scale(self) -> None:
        r = np.ones(100) * 0.01
        rv = _realised_vol(r, 0, horizon=21, ann_factor=252)
        assert rv is not None and rv < 1e-6  # constant series → zero vol


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRunVolForecastCV:
    def test_returns_dict_with_three_methods(self) -> None:
        df = _make_features()
        results = run_vol_forecast_cv(
            df,
            feature_cols=["log_return", "volatility_w20", "smoothed_return_w5"],
            returns_col="log_return",
            n_states=2,
            train_bars=100,
            test_bars=50,
            embargo_bars=5,
            horizons=[5, 10],
            ann_factor=252,
        )
        assert set(results.keys()) == {"hmm", "ewma", "hist_vol"}

    def test_each_method_has_results_per_horizon(self) -> None:
        df = _make_features()
        results = run_vol_forecast_cv(
            df,
            feature_cols=["log_return", "volatility_w20"],
            returns_col="log_return",
            n_states=2,
            train_bars=100,
            test_bars=50,
            embargo_bars=5,
            horizons=[5, 21],
            ann_factor=252,
        )
        for method in ("hmm", "ewma", "hist_vol"):
            assert len(results[method]) == 2  # two horizons

    def test_result_types(self) -> None:
        df = _make_features()
        results = run_vol_forecast_cv(
            df,
            feature_cols=["log_return", "volatility_w20"],
            returns_col="log_return",
            n_states=2,
            train_bars=100,
            test_bars=50,
            embargo_bars=5,
            horizons=[5],
            ann_factor=252,
        )
        r = results["hmm"][0]
        assert isinstance(r, VolForecastResult)
        assert r.method == "hmm"
        assert r.horizon_bars == 5
        assert np.isfinite(r.mse_mean)
        assert np.isfinite(r.mae_mean)
        assert np.isfinite(r.correlation)
        assert np.isfinite(r.ic)
        assert r.n_observations > 0

    def test_mse_positive(self) -> None:
        df = _make_features()
        results = run_vol_forecast_cv(
            df,
            feature_cols=["log_return", "volatility_w20"],
            returns_col="log_return",
            n_states=2,
            train_bars=100,
            test_bars=50,
            embargo_bars=5,
            horizons=[5],
            ann_factor=252,
        )
        for method in ("hmm", "ewma", "hist_vol"):
            assert results[method][0].mse_mean >= 0

    def test_correlation_bounded(self) -> None:
        df = _make_features()
        results = run_vol_forecast_cv(
            df,
            feature_cols=["log_return", "volatility_w20"],
            returns_col="log_return",
            n_states=2,
            train_bars=100,
            test_bars=50,
            embargo_bars=5,
            horizons=[10],
            ann_factor=252,
        )
        for method in ("hmm", "ewma", "hist_vol"):
            corr = results[method][0].correlation
            assert -1.0 <= corr <= 1.0 or np.isnan(corr)


class TestCompareVolForecasters:
    def test_returns_dataframe(self) -> None:
        df = _make_features()
        results = run_vol_forecast_cv(
            df,
            feature_cols=["log_return", "volatility_w20"],
            returns_col="log_return",
            n_states=2,
            train_bars=100,
            test_bars=50,
            embargo_bars=5,
            horizons=[5],
            ann_factor=252,
        )
        comparison = compare_vol_forecasters(results)
        assert isinstance(comparison, pd.DataFrame)
        assert set(comparison.columns) >= {"method", "horizon_bars", "mse", "mae", "correlation", "ic"}

    def test_all_methods_present(self) -> None:
        df = _make_features()
        results = run_vol_forecast_cv(
            df,
            feature_cols=["log_return", "volatility_w20"],
            returns_col="log_return",
            n_states=2,
            train_bars=100,
            test_bars=50,
            embargo_bars=5,
            horizons=[5],
            ann_factor=252,
        )
        comparison = compare_vol_forecasters(results)
        assert set(comparison["method"]) == {"hmm", "ewma", "hist_vol"}
