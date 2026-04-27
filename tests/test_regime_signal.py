"""Tests for rde.signals.regime_signal (Phase 8)."""

import numpy as np
import pandas as pd
import pytest

from rde.evaluation.regime_analytics import RegimeStats, compute_regime_stats
from rde.signals.regime_signal import (
    RegimeSignalConfig,
    RegimeSignalGenerator,
    _ema,
    score_states,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_stats(sharpes: list[float]) -> list[RegimeStats]:
    """Build minimal RegimeStats list from a list of Sharpe values."""
    return [
        RegimeStats(
            state=i,
            label=f"S{i}",
            count=100,
            weight=1.0 / len(sharpes),
            mean_return_ann=s * 0.2,
            vol_ann=0.2,
            sharpe_ann=s,
            skewness=0.0,
            excess_kurtosis=0.0,
            max_drawdown=0.05,
            var_95=-0.01,
            cvar_95=-0.02,
            win_rate=0.5,
        )
        for i, s in enumerate(sharpes)
    ]


def _uniform_posteriors(T: int, K: int) -> np.ndarray:
    return np.full((T, K), 1.0 / K)


# ── _ema ─────────────────────────────────────────────────────────────────────

class TestEma:
    def test_shape_preserved(self):
        x = np.random.default_rng(0).standard_normal(50)
        assert _ema(x, span=5).shape == (50,)

    def test_constant_series_unchanged(self):
        x = np.full(20, 3.7)
        np.testing.assert_allclose(_ema(x, span=5), x, atol=1e-12)

    def test_first_element_unchanged(self):
        x = np.random.default_rng(1).standard_normal(30)
        assert _ema(x, span=10)[0] == pytest.approx(x[0])

    def test_invalid_span_raises(self):
        with pytest.raises(ValueError):
            _ema(np.ones(10), span=0)

    def test_smooths_step_function(self):
        x = np.concatenate([np.zeros(50), np.ones(50)])
        smoothed = _ema(x, span=10)
        # After the step, smoothed should lag behind the raw signal
        assert smoothed[51] < 1.0
        assert smoothed[-1] > smoothed[51]


# ── score_states ─────────────────────────────────────────────────────────────

class TestScoreStates:
    def test_shape(self):
        stats = _make_stats([1.0, 2.0, -1.0])
        scores = score_states(stats, scoring="sharpe")
        assert scores.shape == (3,)

    def test_best_state_gets_plus_one(self):
        stats = _make_stats([0.5, 3.0, -0.5])
        scores = score_states(stats, scoring="sharpe")
        assert scores.max() == pytest.approx(1.0)

    def test_worst_state_gets_minus_one(self):
        stats = _make_stats([0.5, 3.0, -0.5])
        scores = score_states(stats, scoring="sharpe")
        assert scores.min() == pytest.approx(-1.0)

    def test_ordering_preserved(self):
        stats = _make_stats([-1.0, 0.0, 1.0])
        scores = score_states(stats, scoring="sharpe")
        assert scores[0] < scores[1] < scores[2]

    def test_single_state_scores_zero(self):
        stats = _make_stats([2.5])
        scores = score_states(stats, scoring="sharpe")
        assert scores[0] == pytest.approx(0.0)

    def test_two_states(self):
        stats = _make_stats([-1.0, 1.0])
        scores = score_states(stats, scoring="sharpe")
        np.testing.assert_allclose(scores, [-1.0, 1.0])

    def test_nan_sharpe_gets_neutral_score(self):
        stats = _make_stats([2.0, float("nan"), -1.0])
        stats[1] = RegimeStats(
            state=1, label="S1", count=1, weight=0.33,
            mean_return_ann=float("nan"), vol_ann=float("nan"),
            sharpe_ann=float("nan"), skewness=float("nan"),
            excess_kurtosis=float("nan"), max_drawdown=0.0,
            var_95=float("nan"), cvar_95=float("nan"), win_rate=float("nan"),
        )
        scores = score_states(stats, scoring="sharpe")
        assert np.all(np.isfinite(scores))

    def test_return_scoring(self):
        stats = _make_stats([1.0, 0.5, 2.0])
        s_sharpe = score_states(stats, scoring="sharpe")
        s_return = score_states(stats, scoring="return")
        # Both rank based on same direction (higher Sharpe ~ higher return here)
        np.testing.assert_allclose(s_sharpe, s_return)

    def test_invalid_scoring_raises(self):
        stats = _make_stats([1.0])
        with pytest.raises(ValueError, match="scoring"):
            score_states(stats, scoring="drawdown")

    def test_empty_stats(self):
        assert score_states([], scoring="sharpe").shape == (0,)


# ── RegimeSignalGenerator ─────────────────────────────────────────────────────

class TestRegimeSignalGenerator:
    def test_generate_shape(self):
        stats = _make_stats([1.0, -1.0, 0.5])
        gen = RegimeSignalGenerator(stats)
        posteriors = _uniform_posteriors(100, 3)
        sig = gen.generate(posteriors)
        assert sig.shape == (100,)

    def test_uniform_posterior_near_zero(self):
        """With equal probability across all states, signal ≈ mean of scores ≈ 0."""
        stats = _make_stats([-1.0, 0.0, 1.0])
        gen = RegimeSignalGenerator(stats)
        posteriors = _uniform_posteriors(50, 3)
        sig = gen.generate(posteriors)
        np.testing.assert_allclose(sig, 0.0, atol=1e-10)

    def test_best_state_certain_gives_plus_one(self):
        stats = _make_stats([-0.5, 0.0, 2.0])   # state 2 is best
        gen = RegimeSignalGenerator(stats)
        posteriors = np.zeros((10, 3))
        posteriors[:, 2] = 1.0                   # fully confident in best state
        sig = gen.generate(posteriors)
        np.testing.assert_allclose(sig, 1.0, atol=1e-10)

    def test_worst_state_certain_gives_minus_one(self):
        stats = _make_stats([-0.5, 0.0, 2.0])   # state 0 is worst
        gen = RegimeSignalGenerator(stats)
        posteriors = np.zeros((10, 3))
        posteriors[:, 0] = 1.0
        sig = gen.generate(posteriors)
        np.testing.assert_allclose(sig, -1.0, atol=1e-10)

    def test_signal_in_minus_one_plus_one(self):
        stats = _make_stats([3.0, -2.0, 0.1, 1.5])
        gen = RegimeSignalGenerator(stats)
        rng = np.random.default_rng(0)
        raw = rng.dirichlet(np.ones(4), size=200)
        sig = gen.generate(raw)
        assert np.all(sig >= -1.0)
        assert np.all(sig <= 1.0)

    def test_ema_smoothing_reduces_variance(self):
        stats = _make_stats([1.0, -1.0])
        rng = np.random.default_rng(42)
        posteriors = rng.dirichlet([1, 1], size=500)

        gen_raw = RegimeSignalGenerator(stats, config=RegimeSignalConfig(ema_span=None))
        gen_ema = RegimeSignalGenerator(stats, config=RegimeSignalConfig(ema_span=20))

        sig_raw = gen_raw.generate(posteriors)
        sig_ema = gen_ema.generate(posteriors)

        assert sig_raw.std() > sig_ema.std()

    def test_wrong_posterior_width_raises(self):
        stats = _make_stats([1.0, -1.0])
        gen = RegimeSignalGenerator(stats)
        with pytest.raises(ValueError):
            gen.generate(np.ones((10, 3)) / 3)    # 3 cols but only 2 states

    def test_generate_series_has_correct_index(self):
        stats = _make_stats([1.0, -1.0])
        gen = RegimeSignalGenerator(stats)
        idx = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")
        posteriors = _uniform_posteriors(50, 2)
        s = gen.generate_series(posteriors, idx)
        assert isinstance(s, pd.Series)
        assert s.name == "signal"
        assert list(s.index) == list(idx)

    def test_generate_dataframe_columns(self):
        stats = _make_stats([1.0, 0.0, -1.0])
        gen = RegimeSignalGenerator(stats)
        T, K = 30, 3
        posteriors = _uniform_posteriors(T, K)
        idx = pd.date_range("2024-01-01", periods=T, freq="h", tz="UTC")
        states = np.zeros(T, dtype=int)
        df = gen.generate_dataframe(posteriors, idx, viterbi_states=states,
                                    labels=["a", "b", "c"])
        assert "signal" in df.columns
        assert "regime_viterbi" in df.columns
        assert "regime_label" in df.columns
        for k in range(K):
            assert f"regime_proba_{k}" in df.columns

    def test_return_scoring_config(self):
        stats = _make_stats([3.0, -1.0])
        cfg = RegimeSignalConfig(scoring="return")
        gen = RegimeSignalGenerator(stats, config=cfg)
        posteriors = np.zeros((5, 2))
        posteriors[:, 0] = 1.0
        sig = gen.generate(posteriors)
        np.testing.assert_allclose(sig, 1.0, atol=1e-10)
