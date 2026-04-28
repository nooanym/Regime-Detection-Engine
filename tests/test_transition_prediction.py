"""Tests for rde.analysis.transition_prediction (Phase 27)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rde.analysis.transition_prediction import (
    RegimeTransitionPredictor,
    TransitionPredictorConfig,
    blend_hmm_logistic,
    expected_regime_duration,
    multi_step_transition,
    persistence_forecast,
    transition_entropy_trajectory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _transmat(K: int = 2, persistence: float = 0.9) -> np.ndarray:
    mat = np.full((K, K), (1.0 - persistence) / (K - 1))
    np.fill_diagonal(mat, persistence)
    return mat


def _uniform_post(T: int, K: int) -> np.ndarray:
    return np.full((T, K), 1.0 / K)


def _features_states(T: int = 200, d: int = 3, K: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, d)) * 0.01
    states = rng.integers(0, K, size=T)
    return X, states


# ---------------------------------------------------------------------------
# multi_step_transition
# ---------------------------------------------------------------------------


class TestMultiStepTransition:
    def test_zero_steps_is_identity(self) -> None:
        A = _transmat()
        np.testing.assert_array_equal(multi_step_transition(A, 0), np.eye(2))

    def test_one_step_is_transmat(self) -> None:
        A = _transmat()
        np.testing.assert_allclose(multi_step_transition(A, 1), A)

    def test_two_steps_is_a_squared(self) -> None:
        A = _transmat()
        np.testing.assert_allclose(multi_step_transition(A, 2), A @ A, atol=1e-12)

    def test_row_stochastic(self) -> None:
        A = _transmat(K=3)
        for h in [1, 5, 20, 100]:
            mat = multi_step_transition(A, h)
            np.testing.assert_allclose(mat.sum(axis=1), np.ones(3), atol=1e-10)

    def test_converges_to_stationary(self) -> None:
        A = _transmat(K=2, persistence=0.8)
        # Long-run matrix: each row should equal stationary distribution
        mat = multi_step_transition(A, 1000)
        np.testing.assert_allclose(mat[0], mat[1], atol=1e-6)

    def test_identity_transmat(self) -> None:
        A = np.eye(3)
        for h in [1, 10, 100]:
            np.testing.assert_allclose(multi_step_transition(A, h), np.eye(3))


# ---------------------------------------------------------------------------
# persistence_forecast
# ---------------------------------------------------------------------------


class TestPersistenceForecast:
    def test_returns_array(self) -> None:
        A = _transmat()
        result = persistence_forecast(0, A, 10)
        assert isinstance(result, np.ndarray)

    def test_length(self) -> None:
        A = _transmat()
        result = persistence_forecast(0, A, 15)
        assert len(result) == 15

    def test_decreasing_for_transient_regime(self) -> None:
        A = _transmat(persistence=0.7)
        result = persistence_forecast(0, A, 20)
        # Persistence probability should be non-increasing
        assert np.all(np.diff(result) <= 1e-10)

    def test_unit_persistence_stays_at_one(self) -> None:
        A = np.eye(2)
        result = persistence_forecast(0, A, 10)
        np.testing.assert_allclose(result, np.ones(10))

    def test_values_in_range(self) -> None:
        A = _transmat()
        result = persistence_forecast(0, A, 20)
        assert np.all(result >= 0)
        assert np.all(result <= 1.0 + 1e-10)

    def test_first_step_equals_transmat_diagonal(self) -> None:
        A = _transmat(persistence=0.85)
        result = persistence_forecast(0, A, 5)
        assert result[0] == pytest.approx(A[0, 0])


# ---------------------------------------------------------------------------
# expected_regime_duration
# ---------------------------------------------------------------------------


class TestExpectedRegimeDuration:
    def test_returns_float(self) -> None:
        A = _transmat()
        d = expected_regime_duration(0, A)
        assert isinstance(d, float)

    def test_positive(self) -> None:
        A = _transmat()
        d = expected_regime_duration(0, A)
        assert d > 0

    def test_high_persistence_longer_duration(self) -> None:
        A_lo = _transmat(persistence=0.6)
        A_hi = _transmat(persistence=0.95)
        d_lo = expected_regime_duration(0, A_lo)
        d_hi = expected_regime_duration(0, A_hi)
        assert d_hi > d_lo

    def test_identity_transmat_large_duration(self) -> None:
        A = np.eye(2)
        d = expected_regime_duration(0, A, max_horizon=100)
        assert d >= 99  # absorbing state: sums to max_horizon

    def test_approx_matches_analytical(self) -> None:
        # Geometric first-exit: E[T_exit - 1] = p / (1 - p)
        p = 0.8
        A = _transmat(persistence=p)
        d = expected_regime_duration(0, A)
        assert d == pytest.approx(p / (1.0 - p), rel=1e-6)


# ---------------------------------------------------------------------------
# RegimeTransitionPredictor
# ---------------------------------------------------------------------------


class TestRegimeTransitionPredictor:
    def test_fit_returns_self(self) -> None:
        X, y = _features_states()
        pred = RegimeTransitionPredictor()
        result = pred.fit(X, y)
        assert result is pred

    def test_predict_proba_returns_ndarray(self) -> None:
        X, y = _features_states()
        pred = RegimeTransitionPredictor().fit(X, y)
        probs = pred.predict_proba(X)
        assert isinstance(probs, np.ndarray)

    def test_predict_proba_shape(self) -> None:
        T, d, K = 200, 3, 2
        X, y = _features_states(T, d, K)
        pred = RegimeTransitionPredictor().fit(X, y)
        probs = pred.predict_proba(X)
        assert probs.shape[1] == K

    def test_predict_proba_sums_to_one(self) -> None:
        X, y = _features_states()
        pred = RegimeTransitionPredictor().fit(X, y)
        probs = pred.predict_proba(X)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(probs)), atol=1e-8)

    def test_predict_proba_non_negative(self) -> None:
        X, y = _features_states()
        pred = RegimeTransitionPredictor().fit(X, y)
        probs = pred.predict_proba(X)
        assert np.all(probs >= 0)

    def test_predictor_finds_signal(self) -> None:
        """Predictor should do better than uniform on a clearly separable dataset."""
        T, d, K = 500, 2, 2
        rng = np.random.default_rng(42)
        X = rng.standard_normal((T, d))
        # State at t+1 depends on feature 0 at time t (lagged 1 → predictable)
        y = np.zeros(T, dtype=int)
        y[1:] = (X[:-1, 0] > 0).astype(int)
        y[0] = 0
        cfg = TransitionPredictorConfig(lags=1, n_iter=500, l2_reg=0.001,
                                        learning_rate=0.2)
        pred = RegimeTransitionPredictor(cfg).fit(X, y)
        probs = pred.predict_proba(X)
        # Accuracy on training set should be above 60 %
        preds = probs.argmax(axis=1)
        # predict_proba with lags=1 on T rows returns T-lags = T-1 rows;
        # each prediction is for bar t+1 given bar t.
        n = len(preds)
        targets = y[1: 1 + n]
        n_min = min(len(preds), len(targets))
        acc = float((preds[:n_min] == targets[:n_min]).mean())
        assert acc > 0.6

    def test_accepts_dataframe(self) -> None:
        X, y = _features_states()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        pred = RegimeTransitionPredictor().fit(df, y)
        probs = pred.predict_proba(df)
        assert probs.shape[1] == 2

    def test_multiple_lags(self) -> None:
        X, y = _features_states()
        cfg = TransitionPredictorConfig(lags=3)
        pred = RegimeTransitionPredictor(cfg).fit(X, y)
        probs = pred.predict_proba(X)
        assert probs.shape[1] == 2

    def test_three_regime_prediction(self) -> None:
        X, y = _features_states(K=3)
        pred = RegimeTransitionPredictor().fit(X, y)
        probs = pred.predict_proba(X)
        assert probs.shape[1] == 3
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(probs)), atol=1e-8)


# ---------------------------------------------------------------------------
# blend_hmm_logistic
# ---------------------------------------------------------------------------


class TestBlendHMMLogistic:
    def test_returns_ndarray(self) -> None:
        T, K = 50, 2
        hmm = _uniform_post(T, K)
        log_ = _uniform_post(T, K)
        result = blend_hmm_logistic(hmm, log_)
        assert isinstance(result, np.ndarray)

    def test_sums_to_one(self) -> None:
        T, K = 50, 2
        hmm = _uniform_post(T, K)
        log_ = _uniform_post(T, K)
        result = blend_hmm_logistic(hmm, log_)
        np.testing.assert_allclose(result.sum(axis=1), np.ones(T), atol=1e-10)

    def test_alpha_one_equals_hmm(self) -> None:
        T, K = 30, 2
        rng = np.random.default_rng(0)
        hmm = np.abs(rng.standard_normal((T, K)))
        hmm /= hmm.sum(axis=1, keepdims=True)
        log_ = _uniform_post(T, K)
        result = blend_hmm_logistic(hmm, log_, alpha=1.0)
        np.testing.assert_allclose(result, hmm, atol=1e-10)

    def test_alpha_zero_equals_logistic(self) -> None:
        T, K = 30, 2
        rng = np.random.default_rng(0)
        log_ = np.abs(rng.standard_normal((T, K)))
        log_ /= log_.sum(axis=1, keepdims=True)
        hmm = _uniform_post(T, K)
        result = blend_hmm_logistic(hmm, log_, alpha=0.0)
        np.testing.assert_allclose(result, log_, atol=1e-10)

    def test_non_negative(self) -> None:
        T, K = 40, 3
        hmm = _uniform_post(T, K)
        log_ = _uniform_post(T, K)
        result = blend_hmm_logistic(hmm, log_, alpha=0.7)
        assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# transition_entropy_trajectory
# ---------------------------------------------------------------------------


class TestTransitionEntropyTrajectory:
    def test_returns_array(self) -> None:
        T, K = 50, 2
        post = _uniform_post(T, K)
        A = _transmat()
        H = transition_entropy_trajectory(post, A)
        assert isinstance(H, np.ndarray)

    def test_length(self) -> None:
        T, K = 50, 2
        post = _uniform_post(T, K)
        A = _transmat()
        H = transition_entropy_trajectory(post, A)
        assert len(H) == T

    def test_non_negative(self) -> None:
        T, K = 50, 2
        post = _uniform_post(T, K)
        A = _transmat()
        H = transition_entropy_trajectory(post, A)
        assert np.all(H >= -1e-10)

    def test_identity_transmat_zero_entropy(self) -> None:
        # With one-hot posterior: q = e_k @ I = e_k → H = 0
        T, K = 20, 2
        post = np.zeros((T, K))
        post[:, 0] = 1.0  # always in regime 0
        A = np.eye(K)
        H = transition_entropy_trajectory(post, A)
        np.testing.assert_allclose(H, 0.0, atol=1e-8)

    def test_uniform_transmat_maximum_entropy(self) -> None:
        T, K = 20, 3
        post = _uniform_post(T, K)
        A = np.full((K, K), 1.0 / K)
        H = transition_entropy_trajectory(post, A)
        max_H = float(np.log(K))
        np.testing.assert_allclose(H, max_H, atol=1e-8)

    def test_high_persistence_lower_entropy(self) -> None:
        # One-hot posterior: high persistence means q close to one-hot → low entropy
        T, K = 50, 2
        post = np.zeros((T, K))
        post[:, 0] = 1.0  # regime 0
        H_lo = transition_entropy_trajectory(post, _transmat(persistence=0.99))
        H_hi = transition_entropy_trajectory(post, _transmat(persistence=0.55))
        assert H_lo.mean() < H_hi.mean()
