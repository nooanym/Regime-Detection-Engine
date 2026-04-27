"""Tests for rde.inference.online.OnlineDecoder (Phase 8)."""

import numpy as np
import pytest

from rde.inference.online import OnlineDecoder
from rde.inference.smoothing import forward_backward_posteriors
from rde.models.hmm import train_hmm


def _make_fitted(seed=0, T=400, n_states=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, 2))
    return train_hmm(X, n_states, n_restarts=2, n_iter=50, seed_base=seed)


class TestOnlineDecoder:
    def test_init_posterior_sums_to_one(self):
        fitted = _make_fitted()
        dec = OnlineDecoder(fitted)
        assert dec.filtered_posterior.sum() == pytest.approx(1.0, abs=1e-10)
        assert dec.filtered_posterior.shape == (3,)

    def test_step_returns_probability_vector(self):
        fitted = _make_fitted()
        dec = OnlineDecoder(fitted)
        rng = np.random.default_rng(1)
        x = rng.standard_normal(2)
        p = dec.step(x)
        assert p.shape == (3,)
        assert p.sum() == pytest.approx(1.0, abs=1e-10)
        assert np.all(p >= 0)

    def test_step_updates_state(self):
        fitted = _make_fitted()
        dec = OnlineDecoder(fitted)
        init = dec.filtered_posterior.copy()
        rng = np.random.default_rng(2)
        dec.step(rng.standard_normal(2))
        assert not np.allclose(dec.filtered_posterior, init)

    def test_reset_restores_start_dist(self):
        fitted = _make_fitted()
        dec = OnlineDecoder(fitted)
        init = dec.filtered_posterior.copy()
        rng = np.random.default_rng(3)
        for _ in range(10):
            dec.step(rng.standard_normal(2))
        dec.reset()
        np.testing.assert_allclose(dec.filtered_posterior, init, atol=1e-12)

    def test_batch_filter_shape(self):
        fitted = _make_fitted(T=300)
        dec = OnlineDecoder(fitted)
        rng = np.random.default_rng(4)
        X = rng.standard_normal((200, 2))
        out = dec.batch_filter(X)
        assert out.shape == (200, 3)

    def test_batch_filter_rows_sum_to_one(self):
        fitted = _make_fitted()
        dec = OnlineDecoder(fitted)
        rng = np.random.default_rng(5)
        X = rng.standard_normal((100, 2))
        out = dec.batch_filter(X)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-10)

    def test_batch_filter_nonnegative(self):
        fitted = _make_fitted()
        dec = OnlineDecoder(fitted)
        rng = np.random.default_rng(6)
        X = rng.standard_normal((50, 2))
        out = dec.batch_filter(X)
        assert np.all(out >= 0)

    def test_batch_matches_sequential_steps(self):
        """batch_filter must equal calling step() for each row."""
        fitted = _make_fitted(T=300, n_states=2)
        rng = np.random.default_rng(7)
        X = rng.standard_normal((80, 2))

        dec = OnlineDecoder(fitted)
        batch_out = dec.batch_filter(X)

        dec.reset()
        step_out = np.vstack([dec.step(X[t]) for t in range(len(X))])

        np.testing.assert_allclose(batch_out, step_out, atol=1e-12)

    def test_batch_filter_resets_between_calls(self):
        """Two consecutive batch_filter calls should return identical results."""
        fitted = _make_fitted()
        dec = OnlineDecoder(fitted)
        rng = np.random.default_rng(8)
        X = rng.standard_normal((60, 2))
        out1 = dec.batch_filter(X)
        out2 = dec.batch_filter(X)
        np.testing.assert_allclose(out1, out2, atol=1e-12)

    def test_filtered_less_certain_than_smoothed(self):
        """Smoothed (forward-backward) posterior should be more peaked than filtered.

        Smoothed posteriors use future data so they have higher confidence on
        average: max(smoothed[t]) >= max(filtered[t]) on average.
        """
        fitted = _make_fitted(T=500, n_states=3)
        rng = np.random.default_rng(9)
        X_raw = rng.standard_normal((200, 2))

        dec = OnlineDecoder(fitted)
        filtered = dec.batch_filter(X_raw)

        X_scaled = fitted.scaler.transform(X_raw)
        smoothed = forward_backward_posteriors(fitted.hmm, X_scaled)

        # On average, the smoothed peak probability should be >= filtered peak
        avg_smoothed_peak = float(smoothed.max(axis=1).mean())
        avg_filtered_peak = float(filtered.max(axis=1).mean())
        assert avg_smoothed_peak >= avg_filtered_peak - 0.01   # allow 1% tolerance
