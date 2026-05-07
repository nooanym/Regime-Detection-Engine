"""Numerical-stability stress tests for HMM inference on long sequences.

Generates sequences of length 100k, 250k, and 500k from a known 3-state
Gaussian HMM and verifies that forward log-likelihood, Viterbi, and
forward-backward posteriors are all finite and well-behaved.

Run with::

    uv run pytest -m slow tests/numerics/
"""
from __future__ import annotations

import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Known 3-state HMM used for all stress tests
# ---------------------------------------------------------------------------

TRANSMAT = np.array([
    [0.95, 0.03, 0.02],
    [0.02, 0.96, 0.02],
    [0.03, 0.02, 0.95],
])
MEANS = np.array([[-1.5, 0.2], [0.0, 0.8], [1.5, 0.3]])
STDS = np.array([[0.4, 0.15], [0.6, 0.25], [0.5, 0.20]])
STARTPROB = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
SEED = 7


def _make_hmm() -> GaussianHMM:
    """Build a 3-state GaussianHMM with known parameters (no fitting)."""
    model = GaussianHMM(
        n_components=3,
        covariance_type="diag",
        n_iter=1,
        random_state=SEED,
    )
    model.startprob_ = STARTPROB.copy()
    model.transmat_ = TRANSMAT.copy()
    model.means_ = MEANS.copy()
    # hmmlearn diag expects covars_ shape (n_components, n_features)
    model.covars_ = STDS**2
    return model


def _generate_sequence(T: int, seed: int = SEED) -> np.ndarray:
    """Draw T observations from the known HMM."""
    rng = np.random.default_rng(seed)
    D = MEANS.shape[1]
    states = np.empty(T, dtype=int)
    states[0] = rng.choice(3, p=STARTPROB)
    for t in range(1, T):
        states[t] = rng.choice(3, p=TRANSMAT[states[t - 1]])

    X = np.empty((T, D))
    for t in range(T):
        k = states[t]
        X[t] = MEANS[k] + STDS[k] * rng.standard_normal(D)
    return X


def _scaled(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X)


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("T", [100_000, 250_000, 500_000])
class TestForwardLogLikelihoodStability:
    def test_finite(self, T):
        from rde.inference.forward import forward_log_likelihood

        X = _generate_sequence(T)
        model = _make_hmm()
        ll = forward_log_likelihood(model, _scaled(X))
        assert np.isfinite(ll), f"T={T}: log-likelihood is not finite: {ll}"
        assert ll < 0, f"T={T}: log-likelihood should be negative: {ll}"

    def test_scales_with_length(self, T):
        """LL/T should be approximately constant for an i.i.d. approximation."""
        from rde.inference.forward import forward_log_likelihood

        X = _generate_sequence(T)
        model = _make_hmm()
        ll = forward_log_likelihood(model, _scaled(X))
        ll_per_obs = ll / T
        # log-likelihood per observation should be in a reasonable range
        # (not clamped to 0 or exploding to ±inf)
        assert -20.0 < ll_per_obs < 0.0, (
            f"T={T}: LL/T={ll_per_obs:.4f} outside expected range (-20, 0)"
        )


@pytest.mark.slow
@pytest.mark.parametrize("T", [100_000, 250_000, 500_000])
class TestViterbiStability:
    def test_path_is_finite_integers(self, T):
        from rde.inference.viterbi import viterbi_decode

        X = _generate_sequence(T)
        model = _make_hmm()
        path = viterbi_decode(model, _scaled(X))
        assert path.shape == (T,), f"T={T}: path shape {path.shape}"
        assert np.all(path >= 0), f"T={T}: negative state indices"
        assert np.all(path < 3), f"T={T}: state index ≥ n_components"
        # No NaN in int array — just check dtype and values
        assert path.dtype in (np.int32, np.int64, int), f"T={T}: dtype {path.dtype}"


@pytest.mark.slow
@pytest.mark.parametrize("T", [100_000, 250_000, 500_000])
class TestPosteriorStability:
    def test_posteriors_sum_to_one(self, T):
        from rde.inference.smoothing import forward_backward_posteriors

        X = _generate_sequence(T)
        model = _make_hmm()
        posteriors = forward_backward_posteriors(model, _scaled(X))
        assert posteriors.shape == (T, 3), f"T={T}: shape {posteriors.shape}"
        assert np.all(np.isfinite(posteriors)), f"T={T}: non-finite posteriors"
        row_sums = posteriors.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, atol=1e-6,
            err_msg=f"T={T}: posterior rows do not sum to 1"
        )

    def test_posteriors_non_negative(self, T):
        from rde.inference.smoothing import forward_backward_posteriors

        X = _generate_sequence(T)
        model = _make_hmm()
        posteriors = forward_backward_posteriors(model, _scaled(X))
        assert np.all(posteriors >= -1e-9), (
            f"T={T}: negative posterior values (min={posteriors.min():.2e})"
        )
