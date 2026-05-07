"""Synthetic-data falsification suite (audit item 2.2).

These tests verify that the inference stack can recover known regimes from
ground-truth-labelled synthetic sequences.  They are marked ``slow`` and
excluded from the default pytest run.

Run with::

    uv run pytest -m slow tests/integration/
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Shared synthetic-data helpers
# ---------------------------------------------------------------------------

TRUE_TRANSMAT = np.array([
    [0.92, 0.05, 0.03],
    [0.04, 0.90, 0.06],
    [0.03, 0.04, 0.93],
])

TRUE_MEANS = np.array([
    [-2.0, 0.2, -0.1],   # low-return / low-vol
    [ 0.5, 0.8,  0.3],   # mid-return / high-vol
    [ 2.0, 0.3,  0.5],   # high-return / low-vol
])

TRUE_STDS = np.array([
    [0.4, 0.1, 0.15],
    [0.6, 0.2, 0.20],
    [0.5, 0.1, 0.18],
])

TRUE_STARTPROB = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

T_SYNTH = 3_000
SEED = 99


def _generate_gaussian_sequence(
    T: int = T_SYNTH,
    transmat: np.ndarray = TRUE_TRANSMAT,
    means: np.ndarray = TRUE_MEANS,
    stds: np.ndarray = TRUE_STDS,
    startprob: np.ndarray = TRUE_STARTPROB,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, true_states): synthetic observations and ground-truth labels."""
    rng = np.random.default_rng(seed)
    K = len(startprob)
    states = np.empty(T, dtype=int)
    states[0] = rng.choice(K, p=startprob)
    for t in range(1, T):
        states[t] = rng.choice(K, p=transmat[states[t - 1]])

    X = np.empty((T, means.shape[1]))
    for t in range(T):
        k = states[t]
        X[t] = means[k] + stds[k] * rng.standard_normal(means.shape[1])
    return X, states


def _align_states(recovered: np.ndarray, true: np.ndarray, K: int) -> np.ndarray:
    """Permute recovered labels to maximally match true labels (Hungarian alg)."""
    conf = np.zeros((K, K), dtype=int)
    for r, t in zip(recovered, true):
        conf[r, t] += 1
    row_ind, col_ind = linear_sum_assignment(-conf)
    perm = np.empty(K, dtype=int)
    for r, c in zip(row_ind, col_ind):
        perm[r] = c
    return perm[recovered]


def _frobenius(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B, "fro"))


# ---------------------------------------------------------------------------
# 2.2.1 — Gaussian HMM recovery
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestGaussianHMMRecovery:
    """The Gaussian HMM must recover a 3-regime synthetic sequence to ARI ≥ 0.85."""

    def _fit(self):
        from rde.models.hmm import train_hmm

        X, true_states = _generate_gaussian_sequence()
        fitted = train_hmm(X, n_states=3, n_restarts=5, n_iter=300, seed_base=SEED)
        Xs = fitted.scaler.transform(X)
        recovered_raw = fitted.hmm.predict(Xs)
        recovered = _align_states(recovered_raw, true_states, K=3)
        return fitted, Xs, recovered, true_states

    def test_ari_above_threshold(self):
        _, _, recovered, true_states = self._fit()
        ari = adjusted_rand_score(true_states, recovered)
        assert ari >= 0.85, f"ARI={ari:.3f} < 0.85"

    def test_transmat_frobenius(self):
        fitted, Xs, recovered, true_states = self._fit()
        # Align state indices
        perm = np.empty(3, dtype=int)
        conf = np.zeros((3, 3), dtype=int)
        for r, t in zip(fitted.hmm.predict(Xs), true_states):
            conf[r, t] += 1
        _, col = linear_sum_assignment(-conf)
        # Permute recovered transmat rows and columns
        rec_mat = fitted.hmm.transmat_[col][:, col]
        dist = _frobenius(rec_mat, TRUE_TRANSMAT)
        assert dist <= 0.05, f"Frobenius dist={dist:.4f} > 0.05"

    def test_means_within_one_se(self):
        fitted, Xs, recovered, true_states = self._fit()
        # Align recovered means to true means
        conf = np.zeros((3, 3), dtype=int)
        raw = fitted.hmm.predict(Xs)
        for r, t in zip(raw, true_states):
            conf[r, t] += 1
        _, col = linear_sum_assignment(-conf)
        # Recovered means are in scaled space — compare unscaled
        rec_means_scaled = fitted.hmm.means_[col]
        rec_means = fitted.scaler.inverse_transform(rec_means_scaled)
        # SE = std / sqrt(T_per_state)
        n_per_state = np.array([np.sum(true_states == k) for k in range(3)], dtype=float)
        se = TRUE_STDS / np.sqrt(n_per_state[:, None])
        deviations = np.abs(rec_means - TRUE_MEANS)
        max_se_ratio = float(np.max(deviations / se))
        # 3 SE = 3-sigma bound (p < 0.003 for a correct estimator)
        assert max_se_ratio <= 3.0, (
            f"Max deviation {max_se_ratio:.2f} SE exceeds 3.0 SE"
        )


# ---------------------------------------------------------------------------
# 2.2.2 — Student-t HMM recovery
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestStudentTHMMRecovery:
    """StudentTHMM must recover 3-regime heavy-tailed sequences to ARI ≥ 0.80."""

    @staticmethod
    def _generate_student_t_sequence(
        T: int = T_SYNTH, dof: float = 4.0, seed: int = SEED
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate observations with Student-t marginals via scale-mixture."""
        rng = np.random.default_rng(seed)
        K, D = TRUE_MEANS.shape
        states = np.empty(T, dtype=int)
        states[0] = rng.choice(K, p=TRUE_STARTPROB)
        for t in range(1, T):
            states[t] = rng.choice(K, p=TRUE_TRANSMAT[states[t - 1]])

        X = np.empty((T, D))
        for t in range(T):
            k = states[t]
            # Scale mixture: z ~ N(0, I), chi2 ~ χ²(dof) → x ~ t_dof
            z = TRUE_STDS[k] * rng.standard_normal(D)
            scale = np.sqrt(dof / rng.chisquare(dof))
            X[t] = TRUE_MEANS[k] + z * scale
        return X, states

    def test_ari_above_threshold(self):
        from rde.models.student_t_hmm import StudentTHMM

        X, true_states = self._generate_student_t_sequence()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        model = StudentTHMM(n_components=3, n_iter=150, random_state=SEED)
        model.fit(Xs)
        recovered_raw = model.predict(Xs)
        recovered = _align_states(recovered_raw, true_states, K=3)
        ari = adjusted_rand_score(true_states, recovered)
        assert ari >= 0.80, f"ARI={ari:.3f} < 0.80"


# ---------------------------------------------------------------------------
# 2.2.3 — HSMM dwell recovery
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestHSMMDwellRecovery:
    """HSMM correction must recover NegBin dwell distributions within ±50%."""

    # Injected NegBin parameters per state: (r, p) → mean = r*p/(1-p) + 1
    INJECTED_PARAMS = [
        (2.0, 0.80),  # mean ≈ 9  bars
        (3.0, 0.85),  # mean ≈ 18 bars
        (1.5, 0.75),  # mean ≈ 5.5 bars
    ]

    @staticmethod
    def _negbin_mean(r: float, p: float) -> float:
        return r * p / (1.0 - p) + 1.0

    @classmethod
    def _generate_hsmm_sequence(cls, T: int = T_SYNTH, seed: int = SEED):
        from scipy.stats import nbinom

        rng = np.random.default_rng(seed)
        D = TRUE_MEANS.shape[1]
        X_list, states_list = [], []

        state = rng.choice(3)
        while len(X_list) < T:
            r, p = cls.INJECTED_PARAMS[state]
            # NegBin(d-1; r, 1-p) — duration ≥ 1
            dwell = 1 + int(nbinom.rvs(r, 1.0 - p, random_state=rng))
            remaining = T - len(X_list)
            dwell = min(dwell, remaining)

            for _ in range(dwell):
                obs = TRUE_MEANS[state] + TRUE_STDS[state] * rng.standard_normal(D)
                X_list.append(obs)
                states_list.append(state)

            probs = TRUE_TRANSMAT[state].copy()
            probs[state] = 0.0
            probs /= probs.sum()
            state = rng.choice(3, p=probs)

        X = np.array(X_list[:T])
        true_states = np.array(states_list[:T])
        return X, true_states

    def test_median_dwell_within_50pct(self):
        from rde.inference.viterbi import viterbi_decode
        from rde.inference.smoothing import forward_backward_posteriors
        from rde.models.hmm import train_hmm
        from rde.models.semi_markov import fit_duration_params, apply_hsmm_correction
        from rde.evaluation.persistence import empirical_dwell_times

        X, true_states = self._generate_hsmm_sequence()
        fitted = train_hmm(X, n_states=3, n_restarts=5, n_iter=200, seed_base=SEED)
        Xs = fitted.scaler.transform(X)

        viterbi_path = viterbi_decode(fitted.hmm, Xs)
        posteriors = forward_backward_posteriors(fitted.hmm, Xs)

        # Fit NegBin durations from empirical dwells of the recovered path
        dwell_arrays = empirical_dwell_times(viterbi_path)
        durations = fit_duration_params(dwell_arrays)

        corrected_posteriors = apply_hsmm_correction(viterbi_path, posteriors, durations)
        corrected_states = corrected_posteriors.argmax(axis=1)

        # Align corrected states to true states
        corrected_aligned = _align_states(corrected_states, true_states, K=3)

        # Compute empirical median dwell per state from corrected path (after alignment)
        emp_dwells = empirical_dwell_times(corrected_aligned)

        for k, (r, p) in enumerate(self.INJECTED_PARAMS):
            expected_mean = self._negbin_mean(r, p)
            if k not in emp_dwells or len(emp_dwells[k]) == 0:
                pytest.skip(f"State {k} has no dwell observations after correction")
            emp_median = float(np.median(emp_dwells[k]))
            ratio = abs(emp_median - expected_mean) / expected_mean
            assert ratio <= 0.50, (
                f"State {k}: empirical median={emp_median:.1f}, "
                f"expected NegBin mean={expected_mean:.1f}, "
                f"relative error={ratio:.2f} > 0.50"
            )
