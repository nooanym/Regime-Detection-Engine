"""Tests for rde.models.spectral_init.

Covers:
- spectral_kmeans_init: shape contracts, probability constraints, fallback K>d.
- tensor_power_init: shape contracts, NaN-free output, T-too-small fallback.
- spectral_init_hmm_params: dispatch and ValueError on unknown method.
- apply_spectral_init_to_hmm: parameter injection, init_params="".
- Spectral structure: 2-component GMM mean recovery within 2σ.
"""

from __future__ import annotations

import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM

from rde.models.spectral_init import (
    SpectralInitResult,
    apply_spectral_init_to_hmm,
    spectral_init_hmm_params,
    spectral_kmeans_init,
    tensor_power_init,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _make_X(T: int = 500, d: int = 3, seed: int = 0) -> np.ndarray:
    """Return a random (T, d) array with zero mean, unit variance."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, d))


def _make_gmm_data(
    T: int = 2000,
    d: int = 2,
    means: list[np.ndarray] | None = None,
    std: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Generate samples from a K-component GMM with equal weights.

    Parameters
    ----------
    T : int
        Total number of observations.
    d : int
        Dimensionality.
    means : list of np.ndarray, optional
        One mean per component.  Defaults to two well-separated means.
    std : float
        Isotropic standard deviation of each component.
    seed : int

    Returns
    -------
    X : np.ndarray  shape (T, d)
    true_means : list of np.ndarray
    """
    rng = np.random.default_rng(seed)
    if means is None:
        means = [np.array([-3.0, 0.0][:d]), np.array([3.0, 0.0][:d])]
    K = len(means)
    labels = rng.integers(0, K, size=T)
    X = np.zeros((T, d))
    for k, mu in enumerate(means):
        mask = labels == k
        X[mask] = rng.normal(loc=mu, scale=std, size=(mask.sum(), d))
    return X, means


# ---------------------------------------------------------------------------
# 1. spectral_kmeans_init — shape and probability contracts
# ---------------------------------------------------------------------------


class TestSpectralKmeansInit:
    def test_returns_spectral_init_result(self) -> None:
        X = _make_X(300, 3)
        result = spectral_kmeans_init(X, K=2, seed=0)
        assert isinstance(result, SpectralInitResult)

    def test_method_tag(self) -> None:
        X = _make_X(300, 3)
        result = spectral_kmeans_init(X, K=2, seed=0)
        assert result.method == "spectral_kmeans"

    def test_means_shape_K2(self) -> None:
        T, d, K = 400, 3, 2
        result = spectral_kmeans_init(_make_X(T, d), K=K)
        assert result.means.shape == (K, d)

    def test_means_shape_K5(self) -> None:
        T, d, K = 600, 4, 5
        result = spectral_kmeans_init(_make_X(T, d), K=K)
        assert result.means.shape == (K, d)

    def test_covs_shape(self) -> None:
        T, d, K = 400, 3, 2
        result = spectral_kmeans_init(_make_X(T, d), K=K)
        assert result.covs.shape == (K, d, d)

    def test_startprob_sums_to_one(self) -> None:
        X = _make_X(300, 3)
        result = spectral_kmeans_init(X, K=3)
        assert result.startprob.shape == (3,)
        np.testing.assert_allclose(result.startprob.sum(), 1.0, atol=1e-10)

    def test_transmat_rows_sum_to_one(self) -> None:
        X = _make_X(300, 3)
        result = spectral_kmeans_init(X, K=3)
        assert result.transmat.shape == (3, 3)
        np.testing.assert_allclose(result.transmat.sum(axis=1), np.ones(3), atol=1e-10)

    def test_singular_values_length_K(self) -> None:
        T, d, K = 500, 4, 3
        result = spectral_kmeans_init(_make_X(T, d), K=K)
        assert result.singular_values.shape == (K,)

    def test_whitening_matrix_shape(self) -> None:
        T, d, K = 500, 4, 3
        result = spectral_kmeans_init(_make_X(T, d), K=K)
        assert result.whitening_matrix.shape == (d, K)

    def test_covs_are_positive_definite(self) -> None:
        """All per-state covariance matrices must be PSD (regularisation applied)."""
        T, d, K = 500, 3, 2
        result = spectral_kmeans_init(_make_X(T, d), K=K)
        for k in range(K):
            eigvals = np.linalg.eigvalsh(result.covs[k])
            assert np.all(eigvals > 0), f"Covariance for state {k} is not PD."

    def test_fallback_when_K_greater_than_d(self) -> None:
        """When K > d the function should fall back to plain k-means gracefully."""
        T, d, K = 400, 2, 5  # K=5 > d=2
        result = spectral_kmeans_init(_make_X(T, d), K=K)
        assert result.means.shape == (K, d)
        assert result.covs.shape == (K, d, d)
        assert result.method == "spectral_kmeans"

    def test_no_nan_in_means(self) -> None:
        X = _make_X(300, 3)
        result = spectral_kmeans_init(X, K=2)
        assert not np.any(np.isnan(result.means))

    def test_nan_input_raises(self) -> None:
        X = _make_X(300, 3)
        X[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            spectral_kmeans_init(X, K=2)

    def test_K_equals_d(self) -> None:
        """K == d is a boundary case that should work without falling back."""
        T, d, K = 400, 3, 3
        result = spectral_kmeans_init(_make_X(T, d), K=K)
        assert result.means.shape == (K, d)

    def test_transmat_high_self_transition(self) -> None:
        """Diagonal entries of the default transition matrix should dominate."""
        X = _make_X(300, 3)
        result = spectral_kmeans_init(X, K=3)
        diag = np.diag(result.transmat)
        off_diag_max = (result.transmat - np.diag(diag)).max()
        assert np.all(diag > off_diag_max), "Diagonal should dominate off-diagonal."


# ---------------------------------------------------------------------------
# 2. tensor_power_init — shape contracts and fallback
# ---------------------------------------------------------------------------


class TestTensorPowerInit:
    def test_returns_spectral_init_result(self) -> None:
        X = _make_X(2000, 3)
        result = tensor_power_init(X, K=2, seed=0)
        assert isinstance(result, SpectralInitResult)

    def test_means_shape(self) -> None:
        T, d, K = 2000, 3, 2
        result = tensor_power_init(_make_X(T, d), K=K, seed=0)
        assert result.means.shape == (K, d)

    def test_covs_shape(self) -> None:
        T, d, K = 2000, 3, 2
        result = tensor_power_init(_make_X(T, d), K=K, seed=0)
        assert result.covs.shape == (K, d, d)

    def test_startprob_sums_to_one(self) -> None:
        X = _make_X(2000, 3)
        result = tensor_power_init(X, K=2, seed=0)
        np.testing.assert_allclose(result.startprob.sum(), 1.0, atol=1e-10)

    def test_transmat_rows_sum_to_one(self) -> None:
        X = _make_X(2000, 3)
        result = tensor_power_init(X, K=2, seed=0)
        np.testing.assert_allclose(
            result.transmat.sum(axis=1), np.ones(2), atol=1e-10
        )

    def test_no_nan_in_output_means(self) -> None:
        X = _make_X(2000, 3)
        result = tensor_power_init(X, K=2, seed=0)
        assert not np.any(np.isnan(result.means))

    def test_fallback_T_too_small(self) -> None:
        """T < 10*d² triggers fallback to spectral_kmeans_init."""
        d, K = 5, 2
        T = 10 * d ** 2 - 1  # just below threshold: 249
        X = _make_X(T, d)
        result = tensor_power_init(X, K=K, seed=0)
        # We get a valid result regardless (via fallback).
        assert result.means.shape == (K, d)
        assert result.method == "spectral_kmeans"

    def test_fallback_K_greater_than_d(self) -> None:
        T, d, K = 2000, 2, 5
        X = _make_X(T, d)
        result = tensor_power_init(X, K=K, seed=0)
        assert result.means.shape == (K, d)

    def test_nan_input_raises(self) -> None:
        X = _make_X(2000, 3)
        X[10, 1] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            tensor_power_init(X, K=2, seed=0)


# ---------------------------------------------------------------------------
# 3. spectral_init_hmm_params — dispatch and error handling
# ---------------------------------------------------------------------------


class TestSpectralInitHmmParams:
    def test_dispatches_spectral_kmeans(self) -> None:
        X = _make_X(300, 3)
        result = spectral_init_hmm_params(X, K=2, method="spectral_kmeans", seed=0)
        assert result.method == "spectral_kmeans"

    def test_dispatches_tensor_power(self) -> None:
        X = _make_X(2000, 3)
        result = spectral_init_hmm_params(X, K=2, method="tensor_power", seed=0)
        # Method may be tensor_power or spectral_kmeans if fallback triggered.
        assert result.method in ("tensor_power", "spectral_kmeans")

    def test_raises_on_unknown_method(self) -> None:
        X = _make_X(300, 3)
        with pytest.raises(ValueError, match="Unknown spectral init method"):
            spectral_init_hmm_params(X, K=2, method="bogus_method")

    def test_default_method_is_spectral_kmeans(self) -> None:
        X = _make_X(300, 3)
        result = spectral_init_hmm_params(X, K=2, seed=0)
        assert result.method == "spectral_kmeans"

    def test_kwargs_forwarded_to_tensor_power(self) -> None:
        """n_restarts kwarg should be forwarded without raising."""
        X = _make_X(2000, 3)
        result = spectral_init_hmm_params(
            X, K=2, method="tensor_power", seed=0, n_restarts=3, n_iter=5
        )
        assert isinstance(result, SpectralInitResult)


# ---------------------------------------------------------------------------
# 4. apply_spectral_init_to_hmm
# ---------------------------------------------------------------------------


class TestApplySpectralInitToHmm:
    def _make_hmm(self, K: int, d: int, cov_type: str = "full") -> GaussianHMM:
        return GaussianHMM(
            n_components=K,
            covariance_type=cov_type,
            n_iter=1,
        )

    def test_means_set(self) -> None:
        K, d = 2, 3
        hmm = self._make_hmm(K, d)
        result = spectral_kmeans_init(_make_X(300, d), K=K)
        apply_spectral_init_to_hmm(hmm, result)
        np.testing.assert_array_equal(hmm.means_, result.means)

    def test_covars_set_full(self) -> None:
        K, d = 2, 3
        hmm = self._make_hmm(K, d, cov_type="full")
        result = spectral_kmeans_init(_make_X(300, d), K=K)
        apply_spectral_init_to_hmm(hmm, result)
        assert hmm.covars_.shape == (K, d, d)

    def test_covars_set_diag(self) -> None:
        K, d = 2, 3
        hmm = self._make_hmm(K, d, cov_type="diag")
        result = spectral_kmeans_init(_make_X(300, d), K=K)
        apply_spectral_init_to_hmm(hmm, result)
        # Use _covars_ (backing store) — the getter calls fill_covars which expands
        # all non-full types to (K, d, d) in hmmlearn 0.3.x.
        assert hmm._covars_.shape == (K, d)

    def test_covars_set_tied(self) -> None:
        K, d = 2, 3
        hmm = self._make_hmm(K, d, cov_type="tied")
        result = spectral_kmeans_init(_make_X(300, d), K=K)
        apply_spectral_init_to_hmm(hmm, result)
        # Use _covars_ (backing store) — getter expands tied to (K, d, d) via fill_covars.
        assert hmm._covars_.shape == (d, d)

    def test_covars_set_spherical(self) -> None:
        K, d = 2, 3
        hmm = self._make_hmm(K, d, cov_type="spherical")
        result = spectral_kmeans_init(_make_X(300, d), K=K)
        apply_spectral_init_to_hmm(hmm, result)
        # Use _covars_ (backing store) — getter expands spherical to (K, d, d) via fill_covars.
        assert hmm._covars_.shape == (K,)

    def test_init_params_empty(self) -> None:
        K, d = 2, 3
        hmm = self._make_hmm(K, d)
        result = spectral_kmeans_init(_make_X(300, d), K=K)
        apply_spectral_init_to_hmm(hmm, result)
        assert hmm.init_params == ""

    def test_startprob_sums_to_one(self) -> None:
        K, d = 3, 4
        hmm = self._make_hmm(K, d)
        result = spectral_kmeans_init(_make_X(500, d), K=K)
        apply_spectral_init_to_hmm(hmm, result)
        np.testing.assert_allclose(hmm.startprob_.sum(), 1.0, atol=1e-10)

    def test_transmat_rows_sum_to_one(self) -> None:
        K, d = 3, 4
        hmm = self._make_hmm(K, d)
        result = spectral_kmeans_init(_make_X(500, d), K=K)
        apply_spectral_init_to_hmm(hmm, result)
        np.testing.assert_allclose(
            hmm.transmat_.sum(axis=1), np.ones(K), atol=1e-10
        )

    def test_raises_on_component_mismatch(self) -> None:
        K, d = 2, 3
        hmm = self._make_hmm(3, d)  # mismatch: hmm has 3 components
        result = spectral_kmeans_init(_make_X(300, d), K=K)
        with pytest.raises(ValueError, match="n_components"):
            apply_spectral_init_to_hmm(hmm, result)

    def test_hmm_can_fit_after_init(self) -> None:
        """After spectral init, hmmlearn should be able to run Baum-Welch."""
        K, d = 2, 3
        T = 400
        X = _make_X(T, d)
        hmm = GaussianHMM(n_components=K, covariance_type="full", n_iter=3)
        result = spectral_kmeans_init(X, K=K, seed=0)
        apply_spectral_init_to_hmm(hmm, result)
        # Should not raise.
        hmm.fit(X)
        assert hmm.monitor_.converged is not None  # fit ran


# ---------------------------------------------------------------------------
# 5. Spectral structure: GMM mean recovery
# ---------------------------------------------------------------------------


class TestSpectralStructure:
    """For data from a 2-component GMM with well-separated means, spectral
    k-means should recover means that are within 2 standard deviations of
    the true component means."""

    def test_2component_gmm_mean_recovery(self) -> None:
        sep = 4.0   # separation between components
        std = 0.4   # component standard deviation
        d = 2
        K = 2
        true_means = [np.array([-sep / 2, 0.0]), np.array([sep / 2, 0.0])]
        X, _ = _make_gmm_data(T=3000, d=d, means=true_means, std=std, seed=7)

        result = spectral_kmeans_init(X, K=K, seed=7)

        # For each true mean, there should be a recovered mean within 2*std.
        threshold = 2.0 * std
        recovered = result.means  # (K, d)
        for true_mu in true_means:
            dists = np.linalg.norm(recovered - true_mu, axis=1)
            assert dists.min() < threshold, (
                f"No recovered mean within {threshold:.2f} of true mean {true_mu}. "
                f"Min distance: {dists.min():.4f}. Recovered: {recovered}"
            )

    def test_singular_values_are_positive(self) -> None:
        """Top-K singular values of the cross-covariance should be positive."""
        X, _ = _make_gmm_data(T=2000)
        result = spectral_kmeans_init(X, K=2)
        assert np.all(result.singular_values > 0)

    def test_whitening_matrix_orthonormal_columns(self) -> None:
        """Columns of W (= U_K from SVD) should be orthonormal."""
        X = _make_X(500, 4)
        result = spectral_kmeans_init(X, K=3)
        W = result.whitening_matrix  # (d, K)
        gram = W.T @ W              # (K, K)
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-10)
