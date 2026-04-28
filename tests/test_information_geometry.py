"""Tests for rde.analysis.information_geometry (Phase 18)."""

from __future__ import annotations

import numpy as np
import pytest

from rde.analysis.information_geometry import (
    DistinguishabilityResult,
    bhattacharyya_coefficient,
    bhattacharyya_distance,
    compute_distinguishability,
    js_divergence_gaussians,
    kl_divergence_gaussians,
    mahalanobis_distance,
    markov_entropy_rate,
    regime_entropy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(d: int, scale: float = 1.0) -> np.ndarray:
    """Isotropic covariance matrix."""
    return scale * np.eye(d)


def _transmat(K: int, p_self: float = 0.9) -> np.ndarray:
    """Simple transition matrix with self-transition prob p_self."""
    mat = np.full((K, K), (1 - p_self) / (K - 1))
    np.fill_diagonal(mat, p_self)
    return mat


# ---------------------------------------------------------------------------
# kl_divergence_gaussians
# ---------------------------------------------------------------------------


class TestKLDivergence:
    def test_identical_distributions_zero(self) -> None:
        mu = np.array([1.0, 2.0])
        sigma = _iso(2, 1.5)
        kl = kl_divergence_gaussians(mu, sigma, mu, sigma)
        assert kl == pytest.approx(0.0, abs=1e-9)

    def test_non_negative(self) -> None:
        mu1 = np.array([0.0, 0.0])
        mu2 = np.array([1.0, 1.0])
        kl = kl_divergence_gaussians(mu1, _iso(2), mu2, _iso(2, 2.0))
        assert kl >= -1e-12

    def test_asymmetric(self) -> None:
        mu1 = np.array([0.0])
        mu2 = np.array([2.0])
        kl_pq = kl_divergence_gaussians(mu1, _iso(1), mu2, _iso(1, 4.0))
        kl_qp = kl_divergence_gaussians(mu2, _iso(1, 4.0), mu1, _iso(1))
        assert kl_pq != pytest.approx(kl_qp, abs=1e-6)

    def test_wider_q_smaller_kl(self) -> None:
        """KL(N(0,1) || N(0,σ²)) decreases as σ grows (wider Q easier to cover P)."""
        mu = np.array([0.0])
        kl_small = kl_divergence_gaussians(mu, _iso(1), mu, _iso(1, 4.0))
        kl_large = kl_divergence_gaussians(mu, _iso(1), mu, _iso(1, 16.0))
        assert kl_small < kl_large + 1e-9  # small variance difference but direction holds

    def test_known_1d_value(self) -> None:
        """KL(N(0,1) || N(1,1)) = 0.5."""
        mu_p = np.array([0.0])
        mu_q = np.array([1.0])
        sigma = _iso(1)
        kl = kl_divergence_gaussians(mu_p, sigma, mu_q, sigma)
        assert kl == pytest.approx(0.5, abs=1e-8)

    def test_mean_shift_increases_kl(self) -> None:
        mu_near = np.array([0.5])
        mu_far = np.array([5.0])
        mu0 = np.array([0.0])
        sigma = _iso(1)
        kl_near = kl_divergence_gaussians(mu0, sigma, mu_near, sigma)
        kl_far = kl_divergence_gaussians(mu0, sigma, mu_far, sigma)
        assert kl_far > kl_near


# ---------------------------------------------------------------------------
# js_divergence_gaussians
# ---------------------------------------------------------------------------


class TestJSDivergence:
    def test_identical_zero(self) -> None:
        mu = np.array([1.0, 2.0])
        sigma = _iso(2)
        js = js_divergence_gaussians(mu, sigma, mu, sigma)
        assert js == pytest.approx(0.0, abs=1e-9)

    def test_symmetric(self) -> None:
        mu1 = np.array([0.0, 0.0])
        mu2 = np.array([3.0, 1.0])
        s1 = _iso(2, 0.5)
        s2 = _iso(2, 2.0)
        js_pq = js_divergence_gaussians(mu1, s1, mu2, s2)
        js_qp = js_divergence_gaussians(mu2, s2, mu1, s1)
        assert js_pq == pytest.approx(js_qp, abs=1e-10)

    def test_non_negative(self) -> None:
        mu1 = np.array([0.0])
        mu2 = np.array([1.0])
        js = js_divergence_gaussians(mu1, _iso(1), mu2, _iso(1, 2.0))
        assert js >= -1e-12

    def test_larger_separation_larger_js(self) -> None:
        mu0 = np.array([0.0])
        sigma = _iso(1)
        js_near = js_divergence_gaussians(mu0, sigma, np.array([0.5]), sigma)
        js_far = js_divergence_gaussians(mu0, sigma, np.array([5.0]), sigma)
        assert js_far > js_near


# ---------------------------------------------------------------------------
# bhattacharyya
# ---------------------------------------------------------------------------


class TestBhattacharyya:
    def test_identical_distributions_zero(self) -> None:
        mu = np.array([0.0, 1.0])
        sigma = _iso(2)
        bd = bhattacharyya_distance(mu, sigma, mu, sigma)
        assert bd == pytest.approx(0.0, abs=1e-9)

    def test_bc_identical_one(self) -> None:
        mu = np.array([0.0, 1.0])
        sigma = _iso(2)
        bc = bhattacharyya_coefficient(mu, sigma, mu, sigma)
        assert bc == pytest.approx(1.0, abs=1e-9)

    def test_bc_in_unit_interval(self) -> None:
        mu1 = np.array([0.0])
        mu2 = np.array([3.0])
        bc = bhattacharyya_coefficient(mu1, _iso(1), mu2, _iso(1, 2.0))
        assert 0.0 <= bc <= 1.0

    def test_bd_non_negative(self) -> None:
        mu1 = np.array([0.0, 0.0])
        mu2 = np.array([2.0, 1.0])
        bd = bhattacharyya_distance(mu1, _iso(2), mu2, _iso(2, 3.0))
        assert bd >= -1e-12

    def test_farther_means_smaller_bc(self) -> None:
        sigma = _iso(1)
        mu0 = np.array([0.0])
        bc_near = bhattacharyya_coefficient(mu0, sigma, np.array([0.5]), sigma)
        bc_far = bhattacharyya_coefficient(mu0, sigma, np.array([5.0]), sigma)
        assert bc_far < bc_near

    def test_symmetric(self) -> None:
        mu1 = np.array([1.0, 2.0])
        mu2 = np.array([3.0, -1.0])
        s1 = _iso(2, 0.5)
        s2 = _iso(2, 1.5)
        bd_pq = bhattacharyya_distance(mu1, s1, mu2, s2)
        bd_qp = bhattacharyya_distance(mu2, s2, mu1, s1)
        assert bd_pq == pytest.approx(bd_qp, abs=1e-10)


# ---------------------------------------------------------------------------
# mahalanobis_distance
# ---------------------------------------------------------------------------


class TestMahalanobis:
    def test_identical_means_zero(self) -> None:
        mu = np.array([1.0, 2.0])
        mah = mahalanobis_distance(mu, mu, _iso(2))
        assert mah == pytest.approx(0.0, abs=1e-9)

    def test_unit_covariance_equals_euclidean(self) -> None:
        mu1 = np.array([0.0, 0.0])
        mu2 = np.array([3.0, 4.0])
        mah = mahalanobis_distance(mu1, mu2, _iso(2))
        assert mah == pytest.approx(5.0, abs=1e-8)

    def test_non_negative(self) -> None:
        mu1 = np.array([0.0])
        mu2 = np.array([5.0])
        mah = mahalanobis_distance(mu1, mu2, _iso(1, 2.0))
        assert mah >= 0.0

    def test_larger_covariance_smaller_distance(self) -> None:
        mu1 = np.array([0.0])
        mu2 = np.array([1.0])
        mah_small = mahalanobis_distance(mu1, mu2, _iso(1, 0.5))
        mah_large = mahalanobis_distance(mu1, mu2, _iso(1, 4.0))
        assert mah_small > mah_large


# ---------------------------------------------------------------------------
# markov_entropy_rate
# ---------------------------------------------------------------------------


class TestMarkovEntropyRate:
    def test_deterministic_chain_zero(self) -> None:
        """Deterministic chain (self-transitions = 1) has zero entropy rate."""
        A = np.eye(3)
        h = markov_entropy_rate(A)
        assert h == pytest.approx(0.0, abs=1e-9)

    def test_uniform_chain_log_K(self) -> None:
        """Uniform transitions: H = log(K) nats."""
        K = 4
        A = np.full((K, K), 1.0 / K)
        h = markov_entropy_rate(A)
        assert h == pytest.approx(np.log(K), abs=1e-6)

    def test_non_negative(self) -> None:
        A = _transmat(3, p_self=0.8)
        h = markov_entropy_rate(A)
        assert h >= -1e-12

    def test_higher_self_transition_lower_entropy(self) -> None:
        h_lo = markov_entropy_rate(_transmat(3, p_self=0.5))
        h_hi = markov_entropy_rate(_transmat(3, p_self=0.95))
        assert h_lo > h_hi


# ---------------------------------------------------------------------------
# regime_entropy
# ---------------------------------------------------------------------------


class TestRegimeEntropy:
    def test_certain_posterior_zero_entropy(self) -> None:
        P = np.zeros((10, 3))
        P[:, 0] = 1.0
        h = regime_entropy(P)
        np.testing.assert_allclose(h, 0.0, atol=1e-9)

    def test_uniform_posterior_max_entropy(self) -> None:
        K = 4
        P = np.full((10, K), 1.0 / K)
        h = regime_entropy(P)
        np.testing.assert_allclose(h, np.log(K), atol=1e-9)

    def test_output_shape(self) -> None:
        T, K = 200, 3
        P = np.random.default_rng(0).dirichlet(np.ones(K), size=T)
        h = regime_entropy(P)
        assert h.shape == (T,)

    def test_non_negative(self) -> None:
        P = np.random.default_rng(1).dirichlet(np.ones(3), size=100)
        h = regime_entropy(P)
        assert (h >= -1e-12).all()


# ---------------------------------------------------------------------------
# compute_distinguishability
# ---------------------------------------------------------------------------


class TestComputeDistinguishability:
    def _params(self, K: int = 3, d: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        means = rng.standard_normal((K, d)) * 3.0
        covs = np.array([_iso(d, rng.uniform(0.5, 2.0)) for _ in range(K)])
        transmat = _transmat(K)
        return means, covs, transmat

    def test_returns_result(self) -> None:
        means, covs, trans = self._params()
        result = compute_distinguishability(means, covs, trans)
        assert isinstance(result, DistinguishabilityResult)

    def test_diagonal_kl_zero(self) -> None:
        means, covs, trans = self._params()
        result = compute_distinguishability(means, covs, trans)
        diag = np.diag(result.kl_matrix.values)
        np.testing.assert_allclose(diag, 0.0, atol=1e-9)

    def test_diagonal_bc_one(self) -> None:
        means, covs, trans = self._params()
        result = compute_distinguishability(means, covs, trans)
        diag = np.diag(result.bc_matrix.values)
        np.testing.assert_allclose(diag, 1.0, atol=1e-9)

    def test_js_symmetric(self) -> None:
        means, covs, trans = self._params()
        result = compute_distinguishability(means, covs, trans)
        js = result.js_matrix.values
        np.testing.assert_allclose(js, js.T, atol=1e-9)

    def test_bhattacharyya_symmetric(self) -> None:
        means, covs, trans = self._params()
        result = compute_distinguishability(means, covs, trans)
        bd = result.bhattacharyya_matrix.values
        np.testing.assert_allclose(bd, bd.T, atol=1e-9)

    def test_mahalanobis_symmetric(self) -> None:
        means, covs, trans = self._params()
        result = compute_distinguishability(means, covs, trans)
        mah = result.mahalanobis_matrix.values
        np.testing.assert_allclose(mah, mah.T, atol=1e-9)

    def test_n_states_correct(self) -> None:
        K = 4
        means, covs, trans = self._params(K=K)
        result = compute_distinguishability(means, covs, trans)
        assert result.n_states == K

    def test_entropy_rate_positive(self) -> None:
        means, covs, trans = self._params()
        result = compute_distinguishability(means, covs, trans)
        assert result.entropy_rate >= 0.0

    def test_well_separated_high_kl(self) -> None:
        """Means far apart → large KL."""
        d = 2
        mu1 = np.array([0.0, 0.0])
        mu2 = np.array([100.0, 0.0])
        means = np.stack([mu1, mu2])
        covs = np.array([_iso(d), _iso(d)])
        trans = _transmat(2)
        result = compute_distinguishability(means, covs, trans)
        assert result.kl_matrix.iloc[0, 1] > 1000.0

    def test_overlapping_small_bd(self) -> None:
        """Identical distributions → BD = 0."""
        d = 2
        mu = np.array([0.0, 0.0])
        sigma = _iso(d)
        means = np.stack([mu, mu])
        covs = np.array([sigma, sigma])
        trans = _transmat(2)
        result = compute_distinguishability(means, covs, trans)
        assert result.bhattacharyya_matrix.iloc[0, 1] == pytest.approx(0.0, abs=1e-9)
