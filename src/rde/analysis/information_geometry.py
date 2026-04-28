"""Regime-conditional information geometry (Phase 18).

Provides measures of how *statistically distinct* the HMM emission
distributions are from one another:

- **KL divergence** between Gaussian emissions (closed-form).
- **Jensen–Shannon divergence** (symmetrised KL).
- **Bhattacharyya distance** and overlap coefficient.
- **Mahalanobis distance** between state means.
- **Regime distinguishability matrix** — pairwise summary for all state pairs.
- **Entropy rate** of the Markov chain.

All metrics operate on the *fitted model parameters* (means + covariance
matrices from a GaussianHMM), so no raw data is required after training.

References
----------
Cover, T., & Thomas, J. (2006). *Elements of Information Theory* (2nd ed.).
    Wiley-Interscience.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Pairwise emission metrics
# ---------------------------------------------------------------------------


def kl_divergence_gaussians(
    mu_p: np.ndarray,
    sigma_p: np.ndarray,
    mu_q: np.ndarray,
    sigma_q: np.ndarray,
) -> float:
    """KL divergence KL(P || Q) for two multivariate Gaussians.

    Parameters
    ----------
    mu_p, mu_q : np.ndarray, shape (d,)
        Mean vectors.
    sigma_p, sigma_q : np.ndarray, shape (d, d)
        Covariance matrices.

    Returns
    -------
    float
        KL divergence KL(N(μ_P, Σ_P) || N(μ_Q, Σ_Q)).

    Notes
    -----
    KL(P||Q) = 0.5 * [tr(Σ_Q⁻¹ Σ_P) + (μ_Q - μ_P)ᵀ Σ_Q⁻¹ (μ_Q - μ_P)
                      - d + ln(det(Σ_Q)/det(Σ_P))]
    """
    d = len(mu_p)
    try:
        sigma_q_inv = np.linalg.inv(sigma_q)
        sign_p, logdet_p = np.linalg.slogdet(sigma_p)
        sign_q, logdet_q = np.linalg.slogdet(sigma_q)
        if sign_p <= 0 or sign_q <= 0:
            return np.nan
        trace_term = float(np.trace(sigma_q_inv @ sigma_p))
        diff = mu_q - mu_p
        quad = float(diff @ sigma_q_inv @ diff)
        log_det_ratio = float(logdet_q - logdet_p)
        return 0.5 * (trace_term + quad - d + log_det_ratio)
    except np.linalg.LinAlgError:
        return np.nan


def js_divergence_gaussians(
    mu_p: np.ndarray,
    sigma_p: np.ndarray,
    mu_q: np.ndarray,
    sigma_q: np.ndarray,
) -> float:
    """Jensen–Shannon divergence JS(P, Q) for two multivariate Gaussians.

    JS is symmetric and bounded in [0, ln(2)] (nats).  Uses the mixture
    approximation: JS(P,Q) ≈ 0.5*KL(P||M) + 0.5*KL(Q||M) where
    M = 0.5*(P+Q).  For Gaussians the mixture M is not Gaussian, so we use
    the upper bound:

        JS(P,Q) ≤ 0.5 * KL(P||Q) + 0.5 * KL(Q||P)   (symmetric KL / 2).

    This is the *variational JS* upper bound, identical to the symmetrised
    KL divided by 2.  It is non-negative and zero iff P = Q.
    """
    kl_pq = kl_divergence_gaussians(mu_p, sigma_p, mu_q, sigma_q)
    kl_qp = kl_divergence_gaussians(mu_q, sigma_q, mu_p, sigma_p)
    if np.isnan(kl_pq) or np.isnan(kl_qp):
        return np.nan
    return 0.5 * (kl_pq + kl_qp)


def bhattacharyya_distance(
    mu_p: np.ndarray,
    sigma_p: np.ndarray,
    mu_q: np.ndarray,
    sigma_q: np.ndarray,
) -> float:
    """Bhattacharyya distance D_B(P, Q) for multivariate Gaussians.

    D_B(P,Q) = 0.25 * (μ_P - μ_Q)ᵀ Σ̄⁻¹ (μ_P - μ_Q) + 0.5 * ln(det(Σ̄) / sqrt(det(Σ_P) det(Σ_Q)))

    where Σ̄ = (Σ_P + Σ_Q) / 2.  The Bhattacharyya coefficient is exp(-D_B).
    """
    sigma_bar = 0.5 * (sigma_p + sigma_q)
    try:
        sigma_bar_inv = np.linalg.inv(sigma_bar)
        sign_bar, logdet_bar = np.linalg.slogdet(sigma_bar)
        sign_p, logdet_p = np.linalg.slogdet(sigma_p)
        sign_q, logdet_q = np.linalg.slogdet(sigma_q)
        if sign_bar <= 0 or sign_p <= 0 or sign_q <= 0:
            return np.nan
        diff = mu_p - mu_q
        quad = float(diff @ sigma_bar_inv @ diff)
        log_term = float(logdet_bar - 0.5 * (logdet_p + logdet_q))
        return 0.25 * quad + 0.5 * log_term
    except np.linalg.LinAlgError:
        return np.nan


def bhattacharyya_coefficient(
    mu_p: np.ndarray,
    sigma_p: np.ndarray,
    mu_q: np.ndarray,
    sigma_q: np.ndarray,
) -> float:
    """Bhattacharyya coefficient BC(P, Q) = exp(-D_B(P, Q)).

    In [0, 1].  BC = 1 iff P = Q (fully overlapping distributions);
    BC = 0 iff distributions have disjoint support.
    """
    d_b = bhattacharyya_distance(mu_p, sigma_p, mu_q, sigma_q)
    if np.isnan(d_b):
        return np.nan
    return float(np.exp(-d_b))


def mahalanobis_distance(
    mu_p: np.ndarray,
    mu_q: np.ndarray,
    sigma_pooled: np.ndarray,
) -> float:
    """Mahalanobis distance between two mean vectors under a pooled covariance.

    Parameters
    ----------
    mu_p, mu_q : np.ndarray, shape (d,)
    sigma_pooled : np.ndarray, shape (d, d)
        Pooled covariance.  Often the average of sigma_p and sigma_q.

    Returns
    -------
    float
        sqrt((μ_P - μ_Q)ᵀ Σ_pooled⁻¹ (μ_P - μ_Q))
    """
    try:
        sigma_inv = np.linalg.inv(sigma_pooled)
        diff = mu_p - mu_q
        return float(np.sqrt(diff @ sigma_inv @ diff))
    except np.linalg.LinAlgError:
        return np.nan


# ---------------------------------------------------------------------------
# Markov chain information theory
# ---------------------------------------------------------------------------


def markov_entropy_rate(transmat: np.ndarray) -> float:
    """Shannon entropy rate of the Markov chain.

    H(X) = -Σ_i π_i Σ_j A_ij log A_ij

    where π is the stationary distribution and A is the transition matrix.
    Units: nats (using natural log).

    Parameters
    ----------
    transmat : np.ndarray, shape (K, K)

    Returns
    -------
    float
    """
    K = transmat.shape[0]
    # Stationary distribution via left eigenvector.
    eigvals, eigvecs = np.linalg.eig(transmat.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = np.real(eigvecs[:, idx])
    pi = np.abs(pi)
    pi /= pi.sum()

    h = 0.0
    for i in range(K):
        for j in range(K):
            a = transmat[i, j]
            if a > 1e-15:
                h -= float(pi[i]) * a * np.log(a)
    return h


def regime_entropy(posteriors: np.ndarray) -> np.ndarray:
    """Per-step entropy of the regime posterior.

    H_t = -Σ_k P(s_t=k | x_{1:t}) log P(s_t=k | x_{1:t})

    Parameters
    ----------
    posteriors : np.ndarray, shape (T, K)

    Returns
    -------
    np.ndarray, shape (T,)
    """
    eps = 1e-15
    p = np.clip(posteriors, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


# ---------------------------------------------------------------------------
# Summary dataclass
# ---------------------------------------------------------------------------


@dataclass
class DistinguishabilityResult:
    """Pairwise distinguishability metrics for all state pairs.

    Attributes
    ----------
    n_states : int
    kl_matrix : pd.DataFrame
        (K, K) asymmetric KL(i||j) matrix.
    js_matrix : pd.DataFrame
        (K, K) symmetric Jensen–Shannon divergence.
    bhattacharyya_matrix : pd.DataFrame
        (K, K) Bhattacharyya distances.
    bc_matrix : pd.DataFrame
        (K, K) Bhattacharyya coefficients.
    mahalanobis_matrix : pd.DataFrame
        (K, K) Mahalanobis distances between means (pooled covariance).
    entropy_rate : float
        Shannon entropy rate of the Markov chain.
    """

    n_states: int
    kl_matrix: pd.DataFrame
    js_matrix: pd.DataFrame
    bhattacharyya_matrix: pd.DataFrame
    bc_matrix: pd.DataFrame
    mahalanobis_matrix: pd.DataFrame
    entropy_rate: float


def compute_distinguishability(
    means: np.ndarray,
    covars: np.ndarray,
    transmat: np.ndarray,
) -> DistinguishabilityResult:
    """Compute pairwise distinguishability metrics from HMM parameters.

    Parameters
    ----------
    means : np.ndarray, shape (K, d)
        Per-state emission means.
    covars : np.ndarray, shape (K, d, d)
        Per-state full covariance matrices.
    transmat : np.ndarray, shape (K, K)
        Transition probability matrix.

    Returns
    -------
    DistinguishabilityResult
    """
    K = len(means)
    labels = list(range(K))

    kl = np.zeros((K, K))
    js = np.zeros((K, K))
    bd = np.zeros((K, K))
    bc = np.zeros((K, K))
    mah = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            if i == j:
                kl[i, j] = 0.0
                js[i, j] = 0.0
                bd[i, j] = 0.0
                bc[i, j] = 1.0
                mah[i, j] = 0.0
            else:
                kl[i, j] = kl_divergence_gaussians(
                    means[i], covars[i], means[j], covars[j]
                )
                js[i, j] = js_divergence_gaussians(
                    means[i], covars[i], means[j], covars[j]
                )
                bd[i, j] = bhattacharyya_distance(
                    means[i], covars[i], means[j], covars[j]
                )
                bc[i, j] = bhattacharyya_coefficient(
                    means[i], covars[i], means[j], covars[j]
                )
                pooled = 0.5 * (covars[i] + covars[j])
                mah[i, j] = mahalanobis_distance(means[i], means[j], pooled)

    h_rate = markov_entropy_rate(transmat)

    return DistinguishabilityResult(
        n_states=K,
        kl_matrix=pd.DataFrame(kl, index=labels, columns=labels),
        js_matrix=pd.DataFrame(js, index=labels, columns=labels),
        bhattacharyya_matrix=pd.DataFrame(bd, index=labels, columns=labels),
        bc_matrix=pd.DataFrame(bc, index=labels, columns=labels),
        mahalanobis_matrix=pd.DataFrame(mah, index=labels, columns=labels),
        entropy_rate=h_rate,
    )
