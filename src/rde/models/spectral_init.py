"""Spectral (Method of Moments) initialization for Hidden Markov Models.

Implements two strategies:

1. ``spectral_kmeans_init``: Projects data onto the top-K left singular vectors
   of the lag-1 cross-covariance matrix, then runs k-means to find cluster
   assignments. Lightweight and robust.

2. ``tensor_power_init``: Full Anandkumar et al. (2012/2014) observable-operator
   method using a third-order moment tensor and the robust tensor power method.
   Requires many observations; falls back to spectral k-means automatically.

Both return a :class:`SpectralInitResult` that can be applied to an unfitted
``hmmlearn.hmm.GaussianHMM`` via :func:`apply_spectral_init_to_hmm`.

References
----------
Anandkumar, A., Ge, R., Hsu, D., Kakade, S. M., & Telgarsky, M. (2014).
    Tensor decompositions for learning latent variable models.
    *Journal of Machine Learning Research*, 15(1), 2773–2832.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Public dataclass
# ------------------------------------------------------------------


@dataclass
class SpectralInitResult:
    """Output of :func:`spectral_init_hmm_params`.

    Attributes
    ----------
    means : np.ndarray
        Shape ``(K, d)``.  Initial state means in the original feature space.
    covs : np.ndarray
        Shape ``(K, d, d)``.  Initial state covariances (empirical within-cluster,
        regularised by ``1e-4 * I``).
    startprob : np.ndarray
        Shape ``(K,)``.  Uniform ``1/K`` initial state probabilities.
    transmat : np.ndarray
        Shape ``(K, K)``.  Initial transition matrix.  Rows sum to 1.
        Defaults to ``0.9 * I + 0.1/K * ones``, which favours self-transitions
        — a good prior for financial regimes.
    whitening_matrix : np.ndarray
        Shape ``(d, K)``.  The whitening matrix *W* used in the spectral step.
        Columns are the top-K left singular vectors of the cross-covariance,
        optionally scaled by ``S^{-1/2}``.
    singular_values : np.ndarray
        Shape ``(K,)``.  Top-K singular values of the cross-covariance matrix.
    method : str
        ``"spectral_kmeans"`` or ``"tensor_power"`` — which method was used.
    """

    means: np.ndarray
    covs: np.ndarray
    startprob: np.ndarray
    transmat: np.ndarray
    whitening_matrix: np.ndarray
    singular_values: np.ndarray
    method: str


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _default_transmat(K: int) -> np.ndarray:
    """Return a ``(K, K)`` transition matrix with high self-transition probability.

    Parameters
    ----------
    K : int
        Number of states.

    Returns
    -------
    np.ndarray
        ``0.9 * I + 0.1/K * ones``, normalised so rows sum to 1.
    """
    mat = np.full((K, K), 0.1 / K)
    np.fill_diagonal(mat, 0.9 + 0.1 / K)
    # Normalise rows to guard against floating-point drift.
    mat /= mat.sum(axis=1, keepdims=True)
    return mat


def _per_cluster_stats(
    X: np.ndarray,
    labels: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-cluster means and regularised full covariances.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(T, d)``.  Original-space observations.
    labels : np.ndarray
        Shape ``(T,)``.  Integer cluster assignments in ``[0, K)``.
    K : int
        Number of clusters.

    Returns
    -------
    means : np.ndarray
        Shape ``(K, d)``.
    covs : np.ndarray
        Shape ``(K, d, d)``.  Regularised by ``1e-4 * I``.
    """
    T, d = X.shape
    global_cov = np.cov(X, rowvar=False) if T > 1 else np.eye(d)
    reg = 1e-4 * np.eye(d)

    means = np.zeros((K, d))
    covs = np.zeros((K, d, d))

    for k in range(K):
        mask = labels == k
        if mask.sum() >= 2:
            means[k] = X[mask].mean(axis=0)
            covs[k] = np.cov(X[mask], rowvar=False) + reg
        elif mask.sum() == 1:
            means[k] = X[mask][0]
            covs[k] = global_cov + reg
        else:
            # Empty cluster — fall back to global mean/cov.
            logger.warning("Cluster %d is empty; using global statistics.", k)
            means[k] = X.mean(axis=0)
            covs[k] = global_cov + reg

    return means, covs


def _lag1_cross_covariance(X: np.ndarray) -> np.ndarray:
    """Compute the lag-1 cross-covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(T, d)``.

    Returns
    -------
    np.ndarray
        Shape ``(d, d)``.  ``C[i, j] = (1/(T-1)) Σ_t X[t+1, i] * X[t, j]``.
    """
    T = X.shape[0]
    # X[1:] shape (T-1, d), X[:-1] shape (T-1, d)
    # C = X[1:].T @ X[:-1] / (T - 1)
    return (X[1:].T @ X[:-1]) / (T - 1)


# ------------------------------------------------------------------
# spectral_kmeans_init
# ------------------------------------------------------------------


def spectral_kmeans_init(
    X: np.ndarray,
    K: int,
    *,
    seed: int = 42,
) -> SpectralInitResult:
    """Spectral k-means initialization for HMM.

    Projects data onto the top-K left singular vectors of the lag-1
    cross-covariance matrix, then runs k-means to find cluster assignments.
    State means are the original-space centroids of each cluster.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(T, d)``.  Already-standardised feature matrix (e.g., after
        ``StandardScaler``).  Must not contain ``NaN``.
    K : int
        Number of hidden states.
    seed : int
        Random seed for k-means.

    Returns
    -------
    SpectralInitResult
        With ``method="spectral_kmeans"``.

    Notes
    -----
    Cross-covariance: ``C[i,j] = (1/(T-1)) Σ_t X[t+1,i] * X[t,j]``.

    Uses ``scipy.linalg.svd`` with ``full_matrices=False`` for efficiency.

    If ``K > d``, falls back to standard k-means in the original space.
    The whitening matrix in that case is set to the identity (first ``K``
    columns, padded with zeros).
    """
    if np.any(np.isnan(X)):
        raise ValueError("X must not contain NaN values.")

    T, d = X.shape
    logger.info(
        "spectral_kmeans_init: T=%d, d=%d, K=%d, seed=%d", T, d, K, seed
    )

    fallback_whitening = np.eye(d, K)  # used when K > d
    fallback_sv = np.ones(min(K, d))

    if K > d:
        logger.warning(
            "K=%d > d=%d; falling back to standard k-means in original space.", K, d
        )
        km = KMeans(n_clusters=K, random_state=seed, n_init=10)
        km.fit(X)
        means, covs = _per_cluster_stats(X, km.labels_, K)
        return SpectralInitResult(
            means=means,
            covs=covs,
            startprob=np.full(K, 1.0 / K),
            transmat=_default_transmat(K),
            whitening_matrix=fallback_whitening,
            singular_values=fallback_sv,
            method="spectral_kmeans",
        )

    # Step 1: Compute lag-1 cross-covariance C of shape (d, d).
    C = _lag1_cross_covariance(X)

    # Step 2: SVD of C — keep top K left singular vectors.
    # scipy.linalg.svd returns U (d,d), s (min(d,d),), Vt (d,d) with
    # full_matrices=False giving U (d, min(d,d)).
    U, s, _ = svd(C, full_matrices=False)
    U_K = U[:, :K]          # (d, K)
    s_K = s[:K]             # (K,)

    # Whitening matrix: columns are the top-K left singular vectors.
    W = U_K  # shape (d, K)

    # Step 3: Project data.
    Z = X @ W  # (T, K) — project each observation into spectral subspace

    # Step 4: k-means in the spectral subspace.
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    km.fit(Z)
    labels = km.labels_  # (T,)

    # Step 5: Recover original-space statistics.
    # We use the original X observations assigned to each cluster.
    means, covs = _per_cluster_stats(X, labels, K)

    return SpectralInitResult(
        means=means,
        covs=covs,
        startprob=np.full(K, 1.0 / K),
        transmat=_default_transmat(K),
        whitening_matrix=W,
        singular_values=s_K,
        method="spectral_kmeans",
    )


# ------------------------------------------------------------------
# tensor_power_init
# ------------------------------------------------------------------


def _whiten_tensor_m3(M3: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply whitening matrix W to third-order moment tensor M3.

    Computes ``T3[i,j,k] = Σ_{a,b,c} W[a,i]*W[b,j]*W[c,k]*M3[a,b,c]``.

    Parameters
    ----------
    M3 : np.ndarray
        Shape ``(d, d, d)``.
    W : np.ndarray
        Shape ``(d, K)``.

    Returns
    -------
    np.ndarray
        Shape ``(K, K, K)``.
    """
    # Einsum contraction along all three modes simultaneously.
    return np.einsum("ai,bj,ck,abc->ijk", W, W, W, M3)


def _tensor_power_method(
    T3: np.ndarray,
    K: int,
    *,
    n_restarts: int = 10,
    n_iter: int = 20,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust tensor power method (Anandkumar et al., 2014, Algorithm 1).

    Performs a symmetric rank-K decomposition of *T3*.

    Parameters
    ----------
    T3 : np.ndarray
        Shape ``(K, K, K)``.  The whitened third-order moment tensor.
    K : int
        Number of components to extract.
    n_restarts : int
        Random restarts per component.
    n_iter : int
        Power-method iterations per restart.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    eigenvalues : np.ndarray
        Shape ``(K,)``.
    eigenvectors : np.ndarray
        Shape ``(K, K)``.  Row *k* is the *k*-th recovered eigenvector.
    """
    T3_work = T3.copy()
    eigenvalues = np.zeros(K)
    eigenvectors = np.zeros((K, K))

    for comp in range(K):
        best_lam = -np.inf
        best_vec = rng.standard_normal(K)
        best_vec /= np.linalg.norm(best_vec) + 1e-12

        for _ in range(n_restarts):
            theta = rng.standard_normal(K)
            theta /= np.linalg.norm(theta) + 1e-12

            for _ in range(n_iter):
                # Contract T3 with theta along the third mode:
                # T2[i,j] = Σ_k T3[i,j,k] * theta[k]
                T2 = np.einsum("ijk,k->ij", T3_work, theta)
                # Leading eigenvector of symmetric matrix T2.
                eigvals, eigvecs = np.linalg.eigh(T2)
                # Take the eigenvector with the largest |eigenvalue|.
                idx = np.argmax(np.abs(eigvals))
                theta = eigvecs[:, idx]
                theta /= np.linalg.norm(theta) + 1e-12

            # Estimate eigenvalue as λ = θ^T T2 θ.
            T2 = np.einsum("ijk,k->ij", T3_work, theta)
            lam = float(theta @ T2 @ theta)

            if lam > best_lam:
                best_lam = lam
                best_vec = theta.copy()

        eigenvalues[comp] = best_lam
        eigenvectors[comp] = best_vec

        # Deflate: T3 ← T3 − λ * v⊗v⊗v
        T3_work -= best_lam * np.einsum("i,j,k->ijk", best_vec, best_vec, best_vec)

    return eigenvalues, eigenvectors


def tensor_power_init(
    X: np.ndarray,
    K: int,
    *,
    n_restarts: int = 10,
    n_iter: int = 20,
    seed: int = 42,
) -> SpectralInitResult:
    """Tensor power method initialization for HMM (Anandkumar et al., 2014).

    Implements the observable-operator method: computes the first three
    empirical moment matrices, whitens using the SVD of the second moment,
    decomposes the whitened third-order moment tensor via the robust tensor
    power method, and recovers state emission means.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(T, d)``.  Already-standardised feature matrix.
    K : int
        Number of hidden states.
    n_restarts : int
        Number of random restarts for the tensor power method.
    n_iter : int
        Power-method iterations per restart.
    seed : int
        Base seed for random restarts.

    Returns
    -------
    SpectralInitResult
        With ``method="tensor_power"``.

    Notes
    -----
    Requires ``T >> d³`` for reliable moment estimation.  Falls back to
    :func:`spectral_kmeans_init` if ``K > d`` or ``T < 10 * d**2``.

    The rank of the cross-covariance matrix ``M2`` is checked before
    whitening.  If ``rank(M2) < K``, the method falls back to
    :func:`spectral_kmeans_init`.
    """
    if np.any(np.isnan(X)):
        raise ValueError("X must not contain NaN values.")

    T, d = X.shape
    logger.info(
        "tensor_power_init: T=%d, d=%d, K=%d, n_restarts=%d, seed=%d",
        T, d, K, n_restarts, seed,
    )

    # Fallback conditions.
    if K > d:
        logger.warning(
            "K=%d > d=%d; tensor power method not applicable. "
            "Falling back to spectral_kmeans_init.",
            K, d,
        )
        return spectral_kmeans_init(X, K, seed=seed)

    if T < 10 * d ** 2:
        logger.warning(
            "T=%d < 10*d²=%d; insufficient observations for reliable moment "
            "estimation.  Falling back to spectral_kmeans_init.",
            T, 10 * d ** 2,
        )
        return spectral_kmeans_init(X, K, seed=seed)

    # ------------------------------------------------------------------
    # Step 1: Empirical moment matrices.
    # ------------------------------------------------------------------
    # M2: lag-1 cross-covariance, shape (d, d).
    M2 = _lag1_cross_covariance(X)  # E[x_{t+1} ⊗ x_t]

    # M3: lag-2 triple cross-covariance, shape (d, d, d).
    # M3[a,b,c] = (1/(T-2)) Σ_t X[t,a]*X[t+1,b]*X[t+2,c]
    M3 = np.einsum("ta,tb,tc->abc", X[:-2], X[1:-1], X[2:]) / (T - 2)

    # ------------------------------------------------------------------
    # Step 2: Whiten using SVD of M2.
    # ------------------------------------------------------------------
    rank_M2 = int(np.linalg.matrix_rank(M2, tol=1e-6))
    if rank_M2 < K:
        logger.warning(
            "rank(M2)=%d < K=%d; cross-covariance is rank-deficient. "
            "Falling back to spectral_kmeans_init.",
            rank_M2, K,
        )
        return spectral_kmeans_init(X, K, seed=seed)

    U, s, _ = svd(M2, full_matrices=False)
    U_K = U[:, :K]      # (d, K)
    s_K = s[:K]         # (K,)

    # Whitening matrix: W[a,i] = U_K[a,i] * s_K[i]^{-1/2}
    s_K_sqrt_inv = 1.0 / np.sqrt(s_K)
    W = U_K * s_K_sqrt_inv[np.newaxis, :]  # (d, K)
    # Pseudo-inverse: W_pinv[i,a] = U_K[a,i] * s_K[i]^{1/2}  → shape (K, d)
    W_pinv = (U_K * np.sqrt(s_K)[np.newaxis, :]).T  # (K, d)

    # ------------------------------------------------------------------
    # Step 3: Whiten M3.
    # ------------------------------------------------------------------
    T3 = _whiten_tensor_m3(M3, W)  # (K, K, K)

    # ------------------------------------------------------------------
    # Step 4: Tensor power decomposition.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    eigenvalues, eigenvectors = _tensor_power_method(
        T3, K, n_restarts=n_restarts, n_iter=n_iter, rng=rng
    )

    # ------------------------------------------------------------------
    # Step 5: Recover state means in original space.
    # ------------------------------------------------------------------
    # μ_k ≈ W_pinv^T @ (v_k / λ_k^{1/2})
    # Clamp eigenvalues to avoid divide-by-zero.
    lam_safe = np.maximum(np.abs(eigenvalues), 1e-8)
    means = np.zeros((K, d))
    for k in range(K):
        v_k = eigenvectors[k]  # (K,)
        means[k] = W_pinv.T @ (v_k / np.sqrt(lam_safe[k]))

    # Guard against NaN in recovered means (degenerate decomposition).
    if np.any(np.isnan(means)):
        logger.warning(
            "NaN detected in tensor-power recovered means; "
            "falling back to spectral_kmeans_init."
        )
        return spectral_kmeans_init(X, K, seed=seed)

    # Assign each observation to the nearest recovered mean (in original space)
    # to compute empirical cluster statistics.
    dists = np.linalg.norm(
        X[:, np.newaxis, :] - means[np.newaxis, :, :], axis=2
    )  # (T, K)
    labels = dists.argmin(axis=1)  # (T,)
    means_emp, covs = _per_cluster_stats(X, labels, K)

    return SpectralInitResult(
        means=means_emp,
        covs=covs,
        startprob=np.full(K, 1.0 / K),
        transmat=_default_transmat(K),
        whitening_matrix=W,
        singular_values=s_K,
        method="tensor_power",
    )


# ------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------


def spectral_init_hmm_params(
    X: np.ndarray,
    K: int,
    *,
    method: str = "spectral_kmeans",
    seed: int = 42,
    **kwargs,
) -> SpectralInitResult:
    """Dispatch to :func:`spectral_kmeans_init` or :func:`tensor_power_init`.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(T, d)``.
    K : int
        Number of states.
    method : str
        ``"spectral_kmeans"`` (default) or ``"tensor_power"``.
    seed : int
        Random seed forwarded to the chosen method.
    **kwargs
        Additional keyword arguments forwarded to the chosen method.

    Returns
    -------
    SpectralInitResult

    Raises
    ------
    ValueError
        If *method* is not one of the supported values.
    """
    if method == "spectral_kmeans":
        return spectral_kmeans_init(X, K, seed=seed, **kwargs)
    elif method == "tensor_power":
        return tensor_power_init(X, K, seed=seed, **kwargs)
    else:
        raise ValueError(
            f"Unknown spectral init method: {method!r}. "
            "Expected 'spectral_kmeans' or 'tensor_power'."
        )


# ------------------------------------------------------------------
# Apply to hmmlearn estimator
# ------------------------------------------------------------------


def apply_spectral_init_to_hmm(
    hmm,  # hmmlearn GaussianHMM instance (unfitted)
    result: SpectralInitResult,
) -> None:
    """Apply :class:`SpectralInitResult` to an unfitted hmmlearn HMM in place.

    Sets ``hmm.startprob_``, ``hmm.transmat_``, ``hmm.means_``,
    ``hmm.covars_``, and sets ``hmm.init_params`` to ``""`` so
    Baum-Welch does not overwrite the initialisation.

    Parameters
    ----------
    hmm : GaussianHMM
        An unfitted ``hmmlearn`` estimator.  ``n_components`` must already
        be set and must equal ``result.means.shape[0]``.
    result : SpectralInitResult
        Output of :func:`spectral_init_hmm_params`.

    Raises
    ------
    ValueError
        If ``hmm.n_components`` does not match the number of states in
        *result*.

    Notes
    -----
    hmmlearn stores covariances differently per ``covariance_type``:

    * ``"full"``: ``covars_`` shape ``(K, d, d)``
    * ``"diag"``: ``covars_`` shape ``(K, d)``
    * ``"tied"``: ``covars_`` shape ``(d, d)``
    * ``"spherical"``: ``covars_`` shape ``(K,)``

    This function always provides full covariance matrices from the
    spectral result.  For non-full covariance types the stored value is
    adapted accordingly (diagonal extracted, mean of diagonal for spherical,
    etc.) to remain consistent with hmmlearn's expected layout.
    """
    K, d = result.means.shape
    if hmm.n_components != K:
        raise ValueError(
            f"hmm.n_components={hmm.n_components} does not match "
            f"result.means.shape[0]={K}."
        )

    hmm.startprob_ = result.startprob.copy()
    hmm.transmat_ = result.transmat.copy()
    hmm.means_ = result.means.copy()
    # hmmlearn 0.3.x stores n_features as an instance attribute set during
    # _check() / fit().  Set it manually so covars_ getter works before fitting.
    hmm.n_features = d

    cov_type = getattr(hmm, "covariance_type", "full")

    if cov_type == "full":
        hmm.covars_ = result.covs.copy()  # (K, d, d)
    elif cov_type == "diag":
        # Extract the diagonal of each covariance matrix.
        hmm.covars_ = np.array(
            [np.diag(result.covs[k]) for k in range(K)]
        )  # (K, d)
    elif cov_type == "tied":
        # Average the per-state covariances.
        hmm.covars_ = result.covs.mean(axis=0)  # (d, d)
    elif cov_type == "spherical":
        # Mean of diagonal entries per state.
        hmm.covars_ = np.array(
            [np.diag(result.covs[k]).mean() for k in range(K)]
        )  # (K,)
    else:
        # Unknown type — try setting as full and let hmmlearn raise.
        logger.warning(
            "Unknown covariance_type %r; setting covars_ as full (K,d,d).",
            cov_type,
        )
        hmm.covars_ = result.covs.copy()

    # Tell Baum-Welch not to re-initialise any parameters.
    hmm.init_params = ""
