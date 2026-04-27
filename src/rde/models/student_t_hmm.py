"""Multivariate Student-t HMM via ECM algorithm.

Replaces Gaussian emissions with multivariate Student-t emissions, allowing
each hidden state to have its own tail-heaviness parameter ν_k. This corrects
the Gaussian misspecification for financial returns, which have excess kurtosis
of 5-10 on hourly data.

References
----------
Liu, C. & Rubin, D. B. (1995). ML estimation of the t distribution using EM
and its extensions, ECM and ECME. Statistica Sinica, 5, 19-39.
"""

import logging
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import brentq
from scipy.special import digamma, gammaln

logger = logging.getLogger(__name__)

_DOF_LO = 2.001   # lower bound — keeps variance finite
_DOF_HI = 200.0   # upper bound — effectively Gaussian above ~100
_DOF_INIT = 5.0   # mildly fat-tailed starting point


def _mvt_log_pdf(
    X: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    dof: float,
) -> np.ndarray:
    """Multivariate Student-t log-density for a batch of observations.

    Parameters
    ----------
    X : np.ndarray
        Shape (T, d) — observations.
    mean : np.ndarray
        Shape (d,) — state mean.
    cov : np.ndarray
        Shape (d, d) — state covariance (positive-definite).
    dof : float
        Degrees of freedom ν > 2.

    Returns
    -------
    np.ndarray
        Shape (T,) — per-observation log-density values.

    Notes
    -----
    Uses Cholesky decomposition for numerically stable log-det and Mahalanobis
    distance. Falls back to adding a jitter of 1e-6 * I if Cholesky fails.
    Uses np.log1p(maha / dof) for the tail term to avoid cancellation near 0.
    """
    d = mean.shape[0]
    diff = X - mean  # (T, d)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov + 1e-6 * np.eye(d))
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    y = np.linalg.solve(L, diff.T)   # (d, T)
    maha = np.sum(y ** 2, axis=0)    # (T,)
    log_pdf = (
        gammaln((dof + d) / 2.0)
        - gammaln(dof / 2.0)
        - (d / 2.0) * np.log(dof * np.pi)
        - 0.5 * log_det
        - ((dof + d) / 2.0) * np.log1p(maha / dof)
    )
    return log_pdf


def _update_dof(nu_current: float, c_k: float) -> float:
    """Fixed-point update for degrees of freedom via Liu & Rubin (1995).

    Solves log(ν/2) - ψ(ν/2) + 1 + c_k = 0 using Brent's method,
    where c_k = (1/N_k) Σ_t γ_t(k) [log(u_t(k)) - u_t(k)].

    Parameters
    ----------
    nu_current : float
        Current estimate of ν_k; used as fallback if root-finding fails.
    c_k : float
        Normalised ECM auxiliary constant for this state.

    Returns
    -------
    float
        Updated degrees of freedom, clamped to (_DOF_LO, _DOF_HI).
    """
    def objective(nu: float) -> float:
        return np.log(nu / 2.0) - digamma(nu / 2.0) + 1.0 + c_k

    try:
        f_lo = objective(_DOF_LO)
        f_hi = objective(_DOF_HI)
        if f_lo * f_hi > 0:
            # Root not bracketed — clamp current estimate
            return float(np.clip(nu_current, _DOF_LO, _DOF_HI))
        nu_new = float(brentq(objective, _DOF_LO, _DOF_HI, xtol=1e-4, maxiter=100))
        return float(np.clip(nu_new, _DOF_LO, _DOF_HI))
    except (ValueError, RuntimeError):
        return float(np.clip(nu_current, _DOF_LO, _DOF_HI))


class StudentTHMM(GaussianHMM):
    """Hidden Markov Model with multivariate Student-t emissions.

    Subclasses GaussianHMM and overrides the emission-related methods to
    implement the ECM (Expectation Conditional Maximisation) algorithm with
    Student-t emissions. The transition matrix M-step is inherited unchanged.

    Parameters
    ----------
    n_components : int
        Number of hidden states.
    n_iter : int
        Maximum ECM iterations.
    random_state : int | None
        Random seed for reproducibility.

    Attributes
    ----------
    dofs_ : np.ndarray
        Shape (n_components,). Estimated degrees of freedom per state after
        fitting. Values in (_DOF_LO, _DOF_HI).

    Notes
    -----
    Only ``covariance_type="full"`` is supported because Student-t is defined
    with a full precision matrix. Passing any other covariance_type will be
    silently overridden to "full".

    The auxiliary weights u_t(k) = (ν_k + d) / (ν_k + δ_t(k)) upweight
    central observations and downweight outliers relative to Gaussian EM.
    This shrinks the influence of crash/spike bars on state parameter estimates.
    """

    def __init__(
        self,
        n_components: int = 1,
        n_iter: int = 1000,
        random_state=None,
        **kwargs,
    ) -> None:
        kwargs.pop("covariance_type", None)
        super().__init__(
            n_components=n_components,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
            **kwargs,
        )
        self.dofs_: np.ndarray = np.full(n_components, _DOF_INIT)

    # ── Emission log-likelihood ──────────────────────────────────────────────

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Return log p(x_t | state k) for all t and k.

        Parameters
        ----------
        X : np.ndarray
            Shape (T, n_features).

        Returns
        -------
        np.ndarray
            Shape (T, n_components) — per-state log-emission probabilities.
        """
        T = X.shape[0]
        log_ll = np.zeros((T, self.n_components))
        for k in range(self.n_components):
            log_ll[:, k] = _mvt_log_pdf(
                X, self.means_[k], self.covars_[k], float(self.dofs_[k])
            )
        return log_ll

    # ── Sufficient statistics ────────────────────────────────────────────────

    def _initialize_sufficient_statistics(self) -> dict:
        """Allocate accumulators for the ECM sufficient statistics."""
        stats = super()._initialize_sufficient_statistics()
        n, d = self.n_components, self.means_.shape[1]
        stats["t_post"]  = np.zeros(n)           # Σ_t γ_t(k)
        stats["t_wpost"] = np.zeros(n)           # Σ_t γ_t(k) u_t(k)
        stats["t_wobs"]  = np.zeros((n, d))      # Σ_t γ_t(k) u_t(k) x_t
        stats["t_wcov"]  = np.zeros((n, d, d))   # Σ_t γ_t(k) u_t(k) x_t x_t^T  (raw)
        stats["t_dof_c"] = np.zeros(n)           # Σ_t γ_t(k) [log(u) - u]
        return stats

    def _accumulate_sufficient_statistics(
        self,
        stats: dict,
        X: np.ndarray,
        framelogprob: np.ndarray,
        posteriors: np.ndarray,
        fwdlattice: np.ndarray,
        bwdlattice: np.ndarray,
    ) -> None:
        """Accumulate ECM sufficient statistics weighted by auxiliary u_t(k).

        Parameters
        ----------
        stats : dict
            Accumulator dict from ``_initialize_sufficient_statistics``.
        X : np.ndarray
            Shape (T, d).
        posteriors : np.ndarray
            Shape (T, n_components) — γ_t(k) smoothed posteriors.
        """
        super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice
        )

        T, d = X.shape
        for k in range(self.n_components):
            nu_k = float(self.dofs_[k])
            diff = X - self.means_[k]  # (T, d)
            try:
                L = np.linalg.cholesky(self.covars_[k])
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(self.covars_[k] + 1e-6 * np.eye(d))
            y = np.linalg.solve(L, diff.T)
            maha = np.sum(y ** 2, axis=0)          # (T,)
            u = (nu_k + d) / (nu_k + maha)         # (T,) auxiliary weights
            gamma = posteriors[:, k]               # (T,)
            gu = gamma * u                         # (T,)

            stats["t_post"][k]  += gamma.sum()
            stats["t_wpost"][k] += gu.sum()
            stats["t_wobs"][k]  += gu @ X          # (d,)
            # Raw outer product accumulation — avoids recentering artefacts
            stats["t_wcov"][k]  += (gu[:, None] * X).T @ X   # (d, d)
            # Fixed-point auxiliary: E[log u - u | k]
            with np.errstate(divide="ignore", invalid="ignore"):
                log_u = np.where(u > 0, np.log(u), 0.0)
            stats["t_dof_c"][k] += (gamma * (log_u - u)).sum()

    # ── M-step ───────────────────────────────────────────────────────────────

    def _do_mstep(self, stats: dict) -> None:
        """ECM M-step: update means, covariances, and degrees of freedom.

        Parameters
        ----------
        stats : dict
            Accumulated sufficient statistics.

        Notes
        -----
        Means and covariances use u-weighted sums (ECM).
        Degrees of freedom are updated via the Liu & Rubin (1995) fixed-point
        iteration after the M-step for means and covariances.
        The raw-outer-product trick:
            Σ_new = (Σ_t γu x x^T  −  W_k * μ_new μ_new^T) / N_k
        avoids numerical cancellation from recentering around a running mean.
        """
        super()._do_mstep(stats)

        d = self.means_.shape[1]
        for k in range(self.n_components):
            N_k = stats["t_post"][k]
            W_k = stats["t_wpost"][k]
            if N_k < 1e-10 or W_k < 1e-10:
                continue

            mu_new = stats["t_wobs"][k] / W_k
            cov_new = (stats["t_wcov"][k] - W_k * np.outer(mu_new, mu_new)) / N_k
            cov_new += 1e-6 * np.eye(d)

            # Symmetrise to correct floating-point skew
            cov_new = (cov_new + cov_new.T) / 2.0

            # Ensure positive-definiteness
            eigvals = np.linalg.eigvalsh(cov_new)
            if eigvals.min() <= 0:
                cov_new += (-eigvals.min() + 1e-6) * np.eye(d)

            self.means_[k] = mu_new
            self.covars_[k] = cov_new

            c_k = stats["t_dof_c"][k] / N_k
            self.dofs_[k] = _update_dof(float(self.dofs_[k]), c_k)

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init(self, X: np.ndarray, lengths=None) -> None:
        """Initialise model parameters before EM, then reset dofs_."""
        super()._init(X, lengths)
        self.dofs_ = np.full(self.n_components, _DOF_INIT)
