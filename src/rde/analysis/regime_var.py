"""Regime-conditional Vector Autoregression (VAR) fitted with HMM responsibility weights.

For a K-state HMM over N assets, this module fits a separate VAR(p) for each
hidden state using weighted least squares, where the weight of observation t
for state k is the forward-backward posterior γ_t(k).

The weighted OLS formulation means that the VAR for a low-activity state is
estimated primarily from bars where the HMM assigns high probability to that
state, giving a genuine regime-conditional view of return dynamics.

Impulse-response functions (Cholesky-identified) and a Granger-causality
summary table are also provided.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeVARConfig:
    """Configuration for regime-conditional VAR fitting.

    Attributes
    ----------
    lag_order : int
        Number of VAR lags (p).
    min_effective_n : float
        Minimum effective sample size Σ_t γ_t(k) required to fit the VAR
        for state k. States below this threshold are skipped.
    regularization : float
        Ridge penalty added to the diagonal of Z'WZ before solving, for
        numerical stability.
    """

    lag_order: int = 1
    min_effective_n: float = 20.0
    regularization: float = 1e-6


@dataclass
class RegimeVARResult:
    """Fitted regime-conditional VAR(p) for a single HMM state.

    Attributes
    ----------
    state : int
        HMM state index (0-based).
    n_assets : int
        Number of assets (N).
    lag_order : int
        Number of lags (p).
    intercept : np.ndarray
        Shape (N,). Regime-conditional intercept vector ĉ_k.
    coef_matrices : list[np.ndarray]
        Length p. ``coef_matrices[l]`` is the (N, N) coefficient matrix for
        lag l+1, i.e. Â_k^(l+1).  ``coef_matrices[0]`` is for lag 1.
    residual_cov : np.ndarray
        Shape (N, N). Responsibility-weighted residual covariance Σ̂_k.
        Symmetric and positive semi-definite.
    effective_n : float
        Σ_t γ_t(k) — the effective sample size used to fit this state.
    asset_names : list[str]
        Asset names in column order.
    """

    state: int
    n_assets: int
    lag_order: int
    intercept: np.ndarray
    coef_matrices: list[np.ndarray]
    residual_cov: np.ndarray
    effective_n: float
    asset_names: list[str]


def _build_lag_matrix(
    returns: np.ndarray,
    lag_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the lagged regressor matrix Z and the response matrix R.

    Parameters
    ----------
    returns : np.ndarray
        Shape (T, N). Log returns.
    lag_order : int
        Number of lags p ≥ 1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(Z, R)`` where:

        * ``Z`` has shape (T-p, 1+N*p). The first column is all-ones
          (intercept). Remaining columns are lags 1..p of all assets,
          ordered as [r_{t-1}, r_{t-2}, ..., r_{t-p}] for a given t.
        * ``R`` has shape (T-p, N). The response matrix (rows p..T-1).
    """
    T, N = returns.shape
    p = lag_order
    n_obs = T - p  # effective number of observations

    R = returns[p:]  # (T-p, N)

    # Build Z: rows are [1, r_{t-1}^T, r_{t-2}^T, ..., r_{t-p}^T]
    Z_cols = [np.ones((n_obs, 1))]
    for lag in range(1, p + 1):
        Z_cols.append(returns[p - lag: T - lag])  # (T-p, N)

    Z = np.hstack(Z_cols)  # (T-p, 1+N*p)
    return Z, R


def _weighted_ols(
    Z: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Weighted OLS with ridge regularisation.

    Solves B = (Z'WZ + λI)^{-1} Z'WR for all N response variables jointly.

    Parameters
    ----------
    Z : np.ndarray
        Shape (T_eff, q). Design matrix (q = 1 + N*p).
    R : np.ndarray
        Shape (T_eff, N). Response matrix.
    W : np.ndarray
        Shape (T_eff,). Non-negative weights (responsibility values).
    regularization : float
        Ridge penalty λ ≥ 0.

    Returns
    -------
    np.ndarray
        Shape (q, N). Coefficient matrix B where B[0] = intercept,
        B[1:] = stacked lag coefficients.

    Notes
    -----
    We use :func:`numpy.linalg.solve` instead of explicit inversion for
    numerical stability.
    """
    W_diag = W[:, np.newaxis]  # broadcast-friendly column vector
    ZtW = Z.T * W_diag.T  # (q, T_eff): each column scaled by weight
    ZtWZ = ZtW @ Z  # (q, q)
    ZtWR = ZtW @ R  # (q, N)

    # Ridge regularisation on the design matrix (not the intercept column)
    lam_mat = regularization * np.eye(ZtWZ.shape[0])
    lam_mat[0, 0] = 0.0  # do not regularise the intercept

    B = np.linalg.solve(ZtWZ + lam_mat, ZtWR)  # (q, N)
    return B


def fit_regime_var(
    returns: pd.DataFrame,
    posteriors: np.ndarray,
    config: RegimeVARConfig | None = None,
) -> list[RegimeVARResult]:
    """Fit responsibility-weighted VAR(p) within each HMM state.

    For each state k, the VAR is estimated via weighted OLS where
    observation t has weight γ_t(k) (the forward-backward posterior).

    Parameters
    ----------
    returns : pd.DataFrame
        Shape (T, N). Each column is a log-return series for one asset.
        The column names are used as ``asset_names`` in the output.
    posteriors : np.ndarray
        Shape (T, K). Smoothed HMM posteriors from the forward-backward
        algorithm. Rows must sum to 1.0.
    config : RegimeVARConfig, optional
        Defaults to ``RegimeVARConfig()``.

    Returns
    -------
    list[RegimeVARResult]
        One entry per state whose effective sample size ≥
        ``config.min_effective_n``. States that do not meet the threshold
        are logged at WARNING level and excluded from the output.

    Raises
    ------
    ValueError
        If ``returns`` and ``posteriors`` have different time dimensions,
        or if ``lag_order`` is less than 1.

    Notes
    -----
    The effective sample size for state k is Σ_t γ_t(k), which equals the
    expected number of observations in that regime under the model.

    Weighted residual covariance is computed as:
        Σ̂_k = (1/Σγ) Σ_t γ_t(k) ε_t ε_t'
    where ε_t = r_t - ĉ_k - Σ_l Â_k^(l) r_{t-l}.
    """
    if config is None:
        config = RegimeVARConfig()

    T_ret, N = returns.shape
    T_post, K = posteriors.shape

    if T_ret != T_post:
        raise ValueError(
            f"returns has {T_ret} rows but posteriors has {T_post} rows. "
            "They must have the same time dimension."
        )
    if config.lag_order < 1:
        raise ValueError(f"lag_order must be ≥ 1, got {config.lag_order}.")

    asset_names = list(returns.columns)
    ret_arr = returns.to_numpy(dtype=float)  # (T, N)
    p = config.lag_order

    Z, R = _build_lag_matrix(ret_arr, p)  # (T-p, 1+N*p), (T-p, N)
    # posteriors rows p..T-1 correspond to response rows
    gamma = posteriors[p:]  # (T-p, K)

    results: list[RegimeVARResult] = []

    for k in range(K):
        w_k = gamma[:, k]  # (T-p,)
        eff_n = float(w_k.sum())

        if eff_n < config.min_effective_n:
            logger.warning(
                "State %d: effective_n=%.1f < min_effective_n=%.1f — skipping VAR fit.",
                k, eff_n, config.min_effective_n,
            )
            continue

        logger.info(
            "Fitting VAR(p=%d) for state %d  effective_n=%.1f", p, k, eff_n
        )

        B = _weighted_ols(Z, R, w_k, config.regularization)  # (1+N*p, N)

        intercept = B[0]  # (N,)
        # Lag coefficient matrices: B[1:N+1] is lag 1, B[N+1:2N+1] is lag 2, ...
        coef_matrices: list[np.ndarray] = []
        for lag_idx in range(p):
            start = 1 + lag_idx * N
            end = 1 + (lag_idx + 1) * N
            # B is (q, N); A_k^(l)[i,j] = coefficient of asset j's lag l on asset i's future
            # Columns of B correspond to response assets, rows 1..N are lag-1 predictors
            A_l = B[start:end].T  # (N, N): A_l[i,j] = effect of asset j on asset i
            coef_matrices.append(A_l)

        # Compute weighted residuals
        # Predicted: R_hat[t] = intercept + Σ_l A_l @ r_{t-l}
        R_hat = Z @ B  # (T-p, N): Z is [1, r_{t-1}, ..., r_{t-p}]
        residuals = R - R_hat  # (T-p, N)

        # Weighted residual covariance
        w_col = w_k[:, np.newaxis]  # (T-p, 1)
        weighted_residuals = residuals * w_col  # (T-p, N)
        cov = (weighted_residuals.T @ residuals) / eff_n  # (N, N)

        # Symmetrise for numerical hygiene
        cov = 0.5 * (cov + cov.T)

        results.append(RegimeVARResult(
            state=k,
            n_assets=N,
            lag_order=p,
            intercept=intercept,
            coef_matrices=coef_matrices,
            residual_cov=cov,
            effective_n=eff_n,
            asset_names=asset_names,
        ))

    return results


def _build_companion_matrix(
    coef_matrices: list[np.ndarray],
    n_assets: int,
) -> np.ndarray:
    """Build the VAR companion matrix for IRF computation.

    For a VAR(p), the companion matrix has shape (N*p, N*p):

        A_comp = [[A^1, A^2, ..., A^p],
                  [I_{N*(p-1)}, 0       ]]

    Parameters
    ----------
    coef_matrices : list[np.ndarray]
        Length p, each (N, N). Lag coefficient matrices A^1..A^p.
    n_assets : int
        N.

    Returns
    -------
    np.ndarray
        Shape (N*p, N*p). Companion matrix.
    """
    N = n_assets
    p = len(coef_matrices)
    size = N * p
    A_comp = np.zeros((size, size))

    # Top block: [A^1 | A^2 | ... | A^p]
    for l, A_l in enumerate(coef_matrices):
        A_comp[:N, l * N:(l + 1) * N] = A_l

    # Lower-left identity block
    if p > 1:
        A_comp[N:, :N * (p - 1)] = np.eye(N * (p - 1))

    return A_comp


def compute_irf(
    var_result: RegimeVARResult,
    shock_asset: int,
    horizons: int = 10,
) -> pd.DataFrame:
    """Compute Cholesky-identified impulse response functions for a VAR state.

    The Cholesky decomposition of the residual covariance is used to
    orthogonalise the shocks.  The IRF at horizon h is:

        IRF(h) = J (A_comp)^h J' L_k

    where J = [I_N | 0_{N, N*(p-1)}] selects the first N rows/columns,
    and L_k = chol(Σ̂_k) (lower triangular).

    Parameters
    ----------
    var_result : RegimeVARResult
        A fitted :class:`RegimeVARResult` from :func:`fit_regime_var`.
    shock_asset : int
        Index (0-based) of the asset receiving the unit Cholesky shock.
    horizons : int
        Number of periods ahead to compute.  The output includes h=0
        (contemporaneous response) through h=``horizons``.

    Returns
    -------
    pd.DataFrame
        Shape (horizons+1, N). Column names = asset names.
        ``irf.iloc[h, i]`` = response of asset i at horizon h to a unit
        Cholesky shock to asset ``shock_asset`` at horizon 0.

    Notes
    -----
    The Cholesky ordering follows the column order of ``var_result.asset_names``.
    A small identity ridge (1e-8 * I) is added to ``residual_cov`` before
    Cholesky factorisation to ensure positive definiteness.
    """
    N = var_result.n_assets
    p = var_result.lag_order

    if not (0 <= shock_asset < N):
        raise ValueError(
            f"shock_asset={shock_asset} is out of range [0, {N - 1}]."
        )

    A_comp = _build_companion_matrix(var_result.coef_matrices, N)  # (N*p, N*p)

    # Cholesky of residual covariance
    cov_reg = var_result.residual_cov + 1e-8 * np.eye(N)
    try:
        L = np.linalg.cholesky(cov_reg)  # (N, N) lower triangular
    except np.linalg.LinAlgError:
        logger.warning(
            "State %d: Cholesky failed; using SVD-based pseudo-sqrt.",
            var_result.state,
        )
        # Fall back to symmetric square root via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov_reg)
        eigvals = np.maximum(eigvals, 0.0)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    # Cholesky shock vector for shock_asset
    shock_col = L[:, shock_asset]  # (N,)

    # IRF computation
    # Selector J: extracts first N rows from companion-space (N*p)
    # J @ A_comp^h @ J' has shape (N, N)
    irf_rows: list[np.ndarray] = []

    A_h = np.eye(N * p)  # A_comp^0 = I
    for h in range(horizons + 1):
        if h > 0:
            A_h = A_h @ A_comp

        # J @ A_h @ J': first N rows, first N cols of A_h
        A_h_top = A_h[:N, :N]  # (N, N)
        response = A_h_top @ shock_col  # (N,)
        irf_rows.append(response)

    irf_arr = np.stack(irf_rows, axis=0)  # (horizons+1, N)

    return pd.DataFrame(
        irf_arr,
        index=pd.RangeIndex(horizons + 1, name="horizon"),
        columns=var_result.asset_names,
    )


def granger_causality_table(
    var_results: list[RegimeVARResult],
) -> pd.DataFrame:
    """Summarise Granger causality strength across all fitted regimes.

    For each state and each ordered pair (from_asset → to_asset), the
    Granger causality strength is the sum of squared (j, i)-th elements
    across all p lag matrices: Σ_l |A_k^(l)[j, i]|.

    This is a descriptive summary, not a formal Granger-causality test.

    Parameters
    ----------
    var_results : list[RegimeVARResult]
        Output of :func:`fit_regime_var`.

    Returns
    -------
    pd.DataFrame
        Columns: ``state``, ``from_asset``, ``to_asset``,
        ``granger_strength``.  Sorted by ``state``, then
        ``granger_strength`` descending.

    Notes
    -----
    ``granger_strength`` uses the L1 (absolute-sum) norm across lags:
        Σ_{l=1}^{p} |A_k^(l)[to_idx, from_idx]|
    A larger value indicates stronger predictive content of ``from_asset``
    for ``to_asset`` within state k, but significance requires a formal F-test.
    """
    rows: list[dict] = []

    for result in var_results:
        N = result.n_assets
        names = result.asset_names

        for from_idx in range(N):
            for to_idx in range(N):
                # Sum |A_k^(l)[to_idx, from_idx]| across all lags
                strength = sum(
                    abs(A_l[to_idx, from_idx])
                    for A_l in result.coef_matrices
                )
                rows.append({
                    "state": result.state,
                    "from_asset": names[from_idx],
                    "to_asset": names[to_idx],
                    "granger_strength": float(strength),
                })

    if not rows:
        return pd.DataFrame(
            columns=["state", "from_asset", "to_asset", "granger_strength"]
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["state", "granger_strength"], ascending=[True, False]
    ).reset_index(drop=True)
    return df
