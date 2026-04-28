"""Regime-conditional factor analysis (Phase 23).

Decomposes multi-asset or multi-feature returns into regime-specific
principal components / latent factors.  Each regime has its own
covariance structure; PCA on each reveals how dominant risk factors
shift as the market changes state.

Implements:

1. **RegimeFactorModel** — fits PCA per regime using responsibility-weighted
   covariance matrices, returning per-regime factor loadings, explained
   variance ratios, and factor scores.

2. **factor_alignment** — aligns factor signs across regimes so that the
   first factor always has positive mean loading (avoids sign flip
   ambiguity in PCA).

3. **rolling_factor_exposure** — computes a rolling posterior-blended
   factor exposure series for a single asset within a multi-asset frame.

4. **factor_stability** — measures how stable factors are across regimes
   (cosine similarity of loading vectors; high = factor survives regime
   change, low = factor is regime-specific).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FactorConfig:
    """Configuration for regime-conditional factor analysis.

    Attributes
    ----------
    n_factors : int
        Number of principal components to retain per regime.
    min_eff_obs : float
        Minimum effective observations (sum of weights) before fitting
        a regime; regimes with fewer observations use the global PCA.
    center : bool
        Whether to demean the data before computing covariance.
    """

    n_factors: int = 3
    min_eff_obs: float = 20.0
    center: bool = True


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RegimeFactors:
    """PCA result for a single regime.

    Attributes
    ----------
    regime : int
    loadings : np.ndarray, shape (d, n_factors)
        Principal component vectors (columns are eigenvectors).
    explained_variance_ratio : np.ndarray, shape (n_factors,)
    mean : np.ndarray, shape (d,)
        Responsibility-weighted mean (used for centering).
    eff_obs : float
        Effective number of observations (sum of weights).
    """

    regime: int
    loadings: np.ndarray
    explained_variance_ratio: np.ndarray
    mean: np.ndarray
    eff_obs: float


@dataclass
class RegimeFactorResult:
    """Output of :func:`fit_regime_factors`.

    Attributes
    ----------
    regime_factors : list[RegimeFactors]
        One entry per regime.
    global_factors : RegimeFactors
        Unconditional PCA (regime=-1, equal weights).
    factor_stability : np.ndarray, shape (n_factors,)
        Mean pairwise cosine similarity for each factor across regimes.
    """

    regime_factors: list[RegimeFactors]
    global_factors: RegimeFactors
    factor_stability: np.ndarray


# ---------------------------------------------------------------------------
# Core fitting
# ---------------------------------------------------------------------------


def _weighted_pca(
    X: np.ndarray,
    weights: np.ndarray,
    n_factors: int,
    center: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute weighted PCA.

    Returns (loadings (d, n_factors), explained_var_ratio (n_factors,), mean (d,)).
    """
    w = weights / (weights.sum() + 1e-15)
    mean = (w[:, None] * X).sum(axis=0) if center else np.zeros(X.shape[1])
    Xc = X - mean if center else X

    cov = (w[:, None] * Xc).T @ Xc  # (d, d) weighted covariance

    # Regularise for numerical stability.
    cov += np.eye(cov.shape[0]) * 1e-10

    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending order; reverse to get descending.
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    nf = min(n_factors, X.shape[1])
    loadings = eigvecs[:, :nf]
    total_var = max(eigvals.sum(), 1e-15)
    evr = eigvals[:nf] / total_var

    return loadings, evr, mean


def fit_regime_factors(
    X: pd.DataFrame | np.ndarray,
    posteriors: np.ndarray,
    config: FactorConfig | None = None,
) -> RegimeFactorResult:
    """Fit regime-conditional PCA models.

    Parameters
    ----------
    X : array-like, shape (T, d)
        Feature/returns matrix.
    posteriors : np.ndarray, shape (T, K)
        Regime posterior responsibilities.
    config : FactorConfig, optional

    Returns
    -------
    RegimeFactorResult
    """
    cfg = config or FactorConfig()
    if isinstance(X, pd.DataFrame):
        Xarr = X.values.astype(float)
    else:
        Xarr = np.asarray(X, dtype=float)

    T, d = Xarr.shape
    _, K = posteriors.shape
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)

    nf = min(cfg.n_factors, d)

    # Per-regime PCA.
    regime_factors: list[RegimeFactors] = []
    for k in range(K):
        wk = P_norm[:, k]
        eff_n = float(wk.sum())
        if eff_n < cfg.min_eff_obs:
            # Fall back to equal weights (global).
            wk_use = np.ones(T) / T
        else:
            wk_use = wk

        loadings, evr, mean = _weighted_pca(Xarr, wk_use, nf, cfg.center)
        regime_factors.append(
            RegimeFactors(
                regime=k,
                loadings=loadings,
                explained_variance_ratio=evr,
                mean=mean,
                eff_obs=eff_n,
            )
        )

    # Global PCA (equal weights).
    global_loadings, global_evr, global_mean = _weighted_pca(
        Xarr, np.ones(T) / T, nf, cfg.center
    )
    global_factors = RegimeFactors(
        regime=-1,
        loadings=global_loadings,
        explained_variance_ratio=global_evr,
        mean=global_mean,
        eff_obs=float(T),
    )

    # Align factor signs.
    for rf in regime_factors:
        rf.loadings = _align_signs(rf.loadings, global_loadings)

    stab = _factor_stability(regime_factors, nf)

    return RegimeFactorResult(
        regime_factors=regime_factors,
        global_factors=global_factors,
        factor_stability=stab,
    )


# ---------------------------------------------------------------------------
# Sign alignment
# ---------------------------------------------------------------------------


def _align_signs(loadings: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip factor signs so each factor has positive dot product with reference."""
    aligned = loadings.copy()
    n_factors = min(loadings.shape[1], reference.shape[1])
    for f in range(n_factors):
        if np.dot(aligned[:, f], reference[:, f]) < 0:
            aligned[:, f] = -aligned[:, f]
    return aligned


def factor_alignment(result: RegimeFactorResult) -> RegimeFactorResult:
    """Re-align all regime factor signs against the global factors in-place."""
    for rf in result.regime_factors:
        rf.loadings = _align_signs(rf.loadings, result.global_factors.loadings)
    result.factor_stability = _factor_stability(
        result.regime_factors, result.global_factors.loadings.shape[1]
    )
    return result


# ---------------------------------------------------------------------------
# Factor stability
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-15
    return float(abs(np.dot(a, b)) / denom)


def _factor_stability(regime_factors: list[RegimeFactors], n_factors: int) -> np.ndarray:
    """Mean pairwise absolute cosine similarity for each factor across regimes."""
    K = len(regime_factors)
    if K < 2:
        return np.ones(n_factors)

    stab = np.zeros(n_factors)
    for f in range(n_factors):
        sims = []
        for i in range(K):
            for j in range(i + 1, K):
                nf_i = regime_factors[i].loadings.shape[1]
                nf_j = regime_factors[j].loadings.shape[1]
                if f < nf_i and f < nf_j:
                    sims.append(
                        _cosine_similarity(
                            regime_factors[i].loadings[:, f],
                            regime_factors[j].loadings[:, f],
                        )
                    )
        stab[f] = float(np.mean(sims)) if sims else 1.0
    return stab


# ---------------------------------------------------------------------------
# Rolling factor exposure
# ---------------------------------------------------------------------------


def rolling_factor_exposure(
    X: pd.DataFrame | np.ndarray,
    posteriors: np.ndarray,
    result: RegimeFactorResult,
    asset_idx: int = 0,
) -> pd.DataFrame:
    """Compute per-bar blended factor exposure for a given asset.

    At each bar, blends the per-regime factor loadings by the posterior
    and returns the factor exposure vector for ``asset_idx``.

    Parameters
    ----------
    X : array-like, shape (T, d)
        Feature matrix (rows = bars, columns = assets/features).
    posteriors : np.ndarray, shape (T, K)
    result : RegimeFactorResult
    asset_idx : int
        Column index in X whose factor exposures to compute.

    Returns
    -------
    pd.DataFrame
        Shape (T, n_factors), columns ``factor_0``, ``factor_1``, ...
        Values are responsibility-blended loadings for ``asset_idx``.
    """
    if isinstance(X, pd.DataFrame):
        idx = X.index
    else:
        idx = pd.RangeIndex(len(X))

    T, K = posteriors.shape
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)
    nf = result.global_factors.loadings.shape[1]

    exposures = np.zeros((T, nf))
    for t in range(T):
        blended_loading = np.zeros(nf)
        for k, rf in enumerate(result.regime_factors):
            actual_nf = rf.loadings.shape[1]
            for f in range(min(nf, actual_nf)):
                blended_loading[f] += P_norm[t, k] * rf.loadings[asset_idx, f]
        exposures[t] = blended_loading

    cols = [f"factor_{f}" for f in range(nf)]
    return pd.DataFrame(exposures, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# Factor score projection
# ---------------------------------------------------------------------------


def project_to_factors(
    X: pd.DataFrame | np.ndarray,
    posteriors: np.ndarray,
    result: RegimeFactorResult,
) -> pd.DataFrame:
    """Project observations onto per-bar blended factor space.

    At each bar, uses the posterior-blended loading matrix to project the
    mean-subtracted observation onto the factor axes.

    Parameters
    ----------
    X : array-like, shape (T, d)
    posteriors : np.ndarray, shape (T, K)
    result : RegimeFactorResult

    Returns
    -------
    pd.DataFrame
        Shape (T, n_factors).
    """
    if isinstance(X, pd.DataFrame):
        Xarr = X.values.astype(float)
        idx = X.index
    else:
        Xarr = np.asarray(X, dtype=float)
        idx = pd.RangeIndex(len(Xarr))

    T, K = posteriors.shape
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)
    nf = result.global_factors.loadings.shape[1]

    scores = np.zeros((T, nf))
    for t in range(T):
        # Blended mean and loadings.
        blended_mean = np.zeros(Xarr.shape[1])
        blended_L = np.zeros((Xarr.shape[1], nf))
        for k, rf in enumerate(result.regime_factors):
            wk = P_norm[t, k]
            blended_mean += wk * rf.mean
            actual_nf = rf.loadings.shape[1]
            blended_L[:, :actual_nf] += wk * rf.loadings[:, :actual_nf]
        xc = Xarr[t] - blended_mean
        scores[t] = xc @ blended_L

    cols = [f"factor_{f}" for f in range(nf)]
    return pd.DataFrame(scores, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# Factor return decomposition
# ---------------------------------------------------------------------------


def factor_return_decomposition(
    returns: pd.Series,
    factor_scores: pd.DataFrame,
) -> pd.DataFrame:
    """OLS decomposition of returns onto factor scores.

    Parameters
    ----------
    returns : pd.Series, length T
    factor_scores : pd.DataFrame, shape (T, n_factors)

    Returns
    -------
    pd.DataFrame
        Columns: factor names + ``residual``.  Values: contribution per bar
        (beta_f * score_f_t).
    """
    T = len(returns)
    r = returns.values.astype(float)
    F = factor_scores.values.astype(float)

    # OLS: β = (FᵀF)⁻¹ Fᵀ r
    try:
        betas, _, _, _ = np.linalg.lstsq(F, r, rcond=None)
    except np.linalg.LinAlgError:
        betas = np.zeros(F.shape[1])

    contributions = F * betas[None, :]
    residual = r - contributions.sum(axis=1)

    result = pd.DataFrame(
        contributions,
        index=returns.index,
        columns=factor_scores.columns,
    )
    result["residual"] = residual
    return result
