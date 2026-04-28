"""Regime-conditional correlation and dependence structure (Phase 30).

Implements:

1. **regime_correlation** — per-regime Pearson, Spearman, and Kendall
   correlation matrices estimated via responsibility-weighted statistics.

2. **tail_dependence** — upper and lower tail dependence coefficients
   per regime (empirical estimator from joint exceedance proportions).

3. **dynamic_conditional_correlation** — posterior-blended DCC:
   at each bar, blends per-regime correlation matrices into a single
   instantaneous correlation estimate.

4. **correlation_stability** — measures how stable correlations are
   across regimes (Frobenius distance between correlation matrices).

5. **correlation_breakdown** — detects correlation regime breaks
   (large change in blended correlation over a rolling window).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CorrelationConfig:
    """Configuration for regime correlation analysis.

    Attributes
    ----------
    method : str
        ``"pearson"``, ``"spearman"``, or ``"kendall"``.
    min_eff_obs : float
        Minimum effective observations to fit a per-regime model.
    tail_quantile : float
        Quantile used for tail dependence (e.g. 0.10 = bottom/top 10 %).
    dcc_window : int
        Rolling window for DCC smoothing.
    """

    method: str = "pearson"
    min_eff_obs: float = 20.0
    tail_quantile: float = 0.10
    dcc_window: int = 20


# ---------------------------------------------------------------------------
# Weighted Pearson correlation
# ---------------------------------------------------------------------------


def _weighted_corr(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Responsibility-weighted Pearson correlation matrix."""
    w = weights / (weights.sum() + 1e-15)
    mean = (w[:, None] * X).sum(axis=0)
    Xc = X - mean
    cov = (w[:, None] * Xc).T @ Xc
    std = np.sqrt(np.diag(cov))
    outer_std = np.outer(std, std)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(outer_std > 1e-15, cov / outer_std, 0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _rank_transform(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted rank transform for Spearman correlation."""
    T, d = X.shape
    ranked = np.zeros_like(X)
    w = weights / (weights.sum() + 1e-15)
    for j in range(d):
        order = np.argsort(X[:, j])
        cum_w = np.cumsum(w[order])
        ranks = np.zeros(T)
        ranks[order] = cum_w - w[order] / 2.0  # midpoint rank
        ranked[:, j] = ranks
    return ranked


def _weighted_pearson_matrix(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return _weighted_corr(X, weights)


def _weighted_spearman_matrix(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    ranked = _rank_transform(X, weights)
    return _weighted_corr(ranked, weights)


def _weighted_kendall_pair(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Kendall's tau (O(n²) — use only for small n or d=2)."""
    n = len(x)
    w_norm = w / (w.sum() + 1e-15)
    concordant = 0.0
    discordant = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            wij = w_norm[i] * w_norm[j]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dy > 0:
                concordant += wij
            elif dx * dy < 0:
                discordant += wij
    denom = concordant + discordant
    return float((concordant - discordant) / max(denom, 1e-15))


def _weighted_kendall_matrix(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    C = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau = _weighted_kendall_pair(X[:, i], X[:, j], weights)
            C[i, j] = tau
            C[j, i] = tau
    return C


# ---------------------------------------------------------------------------
# Regime correlation result
# ---------------------------------------------------------------------------


@dataclass
class RegimeCorrelation:
    """Correlation matrix for one regime.

    Attributes
    ----------
    regime : int
    corr : np.ndarray, shape (d, d)
    method : str
    eff_obs : float
    """

    regime: int
    corr: np.ndarray
    method: str
    eff_obs: float


@dataclass
class RegimeCorrelationResult:
    """Output of :func:`regime_correlation`.

    Attributes
    ----------
    regime_corrs : list[RegimeCorrelation]
    global_corr : RegimeCorrelation
    stability : float
        Mean pairwise Frobenius distance between regime correlations
        (lower = more stable across regimes).
    """

    regime_corrs: list[RegimeCorrelation]
    global_corr: RegimeCorrelation
    stability: float


def regime_correlation(
    X: pd.DataFrame | np.ndarray,
    posteriors: np.ndarray,
    config: CorrelationConfig | None = None,
) -> RegimeCorrelationResult:
    """Compute per-regime correlation matrices.

    Parameters
    ----------
    X : array-like, shape (T, d)
    posteriors : np.ndarray, shape (T, K)
    config : CorrelationConfig, optional

    Returns
    -------
    RegimeCorrelationResult
    """
    cfg = config or CorrelationConfig()
    if isinstance(X, pd.DataFrame):
        Xarr = X.values.astype(float)
    else:
        Xarr = np.asarray(X, dtype=float)

    T, d = Xarr.shape
    _, K = posteriors.shape
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)

    corr_fn = {
        "pearson": _weighted_pearson_matrix,
        "spearman": _weighted_spearman_matrix,
        "kendall": _weighted_kendall_matrix,
    }.get(cfg.method, _weighted_pearson_matrix)

    regime_corrs: list[RegimeCorrelation] = []
    for k in range(K):
        wk = P_norm[:, k]
        eff_n = float(wk.sum())
        wk_use = wk if eff_n >= cfg.min_eff_obs else np.ones(T) / T
        corr = corr_fn(Xarr, wk_use)
        regime_corrs.append(RegimeCorrelation(
            regime=k, corr=corr, method=cfg.method, eff_obs=eff_n
        ))

    global_corr_mat = corr_fn(Xarr, np.ones(T) / T)
    global_corr = RegimeCorrelation(
        regime=-1, corr=global_corr_mat, method=cfg.method, eff_obs=float(T)
    )

    stab = _correlation_stability_scalar(regime_corrs)
    return RegimeCorrelationResult(
        regime_corrs=regime_corrs,
        global_corr=global_corr,
        stability=stab,
    )


def _correlation_stability_scalar(regime_corrs: list[RegimeCorrelation]) -> float:
    """Mean pairwise Frobenius distance between correlation matrices."""
    K = len(regime_corrs)
    if K < 2:
        return 0.0
    dists = []
    for i in range(K):
        for j in range(i + 1, K):
            diff = regime_corrs[i].corr - regime_corrs[j].corr
            dists.append(float(np.linalg.norm(diff, "fro")))
    return float(np.mean(dists))


# ---------------------------------------------------------------------------
# Dynamic conditional correlation (posterior-blended)
# ---------------------------------------------------------------------------


def dynamic_conditional_correlation(
    X: pd.DataFrame | np.ndarray,
    posteriors: np.ndarray,
    result: RegimeCorrelationResult,
) -> pd.DataFrame:
    """Per-bar posterior-blended correlation matrix (upper-triangle as columns).

    At each bar t, C_t = Σ_k p_{tk} * C_k.

    Returns
    -------
    pd.DataFrame
        Rows = bars, columns = ``"corr_i_j"`` for i < j.
    """
    if isinstance(X, pd.DataFrame):
        idx = X.index
        d = X.shape[1]
    else:
        idx = pd.RangeIndex(len(X))
        d = X.shape[1] if X.ndim > 1 else 1

    T, K = posteriors.shape
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)

    # Upper-triangle index pairs.
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    col_names = [f"corr_{i}_{j}" for i, j in pairs]
    out = np.zeros((T, len(pairs)))

    for t in range(T):
        C_t = np.zeros((d, d))
        for k, rc in enumerate(result.regime_corrs):
            C_t += P_norm[t, k] * rc.corr
        for m, (i, j) in enumerate(pairs):
            out[t, m] = C_t[i, j]

    return pd.DataFrame(out, index=idx, columns=col_names)


# ---------------------------------------------------------------------------
# Tail dependence
# ---------------------------------------------------------------------------


def tail_dependence(
    X: pd.DataFrame | np.ndarray,
    posteriors: np.ndarray,
    config: CorrelationConfig | None = None,
) -> pd.DataFrame:
    """Empirical lower and upper tail dependence per regime.

    For each pair (i, j) and each regime k:

    Lower: λ_L(k) = P(F_i(x_i) ≤ q | F_j(x_j) ≤ q) ≈ weighted joint count / weighted count
    Upper: λ_U(k) = P(F_i(x_i) > 1-q | F_j(x_j) > 1-q)

    Parameters
    ----------
    X : array-like, shape (T, d)
    posteriors : np.ndarray, shape (T, K)
    config : CorrelationConfig, optional

    Returns
    -------
    pd.DataFrame
        Columns: regime, pair (e.g. "0_1"), lower_tail, upper_tail.
    """
    cfg = config or CorrelationConfig()
    Xarr = X.values.astype(float) if isinstance(X, pd.DataFrame) else np.asarray(X, float)
    T, d = Xarr.shape
    _, K = posteriors.shape
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)
    q = cfg.tail_quantile

    # Convert to pseudo-copula ranks.
    U = np.zeros_like(Xarr)
    for j in range(d):
        ranks = Xarr[:, j].argsort().argsort()
        U[:, j] = (ranks + 1) / (T + 1)

    rows = []
    for k in range(K):
        wk = P_norm[:, k]
        for i in range(d):
            for j in range(i + 1, d):
                # Lower tail: both below q.
                lower_mask = (U[:, i] <= q) & (U[:, j] <= q)
                j_mask = U[:, j] <= q
                w_j = (wk * j_mask.astype(float)).sum()
                w_joint_lo = (wk * lower_mask.astype(float)).sum()
                lower_td = float(w_joint_lo / max(w_j, 1e-15))

                # Upper tail: both above 1-q.
                upper_mask = (U[:, i] >= 1.0 - q) & (U[:, j] >= 1.0 - q)
                j_upper = U[:, j] >= 1.0 - q
                w_j_up = (wk * j_upper.astype(float)).sum()
                w_joint_up = (wk * upper_mask.astype(float)).sum()
                upper_td = float(w_joint_up / max(w_j_up, 1e-15))

                rows.append({
                    "regime": k,
                    "pair": f"{i}_{j}",
                    "lower_tail": lower_td,
                    "upper_tail": upper_td,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correlation breakdown detection
# ---------------------------------------------------------------------------


def correlation_breakdown(
    dcc: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """Rolling standard deviation of the DCC columns as a change indicator.

    High values indicate the correlation is shifting rapidly — a sign
    of a regime-correlation breakdown.

    Parameters
    ----------
    dcc : pd.DataFrame
        Output of :func:`dynamic_conditional_correlation`.
    window : int

    Returns
    -------
    pd.Series — rolling std of the DCC magnitude.
    """
    magnitude = dcc.values.__abs__().mean(axis=1)
    s = pd.Series(magnitude, index=dcc.index)
    return s.rolling(window, min_periods=2).std()
