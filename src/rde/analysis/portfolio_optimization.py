"""Regime-aware portfolio optimization (Phase 25).

Implements:

1. **RegimeMVOResult** — per-regime mean-variance optimal weights
   (minimum variance, maximum Sharpe, and target-return frontier).

2. **regime_mvo** — fits Markowitz mean-variance optimization per regime
   using responsibility-weighted moments.

3. **blended_mvo** — posterior-blends regime MVO weights into a single
   allocation.

4. **efficient_frontier** — traces the efficient frontier (risk vs.
   return) for a given regime's moment estimates.

5. **black_litterman_posterior** — simple Black-Litterman update:
   blends the market-implied equilibrium returns with analyst views
   weighted by the current regime posterior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MVOConfig:
    """Configuration for mean-variance optimization.

    Attributes
    ----------
    risk_free_rate : float
        Annualised risk-free rate (e.g. 0.05 = 5 %).  Used for Sharpe.
    regularisation : float
        Ridge regularisation added to the covariance diagonal to ensure
        invertibility.
    min_weight : float
        Minimum weight per asset (0 = long-only floor, -1 = short allowed).
    max_weight : float
        Maximum weight per asset.
    n_frontier_points : int
        Number of points along the efficient frontier.
    min_eff_obs : float
        Minimum effective observations to fit per-regime model; falls
        back to global moments otherwise.
    """

    risk_free_rate: float = 0.0
    regularisation: float = 1e-6
    min_weight: float = 0.0
    max_weight: float = 1.0
    n_frontier_points: int = 30
    min_eff_obs: float = 20.0


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RegimeMVOWeights:
    """Optimal weights for one regime.

    Attributes
    ----------
    regime : int
    min_var_weights : np.ndarray, shape (d,)
    max_sharpe_weights : np.ndarray, shape (d,)
    mean : np.ndarray, shape (d,)
    cov : np.ndarray, shape (d, d)
    eff_obs : float
    """

    regime: int
    min_var_weights: np.ndarray
    max_sharpe_weights: np.ndarray
    mean: np.ndarray
    cov: np.ndarray
    eff_obs: float


@dataclass
class RegimeMVOResult:
    """Output of :func:`regime_mvo`.

    Attributes
    ----------
    regime_weights : list[RegimeMVOWeights]
    global_weights : RegimeMVOWeights
        Weights estimated from equal-weight moments (regime=-1).
    asset_names : list[str] | None
    """

    regime_weights: list[RegimeMVOWeights]
    global_weights: RegimeMVOWeights
    asset_names: list[str] | None


@dataclass
class EfficientFrontier:
    """Points along the efficient frontier.

    Attributes
    ----------
    volatilities : np.ndarray, shape (n,)
    returns : np.ndarray, shape (n,)
    sharpe_ratios : np.ndarray, shape (n,)
    weights : np.ndarray, shape (n, d)
    """

    volatilities: np.ndarray
    returns: np.ndarray
    sharpe_ratios: np.ndarray
    weights: np.ndarray


# ---------------------------------------------------------------------------
# Weighted moments
# ---------------------------------------------------------------------------


def _weighted_moments(
    returns: np.ndarray,
    weights: np.ndarray,
    regularisation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, cov) estimated with observation weights."""
    w = weights / (weights.sum() + 1e-15)
    mean = (w[:, None] * returns).sum(axis=0)
    Xc = returns - mean
    cov = (w[:, None] * Xc).T @ Xc
    cov += np.eye(cov.shape[0]) * regularisation
    return mean, cov


# ---------------------------------------------------------------------------
# Quadratic-programming-free MVO (analytical solutions)
# ---------------------------------------------------------------------------


def _min_variance_weights(
    cov: np.ndarray,
    min_w: float,
    max_w: float,
) -> np.ndarray:
    """Minimum-variance weights via projected gradient / analytical solution.

    For unconstrained (no box constraints), the solution is the standard
    w = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1).  Box constraints are enforced via iterative
    clipping and renormalisation (heuristic but numerically stable and
    avoids a full QP solver dependency).
    """
    d = cov.shape[0]
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.full(d, 1.0 / d)

    ones = np.ones(d)
    raw = cov_inv @ ones
    raw = raw / (raw.sum() + 1e-15)

    # Project onto box constraints.
    raw = _box_project(raw, min_w, max_w, d)
    return raw


def _max_sharpe_weights(
    mean: np.ndarray,
    cov: np.ndarray,
    rf: float,
    min_w: float,
    max_w: float,
) -> np.ndarray:
    """Maximum-Sharpe-ratio weights.

    Analytical: w ∝ Σ⁻¹ (μ - rf·1), then project onto box constraints.
    """
    d = cov.shape[0]
    excess = mean - rf

    # If all excess returns are non-positive, fall back to min-variance.
    if excess.max() <= 0.0:
        return _min_variance_weights(cov, min_w, max_w)

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.full(d, 1.0 / d)

    raw = cov_inv @ excess
    # If sum is negative (e.g. all weights negative before projection),
    # flip sign to get long portfolio.
    if raw.sum() < 0:
        raw = -raw
    denom = raw.sum()
    if abs(denom) < 1e-15:
        return _min_variance_weights(cov, min_w, max_w)
    raw = raw / denom

    raw = _box_project(raw, min_w, max_w, d)
    return raw


def _box_project(w: np.ndarray, lo: float, hi: float, d: int) -> np.ndarray:
    """Project w onto [lo, hi]^d ∩ {sum=1} via iterative clipping."""
    for _ in range(50):
        w = np.clip(w, lo, hi)
        total = w.sum()
        if abs(total) < 1e-15:
            w = np.full(d, 1.0 / d)
            break
        w = w / total
        if np.all(w >= lo - 1e-12) and np.all(w <= hi + 1e-12):
            break
    return w


# ---------------------------------------------------------------------------
# Regime MVO
# ---------------------------------------------------------------------------


def regime_mvo(
    returns: pd.DataFrame | np.ndarray,
    posteriors: np.ndarray,
    config: MVOConfig | None = None,
) -> RegimeMVOResult:
    """Fit per-regime mean-variance optimal portfolios.

    Parameters
    ----------
    returns : array-like, shape (T, d)
        Multi-asset return matrix.
    posteriors : np.ndarray, shape (T, K)
    config : MVOConfig, optional

    Returns
    -------
    RegimeMVOResult
    """
    cfg = config or MVOConfig()
    if isinstance(returns, pd.DataFrame):
        asset_names = list(returns.columns)
        R = returns.values.astype(float)
    else:
        asset_names = None
        R = np.asarray(returns, dtype=float)

    T, d = R.shape
    _, K = posteriors.shape
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)

    regime_weights: list[RegimeMVOWeights] = []
    for k in range(K):
        wk = P_norm[:, k]
        eff_n = float(wk.sum())

        if eff_n < cfg.min_eff_obs:
            wk_use = np.ones(T) / T
        else:
            wk_use = wk

        mean, cov = _weighted_moments(R, wk_use, cfg.regularisation)
        min_var = _min_variance_weights(cov, cfg.min_weight, cfg.max_weight)
        max_sharpe = _max_sharpe_weights(mean, cov, cfg.risk_free_rate,
                                         cfg.min_weight, cfg.max_weight)

        regime_weights.append(
            RegimeMVOWeights(
                regime=k,
                min_var_weights=min_var,
                max_sharpe_weights=max_sharpe,
                mean=mean,
                cov=cov,
                eff_obs=eff_n,
            )
        )

    # Global (equal-weight moments).
    g_mean, g_cov = _weighted_moments(R, np.ones(T) / T, cfg.regularisation)
    global_weights = RegimeMVOWeights(
        regime=-1,
        min_var_weights=_min_variance_weights(g_cov, cfg.min_weight, cfg.max_weight),
        max_sharpe_weights=_max_sharpe_weights(g_mean, g_cov, cfg.risk_free_rate,
                                                cfg.min_weight, cfg.max_weight),
        mean=g_mean,
        cov=g_cov,
        eff_obs=float(T),
    )

    return RegimeMVOResult(
        regime_weights=regime_weights,
        global_weights=global_weights,
        asset_names=asset_names,
    )


# ---------------------------------------------------------------------------
# Blended weights
# ---------------------------------------------------------------------------


def blended_mvo(
    result: RegimeMVOResult,
    current_posteriors: np.ndarray,
    kind: str = "max_sharpe",
) -> np.ndarray:
    """Blend per-regime MVO weights by current posterior.

    Parameters
    ----------
    result : RegimeMVOResult
    current_posteriors : np.ndarray, shape (K,)
    kind : str
        ``"max_sharpe"`` or ``"min_var"``.

    Returns
    -------
    np.ndarray, shape (d,)
    """
    pt = current_posteriors / (current_posteriors.sum() + 1e-15)
    K = len(result.regime_weights)
    d = result.global_weights.min_var_weights.shape[0]
    blended = np.zeros(d)
    for k, rw in enumerate(result.regime_weights):
        if kind == "min_var":
            w = rw.min_var_weights
        else:
            w = rw.max_sharpe_weights
        blended += pt[k] * w
    total = blended.sum()
    if abs(total) > 1e-15:
        blended /= total
    return blended


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------


def efficient_frontier(
    weights_obj: RegimeMVOWeights,
    config: MVOConfig | None = None,
) -> EfficientFrontier:
    """Trace the efficient frontier for one regime's moments.

    Parameterises the frontier by a target-return grid between the
    minimum-variance return and the maximum-mean-return portfolio.

    Parameters
    ----------
    weights_obj : RegimeMVOWeights
    config : MVOConfig, optional

    Returns
    -------
    EfficientFrontier
    """
    cfg = config or MVOConfig()
    mean = weights_obj.mean
    cov = weights_obj.cov
    d = len(mean)
    n = cfg.n_frontier_points

    ret_min = float(mean.min())
    ret_max = float(mean.max())
    if ret_max - ret_min < 1e-10:
        # Degenerate: all returns equal.
        min_var_w = _min_variance_weights(cov, cfg.min_weight, cfg.max_weight)
        r = float(min_var_w @ mean)
        v = float(np.sqrt(max(min_var_w @ cov @ min_var_w, 0.0)))
        s = (r - cfg.risk_free_rate) / (v + 1e-15)
        return EfficientFrontier(
            volatilities=np.full(n, v),
            returns=np.full(n, r),
            sharpe_ratios=np.full(n, s),
            weights=np.tile(min_var_w, (n, 1)),
        )

    target_returns = np.linspace(ret_min, ret_max, n)
    vols = np.zeros(n)
    rets = np.zeros(n)
    sharpes = np.zeros(n)
    all_weights = np.zeros((n, d))

    # For each target return, use a two-fund separation heuristic:
    # blend min-var weights (low return) with max-mean weights (high return).
    min_var_w = _min_variance_weights(cov, cfg.min_weight, cfg.max_weight)
    max_mean_w = np.zeros(d)
    max_mean_w[int(np.argmax(mean))] = 1.0

    r_low = float(min_var_w @ mean)
    r_high = float(max_mean_w @ mean)
    rng = max(r_high - r_low, 1e-15)

    for i, tgt in enumerate(target_returns):
        alpha = np.clip((tgt - r_low) / rng, 0.0, 1.0)
        w = (1.0 - alpha) * min_var_w + alpha * max_mean_w
        w = _box_project(w, cfg.min_weight, cfg.max_weight, d)
        port_ret = float(w @ mean)
        port_var = float(w @ cov @ w)
        port_vol = float(np.sqrt(max(port_var, 0.0)))
        all_weights[i] = w
        rets[i] = port_ret
        vols[i] = port_vol
        sharpes[i] = (port_ret - cfg.risk_free_rate) / (port_vol + 1e-15)

    return EfficientFrontier(
        volatilities=vols,
        returns=rets,
        sharpe_ratios=sharpes,
        weights=all_weights,
    )


# ---------------------------------------------------------------------------
# Black-Litterman posterior
# ---------------------------------------------------------------------------


def black_litterman_posterior(
    equilibrium_returns: np.ndarray,
    cov: np.ndarray,
    views: np.ndarray,
    view_confidence: np.ndarray,
    posteriors: np.ndarray | None = None,
    tau: float = 0.05,
) -> np.ndarray:
    """Black-Litterman posterior expected returns.

    Blends market-implied equilibrium μ_eq with absolute views using the
    BL formula:

        μ_BL = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹ [(τΣ)⁻¹ μ_eq + Pᵀ Ω⁻¹ q]

    where P = I (absolute views), Ω = diag(1/confidence).

    If ``posteriors`` is provided, it is used as an additional confidence
    weight on each view asset (blended across regimes).

    Parameters
    ----------
    equilibrium_returns : np.ndarray, shape (d,)
    cov : np.ndarray, shape (d, d)
    views : np.ndarray, shape (d,)
        Analyst view on expected return per asset.
    view_confidence : np.ndarray, shape (d,)
        Confidence in each view (higher = more weight).  Values in (0, 1].
    posteriors : np.ndarray, shape (K,) or None
        Current regime posterior (used to scale view confidence by
        blended-regime conviction).  Optional.
    tau : float
        Uncertainty scaling on prior (typically 0.01–0.1).

    Returns
    -------
    np.ndarray, shape (d,)
        Posterior expected returns.
    """
    d = len(equilibrium_returns)
    Sigma_tau_inv = np.linalg.inv(tau * cov + np.eye(d) * 1e-10)

    # Absolute views: P = I.
    conf = np.clip(view_confidence, 1e-8, 1.0)
    if posteriors is not None:
        pt = posteriors / (posteriors.sum() + 1e-15)
        # Scale confidence up by max posterior (regime conviction).
        conf = conf * float(pt.max())

    Omega_inv = np.diag(conf)  # higher confidence → more weight

    lhs = Sigma_tau_inv + Omega_inv
    rhs = Sigma_tau_inv @ equilibrium_returns + Omega_inv @ views

    try:
        mu_bl = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        mu_bl = equilibrium_returns.copy()

    return mu_bl
