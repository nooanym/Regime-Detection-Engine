"""Regime-conditional cointegration and spread analysis (Phase 24).

Implements:

1. **RegimeSpreadModel** — fits an OLS hedge ratio and computes the spread
   (residual) per regime using responsibility-weighted OLS.

2. **half_life_ornstein_uhlenbeck** — fits a discrete AR(1) to the spread
   and derives the half-life of mean reversion.

3. **regime_half_lives** — per-regime half-life estimation.

4. **spread_zscore** — rolling Z-score of the spread using
   responsibility-weighted mean and std.

5. **cointegration_eigenvalues** — simplified cointegration test via
   eigendecomposition of the regime-conditional difference covariance:
   a large leading eigenvalue relative to the gap indicates a
   cointegrating relationship.

6. **RegimeCointegrationResult** — aggregates per-regime spread statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SpreadConfig:
    """Configuration for spread / cointegration analysis.

    Attributes
    ----------
    zscore_window : int
        Rolling window for spread Z-score normalisation.
    min_eff_obs : float
        Minimum effective observations to fit a per-regime model.
    half_life_max : float
        Cap for reported half-life (bars).  Half-lives above this are
        clamped (indicates near unit-root — no mean reversion).
    """

    zscore_window: int = 20
    min_eff_obs: float = 20.0
    half_life_max: float = 1000.0


# ---------------------------------------------------------------------------
# Hedge ratio (weighted OLS)
# ---------------------------------------------------------------------------


def weighted_ols(
    y: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Weighted OLS: y = alpha + beta * x.

    Returns (alpha, beta).
    """
    w = weights / (weights.sum() + 1e-15)
    x_mean = float((w * x).sum())
    y_mean = float((w * y).sum())
    cov_xy = float((w * (x - x_mean) * (y - y_mean)).sum())
    var_x = float((w * (x - x_mean) ** 2).sum())
    beta = cov_xy / (var_x + 1e-15)
    alpha = y_mean - beta * x_mean
    return alpha, beta


# ---------------------------------------------------------------------------
# Half-life of mean reversion
# ---------------------------------------------------------------------------


def half_life_ornstein_uhlenbeck(spread: np.ndarray) -> float:
    """Estimate mean-reversion half-life from an AR(1) fit to the spread.

    Fits: Δspread_t = a + b * spread_{t-1} + ε

    Half-life = -log(2) / log(1 + b)

    Returns ``np.inf`` if ``b >= 0`` (no mean reversion) or ``b <= -2``
    (explosive).
    """
    if len(spread) < 3:
        return np.inf
    s = np.asarray(spread, dtype=float)
    s_lag = s[:-1]
    ds = np.diff(s)

    # OLS with intercept.
    A = np.column_stack([np.ones(len(s_lag)), s_lag])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, ds, rcond=None)
    except np.linalg.LinAlgError:
        return np.inf
    b = float(coeffs[1])

    if b >= 0.0 or b <= -2.0:
        return np.inf
    phi = 1.0 + b
    if phi <= 0.0:
        # Oscillating mean-reversion (-2 < b < -1); use |phi| for half-life.
        phi = abs(phi)
    if phi <= 1e-15:
        return np.inf
    return float(-np.log(2.0) / np.log(phi))


# ---------------------------------------------------------------------------
# Per-regime spread model
# ---------------------------------------------------------------------------


@dataclass
class RegimeSpreadParams:
    """Spread model parameters for one regime.

    Attributes
    ----------
    regime : int
    alpha : float
        OLS intercept.
    beta : float
        Hedge ratio.
    spread_mean : float
        Responsibility-weighted spread mean.
    spread_std : float
        Responsibility-weighted spread std.
    half_life : float
        Mean-reversion half-life in bars.
    eff_obs : float
    """

    regime: int
    alpha: float
    beta: float
    spread_mean: float
    spread_std: float
    half_life: float
    eff_obs: float


@dataclass
class RegimeCointegrationResult:
    """Output of :func:`fit_regime_spreads`.

    Attributes
    ----------
    regime_params : list[RegimeSpreadParams]
    spread : pd.Series
        Full-sample spread using global OLS hedge ratio.
    global_alpha : float
    global_beta : float
    global_half_life : float
    eigenvalues : np.ndarray
        Leading eigenvalues from the difference-covariance matrix,
        one per regime.
    """

    regime_params: list[RegimeSpreadParams]
    spread: pd.Series
    global_alpha: float
    global_beta: float
    global_half_life: float
    eigenvalues: np.ndarray


def fit_regime_spreads(
    y: pd.Series | np.ndarray,
    x: pd.Series | np.ndarray,
    posteriors: np.ndarray,
    config: SpreadConfig | None = None,
) -> RegimeCointegrationResult:
    """Fit per-regime spread models.

    Parameters
    ----------
    y : array-like, length T
        Dependent series (e.g. asset 1 log-price).
    x : array-like, length T
        Independent series (e.g. asset 2 log-price).
    posteriors : np.ndarray, shape (T, K)
    config : SpreadConfig, optional

    Returns
    -------
    RegimeCointegrationResult
    """
    cfg = config or SpreadConfig()
    yarr = np.asarray(y, dtype=float)
    xarr = np.asarray(x, dtype=float)
    T, K = posteriors.shape
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)
    idx = y.index if isinstance(y, pd.Series) else pd.RangeIndex(T)

    # Global OLS.
    global_alpha, global_beta = weighted_ols(yarr, xarr, np.ones(T))
    global_spread = yarr - global_alpha - global_beta * xarr
    global_hl = half_life_ornstein_uhlenbeck(global_spread)
    global_hl = min(global_hl, cfg.half_life_max)

    # Per-regime models.
    regime_params: list[RegimeSpreadParams] = []
    eigenvalues = np.zeros(K)

    for k in range(K):
        wk = P_norm[:, k]
        eff_n = float(wk.sum())

        if eff_n < cfg.min_eff_obs:
            alpha_k, beta_k = global_alpha, global_beta
        else:
            alpha_k, beta_k = weighted_ols(yarr, xarr, wk)

        spread_k = yarr - alpha_k - beta_k * xarr
        s_mean = float((wk * spread_k).sum() / max(eff_n, 1.0))
        s_var = float((wk * (spread_k - s_mean) ** 2).sum() / max(eff_n, 1.0))
        s_std = float(np.sqrt(max(s_var, 0.0)))

        # Half-life on the regime-weighted spread (sample those bars).
        if eff_n >= cfg.min_eff_obs:
            hl = half_life_ornstein_uhlenbeck(spread_k)
        else:
            hl = global_hl
        hl = min(hl, cfg.half_life_max)

        # Leading eigenvalue of weighted difference covariance.
        stack = np.column_stack([yarr, xarr])
        diff_stack = np.diff(stack, axis=0)
        w_diff = wk[1:]
        eigenvalues[k] = _leading_eigenvalue(diff_stack, w_diff)

        regime_params.append(
            RegimeSpreadParams(
                regime=k,
                alpha=alpha_k,
                beta=beta_k,
                spread_mean=s_mean,
                spread_std=s_std,
                half_life=hl,
                eff_obs=eff_n,
            )
        )

    return RegimeCointegrationResult(
        regime_params=regime_params,
        spread=pd.Series(global_spread, index=idx),
        global_alpha=global_alpha,
        global_beta=global_beta,
        global_half_life=global_hl,
        eigenvalues=eigenvalues,
    )


def _leading_eigenvalue(X: np.ndarray, weights: np.ndarray) -> float:
    """Largest eigenvalue of the responsibility-weighted covariance of X."""
    w = weights / (weights.sum() + 1e-15)
    mean = (w[:, None] * X).sum(axis=0)
    Xc = X - mean
    cov = (w[:, None] * Xc).T @ Xc + np.eye(X.shape[1]) * 1e-10
    eigvals = np.linalg.eigvalsh(cov)
    return float(eigvals.max())


# ---------------------------------------------------------------------------
# Spread Z-score
# ---------------------------------------------------------------------------


def spread_zscore(
    spread: pd.Series | np.ndarray,
    window: int = 20,
) -> pd.Series:
    """Rolling Z-score of a spread series.

    Z_t = (spread_t - μ_{t-window:t}) / σ_{t-window:t}

    Parameters
    ----------
    spread : array-like, length T
    window : int

    Returns
    -------
    pd.Series, same index as spread.
    """
    s = pd.Series(spread) if not isinstance(spread, pd.Series) else spread
    roll_mean = s.rolling(window, min_periods=2).mean()
    roll_std = s.rolling(window, min_periods=2).std()
    z = (s - roll_mean) / (roll_std + 1e-15)
    return z


# ---------------------------------------------------------------------------
# Regime-conditional spread Z-score
# ---------------------------------------------------------------------------


def regime_spread_zscore(
    spread: pd.Series | np.ndarray,
    posteriors: np.ndarray,
    regime_params: list[RegimeSpreadParams],
) -> pd.Series:
    """Posterior-blended Z-score: normalise by blended regime mean/std.

    Z_t = (spread_t - Σ_k p_{tk} * μ_k) / (Σ_k p_{tk} * σ_k + ε)

    Parameters
    ----------
    spread : array-like, length T
    posteriors : np.ndarray, shape (T, K)
    regime_params : list[RegimeSpreadParams]

    Returns
    -------
    pd.Series
    """
    s = np.asarray(spread, dtype=float)
    idx = spread.index if isinstance(spread, pd.Series) else pd.RangeIndex(len(s))
    T, K = posteriors.shape
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)

    blended_mean = np.zeros(T)
    blended_std = np.zeros(T)
    for k, rp in enumerate(regime_params):
        blended_mean += P_norm[:, k] * rp.spread_mean
        blended_std += P_norm[:, k] * rp.spread_std

    z = (s - blended_mean) / (blended_std + 1e-15)
    return pd.Series(z, index=idx)


# ---------------------------------------------------------------------------
# Half-life table
# ---------------------------------------------------------------------------


def regime_half_lives(
    result: RegimeCointegrationResult,
) -> pd.DataFrame:
    """Tabulate half-lives per regime.

    Returns
    -------
    pd.DataFrame
        Columns: ``regime``, ``half_life``, ``eff_obs``, ``beta``.
    """
    rows = [
        {
            "regime": rp.regime,
            "half_life": rp.half_life,
            "eff_obs": rp.eff_obs,
            "beta": rp.beta,
        }
        for rp in result.regime_params
    ]
    return pd.DataFrame(rows)
