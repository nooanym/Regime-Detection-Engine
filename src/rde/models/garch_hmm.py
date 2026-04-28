"""Regime-switching GARCH(1,1) fitted within each HMM hidden state.

For each state k the conditional variance follows:

    σ²_t,k = ω_k + α_k * ε²_{t-1,k} + β_k * σ²_{t-1,k}

where ε²_{t-1,k} = (r_{t-1} - μ_k)² is the squared de-meaned return and the
recursion is initialised at σ²_{0,k} = Var(r).  Parameters are estimated by
maximising the responsibility-weighted Gaussian log-likelihood:

    -L_k = Σ_t γ_t(k) * [0.5 * log(σ²_t,k) + 0.5 * (r_t - μ_k)² / σ²_t,k]

subject to ω_k > 0, α_k ≥ 0, β_k ≥ 0, α_k + β_k < 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GARCHParams:
    """GARCH(1,1) parameters for one hidden state.

    Attributes
    ----------
    state : int
        State index (0-based).
    omega : float
        Intercept term ω > 0.
    alpha : float
        ARCH coefficient α ≥ 0.
    beta : float
        GARCH coefficient β ≥ 0.
    mu : float
        Responsibility-weighted mean return for this state.
    persistence : float
        α + β.  Must be < 1 for covariance stationarity.
    unconditional_vol : float
        Annualised unconditional volatility sqrt(ω / (1 - α - β)) * sqrt(bars_per_year).
        Set to NaN when persistence ≥ 1.
    """

    state: int
    omega: float
    alpha: float
    beta: float
    mu: float
    persistence: float
    unconditional_vol: float


@dataclass
class GARCHFit:
    """Result of :func:`fit_regime_garch`.

    Attributes
    ----------
    params : list[GARCHParams]
        One entry per state, sorted by state index.
    cond_vol : pd.DataFrame
        Shape (T, n_states).  Conditional standard deviation for each state at
        each bar.  Column names are integer state indices.
    weighted_cond_vol : pd.Series
        Shape (T,).  Regime-averaged conditional volatility:
        ``Σ_k γ_t(k) * σ_t,k``.
    """

    params: list[GARCHParams]
    cond_vol: pd.DataFrame
    weighted_cond_vol: pd.Series


# ---------------------------------------------------------------------------
# Core recursion
# ---------------------------------------------------------------------------


def garch_conditional_vol(
    returns: np.ndarray,
    params: GARCHParams,
) -> np.ndarray:
    """Compute the GARCH(1,1) conditional standard-deviation series.

    The recursion is:

        σ²_0   = Var(returns)      (unconditional initialisation)
        σ²_t   = ω + α*(r_{t-1} - μ)² + β*σ²_{t-1}   for t ≥ 1

    Parameters
    ----------
    returns : np.ndarray
        Shape (T,).  Unscaled log returns.  Must not contain NaN.
    params : GARCHParams
        Fitted (or hypothetical) GARCH parameters for this state.

    Returns
    -------
    np.ndarray
        Shape (T,).  Conditional standard deviation at each bar.

    Notes
    -----
    σ² is clamped to a minimum of 1e-12 at every step to prevent numerical
    issues when ω is very small.  The returned array therefore contains only
    strictly positive values.
    """
    T = len(returns)
    omega = params.omega
    alpha = params.alpha
    beta = params.beta
    mu = params.mu

    sigma2 = np.empty(T)
    # Initialise at the unconditional (sample) variance.
    sigma2[0] = max(float(np.var(returns)), 1e-8)

    for t in range(1, T):
        eps2 = (returns[t - 1] - mu) ** 2
        var_t = omega + alpha * eps2 + beta * sigma2[t - 1]
        sigma2[t] = max(var_t, 1e-12)

    return np.sqrt(sigma2)


# ---------------------------------------------------------------------------
# Multi-step forecast
# ---------------------------------------------------------------------------


def forecast_garch_vol(
    params: GARCHParams,
    last_return: float,
    last_sigma: float,
    horizons: list[int],
) -> pd.Series:
    """Multi-step GARCH(1,1) variance forecast.

    The h-step ahead forecast is:

        σ²_{t+h|t} = ω/(1-α-β) + (α+β)^h * (σ²_t - ω/(1-α-β))

    which mean-reverts to the unconditional variance as h → ∞ when α+β < 1.

    Parameters
    ----------
    params : GARCHParams
        Fitted GARCH parameters.
    last_return : float
        r_T — the most recent log return.
    last_sigma : float
        σ_T — the most recent conditional standard deviation.
    horizons : list[int]
        Forecast horizons in bars (positive integers).

    Returns
    -------
    pd.Series
        Index = ``horizons``, values = conditional standard deviation at each
        horizon.

    Notes
    -----
    When persistence α+β ≥ 1 the unconditional variance is undefined; in that
    case the forecast is computed by iterating the recursion one step at a time
    using the last residual, which provides a sensible (though non-stationary)
    answer.

    ``last_return`` is used only in the non-stationary fallback path.
    """
    omega = params.omega
    alpha = params.alpha
    beta = params.beta
    mu = params.mu
    persistence = alpha + beta

    last_var = last_sigma ** 2

    results: dict[int, float] = {}

    if persistence < 1.0:
        uncond_var = omega / (1.0 - persistence)
        for h in horizons:
            var_h = uncond_var + (persistence ** h) * (last_var - uncond_var)
            results[h] = float(np.sqrt(max(var_h, 1e-12)))
    else:
        # Non-stationary: iterate the recursion step by step.
        logger.warning(
            "forecast_garch_vol: persistence=%.4f ≥ 1 for state %d; "
            "using iterative recursion (non-stationary).",
            persistence,
            params.state,
        )
        # For h=1 the shock is the last observed return.
        # For h>1 we assume E[ε²_{t+j}] = σ²_{t+j} (expected future residual).
        prev_var = last_var
        eps2 = (last_return - mu) ** 2
        horizon_set = sorted(horizons)
        step = 0
        for h in horizon_set:
            while step < h:
                next_var = max(omega + alpha * eps2 + beta * prev_var, 1e-12)
                # After the first step use E[ε²] = σ² (i.e. eps2 = prev_var).
                eps2 = next_var
                prev_var = next_var
                step += 1
            results[h] = float(np.sqrt(prev_var))

    return pd.Series(results, name="forecast_sigma")


# ---------------------------------------------------------------------------
# Per-state GARCH fitting
# ---------------------------------------------------------------------------


def _fit_single_state(
    returns: np.ndarray,
    gamma: np.ndarray,
    state_idx: int,
    bars_per_year: float,
    max_iter: int,
) -> GARCHParams:
    """Fit GARCH(1,1) for a single state via responsibility-weighted MLE.

    Parameters
    ----------
    returns : np.ndarray
        Shape (T,).  Unscaled log returns.
    gamma : np.ndarray
        Shape (T,).  Smoothed posterior weights for this state.
    state_idx : int
        State index (used for logging and the returned dataclass).
    bars_per_year : float
        Annualisation factor for the unconditional volatility field.
    max_iter : int
        Maximum L-BFGS-B iterations.

    Returns
    -------
    GARCHParams
        Fitted parameters.  Falls back to constant-variance if optimisation
        fails.
    """
    T = len(returns)
    total_weight = float(gamma.sum())

    # ── Responsibility-weighted mean ────────────────────────────────────────
    if total_weight < 1e-12:
        # All-zero posteriors for this state; return constant variance fallback.
        logger.warning(
            "State %d has near-zero total posterior weight (%.2e); "
            "returning constant-variance GARCH.",
            state_idx,
            total_weight,
        )
        uncond_var = float(np.var(returns)) if len(returns) > 1 else 1e-8
        return _constant_variance_params(
            state_idx, uncond_var, float(np.mean(returns)), bars_per_year
        )

    mu_k = float(np.dot(gamma, returns) / total_weight)
    uncond_var = float(np.var(returns))
    uncond_var = max(uncond_var, 1e-8)

    # ── Objective: responsibility-weighted negative log-likelihood ───────────
    def neg_log_lik(params: np.ndarray) -> float:
        omega_p, alpha_p, beta_p = params
        sigma2 = np.empty(T)
        sigma2[0] = uncond_var

        for t in range(1, T):
            eps2 = (returns[t - 1] - mu_k) ** 2
            var_t = omega_p + alpha_p * eps2 + beta_p * sigma2[t - 1]
            sigma2[t] = max(var_t, 1e-12)

        sigma2 = np.maximum(sigma2, 1e-12)
        nll = np.sum(
            gamma * (0.5 * np.log(sigma2) + 0.5 * (returns - mu_k) ** 2 / sigma2)
        )
        return float(nll)

    # ── Initial guess ────────────────────────────────────────────────────────
    x0 = np.array([uncond_var * 0.05, 0.1, 0.85])

    # ── Bounds: ω ∈ (1e-8, ∞), α ∈ [0, 0.999], β ∈ [0, 0.999] ─────────────
    bounds = [
        (1e-8, None),   # omega
        (0.0, 0.999),   # alpha
        (0.0, 0.999),   # beta
    ]

    # ── Linear constraint: α + β ≤ 0.999 ────────────────────────────────────
    lin_constraint = LinearConstraint(
        A=np.array([[0.0, 1.0, 1.0]]),
        lb=-np.inf,
        ub=0.999,
    )

    result = minimize(
        neg_log_lik,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8},
    )

    # L-BFGS-B does not support nonlinear constraints natively; enforce via
    # a second pass with SLSQP if the constraint is violated.
    if result.x[1] + result.x[2] >= 0.999:
        result2 = minimize(
            neg_log_lik,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": lambda p: 0.999 - p[1] - p[2]}],
            options={"maxiter": max_iter, "ftol": 1e-12},
        )
        if result2.success or result2.fun < result.fun:
            result = result2

    success = result.success or "Optimization terminated successfully" in result.message

    if not success:
        logger.warning(
            "GARCH optimisation for state %d did not converge (message: %s). "
            "Falling back to constant-variance model.",
            state_idx,
            result.message,
        )
        return _constant_variance_params(state_idx, uncond_var, mu_k, bars_per_year)

    omega_fit, alpha_fit, beta_fit = float(result.x[0]), float(result.x[1]), float(result.x[2])
    # Hard-clamp for safety (shouldn't be needed after SLSQP correction).
    alpha_fit = min(alpha_fit, 0.999)
    beta_fit = min(beta_fit, 0.999)
    if alpha_fit + beta_fit >= 1.0:
        scale = 0.999 / (alpha_fit + beta_fit)
        alpha_fit *= scale
        beta_fit *= scale

    persistence = alpha_fit + beta_fit
    if persistence < 1.0:
        uncond_vol_ann = float(
            np.sqrt(omega_fit / (1.0 - persistence)) * np.sqrt(bars_per_year)
        )
    else:
        uncond_vol_ann = float("nan")

    return GARCHParams(
        state=state_idx,
        omega=omega_fit,
        alpha=alpha_fit,
        beta=beta_fit,
        mu=mu_k,
        persistence=persistence,
        unconditional_vol=uncond_vol_ann,
    )


def _constant_variance_params(
    state_idx: int,
    uncond_var: float,
    mu: float,
    bars_per_year: float,
) -> GARCHParams:
    """Return a constant-variance (α=β=0) GARCHParams as a fallback.

    Parameters
    ----------
    state_idx : int
        State index.
    uncond_var : float
        Unconditional variance to use as ω.
    mu : float
        Responsibility-weighted mean return.
    bars_per_year : float
        Annualisation factor.

    Returns
    -------
    GARCHParams
    """
    uncond_vol_ann = float(np.sqrt(uncond_var) * np.sqrt(bars_per_year))
    return GARCHParams(
        state=state_idx,
        omega=uncond_var,
        alpha=0.0,
        beta=0.0,
        mu=mu,
        persistence=0.0,
        unconditional_vol=uncond_vol_ann,
    )


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------


def fit_regime_garch(
    returns: pd.Series | np.ndarray,
    posteriors: np.ndarray,
    *,
    bars_per_year: float = 8760.0,
    max_iter: int = 200,
) -> GARCHFit:
    """Fit GARCH(1,1) within each HMM state using responsibility weighting.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Shape (T,).  Raw (unscaled) log returns.  Must not contain NaN.
    posteriors : np.ndarray
        Shape (T, K).  Smoothed state posteriors from the forward-backward
        algorithm.  Rows should sum to 1 but this is not enforced.
    bars_per_year : float
        Used to annualise the unconditional volatility in :class:`GARCHParams`.
        Default 8760 for hourly data (365 * 24).
    max_iter : int
        Maximum L-BFGS-B iterations per state.

    Returns
    -------
    GARCHFit
        Fitted parameters, per-state conditional volatility, and the
        regime-averaged conditional volatility.

    Raises
    ------
    ValueError
        If ``returns`` and ``posteriors`` have incompatible lengths.
    ValueError
        If ``returns`` contains NaN values.

    Notes
    -----
    The responsibility-weighted mean μ_k = Σ_t γ_t(k) r_t / Σ_t γ_t(k) is
    computed before the GARCH optimisation so that the innovations are
    de-meaned consistently.

    Initialisation: (ω₀, α₀, β₀) = (Var(r)*0.05, 0.1, 0.85).  This is a
    standard starting point for financial hourly data.
    """
    returns_arr: np.ndarray = (
        np.asarray(returns, dtype=float)
        if not isinstance(returns, np.ndarray)
        else returns.astype(float)
    )

    index = returns.index if isinstance(returns, pd.Series) else None

    T = len(returns_arr)
    if posteriors.shape[0] != T:
        raise ValueError(
            f"returns has length {T} but posteriors has {posteriors.shape[0]} rows."
        )
    if np.any(np.isnan(returns_arr)):
        raise ValueError("returns contains NaN values; clean the series before calling fit_regime_garch.")

    K = posteriors.shape[1]
    logger.info(
        "fit_regime_garch: T=%d bars, K=%d states, bars_per_year=%.0f",
        T, K, bars_per_year,
    )

    fitted_params: list[GARCHParams] = []
    cond_vol_matrix = np.empty((T, K))

    for k in range(K):
        gamma_k = posteriors[:, k]
        gp = _fit_single_state(
            returns_arr, gamma_k, k, bars_per_year, max_iter
        )
        fitted_params.append(gp)
        cond_vol_matrix[:, k] = garch_conditional_vol(returns_arr, gp)
        logger.info(
            "  State %d: ω=%.3e  α=%.4f  β=%.4f  persistence=%.4f  "
            "uncond_vol_ann=%.2f%%",
            k,
            gp.omega,
            gp.alpha,
            gp.beta,
            gp.persistence,
            gp.unconditional_vol * 100 if not np.isnan(gp.unconditional_vol) else float("nan"),
        )

    # Regime-averaged conditional vol: Σ_k γ_t(k) * σ_t,k
    weighted_vol_arr = np.sum(posteriors * cond_vol_matrix, axis=1)

    col_names = list(range(K))
    if index is not None:
        cond_vol_df = pd.DataFrame(cond_vol_matrix, index=index, columns=col_names)
        weighted_vol_series = pd.Series(weighted_vol_arr, index=index, name="weighted_cond_vol")
    else:
        cond_vol_df = pd.DataFrame(cond_vol_matrix, columns=col_names)
        weighted_vol_series = pd.Series(weighted_vol_arr, name="weighted_cond_vol")

    return GARCHFit(
        params=fitted_params,
        cond_vol=cond_vol_df,
        weighted_cond_vol=weighted_vol_series,
    )
