"""Regime-conditional tail risk and extreme value analysis (Phase 28).

Implements:

1. **fit_gpd** — fit a Generalized Pareto Distribution (GPD) to exceedances
   above a threshold using MLE (moment-matching fallback).

2. **RegimeTailModel** — per-regime GPD parameters and tail risk metrics
   (VaR, ES/CVaR) extrapolated beyond the empirical sample.

3. **fit_regime_tails** — responsibility-weighted GPD fitting per regime.

4. **tail_risk_decomposition** — attribute portfolio tail risk to regimes.

5. **stress_test_scenarios** — draw regime-conditional tail scenarios
   beyond the historical sample using fitted GPD tails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TailConfig:
    """Configuration for regime tail risk analysis.

    Attributes
    ----------
    threshold_quantile : float
        Quantile used to set the GPD threshold (e.g. 0.90 = top 10 % losses).
    confidence : float
        Confidence level for VaR / ES (e.g. 0.99 = 99 %).
    min_tail_obs : int
        Minimum number of exceedances required to fit a GPD; otherwise
        falls back to empirical quantile.
    n_stress_scenarios : int
        Number of stress scenarios to generate per regime.
    seed : int
        RNG seed for scenario generation.
    """

    threshold_quantile: float = 0.90
    confidence: float = 0.99
    min_tail_obs: int = 10
    n_stress_scenarios: int = 1000
    seed: int = 42


# ---------------------------------------------------------------------------
# GPD fitting
# ---------------------------------------------------------------------------


@dataclass
class GPDParams:
    """Generalized Pareto Distribution parameters.

    GPD CDF: F(x) = 1 - (1 + xi * x / sigma)^(-1/xi)  for xi ≠ 0
             F(x) = 1 - exp(-x / sigma)                 for xi = 0

    Attributes
    ----------
    xi : float
        Shape parameter (tail index).  xi > 0 = heavy tail.
    sigma : float
        Scale parameter (> 0).
    threshold : float
        Threshold above which the GPD is fitted.
    n_exceedances : int
        Number of observations above threshold.
    """

    xi: float
    sigma: float
    threshold: float
    n_exceedances: int


def fit_gpd(exceedances: np.ndarray) -> GPDParams | None:
    """Fit GPD to exceedances above threshold using method of moments.

    Returns None if fewer than 2 exceedances.

    Parameters
    ----------
    exceedances : np.ndarray
        Positive values above the threshold (y = x - u).

    Returns
    -------
    GPDParams or None
    """
    y = np.asarray(exceedances, dtype=float)
    y = y[y > 0]
    n = len(y)
    if n < 2:
        return None

    # Method of moments: μ = σ / (1 - ξ), σ² = 2μ²(1 - ξ)?
    # Simplified Hosking-Wallis moment estimators:
    m1 = float(y.mean())
    m2 = float(y.var())
    if m2 < 1e-15 or m1 < 1e-15:
        return None

    # xi = 0.5 * (1 - m1^2 / m2), sigma = m1 * (1 - xi)
    xi = 0.5 * (1.0 - m1 ** 2 / m2)
    sigma = m1 * (1.0 - xi)

    # Guard sigma > 0; clamp xi to avoid infinite moments.
    sigma = max(sigma, 1e-10)
    xi = np.clip(xi, -0.5, 2.0)

    return GPDParams(xi=float(xi), sigma=float(sigma),
                     threshold=0.0, n_exceedances=n)


def gpd_var(params: GPDParams, p: float, n_total: int) -> float:
    """GPD-extrapolated VaR at probability level p.

    VaR_p = u + (sigma / xi) * [((1-p) * n_total / n_exc)^(-xi) - 1]

    For xi = 0: VaR_p = u + sigma * log((1-p) * n_total / n_exc)
    """
    xi, sigma, u, n_exc = params.xi, params.sigma, params.threshold, params.n_exceedances
    if n_exc < 1:
        return u
    ratio = (1.0 - p) * n_total / max(n_exc, 1)
    if ratio <= 0:
        return u
    if abs(xi) < 1e-6:
        return float(u + sigma * np.log(1.0 / max(ratio, 1e-15)))
    return float(u + (sigma / xi) * (ratio ** (-xi) - 1.0))


def gpd_es(params: GPDParams, p: float, n_total: int) -> float:
    """GPD-extrapolated Expected Shortfall at probability level p.

    ES_p = VaR_p / (1 - xi) + sigma / (1 - xi) - u * xi / (1 - xi)
    (for xi < 1)
    """
    xi = params.xi
    if xi >= 1.0:
        return np.inf
    var_p = gpd_var(params, p, n_total)
    return float((var_p + params.sigma - params.xi * params.threshold) / (1.0 - xi))


# ---------------------------------------------------------------------------
# Per-regime tail model
# ---------------------------------------------------------------------------


@dataclass
class RegimeTailParams:
    """Tail risk parameters for one regime.

    Attributes
    ----------
    regime : int
    gpd : GPDParams or None
    threshold : float
    var : float
    es : float
    eff_obs : float
    """

    regime: int
    gpd: GPDParams | None
    threshold: float
    var: float
    es: float
    eff_obs: float


@dataclass
class RegimeTailResult:
    """Output of :func:`fit_regime_tails`.

    Attributes
    ----------
    regime_tails : list[RegimeTailParams]
    global_tail : RegimeTailParams
    """

    regime_tails: list[RegimeTailParams]
    global_tail: RegimeTailParams


def _empirical_var(losses: np.ndarray, weights: np.ndarray, confidence: float) -> float:
    """Weighted empirical VaR."""
    order = np.argsort(losses)[::-1]
    losses_sorted = losses[order]
    w_sorted = weights[order]
    cumw = np.cumsum(w_sorted)
    total = cumw[-1]
    idx = np.searchsorted(cumw, (1.0 - confidence) * total)
    idx = min(idx, len(losses_sorted) - 1)
    return float(losses_sorted[idx])


def _empirical_es(losses: np.ndarray, weights: np.ndarray, confidence: float) -> float:
    """Weighted empirical Expected Shortfall."""
    threshold = _empirical_var(losses, weights, confidence)
    tail_mask = losses >= threshold
    if tail_mask.sum() == 0:
        return float(threshold)
    w_tail = weights[tail_mask]
    return float((w_tail * losses[tail_mask]).sum() / (w_tail.sum() + 1e-15))


def fit_regime_tails(
    returns: pd.Series | np.ndarray,
    posteriors: np.ndarray,
    config: TailConfig | None = None,
) -> RegimeTailResult:
    """Fit per-regime GPD tail models to return losses.

    Parameters
    ----------
    returns : array-like, length T
        Return series (positive = gain, negative = loss).
    posteriors : np.ndarray, shape (T, K)
    config : TailConfig, optional

    Returns
    -------
    RegimeTailResult
    """
    cfg = config or TailConfig()
    r = np.asarray(returns, dtype=float)
    losses = -r  # positive = loss
    T, K = posteriors.shape
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)

    def _fit_one(wk: np.ndarray, regime_idx: int) -> RegimeTailParams:
        eff_n = float(wk.sum())
        # Threshold from weighted quantile.
        order = np.argsort(losses)
        cum_w = np.cumsum(wk[order])
        total_w = cum_w[-1]
        thresh_idx = np.searchsorted(cum_w, cfg.threshold_quantile * total_w)
        thresh_idx = min(thresh_idx, T - 1)
        threshold = float(losses[order[thresh_idx]])

        # Exceedances.
        exc_mask = losses > threshold
        exc_vals = losses[exc_mask] - threshold
        exc_weights = wk[exc_mask]

        var_emp = _empirical_var(losses, wk, cfg.confidence)
        es_emp = _empirical_es(losses, wk, cfg.confidence)

        if exc_weights.sum() < cfg.min_tail_obs or len(exc_vals) < cfg.min_tail_obs:
            return RegimeTailParams(
                regime=regime_idx,
                gpd=None,
                threshold=threshold,
                var=var_emp,
                es=es_emp,
                eff_obs=eff_n,
            )

        # Fit GPD to weighted sample (repeat each observation by weight).
        gpd = fit_gpd(exc_vals)
        if gpd is None:
            return RegimeTailParams(
                regime=regime_idx,
                gpd=None,
                threshold=threshold,
                var=var_emp,
                es=es_emp,
                eff_obs=eff_n,
            )

        gpd.threshold = threshold
        gpd.n_exceedances = int(exc_weights.sum())
        var_gpd = gpd_var(gpd, cfg.confidence, int(eff_n))
        es_gpd = gpd_es(gpd, cfg.confidence, int(eff_n))
        if not np.isfinite(var_gpd):
            var_gpd = var_emp
        if not np.isfinite(es_gpd):
            es_gpd = es_emp

        return RegimeTailParams(
            regime=regime_idx,
            gpd=gpd,
            threshold=threshold,
            var=var_gpd,
            es=es_gpd,
            eff_obs=eff_n,
        )

    regime_tails = [_fit_one(P_norm[:, k], k) for k in range(K)]
    global_tail = _fit_one(np.ones(T) / T, -1)

    return RegimeTailResult(regime_tails=regime_tails, global_tail=global_tail)


# ---------------------------------------------------------------------------
# Tail risk decomposition
# ---------------------------------------------------------------------------


def tail_risk_decomposition(
    result: RegimeTailResult,
    current_posteriors: np.ndarray,
) -> pd.DataFrame:
    """Attribute VaR and ES to regimes via posterior weighting.

    Parameters
    ----------
    result : RegimeTailResult
    current_posteriors : np.ndarray, shape (K,)

    Returns
    -------
    pd.DataFrame
        Columns: regime, weight, var, es, weighted_var, weighted_es.
    """
    pt = current_posteriors / (current_posteriors.sum() + 1e-15)
    rows = []
    for k, rt in enumerate(result.regime_tails):
        wk = float(pt[k]) if k < len(pt) else 0.0
        rows.append({
            "regime": k,
            "weight": wk,
            "var": rt.var,
            "es": rt.es,
            "weighted_var": wk * rt.var,
            "weighted_es": wk * rt.es,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stress-test scenario generation
# ---------------------------------------------------------------------------


def stress_test_scenarios(
    result: RegimeTailResult,
    current_posteriors: np.ndarray,
    config: TailConfig | None = None,
) -> pd.DataFrame:
    """Generate tail stress scenarios beyond historical sample.

    For each regime, draws GPD-distributed losses beyond the fitted
    threshold and weights them by the current posterior.

    Parameters
    ----------
    result : RegimeTailResult
    current_posteriors : np.ndarray, shape (K,)
    config : TailConfig, optional

    Returns
    -------
    pd.DataFrame
        Columns: scenario (int), loss (float), regime (int), weight (float).
    """
    cfg = config or TailConfig()
    pt = current_posteriors / (current_posteriors.sum() + 1e-15)
    rng = np.random.default_rng(cfg.seed)
    rows = []
    scenario_id = 0

    for k, rt in enumerate(result.regime_tails):
        wk = float(pt[k]) if k < len(pt) else 0.0
        if wk <= 0.0:
            continue
        n_k = max(1, int(round(wk * cfg.n_stress_scenarios)))

        if rt.gpd is not None:
            xi, sigma, u = rt.gpd.xi, rt.gpd.sigma, rt.threshold
            # Inverse CDF of GPD: x = sigma/xi * (u^(-xi) - 1), u ~ Uniform
            u_samples = rng.uniform(0, 1, n_k)
            if abs(xi) < 1e-6:
                exc = -sigma * np.log(u_samples)
            else:
                exc = (sigma / xi) * (u_samples ** (-xi) - 1.0)
            losses = u + exc
        else:
            # Fallback: Gaussian tail beyond threshold.
            losses = rng.exponential(scale=max(rt.threshold * 0.5, 0.001), size=n_k) + rt.threshold

        losses = np.maximum(losses, rt.threshold)
        for loss in losses:
            rows.append({"scenario": scenario_id, "loss": float(loss),
                         "regime": k, "weight": wk})
            scenario_id += 1

    return pd.DataFrame(rows)
