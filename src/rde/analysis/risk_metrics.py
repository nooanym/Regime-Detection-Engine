"""Regime-conditional risk analytics (Phase 16).

Provides per-regime Value-at-Risk (VaR), Conditional VaR (CVaR / Expected
Shortfall), drawdown analysis, and summary risk tables.  Functions accept
either hard Viterbi state assignments or soft responsibility-weighted
posteriors for all key metrics.

All ratio metrics (Sharpe, Sortino, Calmar) are annualised using
``config.annualisation_factor`` — default 8760 for hourly data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class RegimeRiskConfig:
    """Configuration for regime risk metric computation.

    Attributes
    ----------
    confidence : float
        VaR / CVaR confidence level.  0.95 means the 5th-percentile loss
        (worst 5 % of observations).
    min_obs : int
        Minimum effective observations required.  Regimes below this
        threshold return NaN for all metrics.
    annualisation_factor : float
        Multiplier applied to per-bar mean / vol to annualise.
        Default 8760 = 24 * 365 (hourly bars).
    mar : float
        Minimum acceptable return (per bar) for Sortino ratio.
    """

    confidence: float = 0.95
    min_obs: int = 20
    annualisation_factor: float = 8760.0
    mar: float = 0.0


@dataclass
class RegimeRiskResult:
    """Per-regime risk metrics.

    Attributes
    ----------
    n_states : int
    metrics : pd.DataFrame
        One row per state, indexed by state index.
        Columns: ``mean_ann``, ``vol_ann``, ``sharpe``, ``sortino``,
        ``var``, ``cvar``, ``max_drawdown``, ``calmar``, ``n_obs``.
    drawdown_series : dict[int, pd.Series]
        Running drawdown (0 to -inf) for observations assigned to each
        state under hard decoding.  Empty if states were not supplied.
    """

    n_states: int
    metrics: pd.DataFrame
    drawdown_series: dict[int, pd.Series] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _var(returns: np.ndarray, confidence: float) -> float:
    """Historical VaR: negative of the (1-confidence) quantile of returns."""
    if len(returns) == 0:
        return np.nan
    return float(-np.quantile(returns, 1.0 - confidence))


def _cvar(returns: np.ndarray, confidence: float) -> float:
    """Historical CVaR (Expected Shortfall): mean of losses beyond VaR."""
    if len(returns) == 0:
        return np.nan
    threshold = np.quantile(returns, 1.0 - confidence)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return float(-threshold)
    return float(-tail.mean())


def _max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown of a return series (as a positive fraction)."""
    if len(returns) == 0:
        return np.nan
    cum = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    return float(-dd.min())


def _drawdown_series(returns: np.ndarray) -> np.ndarray:
    """Running drawdown series (non-positive values)."""
    if len(returns) == 0:
        return np.array([])
    cum = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cum)
    return (cum - running_max) / running_max


def _weighted_mean_var(
    returns: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Weighted mean and variance.  Returns (nan, nan) if weight sum is zero."""
    w_sum = weights.sum()
    if w_sum < 1e-12:
        return np.nan, np.nan
    mu = (weights * returns).sum() / w_sum
    var = (weights * (returns - mu) ** 2).sum() / w_sum
    return float(mu), float(var)


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    """Weighted quantile via cumulative weight interpolation."""
    if weights.sum() < 1e-12:
        return np.nan
    sorter = np.argsort(values)
    sv = values[sorter]
    sw = weights[sorter]
    cumw = np.cumsum(sw) / sw.sum()
    idx = int(np.searchsorted(cumw, quantile))
    idx = min(idx, len(sv) - 1)
    return float(sv[idx])


def _sortino(
    returns: np.ndarray, mar: float, ann: float
) -> float:
    """Annualised Sortino ratio."""
    if len(returns) < 2:
        return np.nan
    excess = returns - mar
    mean_excess = excess.mean()
    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean())
    if downside < 1e-12:
        return np.inf if mean_excess > 0 else np.nan
    return float(mean_excess / downside * np.sqrt(ann))


def _calmar(
    returns: np.ndarray, ann: float
) -> float:
    """Annualised Calmar ratio (annualised return / max drawdown)."""
    if len(returns) < 2:
        return np.nan
    ann_ret = returns.mean() * ann
    mdd = _max_drawdown(returns)
    if mdd < 1e-12:
        return np.inf if ann_ret > 0 else np.nan
    return float(ann_ret / mdd)


# ---------------------------------------------------------------------------
# Hard-state per-regime risk
# ---------------------------------------------------------------------------


def compute_regime_risk(
    returns: pd.Series,
    states: np.ndarray,
    config: RegimeRiskConfig | None = None,
) -> RegimeRiskResult:
    """Compute risk metrics per regime using hard Viterbi state assignments.

    Parameters
    ----------
    returns : pd.Series
        Per-bar log returns, length T.
    states : np.ndarray
        Integer state array, shape (T,).
    config : RegimeRiskConfig, optional

    Returns
    -------
    RegimeRiskResult
    """
    cfg = config or RegimeRiskConfig()
    ann = cfg.annualisation_factor
    r = np.asarray(returns, dtype=float)
    s = np.asarray(states, dtype=int)
    if len(r) != len(s):
        raise ValueError(
            f"returns length {len(r)} != states length {len(s)}"
        )

    n_states = int(s.max()) + 1
    rows: list[dict] = []
    dd_series: dict[int, pd.Series] = {}

    for k in range(n_states):
        mask = s == k
        rk = r[mask]
        n = len(rk)
        if n < cfg.min_obs:
            rows.append(_nan_row(k, n))
            continue

        mu = rk.mean()
        vol = rk.std(ddof=1)
        mu_ann = mu * ann
        vol_ann = vol * np.sqrt(ann)
        sharpe = mu_ann / vol_ann if vol_ann > 1e-12 else np.nan
        sortino = _sortino(rk, cfg.mar, ann)
        var = _var(rk, cfg.confidence)
        cvar = _cvar(rk, cfg.confidence)
        mdd = _max_drawdown(rk)
        calmar = _calmar(rk, ann)

        rows.append(
            {
                "state": k,
                "n_obs": n,
                "mean_ann": mu_ann,
                "vol_ann": vol_ann,
                "sharpe": sharpe,
                "sortino": sortino,
                "var": var,
                "cvar": cvar,
                "max_drawdown": mdd,
                "calmar": calmar,
            }
        )

        idx_k = returns.index[mask] if hasattr(returns, "index") else None
        dd_vals = _drawdown_series(rk)
        dd_series[k] = pd.Series(
            dd_vals, index=idx_k, name=f"drawdown_state_{k}"
        )

    metrics = pd.DataFrame(rows).set_index("state")
    return RegimeRiskResult(
        n_states=n_states, metrics=metrics, drawdown_series=dd_series
    )


# ---------------------------------------------------------------------------
# Soft (responsibility-weighted) per-regime risk
# ---------------------------------------------------------------------------


def compute_regime_risk_weighted(
    returns: pd.Series,
    posteriors: np.ndarray,
    config: RegimeRiskConfig | None = None,
) -> RegimeRiskResult:
    """Compute risk metrics per regime using responsibility-weighted posteriors.

    Weighted VaR and CVaR use a weighted-quantile estimator.  Weighted Sharpe,
    Sortino and Calmar use responsibility-weighted mean and variance.

    Parameters
    ----------
    returns : pd.Series
        Per-bar log returns, length T.
    posteriors : np.ndarray
        Shape (T, K).  Rows must sum to approximately 1 (will be normalised).
    config : RegimeRiskConfig, optional

    Returns
    -------
    RegimeRiskResult
        ``drawdown_series`` is empty — drawdown is only computed for hard states.
    """
    cfg = config or RegimeRiskConfig()
    ann = cfg.annualisation_factor
    r = np.asarray(returns, dtype=float)
    P = np.asarray(posteriors, dtype=float)
    T, K = P.shape
    if len(r) != T:
        raise ValueError(
            f"returns length {len(r)} != posteriors rows {T}"
        )

    P_norm = P / (P.sum(axis=1, keepdims=True) + 1e-15)
    rows: list[dict] = []

    for k in range(K):
        wk = P_norm[:, k]
        eff_n = wk.sum()
        if eff_n < cfg.min_obs:
            rows.append(_nan_row(k, float(eff_n)))
            continue

        mu, variance = _weighted_mean_var(r, wk)
        vol = float(np.sqrt(variance)) if variance >= 0 else np.nan
        mu_ann = mu * ann
        vol_ann = vol * np.sqrt(ann) if not np.isnan(vol) else np.nan
        sharpe = mu_ann / vol_ann if (vol_ann and vol_ann > 1e-12) else np.nan

        # Weighted VaR / CVaR
        var_threshold = _weighted_quantile(r, wk, 1.0 - cfg.confidence)
        var = float(-var_threshold)
        tail_mask = r <= var_threshold
        if tail_mask.any():
            w_tail = wk[tail_mask]
            cvar_val = float(
                -(r[tail_mask] * w_tail).sum() / (w_tail.sum() + 1e-15)
            )
        else:
            cvar_val = var

        # Weighted Sortino: use soft-weighted semideviance
        excess = r - cfg.mar
        w_mu_excess = (wk * excess).sum() / eff_n
        downside_var = (wk * np.minimum(excess, 0.0) ** 2).sum() / eff_n
        downside_std = float(np.sqrt(downside_var))
        sortino = (
            float(w_mu_excess / downside_std * np.sqrt(ann))
            if downside_std > 1e-12
            else np.nan
        )

        # Max drawdown — use hard-assignment to highest-weight state for each bar
        hard_k = np.argmax(P_norm, axis=1) == k
        rk_hard = r[hard_k]
        mdd = _max_drawdown(rk_hard) if len(rk_hard) >= cfg.min_obs else np.nan
        calmar = _calmar(rk_hard, ann) if len(rk_hard) >= cfg.min_obs else np.nan

        rows.append(
            {
                "state": k,
                "n_obs": float(eff_n),
                "mean_ann": mu_ann,
                "vol_ann": vol_ann,
                "sharpe": sharpe,
                "sortino": sortino,
                "var": var,
                "cvar": cvar_val,
                "max_drawdown": mdd,
                "calmar": calmar,
            }
        )

    metrics = pd.DataFrame(rows).set_index("state")
    return RegimeRiskResult(n_states=K, metrics=metrics, drawdown_series={})


# ---------------------------------------------------------------------------
# Regime-level risk divergence
# ---------------------------------------------------------------------------


def regime_risk_divergence(result: RegimeRiskResult) -> pd.DataFrame:
    """Pairwise absolute differences in key risk metrics across regimes.

    Returns a long-form DataFrame with columns
    ``[state_a, state_b, metric, abs_diff]`` for each unique pair.
    Useful for quantifying how distinct regimes are in risk terms.

    Parameters
    ----------
    result : RegimeRiskResult

    Returns
    -------
    pd.DataFrame
    """
    m = result.metrics
    metrics_cols = ["mean_ann", "vol_ann", "var", "cvar", "max_drawdown"]
    existing = [c for c in metrics_cols if c in m.columns]
    states = list(m.index)
    rows = []
    for i, a in enumerate(states):
        for b in states[i + 1 :]:
            for col in existing:
                va, vb = m.loc[a, col], m.loc[b, col]
                if np.isnan(va) or np.isnan(vb):
                    diff = np.nan
                else:
                    diff = abs(float(va) - float(vb))
                rows.append(
                    {"state_a": a, "state_b": b, "metric": col, "abs_diff": diff}
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nan_row(state: int, n: float) -> dict:
    return {
        "state": state,
        "n_obs": n,
        "mean_ann": np.nan,
        "vol_ann": np.nan,
        "sharpe": np.nan,
        "sortino": np.nan,
        "var": np.nan,
        "cvar": np.nan,
        "max_drawdown": np.nan,
        "calmar": np.nan,
    }
