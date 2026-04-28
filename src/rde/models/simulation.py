"""Stochastic scenario simulation for regime stress-testing (Phase 21).

Generates Monte Carlo return paths from regime-calibrated models:

1. **GaussianRegimeSimulator** — draws IID samples from each regime's
   calibrated Gaussian emission (mean, covariance).

2. **GARCHRegimeSimulator** — generates correlated paths under
   regime-switching GARCH(1,1) dynamics, respecting each regime's
   (ω, α, β, μ) parameters from :class:`rde.models.garch_hmm.GARCHParams`.

3. **ScenarioAggregator** — runs the portfolio through simulated paths
   and summarises the distribution of cumulative returns, max drawdown,
   and Sharpe ratio across paths.

All simulators are reproducible (seed-controlled) and return DataFrames
indexed by (path, time).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """Parameters controlling Monte Carlo simulation.

    Attributes
    ----------
    n_paths : int
        Number of independent sample paths.
    horizon : int
        Simulation length in bars.
    seed : int
        Master RNG seed.  Path i uses ``seed + i``.
    """

    n_paths: int = 100
    horizon: int = 252
    seed: int = 42


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """Output of a regime simulator.

    Attributes
    ----------
    returns : np.ndarray
        Shape (n_paths, horizon).  Simulated log returns.
    cum_returns : np.ndarray
        Shape (n_paths, horizon).  Cumulative log returns (log-price change).
    regime_sequence : np.ndarray
        Shape (n_paths, horizon).  Integer regime labels per bar.
    path_stats : pd.DataFrame
        One row per path; columns include ``sharpe``, ``max_drawdown``,
        ``total_return``, ``ann_vol``.
    """

    returns: np.ndarray
    cum_returns: np.ndarray
    regime_sequence: np.ndarray
    path_stats: pd.DataFrame


# ---------------------------------------------------------------------------
# Gaussian regime simulator
# ---------------------------------------------------------------------------


def simulate_gaussian_regimes(
    means: np.ndarray,
    covars: np.ndarray,
    transmat: np.ndarray,
    startprob: np.ndarray,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    """Simulate multivariate return paths from a Gaussian HMM.

    At each step, the regime evolves according to ``transmat``, and a
    return vector is drawn from N(μ_k, Σ_k).  The scalar return per path
    is the mean of the d dimensions (suitable for single-asset simulations
    where d = n_features).

    Parameters
    ----------
    means : np.ndarray, shape (K, d)
    covars : np.ndarray, shape (K, d, d)
    transmat : np.ndarray, shape (K, K)
        Row-stochastic transition matrix.
    startprob : np.ndarray, shape (K,)
    config : SimulationConfig, optional

    Returns
    -------
    SimulationResult
    """
    cfg = config or SimulationConfig()
    K, d = means.shape
    rng = np.random.default_rng(cfg.seed)

    all_returns = np.zeros((cfg.n_paths, cfg.horizon))
    all_regimes = np.zeros((cfg.n_paths, cfg.horizon), dtype=int)

    for p in range(cfg.n_paths):
        path_rng = np.random.default_rng(cfg.seed + p)

        # Initial regime.
        state = int(path_rng.choice(K, p=startprob))
        for t in range(cfg.horizon):
            all_regimes[p, t] = state
            sample = path_rng.multivariate_normal(means[state], covars[state])
            all_returns[p, t] = sample.mean()  # scalar for 1-asset case
            state = int(path_rng.choice(K, p=transmat[state]))

    cum_returns = np.cumsum(all_returns, axis=1)
    path_stats = _compute_path_stats(all_returns)

    return SimulationResult(
        returns=all_returns,
        cum_returns=cum_returns,
        regime_sequence=all_regimes,
        path_stats=path_stats,
    )


# ---------------------------------------------------------------------------
# GARCH regime simulator
# ---------------------------------------------------------------------------


def simulate_garch_regimes(
    garch_params: list,  # list[GARCHParams] — avoid hard import cycle
    transmat: np.ndarray,
    startprob: np.ndarray,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    """Simulate univariate return paths under regime-switching GARCH(1,1).

    Each regime k has GARCH parameters (ω_k, α_k, β_k, μ_k).
    Within a regime, the conditional variance evolves as:
        σ²_t = ω_k + α_k * ε²_{t-1} + β_k * σ²_{t-1}
    Returns are: r_t = μ_k + σ_t * z_t,  z_t ~ N(0, 1)

    Parameters
    ----------
    garch_params : list[GARCHParams]
        Fitted GARCH parameters per regime.
    transmat : np.ndarray, shape (K, K)
    startprob : np.ndarray, shape (K,)
    config : SimulationConfig, optional

    Returns
    -------
    SimulationResult
    """
    cfg = config or SimulationConfig()
    K = len(garch_params)

    all_returns = np.zeros((cfg.n_paths, cfg.horizon))
    all_regimes = np.zeros((cfg.n_paths, cfg.horizon), dtype=int)

    for p in range(cfg.n_paths):
        path_rng = np.random.default_rng(cfg.seed + p)

        # Initialise GARCH state: unconditional variance per regime.
        sigma2 = np.array([
            float(gp.omega / (1.0 - gp.alpha - gp.beta + 1e-10))
            for gp in garch_params
        ])
        eps = np.zeros(K)  # last shock per regime

        state = int(path_rng.choice(K, p=startprob))

        for t in range(cfg.horizon):
            all_regimes[p, t] = state
            gp = garch_params[state]

            # Update GARCH variance for current regime.
            sigma2[state] = (
                gp.omega
                + gp.alpha * eps[state] ** 2
                + gp.beta * sigma2[state]
            )
            sigma2[state] = max(sigma2[state], 1e-12)

            z = float(path_rng.standard_normal())
            r = gp.mu + np.sqrt(sigma2[state]) * z
            eps[state] = r - gp.mu

            all_returns[p, t] = r
            state = int(path_rng.choice(K, p=transmat[state]))

    cum_returns = np.cumsum(all_returns, axis=1)
    path_stats = _compute_path_stats(all_returns)

    return SimulationResult(
        returns=all_returns,
        cum_returns=cum_returns,
        regime_sequence=all_regimes,
        path_stats=path_stats,
    )


# ---------------------------------------------------------------------------
# Scenario aggregation
# ---------------------------------------------------------------------------


@dataclass
class ScenarioStats:
    """Summary statistics across Monte Carlo paths.

    Attributes
    ----------
    mean_total_return : float
    std_total_return : float
    var_95 : float
        5th-percentile total return (Value-at-Risk).
    cvar_95 : float
        Mean of paths below the 5th percentile (CVaR).
    mean_max_drawdown : float
    p95_max_drawdown : float
    mean_sharpe : float
    regime_freq : np.ndarray
        Shape (K,).  Fraction of time spent in each regime.
    """

    mean_total_return: float
    std_total_return: float
    var_95: float
    cvar_95: float
    mean_max_drawdown: float
    p95_max_drawdown: float
    mean_sharpe: float
    regime_freq: np.ndarray


def aggregate_scenarios(result: SimulationResult) -> ScenarioStats:
    """Compute distribution statistics from a SimulationResult.

    Parameters
    ----------
    result : SimulationResult

    Returns
    -------
    ScenarioStats
    """
    total_ret = result.cum_returns[:, -1]
    mean_tr = float(total_ret.mean())
    std_tr = float(total_ret.std())

    var_95 = float(np.percentile(total_ret, 5))
    tail = total_ret[total_ret <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95

    mdd_per_path = np.array([
        _path_max_drawdown(result.returns[p]) for p in range(len(result.returns))
    ])
    mean_mdd = float(mdd_per_path.mean())
    p95_mdd = float(np.percentile(mdd_per_path, 95))
    mean_sharpe = float(result.path_stats["sharpe"].mean())

    K = result.regime_sequence.max() + 1
    counts = np.bincount(result.regime_sequence.ravel(), minlength=K)
    regime_freq = counts / counts.sum()

    return ScenarioStats(
        mean_total_return=mean_tr,
        std_total_return=std_tr,
        var_95=var_95,
        cvar_95=cvar_95,
        mean_max_drawdown=mean_mdd,
        p95_max_drawdown=p95_mdd,
        mean_sharpe=mean_sharpe,
        regime_freq=regime_freq,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _path_max_drawdown(returns: np.ndarray) -> float:
    cum = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / (peak + 1e-15)
    return float(-dd.min())


def _compute_path_stats(returns: np.ndarray) -> pd.DataFrame:
    """Compute per-path statistics.  ``returns`` shape (n_paths, horizon)."""
    n_paths, horizon = returns.shape
    rows = []
    for p in range(n_paths):
        r = returns[p]
        mu = r.mean()
        vol = r.std(ddof=1)
        ann_factor = horizon  # horizon bars per path, annualise by horizon
        sharpe = mu / (vol + 1e-12) * np.sqrt(ann_factor)
        total = float(np.exp(r.sum()) - 1.0)  # log returns → simple
        mdd = _path_max_drawdown(r)
        rows.append(
            {
                "path": p,
                "total_return": total,
                "ann_vol": vol * np.sqrt(ann_factor),
                "sharpe": sharpe,
                "max_drawdown": mdd,
            }
        )
    return pd.DataFrame(rows).set_index("path")
