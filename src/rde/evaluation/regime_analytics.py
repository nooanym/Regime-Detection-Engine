"""Per-regime performance statistics and transition forecasting.

Transforms raw regime assignments into quantitative analytics: annualised
return/vol/Sharpe per state, drawdown, tail-risk metrics, and forward-looking
transition probability tables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_HOURS_PER_YEAR = 8_760  # 365 * 24; default for hourly bars


@dataclass
class RegimeStats:
    """Performance statistics for a single hidden state.

    All return/vol metrics are annualised assuming ``bars_per_year`` bars/year.
    Drawdown is computed within each contiguous regime run, then the maximum
    across all runs is reported.

    Attributes
    ----------
    state : int
        State index.
    label : str
        Heuristic label string (may be 'State N' if no label provided).
    count : int
        Number of bars assigned to this state.
    weight : float
        Fraction of total bars, in [0, 1].
    mean_return_ann : float
        Annualised mean log return.
    vol_ann : float
        Annualised standard deviation of log returns.
    sharpe_ann : float
        Annualised Sharpe ratio (risk-free rate = 0).
    skewness : float
        Third standardised moment of log returns in this state.
    excess_kurtosis : float
        Fourth standardised moment minus 3.
    max_drawdown : float
        Maximum peak-to-trough drawdown across contiguous regime spells, in [0, 1].
    var_95 : float
        Empirical 5th-percentile log return (value-at-risk at 95% confidence).
    cvar_95 : float
        Mean of log returns below var_95 (conditional VaR / expected shortfall).
    win_rate : float
        Fraction of bars with strictly positive log return.
    """

    state: int
    label: str
    count: int
    weight: float
    mean_return_ann: float
    vol_ann: float
    sharpe_ann: float
    skewness: float
    excess_kurtosis: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    win_rate: float


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown from a series of log returns.

    Parameters
    ----------
    returns : np.ndarray
        Shape (T,). Log returns; may be empty.

    Returns
    -------
    float
        Maximum drawdown in [0, 1]. Returns 0.0 for empty or singleton input.
    """
    if len(returns) < 2:
        return 0.0
    cum = np.exp(np.cumsum(returns))
    rolling_max = np.maximum.accumulate(cum)
    drawdowns = (rolling_max - cum) / rolling_max
    return float(drawdowns.max())


def compute_regime_stats(
    returns: np.ndarray,
    states: np.ndarray,
    labels: list[str] | None = None,
    bars_per_year: float = _HOURS_PER_YEAR,
) -> list[RegimeStats]:
    """Compute per-regime performance statistics.

    Parameters
    ----------
    returns : np.ndarray
        Shape (T,). Log returns aligned with ``states``.
    states : np.ndarray
        Shape (T,). Integer state assignments (0-indexed).
    labels : list[str] | None
        Human-readable label for each state index. Falls back to 'State N'.
    bars_per_year : float
        Annualisation factor. 8760 for hourly, 252 for daily, 252*6.5 for
        hourly US equities if only market hours are included.

    Returns
    -------
    list[RegimeStats]
        One entry per unique state, sorted by state index.

    Raises
    ------
    ValueError
        If ``returns`` and ``states`` have different lengths.
    """
    if len(returns) != len(states):
        raise ValueError(
            f"returns (len={len(returns)}) and states (len={len(states)}) must have equal length."
        )

    unique_states = sorted(set(int(s) for s in states))
    T = len(returns)
    result: list[RegimeStats] = []

    for k in unique_states:
        mask = states == k
        r_k = returns[mask]
        n_k = int(mask.sum())

        label = labels[k] if labels and k < len(labels) else f"State {k}"

        if n_k == 0:
            result.append(RegimeStats(
                state=k, label=label, count=0, weight=0.0,
                mean_return_ann=np.nan, vol_ann=np.nan, sharpe_ann=np.nan,
                skewness=np.nan, excess_kurtosis=np.nan,
                max_drawdown=np.nan, var_95=np.nan, cvar_95=np.nan,
                win_rate=np.nan,
            ))
            continue

        mean_r = float(r_k.mean())
        std_r  = float(r_k.std(ddof=1)) if n_k > 1 else 0.0

        mean_ann = mean_r * bars_per_year
        vol_ann  = std_r  * np.sqrt(bars_per_year)
        sharpe   = mean_ann / vol_ann if vol_ann > 1e-12 else np.nan

        skew = float(stats.skew(r_k)) if n_k >= 3 else np.nan
        kurt = float(stats.kurtosis(r_k)) if n_k >= 4 else np.nan  # excess by default

        var_95  = float(np.percentile(r_k, 5))
        cvar_95 = float(r_k[r_k <= var_95].mean()) if (r_k <= var_95).any() else var_95

        win_rate = float((r_k > 0).mean())

        # Per-run drawdown: compute within each contiguous spell of this regime
        run_mdd = _compute_per_run_mdd(returns, states, k)

        result.append(RegimeStats(
            state=k,
            label=label,
            count=n_k,
            weight=float(n_k / T),
            mean_return_ann=mean_ann,
            vol_ann=vol_ann,
            sharpe_ann=sharpe,
            skewness=skew,
            excess_kurtosis=kurt,
            max_drawdown=run_mdd,
            var_95=var_95,
            cvar_95=cvar_95,
            win_rate=win_rate,
        ))

    return result


def _compute_per_run_mdd(
    returns: np.ndarray,
    states: np.ndarray,
    target_state: int,
) -> float:
    """Maximum drawdown across all contiguous runs of ``target_state``.

    Parameters
    ----------
    returns : np.ndarray
        Full return series.
    states : np.ndarray
        Full state series.
    target_state : int
        State index to analyse.

    Returns
    -------
    float
        Maximum per-run drawdown. Returns 0.0 if the state has no runs.
    """
    max_mdd = 0.0
    i = 0
    T = len(states)
    while i < T:
        if states[i] != target_state:
            i += 1
            continue
        j = i
        while j < T and states[j] == target_state:
            j += 1
        run_mdd = _max_drawdown_from_returns(returns[i:j])
        max_mdd = max(max_mdd, run_mdd)
        i = j
    return max_mdd


def regime_stats_to_dataframe(stats_list: list[RegimeStats]) -> pd.DataFrame:
    """Convert a list of RegimeStats to a formatted DataFrame.

    Parameters
    ----------
    stats_list : list[RegimeStats]
        Output from ``compute_regime_stats``.

    Returns
    -------
    pd.DataFrame
        One row per state; columns are the RegimeStats fields.
    """
    rows = []
    for s in stats_list:
        rows.append({
            "state": s.state,
            "label": s.label,
            "count": s.count,
            "weight_%": round(100.0 * s.weight, 2),
            "mean_ret_ann_%": round(100.0 * s.mean_return_ann, 3),
            "vol_ann_%": round(100.0 * s.vol_ann, 3),
            "sharpe_ann": round(s.sharpe_ann, 3) if not np.isnan(s.sharpe_ann) else np.nan,
            "skewness": round(s.skewness, 3) if not np.isnan(s.skewness) else np.nan,
            "exc_kurtosis": round(s.excess_kurtosis, 3) if not np.isnan(s.excess_kurtosis) else np.nan,
            "max_drawdown_%": round(100.0 * s.max_drawdown, 2),
            "var_95_%": round(100.0 * s.var_95, 4),
            "cvar_95_%": round(100.0 * s.cvar_95, 4),
            "win_rate_%": round(100.0 * s.win_rate, 2),
        })
    return pd.DataFrame(rows).set_index("state")


def transition_forecast(
    transmat: np.ndarray,
    current_state: int,
    horizons: list[int],
) -> pd.DataFrame:
    """Forward-looking regime probabilities at multiple horizons.

    Computes P(state=j at t+h | state=current_state at t) = (A^h)[current, j]
    using matrix exponentiation. This gives the unconditional probability of
    being in each state h bars from now, starting from ``current_state``.

    Parameters
    ----------
    transmat : np.ndarray
        Shape (K, K). Row-stochastic transition matrix.
    current_state : int
        Starting state index.
    horizons : list[int]
        Forecast horizons in bars.

    Returns
    -------
    pd.DataFrame
        Index = horizons, columns = state indices, values = probabilities.
        Rows sum to 1.0.
    """
    K = transmat.shape[0]
    rows = {}
    A_h = np.eye(K)
    prev_h = 0

    for h in sorted(horizons):
        steps = h - prev_h
        for _ in range(steps):
            A_h = A_h @ transmat
        prev_h = h
        rows[h] = A_h[current_state]

    df = pd.DataFrame(rows, index=[f"S{k}" for k in range(K)]).T
    df.index.name = "horizon_bars"
    return df


def regime_transition_table(
    transmat: np.ndarray,
    labels: list[str] | None = None,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Transition probability table for all states across multiple horizons.

    Parameters
    ----------
    transmat : np.ndarray
        Shape (K, K). Row-stochastic transition matrix.
    labels : list[str] | None
        State labels for column/index names. Falls back to 'S0', 'S1', etc.
    horizons : list[int] | None
        Forecast horizons in bars. Defaults to [1, 6, 24, 72, 168]
        (1 h, 6 h, 1 day, 3 days, 1 week for hourly data).

    Returns
    -------
    pd.DataFrame
        MultiIndex (from_state, horizon_bars) × to_state probabilities.
    """
    K = transmat.shape[0]
    if horizons is None:
        horizons = [1, 6, 24, 72, 168]
    if labels is None:
        labels = [f"S{k}" for k in range(K)]

    all_rows = []
    for k in range(K):
        df = transition_forecast(transmat, k, horizons)
        df.columns = labels[:K]
        df.index = pd.MultiIndex.from_tuples(
            [(labels[k], h) for h in df.index],
            names=["from_state", "horizon_bars"],
        )
        all_rows.append(df)

    return pd.concat(all_rows)
