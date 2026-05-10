"""Regime-conditional volatility forecasting quality test (Phase 46).

Tests whether the HMM posterior-weighted vol forecast adds quality over
simple EWMA/historical-vol baselines.

At each test bar t (causal):
  - HMM vol forecast: sqrt(sum_k(gamma_k * sigma_k^2))
    where sigma_k^2 = state-k variance of the log-return feature
  - EWMA vol forecast: EWMA of squared returns (lambda=0.94, annualised)
  - Historical vol: rolling std over vol_window bars (annualised)

Target: realized vol at horizon h = std(returns[t+1:t+h+1]) * sqrt(ann_factor)

Metrics: MSE, MAE, Pearson correlation, and Information Coefficient (IC)
relative to each baseline.

Public API
----------
VolForecastResult
run_vol_forecast_cv
compare_vol_forecasters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rde.evaluation.purged_cv import purged_k_fold_splits
from rde.inference.online import OnlineDecoder
from rde.models.hmm import train_hmm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class VolForecastResult:
    """Per-fold and aggregate vol forecasting metrics.

    Attributes
    ----------
    method : str
        Forecaster name: ``"hmm"``, ``"ewma"``, ``"hist_vol"``.
    horizon_bars : int
        Forecast horizon in bars.
    ann_factor : int
        Bars per year used to annualise.
    mse_mean : float
        Mean squared error (annualised vol scale) across all test bars.
    mae_mean : float
        Mean absolute error (annualised vol scale).
    correlation : float
        Pearson correlation between forecast and realised vol.
    ic : float
        Information coefficient = Spearman rank correlation.
    n_observations : int
        Total number of forecast/realised pairs across all folds.
    fold_mse : list[float]
        Per-fold MSE (for stability analysis).
    fold_corr : list[float]
        Per-fold correlation.
    """

    method: str
    horizon_bars: int
    ann_factor: int
    mse_mean: float
    mae_mean: float
    correlation: float
    ic: float
    n_observations: int
    fold_mse: list[float] = field(default_factory=list)
    fold_corr: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Low-level forecasters
# ---------------------------------------------------------------------------


def _ewma_vol_forecast(
    returns: np.ndarray,
    t: int,
    lam: float = 0.94,
    ann_factor: int = 252,
    min_obs: int = 10,
) -> float:
    """EWMA variance estimate at bar t (uses only past returns).

    Returns annualised vol.
    """
    if t < min_obs:
        return float(returns[:max(t, 1)].std() * np.sqrt(ann_factor))
    r = returns[:t]
    weights = lam ** np.arange(len(r) - 1, -1, -1)
    weights /= weights.sum()
    var = float((weights * r ** 2).sum())
    return float(np.sqrt(var * ann_factor))


def _hist_vol_forecast(
    returns: np.ndarray,
    t: int,
    window: int = 20,
    ann_factor: int = 252,
) -> float:
    """Rolling historical vol at bar t using past ``window`` returns (annualised)."""
    start = max(0, t - window)
    r = returns[start:t]
    if len(r) < 2:
        return float(np.abs(returns[:max(t, 1)]).mean() * np.sqrt(ann_factor))
    return float(r.std(ddof=1) * np.sqrt(ann_factor))


def _realised_vol(
    returns: np.ndarray,
    t: int,
    horizon: int,
    ann_factor: int = 252,
) -> float | None:
    """Realised vol from bar t+1 through t+horizon (annualised).

    Returns None if insufficient bars remain.
    """
    if t + horizon > len(returns):
        return None
    r = returns[t: t + horizon]
    if len(r) < 2:
        return None
    return float(r.std(ddof=1) * np.sqrt(ann_factor))


def _hmm_vol_forecast(
    decoder: OnlineDecoder,
    X_raw: np.ndarray,
    t: int,
    log_return_idx: int,
    ann_factor: int = 252,
    posteriors_cache: np.ndarray | None = None,
) -> float:
    """HMM posterior-weighted vol forecast at bar t (causal).

    Parameters
    ----------
    decoder : OnlineDecoder
        Fitted and pre-filtered decoder (posteriors_cache should be provided
        to avoid re-filtering from scratch).
    X_raw : np.ndarray
        Shape (T, n_features). Raw features for the full window.
    t : int
        Bar index.  Uses posteriors from bar t (posterior given x_{1:t}).
    log_return_idx : int
        Feature index for log-return in X_raw.
    ann_factor : int
    posteriors_cache : np.ndarray, optional
        Shape (T, K).  If provided, uses pre-computed posteriors.

    Returns
    -------
    float
        Annualised vol forecast.
    """
    if posteriors_cache is not None:
        gamma = posteriors_cache[t]  # (K,)
    else:
        gamma = decoder.batch_filter(X_raw[: t + 1])[-1]  # (K,)

    covars = decoder._model.covars_  # (K, n_features, n_features)
    state_vars = covars[:, log_return_idx, log_return_idx]  # (K,)

    # Original-space variance: unscale from scaler (StandardScaler).
    scale = decoder._scaler.scale_[log_return_idx]  # scaler std for that feature
    state_vars_orig = state_vars * (scale ** 2)  # back to original units

    weighted_var = float(gamma @ state_vars_orig)
    return float(np.sqrt(max(weighted_var, 0.0) * ann_factor))


# ---------------------------------------------------------------------------
# Main CV function
# ---------------------------------------------------------------------------


def run_vol_forecast_cv(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    returns_col: str,
    n_states: int,
    *,
    train_bars: int,
    test_bars: int,
    embargo_bars: int = 24,
    horizons: list[int] | None = None,
    ann_factor: int = 252,
    ewma_lambda: float = 0.94,
    hist_vol_window: int = 20,
    train_kwargs: dict | None = None,
    seed: int = 42,
) -> dict[str, list[VolForecastResult]]:
    """Run walk-forward vol forecasting quality test.

    For each purged k-fold split:
    1. Train HMM on train window.
    2. Run OnlineDecoder causally through test window.
    3. For each test bar t and horizon h, compare HMM/EWMA/HistVol forecasts
       to realised vol.

    Parameters
    ----------
    df_features : pd.DataFrame
        Feature DataFrame; must include ``returns_col``.
    feature_cols : list[str]
        Columns used as HMM features.
    returns_col : str
        Column used as the log-return (also the target for vol forecast).
    n_states : int
        HMM states.
    train_bars : int
        Training window length.
    test_bars : int
        Test window length.
    embargo_bars : int
        Gap between train end and test start.
    horizons : list[int], optional
        Forecast horizons in bars.  Default: [5, 10, 21].
    ann_factor : int
        Bars per year for annualisation.
    ewma_lambda : float
        Decay parameter for EWMA baseline (default 0.94, RiskMetrics).
    hist_vol_window : int
        Window for historical vol baseline.
    train_kwargs : dict, optional
        Extra arguments for :func:`rde.models.hmm.train_hmm`.
    seed : int
        Random seed base.

    Returns
    -------
    dict[str, list[VolForecastResult]]
        Keys: ``"hmm"``, ``"ewma"``, ``"hist_vol"``.
        Values: list of VolForecastResult — one per (method, horizon) pair.
    """
    if horizons is None:
        horizons = [5, 10, 21]
    tkwargs = {**(train_kwargs or {}), "seed_base": seed}

    X = df_features[feature_cols].values.astype(float)
    returns_arr = df_features[returns_col].values.astype(float)
    n_obs = len(X)

    log_return_idx = (
        feature_cols.index(returns_col) if returns_col in feature_cols else 0
    )

    splits = list(purged_k_fold_splits(
        n_obs,
        train_bars=train_bars,
        test_bars=test_bars,
        embargo_bars=embargo_bars,
    ))

    # Accumulate (forecast, realised) pairs per method per horizon
    records: dict[str, dict[int, list[tuple[float, float]]]] = {
        "hmm": {h: [] for h in horizons},
        "ewma": {h: [] for h in horizons},
        "hist_vol": {h: [] for h in horizons},
    }
    # Per-fold MSE/corr for stability
    fold_data: dict[str, dict[int, list[tuple[float, float]]]] = {
        m: {h: [] for h in horizons} for m in ("hmm", "ewma", "hist_vol")
    }

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = X[train_idx]
        X_test = X[test_idx]
        returns_full = returns_arr[: test_idx[-1] + 1]

        if X_train.shape[0] < n_states * 2:
            continue

        try:
            fitted = train_hmm(X_train, n_states=n_states, n_restarts=1, **tkwargs)
            decoder = OnlineDecoder(fitted)
        except Exception as exc:
            logger.warning("Fold %d HMM fit failed: %s", fold_idx, exc)
            continue

        # Compute posteriors for the FULL window up to test_end (causal).
        # We re-run through train+test so posteriors at test bars carry
        # information from train bars (natural warm start).
        X_combined = np.vstack([X_train, X_test])
        posteriors_combined = decoder.batch_filter(X_combined)  # (train+test, K)
        n_train = len(train_idx)

        fold_hmm: dict[int, list[tuple[float, float]]] = {h: [] for h in horizons}
        fold_ewma: dict[int, list[tuple[float, float]]] = {h: [] for h in horizons}
        fold_hist: dict[int, list[tuple[float, float]]] = {h: [] for h in horizons}

        for ti, abs_t in enumerate(test_idx):
            combined_t = n_train + ti  # index in X_combined

            for h in horizons:
                rv = _realised_vol(returns_full, abs_t, h, ann_factor)
                if rv is None:
                    continue

                v_hmm = _hmm_vol_forecast(
                    decoder, X_combined, combined_t, log_return_idx,
                    ann_factor, posteriors_combined,
                )
                v_ewma = _ewma_vol_forecast(
                    returns_full, abs_t, ewma_lambda, ann_factor,
                )
                v_hist = _hist_vol_forecast(
                    returns_full, abs_t, hist_vol_window, ann_factor,
                )

                fold_hmm[h].append((v_hmm, rv))
                fold_ewma[h].append((v_ewma, rv))
                fold_hist[h].append((v_hist, rv))
                records["hmm"][h].append((v_hmm, rv))
                records["ewma"][h].append((v_ewma, rv))
                records["hist_vol"][h].append((v_hist, rv))

        # Store fold-level per-horizon results for stability
        for h in horizons:
            for method, fd in (("hmm", fold_hmm), ("ewma", fold_ewma), ("hist_vol", fold_hist)):
                pairs = fd[h]
                if len(pairs) > 2:
                    fcast = np.array([p[0] for p in pairs])
                    real = np.array([p[1] for p in pairs])
                    fold_mse = float(np.mean((fcast - real) ** 2))
                    fold_corr = float(np.corrcoef(fcast, real)[0, 1]) if len(pairs) > 2 else 0.0
                    fold_data[method][h].append((fold_mse, fold_corr))

    # Aggregate into VolForecastResult
    results: dict[str, list[VolForecastResult]] = {}
    for method in ("hmm", "ewma", "hist_vol"):
        method_results = []
        for h in horizons:
            pairs = records[method][h]
            if len(pairs) < 10:
                logger.warning("%s h=%d: too few observations (%d)", method, h, len(pairs))
                continue
            fcast = np.array([p[0] for p in pairs])
            real = np.array([p[1] for p in pairs])
            mse = float(np.mean((fcast - real) ** 2))
            mae = float(np.mean(np.abs(fcast - real)))
            corr = float(np.corrcoef(fcast, real)[0, 1]) if len(pairs) > 2 else 0.0
            # Spearman IC
            from scipy.stats import spearmanr
            ic_val, _ = spearmanr(fcast, real)
            fold_mses = [fd[0] for fd in fold_data[method][h]]
            fold_corrs = [fd[1] for fd in fold_data[method][h]]
            method_results.append(VolForecastResult(
                method=method,
                horizon_bars=h,
                ann_factor=ann_factor,
                mse_mean=mse,
                mae_mean=mae,
                correlation=corr,
                ic=float(ic_val),
                n_observations=len(pairs),
                fold_mse=fold_mses,
                fold_corr=fold_corrs,
            ))
        results[method] = method_results
    return results


def compare_vol_forecasters(
    results: dict[str, list[VolForecastResult]],
) -> pd.DataFrame:
    """Tabulate vol forecaster metrics across methods and horizons.

    Returns a DataFrame with columns:
    method, horizon_bars, mse, mae, correlation, ic, n_obs.
    """
    rows = []
    for method, method_results in results.items():
        for r in method_results:
            rows.append({
                "method": r.method,
                "horizon_bars": r.horizon_bars,
                "mse": round(r.mse_mean, 6),
                "mae": round(r.mae_mean, 4),
                "correlation": round(r.correlation, 3),
                "ic": round(r.ic, 3),
                "n_obs": r.n_observations,
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["horizon_bars", "mse"]).reset_index(drop=True)
