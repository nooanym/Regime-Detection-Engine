"""Integration pipeline that orchestrates all analysis modules (Phase 31).

Provides :class:`AnalysisPipeline`, which accepts the core HMM outputs
(feature matrix, posteriors, transition matrix, returns, regime labels) and
runs each analysis module in sequence, collecting results into an
:class:`AnalysisReport`.

Each module is called inside a try/except; failures are logged as warnings
and the corresponding report field is set to ``None``.  This ensures a
partial failure in one module does not abort the full pipeline.
"""

from __future__ import annotations

import datetime
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AnalysisConfig:
    """Configuration for :class:`AnalysisPipeline`.

    Attributes
    ----------
    symbol : str
        Asset symbol for labelling purposes.
    returns_col_idx : int
        Which column of ``X`` is the return series.
    run_execution : bool
        Whether to run the execution impact module.
    run_factors : bool
        Whether to run the factor analysis module.
    run_cointegration : bool
        Whether to run the cointegration module (auto-skipped if ``d < 2``).
    run_portfolio : bool
        Whether to run the portfolio optimisation module (auto-skipped if
        ``d < 2``).
    run_signal_filtering : bool
        Whether to run the signal filtering module.
    run_transition : bool
        Whether to run the transition prediction module.
    run_tail_risk : bool
        Whether to run the tail risk / EVT module.
    run_backtest : bool
        Whether to run the regime backtest module.
    run_correlation : bool
        Whether to run the correlation module.
    """

    symbol: str = "UNKNOWN"
    returns_col_idx: int = 0
    run_execution: bool = True
    run_factors: bool = True
    run_cointegration: bool = True   # auto-skipped if d < 2
    run_portfolio: bool = True       # auto-skipped if d < 2
    run_signal_filtering: bool = True
    run_transition: bool = True
    run_tail_risk: bool = True
    run_backtest: bool = True
    run_correlation: bool = True


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class AnalysisReport:
    """Aggregated output of all analysis modules.

    Attributes
    ----------
    symbol : str
    run_timestamp : str
        ISO 8601 UTC timestamp of when the pipeline ran.
    n_obs : int
    n_states : int
    n_features : int
    feature_names : list[str]
    execution : dict or None
    factor_result : dict or None
    cointegration : dict or None
    portfolio : dict or None
    signal_filtering : dict or None
    transition : dict or None
    tail_risk : dict or None
    backtest : dict or None
    correlation : dict or None

    Notes
    -----
    Each populated field is a plain dict with JSON-serialisable values
    (numpy arrays converted to lists, DataFrames to records, dataclass
    fields extracted manually).  ``None`` means the module was skipped or
    raised an exception.
    """

    symbol: str
    run_timestamp: str            # ISO 8601 UTC
    n_obs: int
    n_states: int
    n_features: int
    feature_names: list[str]

    execution: dict[str, Any] | None = None
    factor_result: dict[str, Any] | None = None
    cointegration: dict[str, Any] | None = None
    portfolio: dict[str, Any] | None = None
    signal_filtering: dict[str, Any] | None = None
    transition: dict[str, Any] | None = None
    tail_risk: dict[str, Any] | None = None
    backtest: dict[str, Any] | None = None
    correlation: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class AnalysisPipeline:
    """Orchestrates all analysis modules for a fitted HMM result.

    Parameters
    ----------
    config : AnalysisConfig, optional
        Pipeline configuration.  Defaults to :class:`AnalysisConfig` with
        all modules enabled and ``symbol="UNKNOWN"``.
    """

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        self.config = config or AnalysisConfig()

    def run(
        self,
        X: np.ndarray,
        posteriors: np.ndarray,
        transmat: np.ndarray,
        returns: pd.Series | np.ndarray,
        regimes: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> AnalysisReport:
        """Run all enabled analysis modules and return a consolidated report.

        Parameters
        ----------
        X : np.ndarray, shape (T, d)
            Feature matrix (standardised or raw; all modules handle scaling
            internally).
        posteriors : np.ndarray, shape (T, K)
            HMM forward-backward posteriors (soft regime assignments).
        transmat : np.ndarray, shape (K, K)
            HMM transition matrix.
        returns : pd.Series or np.ndarray, length T
            Raw return series (e.g. log returns) for the primary asset.
        regimes : np.ndarray, shape (T,)
            Viterbi-decoded hard regime labels.
        feature_names : list[str], optional
            Human-readable names for the ``d`` columns of ``X``.  Defaults
            to ``["feat_0", "feat_1", ...]``.

        Returns
        -------
        AnalysisReport
        """
        cfg = self.config

        X = np.asarray(X, dtype=float)
        T, d = X.shape
        K = posteriors.shape[1]

        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(d)]

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        report = AnalysisReport(
            symbol=cfg.symbol,
            run_timestamp=timestamp,
            n_obs=T,
            n_states=K,
            n_features=d,
            feature_names=list(feature_names),
        )

        # ------------------------------------------------------------------
        # Execution impact
        # ------------------------------------------------------------------
        if cfg.run_execution:
            try:
                from rde.analysis.execution import (
                    estimate_regime_impact,
                    expected_slippage,
                    twap_schedule,
                )

                # estimate_regime_impact expects a pd.DataFrame; pass X wrapped
                X_df = pd.DataFrame(X, columns=feature_names)
                impact_params = estimate_regime_impact(X_df, posteriors)

                # Per-regime slippage at size=0.01: use the hard regime
                # posterior (one-hot) as the current posterior vector
                slippage_by_regime: dict[int, float] = {}
                for k in range(K):
                    one_hot = np.zeros(K)
                    one_hot[k] = 1.0
                    slippage_by_regime[k] = float(
                        expected_slippage(0.01, impact_params, one_hot)
                    )

                sched = twap_schedule(100.0, 5)
                report.execution = {
                    "n_regimes": K,
                    "slippage_by_regime": slippage_by_regime,
                    "twap_example": sched.tolist(),
                }
            except Exception as e:
                logger.warning("execution module failed: %s", e)

        # ------------------------------------------------------------------
        # Factor analysis
        # ------------------------------------------------------------------
        if cfg.run_factors:
            try:
                from rde.analysis.factor_analysis import fit_regime_factors

                fres = fit_regime_factors(X, posteriors)
                report.factor_result = {
                    "n_factors_per_regime": [
                        len(rf.loadings[0]) if rf.loadings.ndim > 1 else rf.loadings.shape[0]
                        for rf in fres.regime_factors
                    ],
                    "explained_variance_ratios": [
                        rf.explained_variance_ratio.tolist()
                        for rf in fres.regime_factors
                    ],
                }
            except Exception as e:
                logger.warning("factor_result module failed: %s", e)

        # ------------------------------------------------------------------
        # Cointegration (only if d >= 2)
        # ------------------------------------------------------------------
        if cfg.run_cointegration and d >= 2:
            try:
                from rde.analysis.cointegration import fit_regime_spreads

                spread_result = fit_regime_spreads(X[:, 0], X[:, 1], posteriors)
                hl_by_regime = {
                    str(rp.regime): (float(rp.half_life) if np.isfinite(rp.half_life) else None)
                    for rp in spread_result.regime_params
                }
                ghl = spread_result.global_half_life
                report.cointegration = {
                    "pair": "feat_0_vs_feat_1",
                    "regime_half_lives": hl_by_regime,
                    "global_half_life": float(ghl) if np.isfinite(ghl) else None,
                }
            except Exception as e:
                logger.warning("cointegration module failed: %s", e)
        elif cfg.run_cointegration and d < 2:
            logger.warning("cointegration module skipped: d=%d < 2", d)

        # ------------------------------------------------------------------
        # Portfolio optimisation (only if d >= 2)
        # ------------------------------------------------------------------
        if cfg.run_portfolio and d >= 2:
            try:
                from rde.analysis.portfolio_optimization import regime_mvo

                mvo_result = regime_mvo(X, posteriors)
                blend_w = mvo_result.global_weights.min_var_weights
                report.portfolio = {
                    "blended_weights": blend_w.tolist() if blend_w is not None else None,
                    "regime_weights": [
                        rw.min_var_weights.tolist()
                        for rw in mvo_result.regime_weights
                    ],
                }
            except Exception as e:
                logger.warning("portfolio module failed: %s", e)
        elif cfg.run_portfolio and d < 2:
            logger.warning("portfolio module skipped: d=%d < 2", d)

        # ------------------------------------------------------------------
        # Signal filtering
        # ------------------------------------------------------------------
        if cfg.run_signal_filtering:
            try:
                from rde.analysis.signal_filtering import (
                    regime_ema,
                    regime_hp_filter,
                    adaptive_kalman_smoother,
                )

                sig = X[:, cfg.returns_col_idx]
                ema_out = regime_ema(sig, posteriors)
                hp_out = regime_hp_filter(sig, posteriors)
                kal_out = adaptive_kalman_smoother(sig, posteriors)

                # Convert Series to array if needed
                ema_arr = np.asarray(ema_out)
                hp_arr = np.asarray(hp_out)
                kal_arr = np.asarray(kal_out)

                report.signal_filtering = {
                    "ema_mean": float(np.mean(ema_arr)),
                    "ema_std": float(np.std(ema_arr)),
                    "hp_mean": float(np.mean(hp_arr)),
                    "kalman_mean": float(np.mean(kal_arr)),
                }
            except Exception as e:
                logger.warning("signal_filtering module failed: %s", e)

        # ------------------------------------------------------------------
        # Transition prediction
        # ------------------------------------------------------------------
        if cfg.run_transition:
            try:
                from rde.analysis.transition_prediction import (
                    multi_step_transition,
                    expected_regime_duration,
                    transition_entropy_trajectory,
                )

                ms = multi_step_transition(transmat, steps=5)
                durations = [
                    float(expected_regime_duration(k, transmat)) for k in range(K)
                ]
                ent_traj = transition_entropy_trajectory(posteriors, transmat)

                report.transition = {
                    "expected_durations": durations,
                    "5step_transmat": ms.tolist(),
                    "mean_transition_entropy": float(np.mean(ent_traj)),
                }
            except Exception as e:
                logger.warning("transition module failed: %s", e)

        # ------------------------------------------------------------------
        # Tail risk
        # ------------------------------------------------------------------
        if cfg.run_tail_risk:
            try:
                from rde.analysis.tail_risk import fit_regime_tails

                # fit_regime_tails expects raw returns (it negates internally)
                raw_returns = X[:, cfg.returns_col_idx]
                tr = fit_regime_tails(raw_returns, posteriors)

                regime_tail_summaries = []
                for rt in tr.regime_tails:
                    xi = float(rt.gpd.xi) if rt.gpd is not None else None
                    sigma = float(rt.gpd.sigma) if rt.gpd is not None else None
                    regime_tail_summaries.append({
                        "regime": rt.regime,
                        "xi": xi,
                        "sigma": sigma,
                        "var_95": float(rt.var),
                        "es_95": float(rt.es) if np.isfinite(rt.es) else None,
                    })

                report.tail_risk = {"regime_tails": regime_tail_summaries}
            except Exception as e:
                logger.warning("tail_risk module failed: %s", e)

        # ------------------------------------------------------------------
        # Backtest
        # ------------------------------------------------------------------
        if cfg.run_backtest:
            try:
                from rde.analysis.backtest import (
                    run_backtest,
                    backtest_tearsheet,
                    regime_performance_attribution,
                    RegimeStrategyConfig,
                    RegimeRule,
                )

                # Default: long in regime 0, flat in all other regimes
                rules = [RegimeRule(target_position=1.0)] + [
                    RegimeRule(target_position=0.0) for _ in range(K - 1)
                ]
                cfg_bt = RegimeStrategyConfig(rules=rules, transaction_cost=0.001)
                ret_series = pd.Series(np.asarray(returns, dtype=float))
                bt_result = run_backtest(ret_series, regimes, cfg_bt)
                ts = backtest_tearsheet(bt_result)
                attr = regime_performance_attribution(bt_result)

                report.backtest = {
                    "tearsheet": ts.to_dict(),
                    "attribution": attr.to_dict("records"),
                }
            except Exception as e:
                logger.warning("backtest module failed: %s", e)

        # ------------------------------------------------------------------
        # Correlation
        # ------------------------------------------------------------------
        if cfg.run_correlation:
            try:
                from rde.analysis.correlation import (
                    regime_correlation,
                    tail_dependence,
                )

                corr_result = regime_correlation(X, posteriors)
                td = tail_dependence(X, posteriors)

                report.correlation = {
                    "stability": float(corr_result.stability),
                    "global_corr": corr_result.global_corr.corr.tolist(),
                    "regime_corrs": [
                        rc.corr.tolist() for rc in corr_result.regime_corrs
                    ],
                    "tail_dependence": td.to_dict("records"),
                }
            except Exception as e:
                logger.warning("correlation module failed: %s", e)

        return report
