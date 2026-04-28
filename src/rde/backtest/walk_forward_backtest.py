"""Walk-forward out-of-sample backtest with causal signal generation.

The standard run_backtest uses full-sample HMM signals, which are
in-sample and overfit. This module implements a rigorous OOS evaluation:

For each monthly recalibration fold:
  1. Fit HMM on the trailing train_window (strictly before the boundary).
  2. Decode training data with Viterbi; compute per-state regime stats.
  3. Score states by Sharpe (or return) on the training data.
  4. Run OnlineDecoder on the OOS window → causal filtered posteriors.
  5. Signal = posteriors @ state_scores (optional EMA smoothing).

All OOS (returns, signal) pairs are concatenated and passed to
run_backtest for a single performance evaluation.

The signal at every bar uses only:
  - Information from the training window before the recalibration date.
  - Information accumulated causally within the OOS window up to that bar.

No future data touches the signal at any point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rde.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from rde.evaluation.regime_analytics import compute_regime_stats
from rde.evaluation.walk_forward import (
    _monthly_boundaries,
    _parse_window_days,
)
from rde.inference.online import OnlineDecoder
from rde.inference.viterbi import viterbi_decode
from rde.models.hmm import train_hmm
from rde.signals.regime_signal import RegimeSignalConfig, RegimeSignalGenerator

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardBacktestConfig:
    """Configuration for the walk-forward OOS backtest.

    Attributes
    ----------
    train_window : str
        Rolling training window length, e.g. ``"180d"``.
    n_restarts : int
        HMM restarts per fold. Lower than a full run for speed.
    signal_config : RegimeSignalConfig
        Controls state scoring method and EMA smoothing of the signal.
    backtest_config : BacktestConfig
        Position sizing and transaction-cost settings passed to run_backtest.
    return_col : str
        Column in df_features holding per-bar log returns. Used for both
        regime stats (on training data) and run_backtest.
    """

    train_window: str = "180d"
    n_restarts: int = 5
    signal_config: RegimeSignalConfig = field(default_factory=RegimeSignalConfig)
    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)
    return_col: str = "log_return"


@dataclass
class WalkForwardBacktestResult:
    """Output from :func:`walk_forward_backtest`.

    Attributes
    ----------
    backtest : BacktestResult
        Full OOS backtest (equity curve, metrics, etc.) over all folds.
    fold_summary : pd.DataFrame
        One row per fold with columns:
        ``[train_bars, oos_bars, train_log_lik, sharpe_ann, max_drawdown]``.
    oos_signal : pd.Series
        Concatenated causal OOS signal across all folds.
    oos_returns : pd.Series
        Concatenated OOS log returns across all folds.
    n_folds : int
        Number of successfully evaluated folds.
    """

    backtest: BacktestResult
    fold_summary: pd.DataFrame
    oos_signal: pd.Series
    oos_returns: pd.Series
    n_folds: int


def walk_forward_backtest(
    df_features: pd.DataFrame,
    n_states: int,
    feature_cols: list[str],
    config: WalkForwardBacktestConfig | None = None,
    **train_kwargs,
) -> WalkForwardBacktestResult:
    """Run a fully out-of-sample walk-forward backtest.

    Parameters
    ----------
    df_features : pd.DataFrame
        Feature DataFrame with a tz-aware DatetimeIndex. Must include
        ``config.return_col`` and all ``feature_cols``. NaN rows should
        already be dropped.
    n_states : int
        Number of HMM states for every fold (fixed).
    feature_cols : list[str]
        Columns used as HMM inputs (not the return column).
    config : WalkForwardBacktestConfig, optional
        Walk-forward and backtest settings. Defaults to
        ``WalkForwardBacktestConfig()``.
    **train_kwargs
        Forwarded to :func:`rde.models.hmm.train_hmm` (e.g.
        ``covariance_type``, ``n_iter``, ``seed_base``).

    Returns
    -------
    WalkForwardBacktestResult

    Raises
    ------
    ValueError
        If no recalibration boundaries are found, or all folds are skipped.

    Notes
    -----
    The ``n_restarts`` from ``config`` takes precedence over any value
    passed via ``train_kwargs``.
    """
    if config is None:
        config = WalkForwardBacktestConfig()

    index = df_features.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("df_features must have a tz-aware DatetimeIndex.")
    if config.return_col not in df_features.columns:
        raise ValueError(
            f"return_col={config.return_col!r} not found in df_features. "
            f"Available: {list(df_features.columns)}"
        )
    for col in feature_cols:
        if col not in df_features.columns:
            raise ValueError(f"feature_col={col!r} not in df_features.")

    train_delta = pd.Timedelta(days=_parse_window_days(config.train_window))
    first_boundary = index[0] + train_delta
    boundaries = _monthly_boundaries(index, first_boundary)

    if not boundaries:
        raise ValueError(
            f"No walk-forward boundaries found after warm-up period "
            f"(train_window={config.train_window}). "
            "Use a shorter train_window or a longer data span."
        )

    kw = dict(train_kwargs)
    kw["n_restarts"] = config.n_restarts
    kw.setdefault("feature_names", feature_cols)

    oos_returns_parts: list[pd.Series] = []
    oos_signal_parts: list[pd.Series] = []
    fold_records: list[dict] = []

    for b_idx, boundary in enumerate(boundaries):
        next_boundary = (
            boundaries[b_idx + 1]
            if b_idx + 1 < len(boundaries)
            else index[-1] + pd.Timedelta(hours=1)
        )

        train_mask = (index >= boundary - train_delta) & (index < boundary)
        oos_mask = (index >= boundary) & (index < next_boundary)

        X_train = df_features.loc[train_mask, feature_cols].values
        X_oos = df_features.loc[oos_mask, feature_cols].values
        oos_idx = df_features.index[oos_mask]
        oos_ret_vals = df_features.loc[oos_mask, config.return_col].values

        if len(X_train) < n_states:
            logger.warning(
                "Fold %s: only %d training bars (need ≥ %d). Skipping.",
                boundary.date(), len(X_train), n_states,
            )
            continue
        if len(X_oos) == 0:
            logger.debug("Fold %s: no OOS bars. Skipping.", boundary.date())
            continue

        logger.info(
            "Fold %s: train=%d bars, oos=%d bars",
            boundary.date(), len(X_train), len(X_oos),
        )

        fitted = train_hmm(X_train, n_states, **kw)

        # Per-state scores from training data only
        X_train_sc = fitted.scaler.transform(X_train)
        train_states = viterbi_decode(fitted.hmm, X_train_sc)
        train_returns = df_features.loc[train_mask, config.return_col].values
        regime_stats = compute_regime_stats(train_returns, train_states)

        # Causal OOS signal via online filtering
        decoder = OnlineDecoder(fitted)
        oos_posteriors = decoder.batch_filter(X_oos)

        sig_gen = RegimeSignalGenerator(regime_stats, config=config.signal_config)
        oos_signal_vals = sig_gen.generate(oos_posteriors)

        oos_returns_parts.append(
            pd.Series(oos_ret_vals, index=oos_idx, name=config.return_col)
        )
        oos_signal_parts.append(
            pd.Series(oos_signal_vals, index=oos_idx, name="signal")
        )
        fold_records.append(
            {
                "boundary": boundary,
                "train_bars": len(X_train),
                "oos_bars": len(X_oos),
                "train_log_lik": fitted.log_likelihood,
            }
        )

    if not fold_records:
        raise ValueError(
            "All walk-forward folds were skipped. "
            "Check train_window vs data span and n_states."
        )

    all_returns = pd.concat(oos_returns_parts)
    all_signal = pd.concat(oos_signal_parts)

    bt_result = run_backtest(
        returns=all_returns,
        signal=all_signal,
        config=config.backtest_config,
    )

    fold_df = pd.DataFrame(fold_records).set_index("boundary")

    logger.info(
        "Walk-forward OOS backtest: %d folds | %d bars | Sharpe=%.3f | MaxDD=%.3f",
        len(fold_records),
        len(all_returns),
        bt_result.metrics["sharpe_ann"],
        bt_result.metrics["max_drawdown"],
    )

    return WalkForwardBacktestResult(
        backtest=bt_result,
        fold_summary=fold_df,
        oos_signal=all_signal,
        oos_returns=all_returns,
        n_folds=len(fold_records),
    )
