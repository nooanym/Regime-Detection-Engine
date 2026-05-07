"""Purged k-fold cross-validation with embargo for time-series HMM evaluation.

Implements:

1. :func:`purged_k_fold_splits` — rolling walk-forward folds with embargo gap.
2. :func:`combinatorial_purged_splits` — all C(N, K) train/test combinations
   (de Prado, *Advances in Financial Machine Learning*, ch. 7).
3. :class:`FoldResult` — per-fold performance and stability metrics.
4. :func:`run_purged_cv` — orchestrates purged k-fold CV, saves parquet.
5. :func:`run_combinatorial_purged_cv` — combinatorial variant.

Key design decisions
--------------------
- Strategy simulation uses :class:`rde.inference.online.OnlineDecoder`
  (causal, forward-only) rather than Viterbi, which has within-fold look-ahead.
- Default strategy: long (+1.0) in regimes whose scaled emission mean is
  positive (above-average return in training data), flat otherwise.
- State alignment across folds uses mean-return rank for Frobenius dispersion.
- Output parquet has one row per fold — never collapsed to a single number.

References
----------
de Prado, M. L. (2018). *Advances in Financial Machine Learning*, Wiley, ch. 7.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from rde.analysis.backtest import (
    BacktestResult,
    RegimeRule,
    RegimeStrategyConfig,
    backtest_tearsheet,
    run_backtest,
)
from rde.inference.online import OnlineDecoder
from rde.models.hmm import FittedModel, train_hmm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FoldResult
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """Performance and stability metrics for a single CV fold.

    Attributes
    ----------
    fold_id : int
        Zero-based fold index.
    cv_type : str
        ``"purged_kfold"`` or ``"combinatorial_purged"``.
    train_start, train_end : pd.Timestamp
        First and last bar timestamps in the training window.
    test_start, test_end : pd.Timestamp
        First and last bar timestamps in the test window.
    n_train, n_test : int
        Bar counts.
    sharpe : float
        Annualised Sharpe ratio of the default strategy in the test window.
    calmar : float
        Annualised return / max drawdown.
    max_drawdown : float
        Peak-to-trough drawdown fraction in the test window.
    hit_rate : float
        Fraction of test bars with positive strategy return.
    ann_return : float
        Annualised strategy return.
    ann_vol : float
        Annualised strategy return volatility.
    n_trades : int
        Number of position changes in the test window.
    bic : float
        Bayesian Information Criterion of the HMM on training data.
    log_likelihood : float
        Log-likelihood on training data.
    n_params : int
        Free parameters in the model.
    n_states_used : int
        ``n_states`` value passed to this fold.
    regime_conditional_returns : dict[int, float]
        Mean asset return in test window conditional on each regime.
    transmat_frobenius_from_mean : float
        Frobenius distance of the aligned transition matrix from the
        cross-fold mean. ``NaN`` until dispersion computation.
    means_frobenius_from_mean : float
        Frobenius distance of the aligned emission means from the cross-fold
        mean. ``NaN`` until dispersion computation.
    """

    fold_id: int
    cv_type: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    sharpe: float
    calmar: float
    max_drawdown: float
    hit_rate: float
    ann_return: float
    ann_vol: float
    n_trades: int
    bic: float
    log_likelihood: float
    n_params: int
    n_states_used: int
    regime_conditional_returns: dict[int, float] = field(default_factory=dict)
    transmat_frobenius_from_mean: float = float("nan")
    means_frobenius_from_mean: float = float("nan")

    def to_dict(self) -> dict:
        """Flat dict for DataFrame / parquet serialisation.

        Regime-conditional returns are expanded to ``regime_k_mean_return``
        columns so the parquet is fully columnar.
        """
        d: dict = {
            "fold_id": self.fold_id,
            "cv_type": self.cv_type,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "sharpe": self.sharpe,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.hit_rate,
            "ann_return": self.ann_return,
            "ann_vol": self.ann_vol,
            "n_trades": self.n_trades,
            "bic": self.bic,
            "log_likelihood": self.log_likelihood,
            "n_params": self.n_params,
            "n_states_used": self.n_states_used,
            "transmat_frobenius_from_mean": self.transmat_frobenius_from_mean,
            "means_frobenius_from_mean": self.means_frobenius_from_mean,
        }
        for k, v in self.regime_conditional_returns.items():
            d[f"regime_{k}_mean_return"] = v
        return d


# ---------------------------------------------------------------------------
# Split generators
# ---------------------------------------------------------------------------


def purged_k_fold_splits(
    n_obs: int,
    *,
    train_bars: int,
    test_bars: int,
    embargo_bars: int = 24,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) for rolling walk-forward with embargo.

    Advances the window by ``test_bars`` at each step. The ``embargo_bars``
    gap between the end of training and the start of the test window prevents
    autocorrelation leakage at the boundary.

    Parameters
    ----------
    n_obs : int
        Total number of observations.
    train_bars : int
        Number of bars in the training window.
    test_bars : int
        Number of bars in the test window.
    embargo_bars : int
        Bars excluded between train end and test start (default 24 — one day).

    Yields
    ------
    train_idx : np.ndarray[int]
        Bar indices for training.
    test_idx : np.ndarray[int]
        Bar indices for testing.
    """
    fold_start = 0
    while True:
        train_end = fold_start + train_bars
        test_start = train_end + embargo_bars
        test_end = test_start + test_bars
        if test_end > n_obs:
            break
        yield np.arange(fold_start, train_end), np.arange(test_start, test_end)
        fold_start += test_bars


def combinatorial_purged_splits(
    n_obs: int,
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_bars: int = 24,
) -> Iterator[tuple[np.ndarray, np.ndarray, tuple[int, ...]]]:
    """Yield all C(n_groups, n_test_groups) train/test splits with embargo.

    Implements de Prado (AFML ch. 7) combinatorial purged CV. The series is
    divided into ``n_groups`` equal groups; each of the
    ``C(n_groups, n_test_groups)`` combinations uses ``n_test_groups`` groups
    as test and the remainder as training (minus embargo zones at each
    train→test boundary).

    The result is a **distribution** of fold outcomes (each observation may
    appear in multiple test sets) rather than a single performance path.

    Parameters
    ----------
    n_obs : int
        Total number of observations.
    n_groups : int
        Number of equal-sized groups to partition the series into.
    n_test_groups : int
        Number of groups in each test set.
    embargo_bars : int
        Bars excluded from training immediately before each test group start.

    Yields
    ------
    train_idx : np.ndarray[int]
    test_idx : np.ndarray[int]
    test_group_ids : tuple[int, ...]
        Which group indices form the test set for this combination.
    """
    group_size = n_obs // n_groups
    bounds = [
        (i * group_size, min((i + 1) * group_size, n_obs))
        for i in range(n_groups)
    ]

    for test_groups in itertools.combinations(range(n_groups), n_test_groups):
        test_mask = np.zeros(n_obs, dtype=bool)
        embargo_mask = np.zeros(n_obs, dtype=bool)

        for g in test_groups:
            start, end = bounds[g]
            test_mask[start:end] = True
            emb_start = max(0, start - embargo_bars)
            embargo_mask[emb_start:start] = True

        yield (
            np.where(~test_mask & ~embargo_mask)[0],
            np.where(test_mask)[0],
            test_groups,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_params(fitted: FittedModel) -> int:
    """Count free model parameters (transitions + initial + means + covars)."""
    K = fitted.n_states
    D = fitted.hmm.means_.shape[1]
    n = K * (K - 1) + (K - 1) + K * D
    cov_type = getattr(fitted.hmm, "covariance_type", "full")
    if cov_type == "full":
        n += K * D * (D + 1) // 2
    elif cov_type == "diag":
        n += K * D
    elif cov_type == "tied":
        n += D * (D + 1) // 2
    else:  # spherical
        n += K
    return n


def _default_strategy_config(
    fitted: FittedModel,
    ann_factor: int,
    transaction_cost: float,
) -> RegimeStrategyConfig:
    """Long in above-average-return regimes per training emission means.

    Uses ``fitted.hmm.means_[k, ret_idx] > 0`` where ``ret_idx`` is the
    index of ``"log_return"`` in ``fitted.feature_names``. A positive scaled
    mean means the regime has above-training-average returns.
    """
    try:
        ret_idx = fitted.feature_names.index("log_return")
    except ValueError:
        ret_idx = 0

    rules: list[RegimeRule] = []
    for k in range(fitted.n_states):
        scaled_mean = float(fitted.hmm.means_[k, ret_idx])
        rules.append(RegimeRule(target_position=1.0 if scaled_mean > 0 else 0.0))

    return RegimeStrategyConfig(
        rules=rules,
        transaction_cost=transaction_cost,
        ann_factor=ann_factor,
    )


def _regime_conditional_returns(
    asset_returns: np.ndarray,
    regimes: np.ndarray,
    n_states: int,
) -> dict[int, float]:
    """Mean asset return per regime (test window)."""
    out: dict[int, float] = {}
    for k in range(n_states):
        mask = regimes == k
        out[k] = float(asset_returns[mask].mean()) if mask.any() else float("nan")
    return out


def _align_state_order(means: np.ndarray, return_idx: int) -> np.ndarray:
    """Permutation sorting states by emission mean of ``return_idx`` (ascending)."""
    return np.argsort(means[:, return_idx])


def _compute_parameter_dispersion(
    transmats: list[np.ndarray],
    means_list: list[np.ndarray],
    return_feature_idx: int,
) -> tuple[list[float], list[float]]:
    """Frobenius distances from cross-fold mean after state alignment.

    Parameters
    ----------
    transmats : list of (K, K) arrays
    means_list : list of (K, D) arrays
    return_feature_idx : int
        Feature index used for consistent state ordering across folds.

    Returns
    -------
    transmat_dists, means_dists : list[float], list[float]
        Per-fold Frobenius distances from the cross-fold mean.
        All ``NaN`` if fewer than two folds are provided.
    """
    n = len(transmats)
    nan_list: list[float] = [float("nan")] * n
    if n < 2:
        return nan_list, nan_list

    aligned_t: list[np.ndarray] = []
    aligned_m: list[np.ndarray] = []
    for transmat, means in zip(transmats, means_list):
        perm = _align_state_order(means, return_feature_idx)
        aligned_t.append(transmat[np.ix_(perm, perm)])
        aligned_m.append(means[perm])

    mean_t = np.mean(aligned_t, axis=0)
    mean_m = np.mean(aligned_m, axis=0)

    return (
        [float(np.linalg.norm(t - mean_t, "fro")) for t in aligned_t],
        [float(np.linalg.norm(m - mean_m, "fro")) for m in aligned_m],
    )


def _evaluate_fold(
    fold_id: int,
    cv_type: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X_raw: np.ndarray,
    asset_returns: np.ndarray,
    index: pd.DatetimeIndex,
    n_states: int,
    ann_factor: int,
    transaction_cost: float,
    train_kwargs: dict,
) -> tuple[FoldResult, np.ndarray, np.ndarray]:
    """Fit, decode (causal), backtest a single fold.

    Returns
    -------
    result : FoldResult
    transmat : np.ndarray  shape (K, K)
    means : np.ndarray  shape (K, D)
    """
    X_train = X_raw[train_idx]
    X_test = X_raw[test_idx]
    ret_test = asset_returns[test_idx]

    logger.info(
        "Fold %d (%s): train [%s .. %s] n=%d  test [%s .. %s] n=%d",
        fold_id, cv_type,
        index[train_idx[0]].date(), index[train_idx[-1]].date(), len(train_idx),
        index[test_idx[0]].date(), index[test_idx[-1]].date(), len(test_idx),
    )

    fitted = train_hmm(X_train, n_states, **train_kwargs)

    strategy = _default_strategy_config(fitted, ann_factor, transaction_cost)

    decoder = OnlineDecoder(fitted)
    posteriors = decoder.batch_filter(X_test)  # (n_test, K), causal
    regimes = posteriors.argmax(axis=1)

    bt = run_backtest(pd.Series(ret_test), regimes, strategy)
    ts = backtest_tearsheet(bt, ann_factor=ann_factor)

    rcr = _regime_conditional_returns(ret_test, regimes, n_states)

    result = FoldResult(
        fold_id=fold_id,
        cv_type=cv_type,
        train_start=index[train_idx[0]],
        train_end=index[train_idx[-1]],
        test_start=index[test_idx[0]],
        test_end=index[test_idx[-1]],
        n_train=int(len(train_idx)),
        n_test=int(len(test_idx)),
        sharpe=float(ts["sharpe"]),
        calmar=float(ts["calmar"]),
        max_drawdown=float(ts["max_drawdown"]),
        hit_rate=float(ts["win_rate"]),
        ann_return=float(ts["ann_return"]),
        ann_vol=float(ts["ann_vol"]),
        n_trades=int(ts["n_trades"]),
        bic=float(fitted.bic),
        log_likelihood=float(fitted.log_likelihood),
        n_params=_count_params(fitted),
        n_states_used=n_states,
        regime_conditional_returns=rcr,
    )

    return result, fitted.hmm.transmat_.copy(), fitted.hmm.means_.copy()


# ---------------------------------------------------------------------------
# Public orchestrators
# ---------------------------------------------------------------------------


def run_purged_cv(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    returns_col: str,
    n_states: int,
    *,
    train_bars: int,
    test_bars: int,
    embargo_bars: int = 24,
    ann_factor: int = 8760,
    transaction_cost: float = 0.0001,
    train_kwargs: dict | None = None,
    results_dir: Path | None = None,
    asset: str = "unknown",
) -> tuple[list[FoldResult], pd.DataFrame]:
    """Run purged k-fold CV with embargo and return per-fold results.

    Each fold:

    1. Trains an HMM on the train window.
    2. Builds a default strategy (long in above-average-return regimes).
    3. Decodes the test window causally via :class:`OnlineDecoder`.
    4. Runs :func:`rde.analysis.backtest.run_backtest`.
    5. Collects :class:`FoldResult`.

    After all folds, computes cross-fold parameter dispersion (Frobenius
    distances of aligned transition matrices and emission means).

    Parameters
    ----------
    df_features : pd.DataFrame
        Full feature + return DataFrame with a tz-aware DatetimeIndex.
    feature_cols : list[str]
        Columns to use as HMM feature matrix.
    returns_col : str
        Column with per-bar asset returns (used for backtesting).
    n_states : int
        Number of hidden states per fold.
    train_bars : int
        Training window length in bars.
    test_bars : int
        Test window length in bars.
    embargo_bars : int
        Gap between train end and test start (default 24).
    ann_factor : int
        Bars per year for Sharpe/Calmar annualisation (default 8760 for
        hourly crypto; use 1638 ≈ 252 × 6.5 for hourly equities).
    transaction_cost : float
        One-way transaction cost fraction (default 0.0001 = 1 bp).
    train_kwargs : dict | None
        Extra keyword arguments forwarded to :func:`rde.models.hmm.train_hmm`.
    results_dir : Path | None
        If provided, writes ``{results_dir}/purged_cv_{date}.parquet``.
    asset : str
        Asset label used in log messages.

    Returns
    -------
    fold_results : list[FoldResult]
        One entry per fold, with dispersion metrics filled.
    summary_df : pd.DataFrame
        One row per fold, suitable for parquet serialisation.

    Raises
    ------
    ValueError
        If ``df_features`` has no DatetimeIndex or fewer than
        ``train_bars + embargo_bars + test_bars`` rows.
    """
    if not isinstance(df_features.index, pd.DatetimeIndex):
        raise ValueError("df_features must have a DatetimeIndex.")

    min_obs = train_bars + embargo_bars + test_bars
    if len(df_features) < min_obs:
        raise ValueError(
            f"Need at least {min_obs} bars "
            f"(train={train_bars} + embargo={embargo_bars} + test={test_bars}), "
            f"got {len(df_features)}."
        )

    kw = dict(train_kwargs or {})
    kw.setdefault("feature_names", feature_cols)

    X_raw = df_features[feature_cols].values
    asset_returns = df_features[returns_col].values
    index = df_features.index

    ret_idx = feature_cols.index("log_return") if "log_return" in feature_cols else 0

    raw_folds: list[tuple[FoldResult, np.ndarray, np.ndarray]] = []
    for fold_id, (train_idx, test_idx) in enumerate(
        purged_k_fold_splits(
            len(df_features),
            train_bars=train_bars,
            test_bars=test_bars,
            embargo_bars=embargo_bars,
        )
    ):
        if len(train_idx) < n_states or len(test_idx) == 0:
            logger.warning("Fold %d: skipping (insufficient bars).", fold_id)
            continue
        raw_folds.append(
            _evaluate_fold(
                fold_id, "purged_kfold",
                train_idx, test_idx,
                X_raw, asset_returns, index,
                n_states, ann_factor, transaction_cost, kw,
            )
        )

    return _finalise_folds(raw_folds, results_dir, asset, return_feature_idx=ret_idx)


def run_combinatorial_purged_cv(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    returns_col: str,
    n_states: int,
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_bars: int = 24,
    ann_factor: int = 8760,
    transaction_cost: float = 0.0001,
    train_kwargs: dict | None = None,
    results_dir: Path | None = None,
    asset: str = "unknown",
) -> tuple[list[FoldResult], pd.DataFrame]:
    """Run combinatorial purged CV (de Prado AFML ch. 7).

    Generates all ``C(n_groups, n_test_groups)`` train/test splits, producing
    a **distribution** of fold Sharpe ratios. Each observation may appear in
    multiple test sets; this is by design (it is not a flaw).

    Parameters
    ----------
    df_features : pd.DataFrame
        Full feature + return DataFrame with a tz-aware DatetimeIndex.
    feature_cols : list[str]
        Columns to use as HMM feature matrix.
    returns_col : str
        Column with per-bar asset returns (used for backtesting).
    n_states : int
        Number of hidden states per fold.
    n_groups : int
        Number of equal-sized groups to split the series into (default 6).
    n_test_groups : int
        Groups used as test set in each combination (default 2).
    embargo_bars : int
        Gap before each test group (default 24).
    ann_factor : int
        Bars per year (default 8760 for hourly crypto).
    transaction_cost : float
        One-way transaction cost (default 0.0001 = 1 bp).
    train_kwargs : dict | None
        Extra kwargs forwarded to ``train_hmm``.
    results_dir : Path | None
        If provided, writes ``{results_dir}/combinatorial_cv_{date}.parquet``.
    asset : str
        Asset label for logging.

    Returns
    -------
    fold_results : list[FoldResult]
    summary_df : pd.DataFrame
    """
    if not isinstance(df_features.index, pd.DatetimeIndex):
        raise ValueError("df_features must have a DatetimeIndex.")

    kw = dict(train_kwargs or {})
    kw.setdefault("feature_names", feature_cols)

    X_raw = df_features[feature_cols].values
    asset_returns = df_features[returns_col].values
    index = df_features.index

    n_combinations = len(list(itertools.combinations(range(n_groups), n_test_groups)))
    logger.info(
        "%s combinatorial purged CV: %d groups, %d test groups → %d combinations",
        asset, n_groups, n_test_groups, n_combinations,
    )

    ret_idx = feature_cols.index("log_return") if "log_return" in feature_cols else 0

    raw_folds: list[tuple[FoldResult, np.ndarray, np.ndarray]] = []
    for fold_id, (train_idx, test_idx, test_groups) in enumerate(
        combinatorial_purged_splits(
            len(df_features),
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            embargo_bars=embargo_bars,
        )
    ):
        if len(train_idx) < n_states or len(test_idx) == 0:
            logger.warning(
                "Fold %d (groups %s): skipping (insufficient bars).",
                fold_id, test_groups,
            )
            continue
        raw_folds.append(
            _evaluate_fold(
                fold_id, "combinatorial_purged",
                train_idx, test_idx,
                X_raw, asset_returns, index,
                n_states, ann_factor, transaction_cost, kw,
            )
        )

    return _finalise_folds(
        raw_folds, results_dir, asset,
        cv_prefix="combinatorial_cv", return_feature_idx=ret_idx,
    )


# ---------------------------------------------------------------------------
# Shared finalisation
# ---------------------------------------------------------------------------


def _finalise_folds(
    raw_folds: list[tuple[FoldResult, np.ndarray, np.ndarray]],
    results_dir: Path | None,
    asset: str,
    cv_prefix: str = "purged_cv",
    return_feature_idx: int = 0,
) -> tuple[list[FoldResult], pd.DataFrame]:
    """Attach dispersion metrics, build summary DataFrame, optionally save."""
    if not raw_folds:
        logger.warning("%s: no folds completed.", asset)
        return [], pd.DataFrame()

    fold_results = [fr for fr, _, _ in raw_folds]
    transmats = [tm for _, tm, _ in raw_folds]
    means_list = [m for _, _, m in raw_folds]

    t_dists, m_dists = _compute_parameter_dispersion(
        transmats, means_list, return_feature_idx
    )

    for fr, td, md in zip(fold_results, t_dists, m_dists):
        fr.transmat_frobenius_from_mean = td
        fr.means_frobenius_from_mean = md

    summary_df = pd.DataFrame([fr.to_dict() for fr in fold_results])

    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{cv_prefix}_{date.today().isoformat()}.parquet"
        out_path = results_dir / filename
        summary_df.to_parquet(out_path, index=False)
        logger.info("%s: wrote %d-fold summary to %s", asset, len(fold_results), out_path)

    # Summary statistics log
    sharpes = [fr.sharpe for fr in fold_results if not np.isnan(fr.sharpe)]
    if sharpes:
        logger.info(
            "%s %s: %d folds  Sharpe mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
            asset, cv_prefix,
            len(sharpes),
            float(np.mean(sharpes)),
            float(np.std(sharpes)),
            float(np.min(sharpes)),
            float(np.max(sharpes)),
        )

    return fold_results, summary_df
