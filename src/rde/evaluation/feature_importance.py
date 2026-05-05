"""Permutation feature importance with fold-stability scoring (Phase 37.4).

Distinct from the ablation test in :mod:`rde.evaluation.skeptics`:

- **Ablation** (37.2): retrain the model without a feature → different HMM.
- **Permutation** (37.4): keep the fitted model, randomly shuffle one feature
  in the *test* data → same HMM, different predictions.

Permutation importance is cheaper (no refit) and follows the standard
scikit-learn convention. It is computed within each purged CV fold so the
importance estimate comes with a fold-level distribution, and stability is
assessed by the fraction of folds where a feature's importance is positive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rde.analysis.backtest import backtest_tearsheet, run_backtest
from rde.evaluation.purged_cv import _default_strategy_config, purged_k_fold_splits
from rde.inference.online import OnlineDecoder
from rde.models.hmm import FittedModel, train_hmm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FoldImportance:
    """Per-fold permutation importance for all features.

    Attributes
    ----------
    fold_id : int
    importances : dict[str, float]
        Mapping feature name → mean(model_sharpe − permuted_sharpe) over
        ``n_permutations`` permutations. Positive means the feature contributes
        positively to edge.
    """

    fold_id: int
    importances: dict[str, float]


@dataclass
class FeatureImportanceResult:
    """Aggregate permutation importance across all folds.

    Attributes
    ----------
    feature_names : list[str]
    fold_importances : list[FoldImportance]
    mean_importance : dict[str, float]
        Cross-fold mean importance per feature.
    std_importance : dict[str, float]
        Cross-fold std of importance per feature.
    positive_fold_fraction : dict[str, float]
        Fraction of folds where the feature's importance > 0.
    is_stable : dict[str, bool]
        ``True`` if ``positive_fold_fraction >= 0.7``.
    """

    feature_names: list[str]
    fold_importances: list[FoldImportance]
    mean_importance: dict[str, float]
    std_importance: dict[str, float]
    positive_fold_fraction: dict[str, float]
    is_stable: dict[str, bool]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _permutation_importance_single_fold(
    X_test_raw: np.ndarray,
    regimes_model: np.ndarray,
    model_sharpe: float,
    returns_test: np.ndarray,
    fitted: FittedModel,
    feature_cols: list[str],
    ann_factor: int,
    transaction_cost: float,
    n_permutations: int = 10,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Compute permutation importance for each feature within one test fold.

    For each feature:
    1. Permute that column ``n_permutations`` times.
    2. Decode the perturbed test window causally.
    3. Backtest with the same strategy config.
    4. Importance = ``mean(model_sharpe - permuted_sharpe)``.

    Parameters
    ----------
    X_test_raw : np.ndarray
        Raw (unscaled) feature matrix for the test window, shape (n_test, D).
    regimes_model : np.ndarray
        Regime labels from the unperturbed model decode, shape (n_test,).
    model_sharpe : float
        Pre-computed Sharpe for the unperturbed model on this fold.
    returns_test : np.ndarray
        Per-bar asset returns for the test window.
    fitted : FittedModel
        The trained HMM (used for decoding perturbed test data).
    feature_cols : list[str]
    ann_factor : int
    transaction_cost : float
    n_permutations : int
        Number of permutations per feature (default 10).
    rng : np.random.Generator | None
        Random number generator (created if ``None``).

    Returns
    -------
    dict[str, float]
        Feature name → mean importance score.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    strategy = _default_strategy_config(fitted, ann_factor, transaction_cost)
    ret_series = pd.Series(returns_test)
    decoder = OnlineDecoder(fitted)
    importances: dict[str, float] = {}

    for feat_idx, feat_name in enumerate(feature_cols):
        perm_sharpes: list[float] = []
        for _ in range(n_permutations):
            X_perturbed = X_test_raw.copy()
            perm = rng.permutation(len(X_perturbed))
            X_perturbed[:, feat_idx] = X_perturbed[perm, feat_idx]
            posteriors = decoder.batch_filter(X_perturbed)
            perm_regimes = posteriors.argmax(axis=1)
            bt = run_backtest(ret_series, perm_regimes, strategy)
            ts = backtest_tearsheet(bt, ann_factor=ann_factor)
            perm_sharpes.append(float(ts["sharpe"]))

        importances[feat_name] = float(model_sharpe - np.mean(perm_sharpes))

    return importances


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def run_permutation_importance(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    returns_col: str,
    n_states: int,
    *,
    train_bars: int,
    test_bars: int,
    embargo_bars: int = 24,
    n_permutations: int = 10,
    ann_factor: int = 8760,
    transaction_cost: float = 0.0001,
    train_kwargs: dict | None = None,
    seed: int = 42,
) -> FeatureImportanceResult:
    """Run purged k-fold CV and compute permutation importance in each fold.

    For each fold:
    1. Train HMM on the training window.
    2. Decode the test window causally (forward filter).
    3. Compute model Sharpe.
    4. For each feature, permute it ``n_permutations`` times and measure
       Sharpe degradation.

    Aggregates importances across folds into mean, std, and positive-fold
    fraction (stability indicator).

    Parameters
    ----------
    df_features : pd.DataFrame
        Full feature + return DataFrame with a tz-aware DatetimeIndex.
    feature_cols : list[str]
    returns_col : str
    n_states : int
    train_bars : int
    test_bars : int
    embargo_bars : int
    n_permutations : int
        Permutations per feature per fold (default 10).
    ann_factor : int
    transaction_cost : float
    train_kwargs : dict | None
    seed : int

    Returns
    -------
    FeatureImportanceResult
    """
    kw = dict(train_kwargs or {})
    kw.setdefault("feature_names", feature_cols)

    X_raw = df_features[feature_cols].values
    asset_returns = df_features[returns_col].values
    rng = np.random.default_rng(seed)

    fold_importances: list[FoldImportance] = []

    for fold_id, (train_idx, test_idx) in enumerate(
        purged_k_fold_splits(
            len(df_features),
            train_bars=train_bars,
            test_bars=test_bars,
            embargo_bars=embargo_bars,
        )
    ):
        if len(train_idx) < n_states or len(test_idx) == 0:
            continue

        X_train = X_raw[train_idx]
        X_test = X_raw[test_idx]
        ret_test = asset_returns[test_idx]

        try:
            fitted = train_hmm(X_train, n_states, **kw)
        except Exception:
            logger.warning("Permutation importance fold %d: train_hmm failed.", fold_id)
            continue

        strategy = _default_strategy_config(fitted, ann_factor, transaction_cost)
        decoder = OnlineDecoder(fitted)
        posteriors = decoder.batch_filter(X_test)
        regimes = posteriors.argmax(axis=1)

        bt = run_backtest(pd.Series(ret_test), regimes, strategy)
        ts = backtest_tearsheet(bt, ann_factor=ann_factor)
        model_sharpe = float(ts["sharpe"])

        fold_imp = _permutation_importance_single_fold(
            X_test_raw=X_test,
            regimes_model=regimes,
            model_sharpe=model_sharpe,
            returns_test=ret_test,
            fitted=fitted,
            feature_cols=feature_cols,
            ann_factor=ann_factor,
            transaction_cost=transaction_cost,
            n_permutations=n_permutations,
            rng=rng,
        )
        fold_importances.append(FoldImportance(fold_id=fold_id, importances=fold_imp))
        logger.info(
            "Fold %d importance: %s",
            fold_id,
            {k: f"{v:.3f}" for k, v in fold_imp.items()},
        )

    # Aggregate across folds
    n_folds = len(fold_importances)
    if n_folds == 0:
        empty: dict[str, float] = {f: float("nan") for f in feature_cols}
        return FeatureImportanceResult(
            feature_names=feature_cols,
            fold_importances=[],
            mean_importance=empty,
            std_importance=empty,
            positive_fold_fraction=empty,
            is_stable={f: False for f in feature_cols},
        )

    mean_imp: dict[str, float] = {}
    std_imp: dict[str, float] = {}
    pos_frac: dict[str, float] = {}

    for feat in feature_cols:
        vals = [fi.importances[feat] for fi in fold_importances if feat in fi.importances]
        if not vals:
            mean_imp[feat] = float("nan")
            std_imp[feat] = float("nan")
            pos_frac[feat] = float("nan")
        else:
            mean_imp[feat] = float(np.mean(vals))
            std_imp[feat] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            pos_frac[feat] = float(np.mean([v > 0 for v in vals]))

    is_stable = {f: pos_frac.get(f, 0.0) >= 0.7 for f in feature_cols}

    logger.info(
        "Permutation importance complete: %d folds. Mean importances: %s",
        n_folds,
        {k: f"{v:.3f}" for k, v in mean_imp.items()},
    )
    return FeatureImportanceResult(
        feature_names=feature_cols,
        fold_importances=fold_importances,
        mean_importance=mean_imp,
        std_importance=std_imp,
        positive_fold_fraction=pos_frac,
        is_stable=is_stable,
    )
