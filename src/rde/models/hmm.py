"""GaussianHMM training with multi-restart Baum-Welch and AIC/BIC scoring."""

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class FittedModel:
    """Container for a trained Gaussian HMM and its associated metadata.

    Attributes
    ----------
    hmm : GaussianHMM
        The fitted hmmlearn estimator. Use with X already scaled by ``scaler``.
    scaler : StandardScaler
        The scaler fitted on the training data. Apply before passing X to hmm.
    n_states : int
        Number of hidden states.
    log_likelihood : float
        Total log-likelihood of the training data under the best restart.
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.
    feature_names : list[str]
        Names of the input features in column order.
    seed : int
        Random seed of the winning restart.
    all_restart_scores : list[float]
        Log-likelihood from each restart (-inf for failed restarts).

    """

    hmm: GaussianHMM
    scaler: StandardScaler
    n_states: int
    log_likelihood: float
    aic: float
    bic: float
    feature_names: list[str]
    seed: int
    all_restart_scores: list[float] = field(default_factory=list)


def _compute_n_params(n_states: int, n_features: int, covariance_type: str) -> int:
    """Compute the number of free parameters in a GaussianHMM.

    Parameters
    ----------
    n_states : int
        Number of hidden states.
    n_features : int
        Observation dimensionality.
    covariance_type : str
        One of "full", "diag", "tied", "spherical".

    Returns
    -------
    int
        Total number of free parameters.

    Notes
    -----
    Transition matrix contributes n_states*(n_states-1) free parameters
    (each row sums to 1). Initial distribution contributes n_states-1.
    Means contribute n_states*n_features. Covariance count depends on type.

    """
    trans_params = n_states * (n_states - 1)
    init_params = n_states - 1
    mean_params = n_states * n_features

    if covariance_type == "full":
        cov_params = n_states * n_features * (n_features + 1) // 2
    elif covariance_type == "diag":
        cov_params = n_states * n_features
    elif covariance_type == "tied":
        cov_params = n_features * (n_features + 1) // 2
    elif covariance_type == "spherical":
        cov_params = n_states
    else:
        raise ValueError(f"Unknown covariance_type: {covariance_type!r}")

    return trans_params + init_params + mean_params + cov_params


def train_hmm(
    X: np.ndarray,
    n_states: int,
    *,
    n_restarts: int = 10,
    covariance_type: str = "full",
    n_iter: int = 1000,
    init_strategy: str = "kmeans",
    seed_base: int = 42,
    feature_names: list[str] | None = None,
) -> FittedModel:
    """Fit a GaussianHMM with multiple random restarts; return the best by log-likelihood.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (T, n_features). Will be standardized internally.
    n_states : int
        Number of hidden states.
    n_restarts : int
        Number of Baum-Welch restarts. Must be ≥ 1.
    covariance_type : str
        One of "full", "diag", "tied", "spherical".
    n_iter : int
        Maximum EM iterations per restart.
    init_strategy : str
        Initialization strategy label (informational; hmmlearn uses k-means by default).
    seed_base : int
        Base random seed; restart i uses seed_base + i.
    feature_names : list[str] | None
        Names of the feature columns, used for diagnostics.

    Returns
    -------
    FittedModel
        The best-fitting model across all restarts.

    Raises
    ------
    RuntimeError
        If all restarts fail to produce a valid model.

    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    T, n_features = X_scaled.shape

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    logger.info(
        "Training GaussianHMM: n_states=%d, n_restarts=%d, covariance_type=%s, seed_base=%d",
        n_states,
        n_restarts,
        covariance_type,
        seed_base,
    )

    best_model: GaussianHMM | None = None
    best_score = -np.inf
    best_seed = seed_base
    all_scores: list[float] = []
    converged_count = 0

    for i in range(n_restarts):
        seed = seed_base + i
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=seed,
        )
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                model.fit(X_scaled)
                score = float(model.score(X_scaled))
                did_not_converge = any(
                    issubclass(w.category, ConvergenceWarning) for w in caught_warnings
                )
                if did_not_converge:
                    logger.warning(
                        "Restart %d (seed=%d) did not converge within %d iterations.",
                        i,
                        seed,
                        n_iter,
                    )
                else:
                    converged_count += 1

                all_scores.append(score)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_seed = seed
                    logger.debug("Restart %d: new best log-lik=%.4f", i, score)

            except Exception as exc:
                logger.warning("Restart %d (seed=%d) raised: %s", i, seed, exc)
                all_scores.append(-np.inf)

    if best_model is None:
        raise RuntimeError(
            f"All {n_restarts} restarts failed for n_states={n_states}. "
            "Check your data for NaNs, near-constant features, or insufficient observations."
        )

    logger.info(
        "Best restart: seed=%d  log-lik=%.4f  (%d/%d restarts converged)",
        best_seed,
        best_score,
        converged_count,
        n_restarts,
    )

    n_params = _compute_n_params(n_states, n_features, covariance_type)
    aic = 2.0 * n_params - 2.0 * best_score
    bic = n_params * np.log(T) - 2.0 * best_score

    return FittedModel(
        hmm=best_model,
        scaler=scaler,
        n_states=n_states,
        log_likelihood=best_score,
        aic=aic,
        bic=bic,
        feature_names=list(feature_names),
        seed=best_seed,
        all_restart_scores=all_scores,
    )
