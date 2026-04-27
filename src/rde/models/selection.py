"""AIC/BIC model selection across a candidate set of hidden-state counts."""

import logging

import numpy as np
import pandas as pd

from rde.models.hmm import FittedModel, _compute_n_params, train_hmm

logger = logging.getLogger(__name__)


def select_n_states(
    X: np.ndarray,
    candidate_states: list[int],
    *,
    criterion: str = "bic",
    **train_kwargs,
) -> tuple[FittedModel, pd.DataFrame]:
    """Train HMMs for each candidate n_states and select the best by AIC or BIC.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (T, n_features).
    candidate_states : list[int]
        State counts to evaluate, e.g. [2, 3, 4, 5, 6]. All must be ≥ 2.
    criterion : str
        "aic" or "bic". Lower is better.
    **train_kwargs
        Passed directly to ``train_hmm`` (n_restarts, covariance_type, etc.).

    Returns
    -------
    tuple[FittedModel, pd.DataFrame]
        The chosen model and a scores DataFrame with columns:
        [n_states, log_likelihood, aic, bic, n_params, converged_restarts].

    Raises
    ------
    ValueError
        If criterion is not "aic" or "bic".

    """
    if criterion not in ("aic", "bic"):
        raise ValueError(f"criterion must be 'aic' or 'bic', got {criterion!r}")

    _, n_features = X.shape
    covariance_type = train_kwargs.get("covariance_type", "full")

    rows = []
    models: dict[int, FittedModel] = {}

    for n in candidate_states:
        logger.info("Evaluating n_states=%d ...", n)
        fitted = train_hmm(X, n, **train_kwargs)
        models[n] = fitted
        n_params = _compute_n_params(n, n_features, covariance_type)
        converged = sum(s > -np.inf for s in fitted.all_restart_scores)
        rows.append(
            {
                "n_states": n,
                "log_likelihood": fitted.log_likelihood,
                "aic": fitted.aic,
                "bic": fitted.bic,
                "n_params": n_params,
                "converged_restarts": converged,
            }
        )

    scores_df = pd.DataFrame(rows)
    best_idx = int(scores_df[criterion].idxmin())
    best_n = int(scores_df.loc[best_idx, "n_states"])
    best_model = models[best_n]

    logger.info(
        "Selected n_states=%d by %s=%.4f",
        best_n,
        criterion.upper(),
        scores_df.loc[best_idx, criterion],
    )
    return best_model, scores_df
