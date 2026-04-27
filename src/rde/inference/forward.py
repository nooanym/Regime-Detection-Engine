"""Forward algorithm wrapper: compute sequence log-likelihood."""

import numpy as np
from hmmlearn.hmm import GaussianHMM


def forward_log_likelihood(model: GaussianHMM, X: np.ndarray) -> float:
    """Compute the log-likelihood of the observation sequence under the model.

    Parameters
    ----------
    model : GaussianHMM
        A fitted hmmlearn model.
    X : np.ndarray
        Observation matrix of shape (T, n_features). Must already be scaled
        with the scaler stored in the associated FittedModel.

    Returns
    -------
    float
        log P(X | model).

    """
    return float(model.score(X))
