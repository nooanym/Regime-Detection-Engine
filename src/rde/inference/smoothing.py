"""Forward-backward algorithm wrapper: compute posterior state probabilities."""

import numpy as np
from hmmlearn.hmm import GaussianHMM


def forward_backward_posteriors(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Compute posterior state probabilities via the forward-backward algorithm.

    Parameters
    ----------
    model : GaussianHMM
        A fitted hmmlearn model.
    X : np.ndarray
        Observation matrix of shape (T, n_features). Must already be scaled
        with the scaler stored in the associated FittedModel.

    Returns
    -------
    np.ndarray
        Array of shape (T, n_states) where entry [t, k] = P(state_t=k | X).
        Rows sum to 1.0.

    """
    return model.predict_proba(X)
