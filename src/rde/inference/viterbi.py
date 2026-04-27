"""Viterbi algorithm wrapper: compute MAP state sequence."""

import numpy as np
from hmmlearn.hmm import GaussianHMM


def viterbi_decode(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Compute the MAP state sequence via the Viterbi algorithm.

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
        Integer array of shape (T,) giving the most likely state at each step.

    """
    _, states = model.decode(X, algorithm="viterbi")
    return states.astype(int)
