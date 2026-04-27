"""Transition matrix analysis: stationary distribution and per-state entropy."""

import numpy as np


def stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of a row-stochastic transition matrix.

    Parameters
    ----------
    transmat : np.ndarray
        Row-stochastic transition matrix of shape (n_states, n_states).

    Returns
    -------
    np.ndarray
        Stationary distribution π of shape (n_states,), satisfying π @ transmat = π
        and π.sum() == 1.

    Notes
    -----
    Computed as the left eigenvector corresponding to eigenvalue 1 of the
    transpose of the transition matrix. For an ergodic chain this is unique.
    The result is normalised to sum to 1 to handle numerical imprecision.

    """
    eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
    idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
    stationary = np.real(eigenvectors[:, idx])
    stationary = stationary / stationary.sum()
    return stationary


def transition_entropy(transmat: np.ndarray) -> np.ndarray:
    """Compute per-state Shannon entropy of outgoing transition probabilities.

    Parameters
    ----------
    transmat : np.ndarray
        Row-stochastic transition matrix of shape (n_states, n_states).

    Returns
    -------
    np.ndarray
        Entropy (nats) for each row (state), shape (n_states,).
        High entropy means the state transitions unpredictably; zero entropy
        means it always transitions to the same successor.

    """
    safe = np.where(transmat > 0, transmat, 1.0)
    return -np.sum(transmat * np.log(safe), axis=1)
