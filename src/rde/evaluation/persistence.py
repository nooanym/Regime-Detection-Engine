"""Regime persistence metrics: expected and empirical dwell times."""

import numpy as np


def expected_dwell_times(transmat: np.ndarray) -> np.ndarray:
    """Compute expected dwell time per state as 1 / (1 - p_ii).

    Parameters
    ----------
    transmat : np.ndarray
        Row-stochastic transition matrix of shape (n_states, n_states).

    Returns
    -------
    np.ndarray
        Expected dwell time (in bars) per state, shape (n_states,).

    Notes
    -----
    Derived from the geometric distribution: if p_ii is the self-transition
    probability, the expected number of consecutive bars in state i is
    1 / (1 - p_ii). Clipped to avoid division by zero when p_ii ≈ 1.

    """
    diag = np.diag(transmat)
    return 1.0 / np.maximum(1.0 - diag, 1e-9)


def empirical_dwell_times(states: np.ndarray) -> dict[int, np.ndarray]:
    """Compute empirical dwell times (run lengths) for each state.

    Parameters
    ----------
    states : np.ndarray
        Integer state sequence of shape (T,).

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from state index to an array of observed run lengths for
        that state. An empty sequence returns an empty dict.

    """
    if len(states) == 0:
        return {}

    result: dict[int, list[int]] = {}
    current = int(states[0])
    run = 1

    for s in states[1:]:
        s = int(s)
        if s == current:
            run += 1
        else:
            result.setdefault(current, []).append(run)
            current = s
            run = 1
    result.setdefault(current, []).append(run)

    return {k: np.array(v) for k, v in result.items()}
