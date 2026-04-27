"""Online (causal) forward filtering for real-time regime state estimation.

The standard forward-backward algorithm is non-causal: it uses the entire
observation sequence to compute P(state_t | x_{1:T}). For live trading we
can only condition on past data, so this module implements the forward
(filtering) pass only:

    P(state_t = k | x_{1:t})

This is strictly causal — no future leakage. The filtered posterior is less
certain than the smoothed posterior but is the correct quantity to use when
producing signals at the current bar.

The filtering recursion (in log space for numerical stability):

    log α_t(k) = log b_k(x_t) + logsumexp_j [ log A[j,k] + log α_{t-1}(j) ]

    P(state_t | x_{1:t}) = softmax(log α_t)

References
----------
Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected
applications in speech recognition. Proc. IEEE, 77(2), 257-286.
"""

from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler

from rde.models.hmm import FittedModel


class OnlineDecoder:
    """Causal filtered-posterior estimator for a fitted HMM.

    Call :meth:`step` for each new observation to get the current filtered
    state posterior without looking ahead, or :meth:`batch_filter` to process
    a full array in a single call (sequentially, with no future leakage).

    Parameters
    ----------
    fitted : FittedModel
        The trained model. Both ``fitted.hmm`` and ``fitted.scaler`` are used.

    Attributes
    ----------
    n_states : int
        Number of hidden states.
    filtered_posterior : np.ndarray
        Shape (K,). The most recently computed filtered posterior. Updated by
        :meth:`step`. Initialised to the model's start probability distribution.
    """

    def __init__(self, fitted: FittedModel) -> None:
        self._model: GaussianHMM = fitted.hmm
        self._scaler: StandardScaler = fitted.scaler
        self.n_states: int = fitted.n_states

        log_start = np.log(np.clip(self._model.startprob_, 1e-300, None))
        self._log_alpha: np.ndarray = log_start - logsumexp(log_start)
        self._log_transmat: np.ndarray = np.log(
            np.clip(self._model.transmat_, 1e-300, None)
        )

    @property
    def filtered_posterior(self) -> np.ndarray:
        """Current filtered state probabilities — shape (K,), sums to 1."""
        return np.exp(self._log_alpha)

    def reset(self) -> None:
        """Reset state to the model's start probability distribution."""
        log_start = np.log(np.clip(self._model.startprob_, 1e-300, None))
        self._log_alpha = log_start - logsumexp(log_start)

    def step(self, x: np.ndarray) -> np.ndarray:
        """Update filtered posterior with a single new observation.

        Parameters
        ----------
        x : np.ndarray
            Shape (n_features,). A single raw (unscaled) feature vector.

        Returns
        -------
        np.ndarray
            Shape (K,). Updated filtered posterior P(state_t | x_{1:t}).
            Sums to 1.

        Notes
        -----
        Updates ``self._log_alpha`` in-place. Call :meth:`reset` to restart
        from the start distribution (e.g. when processing a new independent
        sequence).
        """
        x_scaled = self._scaler.transform(x.reshape(1, -1))
        log_b = self._model._compute_log_likelihood(x_scaled)[0]   # (K,)

        # Predict: log Σ_j A[j,k] α_{t-1}(j)  for each k
        # log_transmat[j, k] + log_alpha[j], logsumexp over j
        log_pred = logsumexp(
            self._log_transmat + self._log_alpha[:, np.newaxis],
            axis=0,
        )  # (K,)

        log_alpha_new = log_b + log_pred
        self._log_alpha = log_alpha_new - logsumexp(log_alpha_new)
        return self.filtered_posterior.copy()

    def batch_filter(self, X: np.ndarray) -> np.ndarray:
        """Sequentially apply :meth:`step` over an observation array.

        Resets state before processing so the result is independent of any
        previous calls. The loop is sequential (each step depends on the
        previous alpha), but the log-emission computation is vectorised.

        Parameters
        ----------
        X : np.ndarray
            Shape (T, n_features). Raw (unscaled) observations.

        Returns
        -------
        np.ndarray
            Shape (T, K). Filtered posteriors; row t uses only x_{1:t}.
            Every row sums to 1.

        Notes
        -----
        Compare to :func:`rde.inference.smoothing.forward_backward_posteriors`:
        that function conditions on the full sequence (non-causal). Use this
        method when you need strictly causal inference.
        """
        self.reset()
        X_scaled = self._scaler.transform(X)
        log_b = self._model._compute_log_likelihood(X_scaled)   # (T, K)
        T, K = log_b.shape
        posteriors = np.empty((T, K))

        for t in range(T):
            log_pred = logsumexp(
                self._log_transmat + self._log_alpha[:, np.newaxis],
                axis=0,
            )
            log_alpha_new = log_b[t] + log_pred
            self._log_alpha = log_alpha_new - logsumexp(log_alpha_new)
            posteriors[t] = self.filtered_posterior

        return posteriors
