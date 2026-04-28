"""Regime transition prediction (Phase 27).

Implements:

1. **RegimeTransitionPredictor** — logistic regression on lagged features
   to predict next-bar regime probabilities (supplements the HMM's
   Markov transition matrix with exogenous predictors).

2. **multi_step_transition** — K-step ahead regime probabilities via
   matrix exponentiation of the empirical transition matrix.

3. **persistence_forecast** — probability of remaining in the current
   regime for each of the next h bars.

4. **expected_regime_duration** — expected time until a regime switch
   given the current regime and the multi-step transition matrix.

5. **blend_hmm_logistic** — blends HMM forward-backward posteriors with
   logistic-regression next-step predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TransitionPredictorConfig:
    """Configuration for the logistic regime predictor.

    Attributes
    ----------
    lags : int
        Number of lagged bars of features to include.
    l2_reg : float
        L2 regularisation coefficient for the softmax weights.
    learning_rate : float
        Gradient-descent step size.
    n_iter : int
        Maximum gradient-descent iterations.
    tol : float
        Convergence tolerance on loss change.
    """

    lags: int = 1
    l2_reg: float = 0.01
    learning_rate: float = 0.1
    n_iter: int = 200
    tol: float = 1e-6


# ---------------------------------------------------------------------------
# Softmax helper
# ---------------------------------------------------------------------------


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-15)


def _cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    n = len(labels)
    log_p = np.log(probs[np.arange(n), labels] + 1e-15)
    return float(-log_p.mean())


# ---------------------------------------------------------------------------
# Logistic predictor
# ---------------------------------------------------------------------------


@dataclass
class RegimeTransitionPredictor:
    """Logistic (softmax) predictor of next-bar regime.

    Parameters
    ----------
    config : TransitionPredictorConfig

    Attributes
    ----------
    W_ : np.ndarray, shape (K, d_in)
        Learned weight matrix after fitting.
    b_ : np.ndarray, shape (K,)
        Learned biases.
    K_ : int
        Number of regimes (set after fit).
    d_in_ : int
        Input dimension (set after fit).
    """

    config: TransitionPredictorConfig = field(
        default_factory=TransitionPredictorConfig
    )

    def fit(
        self,
        features: pd.DataFrame | np.ndarray,
        states: np.ndarray,
    ) -> "RegimeTransitionPredictor":
        """Fit the softmax predictor.

        Parameters
        ----------
        features : array-like, shape (T, d)
            Feature matrix.
        states : np.ndarray, shape (T,)
            Viterbi-decoded or dominant-posterior regime at each bar.

        Returns
        -------
        self
        """
        cfg = self.config
        X = np.asarray(features, dtype=float)
        y = np.asarray(states, dtype=int)
        T, d = X.shape
        K = int(y.max()) + 1
        lags = cfg.lags

        # Build lagged design matrix: rows lags..T-1, next-bar target lags+1..T.
        Xl = np.hstack([X[lags - lag - 1: T - lag - 1] for lag in range(lags)])
        yt = y[lags:]  # next-bar label
        n = len(yt)
        d_in = Xl.shape[1]

        # Standardise.
        self._mean = Xl.mean(axis=0)
        self._std = Xl.std(axis=0) + 1e-10
        Xl = (Xl - self._mean) / self._std

        # Add bias column.
        Xl_b = np.column_stack([Xl, np.ones(n)])
        d_b = d_in + 1

        # Initialise weights.
        rng = np.random.default_rng(0)
        W = rng.standard_normal((K, d_b)) * 0.01

        prev_loss = np.inf
        lr = cfg.learning_rate

        for _ in range(cfg.n_iter):
            logits = Xl_b @ W.T  # (n, K)
            probs = _softmax(logits)
            loss = _cross_entropy(probs, yt) + 0.5 * cfg.l2_reg * float((W ** 2).sum())

            if abs(prev_loss - loss) < cfg.tol:
                break
            prev_loss = loss

            # Gradient: dL/dW = Xᵀ (probs - one_hot) / n + l2 * W
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(n), yt] = 1.0
            delta = (probs - one_hot) / n  # (n, K)
            grad = delta.T @ Xl_b + cfg.l2_reg * W  # (K, d_b)
            W -= lr * grad

        self.W_ = W[:, :d_in]
        self.b_ = W[:, d_in]
        self.K_ = K
        self.d_in_ = d_in
        self._lags = lags
        return self

    def predict_proba(
        self,
        features: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Predict next-bar regime probabilities.

        Parameters
        ----------
        features : array-like, shape (T, d)

        Returns
        -------
        np.ndarray, shape (T - lags, K)
        """
        X = np.asarray(features, dtype=float)
        T, d = X.shape
        lags = self._lags
        Xl = np.hstack([X[lags - lag - 1: T - lag] for lag in range(lags)])
        # Handle edge: only use rows that have full lag history.
        n = min(len(Xl), T - lags + 1)
        Xl = Xl[:n]
        Xl = (Xl - self._mean) / self._std
        logits = Xl @ self.W_.T + self.b_  # (n, K)
        return _softmax(logits)


# ---------------------------------------------------------------------------
# Multi-step transition
# ---------------------------------------------------------------------------


def multi_step_transition(
    transmat: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Compute h-step transition matrix via matrix exponentiation.

    Parameters
    ----------
    transmat : np.ndarray, shape (K, K)
        Row-stochastic one-step transition matrix.
    steps : int
        Number of steps ahead.

    Returns
    -------
    np.ndarray, shape (K, K)
        h-step row-stochastic matrix.
    """
    if steps <= 0:
        return np.eye(transmat.shape[0])
    if steps == 1:
        return transmat.copy()

    result = np.linalg.matrix_power(transmat, steps)
    # Re-normalise rows to guard against floating-point drift.
    row_sums = result.sum(axis=1, keepdims=True)
    result /= np.where(row_sums > 1e-15, row_sums, 1.0)
    return result


# ---------------------------------------------------------------------------
# Persistence forecast
# ---------------------------------------------------------------------------


def persistence_forecast(
    current_regime: int,
    transmat: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Probability of staying in ``current_regime`` for each bar 1..horizon.

    P(regime_t = k | regime_0 = k) for t = 1..horizon.

    Parameters
    ----------
    current_regime : int
    transmat : np.ndarray, shape (K, K)
    horizon : int

    Returns
    -------
    np.ndarray, shape (horizon,)
    """
    k = current_regime
    probs = np.zeros(horizon)
    row = transmat[k].copy()
    for h in range(horizon):
        mat_h = multi_step_transition(transmat, h + 1)
        probs[h] = float(mat_h[k, k])
    return probs


# ---------------------------------------------------------------------------
# Expected regime duration
# ---------------------------------------------------------------------------


def expected_regime_duration(
    current_regime: int,
    transmat: np.ndarray,
    max_horizon: int = 200,
) -> float:
    """Expected number of additional bars in current regime before first exit.

    For a geometric first-exit distribution with self-transition p_{kk}:

        E[T_exit - 1] = p_{kk} / (1 - p_{kk})

    ``max_horizon`` is unused but kept for API compatibility.

    Returns
    -------
    float
    """
    k = current_regime
    p_kk = float(transmat[k, k])
    if p_kk >= 1.0 - 1e-15:
        return float(max_horizon)
    if p_kk <= 0.0:
        return 0.0
    return p_kk / (1.0 - p_kk)


# ---------------------------------------------------------------------------
# Blend HMM posteriors with logistic predictions
# ---------------------------------------------------------------------------


def blend_hmm_logistic(
    hmm_posteriors: np.ndarray,
    logistic_probs: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Convex combination of HMM posteriors and logistic predictions.

    Parameters
    ----------
    hmm_posteriors : np.ndarray, shape (T, K)
    logistic_probs : np.ndarray, shape (T, K)
        Should have the same length as hmm_posteriors.
    alpha : float
        Weight on HMM posterior.  1.0 = pure HMM, 0.0 = pure logistic.

    Returns
    -------
    np.ndarray, shape (T, K)
        Row-normalised blended probabilities.
    """
    T_hmm = len(hmm_posteriors)
    T_log = len(logistic_probs)
    T = min(T_hmm, T_log)

    blended = alpha * hmm_posteriors[-T:] + (1.0 - alpha) * logistic_probs[-T:]
    row_sums = blended.sum(axis=1, keepdims=True)
    blended /= np.where(row_sums > 1e-15, row_sums, 1.0)
    return blended


# ---------------------------------------------------------------------------
# Regime transition entropy
# ---------------------------------------------------------------------------


def transition_entropy_trajectory(
    posteriors: np.ndarray,
    transmat: np.ndarray,
) -> np.ndarray:
    """Per-bar entropy of the one-step-ahead regime distribution.

    At each bar t, the one-step-ahead distribution is:
        q_t = posteriors[t] @ transmat

    H_t = -Σ_k q_tk log q_tk

    Parameters
    ----------
    posteriors : np.ndarray, shape (T, K)
    transmat : np.ndarray, shape (K, K)

    Returns
    -------
    np.ndarray, shape (T,)
    """
    P_norm = posteriors / (posteriors.sum(axis=1, keepdims=True) + 1e-15)
    q = P_norm @ transmat  # (T, K)
    q = q / (q.sum(axis=1, keepdims=True) + 1e-15)
    log_q = np.log(q + 1e-15)
    H = -np.sum(q * log_q, axis=1)
    return H
