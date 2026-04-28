"""Ensemble HMM: train M independent GaussianHMMs and average state-aligned posteriors.

Each of the M models is fitted with a different random seed. Because hidden-state
labels are arbitrary permutations of each other across independent runs, the states
of every non-reference model are aligned to the reference model (highest
log-likelihood) via the Hungarian algorithm before the ensemble posterior is formed.

This reduces variance in regime assignments and produces a calibrated uncertainty
estimate via Shannon entropy of the averaged posterior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from rde.models.hmm import FittedModel, train_hmm

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble HMM training.

    Attributes
    ----------
    n_models : int
        Number of independent HMM models to fit. Each uses a different seed_base.
    n_restarts_each : int
        Number of Baum-Welch restarts per individual model.
    seed_step : int
        The seed_base for model m is ``m * seed_step``.
    align_states : bool
        Whether to run Hungarian-algorithm state alignment before averaging.
        Set to False only for debugging.
    """

    n_models: int = 5
    n_restarts_each: int = 3
    seed_step: int = 100
    align_states: bool = True


@dataclass
class EnsembleResult:
    """Output of :func:`train_ensemble_hmm`.

    Attributes
    ----------
    models : list[FittedModel]
        All M fitted models (before state alignment).
    reference_model : FittedModel
        The model with the highest log-likelihood. Used as the alignment
        reference. Equal to ``models[argmax log_likelihood]``.
    ensemble_posteriors : np.ndarray
        Shape (T, K). State-aligned, model-averaged posteriors.
        Rows sum to 1.0.
    per_model_posteriors : np.ndarray
        Shape (M, T, K). Individual smoothed posteriors after alignment.
        Rows of each slice sum to 1.0.
    entropy : np.ndarray
        Shape (T,). Shannon entropy of ``ensemble_posteriors`` at each bar.
        Equals 0 when all probability mass is on one state, log(K) when
        uniform.
    agreement : np.ndarray
        Shape (T,). Normalised inter-model agreement in [0, 1].
        Computed as ``1 - H_t / log(K)``. 1 = all mass on a single state,
        0 = completely uniform ensemble posterior.
    alignment_permutations : list[np.ndarray]
        Length M. ``alignment_permutations[m][k]`` is the reference-model
        state that corresponds to state k of model m. For the reference
        model itself this is the identity permutation.
    """

    models: list[FittedModel]
    reference_model: FittedModel
    ensemble_posteriors: np.ndarray
    per_model_posteriors: np.ndarray
    entropy: np.ndarray
    agreement: np.ndarray
    alignment_permutations: list[np.ndarray]


def align_states(
    reference_posteriors: np.ndarray,
    other_posteriors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align ``other_posteriors`` to ``reference_posteriors`` via the Hungarian algorithm.

    For each pair of states (j in reference, k in other), the overlap is
    O[j, k] = Σ_t ref[t, j] * other[t, k].  The Hungarian algorithm finds the
    permutation π that maximises Σ_k O[π(k), k], effectively mapping each state
    of ``other`` to the most-overlapping state of ``reference``.

    Parameters
    ----------
    reference_posteriors : np.ndarray
        Shape (T, K). Smoothed posteriors from the reference model. Rows sum to 1.
    other_posteriors : np.ndarray
        Shape (T, K). Smoothed posteriors from another model. Rows sum to 1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(aligned_posteriors, permutation)`` where:

        * ``permutation`` is an integer array of length K.
          ``permutation[k]`` gives the reference-model state that state k of
          ``other`` corresponds to.
        * ``aligned_posteriors`` has shape (T, K) and is ``other_posteriors``
          reordered so that column k corresponds to reference state k.

    Notes
    -----
    The permutation is computed from the *inverse* of the linear_sum_assignment
    output: linear_sum_assignment returns (row_ind, col_ind) minimising the
    cost matrix, i.e. col_ind[k] is the other-model state assigned to reference
    state k.  We invert this to get the mapping from other → reference.
    """
    T, K = reference_posteriors.shape
    if other_posteriors.shape != (T, K):
        raise ValueError(
            f"reference_posteriors shape {reference_posteriors.shape} does not "
            f"match other_posteriors shape {other_posteriors.shape}."
        )

    # Overlap matrix O[j, k] = Σ_t ref[t,j] * other[t,k]
    overlap = reference_posteriors.T @ other_posteriors  # (K, K)

    # linear_sum_assignment minimises; we negate to maximise.
    # row_ind[i] = reference state, col_ind[i] = other state assigned to it.
    row_ind, col_ind = linear_sum_assignment(-overlap)

    # permutation[k] = reference state for other state k
    permutation = np.empty(K, dtype=int)
    permutation[col_ind] = row_ind

    # Reorder other's columns: column k becomes the posterior for reference state k
    # aligned[:,k] = other[:,col_ind[k]]  (col_ind[k] is the other-state for ref-state k)
    aligned_posteriors = other_posteriors[:, col_ind[np.argsort(row_ind)]]

    return aligned_posteriors, permutation


def _compute_entropy(posteriors: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row of a probability matrix.

    Parameters
    ----------
    posteriors : np.ndarray
        Shape (T, K). Values should be non-negative and sum to 1 per row.

    Returns
    -------
    np.ndarray
        Shape (T,). Per-bar entropy H_t = -Σ_k p_k * log(p_k + ε).
    """
    eps = 1e-10
    return -np.sum(posteriors * np.log(posteriors + eps), axis=1)


def train_ensemble_hmm(
    X: np.ndarray,
    n_states: int,
    posteriors_fn,
    config: EnsembleConfig | None = None,
    **train_kwargs,
) -> EnsembleResult:
    """Train an ensemble of HMMs and average their state-aligned posteriors.

    Each of the M models is trained independently with ``seed_base = m * seed_step``.
    The model with the highest log-likelihood is selected as the alignment reference.
    All other models' states are permuted to match the reference via the Hungarian
    algorithm (maximising posterior overlap). The ensemble posterior is then the
    simple average of all M aligned posteriors.

    Parameters
    ----------
    X : np.ndarray
        Shape (T, d). Raw (unscaled) feature matrix.
    n_states : int
        Number of hidden states.
    posteriors_fn : callable
        ``posteriors_fn(fitted_model: FittedModel, X_scaled: np.ndarray) -> np.ndarray``
        where the return value has shape (T, K).

        Pass :func:`rde.inference.smoothing.forward_backward_posteriors`.
        The function is called with the **scaled** observation matrix
        (``fitted_model.scaler.transform(X)``).
    config : EnsembleConfig, optional
        Ensemble hyper-parameters.  Defaults to ``EnsembleConfig()``.
    **train_kwargs
        Forwarded verbatim to :func:`rde.models.hmm.train_hmm`.
        Common options: ``covariance_type``, ``emission_dist``, ``n_iter``.
        Note: ``n_restarts`` and ``seed_base`` are set by ``config`` and must
        **not** be passed here.

    Returns
    -------
    EnsembleResult

    Raises
    ------
    RuntimeError
        If all M individual model fits fail.

    Notes
    -----
    For M=1, no state alignment is performed and ``agreement`` is all-ones.

    The alignment step introduces a *label* invariance but cannot fix cases
    where two models have genuinely incompatible state structures (e.g.,
    one model collapses two states into one).  The ``entropy`` field is a
    good diagnostic for such cases: high entropy throughout the series
    indicates the ensemble has not converged on a consistent labelling.
    """
    if config is None:
        config = EnsembleConfig()

    # Disallow callers from passing seed_base / n_restarts directly
    for forbidden in ("seed_base", "n_restarts"):
        if forbidden in train_kwargs:
            raise ValueError(
                f"'{forbidden}' must not be passed in train_kwargs when using "
                "train_ensemble_hmm; it is controlled by EnsembleConfig."
            )

    T = X.shape[0]

    # ── Train all M models ────────────────────────────────────────────────────
    models: list[FittedModel] = []
    for m in range(config.n_models):
        seed_base = m * config.seed_step
        logger.info(
            "Ensemble: fitting model %d/%d  seed_base=%d  n_states=%d",
            m + 1, config.n_models, seed_base, n_states,
        )
        model = train_hmm(
            X,
            n_states=n_states,
            n_restarts=config.n_restarts_each,
            seed_base=seed_base,
            **train_kwargs,
        )
        models.append(model)
        logger.debug(
            "Model %d: log_lik=%.4f  seed=%d", m, model.log_likelihood, model.seed
        )

    # ── Select reference model (best log-likelihood) ──────────────────────────
    ref_idx = int(np.argmax([m.log_likelihood for m in models]))
    reference_model = models[ref_idx]
    logger.info(
        "Ensemble reference: model %d  log_lik=%.4f",
        ref_idx, reference_model.log_likelihood,
    )

    # ── Compute raw posteriors for every model ────────────────────────────────
    raw_posteriors: list[np.ndarray] = []
    for m, model in enumerate(models):
        X_scaled = model.scaler.transform(X)
        post = posteriors_fn(model.hmm, X_scaled)  # (T, K)
        if post.shape != (T, n_states):
            raise RuntimeError(
                f"posteriors_fn returned shape {post.shape} for model {m}; "
                f"expected ({T}, {n_states})."
            )
        raw_posteriors.append(post)

    ref_posteriors = raw_posteriors[ref_idx]

    # ── Align states and collect permutations ────────────────────────────────
    aligned: list[np.ndarray] = []
    permutations: list[np.ndarray] = []

    for m, post in enumerate(raw_posteriors):
        if m == ref_idx or not config.align_states or n_states == 1:
            # Reference model: identity permutation
            perm = np.arange(n_states)
            aligned.append(post)
        else:
            aligned_post, perm = align_states(ref_posteriors, post)
            aligned.append(aligned_post)

        permutations.append(perm)
        logger.debug("Model %d permutation: %s", m, perm)

    # ── Ensemble average ──────────────────────────────────────────────────────
    per_model_arr = np.stack(aligned, axis=0)  # (M, T, K)
    ensemble_post = per_model_arr.mean(axis=0)  # (T, K)

    # ── Entropy and agreement ─────────────────────────────────────────────────
    entropy = _compute_entropy(ensemble_post)

    if n_states > 1:
        max_entropy = np.log(n_states)
        agreement = 1.0 - entropy / max_entropy
        # Clip to [0, 1] for numerical safety (log(p+eps) can push slightly negative)
        agreement = np.clip(agreement, 0.0, 1.0)
    else:
        agreement = np.ones(T)
        entropy = np.zeros(T)

    # For M=1, agreement is trivially all-ones regardless of entropy
    if config.n_models == 1:
        agreement = np.ones(T)

    logger.info(
        "Ensemble complete: mean_entropy=%.4f  mean_agreement=%.4f",
        float(entropy.mean()), float(agreement.mean()),
    )

    return EnsembleResult(
        models=models,
        reference_model=reference_model,
        ensemble_posteriors=ensemble_post,
        per_model_posteriors=per_model_arr,
        entropy=entropy,
        agreement=agreement,
        alignment_permutations=permutations,
    )
