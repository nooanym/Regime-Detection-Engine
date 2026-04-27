"""Label stability analysis: ARI across independent HMM restarts."""

import logging

import numpy as np
from sklearn.metrics import adjusted_rand_score

from rde.inference.viterbi import viterbi_decode
from rde.models.hmm import FittedModel, train_hmm

logger = logging.getLogger(__name__)


def stability_across_restarts(
    X: np.ndarray,
    n_states: int,
    n_runs: int,
    **train_kwargs,
) -> dict:
    """Measure label stability by training n_runs independent models and computing pairwise ARI.

    Each run uses a different ``seed_base`` (spaced 100 apart from the base) to ensure
    independent initialisation. The Adjusted Rand Index (ARI) between every pair of
    Viterbi-decoded state sequences quantifies how consistently the model assigns regime
    labels across random starts.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (T, n_features).
    n_states : int
        Number of hidden states.
    n_runs : int
        Number of independent training runs.
    **train_kwargs
        Passed to ``train_hmm``. ``seed_base`` is overridden per run.

    Returns
    -------
    dict with keys:
        ari_matrix : np.ndarray of shape (n_runs, n_runs)
            Pairwise ARI scores. Diagonal is 1.0.
        mean_ari : float
            Mean of the upper-triangle ARI values (agreement across all pairs).
        min_ari : float
            Minimum pairwise ARI — the worst-case label disagreement.
        transmat_std : float
            Mean element-wise std of the transition matrix across runs.
        means_std : float
            Mean element-wise std of the state means across runs.
        models : list[FittedModel]
            All fitted models (useful for diagnostics).
        state_sequences : list[np.ndarray]
            Viterbi-decoded state sequence from each run.

    Notes
    -----
    ARI of 1.0 indicates perfect label agreement (up to permutation).
    ARI near 0 indicates near-random agreement.
    High mean_ari (> 0.9) and low transmat_std signal a stable solution.

    """
    base_seed = int(train_kwargs.pop("seed_base", 42))
    models: list[FittedModel] = []
    sequences: list[np.ndarray] = []

    for run_idx in range(n_runs):
        seed = base_seed + run_idx * 100
        logger.info("Stability run %d/%d  seed_base=%d", run_idx + 1, n_runs, seed)
        fitted = train_hmm(X, n_states, seed_base=seed, **train_kwargs)
        X_scaled = fitted.scaler.transform(X)
        states = viterbi_decode(fitted.hmm, X_scaled)
        models.append(fitted)
        sequences.append(states)

    ari_matrix = np.eye(n_runs)
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            ari = float(adjusted_rand_score(sequences[i], sequences[j]))
            ari_matrix[i, j] = ari
            ari_matrix[j, i] = ari

    upper = ari_matrix[np.triu_indices(n_runs, k=1)]
    mean_ari = float(upper.mean()) if len(upper) > 0 else 1.0
    min_ari  = float(upper.min())  if len(upper) > 0 else 1.0

    transmats = np.stack([m.hmm.transmat_ for m in models])  # (n_runs, n_states, n_states)
    means_arr = np.stack([m.hmm.means_    for m in models])  # (n_runs, n_states, n_features)

    logger.info(
        "Stability result: mean_ARI=%.4f  min_ARI=%.4f  transmat_std=%.4f",
        mean_ari,
        min_ari,
        float(transmats.std(axis=0).mean()),
    )

    return {
        "ari_matrix":   ari_matrix,
        "mean_ari":     mean_ari,
        "min_ari":      min_ari,
        "transmat_std": float(transmats.std(axis=0).mean()),
        "means_std":    float(means_arr.std(axis=0).mean()),
        "models":       models,
        "state_sequences": sequences,
    }
