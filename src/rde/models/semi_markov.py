"""Hidden Semi-Markov Model (HSMM) duration correction for regime detection.

Replaces the implicit geometric state-duration distribution of a standard HMM
with an explicit Negative Binomial duration model per state.  The Negative
Binomial is flexible enough to represent "minimum-duration" regimes (e.g. a
crisis that cannot last fewer than 3 days) while still being tractable to fit
from observed dwell-time histograms via Method of Moments.

This module does NOT re-train a new model.  It takes the Viterbi path and
forward-backward posteriors produced by a ``FittedModel`` (standard Gaussian
or Student-t HMM) and returns *corrected* posteriors that up-weight durations
consistent with the Negative Binomial prior and down-weight durations that are
too short or too long relative to what was empirically observed.

References
----------
Yu, S.-Z. (2010).  Hidden semi-Markov models.  Artificial Intelligence,
174(2), 215-243.

Ferguson, J. D. (1980).  Variable duration models for speech.  Proceedings of
the Symposium on the Application of Hidden Markov Models to Text and Speech.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import nbinom

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HSMMConfig:
    """Hyper-parameters that govern HSMM duration correction.

    Attributes
    ----------
    d_max : int
        Maximum modelled duration in bars.  Probability mass beyond this
        point is truncated (re-normalised).  Default: 200.
    min_r : float
        Minimum allowed shape parameter r_k for the Negative Binomial.
        Prevents degenerate distributions at very short mean durations.
        Default: 0.5.
    max_r : float
        Maximum allowed shape parameter r_k.  Very large r implies a
        near-constant duration; the geometric asymptote is r→∞.
        Default: 50.0.
    min_p : float
        Minimum allowed extension probability p_k ∈ (0, 1).
        Default: 0.01.
    max_p : float
        Maximum allowed extension probability p_k ∈ (0, 1).
        Default: 0.99.

    """

    d_max: int = 200
    min_r: float = 0.5
    max_r: float = 50.0
    min_p: float = 0.01
    max_p: float = 0.99


# ---------------------------------------------------------------------------
# Duration distribution dataclass
# ---------------------------------------------------------------------------


@dataclass
class NegBinDuration:
    """Negative Binomial duration distribution for a single HMM state.

    The duration random variable D_k takes values d ∈ {1, 2, 3, …}:

        P(D_k = d) = NegBin(d - 1; r_k, 1 - p_k)

    where the scipy parameterisation NegBin(k; n, p) is:

        C(k+n-1, k) * p^n * (1-p)^k

    So with k = d - 1, n = r_k, p = 1 - p_k:

        P(D_k = d) = C(d+r_k-2, d-1) * (1-p_k)^r_k * p_k^(d-1)

    Expected duration: E[D_k] = r_k * p_k / (1 - p_k) + 1
    Variance:         Var[D_k] = r_k * p_k / (1 - p_k)^2

    Attributes
    ----------
    state : int
        Zero-based state index.
    r : float
        Shape parameter (number of successes) of the Negative Binomial.
        Must be > 0; can be fractional (generalised NegBin).
    p : float
        Extension probability in (0, 1).  Higher p → longer expected
        duration.
    mean_duration : float
        E[D_k] = r * p / (1 - p) + 1.
    std_duration : float
        sqrt(Var[D_k]) = sqrt(r * p / (1 - p)^2).

    """

    state: int
    r: float
    p: float
    mean_duration: float
    std_duration: float


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _negbin_mean_std(r: float, p: float) -> tuple[float, float]:
    """Return (mean_duration, std_duration) for NegBin(r, p) shifted to d ≥ 1."""
    # E[D] = r*p/(1-p) + 1,   Var[D] = r*p/(1-p)^2
    mean = r * p / (1.0 - p) + 1.0
    var = r * p / (1.0 - p) ** 2
    return mean, float(np.sqrt(max(var, 0.0)))


def fit_duration_params(
    dwell_arrays: dict[int, np.ndarray],
    config: HSMMConfig | None = None,
) -> list[NegBinDuration]:
    """Fit Negative Binomial duration distribution per state from empirical dwells.

    Uses Method of Moments (MOM):

        p_k = (Var - E) / Var    (matches NegBin variance formula)
        r_k = E * (1 - p_k) / p_k

    where E and Var are the empirical mean and variance of the observed dwell
    times for state k.  When there is only one observed dwell (Var = 0), or
    the variance is smaller than the mean (sub-geometric), the distribution
    degenerates and we fall back to fitting a shifted Geometric:

        r_k = 1,  p_k = 1 - 1/E

    Parameters
    ----------
    dwell_arrays : dict[int, np.ndarray]
        Mapping from state index to an array of observed run lengths (each ≥ 1).
        Typically produced by ``rde.evaluation.persistence.empirical_dwell_times``.
    config : HSMMConfig | None
        Configuration with clamping bounds.  Uses defaults if None.

    Returns
    -------
    list[NegBinDuration]
        One ``NegBinDuration`` per state key in ``dwell_arrays``, sorted by
        state index.

    Notes
    -----
    When the empirical variance ≤ empirical mean (Poisson-like or sub-dispersed
    data), the standard MOM estimator for p would be ≤ 0.  We fall back to the
    geometric (r=1) parameterisation in that case.

    The r and p values are clamped to [config.min_r, config.max_r] and
    [config.min_p, config.max_p] respectively after estimation.

    """
    cfg = config if config is not None else HSMMConfig()
    results: list[NegBinDuration] = []

    for state in sorted(dwell_arrays.keys()):
        arr = np.asarray(dwell_arrays[state], dtype=float)
        if arr.size == 0:
            # No observations: fall back to a unit geometric (r=1, p=0.5 → E=2)
            logger.warning(
                "State %d has no dwell observations; using default NegBin(r=1, p=0.5).",
                state,
            )
            r, p = 1.0, 0.5
        elif arr.size == 1:
            # Single observation: mean is known, variance is undefined; use geometric
            e = float(arr[0])
            p_geom = max(cfg.min_p, min(cfg.max_p, 1.0 - 1.0 / max(e, 1.0)))
            r, p = 1.0, p_geom
            logger.debug(
                "State %d has a single dwell; using geometric r=1, p=%.4f.", state, p
            )
        else:
            e = float(arr.mean())
            v = float(arr.var(ddof=1))
            if v > e:
                # Standard MOM: over-dispersed relative to Poisson
                p_mom = (v - e) / v
                r_mom = e * (1.0 - p_mom) / p_mom
            else:
                # Sub-dispersed or Poisson-like: fall back to shifted geometric
                logger.debug(
                    "State %d: Var(%.3f) ≤ Mean(%.3f); using geometric fallback.",
                    state,
                    v,
                    e,
                )
                p_mom = max(cfg.min_p, min(cfg.max_p, 1.0 - 1.0 / max(e, 1.0)))
                r_mom = 1.0

            r = float(np.clip(r_mom, cfg.min_r, cfg.max_r))
            p = float(np.clip(p_mom, cfg.min_p, cfg.max_p))

        mean_d, std_d = _negbin_mean_std(r, p)
        results.append(
            NegBinDuration(
                state=state,
                r=r,
                p=p,
                mean_duration=mean_d,
                std_duration=std_d,
            )
        )

    return results


def duration_log_pmf(
    d: int | np.ndarray,
    dur: NegBinDuration,
) -> float | np.ndarray:
    """Log P(duration = d) under NegBin(r, p) shifted so that durations start at 1.

    The duration variable takes values d ∈ {1, 2, 3, …}.  Internally this
    maps to k = d - 1 in scipy's NegBin parameterisation:

        P(D = d) = NegBin.pmf(d - 1, n=r, p=1 - self.p)

    scipy convention: NegBin(k; n, p) = C(k+n-1, k) * p^n * (1-p)^k,
    where p is the *success* probability (1 - extension probability in our
    convention).

    Parameters
    ----------
    d : int | np.ndarray
        Duration(s) to evaluate.  Must be ≥ 1; values < 1 return -inf.
    dur : NegBinDuration
        Duration parameters for this state.

    Returns
    -------
    float | np.ndarray
        Log probability (≤ 0).  Shape matches ``d`` for array input.

    Notes
    -----
    Scalar input returns a Python float.  Array input returns an ndarray of
    the same shape.

    """
    scalar = np.ndim(d) == 0
    d_arr = np.atleast_1d(np.asarray(d, dtype=float))
    out = np.full(d_arr.shape, -np.inf)
    valid = d_arr >= 1.0
    if valid.any():
        k = d_arr[valid] - 1.0  # shift: scipy's k = duration - 1
        log_p = nbinom.logpmf(k, dur.r, 1.0 - dur.p)
        out[valid] = log_p
    return float(out[0]) if scalar else out


# ---------------------------------------------------------------------------
# Core correction
# ---------------------------------------------------------------------------


def _current_run_lengths(viterbi_states: np.ndarray) -> np.ndarray:
    """Compute the current consecutive run length ending at each time step.

    Parameters
    ----------
    viterbi_states : np.ndarray
        Integer state sequence of shape (T,).

    Returns
    -------
    np.ndarray
        Integer array of shape (T,) where run_lengths[t] is the number of
        consecutive bars (including t itself) immediately preceding and
        including t that share the same state as viterbi_states[t].

    Examples
    --------
    States: [0, 0, 0, 1, 1, 0]
    Run lengths: [1, 2, 3, 1, 2, 1]

    """
    T = len(viterbi_states)
    run_lengths = np.ones(T, dtype=int)
    for t in range(1, T):
        if viterbi_states[t] == viterbi_states[t - 1]:
            run_lengths[t] = run_lengths[t - 1] + 1
        else:
            run_lengths[t] = 1
    return run_lengths


def apply_hsmm_correction(
    viterbi_states: np.ndarray,
    posteriors: np.ndarray,
    durations: list[NegBinDuration],
    config: HSMMConfig | None = None,
) -> np.ndarray:
    """Apply duration-model correction to HMM posteriors.

    Reweights each bar's posterior probability by the ratio:

        w_t(k) = P_NegBin(r_t(k)) / P_Geometric(r_t(k))

    where:
    - r_t(k) is the length of the most recent consecutive run ending at t
      where the Viterbi path assigns state k (i.e., the current run length
      for the Viterbi-decoded state; for all other states it is set to 1
      as a conservative estimate of a potential single-bar excursion).
    - P_NegBin uses the fitted ``NegBinDuration`` parameters.
    - P_Geometric uses p_geom = 1 - 1/mean_duration, which gives the same
      mean duration as the NegBin but with geometric (memoryless) increments.

    The corrected weights are multiplied element-wise into the posterior,
    then each row is L1-renormalised so it sums to 1.

    Parameters
    ----------
    viterbi_states : np.ndarray
        MAP state sequence of shape (T,), typically from
        ``rde.inference.viterbi.viterbi_decode``.
    posteriors : np.ndarray
        Forward-backward smoothed posteriors of shape (T, K), where K is the
        number of states.  Each row should already sum to approximately 1.
    durations : list[NegBinDuration]
        One ``NegBinDuration`` per state, sorted by ``NegBinDuration.state``.
        Must have exactly K entries.
    config : HSMMConfig | None
        Configuration for ``d_max`` and clamping.  Uses defaults if None.

    Returns
    -------
    np.ndarray
        Corrected posteriors of shape (T, K).  Each row sums to 1, all
        values are non-negative.

    Raises
    ------
    ValueError
        If ``len(durations)`` does not match ``posteriors.shape[1]``.

    Notes
    -----
    This is a post-hoc approximation, not a full HSMM forward-backward pass.
    It is exact when the HMM and HSMM share emission parameters and differ
    only in the duration model, which is the case here (we warm-start from
    the HMM and correct only the duration weighting).

    When the Viterbi-decoded state at time t is state k, we use the current
    run length r_t(k) = (length of the consecutive run of k ending at t) to
    evaluate the duration probability.  For states j ≠ k we use r_t(j) = 1
    (a single-bar hypothetical entry); this is conservative because longer
    stays in non-Viterbi states are not evident from the Viterbi path alone.

    The geometric reference distribution uses the same *mean* duration as the
    NegBin, so the ratio w_t(k) is > 1 for durations that the NegBin
    considers more likely than the geometric, and < 1 otherwise.

    """
    cfg = config if config is not None else HSMMConfig()

    T, K = posteriors.shape
    if len(durations) != K:
        raise ValueError(
            f"len(durations)={len(durations)} must equal posteriors.shape[1]={K}."
        )

    # Build a lookup from state index to NegBinDuration for O(1) access
    dur_by_state: dict[int, NegBinDuration] = {dur.state: dur for dur in durations}

    # Current run length for the Viterbi-decoded state at each time step
    run_lengths = _current_run_lengths(viterbi_states)  # (T,)

    # Compute weight matrix (T, K)
    weights = np.ones((T, K), dtype=float)

    for k in range(K):
        if k not in dur_by_state:
            logger.warning("State %d has no fitted duration; skipping correction.", k)
            continue
        dur_k = dur_by_state[k]
        # Geometric reference: same mean duration as NegBin
        p_geom = float(np.clip(1.0 - 1.0 / max(dur_k.mean_duration, 1.0001),
                                cfg.min_p, cfg.max_p))

        for t in range(T):
            # Use actual run length when Viterbi assigns state k; else 1
            if viterbi_states[t] == k:
                d_t = int(min(run_lengths[t], cfg.d_max))
            else:
                d_t = 1

            # NegBin log-probability for this duration
            log_p_negbin = float(duration_log_pmf(d_t, dur_k))

            # Geometric log-probability: P_geom(d) = (1-p_geom) * p_geom^(d-1)
            log_p_geom = (
                np.log1p(-p_geom) + (d_t - 1) * np.log(p_geom)
                if d_t >= 1
                else -np.inf
            )

            # Weight = ratio of probabilities (in linear space)
            if np.isfinite(log_p_negbin) and np.isfinite(log_p_geom):
                weights[t, k] = np.exp(log_p_negbin - log_p_geom)
            elif np.isfinite(log_p_negbin) and not np.isfinite(log_p_geom):
                # Reference is zero but HSMM is non-zero: give large weight
                weights[t, k] = 1.0
            else:
                # NegBin assigns zero probability (d > d_max or invalid)
                weights[t, k] = 0.0

    # Apply weights to posteriors
    corrected = posteriors * weights  # (T, K)

    # Renormalise each row to sum to 1; fall back to uniform if all-zero
    row_sums = corrected.sum(axis=1, keepdims=True)  # (T, 1)
    zero_rows = (row_sums <= 0.0).ravel()
    if zero_rows.any():
        logger.warning(
            "%d time steps had all-zero corrected posteriors; "
            "reverting to original posteriors for those steps.",
            int(zero_rows.sum()),
        )
        corrected[zero_rows] = posteriors[zero_rows]
        row_sums[zero_rows] = posteriors[zero_rows].sum(axis=1, keepdims=True)

    # Final edge case: original posteriors were also all-zero at some step
    still_zero = (row_sums <= 0.0).ravel()
    if still_zero.any():
        corrected[still_zero] = 1.0 / K
        row_sums[still_zero] = 1.0

    corrected /= row_sums
    return corrected
