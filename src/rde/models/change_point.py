"""Bayesian Online Change-Point Detection (BOCPD) — Phase 17.

Implements the algorithm of Adams & MacKay (2007) with a Gaussian
likelihood and Normal-Gamma conjugate prior.  At each time step the
algorithm maintains a distribution over the *run length* — the number
of observations since the last change point — and updates it exactly
using Bayes' rule.

References
----------
Adams, R. P., & MacKay, D. J. C. (2007).
    Bayesian Online Changepoint Detection.
    *arXiv preprint* arXiv:0710.3742.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BOCPDConfig:
    """Hyper-parameters for BOCPD with Normal-Gamma observations.

    Attributes
    ----------
    hazard : float
        Constant hazard rate *h* (probability of a change point at any
        step under the geometric process prior).  E.g. 1/200 = expected
        run length of 200 bars.
    mu0 : float
        Prior mean on the mean parameter.
    kappa0 : float
        Prior pseudo-count on the mean (controls how quickly the mean
        adapts).
    alpha0 : float
        Prior shape of the precision (Gamma).
    beta0 : float
        Prior rate of the precision (Gamma).  When ``auto_scale_prior``
        is True (default), this is multiplied by the sample variance of
        the input series so the predictive distribution is calibrated to
        the actual data scale.
    auto_scale_prior : bool
        If True (default), scale ``beta0`` by the empirical variance of
        the input before fitting.  This is critical for financial return
        series where the true variance (e.g. 1e-4) is orders of magnitude
        smaller than the conventional ``beta0=1.0``.
    threshold : float
        Minimum posterior probability mass on run-length 0 to declare a
        change point.  Used by :func:`detect_change_points`.
    min_run : int
        Minimum run length between successive change points (suppresses
        rapid toggling).
    """

    hazard: float = 1.0 / 200.0
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1.0
    auto_scale_prior: bool = True
    threshold: float = 0.3
    min_run: int = 10


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class BOCPDResult:
    """Output of :func:`bocpd`.

    Attributes
    ----------
    run_length_posterior : np.ndarray
        Shape ``(T, T+1)``.  Row *t* is the posterior distribution over
        run lengths 0..t after observing observation *t*.
        **Memory-intensive for large T** — use ``run_length_probs`` for
        a summary.
    run_length_probs : pd.Series
        Maximum a-posteriori run length at each time step (index matches
        input).
    change_point_probs : pd.Series
        ``P(run_length = 0 | observations_1..t)`` — the probability of a
        change point at each time step.
    change_points : pd.Index
        Detected change-point locations after thresholding.
    """

    run_length_posterior: np.ndarray
    run_length_probs: pd.Series
    change_point_probs: pd.Series
    change_points: pd.Index


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def bocpd(
    series: pd.Series | np.ndarray,
    config: BOCPDConfig | None = None,
) -> BOCPDResult:
    """Run Bayesian Online Change-Point Detection on a univariate series.

    Uses a Normal-Gamma conjugate model: the observations within each
    segment are i.i.d. Gaussian with unknown mean and precision.  The
    conjugate prior is Normal-Gamma(mu0, kappa0, alpha0, beta0).

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Univariate observations.  NaN values are forward-filled before
        processing.
    config : BOCPDConfig, optional
        Algorithm hyper-parameters.

    Returns
    -------
    BOCPDResult
    """
    cfg = config or BOCPDConfig()
    if isinstance(series, pd.Series):
        idx = series.index
        x = series.ffill().values.astype(float)
    else:
        idx = pd.RangeIndex(len(series))
        x = np.asarray(series, dtype=float)
        # forward-fill NaN
        mask = np.isnan(x)
        if mask.any():
            x = pd.Series(x).ffill().values

    T = len(x)
    h = cfg.hazard

    # Calibrate beta0 to the observed data scale so the predictive
    # distribution is not dominated by a miscalibrated prior.
    beta0_eff = cfg.beta0
    if cfg.auto_scale_prior:
        emp_var = float(np.var(x[~np.isnan(x)]))
        if emp_var > 0:
            beta0_eff = cfg.beta0 * emp_var

    # Sufficient statistics per run-length hypothesis.
    # Each hypothesis tracks (mu, kappa, alpha, beta) of the Normal-Gamma.
    # We maintain arrays indexed by run length r in 0..t.
    mu = np.full(T + 1, cfg.mu0)
    kappa = np.full(T + 1, cfg.kappa0)
    alpha = np.full(T + 1, cfg.alpha0)
    beta = np.full(T + 1, beta0_eff)

    # Run-length posteriors R[t, r] = P(run_length=r | x_1..t)
    R = np.zeros((T, T + 1))
    R[0, 0] = 1.0  # at t=0 before seeing data, run-length=0 with prob 1

    cp_probs = np.zeros(T)
    map_rl = np.zeros(T, dtype=int)

    for t in range(T):
        xt = x[t]

        # Predictive probability under each run-length hypothesis.
        # Student-t predictive: p(x | mu, kappa, alpha, beta)
        # t-distribution with 2*alpha dof, location mu, scale sqrt(beta*(kappa+1)/(alpha*kappa))
        dof = 2.0 * alpha[: t + 1]
        scale2 = beta[: t + 1] * (kappa[: t + 1] + 1.0) / (
            alpha[: t + 1] * kappa[: t + 1]
        )
        scale = np.sqrt(scale2)
        pred_prob = _student_t_pdf(xt, dof, mu[: t + 1], scale)

        # R[t, :t+1] = posterior over run lengths after seeing x[0..t-1].
        # At t=0 this is the prior R[0,0]=1.  At t>0 it was stored in the
        # previous iteration.

        # Growth: each current run-length r survives (no CP) → becomes r+1.
        # Predictive uses the posterior after r observations (params indexed by r).
        growth = pred_prob * R[t, : t + 1] * (1.0 - h)

        # Change point: any run-length collapses to 0.  The predictive for the
        # NEW run-length-0 hypothesis uses the PRIOR (pred_prob[0]) because we
        # have 0 observations in the new segment.  Total CP probability is
        # pred_prob[0] * h * Σ_r R[t,r] = pred_prob[0] * h  (R normalised to 1).
        cp_mass = float(pred_prob[0]) * h

        new_R = np.zeros(T + 1)
        new_R[1 : t + 2] = growth
        new_R[0] = cp_mass

        # Normalise.
        total = new_R.sum()
        if total > 1e-300:
            new_R /= total
        else:
            new_R[0] = 1.0  # numerical underflow — reset

        if t + 1 < T:
            R[t + 1, :] = new_R
        cp_probs[t] = new_R[0]
        map_rl[t] = int(np.argmax(new_R[: t + 2]))

        # Update sufficient statistics for each run-length hypothesis.
        # For r > 0: the hypothesis that the run started at t-r updates
        # the Normal-Gamma with one new observation.
        kn = kappa[: t + 1] + 1.0
        mun = (kappa[: t + 1] * mu[: t + 1] + xt) / kn
        an = alpha[: t + 1] + 0.5
        bn = (
            beta[: t + 1]
            + 0.5 * kappa[: t + 1] * (xt - mu[: t + 1]) ** 2 / kn
        )

        # Shift: run-lengths increase by 1 at the next step.
        mu[1 : t + 2] = mun
        kappa[1 : t + 2] = kn
        alpha[1 : t + 2] = an
        beta[1 : t + 2] = bn
        # Reset run-length=0 to prior.
        mu[0] = cfg.mu0
        kappa[0] = cfg.kappa0
        alpha[0] = cfg.alpha0
        beta[0] = beta0_eff

    cp_series = pd.Series(cp_probs, index=idx, name="change_point_prob")
    rl_series = pd.Series(map_rl, index=idx, name="map_run_length")

    cp_locs = _detect_change_points(cp_series, cfg.threshold, cfg.min_run)

    return BOCPDResult(
        run_length_posterior=R,
        run_length_probs=rl_series,
        change_point_probs=cp_series,
        change_points=cp_locs,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _student_t_pdf(
    x: float,
    dof: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Vectorised Student-t PDF (no scipy dependency)."""
    from math import lgamma

    # Use scipy if available for speed; fall back to manual implementation.
    try:
        from scipy.special import gammaln

        nu = dof
        z = (x - loc) / scale
        log_p = (
            gammaln(0.5 * (nu + 1.0))
            - gammaln(0.5 * nu)
            - 0.5 * np.log(nu * np.pi)
            - np.log(scale)
            - 0.5 * (nu + 1.0) * np.log(1.0 + z ** 2 / nu)
        )
        return np.exp(log_p)
    except ImportError:
        # Manual fallback: approximate with normal when dof > 30.
        z = (x - loc) / scale
        log_p = -0.5 * z ** 2 - np.log(scale) - 0.5 * np.log(2.0 * np.pi)
        return np.exp(log_p)


def _detect_change_points(
    cp_probs: pd.Series,
    threshold: float,
    min_run: int,
) -> pd.Index:
    """Return index locations where change-point probability exceeds threshold.

    Enforces a minimum gap of ``min_run`` between successive detections
    (greedy, left-to-right).

    Parameters
    ----------
    cp_probs : pd.Series
        Probability of change point at each time step.
    threshold : float
    min_run : int

    Returns
    -------
    pd.Index
    """
    candidates = cp_probs.index[cp_probs.values >= threshold]
    if len(candidates) == 0:
        return cp_probs.index[:0]  # empty same-type index

    # Greedy suppression: once we accept a CP, skip next `min_run` positions.
    accepted = []
    last_pos = -min_run - 1
    pos_arr = np.searchsorted(cp_probs.index, candidates)

    for i, loc in enumerate(candidates):
        pos = pos_arr[i]
        if pos - last_pos >= min_run:
            accepted.append(loc)
            last_pos = pos

    return pd.Index(accepted)


# ---------------------------------------------------------------------------
# Alignment utility: map HMM regimes to BOCPD change points
# ---------------------------------------------------------------------------


def align_bocpd_to_regimes(
    change_points: pd.Index,
    states: pd.Series | np.ndarray,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """For each BOCPD change point, report the HMM regime before and after.

    Parameters
    ----------
    change_points : pd.Index
        Change-point locations (must be a subset of ``index``).
    states : pd.Series or np.ndarray
        Hard-decoded HMM state at each time step.
    index : pd.Index, optional
        Index for ``states`` if it is a raw array.

    Returns
    -------
    pd.DataFrame
        Columns: ``[time, regime_before, regime_after, regime_changed]``.
    """
    if isinstance(states, np.ndarray):
        if index is None:
            index = pd.RangeIndex(len(states))
        states = pd.Series(states, index=index)

    _COLS = ["time", "regime_before", "regime_after", "regime_changed"]
    rows = []
    pos_map = {loc: i for i, loc in enumerate(states.index)}
    for cp in change_points:
        pos = pos_map.get(cp)
        if pos is None:
            continue
        before = int(states.iloc[pos - 1]) if pos > 0 else -1
        after = int(states.iloc[pos]) if pos < len(states) else -1
        rows.append(
            {
                "time": cp,
                "regime_before": before,
                "regime_after": after,
                "regime_changed": before != after,
            }
        )
    if not rows:
        return pd.DataFrame(columns=_COLS)
    return pd.DataFrame(rows)
