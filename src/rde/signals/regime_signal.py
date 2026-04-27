"""Regime-conditioned trading signal generation.

Converts per-state analytics (Sharpe ratios or raw returns) into a continuous
signal in [-1, 1] by weighting the state quality scores with the current
filtered or smoothed posterior probabilities:

    signal_t = sum_k  P(state_t=k | data)  *  score_k

State quality scores are computed from the in-sample regime analytics and
normalised so the best state maps to +1 and the worst to -1.

The signal is posterior-weighted rather than a hard regime assignment, so it
automatically reflects uncertainty: if the model is 60/40 between a strong
bull state and a neutral state the signal lands somewhere in between rather
than snapping to either extreme.

An optional exponential moving-average smoothing can be applied to reduce
signal noise from rapid posterior oscillations near state boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rde.evaluation.regime_analytics import RegimeStats

logger = logging.getLogger(__name__)

_VALID_SCORING = frozenset({"sharpe", "return"})


@dataclass
class RegimeSignalConfig:
    """Configuration for signal generation.

    Attributes
    ----------
    scoring : str
        Metric used to score states: "sharpe" (annualised Sharpe ratio,
        default) or "return" (annualised mean return). Sharpe is preferred
        because it adjusts for state volatility.
    ema_span : int | None
        If set, apply exponential moving average with this span (in bars)
        to the raw signal before returning. None means no smoothing.
    clip : bool
        If True (default), clip the final signal to [-1, 1] after any
        smoothing. The raw signal is always in [-1, 1] before smoothing;
        clipping is a no-op unless EMA pushes beyond that range due to
        initialisation effects.
    """

    scoring: str = "sharpe"
    ema_span: int | None = None
    clip: bool = True


def score_states(
    regime_stats: list[RegimeStats],
    scoring: str = "sharpe",
) -> np.ndarray:
    """Compute a normalised quality score in [-1, 1] for each state.

    States are ranked by the chosen metric; the worst state gets -1 and the
    best gets +1, with equal spacing in between.  NaN values (e.g. a state
    with only one bar) are assigned a neutral score of 0.

    Parameters
    ----------
    regime_stats : list[RegimeStats]
        Per-state statistics from :func:`rde.evaluation.regime_analytics.compute_regime_stats`.
        Must be sorted by state index (as returned by that function).
    scoring : str
        "sharpe" or "return".

    Returns
    -------
    np.ndarray
        Shape (K,). Scores in [-1, 1] indexed by state index.

    Raises
    ------
    ValueError
        If ``scoring`` is not "sharpe" or "return".
    """
    if scoring not in _VALID_SCORING:
        raise ValueError(f"scoring must be one of {sorted(_VALID_SCORING)}, got {scoring!r}")

    K = len(regime_stats)
    if K == 0:
        return np.array([], dtype=float)

    raw = np.array([
        rs.sharpe_ann if scoring == "sharpe" else rs.mean_return_ann
        for rs in regime_stats
    ])

    # Replace NaN with the median of valid values so those states score neutral
    valid = raw[np.isfinite(raw)]
    if len(valid) == 0:
        return np.zeros(K)
    fill = float(np.median(valid))
    raw = np.where(np.isfinite(raw), raw, fill)

    # Rank-based normalisation: map ranks to [-1, 1]
    ranks = raw.argsort().argsort().astype(float)   # rank 0 = worst
    if K == 1:
        return np.array([0.0])
    scores = 2.0 * ranks / (K - 1) - 1.0            # linearly spaced in [-1, 1]
    return scores


class RegimeSignalGenerator:
    """Produces a continuous directional signal from posterior probabilities.

    Parameters
    ----------
    regime_stats : list[RegimeStats]
        Per-state analytics (sorted by state index) from which state quality
        scores are derived.
    config : RegimeSignalConfig | None
        Signal generation configuration. Defaults to Sharpe scoring, no EMA.

    Attributes
    ----------
    scores : np.ndarray
        Shape (K,). Normalised state quality scores in [-1, 1].

    Examples
    --------
    >>> gen = RegimeSignalGenerator(regime_stats)
    >>> signal = gen.generate(posteriors)        # (T,) from smoothed posteriors
    >>> causal_signal = gen.generate(filtered)   # (T,) from OnlineDecoder output
    """

    def __init__(
        self,
        regime_stats: list[RegimeStats],
        config: RegimeSignalConfig | None = None,
    ) -> None:
        self.config = config or RegimeSignalConfig()
        self.scores = score_states(regime_stats, scoring=self.config.scoring)
        logger.info(
            "RegimeSignalGenerator: scoring=%s  scores=%s",
            self.config.scoring,
            np.round(self.scores, 3),
        )

    def generate(self, posteriors: np.ndarray) -> np.ndarray:
        """Compute the regime signal from a posterior matrix.

        Parameters
        ----------
        posteriors : np.ndarray
            Shape (T, K). Each row must sum to 1. Can be either filtered
            (causal) or smoothed (non-causal) posteriors.

        Returns
        -------
        np.ndarray
            Shape (T,). Values in [-1, 1] (before any EMA smoothing, the
            signal is a convex combination of the state scores).

        Raises
        ------
        ValueError
            If posteriors.shape[1] != len(self.scores).
        """
        if posteriors.ndim != 2 or posteriors.shape[1] != len(self.scores):
            raise ValueError(
                f"posteriors must be (T, {len(self.scores)}), "
                f"got shape {posteriors.shape}."
            )
        signal = posteriors @ self.scores     # (T,)

        if self.config.ema_span is not None:
            signal = _ema(signal, self.config.ema_span)

        if self.config.clip:
            signal = np.clip(signal, -1.0, 1.0)

        return signal

    def generate_series(
        self,
        posteriors: np.ndarray,
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        """Compute the signal and wrap in a labelled pandas Series.

        Parameters
        ----------
        posteriors : np.ndarray
            Shape (T, K).
        index : pd.DatetimeIndex
            Length T. Typically ``df_features.index``.

        Returns
        -------
        pd.Series
            Float signal with the provided DatetimeIndex. Name = "signal".
        """
        signal = self.generate(posteriors)
        return pd.Series(signal, index=index, name="signal")

    def generate_dataframe(
        self,
        posteriors: np.ndarray,
        index: pd.DatetimeIndex,
        viterbi_states: np.ndarray | None = None,
        labels: list[str] | None = None,
    ) -> pd.DataFrame:
        """Build a full signal DataFrame with signal, hard regime, and posteriors.

        Parameters
        ----------
        posteriors : np.ndarray
            Shape (T, K).
        index : pd.DatetimeIndex
            Length T.
        viterbi_states : np.ndarray | None
            Shape (T,). Optional Viterbi hard-decoded states. If provided,
            adds a ``regime_viterbi`` and ``regime_label`` column.
        labels : list[str] | None
            State labels for the ``regime_label`` column.

        Returns
        -------
        pd.DataFrame
            Columns: ``signal``, optionally ``regime_viterbi`` and
            ``regime_label``, plus ``regime_proba_{k}`` for each state k.
        """
        signal = self.generate(posteriors)
        df = pd.DataFrame({"signal": signal}, index=index)

        if viterbi_states is not None:
            df["regime_viterbi"] = viterbi_states.astype(int)
            if labels:
                df["regime_label"] = [
                    labels[int(s)] if int(s) < len(labels) else f"State {s}"
                    for s in viterbi_states
                ]

        for k in range(posteriors.shape[1]):
            df[f"regime_proba_{k}"] = posteriors[:, k]

        return df


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average with a given span.

    Uses α = 2 / (span + 1). Pandas-equivalent to ``Series.ewm(span=span).mean()``,
    but operates on a raw array without the pandas dependency.

    Parameters
    ----------
    x : np.ndarray
        Shape (T,).
    span : int
        EMA span in bars. Must be ≥ 1.

    Returns
    -------
    np.ndarray
        Shape (T,). EMA-smoothed values.
    """
    if span < 1:
        raise ValueError(f"EMA span must be ≥ 1, got {span}")
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1]
    return out
