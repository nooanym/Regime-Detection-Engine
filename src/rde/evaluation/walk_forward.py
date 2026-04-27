"""Walk-forward validation harness with strict no-leakage enforcement."""

import logging

import numpy as np
import pandas as pd

from rde.inference.smoothing import forward_backward_posteriors
from rde.inference.viterbi import viterbi_decode
from rde.models.hmm import train_hmm

logger = logging.getLogger(__name__)


def _parse_window_days(window: str) -> int:
    if window.endswith("d"):
        return int(window[:-1])
    raise ValueError(f"Unsupported window format: {window!r}. Expected '<N>d'.")


def _monthly_boundaries(
    index: pd.DatetimeIndex, min_boundary: pd.Timestamp
) -> list[pd.Timestamp]:
    """Return month-start timestamps in index range that are ≥ min_boundary."""
    tz = index.tz
    boundaries = pd.date_range(index[0], index[-1], freq="MS", tz=tz)
    return [b for b in boundaries if b >= min_boundary]


class WalkForwardHarness:
    """Rolling re-fit harness that enforces no future leakage.

    At each recalibration boundary, fits an HMM on the trailing ``train_window``
    of data strictly before the boundary, then labels the period up to the next
    boundary. Bars in the first training window receive NaN labels because no
    model has been fitted strictly before them.

    Parameters
    ----------
    recalibration : str
        Recalibration frequency. Currently only ``"monthly"`` is supported.
    train_window : str
        Length of the training window, e.g. ``"180d"``. Must be shorter than
        the total data span.

    Notes
    -----
    Walk-forward correctness invariant: a bar's regime label is set by a model
    whose training window ends strictly before that bar. No future data is ever
    visible during fitting.

    """

    def __init__(self, recalibration: str = "monthly", train_window: str = "180d") -> None:
        """Validate and store recalibration frequency and training window."""
        if recalibration != "monthly":
            raise ValueError(
                f"recalibration={recalibration!r} is not supported. Only 'monthly' for now."
            )
        self._recalibration = recalibration
        self._train_window_days = _parse_window_days(train_window)

    def run(
        self,
        df_features: pd.DataFrame,
        n_states: int,
        feature_cols: list[str] | None = None,
        **train_kwargs,
    ) -> pd.DataFrame:
        """Fit and label out-of-sample bars in a rolling walk-forward fashion.

        Parameters
        ----------
        df_features : pd.DataFrame
            Full feature DataFrame (output of FeaturePipeline.transform), with
            a tz-aware DatetimeIndex. Must already have NaN rows dropped.
        n_states : int
            Number of hidden states for each fitted HMM.
        feature_cols : list[str] | None
            Columns of df_features to use as the HMM feature matrix. If None,
            all columns are used — but this will typically include price columns,
            so always pass explicit feature names.
        **train_kwargs
            Passed to ``train_hmm`` (n_restarts, covariance_type, etc.).

        Returns
        -------
        pd.DataFrame
            Indexed identically to df_features. Contains columns:
            ``regime`` (int, NaN for warm-up bars),
            ``regime_proba_0 … regime_proba_{n_states-1}`` (float, NaN for warm-up).

        Raises
        ------
        ValueError
            If the data span is shorter than train_window.

        """
        index = df_features.index
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("df_features must have a DatetimeIndex.")

        if feature_cols is None:
            feature_cols = list(df_features.columns)

        train_delta = pd.Timedelta(days=self._train_window_days)
        first_possible_boundary = index[0] + train_delta

        if first_possible_boundary > index[-1]:
            raise ValueError(
                f"train_window={self._train_window_days}d exceeds total data span "
                f"({(index[-1] - index[0]).days}d). Shorten the training window."
            )

        boundaries = _monthly_boundaries(index, first_possible_boundary)
        if not boundaries:
            logger.warning(
                "No monthly recalibration boundaries found after the warm-up period. "
                "No out-of-sample labels produced."
            )

        result = pd.DataFrame(index=index)
        result["regime"] = np.nan
        for i in range(n_states):
            result[f"regime_proba_{i}"] = np.nan

        n_calibrations = 0
        for b_idx, boundary in enumerate(boundaries):
            next_boundary = (
                boundaries[b_idx + 1]
                if b_idx + 1 < len(boundaries)
                else index[-1] + pd.Timedelta(hours=1)
            )

            train_start = boundary - train_delta
            train_mask = (index >= train_start) & (index < boundary)
            oos_mask   = (index >= boundary)    & (index < next_boundary)

            X_train = df_features.loc[train_mask, feature_cols].values
            X_oos   = df_features.loc[oos_mask,   feature_cols].values

            if len(X_train) < n_states:
                logger.warning(
                    "Boundary %s: only %d training bars (need ≥ %d). Skipping.",
                    boundary.date(), len(X_train), n_states,
                )
                continue
            if len(X_oos) == 0:
                logger.debug("Boundary %s: no OOS bars. Skipping.", boundary.date())
                continue

            logger.info(
                "Calibrating at %s: train=%d bars, oos=%d bars",
                boundary.date(), len(X_train), len(X_oos),
            )

            kw = dict(train_kwargs)
            kw.setdefault("feature_names", feature_cols)
            fitted = train_hmm(X_train, n_states, **kw)

            X_oos_scaled = fitted.scaler.transform(X_oos)
            states_oos   = viterbi_decode(fitted.hmm, X_oos_scaled)
            post_oos     = forward_backward_posteriors(fitted.hmm, X_oos_scaled)

            result.loc[oos_mask, "regime"] = states_oos
            for i in range(n_states):
                result.loc[oos_mask, f"regime_proba_{i}"] = post_oos[:, i]

            n_calibrations += 1

        oos_count = result["regime"].notna().sum()
        logger.info(
            "Walk-forward complete: %d calibrations, %d/%d bars labelled (%.1f%%)",
            n_calibrations, oos_count, len(result), 100.0 * oos_count / len(result),
        )
        return result
