"""Heuristic state labelling by ranking regimes on return and volatility."""

import logging
from dataclasses import dataclass

import numpy as np

from rde.models.hmm import FittedModel

logger = logging.getLogger(__name__)

HEURISTIC_DISCLAIMER = (
    "NOTICE: Labels are heuristic interpretations of the model's state means "
    "and do not represent ground truth market conditions."
)


@dataclass
class LabelledState:
    """A model state paired with a heuristic human-readable label.

    Attributes
    ----------
    index : int
        State index as used by hmmlearn (0-based).
    label : str
        Heuristic label, e.g. "bearish", "bullish", "low_return_high_vol".
    rank_return : int
        Return rank among all states (0 = lowest mean return).
    rank_volatility : int
        Volatility rank among all states (0 = lowest mean volatility).

    """

    index: int
    label: str
    rank_return: int
    rank_volatility: int


def rank_states(
    model: FittedModel,
    return_feature: str = "log_return",
    vol_feature: str | None = "volatility_w24",
) -> list[LabelledState]:
    """Assign heuristic labels to HMM states based on return and volatility means.

    Parameters
    ----------
    model : FittedModel
        A fitted model whose feature_names contains the requested features.
    return_feature : str
        Column name for the return feature used to rank states.
    vol_feature : str | None
        Column name for the volatility feature. If None or absent, vol rank is 0.

    Returns
    -------
    list[LabelledState]
        One entry per state, in state-index order (i.e. index 0 first).

    Notes
    -----
    Labels are heuristic — see HEURISTIC_DISCLAIMER. A 2-state model gets
    "bearish" / "bullish". A k-state model gets composite labels such as
    "low_return_high_vol". Do not treat these as ground truth.

    """
    feature_names = model.feature_names
    means = model.hmm.means_  # (n_states, n_features)

    if return_feature not in feature_names:
        raise ValueError(
            f"return_feature={return_feature!r} not found in model features: {feature_names}"
        )

    ret_idx = feature_names.index(return_feature)
    return_means = means[:, ret_idx]
    return_ranks = np.argsort(np.argsort(return_means))  # 0 = lowest mean

    use_vol = vol_feature is not None and vol_feature in feature_names
    if use_vol:
        vol_idx = feature_names.index(vol_feature)
        vol_means = means[:, vol_idx]
        vol_ranks = np.argsort(np.argsort(vol_means))  # 0 = lowest vol
    else:
        vol_ranks = np.zeros(model.n_states, dtype=int)

    n = model.n_states

    def _rank_label(rank: int, total: int) -> str:
        if total == 2:
            return ["low", "high"][rank]
        if total == 3:
            return ["low", "mid", "high"][rank]
        return str(rank)

    labelled: list[LabelledState] = []
    for state_idx in range(n):
        r_rank = int(return_ranks[state_idx])
        v_rank = int(vol_ranks[state_idx])

        if n == 2:
            label = "bearish" if r_rank == 0 else "bullish"
        else:
            r_label = _rank_label(r_rank, n)
            v_label = _rank_label(v_rank, n)
            label = f"{r_label}_return_{v_label}_vol"

        labelled.append(
            LabelledState(index=state_idx, label=label, rank_return=r_rank, rank_volatility=v_rank)
        )
        logger.debug(
            "State %d → %r  (return_rank=%d, vol_rank=%d)",
            state_idx,
            label,
            r_rank,
            v_rank,
        )

    return labelled
