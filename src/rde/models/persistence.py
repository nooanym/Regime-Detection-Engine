"""Save and load FittedModel to/from disk via pickle.

Using pickle (protocol 5) rather than joblib because FittedModel contains
hmmlearn and sklearn objects that are standard pickle-serializable, and
pickle avoids an explicit joblib dependency.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from rde.models.hmm import FittedModel

logger = logging.getLogger(__name__)


def save_model(fitted: FittedModel, path: Path | str) -> None:
    """Serialise a FittedModel to disk.

    Parameters
    ----------
    fitted : FittedModel
        The trained model to persist.
    path : Path or str
        Destination file path. Parent directories are created if absent.
        Convention: use the ``.pkl`` extension.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(fitted, fh, protocol=5)
    logger.info("Model saved → %s  (n_states=%d  seed=%d)", path, fitted.n_states, fitted.seed)


def load_model(path: Path | str) -> FittedModel:
    """Load a FittedModel from disk.

    Parameters
    ----------
    path : Path or str
        Path to a ``.pkl`` file created by :func:`save_model`.

    Returns
    -------
    FittedModel

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    TypeError
        If the file does not contain a ``FittedModel``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, FittedModel):
        raise TypeError(
            f"Expected FittedModel at {path}, got {type(obj).__name__}"
        )
    logger.info(
        "Model loaded ← %s  (n_states=%d  seed=%d  log_lik=%.2f)",
        path, obj.n_states, obj.seed, obj.log_likelihood,
    )
    return obj
