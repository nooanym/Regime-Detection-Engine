"""YAML config loader with validation for all Config fields."""

import logging
from pathlib import Path

import yaml

from rde.config.schema import (
    AssetConfig,
    Config,
    EvaluationConfig,
    FeatureConfig,
    ModelConfig,
    RunConfig,
    SelectionConfig,
)

logger = logging.getLogger(__name__)

_VALID_CRITERIA = frozenset({"aic", "bic"})
_VALID_COV_TYPES = frozenset({"full", "diag", "tied", "spherical"})
_VALID_EMISSION_DISTS = frozenset({"gaussian", "student_t"})
_MAX_HOURLY_DAYS = 730


def _parse_period_days(period: str) -> int:
    if period.endswith("d"):
        return int(period[:-1])
    raise ValueError(f"Unsupported period format: {period!r}. Expected '<N>d'.")


def load_config(path: Path | str) -> Config:
    """Load and validate a YAML config file into a typed Config dataclass.

    Parameters
    ----------
    path : Path | str
        Path to the YAML config file.

    Returns
    -------
    Config
        Validated configuration.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If any field fails validation.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        raw = yaml.safe_load(f)

    # Asset
    asset_raw = raw.get("asset", {})
    asset = AssetConfig(
        symbol=asset_raw["symbol"],
        period=str(asset_raw.get("period", "730d")),
        interval=str(asset_raw.get("interval", "1h")),
    )

    if asset.interval == "1h":
        days = _parse_period_days(asset.period)
        if days > _MAX_HOURLY_DAYS:
            raise ValueError(
                f"period={asset.period!r} exceeds the {_MAX_HOURLY_DAYS}d retention cap "
                "for interval='1h'. Use '730d' or shorter."
            )

    # Features
    features_raw = raw.get("features", [])
    features = [
        FeatureConfig(name=str(f["name"]), params=dict(f.get("params", {})))
        for f in features_raw
    ]

    # Model
    model_raw = raw.get("model", {})
    candidate_states: list[int] = [int(n) for n in model_raw.get("candidate_states", [2, 3, 4, 5, 6])]
    if any(n < 2 for n in candidate_states):
        raise ValueError(f"All candidate_states must be ≥ 2, got {candidate_states}")

    cov_type = str(model_raw.get("covariance_type", "full"))
    if cov_type not in _VALID_COV_TYPES:
        raise ValueError(
            f"covariance_type must be one of {sorted(_VALID_COV_TYPES)}, got {cov_type!r}"
        )

    emission_dist = str(model_raw.get("emission_dist", "gaussian"))
    if emission_dist not in _VALID_EMISSION_DISTS:
        raise ValueError(
            f"emission_dist must be one of {sorted(_VALID_EMISSION_DISTS)}, got {emission_dist!r}"
        )

    model = ModelConfig(
        candidate_states=candidate_states,
        covariance_type=cov_type,
        n_restarts=int(model_raw.get("n_restarts", 10)),
        n_iter=int(model_raw.get("n_iter", 1000)),
        init_strategy=str(model_raw.get("init_strategy", "kmeans")),
        seed_base=int(model_raw.get("seed_base", 42)),
        emission_dist=emission_dist,
    )

    # Selection
    sel_raw = raw.get("selection", {})
    criterion = str(sel_raw.get("criterion", "bic"))
    if criterion not in _VALID_CRITERIA:
        raise ValueError(f"criterion must be 'aic' or 'bic', got {criterion!r}")
    selection = SelectionConfig(criterion=criterion)

    # Run
    run_raw = raw.get("run", {})
    run = RunConfig(output_dir=str(run_raw.get("output_dir", "results/{symbol}/")))

    # Evaluation (all optional; defaults to disabled)
    eval_raw = raw.get("evaluation", {})
    wf_recal = str(eval_raw.get("walk_forward_recalibration", "monthly"))
    if wf_recal != "monthly":
        raise ValueError(
            f"walk_forward_recalibration={wf_recal!r} is not supported. Only 'monthly'."
        )
    evaluation = EvaluationConfig(
        run_stability=bool(eval_raw.get("run_stability", False)),
        stability_n_runs=int(eval_raw.get("stability_n_runs", 3)),
        run_walk_forward=bool(eval_raw.get("run_walk_forward", False)),
        walk_forward_train_window=str(eval_raw.get("walk_forward_train_window", "180d")),
        walk_forward_recalibration=wf_recal,
        walk_forward_n_restarts=int(eval_raw.get("walk_forward_n_restarts", 3)),
    )

    config = Config(
        asset=asset, features=features, model=model,
        selection=selection, run=run, evaluation=evaluation,
    )
    logger.info("Loaded config: symbol=%s  period=%s  interval=%s", asset.symbol, asset.period, asset.interval)
    return config
