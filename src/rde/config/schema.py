"""Typed dataclasses for all YAML configuration sections."""

from dataclasses import dataclass, field


@dataclass
class AssetConfig:
    """Configuration for the target asset."""

    symbol: str
    period: str = "730d"
    interval: str = "1h"


@dataclass
class FeatureConfig:
    """Configuration for a single feature transformer."""

    name: str
    params: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for HMM training."""

    candidate_states: list[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    covariance_type: str = "full"
    n_restarts: int = 10
    n_iter: int = 1000
    init_strategy: str = "kmeans"
    seed_base: int = 42
    emission_dist: str = "gaussian"


@dataclass
class SelectionConfig:
    """Configuration for model selection criterion."""

    criterion: str = "bic"


@dataclass
class RunConfig:
    """Configuration for output paths."""

    output_dir: str = "results/{symbol}/"


@dataclass
class EvaluationConfig:
    """Configuration for optional deep evaluation steps.

    Both stability and walk-forward are off by default because they multiply
    the number of model fits and can add significant wall-clock time.
    Enable them per-run via CLI flags or by setting the booleans to true here.
    """

    run_stability: bool = False
    stability_n_runs: int = 3
    run_walk_forward: bool = False
    walk_forward_train_window: str = "180d"
    walk_forward_recalibration: str = "monthly"
    walk_forward_n_restarts: int = 3


@dataclass
class Config:
    """Top-level configuration object."""

    asset: AssetConfig
    features: list[FeatureConfig]
    model: ModelConfig
    selection: SelectionConfig
    run: RunConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
