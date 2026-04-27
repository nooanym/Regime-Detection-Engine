"""Configuration: YAML schema dataclasses and loader with validation."""

from rde.config.loader import load_config
from rde.config.schema import (
    AssetConfig,
    Config,
    EvaluationConfig,
    FeatureConfig,
    ModelConfig,
    RunConfig,
    SelectionConfig,
)

__all__ = [
    "load_config",
    "AssetConfig",
    "Config",
    "EvaluationConfig",
    "FeatureConfig",
    "ModelConfig",
    "RunConfig",
    "SelectionConfig",
]
