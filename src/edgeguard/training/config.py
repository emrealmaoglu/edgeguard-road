"""Safe loading and suite validation for semantic training configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from edgeguard.config import UniqueKeySafeLoader
from edgeguard.training.contracts import (
    ModelFamily,
    SemanticCommonConfig,
    SemanticFrameworkConfig,
    SemanticModelConfig,
)

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def _load(path: Path, model_type: type[ConfigT]) -> ConfigT:
    with path.open("r", encoding="utf-8") as stream:
        payload: Any = yaml.load(stream, Loader=UniqueKeySafeLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"training configuration must be a YAML mapping: {path}")
    return model_type.model_validate(payload)


def load_semantic_framework_config(path: Path) -> SemanticFrameworkConfig:
    """Load the exact framework proposal."""
    return _load(path, SemanticFrameworkConfig)


def load_semantic_common_config(path: Path) -> SemanticCommonConfig:
    """Load the common path-free training policy."""
    return _load(path, SemanticCommonConfig)


def load_semantic_model_config(path: Path) -> SemanticModelConfig:
    """Load one independent model specification."""
    return _load(path, SemanticModelConfig)


def load_semantic_model_suite(directory: Path) -> tuple[SemanticModelConfig, ...]:
    """Load exactly one config for each approved semantic family."""
    configs = tuple(
        load_semantic_model_config(path)
        for path in sorted(directory.glob("*.yaml"))
        if path.name not in {"common_cityscapes.yaml", "framework_mmseg.yaml"}
    )
    families = [config.model_family for config in configs]
    expected = set(ModelFamily)
    if set(families) != expected or len(families) != len(expected):
        raise ValueError("semantic model suite must contain each approved family exactly once")
    experiment_ids = [config.experiment_id for config in configs]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("semantic model suite contains duplicate experiment IDs")
    return configs
