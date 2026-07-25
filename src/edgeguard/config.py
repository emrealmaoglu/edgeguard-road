"""Validated configuration models for the WP-01 foundation."""

from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field
from yaml.nodes import MappingNode

from edgeguard.serialization import sha256_payload

ConfigT = TypeVar("ConfigT", bound=BaseModel)


class DuplicateConfigKeyError(ValueError):
    """Raised when one YAML mapping defines the same key more than once."""


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every mapping level."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        """Validate direct keys before delegating to SafeLoader construction."""
        seen: set[Hashable] = set()
        for key_node, _value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise ValueError(f"Unhashable YAML mapping key at {key_node.start_mark}")
            if key in seen:
                line = key_node.start_mark.line + 1
                raise DuplicateConfigKeyError(f"Duplicate YAML key {key!r} at line {line}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


class StrictConfigModel(BaseModel):
    """Base for immutable configs that reject unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class BaseConfig(StrictConfigModel):
    """Platform-neutral common schema example."""

    schema_version: Literal["1.0"]
    project_name: Literal["edgeguard-road"]
    contract_version: Literal["1.0"]
    record_schema_version: Literal["1.0"]


class SmokeInputConfig(StrictConfigModel):
    """Synthetic input dimensions for the smoke pipeline."""

    batch_size: Literal[1]
    height: int = Field(ge=4, le=4096)
    width: int = Field(ge=4, le=4096)
    channels: Literal[3]


class DummyModelConfig(StrictConfigModel):
    """Configuration for the non-scientific dummy backend."""

    backend: Literal["dummy"]
    num_classes: int = Field(ge=2, le=256)


class DummyScorerConfig(StrictConfigModel):
    """Configuration for the non-scientific dummy anomaly scorer."""

    name: Literal["dummy-normalized-magnitude"]
    epsilon: float = Field(gt=0.0, allow_inf_nan=False)


class SmokeConfig(BaseConfig):
    """Complete, standalone configuration for the WP-01 smoke pipeline."""

    pipeline_name: Literal["dummy-smoke"]
    seed: int = Field(ge=0, le=2**32 - 1)
    input: SmokeInputConfig
    model: DummyModelConfig
    scorer: DummyScorerConfig


def _load_config(path: Path, model_type: type[ConfigT]) -> ConfigT:
    """Load one YAML document without inheritance or composition."""
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.load(stream, Loader=UniqueKeySafeLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return model_type.model_validate(payload)


def load_base_config(path: Path) -> BaseConfig:
    """Load the common schema example."""
    return _load_config(path, BaseConfig)


def load_smoke_config(path: Path) -> SmokeConfig:
    """Load a complete standalone smoke config."""
    return _load_config(path, SmokeConfig)


def config_sha256(config: BaseModel) -> str:
    """Hash a validated config using canonical JSON."""
    return sha256_payload(config)
