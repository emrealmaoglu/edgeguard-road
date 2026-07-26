"""Validated configuration models for the WP-01 foundation."""

from __future__ import annotations

from collections.abc import Hashable
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
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


class PIDNetUpstreamConfig(StrictConfigModel):
    """Pinned official PIDNet source reference for the isolated spike."""

    repository_url: Literal["https://github.com/XuJiacong/PIDNet.git"]
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")


class PIDNetCheckpointConfig(StrictConfigModel):
    """Human-approved checkpoint provenance without claiming a license grant."""

    filename: Literal["PIDNet_S_Cityscapes_val.pt"]
    repository_page: Literal["https://github.com/XuJiacong/PIDNet"]
    official_file_url: Literal[
        "https://drive.google.com/file/d/1JakgBam_GrzyUMp-NbEVVBPEIXLSCssH/view?usp=sharing"
    ]
    official_collection_url: Literal[
        "https://drive.google.com/drive/folders/"
        "0BySIOtxxULinfjlGdGFiT3NQVUdLVDBxWnhhTjB4VXNBRkFOa281WHlkektYY2VBcWVZb1k"
        "?resourcekey=0-w0JIXUekD-FCW-Rm1Z-HfQ&usp=sharing"
    ]
    source_reference_access_date: Literal["2026-07-26"]
    license_status: Literal["OPEN QUESTION"]
    permitted_use: Literal["non-commercial academic thesis research"]


class PIDNetSampleFileConfig(StrictConfigModel):
    """Expected identity of one sample stored in the fixed upstream checkout."""

    relative_path: str = Field(pattern=r"^samples/[^/]+\.png$")
    filename: str = Field(pattern=r"^[^/]+\.png$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_shape: tuple[Literal[1024], Literal[2048], Literal[3]]


class PIDNetSampleConfig(StrictConfigModel):
    """Primary and fallback upstream sample identities and usage boundary."""

    primary: PIDNetSampleFileConfig
    fallback: PIDNetSampleFileConfig
    usage_scope: Literal["noncommercial_internal_plumbing"]
    dataset_role: Literal["plumbing_only"]
    license_status: Literal["OPEN QUESTION"]


class PIDNetSpikeInputConfig(StrictConfigModel):
    """Fixed engineering input grid for the first PIDNet-S spike."""

    batch_size: Literal[1]
    height: Literal[512]
    width: Literal[1024]
    channels: Literal[3]
    color_space: Literal["RGB"]


class PIDNetPreprocessConfig(StrictConfigModel):
    """Explicit preprocessing matched to the official custom inference path."""

    resize_mode: Literal["bilinear"]
    pixel_scale: float = Field(gt=0.0, allow_inf_nan=False)
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @field_validator("mean", "std")
    @classmethod
    def require_finite_channels(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Reject non-finite normalization constants."""
        if not all(float("-inf") < channel < float("inf") for channel in value):
            raise ValueError("normalization channels must be finite")
        return value

    @field_validator("std")
    @classmethod
    def require_positive_std(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        """Reject zero or negative normalization divisors."""
        if any(channel <= 0.0 for channel in value):
            raise ValueError("normalization standard deviations must be positive")
        return value


class PIDNetSpikeModelConfig(StrictConfigModel):
    """PIDNet-S prediction-model settings for a direct semantic output."""

    backend: Literal["pidnet_s"]
    num_classes: Literal[19]
    augment: Literal[False]


class PIDNetAlignmentConfig(StrictConfigModel):
    """Native-to-analysis-grid transformation recorded by the spike."""

    mode: Literal["bilinear"]
    target: Literal["model_input"]
    align_corners: Literal[True]


class PIDNetSpikeConfig(BaseConfig):
    """Complete standalone configuration for the isolated PIDNet-S spike."""

    pipeline_name: Literal["pidnet-s-single-image-spike"]
    seed: int = Field(ge=0, le=2**32 - 1)
    upstream: PIDNetUpstreamConfig
    checkpoint: PIDNetCheckpointConfig
    sample: PIDNetSampleConfig
    input: PIDNetSpikeInputConfig
    preprocess: PIDNetPreprocessConfig
    model: PIDNetSpikeModelConfig
    alignment: PIDNetAlignmentConfig
    scorers: tuple[Literal["msp"], Literal["predictive_entropy"]]


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


def load_pidnet_spike_config(path: Path) -> PIDNetSpikeConfig:
    """Load the complete standalone PIDNet-S spike config."""
    return _load_config(path, PIDNetSpikeConfig)


def config_sha256(config: BaseModel) -> str:
    """Hash a validated config using canonical JSON."""
    return sha256_payload(config)
