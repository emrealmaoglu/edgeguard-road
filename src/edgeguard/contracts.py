"""Array boundary and JSON record contracts."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ContractError(ValueError):
    """Raised when an array violates a pipeline boundary contract."""


def _require_array(name: str, array: npt.NDArray[Any]) -> None:
    if not isinstance(array, np.ndarray):
        raise ContractError(f"{name} must be a NumPy array")


def _require_positive_dimensions(name: str, array: npt.NDArray[Any]) -> None:
    if any(dimension <= 0 for dimension in array.shape):
        raise ContractError(f"{name} dimensions must be positive, got {array.shape}")


def validate_raw_rgb(array: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    """Validate an HWC uint8 raw RGB image."""
    _require_array("raw RGB image", array)
    _require_positive_dimensions("raw RGB image", array)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ContractError(f"raw RGB image must have HWC shape with C=3, got {array.shape}")
    if array.dtype != np.uint8:
        raise ContractError(f"raw RGB image must use uint8, got {array.dtype}")
    return array


def validate_model_input(array: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """Validate an NCHW float32 model input."""
    _require_array("model input", array)
    _require_positive_dimensions("model input", array)
    if array.ndim != 4 or array.shape[1] != 3:
        raise ContractError(f"model input must have NCHW shape with C=3, got {array.shape}")
    if array.dtype != np.float32:
        raise ContractError(f"model input must use float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ContractError("model input must contain only finite values")
    return array


def validate_semantic_logits(array: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """Validate raw, unnormalized NCHW float32 semantic logits."""
    _require_array("semantic logits", array)
    _require_positive_dimensions("semantic logits", array)
    if array.ndim != 4 or array.shape[1] < 2:
        raise ContractError(f"semantic logits must have NCHW shape with C>=2, got {array.shape}")
    if array.dtype != np.float32:
        raise ContractError(f"semantic logits must use float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ContractError("semantic logits must contain only finite values")
    return array


def validate_anomaly_map(array: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """Validate an NHW float32 map where higher means more anomalous."""
    _require_array("anomaly map", array)
    _require_positive_dimensions("anomaly map", array)
    if array.ndim != 3:
        raise ContractError(f"anomaly map must have NHW shape, got {array.shape}")
    if array.dtype != np.float32:
        raise ContractError(f"anomaly map must use float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ContractError("anomaly map must contain only finite values")
    return array


def validate_pipeline_shapes(
    model_input: npt.NDArray[np.float32],
    semantic_logits: npt.NDArray[np.float32],
    anomaly_map: npt.NDArray[np.float32],
) -> None:
    """Ensure batch and spatial dimensions agree across pipeline stages."""
    input_nhw = (model_input.shape[0], model_input.shape[2], model_input.shape[3])
    logits_nhw = (semantic_logits.shape[0], semantic_logits.shape[2], semantic_logits.shape[3])
    if input_nhw != logits_nhw or input_nhw != anomaly_map.shape:
        raise ContractError(
            "pipeline N/H/W mismatch: "
            f"input={input_nhw}, logits={logits_nhw}, anomaly={anomaly_map.shape}"
        )


class GitState(str, Enum):
    """Supported repository states for provenance."""

    UNBORN = "unborn"
    CLEAN = "clean"
    DIRTY = "dirty"
    UNAVAILABLE = "unavailable"


class StrictRecord(BaseModel):
    """Base for immutable, extra-forbidden JSON records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GitProvenanceRecord(StrictRecord):
    """Shared consistency rules for Git-backed records."""

    git_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    git_state: GitState
    git_dirty: bool | None

    @model_validator(mode="after")
    def validate_git_provenance(self) -> GitProvenanceRecord:
        """Reject Git state, commit, and dirty-flag contradictions."""
        if self.git_state is GitState.CLEAN:
            valid = self.git_commit is not None and self.git_dirty is False
        elif self.git_state is GitState.DIRTY:
            valid = self.git_commit is not None and self.git_dirty is True
        elif self.git_state is GitState.UNBORN:
            valid = self.git_commit is None and self.git_dirty is not None
        else:
            valid = self.git_commit is None and self.git_dirty is None
        if not valid:
            raise ValueError(
                "inconsistent Git provenance for "
                f"state={self.git_state.value}, commit={self.git_commit}, dirty={self.git_dirty}"
            )
        return self


class RunMetadata(GitProvenanceRecord):
    """Volatile execution metadata kept outside scientific payloads."""

    schema_version: Literal["1.0"] = "1.0"
    record_type: Literal["run_metadata"] = "run_metadata"
    run_id: str = Field(min_length=1)
    created_at: datetime
    hostname: str = Field(min_length=1)
    command: list[str] = Field(min_length=1)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    experiment_fingerprint: str = Field(pattern=SHA256_PATTERN)
    execution_mode: Literal["normal", "deterministic_smoke"]

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Require an explicit timezone for run timestamps."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class SmokeScientificPayload(StrictRecord):
    """Deterministic synthetic payload; values are not performance metrics."""

    schema_version: Literal["1.0"] = "1.0"
    record_type: Literal["smoke_scientific_payload"] = "smoke_scientific_payload"
    sample_id: Literal["dummy-0000"] = "dummy-0000"
    seed: int
    backend: Literal["dummy"]
    scorer: Literal["dummy-normalized-magnitude"]
    raw_shape: tuple[int, int, int]
    model_input_shape: tuple[int, int, int, int]
    logits_shape: tuple[int, int, int, int]
    anomaly_shape: tuple[int, int, int]
    raw_dtype: Literal["uint8"]
    model_input_dtype: Literal["float32"]
    logits_dtype: Literal["float32"]
    anomaly_dtype: Literal["float32"]
    anomaly_min: float = Field(allow_inf_nan=False)
    anomaly_max: float = Field(allow_inf_nan=False)
    anomaly_mean: float = Field(allow_inf_nan=False)


class SmokeResult(StrictRecord):
    """One smoke record with separate volatile and scientific sections."""

    schema_version: Literal["1.0"] = "1.0"
    record_type: Literal["smoke_result"] = "smoke_result"
    metadata: RunMetadata
    scientific_payload: SmokeScientificPayload


class ArtifactManifest(GitProvenanceRecord):
    """Hash-addressed artifact provenance contract."""

    schema_version: Literal["1.0"] = "1.0"
    record_type: Literal["artifact_manifest"] = "artifact_manifest"
    artifact_type: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    artifact_name: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    model_artifact_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    experiment_fingerprint: str = Field(pattern=SHA256_PATTERN)
    dataset_manifest_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    source_run_id: str | None = None
    model_source: str | None = None
    input_shape: list[int] = Field(min_length=1)
    precision: str = Field(min_length=1)
    environment: dict[str, Any]
    files: dict[str, str] = Field(default_factory=dict)
    created_at: datetime
    created_by: str = Field(min_length=1)
    notes: str | None = None

    @field_validator("input_shape")
    @classmethod
    def require_positive_shape(cls, value: list[int]) -> list[int]:
        """Reject empty or non-positive artifact input dimensions."""
        if any(dimension <= 0 for dimension in value):
            raise ValueError("input_shape dimensions must be positive")
        return value

    @field_validator("files")
    @classmethod
    def require_safe_file_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        """Require relative POSIX artifact names and lowercase SHA-256 values."""
        for name, digest in value.items():
            path = PurePosixPath(name)
            if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError(f"unsafe artifact file name: {name!r}")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"invalid artifact file SHA-256 for {name!r}")
        return value

    @field_validator("created_at")
    @classmethod
    def require_artifact_timezone(cls, value: datetime) -> datetime:
        """Require an explicit timezone for artifact timestamps."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value
