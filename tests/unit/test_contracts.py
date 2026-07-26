"""Tests for explicit NumPy boundary contracts."""

import numpy as np
import pytest
from pydantic import ValidationError

from edgeguard.contracts import (
    ArtifactManifest,
    ContractError,
    RunMetadata,
    validate_anomaly_map,
    validate_model_input,
    validate_pipeline_shapes,
    validate_raw_rgb,
    validate_semantic_logits,
)

COMMIT_SHA = "a" * 40


def _provenance_record(
    record_kind: str,
    git_state: str,
    git_commit: str | None,
    git_dirty: bool | None,
) -> RunMetadata | ArtifactManifest:
    provenance = {
        "git_state": git_state,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    if record_kind == "run_metadata":
        return RunMetadata.model_validate(
            {
                **provenance,
                "run_id": "test-run",
                "created_at": "2026-07-25T00:00:00Z",
                "hostname": "test-host",
                "command": ["edgeguard", "smoke"],
                "config_sha256": "1" * 64,
                "experiment_fingerprint": "2" * 64,
                "execution_mode": "normal",
            }
        )
    return ArtifactManifest.model_validate(
        {
            **provenance,
            "artifact_type": "smoke_result",
            "artifact_name": "test-artifact",
            "sha256": "0" * 64,
            "model_artifact_sha256": None,
            "config_sha256": "1" * 64,
            "experiment_fingerprint": "2" * 64,
            "dataset_manifest_sha256": None,
            "source_run_id": "test-run",
            "model_source": "dummy",
            "input_shape": [1, 3, 8, 12],
            "precision": "float32",
            "environment": {"type": "test"},
            "created_at": "2026-07-25T00:00:00Z",
            "created_by": "pytest",
            "notes": None,
        }
    )


def test_array_contracts_accept_expected_layouts() -> None:
    raw = np.zeros((8, 12, 3), dtype=np.uint8)
    model_input = np.zeros((1, 3, 8, 12), dtype=np.float32)
    logits = np.zeros((1, 4, 8, 12), dtype=np.float32)
    anomaly = np.zeros((1, 8, 12), dtype=np.float32)

    assert validate_raw_rgb(raw) is raw
    assert validate_model_input(model_input) is model_input
    assert validate_semantic_logits(logits) is logits
    assert validate_anomaly_map(anomaly) is anomaly
    validate_pipeline_shapes(model_input, logits, anomaly)


@pytest.mark.parametrize(
    ("validator", "array"),
    [
        (validate_raw_rgb, np.zeros((8, 12, 3), dtype=np.float32)),
        (validate_model_input, np.zeros((1, 8, 12, 3), dtype=np.float32)),
        (validate_semantic_logits, np.zeros((1, 4, 8, 12), dtype=np.float64)),
        (validate_anomaly_map, np.zeros((1, 1, 8, 12), dtype=np.float32)),
    ],
)
def test_array_contracts_reject_wrong_layout_or_dtype(validator: object, array: np.ndarray) -> None:
    with pytest.raises(ContractError):
        validator(array)  # type: ignore[operator]


@pytest.mark.parametrize(
    ("validator", "array"),
    [
        (validate_raw_rgb, np.zeros((0, 12, 3), dtype=np.uint8)),
        (validate_model_input, np.zeros((0, 3, 8, 12), dtype=np.float32)),
        (validate_semantic_logits, np.zeros((1, 4, 8, 0), dtype=np.float32)),
        (validate_anomaly_map, np.zeros((1, 0, 12), dtype=np.float32)),
    ],
)
def test_array_contracts_reject_zero_dimensions(validator: object, array: np.ndarray) -> None:
    with pytest.raises(ContractError, match="dimensions must be positive"):
        validator(array)  # type: ignore[operator]


def test_contracts_reject_non_finite_values() -> None:
    logits = np.zeros((1, 4, 8, 12), dtype=np.float32)
    logits[0, 0, 0, 0] = np.nan

    with pytest.raises(ContractError):
        validate_semantic_logits(logits)


def test_pipeline_contract_rejects_spatial_mismatch() -> None:
    model_input = np.zeros((1, 3, 8, 12), dtype=np.float32)
    logits = np.zeros((1, 4, 8, 12), dtype=np.float32)
    anomaly = np.zeros((1, 7, 12), dtype=np.float32)

    with pytest.raises(ContractError):
        validate_pipeline_shapes(model_input, logits, anomaly)


@pytest.mark.parametrize("record_kind", ["run_metadata", "artifact_manifest"])
@pytest.mark.parametrize(
    ("git_state", "git_commit", "git_dirty"),
    [
        ("clean", COMMIT_SHA, False),
        ("dirty", COMMIT_SHA, True),
        ("unborn", None, False),
        ("unborn", None, True),
        ("unavailable", None, None),
    ],
)
def test_git_provenance_accepts_consistent_combinations(
    record_kind: str,
    git_state: str,
    git_commit: str | None,
    git_dirty: bool | None,
) -> None:
    record = _provenance_record(record_kind, git_state, git_commit, git_dirty)

    assert record.git_state.value == git_state
    assert record.git_commit == git_commit
    assert record.git_dirty is git_dirty


@pytest.mark.parametrize("record_kind", ["run_metadata", "artifact_manifest"])
@pytest.mark.parametrize(
    ("git_state", "git_commit", "git_dirty"),
    [
        ("clean", None, False),
        ("clean", COMMIT_SHA, True),
        ("clean", COMMIT_SHA, None),
        ("dirty", None, True),
        ("dirty", COMMIT_SHA, False),
        ("dirty", COMMIT_SHA, None),
        ("unborn", COMMIT_SHA, True),
        ("unborn", None, None),
        ("unavailable", COMMIT_SHA, None),
        ("unavailable", None, False),
    ],
)
def test_git_provenance_rejects_inconsistent_combinations(
    record_kind: str,
    git_state: str,
    git_commit: str | None,
    git_dirty: bool | None,
) -> None:
    with pytest.raises(ValidationError, match="inconsistent Git provenance"):
        _provenance_record(record_kind, git_state, git_commit, git_dirty)


def test_artifact_manifest_rejects_invalid_hash() -> None:
    with pytest.raises(ValidationError):
        ArtifactManifest.model_validate(
            {
                "artifact_type": "smoke_result",
                "artifact_name": "invalid",
                "sha256": "not-a-hash",
                "model_artifact_sha256": None,
                "git_commit": None,
                "git_state": "unborn",
                "git_dirty": True,
                "config_sha256": "1" * 64,
                "experiment_fingerprint": "2" * 64,
                "dataset_manifest_sha256": None,
                "source_run_id": None,
                "model_source": None,
                "input_shape": [1, 3, 8, 12],
                "precision": "float32",
                "environment": {},
                "created_at": "2026-07-25T00:00:00Z",
                "created_by": "test",
                "notes": None,
            }
        )


def test_artifact_manifest_validates_relative_file_hash_inventory() -> None:
    manifest = _provenance_record("artifact_manifest", "clean", COMMIT_SHA, False)
    payload = manifest.model_dump(mode="json")
    payload["files"] = {"visuals/sample.png": "3" * 64}

    validated = ArtifactManifest.model_validate(payload)

    assert validated.files == {"visuals/sample.png": "3" * 64}
    payload["files"] = {"../checkpoint.pt": "3" * 64}
    with pytest.raises(ValidationError, match="unsafe artifact file name"):
        ArtifactManifest.model_validate(payload)
