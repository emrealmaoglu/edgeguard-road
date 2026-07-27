"""Tests for artifact-gated and generator-gated dataset acquisition."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgeguard.serialization import sha256_file
from scripts.acquire_edgeguard_datasets import (
    evaluate_artifact_dataset_readiness,
    generator_dataset_status,
    load_queue,
    process_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QUEUE = REPO_ROOT / "configs/dataset/acquisition_queue.yaml"


def _entry(dataset_id: str) -> dict[str, object]:
    return next(
        entry for entry in load_queue(QUEUE)["entries"] if entry["dataset_id"] == dataset_id
    )


def _write_artifact_receipt(
    root: Path, dataset_id: str, artifact_id: str, filename: str, contents: bytes
) -> None:
    archive = root / "archives" / dataset_id / filename
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(contents)
    receipt = root / "manifests" / dataset_id / "artifacts" / f"{artifact_id}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "artifact_id": artifact_id,
                "artifact_status": "verified",
                "filename": filename,
                "byte_size": len(contents),
                "sha256": sha256_file(archive),
            }
        ),
        encoding="utf-8",
    )


def test_queue_models_bdd_as_two_artifacts_and_static_as_generator() -> None:
    payload = load_queue(QUEUE)
    identifiers = {entry["dataset_id"] for entry in payload["entries"]}
    bdd = _entry("bdd100k_detection")
    fishyscapes = _entry("fishyscapes_static")

    assert identifiers == {
        "bdd100k_detection",
        "fishyscapes_static",
        "cityscapes_coarse_trainextra",
        "temporal_selected",
        "demo_videos",
    }
    assert bdd["acquisition_kind"] == "required_artifacts"
    assert {artifact["artifact_id"] for artifact in bdd["required_artifacts"]} == {
        "bdd100k_100k_images",
        "bdd100k_detection_2020_labels",
    }
    assert fishyscapes["acquisition_kind"] == "pinned_generator"
    assert "required_artifacts" not in fishyscapes
    assert fishyscapes["generator"]["immutable_revision"] == (
        "03773d621374ed10122866e64943df77ff8fbf50"
    )
    assert fishyscapes["generator"]["progress_resume_required"] is True
    assert fishyscapes["generator"]["deterministic_output_manifest_required"] is True
    assert fishyscapes["generator"]["generated_data_receipt_required"] is True
    assert "SMIYC RoadAnomaly21" in payload["sealed_exclusions"]


def test_one_bdd_artifact_cannot_make_dataset_ready(tmp_path: Path) -> None:
    bdd = _entry("bdd100k_detection")
    _write_artifact_receipt(
        tmp_path,
        "bdd100k_detection",
        "bdd100k_100k_images",
        "images.zip",
        b"images",
    )

    result = evaluate_artifact_dataset_readiness(bdd, tmp_path)

    assert result["dataset_status"] == "blocked_required_artifacts"
    assert result["verified_artifact_ids"] == ["bdd100k_100k_images"]
    assert result["missing_or_invalid_artifact_ids"] == ["bdd100k_detection_2020_labels"]
    assert not (tmp_path / "manifests/bdd100k_detection/dataset_readiness_receipt.json").exists()


def test_bdd_is_ready_only_after_both_receipts_and_hashes_verify(tmp_path: Path) -> None:
    bdd = _entry("bdd100k_detection")
    _write_artifact_receipt(
        tmp_path,
        "bdd100k_detection",
        "bdd100k_100k_images",
        "images.zip",
        b"images",
    )
    _write_artifact_receipt(
        tmp_path,
        "bdd100k_detection",
        "bdd100k_detection_2020_labels",
        "labels.zip",
        b"labels",
    )

    result = evaluate_artifact_dataset_readiness(bdd, tmp_path)

    assert result["dataset_status"] == "ready"
    assert len(result["required_artifacts"]) == 2
    assert (tmp_path / "manifests/bdd100k_detection/dataset_readiness_receipt.json").is_file()


def test_fishyscapes_static_reports_generator_gate_not_archive(
    tmp_path: Path,
) -> None:
    fishyscapes = _entry("fishyscapes_static")

    result = generator_dataset_status(fishyscapes, tmp_path)

    assert result["acquisition_kind"] == "pinned_generator"
    assert result["dataset_status"] == "blocked_generator_inputs"
    assert "archive" not in json.dumps(result).lower()


def test_fishyscapes_static_requires_generated_data_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fishyscapes = _entry("fishyscapes_static")
    generator = fishyscapes["generator"]
    identities = {
        "cityscapes_input_manifest_sha256": "1" * 64,
        "generator_config_sha256": "2" * 64,
        "output_manifest_sha256": "3" * 64,
    }
    environment_fields = {
        "cityscapes_input_manifest_sha256": "cityscapes_input_manifest_sha256_env",
        "generator_config_sha256": "generator_config_sha256_env",
        "output_manifest_sha256": "output_manifest_sha256_env",
    }
    for identity, field in environment_fields.items():
        monkeypatch.setenv(str(generator[field]), identities[identity])

    result = generator_dataset_status(fishyscapes, tmp_path)

    assert result["dataset_status"] == "generator_ready_to_execute"
    assert result["input_identities"] == identities
    assert result["dataset_ready"] is False
    assert not (tmp_path / "manifests/fishyscapes_static/dataset_readiness_receipt.json").exists()

    generated_receipt = tmp_path / "manifests/fishyscapes_static/generated_data_receipt.json"
    generated_receipt.parent.mkdir(parents=True, exist_ok=True)
    generated_receipt.write_text(
        json.dumps(
            {
                "source_repository": generator["source_repository"],
                "immutable_revision": generator["immutable_revision"],
                **identities,
            }
        ),
        encoding="utf-8",
    )

    ready = generator_dataset_status(fishyscapes, tmp_path)

    assert ready["dataset_status"] == "ready"
    assert ready["output_manifest_sha256"] == "3" * 64
    assert (tmp_path / "manifests/fishyscapes_static/dataset_readiness_receipt.json").is_file()


def test_temporal_entry_stays_policy_blocked(tmp_path: Path) -> None:
    temporal = _entry("temporal_selected")

    result = process_dataset(temporal, tmp_path)

    assert result["dataset_status"] == "blocked_policy"
    assert result["human_action"] == temporal["human_action"]
