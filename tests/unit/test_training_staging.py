"""Tests for deterministic Drive-to-Colab Cityscapes training staging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgeguard.data.cityscapes_split_policy import POLICY_CONFIG, POLICY_VERSION
from edgeguard.serialization import sha256_payload
from scripts.stage_cityscapes_training import stage_cityscapes_training

SHA = "2" * 64


def _manifests(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "source"
    samples = []
    roles = ("train_fit", "train_select", "train_calibration")
    for index, _role in enumerate(roles):
        city = ("alpha", "beta", "gamma")[index]
        sample_id = f"{city}_000000_000001"
        image = f"leftImg8bit/train/{city}/{sample_id}_leftImg8bit.png"
        label = f"gtFine/train/trainIds/{city}/{sample_id}_gtFine_trainIds.png"
        for relative, contents in ((image, b"rgb"), (label, bytes([index]))):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        samples.append(
            {
                "sample_id": sample_id,
                "group_id": f"{city}_000000",
                "image_relative_path": image,
                "train_id_relative_path": label,
            }
        )
    dataset = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_train_dataset_manifest",
        "samples": samples,
        "image_count": 3,
        "ontology_sha256": SHA,
    }
    dataset["manifest_sha256"] = sha256_payload(dataset)
    candidate = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_diversity_split_candidate",
        "candidate_id": "CSF-SPLIT-D",
        "status": "policy_candidate",
        "hard_constraints_passed": True,
        "sample_manifest": [
            {
                "sample_id": sample["sample_id"],
                "group_id": sample["group_id"],
                "role": role,
            }
            for sample, role in zip(samples, roles, strict=True)
        ],
    }
    candidate["candidate_sha256"] = sha256_payload(candidate)
    split = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_policy_selected_split",
        "status": "policy_selected",
        "policy_version": POLICY_VERSION,
        "policy_config": POLICY_CONFIG,
        "policy_config_sha256": sha256_payload(POLICY_CONFIG),
        "candidate_id": "CSF-SPLIT-D",
        "candidate_sha256": candidate["candidate_sha256"],
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "ontology_sha256": SHA,
        "candidate": candidate,
    }
    split["manifest_sha256"] = sha256_payload(split)
    dataset_path = tmp_path / "dataset.json"
    split_path = tmp_path / "split.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    split_path.write_text(json.dumps(split), encoding="utf-8")
    return root, dataset_path, split_path


def test_training_bundle_is_reused_and_sha_verified(tmp_path: Path) -> None:
    root, dataset, split = _manifests(tmp_path)
    arguments = {
        "dataset_root": root,
        "dataset_manifest_path": dataset,
        "split_policy_path": split,
        "drive_bundle_directory": tmp_path / "drive-bundles",
        "cache_directory": tmp_path / "cache",
        "allow_bundle_creation": True,
    }
    first = stage_cityscapes_training(
        **arguments,
        staged_dataset_root=tmp_path / "staged-first",
    )
    second = stage_cityscapes_training(
        **arguments,
        staged_dataset_root=tmp_path / "staged-second",
    )

    assert first["status"] == "staged_verified"
    assert second["status"] == "staged_verified"
    assert first["sha256"] == second["sha256"]
    assert first["file_count"] == 6


def test_partial_staged_destination_is_rejected(tmp_path: Path) -> None:
    root, dataset, split = _manifests(tmp_path)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "unexpected").write_text("partial", encoding="utf-8")

    with pytest.raises(ValueError, match="partial or identity-mismatched"):
        stage_cityscapes_training(
            dataset_root=root,
            dataset_manifest_path=dataset,
            split_policy_path=split,
            drive_bundle_directory=tmp_path / "drive-bundles",
            cache_directory=tmp_path / "cache",
            staged_dataset_root=partial,
            allow_bundle_creation=True,
        )


def test_staging_dry_run_is_zero_download_and_requires_explicit_bundle_creation(
    tmp_path: Path,
) -> None:
    root, dataset, split = _manifests(tmp_path)
    arguments = {
        "dataset_root": root,
        "dataset_manifest_path": dataset,
        "split_policy_path": split,
        "drive_bundle_directory": tmp_path / "drive-bundles",
        "cache_directory": tmp_path / "cache",
        "staged_dataset_root": tmp_path / "staged",
    }

    plan = stage_cityscapes_training(**arguments, dry_run=True)

    assert plan["expected_download_bytes"] == 0
    assert plan["expected_drive_write_bytes"] > 0
    assert plan["bundle_reusable"] is False
    with pytest.raises(ValueError, match="explicit --create-bundle"):
        stage_cityscapes_training(**arguments)
