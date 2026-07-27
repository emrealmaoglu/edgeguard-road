"""Tests for the bounded Cityscapes diversity split-policy rebuild."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgeguard.data.cityscapes_split_policy import (
    POLICY_CONFIG,
    build_diversity_split_policy,
)
from edgeguard.serialization import sha256_payload
from scripts.rebuild_cityscapes_splits import rebuild_cityscapes_splits

SHA = "1" * 64


def _evidence(*, class_18_only_in_large_group: bool = False) -> tuple[dict, dict]:
    samples = []
    groups = []
    for city_index in range(12):
        city = f"city{city_index:02d}"
        for sequence_index in range(20):
            sequence = f"{sequence_index:06d}"
            group_id = f"{city}_{sequence}"
            sample_count = 60 if city_index == 0 and sequence_index == 0 else 1
            group_samples = []
            for frame_index in range(sample_count):
                sample_id = f"{group_id}_{frame_index:06d}"
                group_samples.append(sample_id)
                samples.append(
                    {
                        "sample_id": sample_id,
                        "group_id": group_id,
                        "image_relative_path": f"leftImg8bit/train/{city}/{sample_id}.png",
                        "train_id_relative_path": f"gtFine/train/trainIds/{city}/{sample_id}.png",
                    }
                )
            class_18_present = not class_18_only_in_large_group or sample_count > 50
            pixel_counts = [sample_count * (class_id + 1) for class_id in range(19)]
            presence_counts = [sample_count] * 19
            if not class_18_present:
                pixel_counts[18] = 0
                presence_counts[18] = 0
            groups.append(
                {
                    "group_id": group_id,
                    "city": city,
                    "sequence": sequence,
                    "sample_count": sample_count,
                    "valid_pixel_count": sum(pixel_counts),
                    "ignored_pixel_count": 0,
                    "class_pixel_counts": pixel_counts,
                    "class_presence_counts": presence_counts,
                }
            )
    dataset = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_train_dataset_manifest",
        "image_count": len(samples),
        "ontology_sha256": SHA,
        "samples": samples,
    }
    dataset["manifest_sha256"] = sha256_payload(dataset)
    group_summary = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_train_group_summary",
        "group_identity": "city+sequence",
        "group_count": len(groups),
        "groups": groups,
    }
    return dataset, group_summary


def test_diversity_policy_builds_d_and_e_and_selects_lowest_passing_objective() -> None:
    dataset, groups = _evidence()

    first = build_diversity_split_policy(dataset, groups)
    second = build_diversity_split_policy(dataset, groups)

    assert first == second
    assert first["status"] == "policy_selected"
    assert first["policy_config_sha256"] == sha256_payload(POLICY_CONFIG)
    assert {candidate["candidate_id"] for candidate in first["candidates"]} == {
        "CSF-SPLIT-D",
        "CSF-SPLIT-E",
    }
    passing = [
        candidate for candidate in first["candidates"] if candidate["hard_constraints_passed"]
    ]
    expected = min(passing, key=lambda item: (item["objective"]["total"], item["candidate_id"]))
    assert first["selected_candidate_id"] == expected["candidate_id"]
    assert first["selected_manifest"]["status"] == "policy_selected"
    assert first["selected_manifest"]["dataset_manifest_sha256"] == dataset["manifest_sha256"]
    assert first["selected_manifest"]["ontology_sha256"] == SHA

    for candidate in first["candidates"]:
        assert candidate["hard_constraint_failures"] == []
        roles = candidate["roles"]
        assert roles["train_select"]["city_count"] >= 10
        assert roles["train_calibration"]["city_count"] >= 10
        assert roles["train_select"]["group_count"] >= 100
        assert roles["train_calibration"]["group_count"] >= 50
        for role in ("train_select", "train_calibration"):
            assert max(roles[role]["city_counts"].values()) / roles[role]["sample_count"] <= 0.25
            assert all(value > 0 for value in roles[role]["class_presence_counts"])
        large_group = next(row for row in candidate["group_manifest"] if row["sample_count"] == 60)
        assert large_group["role"] == "train_fit"


def test_diversity_policy_fails_when_rare_class_cannot_enter_selection_roles() -> None:
    dataset, groups = _evidence(class_18_only_in_large_group=True)

    with pytest.raises(ValueError, match="cannot satisfy|no diversity"):
        build_diversity_split_policy(dataset, groups)


def test_split_only_command_writes_root_free_identity_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    dataset, groups = _evidence()
    dataset_path = tmp_path / "private-mount" / "dataset_manifest.json"
    group_path = tmp_path / "private-mount" / "group_summary.json"
    dataset_path.parent.mkdir()
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    group_path.write_text(json.dumps(groups), encoding="utf-8")
    output = tmp_path / "results" / "split-policy"

    result = rebuild_cityscapes_splits(dataset_path, group_path, output)

    assert result["status"] == "policy_selected"
    assert (output / "policy_selected_split.json").is_file()
    assert (output / "split_candidates" / "CSF-SPLIT-D.samples.json").is_file()
    assert str(tmp_path) not in json.dumps(result)
    assert str(tmp_path) not in (output / "split_policy_result.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        rebuild_cityscapes_splits(dataset_path, group_path, output)
