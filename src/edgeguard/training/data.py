"""Narrow EG-DATA-002 manifest handoff for semantic training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from edgeguard.serialization import sha256_payload
from edgeguard.training.contracts import (
    DatasetIdentity,
    PolicySelectedSplit,
    SemanticTrainingSample,
)


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is missing or malformed") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def load_policy_selected_cityscapes_split(
    dataset_manifest_path: Path,
    split_manifest_path: Path,
) -> tuple[DatasetIdentity, tuple[SemanticTrainingSample, ...]]:
    """Join one cryptographically verified policy-selected D/E split."""
    dataset = _load_object(dataset_manifest_path, "dataset manifest")
    split = PolicySelectedSplit.model_validate(_load_object(split_manifest_path, "split manifest"))
    if dataset.get("record_type") != "cityscapes_fine_train_dataset_manifest":
        raise ValueError("training requires the EG-DATA-002 dataset manifest")
    embedded_dataset_sha = dataset.get("manifest_sha256")
    if not isinstance(embedded_dataset_sha, str):
        raise ValueError("dataset manifest is missing its canonical identity")
    dataset_without_hash = {
        key: value for key, value in dataset.items() if key != "manifest_sha256"
    }
    if sha256_payload(dataset_without_hash) != embedded_dataset_sha:
        raise ValueError("dataset manifest canonical identity mismatch")
    if split.dataset_manifest_sha256 != embedded_dataset_sha:
        raise ValueError("selected split references a different dataset manifest")
    if split.ontology_sha256 != dataset.get("ontology_sha256"):
        raise ValueError("selected split ontology identity mismatch")
    if sha256_payload(split.policy_config) != split.policy_config_sha256:
        raise ValueError("selected split policy-config identity mismatch")
    split_payload = split.model_dump(mode="json")
    split_without_hash = {
        key: value for key, value in split_payload.items() if key != "manifest_sha256"
    }
    if sha256_payload(split_without_hash) != split.manifest_sha256:
        raise ValueError("selected split manifest identity mismatch")
    candidate = split.candidate
    candidate_without_hash = {
        key: value for key, value in candidate.items() if key != "candidate_sha256"
    }
    if sha256_payload(candidate_without_hash) != split.candidate_sha256:
        raise ValueError("selected split candidate identity mismatch")
    if candidate.get("candidate_id") != split.candidate_id:
        raise ValueError("selected split candidate ID mismatch")
    if candidate.get("candidate_sha256") != split.candidate_sha256:
        raise ValueError("selected split embedded candidate hash mismatch")
    if candidate.get("hard_constraints_passed") is not True:
        raise ValueError("selected split did not pass every hard constraint")

    dataset_samples = {sample["sample_id"]: sample for sample in dataset.get("samples", [])}
    joined: list[SemanticTrainingSample] = []
    group_roles: dict[str, str] = {}
    seen: set[str] = set()
    for split_sample in candidate.get("sample_manifest", []):
        sample_id = split_sample.get("sample_id")
        if sample_id in seen or sample_id not in dataset_samples:
            raise ValueError("selected split contains duplicate or unknown sample")
        seen.add(sample_id)
        group_id = split_sample.get("group_id")
        role = split_sample.get("role")
        previous_role = group_roles.setdefault(group_id, role)
        if previous_role != role:
            raise ValueError("selected split crosses one city+sequence group between roles")
        source = dataset_samples[sample_id]
        if source.get("group_id") != group_id:
            raise ValueError("selected split group identity differs from the dataset manifest")
        joined.append(
            SemanticTrainingSample(
                sample_id=sample_id,
                group_id=group_id,
                role=role,
                image_relative_path=source["image_relative_path"],
                train_id_relative_path=source["train_id_relative_path"],
            )
        )
    if set(dataset_samples) != seen:
        raise ValueError("selected split must cover the complete Fine train manifest")
    identity = DatasetIdentity(
        kind="real_selected_split",
        dataset_manifest_sha256=embedded_dataset_sha,
        split_manifest_sha256=split.manifest_sha256,
    )
    return identity, tuple(sorted(joined, key=lambda sample: sample.sample_id))


def samples_for_training_role(
    samples: tuple[SemanticTrainingSample, ...],
    role: Literal["train_fit", "train_select"],
) -> tuple[SemanticTrainingSample, ...]:
    """Expose only gradient-update or model-selection roles to the runner."""
    selected = tuple(sample for sample in samples if sample.role == role)
    if not selected:
        raise ValueError(f"policy-selected split has no samples for {role}")
    return selected
