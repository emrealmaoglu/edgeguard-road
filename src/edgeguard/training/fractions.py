"""Deterministic group-aware train-fit fractions for later learning-curve runs."""

from __future__ import annotations

import hashlib
from typing import Any

from edgeguard.serialization import sha256_payload
from edgeguard.training.contracts import SemanticTrainingSample

SUPPORTED_FRACTIONS = (0.25, 0.50, 1.00)


def build_train_fit_fraction(
    samples: tuple[SemanticTrainingSample, ...],
    fraction: float,
    *,
    seed: int,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    """Select whole train-fit groups until the deterministic sample target is met."""
    if fraction not in SUPPORTED_FRACTIONS:
        raise ValueError("train-fit fraction must be one of 0.25, 0.50, or 1.00")
    groups: dict[str, list[str]] = {}
    for sample in samples:
        if sample.role == "train_fit":
            groups.setdefault(sample.group_id, []).append(sample.sample_id)
    if not groups:
        raise ValueError("policy-selected split has no train_fit groups")
    total_samples = sum(len(sample_ids) for sample_ids in groups.values())
    target = total_samples if fraction == 1.0 else max(1, round(total_samples * fraction))
    ordered = sorted(
        groups,
        key=lambda group_id: (
            hashlib.sha256(f"{seed}:{split_manifest_sha256}:{group_id}".encode()).hexdigest(),
            group_id,
        ),
    )
    selected: list[str] = []
    selected_count = 0
    for group_id in ordered:
        selected.append(group_id)
        selected_count += len(groups[group_id])
        if selected_count >= target:
            break
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "semantic_train_fit_fraction",
        "fraction": fraction,
        "seed": seed,
        "source_split_manifest_sha256": split_manifest_sha256,
        "target_sample_count": target,
        "selected_sample_count": selected_count,
        "selected_group_count": len(selected),
        "selected_group_ids": sorted(selected),
        "selected_sample_ids": sorted(
            sample_id for group_id in selected for sample_id in groups[group_id]
        ),
        "group_atomicity": "city+sequence",
    }
    payload["manifest_sha256"] = sha256_payload(payload)
    return payload
