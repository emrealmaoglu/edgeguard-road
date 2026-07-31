"""Deterministic diversity-aware Cityscapes Fine internal split policy."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from edgeguard.serialization import sha256_payload

ROLE_NAMES = ("train_fit", "train_select", "train_calibration")
POLICY_VERSION = "cityscapes-diversity-policy-v1"
POLICY_CONFIG: dict[str, Any] = {
    "policy_version": POLICY_VERSION,
    "candidate_specs": [
        {
            "candidate_id": "CSF-SPLIT-D",
            "seed": 2026072704,
            "target_ratios": {
                "train_fit": 0.80,
                "train_select": 0.15,
                "train_calibration": 0.05,
            },
        },
        {
            "candidate_id": "CSF-SPLIT-E",
            "seed": 2026072705,
            "target_ratios": {
                "train_fit": 0.85,
                "train_select": 0.10,
                "train_calibration": 0.05,
            },
        },
    ],
    "hard_constraints": {
        "group_identity": "city+sequence",
        "large_group_train_fit_threshold": 50,
        "minimum_select_cities": 10,
        "minimum_calibration_cities": 10,
        "maximum_city_sample_share": 0.25,
        "minimum_select_groups": 100,
        "minimum_calibration_groups": 50,
        "required_class_ids": list(range(19)),
        "rare_class_absence": "hard_failure",
    },
    "objective_weights": {
        "sample_count_deviation": 2.0,
        "pixel_distribution_divergence": 1.0,
        "image_presence_divergence": 1.0,
        "rare_class_imbalance": 1.5,
        "city_imbalance": 1.0,
    },
}


def _stable_key(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()


def _shares(values: Iterable[int]) -> list[float]:
    counts = [int(value) for value in values]
    total = sum(counts)
    return [value / total if total else 0.0 for value in counts]


def _total_variation(left: Iterable[float], right: Iterable[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right, strict=True))


def _validate_hash(payload: dict[str, Any], field: str) -> str:
    recorded = payload.get(field)
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError(f"recorded {field} is missing or invalid")
    unhashed = {key: value for key, value in payload.items() if key != field}
    if sha256_payload(unhashed) != recorded:
        raise ValueError(f"recorded {field} does not match canonical payload")
    return recorded


def _load_groups(
    dataset_manifest: dict[str, Any], group_summary: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if dataset_manifest.get("record_type") != "cityscapes_fine_train_dataset_manifest":
        raise ValueError("split rebuild requires the EG-DATA-002 dataset manifest")
    if group_summary.get("record_type") != "cityscapes_fine_train_group_summary":
        raise ValueError("split rebuild requires the EG-DATA-002 group summary")
    _validate_hash(dataset_manifest, "manifest_sha256")

    samples: dict[str, dict[str, Any]] = {}
    sample_ids_by_group: dict[str, list[str]] = {}
    for raw_sample in dataset_manifest.get("samples", []):
        if not isinstance(raw_sample, dict):
            raise ValueError("dataset manifest contains a malformed sample")
        sample_id = raw_sample.get("sample_id")
        group_id = raw_sample.get("group_id")
        if not isinstance(sample_id, str) or not isinstance(group_id, str):
            raise ValueError("dataset sample identity is missing")
        if sample_id in samples:
            raise ValueError("dataset manifest contains a duplicate sample")
        samples[sample_id] = raw_sample
        sample_ids_by_group.setdefault(group_id, []).append(sample_id)

    groups: dict[str, dict[str, Any]] = {}
    for raw_group in group_summary.get("groups", []):
        if not isinstance(raw_group, dict):
            raise ValueError("group summary contains a malformed group")
        group_id = raw_group.get("group_id")
        city = raw_group.get("city")
        sequence = raw_group.get("sequence")
        if (
            not isinstance(group_id, str)
            or not isinstance(city, str)
            or not isinstance(sequence, str)
        ):
            raise ValueError("group identity is missing")
        if group_id in groups:
            raise ValueError("group summary contains a duplicate group")
        sample_ids = sorted(sample_ids_by_group.get(group_id, []))
        if len(sample_ids) != raw_group.get("sample_count"):
            raise ValueError(f"group sample count mismatch for {group_id}")
        pixel_counts = raw_group.get("class_pixel_counts")
        presence_counts = raw_group.get("class_presence_counts")
        if (
            not isinstance(pixel_counts, list)
            or not isinstance(presence_counts, list)
            or len(pixel_counts) != 19
            or len(presence_counts) != 19
            or any(not isinstance(value, int) or value < 0 for value in pixel_counts)
            or any(not isinstance(value, int) or value < 0 for value in presence_counts)
        ):
            raise ValueError(f"group class statistics are invalid for {group_id}")
        groups[group_id] = {
            "group_id": group_id,
            "city": city,
            "sequence": sequence,
            "sample_count": len(sample_ids),
            "valid_pixel_count": int(raw_group.get("valid_pixel_count", 0)),
            "ignored_pixel_count": int(raw_group.get("ignored_pixel_count", 0)),
            "class_pixel_counts": [int(value) for value in pixel_counts],
            "class_presence_counts": [int(value) for value in presence_counts],
            "sample_ids": sample_ids,
        }
    if set(groups) != set(sample_ids_by_group):
        raise ValueError("dataset manifest and group summary group identities differ")
    if len(samples) != dataset_manifest.get("image_count"):
        raise ValueError("dataset manifest sample count is inconsistent")
    if len(groups) != group_summary.get("group_count"):
        raise ValueError("group summary group count is inconsistent")
    return samples, groups


def _aggregate_groups(
    group_ids: Iterable[str], groups: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    sample_count = 0
    pixel_counts = [0] * 19
    presence_counts = [0] * 19
    city_counts: dict[str, int] = {}
    selected = sorted(group_ids)
    for group_id in selected:
        group = groups[group_id]
        count = int(group["sample_count"])
        sample_count += count
        city = str(group["city"])
        city_counts[city] = city_counts.get(city, 0) + count
        for class_id in range(19):
            pixel_counts[class_id] += int(group["class_pixel_counts"][class_id])
            presence_counts[class_id] += int(group["class_presence_counts"][class_id])
    return {
        "sample_count": sample_count,
        "group_count": len(selected),
        "city_count": len(city_counts),
        "city_counts": dict(sorted(city_counts.items())),
        "class_pixel_counts": pixel_counts,
        "class_pixel_shares": _shares(pixel_counts),
        "class_presence_counts": presence_counts,
        "class_image_presence_shares": [
            value / sample_count if sample_count else 0.0 for value in presence_counts
        ],
    }


def _add_group_to_aggregate(aggregate: dict[str, Any], group: dict[str, Any]) -> dict[str, Any]:
    """Project one aggregate update without mutating the current selection."""
    sample_count = int(aggregate["sample_count"]) + int(group["sample_count"])
    pixel_counts = [
        int(current) + int(incoming)
        for current, incoming in zip(
            aggregate["class_pixel_counts"], group["class_pixel_counts"], strict=True
        )
    ]
    presence_counts = [
        int(current) + int(incoming)
        for current, incoming in zip(
            aggregate["class_presence_counts"], group["class_presence_counts"], strict=True
        )
    ]
    city_counts = dict(aggregate["city_counts"])
    city = str(group["city"])
    city_counts[city] = city_counts.get(city, 0) + int(group["sample_count"])
    return {
        "sample_count": sample_count,
        "group_count": int(aggregate["group_count"]) + 1,
        "city_count": len(city_counts),
        "city_counts": city_counts,
        "class_pixel_counts": pixel_counts,
        "class_pixel_shares": _shares(pixel_counts),
        "class_presence_counts": presence_counts,
        "class_image_presence_shares": [
            value / sample_count if sample_count else 0.0 for value in presence_counts
        ],
    }


def _rare_class_ids(global_stats: dict[str, Any]) -> list[int]:
    pixel_shares = global_stats["class_pixel_shares"]
    presence_shares = global_stats["class_image_presence_shares"]
    nonzero_by_pixels = [
        class_id
        for class_id, count in sorted(
            enumerate(global_stats["class_pixel_counts"]), key=lambda item: (item[1], item[0])
        )
        if count > 0
    ]
    return sorted(
        {class_id for class_id, share in enumerate(pixel_shares) if 0.0 < share < 0.01}
        | {class_id for class_id, share in enumerate(presence_shares) if 0.0 < share < 0.05}
        | set(nonzero_by_pixels[:5])
    )


def _role_selection_key(
    group_id: str,
    current: dict[str, Any],
    groups: dict[str, dict[str, Any]],
    *,
    seed: int,
    target_samples: int,
    minimum_groups: int,
    global_stats: dict[str, Any],
    rare_ids: set[int],
) -> tuple[Any, ...]:
    projected = _add_group_to_aggregate(current, groups[group_id])
    present = {
        class_id for class_id, count in enumerate(current["class_presence_counts"]) if count > 0
    }
    group_present = {
        class_id
        for class_id, count in enumerate(groups[group_id]["class_presence_counts"])
        if count > 0
    }
    missing = set(range(19)) - present
    missing_rare = rare_ids - present
    new_city = groups[group_id]["city"] not in current["city_counts"]
    maximum_city_share = max(projected["city_counts"].values()) / projected["sample_count"]
    pixel_divergence = _total_variation(
        projected["class_pixel_shares"], global_stats["class_pixel_shares"]
    )
    presence_divergence = _total_variation(
        projected["class_image_presence_shares"],
        global_stats["class_image_presence_shares"],
    )
    sample_deviation = abs(projected["sample_count"] - target_samples) / max(1, target_samples)

    if current["city_count"] < 10:
        phase: tuple[Any, ...] = (
            0 if new_city else 1,
            int(groups[group_id]["sample_count"]),
        )
    elif missing:
        phase = (
            0,
            -len(group_present & missing),
            -len(group_present & missing_rare),
        )
    elif current["group_count"] < minimum_groups:
        phase = (0, maximum_city_share, int(groups[group_id]["sample_count"]))
    elif current["sample_count"] < target_samples:
        phase = (0, sample_deviation, maximum_city_share)
    else:
        phase = (0, maximum_city_share, sample_deviation)
    return (
        *phase,
        pixel_divergence + presence_divergence,
        _stable_key(seed, group_id),
    )


def _role_is_complete(
    aggregate: dict[str, Any], *, target_samples: int, minimum_groups: int, rare_ids: set[int]
) -> bool:
    present = {
        class_id for class_id, count in enumerate(aggregate["class_presence_counts"]) if count > 0
    }
    maximum_city_share = (
        max(aggregate["city_counts"].values()) / aggregate["sample_count"]
        if aggregate["sample_count"]
        else 1.0
    )
    return (
        aggregate["sample_count"] >= target_samples
        and aggregate["group_count"] >= minimum_groups
        and aggregate["city_count"] >= 10
        and maximum_city_share <= 0.25
        and present == set(range(19))
        and rare_ids <= present
    )


def _choose_role_groups(
    available: set[str],
    groups: dict[str, dict[str, Any]],
    *,
    seed: int,
    target_samples: int,
    minimum_groups: int,
    global_stats: dict[str, Any],
    rare_ids: set[int],
) -> set[str]:
    chosen: set[str] = set()
    current = _aggregate_groups(chosen, groups)
    while not _role_is_complete(
        current,
        target_samples=target_samples,
        minimum_groups=minimum_groups,
        rare_ids=rare_ids,
    ):
        if not available:
            break
        group_id = min(
            available,
            key=lambda value: _role_selection_key(
                value,
                current,
                groups,
                seed=seed,
                target_samples=target_samples,
                minimum_groups=minimum_groups,
                global_stats=global_stats,
                rare_ids=rare_ids,
            ),
        )
        chosen.add(group_id)
        available.remove(group_id)
        current = _add_group_to_aggregate(current, groups[group_id])
    return chosen


def _assignment(
    groups: dict[str, dict[str, Any]],
    *,
    seed: int,
    targets: dict[str, float],
    global_stats: dict[str, Any],
    rare_ids: set[int],
) -> dict[str, set[str]]:
    large = {group_id for group_id, group in groups.items() if int(group["sample_count"]) > 50}
    available = set(groups) - large
    total_samples = int(global_stats["sample_count"])
    calibration = _choose_role_groups(
        available,
        groups,
        seed=seed + 1,
        target_samples=max(1, round(total_samples * targets["train_calibration"])),
        minimum_groups=50,
        global_stats=global_stats,
        rare_ids=rare_ids,
    )
    selection = _choose_role_groups(
        available,
        groups,
        seed=seed + 2,
        target_samples=max(1, round(total_samples * targets["train_select"])),
        minimum_groups=100,
        global_stats=global_stats,
        rare_ids=rare_ids,
    )
    return {
        "train_fit": set(groups) - calibration - selection,
        "train_select": selection,
        "train_calibration": calibration,
    }


def _hard_constraint_failures(
    assignment: dict[str, set[str]],
    role_stats: dict[str, dict[str, Any]],
    groups: dict[str, dict[str, Any]],
    rare_ids: set[int],
) -> list[str]:
    failures: list[str] = []
    role_sets = [assignment[role] for role in ROLE_NAMES]
    if set().union(*role_sets) != set(groups):
        failures.append("incomplete_group_coverage")
    if any(role_sets[left] & role_sets[right] for left in range(3) for right in range(left)):
        failures.append("group_leakage")
    if any(
        int(group["sample_count"]) > 50 and group_id not in assignment["train_fit"]
        for group_id, group in groups.items()
    ):
        failures.append("large_group_outside_train_fit")
    for role, minimum_groups in (("train_select", 100), ("train_calibration", 50)):
        stats = role_stats[role]
        present = {
            class_id for class_id, count in enumerate(stats["class_presence_counts"]) if count > 0
        }
        if stats["city_count"] < 10:
            failures.append(f"{role}_city_count")
        if stats["group_count"] < minimum_groups:
            failures.append(f"{role}_group_count")
        if max(stats["city_counts"].values(), default=0) / max(1, stats["sample_count"]) > 0.25:
            failures.append(f"{role}_city_share")
        if present != set(range(19)):
            failures.append(f"{role}_class_absence")
        if not rare_ids <= present:
            failures.append(f"{role}_rare_class_absence")
    fit_present = {
        class_id
        for class_id, count in enumerate(role_stats["train_fit"]["class_presence_counts"])
        if count > 0
    }
    if fit_present != set(range(19)):
        failures.append("train_fit_class_absence")
    if not rare_ids <= fit_present:
        failures.append("train_fit_rare_class_absence")
    return failures


def _objective(
    role_stats: dict[str, dict[str, Any]],
    *,
    targets: dict[str, float],
    global_stats: dict[str, Any],
    rare_ids: list[int],
) -> dict[str, float]:
    total_samples = int(global_stats["sample_count"])
    sample_deviation = sum(
        abs(role_stats[role]["sample_count"] / total_samples - targets[role]) for role in ROLE_NAMES
    )
    pixel_divergence = sum(
        _total_variation(role_stats[role]["class_pixel_shares"], global_stats["class_pixel_shares"])
        for role in ROLE_NAMES
    ) / len(ROLE_NAMES)
    presence_divergence = sum(
        _total_variation(
            role_stats[role]["class_image_presence_shares"],
            global_stats["class_image_presence_shares"],
        )
        for role in ROLE_NAMES
    ) / len(ROLE_NAMES)
    rare_imbalance = sum(
        abs(
            role_stats[role]["class_image_presence_shares"][class_id]
            - global_stats["class_image_presence_shares"][class_id]
        )
        for role in ROLE_NAMES
        for class_id in rare_ids
    ) / max(1, len(ROLE_NAMES) * len(rare_ids))

    global_city_shares = {
        city: count / total_samples for city, count in global_stats["city_counts"].items()
    }
    city_imbalance = 0.0
    for role in ("train_select", "train_calibration"):
        role_total = max(1, role_stats[role]["sample_count"])
        role_city_shares = {
            city: role_stats[role]["city_counts"].get(city, 0) / role_total
            for city in global_city_shares
        }
        city_imbalance += _total_variation(role_city_shares.values(), global_city_shares.values())
    city_imbalance /= 2
    components = {
        "sample_count_deviation": sample_deviation,
        "pixel_distribution_divergence": pixel_divergence,
        "image_presence_divergence": presence_divergence,
        "rare_class_imbalance": rare_imbalance,
        "city_imbalance": city_imbalance,
    }
    weights = POLICY_CONFIG["objective_weights"]
    components["total"] = sum(float(weights[name]) * value for name, value in components.items())
    return components


def build_diversity_split_policy(
    dataset_manifest: dict[str, Any], group_summary: dict[str, Any]
) -> dict[str, Any]:
    """Build D/E candidates and select the lowest-objective valid policy."""
    samples, groups = _load_groups(dataset_manifest, group_summary)
    global_stats = _aggregate_groups(groups, groups)
    if any(count <= 0 for count in global_stats["class_presence_counts"]):
        raise ValueError("all 19 classes must occur in the prepared training dataset")
    rare_ids = _rare_class_ids(global_stats)
    candidates: list[dict[str, Any]] = []
    for spec in POLICY_CONFIG["candidate_specs"]:
        candidate_id = str(spec["candidate_id"])
        targets = {key: float(value) for key, value in spec["target_ratios"].items()}
        assignment = _assignment(
            groups,
            seed=int(spec["seed"]),
            targets=targets,
            global_stats=global_stats,
            rare_ids=set(rare_ids),
        )
        role_stats = {role: _aggregate_groups(assignment[role], groups) for role in ROLE_NAMES}
        failures = _hard_constraint_failures(assignment, role_stats, groups, set(rare_ids))
        group_roles = {group_id: role for role in ROLE_NAMES for group_id in assignment[role]}
        sample_rows = [
            {
                "sample_id": sample_id,
                "group_id": str(sample["group_id"]),
                "role": group_roles[str(sample["group_id"])],
            }
            for sample_id, sample in sorted(samples.items())
        ]
        group_rows = [
            {
                "group_id": group_id,
                "city": str(groups[group_id]["city"]),
                "sequence": str(groups[group_id]["sequence"]),
                "sample_count": int(groups[group_id]["sample_count"]),
                "role": group_roles[group_id],
            }
            for group_id in sorted(groups)
        ]
        candidate: dict[str, Any] = {
            "schema_version": "1.0",
            "record_type": "cityscapes_fine_diversity_split_candidate",
            "candidate_id": candidate_id,
            "status": "policy_candidate",
            "policy_version": POLICY_VERSION,
            "seed": int(spec["seed"]),
            "target_ratios": targets,
            "hard_constraints_passed": not failures,
            "hard_constraint_failures": failures,
            "roles": role_stats,
            "objective": _objective(
                role_stats,
                targets=targets,
                global_stats=global_stats,
                rare_ids=rare_ids,
            ),
            "rare_class_ids": rare_ids,
            "sample_manifest": sample_rows,
            "group_manifest": group_rows,
        }
        candidate["candidate_sha256"] = sha256_payload(candidate)
        candidates.append(candidate)
    passing = [candidate for candidate in candidates if candidate["hard_constraints_passed"]]
    if not passing:
        failure_map = {
            candidate["candidate_id"]: candidate["hard_constraint_failures"]
            for candidate in candidates
        }
        raise ValueError(f"no diversity split candidate passed hard constraints: {failure_map}")
    selected = min(passing, key=lambda item: (item["objective"]["total"], item["candidate_id"]))
    policy_config_sha256 = sha256_payload(POLICY_CONFIG)
    selected_manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_policy_selected_split",
        "status": "policy_selected",
        "policy_version": POLICY_VERSION,
        "policy_config": POLICY_CONFIG,
        "policy_config_sha256": policy_config_sha256,
        "candidate_id": selected["candidate_id"],
        "candidate_sha256": selected["candidate_sha256"],
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "ontology_sha256": dataset_manifest["ontology_sha256"],
        "candidate": selected,
    }
    selected_manifest["manifest_sha256"] = sha256_payload(selected_manifest)
    return {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_diversity_split_policy_result",
        "status": "policy_selected",
        "policy_version": POLICY_VERSION,
        "policy_config_sha256": policy_config_sha256,
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "ontology_sha256": dataset_manifest["ontology_sha256"],
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate_sha256": selected["candidate_sha256"],
        "candidates": candidates,
        "selected_manifest": selected_manifest,
    }
