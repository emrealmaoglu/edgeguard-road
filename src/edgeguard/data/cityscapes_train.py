"""Project-specific Cityscapes Fine train preparation and split analysis."""

from __future__ import annotations

import hashlib
import io
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError

from edgeguard.data.cityscapes import (
    CITYSCAPES_SOURCE_LABEL_NAMES,
    CITYSCAPES_TRAIN_ID_BY_LABEL_ID,
    label_ids_to_train_ids,
)
from edgeguard.data.single_image import load_rgb_image
from edgeguard.serialization import sha256_file, sha256_payload

_IMAGE_PATTERN = re.compile(
    r"^(?P<city>[a-z][a-z0-9]*)_(?P<sequence>[0-9]{6})_"
    r"(?P<frame>[0-9]{6})_leftImg8bit\.png$"
)
_LABEL_PATTERN = re.compile(
    r"^(?P<city>[a-z][a-z0-9]*)_(?P<sequence>[0-9]{6})_"
    r"(?P<frame>[0-9]{6})_gtFine_labelIds\.png$"
)
_TRAIN_ID_PATTERN = re.compile(
    r"^(?P<city>[a-z][a-z0-9]*)_(?P<sequence>[0-9]{6})_"
    r"(?P<frame>[0-9]{6})_gtFine_trainIds\.png$"
)

ROLE_NAMES = ("train_fit", "train_select", "train_calibration")
SPLIT_SPECS = (
    ("CSF-SPLIT-A", 2026072701, (0.85, 0.10, 0.05)),
    ("CSF-SPLIT-B", 2026072702, (0.80, 0.15, 0.05)),
    ("CSF-SPLIT-C", 2026072703, (0.90, 0.05, 0.05)),
)


@dataclass(frozen=True)
class CityscapesTrainSample:
    """One root-free Cityscapes Fine train image/label/mask identity."""

    sample_id: str
    city: str
    sequence: str
    frame: str
    group_id: str
    image_relative_path: str
    label_id_relative_path: str
    train_id_relative_path: str


def _parse_filename(path: Path, pattern: re.Pattern[str], kind: str) -> tuple[str, str, str]:
    match = pattern.fullmatch(path.name)
    if match is None:
        raise ValueError(f"malformed Cityscapes train {kind} filename: {path.name}")
    city = match.group("city")
    if path.parent.name != city:
        raise ValueError(
            f"Cityscapes train {kind} city directory mismatch for {path.name}: {path.parent.name}"
        )
    return city, match.group("sequence"), match.group("frame")


def discover_cityscapes_train(
    root: Path, *, require_train_ids: bool = True
) -> list[CityscapesTrainSample]:
    """Discover deterministic train triples without accepting duplicate identities."""
    image_root = root / "leftImg8bit" / "train"
    label_root = root / "gtFine" / "train" / "labelIds"
    train_id_root = root / "gtFine" / "train" / "trainIds"
    if not image_root.is_dir() or not label_root.is_dir():
        raise ValueError(
            "Cityscapes train root must contain leftImg8bit/train and gtFine/train/labelIds"
        )
    if require_train_ids and not train_id_root.is_dir():
        raise ValueError("prepared Cityscapes train root is missing gtFine/train/trainIds")

    images: dict[str, tuple[Path, str, str, str]] = {}
    for path in sorted(image_root.rglob("*.png")):
        city, sequence, frame = _parse_filename(path, _IMAGE_PATTERN, "image")
        sample_id = f"{city}_{sequence}_{frame}"
        if sample_id in images:
            raise ValueError(f"duplicate Cityscapes train image sample: {sample_id}")
        images[sample_id] = (path, city, sequence, frame)

    labels: dict[str, Path] = {}
    for path in sorted(label_root.rglob("*.png")):
        city, sequence, frame = _parse_filename(path, _LABEL_PATTERN, "label")
        sample_id = f"{city}_{sequence}_{frame}"
        if sample_id in labels:
            raise ValueError(f"duplicate Cityscapes train label sample: {sample_id}")
        labels[sample_id] = path

    train_ids: dict[str, Path] = {}
    if train_id_root.is_dir():
        for path in sorted(train_id_root.rglob("*.png")):
            city, sequence, frame = _parse_filename(path, _TRAIN_ID_PATTERN, "train-ID mask")
            sample_id = f"{city}_{sequence}_{frame}"
            if sample_id in train_ids:
                raise ValueError(f"duplicate Cityscapes train-ID sample: {sample_id}")
            train_ids[sample_id] = path

    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    missing_train_ids = sorted(set(images) - set(train_ids)) if require_train_ids else []
    unexpected_train_ids = sorted(set(train_ids) - set(images))
    if missing_labels or missing_images or missing_train_ids or unexpected_train_ids:
        raise ValueError(
            "Cityscapes train pairing mismatch: "
            f"missing_labels={missing_labels[:10]}, missing_images={missing_images[:10]}, "
            f"missing_train_ids={missing_train_ids[:10]}, "
            f"unexpected_train_ids={unexpected_train_ids[:10]}"
        )

    samples: list[CityscapesTrainSample] = []
    for sample_id in sorted(images):
        image_path, city, sequence, frame = images[sample_id]
        train_id_path = train_id_root / city / f"{sample_id}_gtFine_trainIds.png"
        samples.append(
            CityscapesTrainSample(
                sample_id=sample_id,
                city=city,
                sequence=sequence,
                frame=frame,
                group_id=f"{city}_{sequence}",
                image_relative_path=image_path.relative_to(root).as_posix(),
                label_id_relative_path=labels[sample_id].relative_to(root).as_posix(),
                train_id_relative_path=train_id_path.relative_to(root).as_posix(),
            )
        )
    if not samples:
        raise ValueError("Cityscapes train preparation contains no paired samples")
    return samples


def _load_uint8_mask(path: Path, *, kind: str) -> npt.NDArray[np.uint8]:
    try:
        with Image.open(path) as encoded:
            encoded.load()
            array = np.asarray(encoded)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"could not decode Cityscapes {kind} {path.name!r}") from error
    if array.dtype != np.uint8 or array.ndim != 2 or any(size <= 0 for size in array.shape):
        raise ValueError(f"Cityscapes {kind} must be a positive HW uint8 mask")
    return np.asarray(array, dtype=np.uint8).copy()


def load_source_label_ids(path: Path) -> npt.NDArray[np.uint8]:
    """Load one native label-ID mask and reject unreviewed source identifiers."""
    label_ids = _load_uint8_mask(path, kind="source label-ID mask")
    label_ids_to_train_ids(label_ids)
    return label_ids


def load_train_ids(path: Path) -> npt.NDArray[np.uint8]:
    """Load one generated mask and require only project semantic IDs or ignore."""
    train_ids = _load_uint8_mask(path, kind="generated train-ID mask")
    observed = {int(value) for value in np.unique(train_ids)}
    invalid = sorted(observed - {*range(19), 255})
    if invalid:
        raise ValueError(f"generated Cityscapes mask contains invalid train IDs: {invalid}")
    return train_ids


def encode_train_ids_png(train_ids: npt.NDArray[np.uint8]) -> bytes:
    """Encode a train-ID mask as deterministic metadata-free grayscale PNG bytes."""
    if train_ids.dtype != np.uint8 or train_ids.ndim != 2:
        raise ValueError("train-ID PNG input must be an HW uint8 array")
    observed = {int(value) for value in np.unique(train_ids)}
    invalid = sorted(observed - {*range(19), 255})
    if invalid:
        raise ValueError(f"train-ID PNG input contains invalid IDs: {invalid}")
    stream = io.BytesIO()
    Image.fromarray(train_ids, mode="L").save(stream, format="PNG", compress_level=9)
    return stream.getvalue()


def generate_train_id_masks(root: Path) -> list[CityscapesTrainSample]:
    """Generate deterministic train-ID PNGs from preserved native label-ID masks."""
    samples = discover_cityscapes_train(root, require_train_ids=False)
    for sample in samples:
        label_ids = load_source_label_ids(root / sample.label_id_relative_path)
        train_ids = label_ids_to_train_ids(label_ids)
        encoded = encode_train_ids_png(train_ids)
        if encoded != encode_train_ids_png(train_ids):
            raise ValueError(f"non-deterministic train-ID encoding for {sample.sample_id}")
        output = root / sample.train_id_relative_path
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(encoded)
    return discover_cityscapes_train(root)


def _pixel_shares(counts: Iterable[int]) -> list[float | None]:
    values = [int(value) for value in counts]
    total = sum(values)
    if total == 0:
        return [None for _value in values]
    return [value / total for value in values]


def _empty_aggregate() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "valid_pixel_count": 0,
        "ignored_pixel_count": 0,
        "class_pixel_counts": [0] * 19,
        "class_presence_counts": [0] * 19,
        "sample_ids": [],
    }


def _merge_sample(aggregate: dict[str, Any], sample: dict[str, Any]) -> None:
    aggregate["sample_count"] += 1
    aggregate["valid_pixel_count"] += sample["valid_pixel_count"]
    aggregate["ignored_pixel_count"] += sample["ignored_pixel_count"]
    aggregate["sample_ids"].append(sample["sample_id"])
    for class_id in range(19):
        aggregate["class_pixel_counts"][class_id] += sample["class_pixel_counts"][class_id]
        aggregate["class_presence_counts"][class_id] += sample["class_presence"][class_id]


def _finalize_aggregate(aggregate: dict[str, Any]) -> dict[str, Any]:
    result = dict(aggregate)
    total_pixels = result["valid_pixel_count"] + result["ignored_pixel_count"]
    result["ignored_pixel_ratio"] = (
        result["ignored_pixel_count"] / total_pixels if total_pixels else None
    )
    result["class_pixel_shares"] = _pixel_shares(result["class_pixel_counts"])
    count = result["sample_count"]
    result["class_image_presence_shares"] = [
        value / count if count else None for value in result["class_presence_counts"]
    ]
    result["sample_ids"] = sorted(result["sample_ids"])
    return result


def analyze_prepared_train(root: Path) -> dict[str, Any]:
    """Stream prepared masks into sample, group, city, and class summaries."""
    samples = discover_cityscapes_train(root)
    global_stats = _empty_aggregate()
    city_stats: dict[str, dict[str, Any]] = {}
    group_stats: dict[str, dict[str, Any]] = {}
    source_counts = np.zeros(256, dtype=np.uint64)
    sample_records: list[dict[str, Any]] = []
    train_id_hash_seconds = 0.0
    train_id_total_bytes = 0

    for sample in samples:
        image = load_rgb_image(root / sample.image_relative_path)
        label_ids = load_source_label_ids(root / sample.label_id_relative_path)
        train_ids = load_train_ids(root / sample.train_id_relative_path)
        expected_train_ids = label_ids_to_train_ids(label_ids)
        if image.shape[:2] != label_ids.shape or label_ids.shape != train_ids.shape:
            raise ValueError(
                f"Cityscapes train geometry mismatch for {sample.sample_id}: "
                f"image={image.shape[:2]}, label={label_ids.shape}, train_ids={train_ids.shape}"
            )
        if not np.array_equal(train_ids, expected_train_ids):
            raise ValueError(f"generated train-ID mask mismatch for {sample.sample_id}")

        source_counts += np.bincount(label_ids.ravel(), minlength=256).astype(np.uint64)
        class_counts = np.bincount(train_ids[train_ids != 255].ravel(), minlength=19).astype(
            np.uint64
        )
        ignored = int(np.count_nonzero(train_ids == 255))
        started = time.perf_counter()
        train_id_path = root / sample.train_id_relative_path
        train_id_sha = sha256_file(train_id_path)
        train_id_hash_seconds += time.perf_counter() - started
        train_id_bytes = train_id_path.stat().st_size
        train_id_total_bytes += train_id_bytes
        record = {
            "sample_id": sample.sample_id,
            "city": sample.city,
            "sequence": sample.sequence,
            "frame": sample.frame,
            "group_id": sample.group_id,
            "height": int(label_ids.shape[0]),
            "width": int(label_ids.shape[1]),
            "valid_pixel_count": int(class_counts.sum()),
            "ignored_pixel_count": ignored,
            "class_pixel_counts": [int(value) for value in class_counts],
            "class_presence": [int(value > 0) for value in class_counts],
            "image_relative_path": sample.image_relative_path,
            "label_id_relative_path": sample.label_id_relative_path,
            "train_id_relative_path": sample.train_id_relative_path,
            "train_id_bytes": train_id_bytes,
            "train_id_sha256": train_id_sha,
        }
        record["pairing_sha256"] = sha256_payload(
            {
                "sample_id": sample.sample_id,
                "image_relative_path": sample.image_relative_path,
                "label_id_relative_path": sample.label_id_relative_path,
                "train_id_relative_path": sample.train_id_relative_path,
            }
        )
        sample_records.append(record)
        _merge_sample(global_stats, record)
        city_aggregate = city_stats.setdefault(sample.city, _empty_aggregate())
        _merge_sample(city_aggregate, record)
        group_aggregate = group_stats.setdefault(sample.group_id, _empty_aggregate())
        group_aggregate.setdefault("city", sample.city)
        group_aggregate.setdefault("sequence", sample.sequence)
        _merge_sample(group_aggregate, record)

    finalized_global = _finalize_aggregate(global_stats)
    pixel_shares = finalized_global["class_pixel_shares"]
    presence_shares = finalized_global["class_image_presence_shares"]
    nonzero_by_pixels = [
        class_id
        for class_id, count in sorted(
            enumerate(finalized_global["class_pixel_counts"]), key=lambda item: (item[1], item[0])
        )
        if count > 0
    ]
    rare_definitions = {
        "pixel_share_below_1_percent": [
            class_id
            for class_id, share in enumerate(pixel_shares)
            if share is not None and 0 < share < 0.01
        ],
        "image_presence_below_5_percent": [
            class_id
            for class_id, share in enumerate(presence_shares)
            if share is not None and 0 < share < 0.05
        ],
        "five_lowest_nonzero_pixel_share": nonzero_by_pixels[:5],
    }
    heuristic_rare_ids = sorted(
        set(rare_definitions["pixel_share_below_1_percent"])
        | set(rare_definitions["image_presence_below_5_percent"])
        | set(rare_definitions["five_lowest_nonzero_pixel_share"])
    )
    observed_source_ids = [int(index) for index in np.flatnonzero(source_counts)]
    unknown_source_ids = sorted(set(observed_source_ids) - set(CITYSCAPES_SOURCE_LABEL_NAMES))
    if unknown_source_ids:
        raise ValueError(f"unknown Cityscapes source label IDs: {unknown_source_ids}")
    class_mapping_receipt = [
        {
            "source_label_id": source_id,
            "source_name": CITYSCAPES_SOURCE_LABEL_NAMES[source_id],
            "project_train_id": CITYSCAPES_TRAIN_ID_BY_LABEL_ID.get(source_id),
            "action": "map" if source_id in CITYSCAPES_TRAIN_ID_BY_LABEL_ID else "ignore",
            "observed_pixel_count": int(source_counts[source_id]),
        }
        for source_id in sorted(CITYSCAPES_SOURCE_LABEL_NAMES)
    ]
    file_map = [
        {
            "sample_id": record["sample_id"],
            "train_id_relative_path": record["train_id_relative_path"],
            "train_id_bytes": record["train_id_bytes"],
            "train_id_sha256": record["train_id_sha256"],
        }
        for record in sample_records
    ]
    return {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_train_analysis",
        "samples": sample_records,
        "global": finalized_global,
        "cities": {city: _finalize_aggregate(stats) for city, stats in sorted(city_stats.items())},
        "groups": {
            group_id: _finalize_aggregate(stats) for group_id, stats in sorted(group_stats.items())
        },
        "rare_class_definition_candidates": rare_definitions,
        "heuristic_rare_class_ids": heuristic_rare_ids,
        "class_mapping_receipt": class_mapping_receipt,
        "train_id_integrity": {
            "file_count": len(file_map),
            "total_bytes": train_id_total_bytes,
            "hashing_seconds": train_id_hash_seconds,
            "file_map_sha256": sha256_payload(file_map),
        },
    }


def _stable_order_key(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()


def _distribution_divergence(counts: list[int], reference_shares: list[float | None]) -> float:
    shares = _pixel_shares(counts)
    return 0.5 * sum(
        abs((share or 0.0) - (reference or 0.0))
        for share, reference in zip(shares, reference_shares, strict=True)
    )


def _candidate_objective(
    role_stats: dict[str, dict[str, Any]],
    *,
    targets: dict[str, float],
    global_stats: dict[str, Any],
    rare_ids: list[int],
) -> dict[str, float]:
    total_samples = global_stats["sample_count"]
    sample_deviation = sum(
        abs(role_stats[role]["sample_count"] / total_samples - targets[role]) for role in ROLE_NAMES
    )
    present_global = {
        class_id
        for class_id, count in enumerate(global_stats["class_presence_counts"])
        if count > 0
    }
    coverage_missing = 0
    rare_missing = 0
    divergence = 0.0
    for role in ("train_select", "train_calibration"):
        present = {
            class_id
            for class_id, count in enumerate(role_stats[role]["class_presence_counts"])
            if count > 0
        }
        coverage_missing += len(present_global - present)
        rare_missing += len(set(rare_ids) - present)
        divergence += _distribution_divergence(
            role_stats[role]["class_pixel_counts"], global_stats["class_pixel_shares"]
        )
    coverage_denominator = max(1, 2 * len(present_global))
    rare_denominator = max(1, 2 * len(rare_ids))
    components = {
        "sample_count_deviation": sample_deviation,
        "class_presence_coverage_penalty": coverage_missing / coverage_denominator,
        "pixel_distribution_divergence": divergence / 2,
        "rare_class_absence_penalty": rare_missing / rare_denominator,
    }
    components["total"] = (
        2.0 * components["sample_count_deviation"]
        + components["class_presence_coverage_penalty"]
        + components["pixel_distribution_divergence"]
        + 1.5 * components["rare_class_absence_penalty"]
    )
    return components


def _assign_groups(
    analysis: dict[str, Any], *, seed: int, target_values: tuple[float, float, float]
) -> dict[str, list[str]]:
    groups = analysis["groups"]
    if len(groups) < len(ROLE_NAMES):
        raise ValueError("at least three city+sequence groups are required for split candidates")
    targets = dict(zip(ROLE_NAMES, target_values, strict=True))
    total_samples = analysis["global"]["sample_count"]
    assigned: dict[str, list[str]] = {role: [] for role in ROLE_NAMES}
    role_counts = {role: 0 for role in ROLE_NAMES}
    role_presence: dict[str, set[int]] = {role: set() for role in ROLE_NAMES}
    rare_ids = set(analysis["heuristic_rare_class_ids"])
    ordered_groups = sorted(
        groups,
        key=lambda group_id: (
            -groups[group_id]["sample_count"],
            _stable_order_key(seed, group_id),
        ),
    )
    for index, group_id in enumerate(ordered_groups):
        group = groups[group_id]
        remaining = len(ordered_groups) - index
        empty_roles = [role for role in ROLE_NAMES if not assigned[role]]
        allowed_roles = empty_roles if remaining == len(empty_roles) else list(ROLE_NAMES)
        group_present = {
            class_id for class_id, count in enumerate(group["class_presence_counts"]) if count > 0
        }
        choices: list[tuple[float, str]] = []
        for role in allowed_roles:
            target_count = max(1.0, targets[role] * total_samples)
            load = (role_counts[role] + group["sample_count"]) / target_count
            new_classes = len(group_present - role_presence[role])
            new_rare = len((group_present & rare_ids) - role_presence[role])
            coverage_reward = 0.02 * new_classes + 0.08 * new_rare
            if role == "train_fit":
                coverage_reward *= 0.25
            choices.append((load - coverage_reward, role))
        _score, selected_role = min(choices, key=lambda item: (item[0], ROLE_NAMES.index(item[1])))
        assigned[selected_role].append(group_id)
        role_counts[selected_role] += group["sample_count"]
        role_presence[selected_role].update(group_present)
    if any(not assigned[role] for role in ROLE_NAMES):
        raise ValueError("split candidate produced an empty role")
    return {role: sorted(group_ids) for role, group_ids in assigned.items()}


def build_split_candidates(analysis: dict[str, Any]) -> dict[str, Any]:
    """Build three deterministic group-atomic candidates without approving one."""
    sample_by_id = {sample["sample_id"]: sample for sample in analysis["samples"]}
    candidates: list[dict[str, Any]] = []
    for candidate_id, seed, target_values in SPLIT_SPECS:
        targets = dict(zip(ROLE_NAMES, target_values, strict=True))
        assignment = _assign_groups(analysis, seed=seed, target_values=target_values)
        role_stats: dict[str, dict[str, Any]] = {}
        sample_roles: dict[str, str] = {}
        group_rows: list[dict[str, Any]] = []
        for role in ROLE_NAMES:
            aggregate = _empty_aggregate()
            city_counts: dict[str, int] = {}
            for group_id in assignment[role]:
                group = analysis["groups"][group_id]
                group_rows.append(
                    {
                        "group_id": group_id,
                        "city": group["city"],
                        "sequence": group["sequence"],
                        "role": role,
                        "sample_count": group["sample_count"],
                    }
                )
                city_counts[group["city"]] = city_counts.get(group["city"], 0) + int(
                    group["sample_count"]
                )
                for sample_id in group["sample_ids"]:
                    if sample_id in sample_roles:
                        raise ValueError(f"group leakage detected for sample {sample_id}")
                    sample_roles[sample_id] = role
                    _merge_sample(aggregate, sample_by_id[sample_id])
            finalized = _finalize_aggregate(aggregate)
            finalized["group_count"] = len(assignment[role])
            finalized["city_counts"] = dict(sorted(city_counts.items()))
            role_stats[role] = finalized
        if set(sample_roles) != set(sample_by_id):
            raise ValueError("split candidate does not cover every sample exactly once")
        group_sets = [set(assignment[role]) for role in ROLE_NAMES]
        if any(group_sets[left] & group_sets[right] for left in range(3) for right in range(left)):
            raise ValueError("city+sequence group leakage detected")
        objective = _candidate_objective(
            role_stats,
            targets=targets,
            global_stats=analysis["global"],
            rare_ids=analysis["heuristic_rare_class_ids"],
        )
        sample_rows = [
            {
                "sample_id": sample_id,
                "group_id": sample_by_id[sample_id]["group_id"],
                "role": sample_roles[sample_id],
            }
            for sample_id in sorted(sample_roles)
        ]
        candidate: dict[str, Any] = {
            "schema_version": "1.0",
            "record_type": "cityscapes_fine_split_candidate",
            "candidate_id": candidate_id,
            "seed": seed,
            "target_ratios": targets,
            "status": "candidate_not_human_approved",
            "group_atomicity": "city+sequence",
            "leakage_validated": True,
            "roles": role_stats,
            "sample_manifest": sample_rows,
            "group_manifest": sorted(group_rows, key=lambda row: row["group_id"]),
            "objective": objective,
            "rare_class_ids": analysis["heuristic_rare_class_ids"],
        }
        candidate["candidate_sha256"] = sha256_payload(candidate)
        candidates.append(candidate)
    recommended = min(
        candidates,
        key=lambda item: (item["objective"]["total"], item["candidate_id"]),
    )
    return {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_split_candidate_set",
        "selection_status": "recommended_pending_human_approval",
        "recommended_candidate_id": recommended["candidate_id"],
        "candidates": candidates,
    }


def build_split_comparison(candidate_set: dict[str, Any]) -> dict[str, Any]:
    """Build a compact candidate comparison without full sample manifests."""
    rows: list[dict[str, Any]] = []
    for candidate in candidate_set["candidates"]:
        rare_ids = set(candidate["rare_class_ids"])
        role_class_coverage = {
            role: sum(value > 0 for value in candidate["roles"][role]["class_presence_counts"])
            for role in ROLE_NAMES
        }
        role_rare_coverage = {
            role: [
                class_id
                for class_id, value in enumerate(candidate["roles"][role]["class_presence_counts"])
                if value > 0 and class_id in rare_ids
            ]
            for role in ROLE_NAMES
        }
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_sha256": candidate["candidate_sha256"],
                "target_ratios": candidate["target_ratios"],
                "role_sample_counts": {
                    role: candidate["roles"][role]["sample_count"] for role in ROLE_NAMES
                },
                "role_group_counts": {
                    role: candidate["roles"][role]["group_count"] for role in ROLE_NAMES
                },
                "role_city_counts": {
                    role: candidate["roles"][role]["city_counts"] for role in ROLE_NAMES
                },
                "role_class_coverage_counts": role_class_coverage,
                "role_rare_class_coverage": role_rare_coverage,
                "objective": candidate["objective"],
                "leakage_validated": candidate["leakage_validated"],
                "train_select_suitability": "candidate_requires_human_review",
                "train_calibration_suitability": "candidate_requires_human_review",
                "known_weaknesses": [
                    "Heuristic candidate; ratios and class balance require human review.",
                    "Official Cityscapes val is excluded from all three internal roles.",
                ],
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_split_candidate_comparison",
        "selection_status": candidate_set["selection_status"],
        "recommended_candidate_id": candidate_set["recommended_candidate_id"],
        "candidates": rows,
    }
    payload["comparison_sha256"] = sha256_payload(payload)
    return payload


def render_split_report(comparison: dict[str, Any]) -> str:
    """Render the compact human selection report as stable Markdown."""
    lines = [
        "# Cityscapes Fine Train Split Candidates",
        "",
        f"Status: `{comparison['selection_status']}`",
        f"Recommendation: `{comparison['recommended_candidate_id']}`",
        "",
        "No candidate is frozen or human-approved. Official Cityscapes val is excluded.",
        "",
        "| Candidate | Fit / Select / Calibration samples | Groups | "
        "Class / rare penalty | Pixel divergence | Objective | Leakage |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for candidate in comparison["candidates"]:
        samples = candidate["role_sample_counts"]
        groups = candidate["role_group_counts"]
        lines.append(
            f"| {candidate['candidate_id']} | "
            f"{samples['train_fit']} / {samples['train_select']} / "
            f"{samples['train_calibration']} | "
            f"{groups['train_fit']} / {groups['train_select']} / "
            f"{groups['train_calibration']} | "
            f"{candidate['objective']['class_presence_coverage_penalty']:.6f} / "
            f"{candidate['objective']['rare_class_absence_penalty']:.6f} | "
            f"{candidate['objective']['pixel_distribution_divergence']:.6f} | "
            f"{candidate['objective']['total']:.8f} | "
            f"{'passed' if candidate['leakage_validated'] else 'failed'} |"
        )
    lines.extend(
        [
            "",
            "The recommendation minimizes the recorded bounded heuristic only; the human project ",
            "owner must inspect city, group, class-frequency, rare-class, and calibration ",
            "suitability ",
            "before selecting a split.",
            "",
        ]
    )
    return "\n".join(lines)
