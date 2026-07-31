"""Cityscapes audit and leakage-safe split utilities for the rescue path."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from edgeguard.serialization import canonical_json, sha256_file, sha256_payload

CITYSCAPES_CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)
EXPECTED_ROLE_COUNTS = {"train_fit": 2380, "train_select": 446, "train_calibration": 149}
NEAR_DUPLICATE_HAMMING_DISTANCE = 2
CROP_SURVIVAL_TRIALS = 2


@dataclass(frozen=True)
class SemanticSample:
    """Root-relative image/mask pair with a sequence-atomic group identity."""

    sample_id: str
    city: str
    group_id: str
    image: str
    mask: str
    class_pixel_counts: tuple[int, ...] | None = None


def _sample_id(path: Path) -> str:
    suffix = "_leftImg8bit"
    if not path.stem.endswith(suffix):
        raise ValueError(f"unexpected Cityscapes image name: {path.name}")
    return path.stem[: -len(suffix)]


def _mask_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted((root / "gtFine").glob("**/*.png")):
        name = path.stem
        for suffix in ("_gtFine_labelTrainIds", "_gtFine_trainIds"):
            if name.endswith(suffix):
                key = name[: -len(suffix)]
                if key in index:
                    raise ValueError(f"multiple train-id masks found for {key}")
                index[key] = path
                break
    return index


def discover_cityscapes(
    root: Path, *, split: str = "train"
) -> tuple[list[SemanticSample], list[str]]:
    """Discover Cityscapes train-id pairs without accepting source label IDs."""
    image_root = root / "leftImg8bit" / split
    if not image_root.is_dir():
        raise FileNotFoundError(f"missing Cityscapes image directory: {image_root}")
    masks = _mask_index(root)
    samples: list[SemanticSample] = []
    missing: list[str] = []
    for image_path in sorted(image_root.glob("*/*_leftImg8bit.png")):
        identifier = _sample_id(image_path)
        mask_path = masks.get(identifier)
        if mask_path is None:
            missing.append(identifier)
            continue
        parts = identifier.split("_")
        if len(parts) < 3:
            raise ValueError(f"cannot derive sequence group from {identifier}")
        samples.append(
            SemanticSample(
                sample_id=identifier,
                city=parts[0],
                group_id="_".join(parts[:2]),
                image=image_path.relative_to(root).as_posix(),
                mask=mask_path.relative_to(root).as_posix(),
            )
        )
    if not samples and not missing:
        raise FileNotFoundError(f"no Cityscapes images found under {image_root}")
    return samples, missing


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median_frequency_weights(pixel_counts: Iterable[int], *, maximum: float = 10.0) -> list[float]:
    """Return capped, mean-one median-frequency weights for 19 classes."""
    counts = np.asarray(tuple(pixel_counts), dtype=np.float64)
    if counts.shape != (19,) or bool((counts <= 0).any()):
        raise ValueError("all 19 train classes need positive pixel counts")
    frequencies = counts / counts.sum()
    raw = np.median(frequencies) / frequencies
    capped = np.minimum(raw, maximum)
    normalized = capped / capped.mean()
    return [float(value) for value in normalized]


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_frequency_figures(report_root: Path, class_rows: list[dict[str, Any]]) -> bool:
    """Write the two decision-critical frequency charts when matplotlib is installed."""
    figures = report_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    try:
        pyplot = __import__("matplotlib.pyplot", fromlist=["subplots", "close"])
    except ModuleNotFoundError:
        (figures / "README.md").write_text(
            "Install the `rescue` extra and rerun the audit to render figures.\n",
            encoding="utf-8",
        )
        return False
    names = [str(row["class_name"]) for row in class_rows]
    for field, filename, title in (
        ("pixel_ratio", "class_pixel_frequency.png", "Cityscapes train pixel frequency"),
        ("image_ratio", "class_image_frequency.png", "Cityscapes train image frequency"),
    ):
        figure, axis = pyplot.subplots(figsize=(12, 5))
        axis.bar(names, [float(row[field]) for row in class_rows])
        axis.set_title(title)
        axis.set_ylabel(field)
        axis.tick_params(axis="x", labelrotation=65)
        figure.tight_layout()
        figure.savefig(figures / filename, dpi=160)
        pyplot.close(figure)
    return True


def _difference_hash(image: Image.Image) -> int:
    """Return a compact perceptual hash used only to flag review candidates."""
    gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    bits = (gray[:, 1:] > gray[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _near_duplicate_candidates(
    hashes: list[tuple[str, str, int]], *, maximum_rows: int = 5_000
) -> tuple[int, list[dict[str, Any]]]:
    """Find all Hamming<=2 dHash pairs through bounded-memory pigeonhole bands."""
    bands = ((0, 22), (22, 43), (43, 64))
    buckets: defaultdict[tuple[int, int], list[tuple[str, str, int]]] = defaultdict(list)
    pairs: dict[tuple[str, str], tuple[int, bool]] = {}
    for sample_id, group_id, value in hashes:
        candidates: dict[str, tuple[str, int]] = {}
        for band_id, (start, end) in enumerate(bands):
            key = (band_id, (value >> start) & ((1 << (end - start)) - 1))
            for other_id, other_group, other_value in buckets[key]:
                candidates[other_id] = (other_group, other_value)
        for other_id, (other_group, other_value) in candidates.items():
            distance = (value ^ other_value).bit_count()
            if distance <= NEAR_DUPLICATE_HAMMING_DISTANCE:
                left, right = sorted((sample_id, other_id))
                pair = (left, right)
                pairs[pair] = (distance, group_id == other_group)
        for band_id, (start, end) in enumerate(bands):
            key = (band_id, (value >> start) & ((1 << (end - start)) - 1))
            buckets[key].append((sample_id, group_id, value))
    rows = [
        {
            "sample_id_a": left,
            "sample_id_b": right,
            "hamming_distance": evidence[0],
            "same_sequence_group": evidence[1],
        }
        for (left, right), evidence in sorted(pairs.items())[:maximum_rows]
    ]
    return len(pairs), rows


def _crop_survival_presence(
    mask: np.ndarray, sample_id: str, crop_size: tuple[int, int] = (512, 1024)
) -> tuple[np.ndarray, np.ndarray]:
    """Measure a deterministic proxy for class survival under the training crop policy."""
    present = np.bincount(mask[mask != 255], minlength=19)[:19] > 0
    before = present.astype(np.int64) * CROP_SURVIVAL_TRIALS
    after = np.zeros(19, dtype=np.int64)
    height, width = mask.shape
    seed = int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    crop_height, crop_width = crop_size
    for _ in range(CROP_SURVIVAL_TRIALS):
        scale = float(rng.uniform(0.5, 2.0))
        scaled_height = max(1, round(height * scale))
        scaled_width = max(1, round(width * scale))
        scaled_y = int(rng.integers(0, max(1, scaled_height - crop_height + 1)))
        scaled_x = int(rng.integers(0, max(1, scaled_width - crop_width + 1)))
        y0 = min(height - 1, int(scaled_y / scale))
        x0 = min(width - 1, int(scaled_x / scale))
        y1 = min(height, max(y0 + 1, int(np.ceil((scaled_y + crop_height) / scale))))
        x1 = min(width, max(x0 + 1, int(np.ceil((scaled_x + crop_width) / scale))))
        crop_counts = np.bincount(mask[y0:y1, x0:x1].reshape(-1), minlength=256)[:19]
        after += crop_counts > 0
    return before, after


def _write_audit_figures(
    report_root: Path,
    *,
    cooccurrence: np.ndarray,
    resolutions: Counter[str],
    city_counts: Counter[str],
    ignore_ratio: float | None,
) -> bool:
    figures = report_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    try:
        pyplot = __import__("matplotlib.pyplot", fromlist=["subplots", "close"])
    except ModuleNotFoundError:
        return False
    figure, axis = pyplot.subplots(figsize=(8, 7))
    image = axis.imshow(cooccurrence, cmap="viridis")
    axis.set_title("Cityscapes class co-occurrence (image count)")
    axis.set_xticks(range(19), CITYSCAPES_CLASSES, rotation=90, fontsize=7)
    axis.set_yticks(range(19), CITYSCAPES_CLASSES, fontsize=7)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(figures / "class_cooccurrence.png", dpi=160)
    pyplot.close(figure)
    for values, filename, title in (
        (resolutions, "image_resolution_histogram.png", "Image resolutions"),
        (city_counts, "city_distribution.png", "City distribution"),
    ):
        figure, axis = pyplot.subplots(figsize=(12, 5))
        labels = list(values)
        axis.bar(labels, [values[label] for label in labels])
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=75)
        figure.tight_layout()
        figure.savefig(figures / filename, dpi=160)
        pyplot.close(figure)
    figure, axis = pyplot.subplots(figsize=(5, 4))
    ratio = float(ignore_ratio or 0.0)
    axis.bar(["valid", "ignore"], [1.0 - ratio, ratio])
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Pixel validity ratio")
    figure.tight_layout()
    figure.savefig(figures / "ignore_pixel_ratio.png", dpi=160)
    pyplot.close(figure)
    return True


def audit_cityscapes(root: Path, output_root: Path) -> dict[str, Any]:
    """Audit real Cityscapes pairs and emit deterministic scientific input evidence."""
    report_root = output_root / "dataset_audit"
    report_root.mkdir(parents=True, exist_ok=False)
    samples, missing_masks = discover_cityscapes(root)
    pixel_counts = np.zeros(19, dtype=np.int64)
    image_counts = np.zeros(19, dtype=np.int64)
    ignore_pixels = 0
    total_pixels = 0
    corrupt: list[dict[str, str]] = []
    invalid_labels: list[dict[str, Any]] = []
    black_images: list[dict[str, Any]] = []
    low_information_images: list[dict[str, Any]] = []
    all_ignore_masks: list[dict[str, str]] = []
    digest_samples: defaultdict[str, list[str]] = defaultdict(list)
    perceptual_hashes: list[tuple[str, str, int]] = []
    valid_samples: list[SemanticSample] = []
    resolutions: Counter[str] = Counter()
    city_counts: Counter[str] = Counter()
    cooccurrence = np.zeros((19, 19), dtype=np.int64)
    crop_present_before = np.zeros(19, dtype=np.int64)
    crop_present_after = np.zeros(19, dtype=np.int64)
    for sample in samples:
        image_path = root / sample.image
        mask_path = root / sample.mask
        try:
            with Image.open(image_path) as image:
                image.load()
                image_size = image.size
                perceptual_hash = _difference_hash(image)
                thumbnail = np.asarray(
                    image.convert("L").resize((64, 32), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image, dtype=np.uint8)
            if mask.ndim != 2 or (image_size[1], image_size[0]) != mask.shape:
                raise ValueError(f"image/mask geometry mismatch: {image_size} vs {mask.shape}")
            unique = np.unique(mask)
            invalid = unique[(unique > 18) & (unique != 255)]
            if invalid.size:
                invalid_labels.append(
                    {"sample_id": sample.sample_id, "values": [int(value) for value in invalid]}
                )
                continue
            counts = np.bincount(mask[mask != 255], minlength=19)[:19]
            ignore_pixels += int(np.count_nonzero(mask == 255))
            total_pixels += int(mask.size)
            resolutions[f"{image_size[0]}x{image_size[1]}"] += 1
            city_counts[sample.city] += 1
            digest_samples[_sha256_file(image_path)].append(sample.sample_id)
            perceptual_hashes.append((sample.sample_id, sample.group_id, perceptual_hash))
            intensity_range = float(thumbnail.max() - thumbnail.min())
            intensity_std = float(thumbnail.std())
            if float(thumbnail.max()) <= 1.0:
                black_images.append(
                    {"sample_id": sample.sample_id, "maximum": float(thumbnail.max())}
                )
            if intensity_std < 2.0 or intensity_range < 5.0:
                low_information_images.append(
                    {
                        "sample_id": sample.sample_id,
                        "intensity_std": intensity_std,
                        "intensity_range": intensity_range,
                    }
                )
            if not bool((counts > 0).any()):
                all_ignore_masks.append({"sample_id": sample.sample_id})
                continue
            pixel_counts += counts
            present = counts > 0
            image_counts += present
            cooccurrence += np.outer(present, present)
            before, after = _crop_survival_presence(mask, sample.sample_id)
            crop_present_before += before
            crop_present_after += after
            valid_samples.append(
                SemanticSample(
                    sample_id=sample.sample_id,
                    city=sample.city,
                    group_id=sample.group_id,
                    image=sample.image,
                    mask=sample.mask,
                    class_pixel_counts=tuple(int(value) for value in counts),
                )
            )
        except (OSError, UnidentifiedImageError, ValueError) as error:
            corrupt.append({"sample_id": sample.sample_id, "error": str(error)})
    duplicates = [
        {"sha256": digest, "sample_ids": "|".join(identifiers), "count": len(identifiers)}
        for digest, identifiers in sorted(digest_samples.items())
        if len(identifiers) > 1
    ]
    near_duplicate_count, near_duplicates = _near_duplicate_candidates(perceptual_hashes)
    class_rows = []
    valid_pixel_total = int(pixel_counts.sum())
    for class_id, name in enumerate(CITYSCAPES_CLASSES):
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "pixel_count": int(pixel_counts[class_id]),
                "pixel_ratio": (
                    float(pixel_counts[class_id] / valid_pixel_total) if valid_pixel_total else 0.0
                ),
                "image_count": int(image_counts[class_id]),
                "image_ratio": (
                    float(image_counts[class_id] / len(valid_samples)) if valid_samples else 0.0
                ),
            }
        )
    _write_csv(
        report_root / "class_pixel_frequency.csv",
        ["class_id", "class_name", "pixel_count", "pixel_ratio", "image_count", "image_ratio"],
        class_rows,
    )
    _write_csv(
        report_root / "class_image_frequency.csv",
        ["class_id", "class_name", "image_count", "image_ratio"],
        (
            {
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "image_count": row["image_count"],
                "image_ratio": row["image_ratio"],
            }
            for row in class_rows
        ),
    )
    _write_csv(report_root / "duplicates.csv", ["sha256", "sample_ids", "count"], duplicates)
    _write_csv(
        report_root / "near_duplicates.csv",
        ["sample_id_a", "sample_id_b", "hamming_distance", "same_sequence_group"],
        near_duplicates,
    )
    _write_csv(report_root / "corrupt_files.csv", ["sample_id", "error"], corrupt)
    _write_csv(report_root / "black_images.csv", ["sample_id", "maximum"], black_images)
    _write_csv(
        report_root / "low_information_images.csv",
        ["sample_id", "intensity_std", "intensity_range"],
        low_information_images,
    )
    _write_csv(report_root / "all_ignore_masks.csv", ["sample_id"], all_ignore_masks)
    _write_csv(
        report_root / "invalid_labels.csv",
        ["sample_id", "values"],
        (
            {"sample_id": row["sample_id"], "values": json.dumps(row["values"])}
            for row in invalid_labels
        ),
    )
    crop_rows = [
        {
            "class_id": class_id,
            "class_name": name,
            "present_before_crop": int(crop_present_before[class_id]),
            "present_after_crop": int(crop_present_after[class_id]),
            "survival_rate": (
                float(crop_present_after[class_id] / crop_present_before[class_id])
                if crop_present_before[class_id]
                else None
            ),
        }
        for class_id, name in enumerate(CITYSCAPES_CLASSES)
    ]
    _write_csv(
        report_root / "crop_survival.csv",
        [
            "class_id",
            "class_name",
            "present_before_crop",
            "present_after_crop",
            "survival_rate",
        ],
        crop_rows,
    )
    _write_csv(
        report_root / "class_cooccurrence.csv",
        ["class_id", "class_name", *CITYSCAPES_CLASSES],
        (
            {
                "class_id": class_id,
                "class_name": name,
                **{
                    other_name: int(cooccurrence[class_id, other_id])
                    for other_id, other_name in enumerate(CITYSCAPES_CLASSES)
                },
            }
            for class_id, name in enumerate(CITYSCAPES_CLASSES)
        ),
    )
    _write_csv(
        report_root / "city_distribution.csv",
        ["city", "image_count"],
        ({"city": city, "image_count": count} for city, count in sorted(city_counts.items())),
    )
    weights: list[float] | None = None
    if bool((pixel_counts > 0).all()):
        weights = median_frequency_weights(pixel_counts)
    weights_payload = {
        "schema_version": "1.0",
        "method": "median_frequency_capped_mean_one",
        "maximum_before_renormalization": 10.0,
        "source_role": "train_fit_pending_split_filter",
        "weights": weights,
        "scientific_evidence": True,
        "note": "Recompute on the frozen train_fit role before the weighted-loss run.",
    }
    (report_root / "class_weights.json").write_text(
        canonical_json(weights_payload) + "\n", encoding="utf-8"
    )
    ignore_ratio = float(ignore_pixels / total_pixels) if total_pixels else None
    figures_generated = _write_frequency_figures(report_root, class_rows)
    figures_generated = (
        _write_audit_figures(
            report_root,
            cooccurrence=cooccurrence,
            resolutions=resolutions,
            city_counts=city_counts,
            ignore_ratio=ignore_ratio,
        )
        and figures_generated
    )
    summary = {
        "schema_version": "1.0",
        "record_type": "cityscapes_dataset_audit",
        "dataset": "cityscapes_fine_train",
        "discovered_pairs": len(samples),
        "valid_pairs": len(valid_samples),
        "missing_masks": sorted(missing_masks),
        "corrupt_count": len(corrupt),
        "invalid_label_count": len(invalid_labels),
        "black_image_count": len(black_images),
        "low_information_image_count": len(low_information_images),
        "all_ignore_mask_count": len(all_ignore_masks),
        "exact_duplicate_groups": len(duplicates),
        "near_duplicate_candidate_pairs": near_duplicate_count,
        "near_duplicate_rows_truncated": near_duplicate_count > len(near_duplicates),
        "resolutions": dict(sorted(resolutions.items())),
        "total_pixels": total_pixels,
        "ignore_pixels": ignore_pixels,
        "ignore_ratio": ignore_ratio,
        "class_pixel_counts": [int(value) for value in pixel_counts],
        "class_image_counts": [int(value) for value in image_counts],
        "figures_generated": figures_generated,
        "crop_survival_trials_per_image": CROP_SURVIVAL_TRIALS,
        "crop_survival_method": "deterministic_random-resize-coordinate crop proxy",
        "near_duplicate_method": (
            f"64-bit difference hash, Hamming <= {NEAR_DUPLICATE_HAMMING_DISTANCE}; "
            "review candidates, not confirmed duplicates"
        ),
        "audit_passed": (
            not missing_masks
            and not corrupt
            and not invalid_labels
            and not duplicates
            and not black_images
            and not all_ignore_masks
        ),
    }
    (report_root / "summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
    lines = [
        "# Cityscapes dataset audit",
        "",
        f"- Valid pairs: {len(valid_samples)}",
        f"- Missing masks: {len(missing_masks)}",
        f"- Corrupt/geometry failures: {len(corrupt)}",
        f"- Invalid-label samples: {len(invalid_labels)}",
        f"- Black images: {len(black_images)}",
        f"- Low-information review candidates: {len(low_information_images)}",
        f"- All-ignore masks: {len(all_ignore_masks)}",
        f"- Exact duplicate groups: {len(duplicates)}",
        f"- Near-duplicate review pairs: {near_duplicate_count}",
        f"- Ignore ratio: {summary['ignore_ratio']}",
        f"- Gate: {'PASS' if summary['audit_passed'] else 'FAIL'}",
        "",
        "A failed gate must be resolved before scientific training.",
    ]
    (report_root / "dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["samples"] = [asdict(sample) for sample in valid_samples]
    return summary


def _manifest_without_hash(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "manifest_sha256"}


def validate_split_manifest(
    payload: dict[str, Any], samples: Iterable[SemanticSample], *, exact_counts: bool = True
) -> None:
    """Reject sample overlap, group leakage, missing samples, and role drift."""
    if payload.get("schema_version") != "1.0" or payload.get("split_id") != "CSF-SPLIT-D":
        raise ValueError("split manifest must be CSF-SPLIT-D schema 1.0")
    expected_hash = sha256_payload(_manifest_without_hash(payload))
    if payload.get("manifest_sha256") != expected_hash:
        raise ValueError("split manifest hash mismatch")
    roles = payload.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(EXPECTED_ROLE_COUNTS):
        raise ValueError("split manifest roles are invalid")
    known = {sample.sample_id: sample for sample in samples}
    assigned: set[str] = set()
    group_roles: dict[str, str] = {}
    for role, records in roles.items():
        if not isinstance(records, list):
            raise ValueError(f"split role {role} must be a list")
        if exact_counts and len(records) != EXPECTED_ROLE_COUNTS[role]:
            raise ValueError(f"split role {role} count is not frozen")
        for record in records:
            sample_id = str(record["sample_id"])
            group_id = str(record["group_id"])
            if sample_id not in known or sample_id in assigned:
                raise ValueError(f"unknown or duplicate split sample: {sample_id}")
            if known[sample_id].group_id != group_id:
                raise ValueError(f"group identity mismatch for {sample_id}")
            previous = group_roles.setdefault(group_id, role)
            if previous != role:
                raise ValueError(f"sequence group leakage: {group_id}")
            if "/val/" in str(record["image"]):
                raise ValueError("official validation cannot enter trial roles")
            assigned.add(sample_id)
    if assigned != set(known):
        raise ValueError("split manifest does not cover every audited sample exactly once")


def build_split_manifest(samples: Iterable[SemanticSample], *, seed: int) -> dict[str, Any]:
    """Build the frozen split by deterministic sequence groups and exact counts."""
    sample_list = list(samples)
    if len(sample_list) != sum(EXPECTED_ROLE_COUNTS.values()):
        raise ValueError("CSF-SPLIT-D requires exactly 2,975 valid Cityscapes train samples")
    groups: defaultdict[str, list[SemanticSample]] = defaultdict(list)
    for sample in sample_list:
        groups[sample.group_id].append(sample)
    ordered_groups = sorted(
        groups.values(),
        key=lambda group: hashlib.sha256(f"{seed}:{group[0].group_id}".encode()).hexdigest(),
    )
    roles: dict[str, list[dict[str, Any]]] = {role: [] for role in EXPECTED_ROLE_COUNTS}
    role_order = ("train_select", "train_calibration", "train_fit")
    for group in ordered_groups:
        selected_role = next(
            (
                role
                for role in role_order
                if len(roles[role]) + len(group) <= EXPECTED_ROLE_COUNTS[role]
            ),
            None,
        )
        if selected_role is None:
            raise ValueError("sequence-atomic groups cannot satisfy frozen split counts")
        roles[selected_role].extend(
            asdict(sample) for sample in sorted(group, key=lambda x: x.sample_id)
        )
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "semantic_split_manifest",
        "split_id": "CSF-SPLIT-D",
        "dataset": "cityscapes_fine_train",
        "seed": seed,
        "group_identity": "city+sequence",
        "counts": {role: len(records) for role, records in roles.items()},
        "roles": roles,
        "official_val_role": "final_common_evaluation_only",
        "human_freeze_required": True,
    }
    payload["manifest_sha256"] = sha256_payload(payload)
    validate_split_manifest(payload, sample_list)
    return payload


def validate_or_build_split(
    samples: Iterable[SemanticSample],
    output_path: Path,
    *,
    seed: int,
    existing_path: Path | None = None,
) -> dict[str, Any]:
    """Validate an existing frozen split or deterministically create the candidate."""
    sample_list = list(samples)
    if existing_path is not None:
        payload = json.loads(existing_path.read_text(encoding="utf-8"))
        validate_split_manifest(payload, sample_list)
        status = "validated_existing"
    else:
        payload = build_split_manifest(sample_list, seed=seed)
        status = "generated_candidate_requires_human_freeze"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return {
        "status": status,
        "path": output_path.name,
        "manifest_sha256": payload["manifest_sha256"],
    }


def role_records(manifest_path: Path, role: str) -> list[dict[str, Any]]:
    """Return one validated role from a frozen rescue manifest."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = payload.get("roles")
    if not isinstance(roles, dict) or role not in roles:
        raise ValueError(f"split manifest has no role {role!r}")
    records = roles[role]
    if not isinstance(records, list):
        raise ValueError(f"split role {role!r} is invalid")
    return records


def write_train_fit_statistics(
    dataset_root: Path, manifest_path: Path, report_root: Path
) -> dict[str, Any]:
    """Freeze weights and rare groups from train_fit only after split validation."""
    role_pixel_counts: dict[str, np.ndarray] = {}
    role_image_counts: dict[str, np.ndarray] = {}
    role_sizes: dict[str, int] = {}
    for role in EXPECTED_ROLE_COUNTS:
        records = role_records(manifest_path, role)
        pixels = np.zeros(19, dtype=np.int64)
        images = np.zeros(19, dtype=np.int64)
        for record in records:
            cached_counts = record.get("class_pixel_counts")
            if isinstance(cached_counts, list) and len(cached_counts) == 19:
                sample_counts = np.asarray(cached_counts, dtype=np.int64)
                if bool((sample_counts < 0).any()):
                    raise ValueError(f"negative cached counts in {record['sample_id']}")
            else:
                mask_path = dataset_root / str(record["mask"])
                with Image.open(mask_path) as image:
                    mask = np.asarray(image, dtype=np.uint8)
                invalid = np.unique(mask[(mask > 18) & (mask != 255)])
                if invalid.size:
                    raise ValueError(f"invalid {role} labels in {record['sample_id']}: {invalid}")
                sample_counts = np.bincount(mask[mask != 255], minlength=19)[:19]
            pixels += sample_counts
            images += sample_counts > 0
        role_pixel_counts[role] = pixels
        role_image_counts[role] = images
        role_sizes[role] = len(records)
    counts = role_pixel_counts["train_fit"]
    weights = median_frequency_weights(counts)
    ordered = sorted(range(19), key=lambda class_id: (int(counts[class_id]), class_id))
    groups = {
        "rare": ordered[:6],
        "medium": ordered[6:12],
        "frequent": ordered[12:],
    }
    weights_payload = {
        "schema_version": "1.0",
        "method": "median_frequency_capped_mean_one",
        "maximum_before_renormalization": 10.0,
        "source_role": "train_fit",
        "split_manifest_sha256": sha256_file(manifest_path),
        "pixel_counts": [int(value) for value in counts],
        "weights": weights,
        "scientific_evidence": True,
    }
    rare_payload = {
        "schema_version": "1.0",
        "method": "train_fit_pixel_frequency_tertiles",
        "source_role": "train_fit",
        "groups": groups,
        "class_names": list(CITYSCAPES_CLASSES),
        "scientific_evidence": True,
    }
    (report_root / "class_weights.json").write_text(
        canonical_json(weights_payload) + "\n", encoding="utf-8"
    )
    (report_root / "rare_classes.json").write_text(
        canonical_json(rare_payload) + "\n", encoding="utf-8"
    )
    split_rows: list[dict[str, Any]] = []
    for role in EXPECTED_ROLE_COUNTS:
        role_total = int(role_pixel_counts[role].sum())
        for class_id, class_name in enumerate(CITYSCAPES_CLASSES):
            split_rows.append(
                {
                    "role": role,
                    "role_images": role_sizes[role],
                    "class_id": class_id,
                    "class_name": class_name,
                    "pixel_count": int(role_pixel_counts[role][class_id]),
                    "pixel_ratio": (
                        float(role_pixel_counts[role][class_id] / role_total)
                        if role_total
                        else None
                    ),
                    "image_count": int(role_image_counts[role][class_id]),
                    "image_ratio": (
                        float(role_image_counts[role][class_id] / role_sizes[role])
                        if role_sizes[role]
                        else None
                    ),
                }
            )
    _write_csv(
        report_root / "split_comparison.csv",
        [
            "role",
            "role_images",
            "class_id",
            "class_name",
            "pixel_count",
            "pixel_ratio",
            "image_count",
            "image_ratio",
        ],
        split_rows,
    )
    split_summary = {
        "schema_version": "1.0",
        "split_manifest_sha256": sha256_file(manifest_path),
        "roles": role_sizes,
        "all_roles_have_all_classes": all(
            bool((role_pixel_counts[role] > 0).all()) for role in EXPECTED_ROLE_COUNTS
        ),
    }
    (report_root / "split_summary.json").write_text(
        canonical_json(split_summary) + "\n", encoding="utf-8"
    )
    return {
        "class_weights": weights_payload,
        "rare_classes": rare_payload,
        "split_summary": split_summary,
    }
