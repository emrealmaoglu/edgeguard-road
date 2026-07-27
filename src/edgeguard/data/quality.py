"""Project-specific data quality and compact EDA for CV fixtures/manifests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from edgeguard.data.contracts import DetectionSample, OODSample, SemanticSample, TemporalFrame


@dataclass(frozen=True)
class QualityIssue:
    """One deterministic, machine-readable data-quality finding."""

    code: str
    sample_id: str
    severity: str
    detail: str


def _array_sha256(array: npt.NDArray[np.generic]) -> str:
    header = f"{array.dtype.str}:{array.shape}".encode()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def bounded_perceptual_duplicate_pairs(
    images: dict[str, npt.NDArray[np.uint8]], *, maximum_samples: int = 128, distance: int = 0
) -> list[dict[str, object]]:
    """Find bounded average-hash near duplicates; this is diagnostic, not identity proof."""
    if not 1 <= maximum_samples <= 1024 or not 0 <= distance <= 64:
        raise ValueError("perceptual duplicate bounds are invalid")
    from PIL import Image

    hashes: dict[str, int] = {}
    for sample_id in sorted(images)[:maximum_samples]:
        image = images[sample_id]
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("perceptual duplicate input must be HWC uint8 RGB")
        reduced = np.asarray(
            Image.fromarray(image).convert("L").resize((8, 8), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        bits = reduced >= float(reduced.mean())
        hashes[sample_id] = int("".join("1" if value else "0" for value in bits.flat), 2)
    pairs: list[dict[str, object]] = []
    identities = sorted(hashes)
    for left_index, left in enumerate(identities):
        for right in identities[left_index + 1 :]:
            observed = (hashes[left] ^ hashes[right]).bit_count()
            if observed <= distance:
                pairs.append({"left": left, "right": right, "hamming_distance": observed})
    return pairs


def audit_semantic_samples(samples: tuple[SemanticSample, ...]) -> dict[str, Any]:
    """Validate semantic samples and summarize duplicates, balance, and geometry."""
    issues: list[QualityIssue] = []
    seen_ids: set[str] = set()
    image_hashes: dict[str, str] = {}
    group_roles: dict[str, str] = {}
    pixel_counts = np.zeros(19, dtype=np.uint64)
    presence_counts = np.zeros(19, dtype=np.uint64)
    resolutions: list[tuple[int, int]] = []
    ignored = 0
    total = 0
    spatial = np.zeros((19, 4, 4), dtype=np.uint64)
    for sample in samples:
        try:
            sample.validate()
        except ValueError as error:
            issues.append(
                QualityIssue("invalid_semantic_sample", sample.image_id, "error", str(error))
            )
            continue
        if sample.image_id in seen_ids:
            issues.append(
                QualityIssue("duplicate_sample_id", sample.image_id, "error", "sample ID repeated")
            )
        seen_ids.add(sample.image_id)
        digest = _array_sha256(sample.image)
        if digest in image_hashes:
            issues.append(
                QualityIssue(
                    "exact_duplicate_image",
                    sample.image_id,
                    "error",
                    f"duplicates {image_hashes[digest]}",
                )
            )
        image_hashes[digest] = sample.image_id
        previous = group_roles.setdefault(sample.group_id, sample.split_role)
        if previous != sample.split_role:
            issues.append(QualityIssue("group_leakage", sample.image_id, "error", sample.group_id))
        valid = sample.train_ids != 255
        counts = np.bincount(sample.train_ids[valid], minlength=19).astype(np.uint64)
        pixel_counts += counts
        presence_counts += (counts > 0).astype(np.uint64)
        ignored += int(np.count_nonzero(~valid))
        total += int(valid.size)
        resolutions.append((int(sample.train_ids.shape[0]), int(sample.train_ids.shape[1])))
        height, width = sample.train_ids.shape
        for y_bin in range(4):
            for x_bin in range(4):
                crop = sample.train_ids[
                    y_bin * height // 4 : (y_bin + 1) * height // 4,
                    x_bin * width // 4 : (x_bin + 1) * width // 4,
                ]
                values = crop[crop != 255]
                if values.size:
                    spatial[:, y_bin, x_bin] += np.bincount(values, minlength=19).astype(np.uint64)
    valid_pixels = int(pixel_counts.sum())
    return {
        "sample_count": len(samples),
        "valid_sample_count": len(samples)
        - sum(issue.code == "invalid_semantic_sample" for issue in issues),
        "issues": [issue.__dict__ for issue in issues],
        "class_pixel_counts": [int(value) for value in pixel_counts],
        "class_pixel_frequencies": [
            float(value / valid_pixels) if valid_pixels else None for value in pixel_counts
        ],
        "class_image_presence_counts": [int(value) for value in presence_counts],
        "ignored_pixel_ratio": ignored / total if total else None,
        "resolution_distribution": [list(item) for item in sorted(resolutions)],
        "spatial_class_heatmap_4x4": spatial.tolist(),
        "scientific_evidence": False,
    }


def audit_detection_samples(samples: tuple[DetectionSample, ...]) -> dict[str, Any]:
    """Validate detection boxes and summarize count, size, aspect, and source balance."""
    issues: list[QualityIssue] = []
    counts: list[int] = []
    areas: list[float] = []
    aspects: list[float] = []
    sources: dict[str, int] = {}
    for sample in samples:
        try:
            sample.validate()
        except ValueError as error:
            issues.append(
                QualityIssue("invalid_detection_sample", sample.image_id, "error", str(error))
            )
            continue
        counts.append(int(sample.boxes_xyxy.shape[0]))
        sources[sample.source] = sources.get(sample.source, 0) + 1
        widths = sample.source_boxes_xyxy[:, 2] - sample.source_boxes_xyxy[:, 0]
        heights = sample.source_boxes_xyxy[:, 3] - sample.source_boxes_xyxy[:, 1]
        areas.extend(float(value) for value in widths * heights)
        aspects.extend(float(value) for value in widths / heights)
    return {
        "sample_count": len(samples),
        "issues": [issue.__dict__ for issue in issues],
        "object_counts": counts,
        "box_areas": areas,
        "box_aspect_ratios": aspects,
        "source_counts": dict(sorted(sources.items())),
        "scientific_evidence": False,
    }


def audit_ood_samples(samples: tuple[OODSample, ...]) -> dict[str, Any]:
    """Validate OOD masks and summarize anomaly locations and source balance."""
    issues: list[QualityIssue] = []
    heatmap = np.zeros((4, 4), dtype=np.uint64)
    sources: dict[str, int] = {}
    anomaly_pixels = 0
    valid_pixels = 0
    for sample in samples:
        try:
            sample.validate()
        except ValueError as error:
            issues.append(QualityIssue("invalid_ood_sample", sample.image_id, "error", str(error)))
            continue
        sources[sample.source] = sources.get(sample.source, 0) + 1
        target = sample.anomaly_target
        valid_pixels += int(np.count_nonzero(sample.valid_mask))
        anomaly_pixels += int(np.count_nonzero((target == 1) & sample.valid_mask))
        height, width = target.shape
        for y_bin in range(4):
            for x_bin in range(4):
                crop = target[
                    y_bin * height // 4 : (y_bin + 1) * height // 4,
                    x_bin * width // 4 : (x_bin + 1) * width // 4,
                ]
                heatmap[y_bin, x_bin] += np.count_nonzero(crop == 1)
    return {
        "sample_count": len(samples),
        "issues": [issue.__dict__ for issue in issues],
        "source_counts": dict(sorted(sources.items())),
        "anomaly_pixel_count": anomaly_pixels,
        "valid_pixel_count": valid_pixels,
        "anomaly_location_heatmap_4x4": heatmap.tolist(),
        "scientific_evidence": False,
    }


def audit_temporal_frames(frames: tuple[TemporalFrame, ...]) -> dict[str, Any]:
    """Validate deterministic sequence ordering, gaps, and sequence lengths."""
    issues: list[QualityIssue] = []
    by_sequence: dict[str, list[TemporalFrame]] = {}
    for frame in frames:
        try:
            frame.validate()
        except ValueError as error:
            issues.append(
                QualityIssue("invalid_temporal_frame", frame.image_id, "error", str(error))
            )
            continue
        by_sequence.setdefault(frame.sequence_id, []).append(frame)
    lengths: dict[str, int] = {}
    for sequence_id, items in sorted(by_sequence.items()):
        ordered = sorted(items, key=lambda item: item.frame_index)
        indices = [item.frame_index for item in ordered]
        if len(indices) != len(set(indices)):
            issues.append(QualityIssue("duplicate_frame_index", sequence_id, "error", str(indices)))
        for previous, current in zip(ordered, ordered[1:], strict=False):
            expected_gap = current.frame_index - previous.frame_index - 1
            if current.gap_from_previous != expected_gap:
                issues.append(
                    QualityIssue(
                        "gap_metadata_mismatch",
                        current.image_id,
                        "error",
                        f"expected {expected_gap}",
                    )
                )
            if (
                previous.timestamp_seconds is not None
                and current.timestamp_seconds is not None
                and current.timestamp_seconds <= previous.timestamp_seconds
            ):
                issues.append(
                    QualityIssue("non_monotonic_timestamp", current.image_id, "error", sequence_id)
                )
        lengths[sequence_id] = len(ordered)
    return {
        "sequence_count": len(by_sequence),
        "sequence_lengths": lengths,
        "issues": [issue.__dict__ for issue in issues],
        "scientific_evidence": False,
    }
