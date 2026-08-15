"""Semantic-to-region perception and explainable operational attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from edgeguard.rescue.dataset import CITYSCAPES_CLASSES

REGION_CLASS_IDS = (6, 7, 11, 12, 13, 14, 15, 16, 17, 18)
CLASS_IMPORTANCE = {
    6: 0.4,
    7: 0.4,
    11: 1.0,
    12: 1.0,
    13: 0.8,
    14: 0.8,
    15: 0.8,
    16: 0.8,
    17: 1.0,
    18: 1.0,
}
ATTENTION_WEIGHTS = {
    "class_importance": 0.35,
    "image_proximity": 0.25,
    "drivable_corridor_relation": 0.25,
    "uncertainty": 0.15,
}


@dataclass(frozen=True)
class SemanticRegion:
    """One class-connected region and its operational-attention evidence."""

    region_id: int
    class_id: int
    class_name: str
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    area_pixels: int
    mean_confidence: float
    mean_entropy: float
    attention_score: float
    attention_level: str
    contributions: dict[str, float]
    corridor_distance_pixels: float | None
    mask: np.ndarray

    def to_dict(self, *, mask_path: str | None = None) -> dict[str, Any]:
        """Serialize region evidence without embedding a full pixel array."""
        return {
            "region_id": self.region_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox_xyxy": list(self.bbox_xyxy),
            "centroid_xy": [float(value) for value in self.centroid_xy],
            "area_pixels": self.area_pixels,
            "mean_confidence": self.mean_confidence,
            "mean_entropy": self.mean_entropy,
            "attention_score": self.attention_score,
            "attention_level": self.attention_level,
            "contributions": self.contributions,
            "corridor_distance_pixels": self.corridor_distance_pixels,
            "mask": mask_path,
            "instance_detection": False,
        }


@dataclass(frozen=True)
class PerceptionResult:
    """Derived drivable, unreliable, region, and attention outputs."""

    road_mask: np.ndarray
    drivable_corridor: np.ndarray
    unreliable_mask: np.ndarray
    attention_map: np.ndarray
    regions: tuple[SemanticRegion, ...]
    confidence_threshold: float
    entropy_threshold: float


def _validate_semantic_inputs(
    mask: np.ndarray, confidence: np.ndarray, entropy: np.ndarray
) -> None:
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("semantic mask must be a two-dimensional integer array")
    if bool(((mask < 0) | (mask > 18)).any()):
        raise ValueError("semantic mask contains IDs outside Cityscapes19")
    for name, values in (("confidence", confidence), ("entropy", entropy)):
        if values.shape != mask.shape or not np.issubdtype(values.dtype, np.floating):
            raise ValueError(f"{name} must be floating and match semantic geometry")
        if not bool(np.isfinite(values).all()) or bool(((values < 0) | (values > 1)).any()):
            raise ValueError(f"{name} must be finite and lie in [0,1]")


def _label_components(binary: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Label four-connected components, numbered by raster order of first pixel.

    Labels propagate by whole-array maxima rather than a per-pixel breadth-first search.
    The two were timed against each other twice, and the answer depended on the machine:
    on the development Mac the search won at 1.4-2.5x, so propagation was written and
    reverted; on the Jetson Orin Nano Super that actually runs this, propagation won at
    1.59x (3.154 ms -> 1.989 ms per call over 220 real 64x128 masks), because an ARM CPU
    runs the Python interpreter far slower relative to NumPy. `derive_perception` calls
    this eleven times per frame, so that is 34.7 ms against 21.9 ms of a frame already
    dominated by CPU-side work. The target device decides.

    Component order and the returned label values are part of the contract: callers map a
    component's index onto its label (`labels == selected`) and derive `region_id` from
    list position. Pixel order *within* a component is not -- every consumer takes areas,
    extents or means -- so pixels come back in raster order rather than BFS order.
    """
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.int32), []
    seeds = np.arange(1, binary.size + 1, dtype=np.int64).reshape(binary.shape)
    labels = np.where(binary, seeds, 0)
    while True:
        merged = labels.copy()
        merged[1:, :] = np.maximum(merged[1:, :], labels[:-1, :])
        merged[:-1, :] = np.maximum(merged[:-1, :], labels[1:, :])
        merged[:, 1:] = np.maximum(merged[:, 1:], labels[:, :-1])
        merged[:, :-1] = np.maximum(merged[:, :-1], labels[:, 1:])
        merged[~binary] = 0
        if np.array_equal(merged, labels):
            break
        labels = merged

    flat = labels.reshape(-1)
    foreground = np.flatnonzero(flat)
    unique, inverse = np.unique(flat[foreground], return_inverse=True)
    # The search numbered components by the raster position of the pixel that started
    # them, so rank each component by its own first pixel to reproduce that numbering.
    first_index = np.full(unique.size, flat.size, dtype=np.int64)
    np.minimum.at(first_index, inverse, foreground)
    ranking = np.empty(unique.size, dtype=np.int32)
    ranking[np.argsort(first_index)] = np.arange(1, unique.size + 1, dtype=np.int32)
    renumbered = ranking[inverse]

    ordered = np.zeros(flat.size, dtype=np.int32)
    ordered[foreground] = renumbered
    grouped = np.argsort(renumbered, kind="stable")
    boundaries = np.searchsorted(renumbered[grouped], np.arange(1, unique.size + 2))
    width = binary.shape[1]
    components = [
        np.stack(
            (foreground[grouped[start:end]] // width, foreground[grouped[start:end]] % width),
            axis=1,
        ).astype(np.int32)
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    return ordered.reshape(binary.shape), components


def _label_components_by_class(
    semantic_mask: np.ndarray, class_ids: tuple[int, ...]
) -> dict[int, list[np.ndarray]]:
    """Label every requested class's components in a single propagation.

    Calling `_label_components` once per class walks the whole array once per class --
    eleven full passes per frame for the road plus ten attention classes, on the stage
    that already dominates the Jetson frame budget. Components of different classes can
    never merge, so restricting propagation to same-class neighbours resolves all of them
    at once for the cost of one.

    Ordering matches the per-class calls it replaces: within each class, components are
    numbered by the raster position of their first pixel.
    """
    wanted = np.isin(semantic_mask, class_ids)
    if not wanted.any():
        return {class_id: [] for class_id in class_ids}
    seeds = np.arange(1, semantic_mask.size + 1, dtype=np.int64).reshape(semantic_mask.shape)
    labels = np.where(wanted, seeds, 0)
    vertical = semantic_mask[1:, :] == semantic_mask[:-1, :]
    horizontal = semantic_mask[:, 1:] == semantic_mask[:, :-1]
    while True:
        merged = labels.copy()
        merged[1:, :] = np.where(vertical, np.maximum(merged[1:, :], labels[:-1, :]), merged[1:, :])
        merged[:-1, :] = np.where(
            vertical, np.maximum(merged[:-1, :], labels[1:, :]), merged[:-1, :]
        )
        merged[:, 1:] = np.where(
            horizontal, np.maximum(merged[:, 1:], labels[:, :-1]), merged[:, 1:]
        )
        merged[:, :-1] = np.where(
            horizontal, np.maximum(merged[:, :-1], labels[:, 1:]), merged[:, :-1]
        )
        merged[~wanted] = 0
        if np.array_equal(merged, labels):
            break
        labels = merged

    flat = labels.reshape(-1)
    foreground = np.flatnonzero(flat)
    unique, inverse = np.unique(flat[foreground], return_inverse=True)
    first_index = np.full(unique.size, flat.size, dtype=np.int64)
    np.minimum.at(first_index, inverse, foreground)
    grouped = np.argsort(inverse, kind="stable")
    boundaries = np.searchsorted(inverse[grouped], np.arange(unique.size + 1))
    width = semantic_mask.shape[1]
    classes = semantic_mask.reshape(-1)

    by_class: dict[int, list[tuple[int, np.ndarray]]] = {int(c): [] for c in class_ids}
    for index in range(unique.size):
        members = foreground[grouped[boundaries[index] : boundaries[index + 1]]]
        class_id = int(classes[members[0]])
        by_class[class_id].append(
            (
                int(first_index[index]),
                np.stack((members // width, members % width), axis=1).astype(np.int32),
            )
        )
    return {
        class_id: [pixels for _, pixels in sorted(entries, key=lambda entry: entry[0])]
        for class_id, entries in by_class.items()
    }


def drivable_corridor_from_semantics(
    semantic_mask: np.ndarray, *, minimum_area: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Return road pixels and the bottom-center-connected ego corridor."""
    if minimum_area <= 0:
        raise ValueError("minimum drivable area must be positive")
    if semantic_mask.ndim != 2 or not np.issubdtype(semantic_mask.dtype, np.integer):
        raise ValueError("semantic mask must be a two-dimensional integer array")
    road = semantic_mask == 0
    labels, components = _label_components(road)
    if not components:
        return road, np.zeros_like(road)
    height, width = road.shape
    anchor = np.zeros_like(road)
    anchor[max(0, height - max(1, height // 10)) :, width * 2 // 5 : width * 3 // 5] = True
    candidates: list[tuple[int, int, int]] = []
    for index, pixels in enumerate(components, start=1):
        area = int(pixels.shape[0])
        if area < minimum_area:
            continue
        anchor_count = int(np.count_nonzero(anchor[pixels[:, 0], pixels[:, 1]]))
        candidates.append((anchor_count, area, index))
    if not candidates:
        return road, np.zeros_like(road)
    anchor_count, _area, selected = max(candidates)
    if anchor_count == 0:
        return road, np.zeros_like(road)
    return road, labels == selected


def _axis_distance(distance: np.ndarray, axis: int) -> np.ndarray:
    """Propagate a one-dimensional min-plus sweep along one axis, in both directions.

    A forward sweep `d[i] = min(d[i], d[i-1] + 1)` expands to `d[i] = i + min(d[j] - j)`
    over `j <= i`, which is a running minimum -- so the sequential scan becomes a single
    `np.minimum.accumulate` instead of a Python loop.
    """
    length = distance.shape[axis]
    shape = [1] * distance.ndim
    shape[axis] = length
    # Both directions index from their own origin, so the reverse sweep reuses this same
    # ramp against the flipped array rather than a flipped ramp.
    offsets = np.arange(length, dtype=np.float32).reshape(shape)
    forward = np.minimum.accumulate(distance - offsets, axis=axis) + offsets
    flipped = np.flip(distance, axis=axis)
    backward = np.flip(np.minimum.accumulate(flipped - offsets, axis=axis) + offsets, axis=axis)
    return np.minimum(forward, backward)


def _distance_from_mask(mask: np.ndarray) -> np.ndarray:
    """Compute four-neighbour distance to a boolean mask without SciPy.

    Breadth-first search on an obstacle-free four-connected grid is exactly the L1
    distance to the nearest source, and the L1 transform is separable: sweep each row,
    then each column. That replaces a per-pixel Python queue -- which shared the blame for
    `derive_perception` costing 96 ms of a 146 ms Jetson frame -- with four accumulate
    passes, while returning the same distances.
    """
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    # A finite sentinel keeps the min-plus arithmetic free of inf-minus-inf; anything
    # still holding it afterwards is genuinely unreachable, which cannot happen here but
    # is restored as `inf` so the caller's `np.isfinite` guard keeps its meaning.
    unreachable = np.float32(mask.size + mask.shape[0] + mask.shape[1])
    distance = np.where(mask, np.float32(0.0), unreachable).astype(np.float32)
    distance = _axis_distance(distance, axis=1)
    distance = _axis_distance(distance, axis=0)
    return np.where(distance >= unreachable, np.inf, distance).astype(np.float32)


def _attention_level(score: float) -> str:
    if score < 0.35:
        return "low"
    if score <= 0.65:
        return "medium"
    return "high"


def derive_perception(
    semantic_mask: np.ndarray,
    confidence: np.ndarray,
    entropy: np.ndarray,
    *,
    confidence_threshold: float = 0.5,
    entropy_threshold: float = 0.5,
    minimum_region_area: int = 8,
    minimum_drivable_area: int = 64,
) -> PerceptionResult:
    """Derive the fixed Phase-1 perception contract from semantic probabilities."""
    _validate_semantic_inputs(semantic_mask, confidence, entropy)
    if not 0.0 <= confidence_threshold <= 1.0 or not 0.0 <= entropy_threshold <= 1.0:
        raise ValueError("uncertainty thresholds must lie in [0,1]")
    if minimum_region_area <= 0:
        raise ValueError("minimum region area must be positive")
    road, corridor = drivable_corridor_from_semantics(
        semantic_mask, minimum_area=minimum_drivable_area
    )
    distance = _distance_from_mask(corridor)
    height, _width = semantic_mask.shape
    relation_scale = max(1.0, height * 0.25)
    regions: list[SemanticRegion] = []
    attention_map = np.zeros(semantic_mask.shape, dtype=np.float32)
    components_by_class = _label_components_by_class(semantic_mask, REGION_CLASS_IDS)
    for class_id in REGION_CLASS_IDS:
        for pixels in components_by_class[class_id]:
            area = int(pixels.shape[0])
            if area < minimum_region_area:
                continue
            ys = pixels[:, 0]
            xs = pixels[:, 1]
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max() + 1), int(ys.max() + 1)
            proximity = float((y2 - 1) / max(1, height - 1))
            minimum_distance = float(np.min(distance[ys, xs]))
            corridor_relation = (
                0.0
                if not np.isfinite(minimum_distance)
                else float(np.clip(1.0 - minimum_distance / relation_scale, 0.0, 1.0))
            )
            features = {
                "class_importance": CLASS_IMPORTANCE[class_id],
                "image_proximity": proximity,
                "drivable_corridor_relation": corridor_relation,
                "uncertainty": float(np.mean(entropy[ys, xs])),
            }
            contributions = {
                name: float(features[name] * weight) for name, weight in ATTENTION_WEIGHTS.items()
            }
            score = float(sum(contributions.values()))
            region_mask = np.zeros(semantic_mask.shape, dtype=np.bool_)
            region_mask[ys, xs] = True
            attention_map[ys, xs] = score
            regions.append(
                SemanticRegion(
                    region_id=len(regions) + 1,
                    class_id=class_id,
                    class_name=CITYSCAPES_CLASSES[class_id],
                    bbox_xyxy=(x1, y1, x2, y2),
                    centroid_xy=(float(np.mean(xs)), float(np.mean(ys))),
                    area_pixels=area,
                    mean_confidence=float(np.mean(confidence[ys, xs])),
                    mean_entropy=features["uncertainty"],
                    attention_score=score,
                    attention_level=_attention_level(score),
                    contributions=contributions,
                    corridor_distance_pixels=(
                        minimum_distance if np.isfinite(minimum_distance) else None
                    ),
                    mask=region_mask,
                )
            )
    unreliable = (confidence < confidence_threshold) | (entropy > entropy_threshold)
    return PerceptionResult(
        road_mask=road,
        drivable_corridor=corridor,
        unreliable_mask=unreliable,
        attention_map=attention_map,
        regions=tuple(regions),
        confidence_threshold=confidence_threshold,
        entropy_threshold=entropy_threshold,
    )


def attention_contract() -> dict[str, Any]:
    """Return the frozen, non-probabilistic attention policy for evidence records."""
    return {
        "schema_version": "1.0",
        "record_type": "edgeguard_operational_attention_contract",
        "weights": ATTENTION_WEIGHTS,
        "thresholds": {"low_max_exclusive": 0.35, "medium_max_inclusive": 0.65},
        "class_importance": {
            CITYSCAPES_CLASSES[class_id]: value for class_id, value in CLASS_IMPORTANCE.items()
        },
        "physical_risk_probability": False,
        "safety_certified": False,
    }
