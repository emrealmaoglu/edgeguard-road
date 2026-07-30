"""Semantic-to-region perception and explainable operational attention."""

from __future__ import annotations

from collections import deque
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
    labels = np.zeros(binary.shape, dtype=np.int32)
    components: list[np.ndarray] = []
    height, width = binary.shape
    next_label = 0
    for start_y, start_x in zip(*np.nonzero(binary & (labels == 0)), strict=True):
        if labels[start_y, start_x] != 0:
            continue
        next_label += 1
        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        labels[start_y, start_x] = next_label
        pixels: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and binary[next_y, next_x]
                    and labels[next_y, next_x] == 0
                ):
                    labels[next_y, next_x] = next_label
                    queue.append((next_y, next_x))
        components.append(np.asarray(pixels, dtype=np.int32))
    return labels, components


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


def _distance_from_mask(mask: np.ndarray) -> np.ndarray:
    """Compute four-neighbour distance to a boolean mask without SciPy."""
    height, width = mask.shape
    distance = np.full(mask.shape, np.inf, dtype=np.float32)
    queue: deque[tuple[int, int]] = deque()
    for y, x in zip(*np.nonzero(mask), strict=True):
        distance[y, x] = 0.0
        queue.append((int(y), int(x)))
    while queue:
        y, x = queue.popleft()
        candidate = distance[y, x] + 1.0
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (
                0 <= next_y < height
                and 0 <= next_x < width
                and candidate < distance[next_y, next_x]
            ):
                distance[next_y, next_x] = candidate
                queue.append((next_y, next_x))
    return distance


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
    for class_id in REGION_CLASS_IDS:
        _labels, components = _label_components(semantic_mask == class_id)
        for pixels in components:
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
