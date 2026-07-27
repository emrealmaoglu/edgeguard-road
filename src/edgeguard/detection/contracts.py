"""Detector-neutral prediction and coordinate-transform contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Detection:
    """One XYXY detection on the original image grid."""

    box_xyxy: tuple[float, float, float, float]
    class_id: int
    confidence: float

    def validated(self, *, width: int, height: int) -> Detection:
        x1, y1, x2, y2 = self.box_xyxy
        if width <= 0 or height <= 0 or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("detection box must be ordered and inside image geometry")
        if self.class_id < 0 or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection class and confidence are invalid")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectorBatch:
    """Predictions tied to one image and transform identity."""

    image_id: str
    image_shape_hw: tuple[int, int]
    transform_id: str
    detections: tuple[Detection, ...]

    def validated(self) -> DetectorBatch:
        height, width = self.image_shape_hw
        if not self.image_id or not self.transform_id or height <= 0 or width <= 0:
            raise ValueError("detector batch identity and image shape are required")
        for detection in self.detections:
            detection.validated(width=width, height=height)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_shape_hw": list(self.image_shape_hw),
            "transform_id": self.transform_id,
            "detections": [detection.to_dict() for detection in self.detections],
        }


@dataclass(frozen=True)
class LetterboxTransform:
    """Reversible aspect-ratio-preserving resize plus symmetric padding."""

    source_hw: tuple[int, int]
    target_hw: tuple[int, int]
    scale: float
    pad_xy: tuple[float, float]

    @classmethod
    def for_shapes(
        cls, source_hw: tuple[int, int], target_hw: tuple[int, int]
    ) -> LetterboxTransform:
        source_h, source_w = source_hw
        target_h, target_w = target_hw
        if min(source_h, source_w, target_h, target_w) <= 0:
            raise ValueError("letterbox dimensions must be positive")
        scale = min(target_w / source_w, target_h / source_h)
        resized_w, resized_h = source_w * scale, source_h * scale
        padding = ((target_w - resized_w) / 2, (target_h - resized_h) / 2)
        return cls(source_hw, target_hw, scale, padding)

    def to_source(
        self, box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        pad_x, pad_y = self.pad_xy
        source_h, source_w = self.source_hw
        x1, y1, x2, y2 = box
        restored = (
            max(0.0, min(float(source_w), (x1 - pad_x) / self.scale)),
            max(0.0, min(float(source_h), (y1 - pad_y) / self.scale)),
            max(0.0, min(float(source_w), (x2 - pad_x) / self.scale)),
            max(0.0, min(float(source_h), (y2 - pad_y) / self.scale)),
        )
        if restored[0] >= restored[2] or restored[1] >= restored[3]:
            raise ValueError("letterbox reversal produced an empty box")
        return restored


def box_mask_overlap(detection: Detection, mask: npt.NDArray[np.bool_]) -> float:
    """Return the fraction of one integer-covered box overlapping a boolean mask."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("overlap mask must be a two-dimensional boolean array")
    detection.validated(width=mask.shape[1], height=mask.shape[0])
    x1, y1, x2, y2 = detection.box_xyxy
    left, top = int(np.floor(x1)), int(np.floor(y1))
    right, bottom = int(np.ceil(x2)), int(np.ceil(y2))
    covered = mask[top:bottom, left:right]
    return float(np.count_nonzero(covered) / covered.size) if covered.size else 0.0


def detection_metrics_schema() -> dict[str, Any]:
    """Return a path-free result schema without inventing detector measurements."""
    return {
        "schema_version": "1.0",
        "record_type": "detection_metrics_contract",
        "metrics": ["precision", "recall", "average_precision", "mean_average_precision"],
        "iou_thresholds": "record_at_runtime",
        "class_ids": "edgeguard_detection_ontology",
        "scientific_evidence": False,
    }
