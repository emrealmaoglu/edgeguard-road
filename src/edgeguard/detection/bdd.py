"""Narrow BDD-style synthetic annotation adapter."""

from __future__ import annotations

from typing import Any

from edgeguard.detection.contracts import Detection, DetectorBatch


def adapt_bdd_record(
    record: dict[str, Any], *, class_mapping: dict[str, int], image_shape_hw: tuple[int, int]
) -> DetectorBatch:
    """Map one inspected BDD-like record and reject unknown categories."""
    image_id = record.get("name")
    labels = record.get("labels")
    if not isinstance(image_id, str) or not isinstance(labels, list):
        raise ValueError("BDD record requires name and labels")
    detections: list[Detection] = []
    for label in labels:
        if not isinstance(label, dict) or label.get("category") not in class_mapping:
            raise ValueError("BDD record contains an unknown category")
        box = label.get("box2d")
        if not isinstance(box, dict):
            raise ValueError("BDD detection is missing box2d")
        detections.append(
            Detection(
                (float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])),
                class_mapping[str(label["category"])],
                float(label.get("score", 1.0)),
            )
        )
    return DetectorBatch(image_id, image_shape_hw, "identity", tuple(detections)).validated()
