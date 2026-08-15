"""Drivable-area and semantic-component localization metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from edgeguard.evaluation.components import label_components
from edgeguard.rescue.perception import REGION_CLASS_IDS


def _component_sizes(mask: np.ndarray) -> list[int]:
    """Return each four-connected component's pixel count, without materialising masks.

    The per-pixel search this replaces allocated a full-size boolean array *per component*
    and walked the image in Python. On a 2048x1024 road mask that is minutes per frame,
    which is why the drivable metrics had no caller. Callers that only need areas should
    ask for areas.
    """
    _, components = label_components(mask)
    return [int(component.shape[0]) for component in components]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return each four-connected component as its own boolean mask.

    Kept for callers that genuinely need the masks (component matching intersects them),
    but the labelling itself is run-based rather than a per-pixel Python search.
    """
    _, components = label_components(mask)
    result: list[np.ndarray] = []
    for pixels in components:
        component = np.zeros(mask.shape, dtype=np.bool_)
        component[pixels[:, 0], pixels[:, 1]] = True
        result.append(component)
    return result


def _boundary(mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(mask)
    result[1:, :] |= mask[1:, :] != mask[:-1, :]
    result[:-1, :] |= mask[:-1, :] != mask[1:, :]
    result[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    result[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return result


def _dilate(mask: np.ndarray) -> np.ndarray:
    result = mask.copy()
    result[1:, :] |= mask[:-1, :]
    result[:-1, :] |= mask[1:, :]
    result[:, 1:] |= mask[:, :-1]
    result[:, :-1] |= mask[:, 1:]
    return result


def _widen(mask: np.ndarray, tolerance: int) -> np.ndarray:
    for _ in range(tolerance):
        mask = _dilate(mask)
    return mask


def _boundary_f1(prediction: np.ndarray, target: np.ndarray, tolerance: int) -> float:
    pred_total = int(np.count_nonzero(prediction))
    target_total = int(np.count_nonzero(target))
    if not pred_total or not target_total:
        return 0.0
    precision = int(np.count_nonzero(prediction & _widen(target, tolerance))) / pred_total
    recall = int(np.count_nonzero(target & _widen(prediction, tolerance))) / target_total
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def drivable_metrics(
    prediction: np.ndarray, target_semantics: np.ndarray, *, output_stride: int = 8
) -> dict[str, Any]:
    """Measure road mask accuracy without treating ignore pixels as non-road.

    Boundary agreement is reported at two tolerances because one alone misleads. At 1 px
    it asks for pixel-exact edges from a mask that was upsampled from stride-8 logits,
    which no amount of training could deliver -- the answer is bounded by the output
    resolution, not by the model. At the stride it asks the question the architecture can
    actually be held to. Both are measured; the gap between them is the cost of predicting
    coarsely and upsampling, and it belongs in the record rather than in a footnote.
    """
    if prediction.shape != target_semantics.shape or prediction.dtype != np.bool_:
        raise ValueError("drivable prediction must be bool and match target geometry")
    if not np.issubdtype(target_semantics.dtype, np.integer):
        raise ValueError("target semantics must use integer IDs")
    if output_stride < 1:
        raise ValueError("output stride must be positive")
    valid = target_semantics != 255
    target = target_semantics == 0
    intersection = int(np.count_nonzero(prediction & target & valid))
    union = int(np.count_nonzero((prediction | target) & valid))
    nonroad = int(np.count_nonzero((~target) & valid))
    false_drivable = int(np.count_nonzero(prediction & (~target) & valid))
    pred_boundary = _boundary(prediction) & valid
    target_boundary = _boundary(target) & valid
    boundary_f1 = _boundary_f1(pred_boundary, target_boundary, 1)
    component_areas = _component_sizes(prediction & valid)
    total_area = sum(component_areas)
    return {
        "road_iou": intersection / union if union else 1.0,
        "road_boundary_f1_tolerance_1px": boundary_f1,
        f"road_boundary_f1_tolerance_{output_stride}px": _boundary_f1(
            pred_boundary, target_boundary, output_stride
        ),
        "false_drivable_rate": false_drivable / nonroad if nonroad else 0.0,
        "predicted_component_count": len(component_areas),
        "largest_component_fraction": max(component_areas, default=0) / max(1, total_area),
        "ignore_pixels_excluded": True,
    }


def component_localization_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    minimum_area: int = 8,
) -> dict[str, Any]:
    """Compare semantic components without claiming instance-detection metrics."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target semantic masks must share 2D geometry")
    if minimum_area <= 0:
        raise ValueError("minimum component area must be positive")
    matched_ious: list[float] = []
    ground_truth_count = 0
    prediction_count = 0
    covered_ground_truth = 0
    fragmentation_counts: list[int] = []
    merge_counts: list[int] = []
    for class_id in REGION_CLASS_IDS:
        predicted = [
            component
            for component in _components(prediction == class_id)
            if np.count_nonzero(component) >= minimum_area
        ]
        expected = [
            component
            for component in _components(target == class_id)
            if np.count_nonzero(component) >= minimum_area
        ]
        prediction_count += len(predicted)
        ground_truth_count += len(expected)
        overlaps = np.zeros((len(expected), len(predicted)), dtype=np.float64)
        for gt_index, gt_component in enumerate(expected):
            for pred_index, pred_component in enumerate(predicted):
                intersection = np.count_nonzero(gt_component & pred_component)
                union = np.count_nonzero(gt_component | pred_component)
                overlaps[gt_index, pred_index] = intersection / union if union else 0.0
        for gt_index in range(len(expected)):
            positive = int(np.count_nonzero(overlaps[gt_index] > 0))
            fragmentation_counts.append(positive)
            if positive:
                covered_ground_truth += 1
                matched_ious.append(float(np.max(overlaps[gt_index])))
        for pred_index in range(len(predicted)):
            merge_counts.append(int(np.count_nonzero(overlaps[:, pred_index] > 0)))
    return {
        "ground_truth_component_count": ground_truth_count,
        "predicted_component_count": prediction_count,
        "component_coverage": covered_ground_truth / max(1, ground_truth_count),
        "mean_best_component_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "mean_fragments_per_ground_truth": (
            float(np.mean(fragmentation_counts)) if fragmentation_counts else 0.0
        ),
        "mean_ground_truths_per_prediction": (
            float(np.mean(merge_counts)) if merge_counts else 0.0
        ),
        "instance_detection_metric": False,
    }
