"""Deterministic confidence intervals, paired summaries, and component metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from edgeguard.evaluation.components import ComponentRecord


def deterministic_bootstrap_interval(
    values: npt.NDArray[np.floating],
    *,
    statistic: Callable[[npt.NDArray[np.float64]], float] = np.mean,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 20260727,
) -> dict[str, float | int]:
    """Return one deterministic percentile bootstrap interval."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 2 or not np.isfinite(data).all():
        raise ValueError("bootstrap values must be a finite vector with at least two items")
    if not 0 < confidence < 1 or resamples < 100:
        raise ValueError("bootstrap confidence/resample contract is invalid")
    generator = np.random.default_rng(seed)
    samples = generator.choice(data, size=(resamples, data.size), replace=True)
    estimates = np.asarray([statistic(sample) for sample in samples], dtype=np.float64)
    alpha = (1 - confidence) / 2
    return {
        "estimate": float(statistic(data)),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1 - alpha)),
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
    }


def paired_comparison(left: dict[str, float], right: dict[str, float]) -> dict[str, Any]:
    """Compare aligned sample identities without asserting statistical significance."""
    if set(left) != set(right) or len(left) < 2:
        raise ValueError("paired comparisons require the same two-or-more sample identities")
    identities = sorted(left)
    differences = np.asarray([right[item] - left[item] for item in identities], dtype=np.float64)
    if not np.isfinite(differences).all():
        raise ValueError("paired comparison values must be finite")
    return {
        "sample_ids": identities,
        "differences": differences.tolist(),
        "mean_difference": float(differences.mean()),
        "bootstrap_interval": deterministic_bootstrap_interval(differences),
        "significance_claim": False,
    }


def component_detection_metrics(
    predictions: Sequence[Sequence[ComponentRecord]],
    targets: Sequence[Sequence[ComponentRecord]],
    *,
    minimum_iou: float = 0.1,
) -> dict[str, float | int | None]:
    """Measure matched target components and false-positive components per image."""
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("component metrics require aligned non-empty image lists")
    if not 0 < minimum_iou <= 1:
        raise ValueError("component IoU threshold must lie in (0, 1]")

    def iou(left: ComponentRecord, right: ComponentRecord) -> float:
        lx1, ly1, lx2, ly2 = left.bbox_xyxy
        rx1, ry1, rx2, ry2 = right.bbox_xyxy
        intersection = max(0, min(lx2, rx2) - max(lx1, rx1)) * max(0, min(ly2, ry2) - max(ly1, ry1))
        union = left.area + right.area - intersection
        return intersection / union if union else 0.0

    target_count = 0
    matched_count = 0
    false_positives = 0
    for predicted, actual in zip(predictions, targets, strict=True):
        target_count += len(actual)
        used: set[int] = set()
        for prediction in predicted:
            candidate = [
                (index, iou(prediction, target))
                for index, target in enumerate(actual)
                if index not in used
            ]
            best = max(candidate, key=lambda item: item[1]) if candidate else None
            if best is not None and best[1] >= minimum_iou:
                used.add(best[0])
                matched_count += 1
            else:
                false_positives += 1
    return {
        "target_component_count": target_count,
        "matched_component_count": matched_count,
        "component_detection_rate": matched_count / target_count if target_count else None,
        "false_positive_components_per_image": false_positives / len(predictions),
        "minimum_iou": minimum_iou,
    }
