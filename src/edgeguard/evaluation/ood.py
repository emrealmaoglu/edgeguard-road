"""Pixel-level OOD development metrics with no threshold selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

ID_LABEL = 0
ANOMALY_LABEL = 1
IGNORE_LABEL = 255


@dataclass(frozen=True)
class OODPixelMetricResult:
    """Threshold-free pixel OOD metrics and evaluated class counts."""

    average_precision: float | None
    fpr_at_95_tpr: float | None
    anomaly_pixel_count: int
    id_pixel_count: int
    ignored_pixel_count: int


def _validated_pixels(
    scores: npt.NDArray[np.floating], labels: npt.NDArray[np.integer], *, ignore_index: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], int]:
    if not isinstance(scores, np.ndarray) or not np.issubdtype(scores.dtype, np.floating):
        raise ValueError("OOD scores must be a floating-point numpy array")
    if not isinstance(labels, np.ndarray) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("OOD labels must be an integer numpy array")
    if scores.shape != labels.shape or scores.size == 0:
        raise ValueError("OOD scores and labels must have the same non-empty shape")
    if not np.isfinite(scores).all():
        raise ValueError("OOD scores must be finite")
    allowed = np.isin(labels, np.array([ID_LABEL, ANOMALY_LABEL, ignore_index]))
    if not bool(np.all(allowed)):
        raise ValueError("OOD labels must contain only ID=0, anomaly=1, or ignore")

    valid = labels != ignore_index
    valid_scores = scores[valid].astype(np.float64, copy=False)
    positives = (labels[valid] == ANOMALY_LABEL).astype(np.bool_, copy=False)
    return valid_scores, positives, int(np.count_nonzero(~valid))


def _ranking_curve(
    scores: npt.NDArray[np.float64], positives: npt.NDArray[np.bool_]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_positive = positives[order]
    group_ends = np.concatenate(
        (np.flatnonzero(np.diff(sorted_scores)) + 1, np.array([scores.size], dtype=np.int64))
    )
    true_positives = np.cumsum(sorted_positive, dtype=np.int64)[group_ends - 1]
    false_positives = group_ends - true_positives
    return true_positives.astype(np.float64), false_positives.astype(np.float64)


def pixel_ood_metrics(
    scores: npt.NDArray[np.floating],
    labels: npt.NDArray[np.integer],
    *,
    ignore_index: int = IGNORE_LABEL,
) -> OODPixelMetricResult:
    """Compute AP and FPR at at least 95% TPR; higher scores mean anomaly."""
    valid_scores, positives, ignored_count = _validated_pixels(
        scores, labels, ignore_index=ignore_index
    )
    positive_count = int(np.count_nonzero(positives))
    negative_count = int(positives.size - positive_count)
    if positive_count == 0:
        return OODPixelMetricResult(None, None, 0, negative_count, ignored_count)

    true_positives, false_positives = _ranking_curve(valid_scores, positives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / positive_count
    recall_steps = np.diff(np.concatenate((np.array([0.0]), recall)))
    average_precision = float(np.sum(recall_steps * precision))

    fpr_at_95_tpr: float | None = None
    if negative_count > 0:
        first_at_target = int(np.flatnonzero(recall >= 0.95)[0])
        fpr_at_95_tpr = float(false_positives[first_at_target] / negative_count)
    return OODPixelMetricResult(
        average_precision=average_precision,
        fpr_at_95_tpr=fpr_at_95_tpr,
        anomaly_pixel_count=positive_count,
        id_pixel_count=negative_count,
        ignored_pixel_count=ignored_count,
    )
