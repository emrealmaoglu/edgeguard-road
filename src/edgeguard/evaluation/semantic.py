"""Streaming semantic-segmentation metrics with explicit empty-class behavior."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


class SemanticConfusionMatrix:
    """Accumulate a fixed-class semantic confusion matrix with one ignore ID."""

    def __init__(self, num_classes: int = 19, ignore_index: int = 255) -> None:
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self._ignored_pixels = 0
        self._evaluated_pixels = 0

    def update(self, prediction: npt.NDArray[Any], target: npt.NDArray[Any]) -> None:
        """Accumulate one prediction/target pair after strict ID validation."""
        if not isinstance(prediction, np.ndarray) or not isinstance(target, np.ndarray):
            raise ValueError("prediction and target must be NumPy arrays")
        if prediction.shape != target.shape or prediction.ndim not in {2, 3}:
            raise ValueError(
                "prediction and target must have matching HW or NHW shapes, "
                f"got {prediction.shape} and {target.shape}"
            )
        if any(dimension <= 0 for dimension in prediction.shape):
            raise ValueError("prediction and target dimensions must be positive")
        if not np.issubdtype(prediction.dtype, np.integer) or not np.issubdtype(
            target.dtype, np.integer
        ):
            raise ValueError("prediction and target must use integer dtypes")

        prediction64 = prediction.astype(np.int64, copy=False)
        target64 = target.astype(np.int64, copy=False)
        if np.any((prediction64 < 0) | (prediction64 >= self.num_classes)):
            raise ValueError("prediction contains an invalid semantic class ID")
        valid_target = (target64 >= 0) & (target64 < self.num_classes)
        ignored_target = target64 == self.ignore_index
        if not np.all(valid_target | ignored_target):
            raise ValueError("target contains an invalid semantic class ID")

        self._ignored_pixels += int(np.count_nonzero(ignored_target))
        self._evaluated_pixels += int(np.count_nonzero(valid_target))
        encoded = self.num_classes * target64[valid_target] + prediction64[valid_target]
        counts = np.bincount(encoded, minlength=self.num_classes**2)
        self._matrix += counts.reshape(self.num_classes, self.num_classes)

    def merge(self, other: SemanticConfusionMatrix) -> None:
        """Merge another accumulator with identical class/ignore contracts."""
        if self.num_classes != other.num_classes or self.ignore_index != other.ignore_index:
            raise ValueError("cannot merge semantic metrics with different contracts")
        self._matrix += other._matrix
        self._ignored_pixels += other._ignored_pixels
        self._evaluated_pixels += other._evaluated_pixels

    def result(self) -> dict[str, Any]:
        """Return JSON-safe aggregate and per-class metrics without NaN values."""
        intersection = np.diag(self._matrix)
        target_count = np.sum(self._matrix, axis=1)
        prediction_count = np.sum(self._matrix, axis=0)
        union = target_count + prediction_count - intersection

        per_class_iou: list[float | None] = []
        per_class_accuracy: list[float | None] = []
        for class_index in range(self.num_classes):
            class_union = int(union[class_index])
            class_target = int(target_count[class_index])
            per_class_iou.append(
                None if class_union == 0 else float(intersection[class_index] / class_union)
            )
            per_class_accuracy.append(
                None if class_target == 0 else float(intersection[class_index] / class_target)
            )

        defined_ious = [value for value in per_class_iou if value is not None]
        defined_accuracies = [value for value in per_class_accuracy if value is not None]
        correct_pixels = int(np.trace(self._matrix))
        return {
            "schema_version": "1.0",
            "record_type": "semantic_metrics",
            "num_classes": self.num_classes,
            "ignore_index": self.ignore_index,
            "confusion_matrix": self._matrix.tolist(),
            "per_class_iou": per_class_iou,
            "mean_iou": (float(np.mean(defined_ious, dtype=np.float64)) if defined_ious else None),
            "pixel_accuracy": (
                correct_pixels / self._evaluated_pixels if self._evaluated_pixels else None
            ),
            "per_class_accuracy": per_class_accuracy,
            "mean_class_accuracy": (
                float(np.mean(defined_accuracies, dtype=np.float64)) if defined_accuracies else None
            ),
            "ignored_pixel_count": self._ignored_pixels,
            "evaluated_pixel_count": self._evaluated_pixels,
        }
