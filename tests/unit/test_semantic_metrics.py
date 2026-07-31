"""Tests for streaming semantic segmentation metrics."""

import numpy as np
import pytest

from edgeguard.evaluation.semantic import SemanticConfusionMatrix


def test_perfect_semantic_prediction() -> None:
    metrics = SemanticConfusionMatrix(num_classes=3)
    target = np.array([[0, 1], [2, 255]], dtype=np.uint8)

    metrics.update(np.array([[0, 1], [2, 0]], dtype=np.int64), target)
    result = metrics.result()

    assert result["mean_iou"] == 1.0
    assert result["pixel_accuracy"] == 1.0
    assert result["mean_class_accuracy"] == 1.0
    assert result["evaluated_pixel_count"] == 3
    assert result["ignored_pixel_count"] == 1


def test_wrong_and_mixed_semantic_prediction() -> None:
    metrics = SemanticConfusionMatrix(num_classes=3)

    metrics.update(
        np.array([[0, 2], [2, 1]], dtype=np.int64),
        np.array([[0, 1], [2, 1]], dtype=np.uint8),
    )
    result = metrics.result()

    assert result["pixel_accuracy"] == 0.75
    assert result["per_class_iou"] == [1.0, 0.5, 0.5]
    assert result["mean_iou"] == pytest.approx(2.0 / 3.0)


def test_absent_class_is_null_and_prediction_only_class_is_zero() -> None:
    metrics = SemanticConfusionMatrix(num_classes=3)

    metrics.update(
        np.array([[0, 1]], dtype=np.int64),
        np.array([[0, 0]], dtype=np.uint8),
    )
    result = metrics.result()

    assert result["per_class_iou"] == [0.5, 0.0, None]
    assert result["per_class_accuracy"] == [0.5, None, None]


def test_streaming_merge_matches_single_accumulator() -> None:
    first = SemanticConfusionMatrix(num_classes=2)
    second = SemanticConfusionMatrix(num_classes=2)
    combined = SemanticConfusionMatrix(num_classes=2)
    prediction_a = np.array([[0, 1]], dtype=np.int64)
    target_a = np.array([[0, 1]], dtype=np.uint8)
    prediction_b = np.array([[1, 1]], dtype=np.int64)
    target_b = np.array([[0, 255]], dtype=np.uint8)

    first.update(prediction_a, target_a)
    second.update(prediction_b, target_b)
    first.merge(second)
    combined.update(
        np.concatenate((prediction_a, prediction_b), axis=1),
        np.concatenate((target_a, target_b), axis=1),
    )

    assert first.result() == combined.result()


def test_empty_evaluated_region_has_null_aggregates() -> None:
    metrics = SemanticConfusionMatrix(num_classes=2)

    metrics.update(
        np.array([[0, 1]], dtype=np.int64),
        np.array([[255, 255]], dtype=np.uint8),
    )
    result = metrics.result()

    assert result["mean_iou"] is None
    assert result["pixel_accuracy"] is None
    assert result["mean_class_accuracy"] is None
    assert result["evaluated_pixel_count"] == 0


@pytest.mark.parametrize(
    ("prediction", "target", "message"),
    [
        (
            np.array([[2]], dtype=np.int64),
            np.array([[0]], dtype=np.uint8),
            "prediction contains",
        ),
        (
            np.array([[0]], dtype=np.int64),
            np.array([[2]], dtype=np.uint8),
            "target contains",
        ),
    ],
)
def test_invalid_semantic_ids_are_rejected(
    prediction: np.ndarray, target: np.ndarray, message: str
) -> None:
    metrics = SemanticConfusionMatrix(num_classes=2)

    with pytest.raises(ValueError, match=message):
        metrics.update(prediction, target)
