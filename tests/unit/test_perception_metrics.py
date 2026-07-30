from __future__ import annotations

import numpy as np

from edgeguard.evaluation.perception import component_localization_metrics, drivable_metrics


def test_drivable_metrics_exclude_ignore_and_report_fragmentation() -> None:
    target = np.full((6, 8), 2, dtype=np.uint8)
    target[3:, 2:6] = 0
    target[0, :] = 255
    prediction = target == 0
    result = drivable_metrics(prediction, target)
    assert result["road_iou"] == 1.0
    assert result["false_drivable_rate"] == 0.0
    assert result["largest_component_fraction"] == 1.0
    assert result["ignore_pixels_excluded"] is True


def test_component_metrics_expose_merge_without_calling_it_detection() -> None:
    target = np.zeros((6, 10), dtype=np.uint8)
    target[1:3, 1:3] = 13
    target[1:3, 5:7] = 13
    prediction = target.copy()
    prediction[1:3, 3:5] = 13
    result = component_localization_metrics(prediction, target, minimum_area=2)
    assert result["ground_truth_component_count"] == 2
    assert result["predicted_component_count"] == 1
    assert result["mean_ground_truths_per_prediction"] == 2.0
    assert result["instance_detection_metric"] is False
