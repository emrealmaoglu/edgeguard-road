"""Synthetic tests for OOD, components, detection, risk, and temporal behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edgeguard.context import RiskWeights, contextual_risk
from edgeguard.detection.bdd import adapt_bdd_record
from edgeguard.detection.contracts import Detection, LetterboxTransform, box_mask_overlap
from edgeguard.evaluation.components import ComponentRecord, connected_components
from edgeguard.evaluation.ood import pixel_ood_metrics, select_anomaly_threshold
from edgeguard.scoring.anomaly_head import LinearAnomalyHead, synthetic_outlier_exposure
from edgeguard.temporal import TemporalPersistence


def _component(identifier: int, x: int, score: float = 0.8) -> ComponentRecord:
    return ComponentRecord(identifier, 4, (x, 2, x + 2, 4), (x + 0.5, 2.5), score, score, 1.0)


def test_ood_auroc_threshold_and_ignore_direction() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1, 100.0], dtype=np.float32)
    labels = np.array([1, 1, 0, 0, 255], dtype=np.uint8)
    result = pixel_ood_metrics(scores, labels)
    threshold = select_anomaly_threshold(scores, labels, target_tpr=0.95)
    assert result.auroc == pytest.approx(1.0)
    assert result.score_direction == "higher_means_more_anomalous"
    assert threshold["threshold"] == pytest.approx(0.8)
    assert threshold["ignored_pixel_count"] == 1


def test_components_report_road_overlap_and_geometry() -> None:
    mask = np.zeros((5, 7), dtype=np.bool_)
    mask[1:3, 1:3] = True
    mask[4, 6] = True
    scores = np.arange(35, dtype=np.float32).reshape(5, 7)
    road = np.zeros_like(mask)
    road[1:3, 1:3] = True
    components = connected_components(mask, scores, road_mask=road)
    assert [item.area for item in components] == [4, 1]
    assert components[0].road_overlap == 1.0
    assert components[1].road_overlap == 0.0


def test_anomaly_head_checkpoint_resume_and_identity(tmp_path: Path) -> None:
    features, targets = synthetic_outlier_exposure(seed=7, sample_count=40, feature_count=3)
    head = LinearAnomalyHead.initialized(3)
    first_losses = head.train_steps(features, targets, steps=2, learning_rate=0.2)
    path = tmp_path / "head.json"
    head.checkpoint(path, identity={"run": "one"})
    resumed = LinearAnomalyHead.resume(path, identity={"run": "one"})
    second_losses = resumed.train_steps(features, targets, steps=1, learning_rate=0.2)
    assert resumed.optimizer_step == 3
    assert second_losses[0] <= first_losses[0]
    with pytest.raises(ValueError, match="identity"):
        LinearAnomalyHead.resume(path, identity={"run": "other"})


def test_detection_adapter_transform_and_overlap() -> None:
    batch = adapt_bdd_record(
        {
            "name": "frame",
            "labels": [
                {
                    "category": "car",
                    "box2d": {"x1": 10, "y1": 10, "x2": 20, "y2": 20},
                }
            ],
        },
        class_mapping={"car": 0},
        image_shape_hw=(40, 80),
    )
    assert batch.detections[0].class_id == 0
    transform = LetterboxTransform.for_shapes((40, 80), (80, 80))
    assert transform.to_source((10, 30, 20, 40)) == pytest.approx((10, 10, 20, 20))
    mask = np.zeros((40, 80), dtype=np.bool_)
    mask[10:20, 10:20] = True
    assert box_mask_overlap(Detection((10, 10, 20, 20), 0, 1.0), mask) == 1.0


def test_context_risk_contributions_and_suppression() -> None:
    weights = RiskWeights(1, 1, 1, 2, 1, 1, 2)
    transient = contextual_risk(
        dict(
            anomaly_score=0.7,
            component_area=0.1,
            image_position=0.2,
            road_overlap=0.0,
            relative_proximity=0.1,
            detector_overlap=0.0,
            temporal_persistence=0.0,
        ),
        weights,
    )
    persistent = contextual_risk(
        dict(
            anomaly_score=0.7,
            component_area=0.6,
            image_position=1.0,
            road_overlap=1.0,
            relative_proximity=0.9,
            detector_overlap=0.5,
            temporal_persistence=1.0,
        ),
        weights,
    )
    assert sum(persistent["normalized_feature_contributions"].values()) == pytest.approx(
        persistent["total_risk_score"]
    )
    assert persistent["total_risk_score"] > transient["total_risk_score"]


def test_temporal_stable_transient_moving_gap_and_restart() -> None:
    tracker = TemporalPersistence(centroid_distance=4, missed_frame_tolerance=1)
    first = tracker.update("one", 0, (_component(1, 2),))
    moving = tracker.update("one", 1, (_component(1, 4),))
    missing = tracker.update("one", 2, ())
    recovered = tracker.update("one", 3, (_component(1, 5),))
    restarted = tracker.update("two", 0, (_component(1, 20),))
    assert first[0]["event"] == "appeared"
    assert moving[0]["persistence_count"] == 2
    assert missing == ()
    assert recovered[0]["persistence_count"] == 3
    assert restarted[0]["track_id"] == 1


def test_temporal_split_merge_is_deterministic() -> None:
    tracker = TemporalPersistence(centroid_distance=8)
    tracker.update("split", 0, (_component(1, 10),))
    split = tracker.update("split", 1, (_component(1, 9), _component(2, 12)))
    assert [record["event"] for record in split] == ["matched", "appeared"]


def test_temporal_snapshot_restore_continues_exact_track() -> None:
    tracker = TemporalPersistence(missed_frame_tolerance=1)
    component = _component(1, 4)
    tracker.update("sequence", 0, (component,))
    restored = TemporalPersistence(missed_frame_tolerance=1)
    restored.restore(tracker.snapshot())
    record = restored.update("sequence", 1, (component,))[0]
    assert record["track_id"] == 1
    assert record["persistence_count"] == 2


def test_temporal_restore_rejects_corrupt_state() -> None:
    with pytest.raises(ValueError, match="malformed"):
        TemporalPersistence().restore({"sequence_id": "sequence"})
