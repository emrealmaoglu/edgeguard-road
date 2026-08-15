"""Synthetic tests for OOD, components, detection, risk, and temporal behavior."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from edgeguard.context import RiskWeights, contextual_risk
from edgeguard.detection.bdd import adapt_bdd_record
from edgeguard.detection.contracts import Detection, LetterboxTransform, box_mask_overlap
from edgeguard.evaluation.components import ComponentRecord, connected_components
from edgeguard.evaluation.components import label_components as _run_label_components
from edgeguard.evaluation.ood import pixel_ood_metrics, select_anomaly_threshold
from edgeguard.rescue import perception as rescue_perception
from edgeguard.rescue.perception import (
    REGION_CLASS_IDS,
    _distance_from_mask,
    _label_components,
)
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


def _bfs_label_components(binary: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """The original per-pixel BFS, retained only as a correctness oracle.

    It cost 96 ms of a 146 ms frame on a real Orin Nano Super -- 19x the TensorRT engine
    it wraps -- which is why the shipped implementation is vectorised. Keeping the naive
    version here pins the fast path to the labelling it replaced.
    """
    from collections import deque

    labels = np.zeros(binary.shape, dtype=np.int32)
    components: list[np.ndarray] = []
    height, width = binary.shape
    next_label = 0
    for start_y, start_x in zip(*np.nonzero(binary), strict=True):
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


def _bfs_distance_from_mask(mask: np.ndarray) -> np.ndarray:
    """The original per-pixel BFS distance, retained as a correctness oracle."""
    from collections import deque

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


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7, 20260728])
@pytest.mark.parametrize("density", [0.15, 0.5, 0.85])
def test_vectorised_labelling_matches_the_breadth_first_search(seed: int, density: float) -> None:
    """Component order and label values are contract: callers map a component's list index
    onto `labels == index + 1` to select the drivable corridor, and derive `region_id` from
    list position. Pixel order inside a component is not, since every consumer takes an
    area, extent or mean -- so pixel sets are compared as sets.
    """
    generator = np.random.default_rng(seed)
    binary = generator.random((24, 37)) < density

    expected_labels, expected_components = _bfs_label_components(binary)
    actual_labels, actual_components = _label_components(binary)

    assert np.array_equal(actual_labels, expected_labels)
    assert len(actual_components) == len(expected_components)
    for actual, expected in zip(actual_components, expected_components, strict=True):
        assert sorted(map(tuple, actual.tolist())) == sorted(map(tuple, expected.tolist()))


def test_labelling_handles_an_entirely_empty_mask() -> None:
    labels, components = _label_components(np.zeros((5, 6), dtype=np.bool_))
    assert components == []
    assert not labels.any()


def test_snake_shaped_component_still_converges_to_one_label() -> None:
    """Label propagation needs one pass per unit of component diameter, so the worst case
    is a single winding component rather than the compact blobs real masks produce.
    """
    binary = np.zeros((9, 9), dtype=np.bool_)
    binary[::2, :] = True
    binary[1::4, -1] = True
    binary[3::4, 0] = True

    labels, components = _label_components(binary)
    assert len(components) == 1
    assert set(np.unique(labels[binary]).tolist()) == {1}


@pytest.mark.parametrize("seed", [0, 5, 20260728])
def test_separable_distance_transform_matches_the_breadth_first_search(seed: int) -> None:
    """BFS on an obstacle-free four-connected grid is exactly the L1 distance transform."""
    generator = np.random.default_rng(seed)
    mask = generator.random((19, 23)) < 0.08
    mask[0, 0] = True  # guarantee a source so both paths return finite distances

    assert np.array_equal(_distance_from_mask(mask), _bfs_distance_from_mask(mask))


def test_distance_from_an_empty_mask_stays_infinite() -> None:
    distance = _distance_from_mask(np.zeros((4, 4), dtype=np.bool_))
    assert np.isinf(distance).all()


@pytest.mark.parametrize("seed", [0, 3, 11, 20260728])
def test_single_pass_class_labelling_matches_the_per_class_calls(seed: int) -> None:
    """`derive_perception` used to label each attention class with its own full-array
    pass -- eleven per frame on the stage that dominates the Jetson budget. Components of
    different classes can never merge, so one propagation restricted to same-class
    neighbours resolves them all; it has to return exactly what the separate calls did,
    including the order components appear in, since `region_id` comes from list position.
    """
    from edgeguard.rescue.perception import REGION_CLASS_IDS, _label_components_by_class

    generator = np.random.default_rng(seed)
    mask = generator.integers(0, 19, (26, 41)).astype(np.uint8)

    grouped = _label_components_by_class(mask, REGION_CLASS_IDS)
    for class_id in REGION_CLASS_IDS:
        _, expected = _label_components(mask == class_id)
        actual = grouped[class_id]
        assert len(actual) == len(expected), class_id
        for one, other in zip(actual, expected, strict=True):
            assert sorted(map(tuple, one.tolist())) == sorted(map(tuple, other.tolist()))


def test_single_pass_labelling_handles_a_mask_with_no_attention_classes() -> None:
    from edgeguard.rescue.perception import REGION_CLASS_IDS, _label_components_by_class

    mask = np.zeros((6, 6), dtype=np.uint8)  # road only
    grouped = _label_components_by_class(mask, REGION_CLASS_IDS)
    assert set(grouped) == set(REGION_CLASS_IDS)
    assert all(not components for components in grouped.values())


def test_adjacent_different_classes_never_merge_into_one_component() -> None:
    """The whole optimisation rests on this: propagation is restricted to same-class
    neighbours, so two touching regions of different classes stay separate.
    """
    from edgeguard.rescue.perception import _label_components_by_class

    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[:, :3] = 11  # person
    mask[:, 3:] = 13  # car

    grouped = _label_components_by_class(mask, (11, 13))
    assert len(grouped[11]) == 1
    assert len(grouped[13]) == 1
    assert grouped[11][0].shape[0] == 12
    assert grouped[13][0].shape[0] == 12


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7, 20260728])
@pytest.mark.parametrize("density", [0.05, 0.15, 0.5, 0.85, 1.0])
def test_labellers_agree(seed: int, density: float) -> None:
    """The evaluation module labels components by unioning horizontal runs instead of
    propagating labels, because propagation costs one pass per unit of component width and
    evaluation scores full-resolution masks where a road spans the image. Two algorithms
    for one definition is a licence to drift, so pin them to each other: same labels, same
    component order, same pixel sets.
    """
    generator = np.random.default_rng(seed)
    binary = generator.random((31, 43)) < density

    expected_labels, expected_components = _label_components(binary)
    actual_labels, actual_components = _run_label_components(binary)

    assert np.array_equal(actual_labels, expected_labels)
    assert len(actual_components) == len(expected_components)
    for actual, expected in zip(actual_components, expected_components, strict=True):
        assert sorted(map(tuple, actual.tolist())) == sorted(map(tuple, expected.tolist()))


def test_run_labelling_handles_an_entirely_empty_mask() -> None:
    labels, components = _run_label_components(np.zeros((5, 6), dtype=np.bool_))
    assert components == []
    assert not labels.any()


def test_run_labelling_does_not_slow_down_with_component_width() -> None:
    """The property that made this algorithm necessary: a component as wide as the image
    is not more expensive than a narrow one. Propagation needs a pass per unit of width, so
    on the 2048-wide road mask this replaces it took 76 s.
    """
    wide = np.zeros((256, 2048), dtype=np.bool_)
    wide[64:192, :] = True
    started = time.perf_counter()
    labels, components = _run_label_components(wide)
    assert time.perf_counter() - started < 5.0
    assert len(components) == 1
    assert int(labels.max()) == 1


def _naive_connected_components(
    anomaly_mask: np.ndarray, scores: np.ndarray, road_mask: np.ndarray
) -> tuple[ComponentRecord, ...]:
    """The per-pixel search `connected_components` replaced, kept to compare against."""
    visited = np.zeros_like(anomaly_mask)
    height, width = anomaly_mask.shape
    records: list[ComponentRecord] = []
    for y in range(height):
        for x in range(width):
            if not anomaly_mask[y, x] or visited[y, x]:
                continue
            queue = [(y, x)]
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                current_y, current_x = queue.pop()
                pixels.append((current_y, current_x))
                for next_y, next_x in (
                    (current_y - 1, current_x),
                    (current_y + 1, current_x),
                    (current_y, current_x - 1),
                    (current_y, current_x + 1),
                ):
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and anomaly_mask[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        queue.append((next_y, next_x))
            ys = np.asarray([item[0] for item in pixels], dtype=np.int64)
            xs = np.asarray([item[1] for item in pixels], dtype=np.int64)
            values = scores[ys, xs]
            records.append(
                ComponentRecord(
                    component_id=len(records) + 1,
                    area=len(pixels),
                    bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
                    centroid_xy=(float(xs.mean()), float(ys.mean())),
                    mean_score=float(values.mean()),
                    max_score=float(values.max()),
                    road_overlap=float(np.count_nonzero(road_mask[ys, xs]) / len(pixels)),
                )
            )
    return tuple(records)


@pytest.mark.parametrize("seed", [0, 3, 11, 20260728])
@pytest.mark.parametrize("density", [0.08, 0.3, 0.75])
def test_component_records_match_the_search_they_replaced(seed: int, density: float) -> None:
    """`component_id` is a contract -- temporal tracks and region IDs are derived from it --
    so the run-based rewrite has to reproduce the search's numbering and every statistic,
    not merely find the same partition.
    """
    generator = np.random.default_rng(seed)
    anomaly = generator.random((29, 41)) < density
    scores = generator.random((29, 41))
    road = generator.random((29, 41)) < 0.4

    actual = connected_components(anomaly, scores, road_mask=road)
    expected = _naive_connected_components(anomaly, scores, road)

    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert left.component_id == right.component_id
        assert left.area == right.area
        assert left.bbox_xyxy == right.bbox_xyxy
        # Summation order inside a component differs, so the means differ in the last bit.
        # Pixel order is explicitly not part of the contract; the partition and the
        # numbering are.
        assert left.centroid_xy == pytest.approx(right.centroid_xy, rel=1e-12)
        assert left.mean_score == pytest.approx(right.mean_score, rel=1e-12)
        assert left.max_score == right.max_score
        assert left.road_overlap == pytest.approx(right.road_overlap, rel=1e-12)


def test_the_size_threshold_does_not_change_what_is_labelled(monkeypatch) -> None:
    """`_label_components` picks its algorithm by mask size, so the branch has to be
    invisible: the same mask must label identically on either side of the limit.
    """
    generator = np.random.default_rng(4)
    binary = generator.random((37, 53)) < 0.35

    monkeypatch.setattr(rescue_perception, "_PROPAGATION_PIXEL_LIMIT", binary.size + 1)
    propagated_labels, propagated = rescue_perception._label_components(binary)
    monkeypatch.setattr(rescue_perception, "_PROPAGATION_PIXEL_LIMIT", 0)
    run_based_labels, run_based = rescue_perception._label_components(binary)

    assert np.array_equal(propagated_labels, run_based_labels)
    assert len(propagated) == len(run_based)
    for left, right in zip(propagated, run_based, strict=True):
        assert sorted(map(tuple, left.tolist())) == sorted(map(tuple, right.tolist()))


def test_the_size_threshold_does_not_change_multi_class_labelling(monkeypatch) -> None:
    generator = np.random.default_rng(9)
    semantic = generator.integers(0, 19, size=(31, 47))

    monkeypatch.setattr(rescue_perception, "_PROPAGATION_PIXEL_LIMIT", semantic.size + 1)
    propagated = rescue_perception._label_components_by_class(semantic, REGION_CLASS_IDS)
    monkeypatch.setattr(rescue_perception, "_PROPAGATION_PIXEL_LIMIT", 0)
    run_based = rescue_perception._label_components_by_class(semantic, REGION_CLASS_IDS)

    assert set(propagated) == set(run_based)
    for class_id in propagated:
        assert len(propagated[class_id]) == len(run_based[class_id])
        for left, right in zip(propagated[class_id], run_based[class_id], strict=True):
            assert sorted(map(tuple, left.tolist())) == sorted(map(tuple, right.tolist()))
