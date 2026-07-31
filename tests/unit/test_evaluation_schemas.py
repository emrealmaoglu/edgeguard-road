import pytest

from edgeguard.evaluation.schemas import (
    aggregate_seed_metrics,
    detection_evaluation_record,
    efficiency_record,
    temporal_evaluation_record,
)


def test_detection_schema_accepts_unknown_real_metrics_without_invention() -> None:
    record = detection_evaluation_record(
        map_value=None,
        ap50=None,
        ap75=None,
        per_class_ap={"car": None},
        size_buckets={"small": None},
        box_error_counts={"localization": 0},
        scientific_evidence=False,
    )
    assert record["mAP"] is None
    assert record["scientific_evidence"] is False


def test_seed_temporal_and_efficiency_records() -> None:
    aggregate = aggregate_seed_metrics(
        ({"seed": 1.0, "mIoU": 0.2}, {"seed": 2.0, "mIoU": 0.4}), "mIoU"
    )
    assert aggregate["mean"] == pytest.approx(0.3)
    temporal = temporal_evaluation_record(
        persistence_gain=0.1,
        transient_suppression=0.5,
        track_continuity=0.8,
        missed_frame_recovery=1.0,
        scientific_evidence=False,
    )
    assert temporal["scientific_evidence"] is False
    efficiency = efficiency_record(
        throughput=2.0,
        batch_latency_seconds=0.5,
        per_frame_latency_seconds=0.5,
        peak_memory_bytes=100,
        model_size_bytes=200,
        preprocessing_seconds=0.1,
        postprocessing_seconds=0.2,
    )
    assert efficiency["includes_ui"] is False
