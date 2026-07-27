from __future__ import annotations

import hashlib
import http.server
import threading
from pathlib import Path

import numpy as np
import pytest

from edgeguard.data.acquisition import (
    copy_with_progress,
    dataset_artifact_readiness,
    download_verified,
    generate_fixture_bundle,
)
from edgeguard.data.contracts import DetectionSample, OODSample, SemanticSample, TemporalFrame
from edgeguard.data.quality import (
    audit_detection_samples,
    audit_ood_samples,
    audit_semantic_samples,
    audit_temporal_frames,
    bounded_perceptual_duplicate_pairs,
)
from edgeguard.data.transforms import (
    horizontal_flip_boxes,
    invert_letterbox_boxes,
    letterbox_detection,
    rare_class_sampling_weights,
    resize_mask,
    semantic_training_transform,
    synthetic_outlier_exposure,
    temporal_sample_indices,
)


def _semantic(sample_id: str = "sample-1", *, role: str = "train_fit") -> SemanticSample:
    image = np.arange(8 * 16 * 3, dtype=np.uint8).reshape(8, 16, 3)
    mask = np.tile(np.arange(16, dtype=np.uint8), (8, 1)) % 19
    mask[0, 0] = 255
    return SemanticSample(
        image_id=sample_id,
        image=image,
        train_ids=mask,
        ignore_mask=mask == 255,
        source="synthetic",
        group_id="city-sequence-1",
        split_role=role,
        city="city",
        sequence_id="sequence-1",
    )


def _detection() -> DetectionSample:
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    boxes = np.asarray([[4, 5, 20, 15]], dtype=np.float32)
    return DetectionSample(
        image_id="det-1",
        image=image,
        boxes_xyxy=boxes.copy(),
        source_boxes_xyxy=boxes.copy(),
        class_ids=np.asarray([2], dtype=np.int64),
        class_names=("car",),
        crowd_or_ignore=np.asarray([False]),
        source="synthetic_bdd",
        original_size_hw=(20, 40),
        model_input_size_hw=(32, 32),
        transform_receipt={},
    )


def test_perception_contracts_and_quality_reports() -> None:
    semantic = _semantic()
    semantic.validate()
    detection = _detection()
    detection.validate()
    ood = OODSample(
        "ood-1",
        np.asarray([[0, 1], [255, 0]], dtype=np.uint8),
        np.asarray([[True, True], [False, True]]),
        "synthetic",
        "development",
    )
    ood.validate()
    ood_report = audit_ood_samples((ood,))
    assert ood_report["anomaly_pixel_count"] == 1
    assert np.asarray(ood_report["anomaly_location_heatmap_4x4"]).shape == (4, 4)
    semantic_report = audit_semantic_samples((semantic,))
    assert semantic_report["sample_count"] == 1
    assert len(semantic_report["class_pixel_counts"]) == 19
    assert np.asarray(semantic_report["spatial_class_heatmap_4x4"]).shape == (19, 4, 4)
    detection_report = audit_detection_samples((detection,))
    assert detection_report["object_counts"] == [1]


def test_quality_detects_duplicate_and_group_leakage() -> None:
    duplicate = _semantic("sample-2", role="train_select")
    report = audit_semantic_samples((_semantic(), duplicate))
    codes = {issue["code"] for issue in report["issues"]}
    assert {"exact_duplicate_image", "group_leakage"} <= codes
    pairs = bounded_perceptual_duplicate_pairs(
        {"first": _semantic().image, "second": _semantic("two").image}
    )
    assert pairs == [{"left": "first", "right": "second", "hamming_distance": 0}]


def test_temporal_quality_detects_gap_and_timestamp_errors() -> None:
    frames = (
        TemporalFrame("f0", "s", 0, 1.0, 0),
        TemporalFrame("f2", "s", 2, 0.5, 0),
    )
    report = audit_temporal_frames(frames)
    codes = {issue["code"] for issue in report["issues"]}
    assert {"gap_metadata_mismatch", "non_monotonic_timestamp"} <= codes
    selected, gaps = temporal_sample_indices((0, 1, 2, 5, 6), stride=2, maximum_gap=2)
    assert selected == (0, 2, 6)
    assert gaps[-1]["exceeds_maximum"] == 1


def test_semantic_transform_preserves_alignment_ignore_and_determinism() -> None:
    sample = _semantic()
    first = semantic_training_transform(
        sample.image,
        sample.train_ids,
        seed=4,
        crop_size_hw=(12, 20),
        scale_range=(1.0, 1.0),
        horizontal_flip_probability=1.0,
        photometric_strength=0.0,
        class_aware_ids=(15,),
    )
    second = semantic_training_transform(
        sample.image,
        sample.train_ids,
        seed=4,
        crop_size_hw=(12, 20),
        scale_range=(1.0, 1.0),
        horizontal_flip_probability=1.0,
        photometric_strength=0.0,
        class_aware_ids=(15,),
    )
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert set(int(value) for value in np.unique(first[1])) <= {*range(19), 255}
    assert resize_mask(sample.train_ids, (16, 32)).dtype == np.uint8
    weights = rare_class_sampling_weights(
        np.asarray([[True, False], [True, True], [True, False]], dtype=np.bool_)
    )
    assert weights.shape == (3,)
    assert weights[1] > weights[0]


def test_detection_transform_round_trip_and_flip() -> None:
    sample = _detection()
    _, transformed, receipt = letterbox_detection(sample.image, sample.boxes_xyxy, (32, 32))
    restored = invert_letterbox_boxes(transformed, receipt)
    assert np.allclose(restored, sample.boxes_xyxy)
    _, flipped = horizontal_flip_boxes(sample.image, sample.boxes_xyxy)
    _, round_trip = horizontal_flip_boxes(sample.image, flipped)
    assert np.array_equal(round_trip, sample.boxes_xyxy)


def test_road_aware_outlier_stays_inside_valid_region() -> None:
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    road = np.zeros((32, 64), dtype=np.bool_)
    road[12:30, 5:60] = True
    output, target, receipt = synthetic_outlier_exposure(
        image, road, seed=8, scale_fraction=(0.02, 0.02)
    )
    assert output.shape == image.shape
    assert np.all(road[target == 1])
    assert receipt["bbox_xyxy"]


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""
    fail_first = False
    request_count = 0

    def do_GET(self) -> None:  # noqa: N802
        type(self).request_count += 1
        if self.path.startswith("/forbidden"):
            self.send_response(403)
            self.end_headers()
            return
        if type(self).fail_first and type(self).request_count == 1:
            self.send_response(503)
            self.end_headers()
            return
        start = 0
        range_header = self.headers.get("Range")
        if range_header:
            start = int(range_header.removeprefix("bytes=").removesuffix("-"))
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}"
            )
        else:
            self.send_response(200)
        body = self.payload[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def _serve(payload: bytes) -> tuple[http.server.ThreadingHTTPServer, str]:
    _RangeHandler.payload = payload
    _RangeHandler.request_count = 0
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}/archive?token=secret"


def test_download_resume_retry_hash_and_url_redaction(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 300
    server, url = _serve(payload)
    try:
        destination = tmp_path / "archive.bin"
        destination.with_name("archive.bin.part").write_bytes(payload[:1000])
        progress: list[dict[str, object]] = []
        result = download_verified(
            url,
            destination,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size=len(payload),
            progress=progress.append,
        )
        assert destination.read_bytes() == payload
        assert result["source"].endswith("/archive")
        assert "token" not in str(result)
        assert progress
    finally:
        server.shutdown()


def test_download_rejects_hash_and_expired_url(tmp_path: Path) -> None:
    payload = b"payload"
    server, url = _serve(payload)
    try:
        with pytest.raises(RuntimeError, match="SHA-256"):
            download_verified(
                url,
                tmp_path / "bad.bin",
                expected_sha256="0" * 64,
                expected_size=len(payload),
                retries=1,
            )
        with pytest.raises(RuntimeError, match="403"):
            download_verified(
                url.replace("/archive", "/forbidden"),
                tmp_path / "forbidden.bin",
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_size=len(payload),
                retries=1,
            )
        with pytest.raises(RuntimeError, match="byte size mismatch"):
            download_verified(
                url,
                tmp_path / "wrong-size.bin",
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_size=len(payload) + 1,
                retries=1,
            )
    finally:
        server.shutdown()


def test_download_retries_after_transient_failure(tmp_path: Path) -> None:
    payload = b"retry-payload"
    server, url = _serve(payload)
    _RangeHandler.fail_first = True
    try:
        result = download_verified(
            url,
            tmp_path / "retried.bin",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size=len(payload),
            retries=2,
            backoff_seconds=0,
        )
        assert result["attempts"] == 2
    finally:
        _RangeHandler.fail_first = False
        server.shutdown()


def test_generator_interruption_resume_readiness_and_slow_copy(tmp_path: Path) -> None:
    root = tmp_path / "generator"
    config = {"seed": 7, "file_count": 3, "generator_version": "fixture-v1"}
    with pytest.raises(InterruptedError):
        generate_fixture_bundle(root, generator_config=config, interrupt_after_files=1)
    result = generate_fixture_bundle(root, generator_config=config)
    assert len(result["manifest"]["files"]) == 3
    readiness = dataset_artifact_readiness(
        [
            {"artifact_id": "images", "verified": True},
            {"artifact_id": "labels", "verified": False},
        ]
    )
    assert readiness == {
        "artifact_count": 2,
        "missing_or_unverified": ["labels"],
        "ready": False,
    }
    progress: list[dict[str, int]] = []
    copy_result = copy_with_progress(
        root / result["archive_filename"],
        tmp_path / "slow-destination.zip",
        chunk_size=8,
        progress=progress.append,
    )
    assert copy_result["sha256"] == result["archive_sha256"]
    assert progress
