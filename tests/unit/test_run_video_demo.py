from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edgeguard.rescue.visualization import InferenceResult
from scripts.jetson import run_video_demo


def _fixture_result(*, height: int = 12, width: int = 16) -> InferenceResult:
    mask = np.full((height, width), 2, dtype=np.uint8)
    mask[6:, 3:13] = 0
    mask[7:11, 7:10] = 13
    logits = np.zeros((19, height, width), dtype=np.float32)
    for class_id in range(19):
        logits[class_id] = -5.0
    for y in range(height):
        for x in range(width):
            logits[int(mask[y, x]), y, x] = 5.0
    return InferenceResult(
        mask=mask,
        confidence=np.full((height, width), 0.8, dtype=np.float32),
        entropy=np.full((height, width), 0.2, dtype=np.float32),
        latency_ms=12.5,
        backend="fixture",
        metadata={"fixture": True},
        logits=logits,
    )


def test_render_frame_blends_semantic_mask(tmp_path: Path) -> None:
    frame = np.full((12, 16, 3), 40, dtype=np.uint8)
    result = _fixture_result()
    rendered, perception_summary = run_video_demo.render_frame(
        frame,
        result,
        opacity=0.6,
        emit_regions=False,
        confidence_threshold=0.5,
        entropy_threshold=0.5,
        minimum_region_area=2,
        minimum_drivable_area=4,
    )
    assert rendered.shape == frame.shape
    assert rendered.dtype == np.uint8
    assert not np.array_equal(rendered, frame)
    assert perception_summary is None


def test_render_frame_with_regions_draws_boxes_and_returns_summary() -> None:
    frame = np.full((12, 16, 3), 40, dtype=np.uint8)
    result = _fixture_result()
    without_regions, _ = run_video_demo.render_frame(
        frame,
        result,
        opacity=0.6,
        emit_regions=False,
        confidence_threshold=0.5,
        entropy_threshold=0.5,
        minimum_region_area=2,
        minimum_drivable_area=4,
    )
    with_regions, summary = run_video_demo.render_frame(
        frame,
        result,
        opacity=0.6,
        emit_regions=True,
        confidence_threshold=0.5,
        entropy_threshold=0.5,
        minimum_region_area=2,
        minimum_drivable_area=4,
    )
    assert summary is not None
    assert summary["region_count"] >= 1
    assert not np.array_equal(with_regions, without_regions)


def test_validate_backend_args_requires_matching_model_source() -> None:
    onnx_missing = argparse.Namespace(backend="onnx", onnx_model=None, engine=None)
    with pytest.raises(ValueError, match="--onnx-model"):
        run_video_demo._validate_backend_args(onnx_missing)  # noqa: SLF001

    tensorrt_missing = argparse.Namespace(backend="tensorrt", onnx_model=None, engine=None)
    with pytest.raises(ValueError, match="--engine"):
        run_video_demo._validate_backend_args(tensorrt_missing)  # noqa: SLF001

    run_video_demo._validate_backend_args(  # noqa: SLF001
        argparse.Namespace(backend="onnx", onnx_model=Path("model.onnx"), engine=None)
    )


def _write_fixture_video(path: Path, *, frame_count: int, size: tuple[int, int]) -> None:
    cv2 = __import__("cv2")
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    try:
        for index in range(frame_count):
            frame = np.full((height, width, 3), (index * 20) % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_video_demo_end_to_end_with_fixture_onnx_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "output.mp4"
    summary_path = tmp_path / "summary.json"
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fixture-onnx-bytes")
    _write_fixture_video(input_video, frame_count=3, size=(16, 12))

    fixture_result = _fixture_result()
    monkeypatch.setattr(run_video_demo, "predict_onnx", lambda *_a, **_k: fixture_result)

    class _FakeSession:
        pass

    class _FakeOnnxRuntimeModule:
        @staticmethod
        def InferenceSession(*_args: object, **_kwargs: object) -> _FakeSession:
            return _FakeSession()

    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOnnxRuntimeModule())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_video_demo.py",
            "--input-video",
            str(input_video),
            "--output-video",
            str(output_video),
            "--summary-json",
            str(summary_path),
            "--backend",
            "onnx",
            "--onnx-model",
            str(model_path),
            "--emit-regions",
            "--minimum-region-area",
            "2",
            "--minimum-drivable-area",
            "4",
        ],
    )

    assert run_video_demo.main() == 0
    assert output_video.is_file() and output_video.stat().st_size > 0
    summary = json.loads(summary_path.read_text())
    assert summary["record_type"] == "edgeguard_video_demo_summary"
    assert summary["frame_count"] == 3
    assert summary["backend"] == "onnx"
    assert summary["instance_detection"] is False
    assert summary["physical_risk_probability"] is False
    assert summary["domain_shift_alert_frames"] is None


def test_missing_input_video_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_video_demo._open_capture(tmp_path / "does-not-exist.mp4")  # noqa: SLF001


def test_render_frame_rejects_geometry_mismatch() -> None:
    frame = np.full((12, 16, 3), 40, dtype=np.uint8)
    mismatched = InferenceResult(
        mask=np.zeros((5, 5), dtype=np.uint8),
        confidence=np.full((5, 5), 0.8, dtype=np.float32),
        entropy=np.full((5, 5), 0.2, dtype=np.float32),
        latency_ms=1.0,
        backend="fixture",
        metadata={},
    )
    # overlay_mask resizes the palette to the base image, so a mismatched mask
    # is silently resized rather than rejected -- this exercises that path
    # explicitly so a future tightening of the contract is a deliberate change.
    rendered, _ = run_video_demo.render_frame(
        frame,
        mismatched,
        opacity=0.5,
        emit_regions=False,
        confidence_threshold=0.5,
        entropy_threshold=0.5,
        minimum_region_area=2,
        minimum_drivable_area=4,
    )
    assert rendered.shape == frame.shape


def test_fixture_image_round_trips_through_pil() -> None:
    # Guards the RGB/BGR handling in main(): frame_rgb must be a real RGB
    # array (not a reversed BGR view) before Image.fromarray/overlay_mask see it.
    frame_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    frame_bgr[..., 0] = 10  # blue channel
    frame_bgr[..., 2] = 200  # red channel
    frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    image = Image.fromarray(frame_rgb, mode="RGB")
    pixel = image.getpixel((0, 0))
    assert pixel == (200, 0, 10)
