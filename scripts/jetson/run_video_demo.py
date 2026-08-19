"""Run semantic segmentation + perception overlay on a video, frame by frame.

Reads an input video, runs each frame through a static-shape semantic engine
(ONNX Runtime for local/CPU development, or a target-device TensorRT engine via
``scripts.jetson.benchmark.TensorRTTorchRunner``), derives the road/drivable
corridor/attention perception layer, and writes an annotated output video plus
a JSON summary. Every output uses the same non-fabrication contract as
``scripts/predict.py``: no instance detection, no physical risk probability,
and TensorRT results are only meaningful when actually run on the target
Jetson device (see ``scripts/jetson/AGENTS.md``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edgeguard.rescue.inference import predict_onnx, predict_tensorrt
from edgeguard.rescue.perception import attention_contract, derive_perception
from edgeguard.rescue.shift import (
    apply_shift_reference,
    frame_uncertainty_summary,
    load_shift_reference,
    uncertainty_maps,
)
from edgeguard.rescue.visualization import (
    InferenceResult,
    overlay_mask,
    render_regions_overlay,
    resize_scalar,
)
from edgeguard.serialization import canonical_json, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--backend", choices=("onnx", "tensorrt"), default="onnx")
    parser.add_argument("--onnx-model", type=Path, help="required when --backend=onnx")
    parser.add_argument("--engine", type=Path, help="required when --backend=tensorrt")
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=1024)
    parser.add_argument("--opacity", type=float, default=0.55)
    parser.add_argument("--emit-regions", action="store_true")
    parser.add_argument("--shift-reference", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--entropy-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-region-area", type=int, default=8)
    parser.add_argument("--minimum-drivable-area", type=int, default=64)
    parser.add_argument(
        "--max-frames", type=int, help="stop after this many frames (dry-run/testing)"
    )
    return parser


def _open_capture(path: Path) -> Any:
    cv2 = __import__("cv2")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open input video: {path}")
    return capture


def _open_writer(path: Path, *, fps: float, size: tuple[int, int]) -> Any:
    cv2 = __import__("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"could not open output video for writing: {path}")
    return writer


def render_frame(
    frame_rgb: np.ndarray,
    result: InferenceResult,
    *,
    opacity: float,
    emit_regions: bool,
    confidence_threshold: float,
    entropy_threshold: float,
    minimum_region_area: int,
    minimum_drivable_area: int,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Blend the semantic mask (and optional region boxes) onto one RGB frame."""
    image = Image.fromarray(frame_rgb, mode="RGB")
    blended = overlay_mask(image, result.mask, opacity)
    perception_summary: dict[str, Any] | None = None
    if emit_regions:
        perception = derive_perception(
            result.mask,
            result.confidence,
            result.entropy,
            confidence_threshold=confidence_threshold,
            entropy_threshold=entropy_threshold,
            minimum_region_area=minimum_region_area,
            minimum_drivable_area=minimum_drivable_area,
        )
        blended = render_regions_overlay(blended, perception)
        perception_summary = {
            "region_count": len(perception.regions),
            "drivable_pixel_ratio": float(np.mean(perception.drivable_corridor)),
            "unreliable_pixel_ratio": float(np.mean(perception.unreliable_mask)),
        }
    return np.asarray(blended, dtype=np.uint8), perception_summary


def _predict_frame(
    frame_rgb: np.ndarray,
    *,
    backend: str,
    onnx_session: Any,
    onnx_model_path: Path | None,
    tensorrt_runner: Any,
    input_size: tuple[int, int],
) -> InferenceResult:
    image = Image.fromarray(frame_rgb, mode="RGB")
    if backend == "onnx":
        assert onnx_model_path is not None
        return predict_onnx(image, onnx_model_path, input_size=input_size, session=onnx_session)
    return predict_tensorrt(image, tensorrt_runner, input_size=input_size)


def _validate_backend_args(args: argparse.Namespace) -> None:
    if args.backend == "onnx" and args.onnx_model is None:
        raise ValueError("--backend=onnx requires --onnx-model")
    if args.backend == "tensorrt" and args.engine is None:
        raise ValueError("--backend=tensorrt requires --engine")


def main() -> int:
    """Run the video demo end to end and print the aggregate summary."""
    args = _parser().parse_args()
    _validate_backend_args(args)

    onnx_session = None
    tensorrt_runner = None
    checkpoint_identity: str | None = None
    if args.backend == "onnx":
        ort = __import__("onnxruntime")
        onnx_session = ort.InferenceSession(
            str(args.onnx_model.resolve()), providers=["CPUExecutionProvider"]
        )
        checkpoint_identity = sha256_file(args.onnx_model)
    else:
        from scripts.jetson.benchmark import TensorRTTorchRunner

        tensorrt_runner = TensorRTTorchRunner(args.engine.resolve())
        checkpoint_identity = sha256_file(args.engine)

    shift_reference = (
        load_shift_reference(args.shift_reference.resolve())
        if args.shift_reference is not None
        else None
    )

    cv2 = __import__("cv2")
    capture = _open_capture(args.input_video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _open_writer(args.output_video, fps=fps, size=(frame_width, frame_height))

    input_size = (args.input_height, args.input_width)
    frame_count = 0
    inference_latency_ms: list[float] = []
    domain_shift_alerts = 0
    confidences: list[float] = []
    entropies: list[float] = []
    started = time.perf_counter()
    try:
        while args.max_frames is None or frame_count < args.max_frames:
            read_ok, frame_bgr = capture.read()
            if not read_ok:
                break
            frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            result = _predict_frame(
                frame_rgb,
                backend=args.backend,
                onnx_session=onnx_session,
                onnx_model_path=args.onnx_model,
                tensorrt_runner=tensorrt_runner,
                input_size=input_size,
            )
            inference_latency_ms.append(result.latency_ms)
            confidences.append(float(np.mean(result.confidence)))
            entropies.append(float(np.mean(result.entropy)))
            if shift_reference is not None:
                energy = None
                if result.logits is not None:
                    energy = uncertainty_maps(result.logits)["energy"]
                    if energy.shape != result.mask.shape:
                        energy = resize_scalar(energy, (frame_width, frame_height))
                summary = frame_uncertainty_summary(
                    result.confidence,
                    result.entropy,
                    energy=energy,
                    confidence_threshold=args.confidence_threshold,
                )
                if apply_shift_reference(summary, shift_reference)["domain_shift_alert"]:
                    domain_shift_alerts += 1
            rendered, _ = render_frame(
                frame_rgb,
                result,
                opacity=args.opacity,
                emit_regions=args.emit_regions,
                confidence_threshold=args.confidence_threshold,
                entropy_threshold=args.entropy_threshold,
                minimum_region_area=args.minimum_region_area,
                minimum_drivable_area=args.minimum_drivable_area,
            )
            writer.write(np.ascontiguousarray(rendered[:, :, ::-1]))
            frame_count += 1
    finally:
        capture.release()
        writer.release()

    if frame_count == 0:
        raise ValueError("input video produced zero readable frames")
    elapsed_seconds = time.perf_counter() - started
    summary = {
        "schema_version": "2.0",
        "record_type": "edgeguard_video_demo_summary",
        "input_video": str(args.input_video),
        "output_video": str(args.output_video),
        "backend": args.backend,
        "checkpoint_sha256": checkpoint_identity,
        "frame_count": frame_count,
        "wall_clock_seconds": elapsed_seconds,
        "achieved_fps": frame_count / elapsed_seconds if elapsed_seconds > 0 else None,
        "mean_inference_latency_ms": float(np.mean(inference_latency_ms)),
        "p95_inference_latency_ms": float(np.percentile(inference_latency_ms, 95)),
        "mean_confidence": float(np.mean(confidences)),
        "mean_normalized_entropy": float(np.mean(entropies)),
        "domain_shift_alert_frames": domain_shift_alerts if shift_reference is not None else None,
        "domain_shift_alert_ratio": (
            domain_shift_alerts / frame_count if shift_reference is not None else None
        ),
        "attention_contract": attention_contract() if args.emit_regions else None,
        "scientific_benchmark": False,
        "instance_detection": False,
        "physical_risk_probability": False,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
