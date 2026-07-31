"""PyTorch/MMSeg and ONNX Runtime inference behind one result contract."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edgeguard.rescue.mmseg_runtime import install_mmcv_lite_guard
from edgeguard.rescue.visualization import (
    InferenceResult,
    confidence_entropy,
    resize_mask,
    resize_scalar,
)

IMAGENET_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def discover_demo_models(run_root: Path) -> list[dict[str, str]]:
    """Discover only complete ONNX or checkpoint/config pairs for the demo."""
    if not run_root.is_dir():
        return []
    records: list[dict[str, str]] = []
    for path in sorted(run_root.glob("**/*.onnx")):
        records.append({"label": f"ONNX · {path.stem}", "backend": "onnx", "model": str(path)})
    for checkpoint in sorted(run_root.glob("**/*.pth")):
        resolved = checkpoint.parent / "resolved.py"
        if resolved.is_file():
            summary_path = checkpoint.parent / "summary.json"
            datasets = "unknown"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                values = summary.get("datasets")
                if isinstance(values, list):
                    datasets = "+".join(str(value) for value in values)
            records.append(
                {
                    "label": (
                        f"PyTorch · {checkpoint.parent.name} · {checkpoint.stem} · {datasets}"
                    ),
                    "backend": "mmseg",
                    "model": str(checkpoint),
                    "config": str(resolved),
                    "datasets": datasets,
                }
            )
    return records


def preprocess_image(image: Image.Image, input_size: tuple[int, int]) -> np.ndarray:
    """Aspect-preserving letterbox followed by shared ImageNet normalization."""
    tensor, _ = _letterbox_image(image, input_size)
    return tensor


def _letterbox_image(
    image: Image.Image, input_size: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return normalized NCHW input and left/top/right/bottom content bounds."""
    height, width = input_size
    rgb = image.convert("RGB")
    scale = min(width / rgb.width, height / rgb.height)
    resized_size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
    resized = rgb.resize(resized_size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    left = (width - resized.width) // 2
    top = (height - resized.height) // 2
    canvas.paste(resized, (left, top))
    array = np.asarray(canvas, dtype=np.float32)
    normalized = (array - IMAGENET_MEAN) / IMAGENET_STD
    tensor = np.transpose(normalized, (2, 0, 1))[None]
    return tensor, (left, top, left + resized.width, top + resized.height)


def predict_onnx(
    image: Image.Image,
    model_path: Path,
    *,
    input_size: tuple[int, int] = (512, 1024),
    session: Any | None = None,
) -> InferenceResult:
    """Run a validated static-shape semantic ONNX graph on CPU."""
    try:
        ort = __import__("onnxruntime")
    except ModuleNotFoundError as error:
        raise RuntimeError("install the rescue/Colab dependencies for ONNX inference") from error
    active_session = session or ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    tensor, bounds = _letterbox_image(image, input_size)
    input_name = active_session.get_inputs()[0].name
    output_name = active_session.get_outputs()[0].name
    started = time.perf_counter()
    logits = active_session.run([output_name], {input_name: tensor})[0]
    latency_ms = (time.perf_counter() - started) * 1000.0
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise RuntimeError(f"unexpected ONNX logits shape: {logits.shape}")
    raw = np.asarray(logits[0], dtype=np.float32)
    left, top, right, bottom = bounds
    scale_y = raw.shape[1] / input_size[0]
    scale_x = raw.shape[2] / input_size[1]
    raw = raw[
        :,
        round(top * scale_y) : max(round(top * scale_y) + 1, round(bottom * scale_y)),
        round(left * scale_x) : max(round(left * scale_x) + 1, round(right * scale_x)),
    ]
    confidence, entropy = confidence_entropy(raw)
    original_size = image.size
    return InferenceResult(
        mask=resize_mask(np.argmax(raw, axis=0), original_size),
        confidence=resize_scalar(confidence, original_size),
        entropy=resize_scalar(entropy, original_size),
        latency_ms=latency_ms,
        backend="onnxruntime_cpu",
        metadata={"model": model_path.name, "raw_logits_shape": list(raw.shape)},
        logits=raw,
    )


def predict_mmseg(
    image: Image.Image,
    resolved_config: Path,
    checkpoint: Path,
    *,
    device: str,
    model: Any | None = None,
) -> InferenceResult:
    """Run one frozen MMSeg checkpoint and preserve native confidence evidence."""
    install_mmcv_lite_guard()
    try:
        apis = __import__("mmseg.apis", fromlist=["inference_model", "init_model"])
    except ModuleNotFoundError as error:
        raise RuntimeError("MMSeg is not installed; use the canonical Colab setup") from error
    active_model = model or apis.init_model(str(resolved_config), str(checkpoint), device=device)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = rgb[:, :, ::-1].copy()
    started = time.perf_counter()
    output = apis.inference_model(active_model, bgr)
    latency_ms = (time.perf_counter() - started) * 1000.0
    mask = output.pred_sem_seg.data.detach().cpu().numpy().squeeze().astype(np.uint8)
    if hasattr(output, "seg_logits"):
        logits = output.seg_logits.data.detach().cpu().numpy().astype(np.float32)
        confidence, entropy = confidence_entropy(logits)
    else:
        confidence = np.ones(mask.shape, dtype=np.float32)
        entropy = np.zeros(mask.shape, dtype=np.float32)
    if mask.shape != (image.height, image.width):
        mask = resize_mask(mask, image.size)
        confidence = resize_scalar(confidence, image.size)
        entropy = resize_scalar(entropy, image.size)
    return InferenceResult(
        mask=mask,
        confidence=confidence,
        entropy=entropy,
        latency_ms=latency_ms,
        backend=f"mmseg_{device}",
        metadata={"checkpoint": checkpoint.name},
        logits=logits if hasattr(output, "seg_logits") else None,
    )
