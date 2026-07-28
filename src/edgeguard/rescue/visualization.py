"""Pure visualization helpers shared by CLI prediction and Streamlit."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edgeguard.rescue.mmseg_runtime import CITYSCAPES_PALETTE


@dataclass(frozen=True)
class InferenceResult:
    """Backend-neutral semantic output at the original image resolution."""

    mask: np.ndarray
    confidence: np.ndarray
    entropy: np.ndarray
    latency_ms: float
    backend: str
    metadata: dict[str, Any]
    logits: np.ndarray | None = None


def stable_probabilities(logits: np.ndarray) -> np.ndarray:
    """Convert finite CHW logits to stable class probabilities."""
    if logits.ndim != 3 or logits.shape[0] != 19 or not bool(np.isfinite(logits).all()):
        raise ValueError("logits must be a finite 19-channel CHW array")
    shifted = logits.astype(np.float64) - np.max(logits, axis=0, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=0, keepdims=True)


def confidence_entropy(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return maximum-softmax confidence and normalized predictive entropy."""
    probabilities = stable_probabilities(logits)
    confidence = np.max(probabilities, axis=0)
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1.0e-12, 1.0)), axis=0)
    entropy /= np.log(probabilities.shape[0])
    return confidence.astype(np.float32), entropy.astype(np.float32)


def calibrate_inference_result(result: InferenceResult, temperature: float) -> InferenceResult:
    """Recompute displayed probabilities from preserved logits and one scalar."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("calibration temperature must be positive and finite")
    if result.logits is None:
        raise ValueError("selected backend did not preserve logits for calibration")
    confidence, entropy = confidence_entropy(result.logits / temperature)
    target_size = (int(result.mask.shape[1]), int(result.mask.shape[0]))
    if confidence.shape != result.mask.shape:
        confidence = resize_scalar(confidence, target_size)
        entropy = resize_scalar(entropy, target_size)
    return replace(
        result,
        confidence=confidence,
        entropy=entropy,
        metadata={**result.metadata, "calibrated": True, "temperature": temperature},
    )


def resize_scalar(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a scalar map to width/height using bilinear interpolation."""
    image = Image.fromarray(values.astype(np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize categorical IDs to width/height using nearest-neighbor only."""
    image = Image.fromarray(mask.astype(np.uint8), mode="L")
    return np.asarray(image.resize(size, Image.Resampling.NEAREST), dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> Image.Image:
    """Convert a 0-18 semantic mask into the frozen Cityscapes palette."""
    if mask.ndim != 2 or bool(((mask < 0) | (mask > 18)).any()):
        raise ValueError("semantic mask must contain only class IDs 0-18")
    palette = np.asarray(CITYSCAPES_PALETTE, dtype=np.uint8)
    return Image.fromarray(palette[mask.astype(np.int64)], mode="RGB")


def overlay_mask(image: Image.Image, mask: np.ndarray, opacity: float) -> Image.Image:
    """Blend a semantic palette onto one RGB image."""
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be between zero and one")
    base = image.convert("RGB")
    colored = colorize_mask(mask)
    if colored.size != base.size:
        colored = colored.resize(base.size, Image.Resampling.NEAREST)
    return Image.blend(base, colored, opacity)


def save_result(
    image: Image.Image, result: InferenceResult, output_dir: Path, *, opacity: float = 0.55
) -> dict[str, str]:
    """Persist mask, overlay, confidence, and entropy visualizations."""
    output_dir.mkdir(parents=True, exist_ok=False)
    mask_path = output_dir / "mask.png"
    overlay_path = output_dir / "overlay.png"
    confidence_path = output_dir / "confidence.png"
    entropy_path = output_dir / "entropy.png"
    Image.fromarray(result.mask.astype(np.uint8), mode="L").save(mask_path)
    overlay_mask(image, result.mask, opacity).save(overlay_path)
    Image.fromarray(np.clip(result.confidence * 255, 0, 255).astype(np.uint8), mode="L").save(
        confidence_path
    )
    Image.fromarray(np.clip(result.entropy * 255, 0, 255).astype(np.uint8), mode="L").save(
        entropy_path
    )
    return {
        "mask": mask_path.name,
        "overlay": overlay_path.name,
        "confidence": confidence_path.name,
        "entropy": entropy_path.name,
    }
