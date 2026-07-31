"""Pure visualization helpers shared by CLI prediction and Streamlit."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

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


def save_perception_result(
    image: Image.Image,
    perception: Any,
    output_dir: Path,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Persist derived drivable, reliability, attention, and region artifacts."""
    required = (
        "road_mask",
        "drivable_corridor",
        "unreliable_mask",
        "attention_map",
        "regions",
    )
    if any(not hasattr(perception, name) for name in required):
        raise TypeError("perception result does not satisfy the EdgeGuard output contract")
    paths = {
        "road_mask": output_dir / "road_mask.png",
        "drivable_mask": output_dir / "drivable_corridor.png",
        "unreliable_mask": output_dir / "unreliable_mask.png",
        "attention_map": output_dir / "attention_map.png",
        "regions_overlay": output_dir / "regions_overlay.png",
    }
    Image.fromarray(perception.road_mask.astype(np.uint8) * 255, mode="L").save(paths["road_mask"])
    Image.fromarray(perception.drivable_corridor.astype(np.uint8) * 255, mode="L").save(
        paths["drivable_mask"]
    )
    Image.fromarray(perception.unreliable_mask.astype(np.uint8) * 255, mode="L").save(
        paths["unreliable_mask"]
    )
    Image.fromarray(
        np.clip(perception.attention_map * 255, 0, 255).astype(np.uint8), mode="L"
    ).save(paths["attention_map"])
    overlay = image.convert("RGB").copy()
    drawer = ImageDraw.Draw(overlay)
    colors = {"low": (0, 200, 0), "medium": (255, 180, 0), "high": (230, 0, 0)}
    region_root = output_dir / "regions"
    region_root.mkdir()
    records: list[dict[str, Any]] = []
    for region in perception.regions:
        mask_name = f"region-{region.region_id:04d}-{region.class_name.replace(' ', '_')}.png"
        Image.fromarray(region.mask.astype(np.uint8) * 255, mode="L").save(region_root / mask_name)
        color = colors[region.attention_level]
        x1, y1, x2, y2 = region.bbox_xyxy
        drawer.rectangle((x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)), outline=color, width=2)
        drawer.text(
            (x1, max(0, y1 - 12)),
            f"{region.class_name} {region.attention_score:.2f}",
            fill=color,
        )
        records.append(region.to_dict(mask_path=f"regions/{mask_name}"))
    overlay.save(paths["regions_overlay"])
    return ({key: path.name for key, path in paths.items()}, records)
