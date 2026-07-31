"""Explicit deterministic and training transforms for project CV contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageEnhance


def resize_rgb(image: npt.NDArray[np.uint8], size_hw: tuple[int, int]) -> npt.NDArray[np.uint8]:
    """Resize RGB with bilinear interpolation."""
    height, width = size_hw
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(size_hw) <= 0:
        raise ValueError("RGB resize requires HWC uint8 input and positive output size")
    return np.asarray(
        Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8
    )


def resize_mask(mask: npt.NDArray[np.uint8], size_hw: tuple[int, int]) -> npt.NDArray[np.uint8]:
    """Resize categorical labels with nearest-neighbour interpolation."""
    height, width = size_hw
    if mask.dtype != np.uint8 or mask.ndim != 2 or min(size_hw) <= 0:
        raise ValueError("mask resize requires HW uint8 input and positive output size")
    return np.asarray(
        Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8
    )


def normalize_rgb(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Convert RGB to ImageNet-normalized CHW float32."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("normalization requires HWC uint8 RGB")
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    normalized = (image.astype(np.float32) / 255.0 - mean) / std
    return np.asarray(normalized.transpose(2, 0, 1), dtype=np.float32)


def semantic_training_transform(
    image: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.uint8],
    *,
    seed: int,
    crop_size_hw: tuple[int, int] = (512, 1024),
    scale_range: tuple[float, float] = (0.5, 2.0),
    horizontal_flip_probability: float = 0.5,
    photometric_strength: float = 0.1,
    class_aware_ids: tuple[int, ...] = (),
    class_aware_attempts: int = 8,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8], dict[str, object]]:
    """Apply aligned scale, crop/pad, flip, and bounded photometric augmentation."""
    if image.shape[:2] != mask.shape:
        raise ValueError("semantic image/mask geometry mismatch")
    if not 0 <= horizontal_flip_probability <= 1 or not 0 <= photometric_strength <= 1:
        raise ValueError("augmentation probabilities/strengths must lie in [0, 1]")
    generator = np.random.default_rng(seed)
    scale = float(generator.uniform(*scale_range))
    scaled_hw = (max(1, round(image.shape[0] * scale)), max(1, round(image.shape[1] * scale)))
    scaled_image = resize_rgb(image, scaled_hw)
    scaled_mask = resize_mask(mask, scaled_hw)
    crop_h, crop_w = crop_size_hw
    pad_h = max(0, crop_h - scaled_hw[0])
    pad_w = max(0, crop_w - scaled_hw[1])
    if pad_h or pad_w:
        scaled_image = np.pad(scaled_image, ((0, pad_h), (0, pad_w), (0, 0)))
        scaled_mask = np.pad(scaled_mask, ((0, pad_h), (0, pad_w)), constant_values=255)
    max_y = scaled_image.shape[0] - crop_h
    max_x = scaled_image.shape[1] - crop_w
    if class_aware_attempts < 1:
        raise ValueError("class-aware crop attempts must be positive")
    top = int(generator.integers(0, max_y + 1)) if max_y else 0
    left = int(generator.integers(0, max_x + 1)) if max_x else 0
    class_aware_matched = False
    if class_aware_ids:
        for _ in range(class_aware_attempts):
            candidate_top = int(generator.integers(0, max_y + 1)) if max_y else 0
            candidate_left = int(generator.integers(0, max_x + 1)) if max_x else 0
            candidate = scaled_mask[
                candidate_top : candidate_top + crop_h,
                candidate_left : candidate_left + crop_w,
            ]
            if bool(np.isin(candidate, np.asarray(class_aware_ids)).any()):
                top, left = candidate_top, candidate_left
                class_aware_matched = True
                break
    output_image = scaled_image[top : top + crop_h, left : left + crop_w]
    output_mask = scaled_mask[top : top + crop_h, left : left + crop_w]
    flipped = bool(generator.random() < horizontal_flip_probability)
    if flipped:
        output_image = output_image[:, ::-1].copy()
        output_mask = output_mask[:, ::-1].copy()
    brightness = 1.0 + float(generator.uniform(-photometric_strength, photometric_strength))
    output_image = np.asarray(
        ImageEnhance.Brightness(Image.fromarray(output_image)).enhance(brightness), dtype=np.uint8
    )
    return (
        normalize_rgb(output_image),
        output_mask,
        {
            "seed": seed,
            "scale": scale,
            "crop_top_left": [top, left],
            "crop_size_hw": list(crop_size_hw),
            "horizontal_flip": flipped,
            "brightness_factor": brightness,
            "class_aware_ids": list(class_aware_ids),
            "class_aware_matched": class_aware_matched,
        },
    )


def rare_class_sampling_weights(
    class_presence: npt.NDArray[np.bool_], *, power: float = 0.5
) -> npt.NDArray[np.float64]:
    """Return normalized sample weights from inverse image-level class presence."""
    if class_presence.dtype != np.bool_ or class_presence.ndim != 2 or not 0 < power <= 1:
        raise ValueError("rare-class sampling requires a boolean sample-by-class matrix")
    frequencies = class_presence.sum(axis=0, dtype=np.float64)
    active = frequencies > 0
    inverse = np.zeros_like(frequencies)
    inverse[active] = np.power(class_presence.shape[0] / frequencies[active], power)
    weights = class_presence.astype(np.float64) @ inverse
    if not np.any(weights > 0):
        raise ValueError("rare-class sampling requires at least one observed class")
    return np.asarray(weights / weights.sum(), dtype=np.float64)


def temporal_sample_indices(
    frame_indices: tuple[int, ...], *, stride: int = 1, maximum_gap: int = 0
) -> tuple[tuple[int, ...], tuple[dict[str, int], ...]]:
    """Select ordered frames and report sequence gaps without concealing discontinuities."""
    if stride < 1 or maximum_gap < 0 or any(index < 0 for index in frame_indices):
        raise ValueError("temporal stride/gap/frame indices are invalid")
    unordered = tuple(sorted(frame_indices)) != frame_indices
    duplicated = len(set(frame_indices)) != len(frame_indices)
    if unordered or duplicated:
        raise ValueError("temporal frame indices must be unique and ordered")
    selected = frame_indices[::stride]
    gaps = tuple(
        {
            "previous": previous,
            "current": current,
            "missing": current - previous - 1,
            "exceeds_maximum": int(current - previous - 1 > maximum_gap),
        }
        for previous, current in zip(selected, selected[1:], strict=False)
    )
    return selected, gaps


@dataclass(frozen=True)
class LetterboxReceipt:
    """Invertible detector resize/letterbox parameters."""

    source_size_hw: tuple[int, int]
    target_size_hw: tuple[int, int]
    scale: float
    pad_xy: tuple[float, float]


def letterbox_detection(
    image: npt.NDArray[np.uint8],
    boxes_xyxy: npt.NDArray[np.float32],
    target_hw: tuple[int, int],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], LetterboxReceipt]:
    """Letterbox RGB and transform boxes without changing their meaning."""
    source_h, source_w = image.shape[:2]
    target_h, target_w = target_hw
    scale = min(target_w / source_w, target_h / source_h)
    resized_hw = (max(1, round(source_h * scale)), max(1, round(source_w * scale)))
    resized = resize_rgb(image, resized_hw)
    pad_x = (target_w - resized_hw[1]) // 2
    pad_y = (target_h - resized_hw[0]) // 2
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[pad_y : pad_y + resized_hw[0], pad_x : pad_x + resized_hw[1]] = resized
    transformed = boxes_xyxy.astype(np.float32, copy=True)
    transformed[:, (0, 2)] = transformed[:, (0, 2)] * scale + pad_x
    transformed[:, (1, 3)] = transformed[:, (1, 3)] * scale + pad_y
    receipt = LetterboxReceipt((source_h, source_w), target_hw, scale, (float(pad_x), float(pad_y)))
    return canvas, transformed, receipt


def invert_letterbox_boxes(
    boxes_xyxy: npt.NDArray[np.float32], receipt: LetterboxReceipt
) -> npt.NDArray[np.float32]:
    """Return detector boxes to source coordinates and clip safely."""
    restored = boxes_xyxy.astype(np.float32, copy=True)
    restored[:, (0, 2)] = (restored[:, (0, 2)] - receipt.pad_xy[0]) / receipt.scale
    restored[:, (1, 3)] = (restored[:, (1, 3)] - receipt.pad_xy[1]) / receipt.scale
    source_h, source_w = receipt.source_size_hw
    restored[:, (0, 2)] = np.clip(restored[:, (0, 2)], 0, source_w)
    restored[:, (1, 3)] = np.clip(restored[:, (1, 3)], 0, source_h)
    return restored


def horizontal_flip_boxes(
    image: npt.NDArray[np.uint8], boxes_xyxy: npt.NDArray[np.float32]
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
    """Flip image and boxes together."""
    width = image.shape[1]
    flipped = boxes_xyxy.astype(np.float32, copy=True)
    flipped[:, 0] = width - boxes_xyxy[:, 2]
    flipped[:, 2] = width - boxes_xyxy[:, 0]
    return image[:, ::-1].copy(), flipped


def synthetic_outlier_exposure(
    image: npt.NDArray[np.uint8],
    road_mask: npt.NDArray[np.bool_],
    *,
    seed: int,
    scale_fraction: tuple[float, float] = (0.05, 0.2),
    alpha: float = 0.75,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], dict[str, object]]:
    """Place one deterministic blended rectangle wholly inside a valid road region."""
    if road_mask.dtype != np.bool_ or road_mask.shape != image.shape[:2] or not road_mask.any():
        raise ValueError("road-aware outlier placement requires a non-empty matching bool mask")
    if not 0 < scale_fraction[0] <= scale_fraction[1] <= 1 or not 0 < alpha <= 1:
        raise ValueError("outlier scale and alpha are invalid")
    generator = np.random.default_rng(seed)
    ys, xs = np.nonzero(road_mask)
    area = image.shape[0] * image.shape[1]
    side = max(1, round(np.sqrt(area * float(generator.uniform(*scale_fraction)))))
    candidates = [
        (int(y), int(x))
        for y, x in zip(ys, xs, strict=True)
        if y + side <= image.shape[0]
        and x + side <= image.shape[1]
        and bool(road_mask[y : y + side, x : x + side].all())
    ]
    if not candidates:
        raise ValueError("no valid road region can contain the requested synthetic outlier")
    top, left = candidates[int(generator.integers(0, len(candidates)))]
    color = generator.integers(0, 256, size=(1, 1, 3), dtype=np.uint8)
    output = image.copy()
    patch = output[top : top + side, left : left + side].astype(np.float32)
    output[top : top + side, left : left + side] = np.clip(
        (1 - alpha) * patch + alpha * color, 0, 255
    ).astype(np.uint8)
    target = np.zeros(image.shape[:2], dtype=np.uint8)
    target[top : top + side, left : left + side] = 1
    return (
        output,
        target,
        {"seed": seed, "bbox_xyxy": [left, top, left + side, top + side], "alpha": alpha},
    )
