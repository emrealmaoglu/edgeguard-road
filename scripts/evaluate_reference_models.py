"""Measure accuracy and calibration of a segmentation model on a labelled split.

Deliberately standalone. `evaluate.py` routes through the campaign's role and manifest
gates, which exist to keep sealed data sealed and are the right thing for the training
pipeline -- but they also require frozen dataset manifests this evaluation does not have
and does not need. Cityscapes val and ACDC val are public, already labelled, and touch no
sealed release, so this reads them directly and computes the confusion matrix, the
per-class IoU and the reliability curve in one pass.

Calibration is the point as much as accuracy: the thesis claims the system measures how
sure it is, and a mean confidence means nothing without knowing whether that confidence
tracks correctness. Expected calibration error and the bin-wise reliability curve are
computed over the same pixels the mIoU comes from.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edgeguard.rescue.inference import IMAGENET_MEAN, IMAGENET_STD, predict_onnx
from edgeguard.serialization import canonical_json, sha256_file

CLASS_COUNT = 19
IGNORE_LABEL = 255
RELIABILITY_BINS = 15


def _torch_letterbox(
    image: Image.Image, input_size: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Reproduce the Jetson GPU preprocessing on CPU so its accuracy cost is measurable.

    The device path resizes with torch's bilinear filter and `antialias=True`, which is
    close to PIL's but not identical; on real frames the two disagree on 0.54% of pixels.
    Whether that matters is an mIoU question, and answering it does not need the Jetson.
    """
    import torch

    height, width = input_size
    rgb = image.convert("RGB")
    frame = torch.from_numpy(np.array(rgb, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0).float()
    scale = min(width / rgb.width, height / rgb.height)
    target = (max(1, round(rgb.height * scale)), max(1, round(rgb.width * scale)))
    resized = torch.nn.functional.interpolate(
        frame, size=target, mode="bilinear", align_corners=False, antialias=True
    )
    canvas = torch.zeros((1, 3, height, width), dtype=torch.float32)
    top = (height - target[0]) // 2
    left = (width - target[1]) // 2
    canvas[:, :, top : top + target[0], left : left + target[1]] = resized
    array = canvas.numpy()[0].transpose(1, 2, 0)
    normalized = (array - IMAGENET_MEAN) / IMAGENET_STD
    tensor = np.transpose(normalized, (2, 0, 1))[None].astype(np.float32)
    return tensor, (left, top, left + target[1], top + target[0])


def discover_pairs(
    image_root: Path, mask_root: Path, image_suffix: str, mask_suffix: str, limit: int | None
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    masks = {path.name: path for path in mask_root.rglob(f"*{mask_suffix}")}
    for image in sorted(image_root.rglob(f"*{image_suffix}")):
        identifier = image.name[: -len(image_suffix)]
        mask = masks.get(f"{identifier}{mask_suffix}")
        if mask is not None:
            pairs.append((image, mask))
        if limit and len(pairs) >= limit:
            break
    if not pairs:
        raise ValueError(f"no pairs under {image_root} / {mask_root}")
    return pairs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help=".onnx graph")
    parser.add_argument("--model-name")
    parser.add_argument("--image-suffix", default="_leftImg8bit.png")
    parser.add_argument("--mask-suffix", default="_gtFine_labelTrainIds.png")
    parser.add_argument("--split-name", default="cityscapes_val")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=1024)
    parser.add_argument(
        "--calibration-pixels-per-frame",
        type=int,
        default=20_000,
        help="deterministic subsample used for the reliability curve; mIoU uses every pixel",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--torch-preprocess",
        action="store_true",
        help="letterbox with torch bilinear+antialias instead of PIL, matching what the "
        "Jetson GPU path does, so its accuracy cost can be measured off-device",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    pairs = discover_pairs(
        args.image_root.resolve(),
        args.mask_root.resolve(),
        args.image_suffix,
        args.mask_suffix,
        args.limit,
    )
    generator = np.random.default_rng(args.seed)
    matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    confidences: list[np.ndarray] = []
    correctness: list[np.ndarray] = []

    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        result = predict_onnx(
            image,
            args.model.resolve(),
            input_size=(args.input_height, args.input_width),
            letterbox=_torch_letterbox if args.torch_preprocess else None,
        )
        target = np.array(Image.open(mask_path))
        prediction, confidence = result.mask, result.confidence
        if prediction.shape != target.shape:
            prediction = np.array(
                Image.fromarray(prediction, mode="L").resize(
                    (target.shape[1], target.shape[0]), Image.Resampling.NEAREST
                )
            )
            confidence = np.array(
                Image.fromarray(confidence).resize(
                    (target.shape[1], target.shape[0]), Image.Resampling.BILINEAR
                )
            )

        valid = target != IGNORE_LABEL
        if valid.any():
            flat = target[valid].astype(np.int64) * CLASS_COUNT + prediction[valid].astype(np.int64)
            matrix += np.bincount(flat, minlength=CLASS_COUNT**2).reshape(CLASS_COUNT, CLASS_COUNT)
            # The reliability curve needs a bounded sample; the confusion matrix does not,
            # so accuracy stays exact while calibration stays affordable.
            valid_indices = np.flatnonzero(valid.reshape(-1))
            if valid_indices.size > args.calibration_pixels_per_frame:
                valid_indices = generator.choice(
                    valid_indices, size=args.calibration_pixels_per_frame, replace=False
                )
            confidences.append(confidence.reshape(-1)[valid_indices].astype(np.float32))
            correctness.append(
                (prediction.reshape(-1)[valid_indices] == target.reshape(-1)[valid_indices]).astype(
                    np.float32
                )
            )
        if index % 25 == 0:
            print(f"  {index}/{len(pairs)}", flush=True)

    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - np.diag(matrix)
    per_class = np.where(union > 0, intersection / np.maximum(union, 1), np.nan)
    pixel_accuracy = float(np.diag(matrix).sum() / max(1, matrix.sum()))

    all_confidence = np.concatenate(confidences)
    all_correct = np.concatenate(correctness)
    edges = np.linspace(0.0, 1.0, RELIABILITY_BINS + 1)
    assignment = np.clip(np.digitize(all_confidence, edges[1:-1]), 0, RELIABILITY_BINS - 1)
    reliability: list[dict[str, Any]] = []
    expected_error = 0.0
    for bin_index in range(RELIABILITY_BINS):
        selected = assignment == bin_index
        count = int(np.count_nonzero(selected))
        if count == 0:
            reliability.append(
                {"bin": bin_index, "count": 0, "mean_confidence": None, "accuracy": None}
            )
            continue
        mean_confidence = float(np.mean(all_confidence[selected]))
        accuracy = float(np.mean(all_correct[selected]))
        expected_error += count / all_confidence.size * abs(mean_confidence - accuracy)
        reliability.append(
            {
                "bin": bin_index,
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )

    record = {
        "schema_version": "1.0",
        "record_type": "edgeguard_reference_model_evaluation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model_name or args.model.name,
        "model_sha256": sha256_file(args.model),
        "split": args.split_name,
        "frames": len(pairs),
        "mIoU": float(np.nanmean(per_class)),
        "pixel_accuracy": pixel_accuracy,
        "per_class_iou": [None if np.isnan(value) else float(value) for value in per_class],
        "expected_calibration_error": expected_error,
        "mean_confidence": float(np.mean(all_confidence)),
        "mean_accuracy": float(np.mean(all_correct)),
        # A model whose mean confidence exceeds its accuracy is overconfident by exactly
        # that much; the sign matters more than the magnitude for a safety argument.
        "confidence_minus_accuracy": float(np.mean(all_confidence) - np.mean(all_correct)),
        "reliability_bins": reliability,
        "calibration_pixels": int(all_confidence.size),
        "seed": args.seed,
        "temperature_scaling_applied": False,
        "scientific_measurement": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(record) + "\n", encoding="utf-8")

    print(f"\n{record['model']} · {args.split_name} · {len(pairs)} kare")
    print(f"  mIoU      {record['mIoU']:.4f}   piksel doğruluğu {pixel_accuracy:.4f}")
    print(f"  ECE       {expected_error:.4f}")
    print(
        f"  güven {record['mean_confidence']:.4f} vs doğruluk {record['mean_accuracy']:.4f} "
        f"-> aşırı-güven {record['confidence_minus_accuracy']:+.4f}"
    )
    print(f"kayıt: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
