"""Measure the deployed FP16 engine's accuracy on the target device.

Every accuracy number in this project is measured from ONNX in FP32, but what ships to the
Jetson is a TensorRT FP16 engine. `build_tensorrt.py` has always written
`numerical_equivalence_pending: true` into its manifest and nothing ever resolved it, so
the deployed model's accuracy has never been checked against the measured model's.

That gap matters more than it sounds: if half precision costs accuracy, then the mIoU
table describes a model nobody deploys. This runs the engine itself over labelled frames
on the device and reports the confusion-matrix mIoU, so the deployment number can be put
next to the FP32 number instead of assumed equal to it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from edgeguard.rescue.inference import preprocess_image
from edgeguard.serialization import canonical_json, sha256_file
from scripts.jetson.benchmark import TensorRTTorchRunner

CLASS_COUNT = 19
IGNORE_LABEL = 255
IMAGE_SUFFIX = "_leftImg8bit.png"
MASK_SUFFIX = "_gtFine_labelTrainIds.png"


def discover_pairs(image_root: Path, mask_root: Path, limit: int | None) -> list[tuple[Path, Path]]:
    """Pair Cityscapes-layout frames with their train-id masks."""
    pairs: list[tuple[Path, Path]] = []
    for image in sorted(image_root.rglob(f"*{IMAGE_SUFFIX}")):
        identifier = image.name[: -len(IMAGE_SUFFIX)]
        matches = list(mask_root.rglob(f"{identifier}{MASK_SUFFIX}"))
        if matches:
            pairs.append((image, matches[0]))
        if limit and len(pairs) >= limit:
            break
    if not pairs:
        raise ValueError(f"no image/mask pairs found under {image_root} and {mask_root}")
    return pairs


def accumulate(matrix: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> None:
    """Add one frame to the confusion matrix, dropping ignore pixels."""
    valid = target != IGNORE_LABEL
    if not valid.any():
        return
    indices = target[valid].astype(np.int64) * CLASS_COUNT + prediction[valid].astype(np.int64)
    counts = np.bincount(indices, minlength=CLASS_COUNT * CLASS_COUNT)
    matrix += counts.reshape(CLASS_COUNT, CLASS_COUNT)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--engine-manifest", type=Path)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--reference-miou",
        type=float,
        help="the FP32 mIoU measured off-device, so the record carries the comparison",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.engine_manifest is not None:
        import json

        manifest = json.loads(args.engine_manifest.read_text(encoding="utf-8"))
        if manifest.get("engine_sha256") != sha256_file(args.engine):
            raise ValueError("TensorRT engine does not match its build manifest")

    pairs = discover_pairs(args.image_root.resolve(), args.mask_root.resolve(), args.limit)
    runner = TensorRTTorchRunner(args.engine)
    input_height, input_width = runner.input_shape[2:]

    matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        logits, _ = runner.infer(preprocess_image(image, (input_height, input_width)))
        prediction = np.argmax(logits[0], axis=0).astype(np.uint8)

        target = np.array(Image.open(mask_path))
        # The engine emits logits at its own stride; compare at label resolution so the
        # score is the one a deployed system would actually produce.
        if prediction.shape != target.shape:
            prediction = np.array(
                Image.fromarray(prediction, mode="L").resize(
                    (target.shape[1], target.shape[0]), Image.Resampling.NEAREST
                )
            )
        accumulate(matrix, prediction, target)
        if index % 25 == 0:
            print(f"  {index}/{len(pairs)}", flush=True)

    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - np.diag(matrix)
    per_class = np.where(union > 0, intersection / np.maximum(union, 1), np.nan)
    miou = float(np.nanmean(per_class))

    record = {
        "schema_version": "1.0",
        "record_type": "edgeguard_jetson_engine_accuracy",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": args.engine.name,
        "engine_sha256": sha256_file(args.engine),
        "backend": "tensorrt_fp16",
        "frames": len(pairs),
        "mIoU": miou,
        "per_class_iou": [None if np.isnan(value) else float(value) for value in per_class],
        "reference_fp32_miou": args.reference_miou,
        "fp16_minus_fp32_miou": (
            None if args.reference_miou is None else miou - args.reference_miou
        ),
        # This is what resolves the manifest's standing `numerical_equivalence_pending`.
        "numerical_equivalence_measured": True,
        "scientific_measurement": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(record) + "\n", encoding="utf-8")

    print(f"\nFP16 motor mIoU : {miou:.4f}  ({len(pairs)} kare)")
    if args.reference_miou is not None:
        print(f"FP32 referans   : {args.reference_miou:.4f}")
        print(f"fark            : {miou - args.reference_miou:+.4f}")
    print(f"kayıt: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
