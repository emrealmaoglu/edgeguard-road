"""Measure the deployed FP16 engine's accuracy on the target device.

Every accuracy number in this project is measured from ONNX in FP32, but what ships to the
Jetson is a TensorRT FP16 engine. `build_tensorrt.py` has always written
`numerical_equivalence_pending: true` into its manifest and nothing ever resolved it, so
the deployed model's accuracy has never been checked against the measured model's.

That gap matters more than it sounds: if half precision costs accuracy, then the mIoU
table describes a model nobody deploys. This runs the engine itself over labelled frames
on the device and reports the confusion-matrix mIoU, so the deployment number can be put
next to the FP32 number instead of assumed equal to it.

With `--onnx`, both precisions are scored on the *same frames in the same run* and the
per-frame difference is bootstrapped. That is a stronger answer than two mIoU values
quoted side by side: a gap of 0.003 between separately-measured numbers could be half
precision or could be the frame subset, and only the paired form can tell them apart. It
is also the difference between "FP16 appears to cost a little" and "the interval includes
zero", which is what the manifest's `numerical_equivalence_pending` flag is actually
asking.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

# Running this as `python scripts/jetson/<name>.py` puts *this directory* on the import
# path, not the repository root, so the shared runner below is not importable. The runbook
# documents exactly that invocation, so the script makes it work rather than asking the
# reader to know about PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edgeguard.evaluation.statistics import paired_comparison
from edgeguard.rescue.inference import predict_onnx, preprocess_image
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
    parser.add_argument(
        "--onnx",
        type=Path,
        help="score this FP32 ONNX graph on the same frames in the same run, and "
        "bootstrap the per-frame difference. Without it the record carries the engine's "
        "mIoU alone, which cannot separate a precision effect from a subset effect.",
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
    reference_matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    engine_frames: dict[str, float] = {}
    reference_frames: dict[str, float] = {}
    agreements: list[float] = []

    def _resize_to(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        # The engine emits logits at its own stride; compare at label resolution so the
        # score is the one a deployed system would actually produce.
        if prediction.shape == target.shape:
            return prediction
        return np.array(
            Image.fromarray(prediction, mode="L").resize(
                (target.shape[1], target.shape[0]), Image.Resampling.NEAREST
            )
        )

    def _frame_miou(prediction: np.ndarray, target: np.ndarray) -> float:
        valid = target != IGNORE_LABEL
        if not valid.any():
            return float("nan")
        indices = target[valid].astype(np.int64) * CLASS_COUNT + prediction[valid].astype(np.int64)
        frame = np.bincount(indices, minlength=CLASS_COUNT**2).reshape(CLASS_COUNT, CLASS_COUNT)
        overlap = np.diag(frame).astype(np.float64)
        total = frame.sum(axis=1) + frame.sum(axis=0) - np.diag(frame)
        present = total > 0
        return float(np.mean(overlap[present] / total[present])) if present.any() else float("nan")

    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        tensor = preprocess_image(image, (input_height, input_width))
        logits, _ = runner.infer(tensor)
        prediction = np.argmax(logits[0], axis=0).astype(np.uint8)

        target = np.array(Image.open(mask_path))
        prediction = _resize_to(prediction, target)
        accumulate(matrix, prediction, target)
        identifier = image_path.name[: -len(IMAGE_SUFFIX)]
        engine_frames[identifier] = _frame_miou(prediction, target)

        if args.onnx is not None:
            reference = predict_onnx(image, args.onnx.resolve())
            expected = _resize_to(reference.mask.astype(np.uint8), target)
            accumulate(reference_matrix, expected, target)
            reference_frames[identifier] = _frame_miou(expected, target)
            # Pixel agreement is reported separately from mIoU: the two answer different
            # questions, and a high agreement with a real mIoU gap would say the changed
            # pixels are the ones that matter.
            agreements.append(float(np.mean(expected == prediction)))
        if index % 25 == 0:
            print(f"  {index}/{len(pairs)}", flush=True)

    def _dataset_miou(counts: np.ndarray) -> np.ndarray:
        overlap = np.diag(counts).astype(np.float64)
        total = counts.sum(axis=1) + counts.sum(axis=0) - np.diag(counts)
        return np.where(total > 0, overlap / np.maximum(total, 1), np.nan)

    per_class = _dataset_miou(matrix)
    miou = float(np.nanmean(per_class))

    paired: dict[str, object] | None = None
    if args.onnx is not None:
        shared = sorted(set(engine_frames) & set(reference_frames))
        usable = [
            frame
            for frame in shared
            if np.isfinite(engine_frames[frame]) and np.isfinite(reference_frames[frame])
        ]
        if len(usable) > 1:
            comparison = paired_comparison(
                {frame: reference_frames[frame] for frame in usable},
                {frame: engine_frames[frame] for frame in usable},
            )
            interval = comparison["bootstrap_interval"]
            paired = {
                "frames": len(usable),
                "onnx": args.onnx.name,
                "fp32_dataset_miou": float(np.nanmean(_dataset_miou(reference_matrix))),
                "mean_per_frame_difference": comparison["mean_difference"],
                "lower_95": interval["lower"],
                "upper_95": interval["upper"],
                # The honest form of the equivalence question: an interval containing zero
                # means half precision costs nothing this measurement can detect, which is
                # a different and weaker statement than "the two are identical".
                "interval_excludes_zero": bool(interval["lower"] > 0 or interval["upper"] < 0),
                "mean_pixel_agreement": float(np.mean(agreements)) if agreements else None,
                "significance_claim": False,
            }

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
        "paired": paired,
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
    if paired:
        verdict = "AYRIŞIYOR" if paired["interval_excludes_zero"] else "ayırt edilemez"
        print(f"\nAynı koşuda FP32 ONNX: {paired['fp32_dataset_miou']:.4f}")
        print(
            f"eşleştirilmiş fark  : {paired['mean_per_frame_difference']:+.4f} "
            f"[{paired['lower_95']:+.4f}, {paired['upper_95']:+.4f}]  {verdict}"
        )
        if paired["mean_pixel_agreement"] is not None:
            print(f"piksel uyumu        : %{paired['mean_pixel_agreement'] * 100:.4f}")
    print(f"kayıt: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
