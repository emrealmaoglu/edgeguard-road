"""Measure how much the dominant class hides in a pooled calibration error.

Every ECE this project reports pools all pixels into one set of confidence bins. In a
driving scene that is dominated by road, building and sky, and those are the classes the
model is most confidently right about, so their well-calibrated mass can absorb severe
overconfidence on the small classes that actually matter -- pedestrian, rider, traffic
light. The pooled number then reads as "well calibrated" while the safety-relevant classes
are not.

That is an argument, and arguments are cheap. This measures it: the same pixels scored
both ways, so the gap between pooled and class-averaged ECE is a number rather than a
concern. Per-class errors come out with it, which is what tells you *which* classes the
pooled figure was covering for.

Classwise ECE here bins the top-1 confidence of the pixels **predicted** as each class and
averages the resulting per-class errors with equal weight. That is one of several
definitions in use and not the one-vs-rest form; the record names it so the two are not
silently compared.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from edgeguard.rescue.dataset import CITYSCAPES_CLASSES
from edgeguard.rescue.inference import predict_onnx
from edgeguard.serialization import canonical_json, sha256_file

CLASS_COUNT = 19
IGNORE_LABEL = 255
IMAGE_SUFFIX = "_leftImg8bit.png"
MASK_SUFFIX = "_gtFine_labelTrainIds.png"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument(
        "--minimum-class-pixels",
        type=int,
        default=1000,
        help="a class with fewer predicted pixels than this is reported but left out of "
        "the class average, since its ECE would be dominated by bin noise",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, bins: int
) -> tuple[float, float, float]:
    """Return ECE, mean confidence and mean accuracy for one set of pixels."""
    if confidence.size == 0:
        return float("nan"), float("nan"), float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, bins - 1)
    total = confidence.size
    error = 0.0
    for bucket in range(bins):
        selected = index == bucket
        count = int(np.count_nonzero(selected))
        if not count:
            continue
        error += (count / total) * abs(
            float(np.mean(correct[selected])) - float(np.mean(confidence[selected]))
        )
    return error, float(np.mean(confidence)), float(np.mean(correct))


def main() -> int:
    args = _parser().parse_args()
    images = sorted(args.image_root.rglob(f"*{IMAGE_SUFFIX}"))[: args.frames]
    masks = {path.name: path for path in args.mask_root.rglob(f"*{MASK_SUFFIX}")}
    if not images:
        raise ValueError(f"no frames under {args.image_root}")

    confidences: list[np.ndarray] = []
    corrects: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    generator = np.random.default_rng(20260728)
    # Every pixel of 200 full-resolution frames will not fit in memory, and does not need
    # to: a fixed per-frame sample with a declared seed estimates the same quantity and
    # keeps the run reproducible.
    per_frame = 20000
    for index, image_path in enumerate(images, start=1):
        identifier = image_path.name[: -len(IMAGE_SUFFIX)]
        mask_path = masks.get(f"{identifier}{MASK_SUFFIX}")
        if mask_path is None:
            continue
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        result = predict_onnx(image, args.model.resolve())
        target = np.array(Image.open(mask_path))
        valid = target != IGNORE_LABEL
        flat_valid = np.flatnonzero(valid.reshape(-1))
        if flat_valid.size == 0:
            continue
        chosen = (
            generator.choice(flat_valid, size=per_frame, replace=False)
            if flat_valid.size > per_frame
            else flat_valid
        )
        confidences.append(result.confidence.reshape(-1)[chosen].astype(np.float64))
        prediction = result.mask.reshape(-1)[chosen].astype(np.int64)
        predictions.append(prediction)
        corrects.append(
            (prediction == target.reshape(-1)[chosen].astype(np.int64)).astype(np.float64)
        )
        if index % 25 == 0:
            print(f"  {index}/{len(images)}", flush=True)

    confidence = np.concatenate(confidences)
    correct = np.concatenate(corrects)
    prediction = np.concatenate(predictions)

    pooled, pooled_confidence, pooled_accuracy = _expected_calibration_error(
        confidence, correct, args.bins
    )

    per_class: dict[str, dict[str, float]] = {}
    averaged: list[float] = []
    for class_id in range(CLASS_COUNT):
        selected = prediction == class_id
        count = int(np.count_nonzero(selected))
        name = CITYSCAPES_CLASSES[class_id]
        if not count:
            per_class[name] = {"predicted_pixels": 0, "ece": None, "counted": False}
            continue
        error, mean_confidence, mean_accuracy = _expected_calibration_error(
            confidence[selected], correct[selected], args.bins
        )
        counted = count >= args.minimum_class_pixels
        per_class[name] = {
            "predicted_pixels": count,
            "pixel_share": count / confidence.size,
            "ece": error,
            "mean_confidence": mean_confidence,
            "mean_accuracy": mean_accuracy,
            "overconfidence": mean_confidence - mean_accuracy,
            "counted": counted,
        }
        if counted:
            averaged.append(error)

    classwise = float(np.mean(averaged)) if averaged else float("nan")
    worst = max(
        (row for row in per_class.items() if row[1].get("counted")),
        key=lambda row: row[1]["ece"],
        default=(None, {}),
    )

    record = {
        "schema_version": "1.0",
        "record_type": "edgeguard_classwise_calibration",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model_name or args.model.name,
        "model_sha256": sha256_file(args.model),
        "frames": len(confidences),
        "sampled_pixels": int(confidence.size),
        "pixels_per_frame": per_frame,
        "seed": 20260728,
        "bins": args.bins,
        "pooled_ece": pooled,
        "pooled_mean_confidence": pooled_confidence,
        "pooled_mean_accuracy": pooled_accuracy,
        "classwise_ece": classwise,
        "classwise_minus_pooled": classwise - pooled,
        "classes_counted": len(averaged),
        "minimum_class_pixels": args.minimum_class_pixels,
        "worst_class": worst[0],
        "worst_class_ece": worst[1].get("ece"),
        "per_class": per_class,
        "classwise_definition": (
            "top-1 confidence of pixels predicted as each class, binned per class and "
            "averaged with equal weight; not the one-vs-rest form"
        ),
        "scientific_measurement": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(record) + "\n", encoding="utf-8")

    print(f"\n{record['model']} · {len(confidences)} kare · {confidence.size:,} piksel")
    print(f"  havuzlanmış ECE   {pooled:.4f}")
    print(f"  sınıf-bazlı ECE   {classwise:.4f}   ({classwise - pooled:+.4f})")
    print(f"  en kötü sınıf     {worst[0]} · ECE {worst[1].get('ece', float('nan')):.4f}")
    print("\n  sınıf bazında (pay > %1):")
    for name, row in sorted(per_class.items(), key=lambda item: -(item[1].get("ece") or 0.0)):
        if not row.get("counted") or (row.get("pixel_share") or 0) < 0.01:
            continue
        print(
            f"    {name:14s} ECE {row['ece']:.4f} · pay %{row['pixel_share'] * 100:.1f} · "
            f"aşırı-güven {row['overconfidence']:+.4f}"
        )
    print(f"kayıt: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
