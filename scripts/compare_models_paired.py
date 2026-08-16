"""Decide which accuracy differences between architectures survive sampling noise.

The comparison table reports one mIoU per model and orders them. Two of those gaps are
small enough that the ordering may be an artefact of which 500 frames Cityscapes happens
to use for validation -- SegFormer-B0 leads DDRNet-23-slim by 0.0084, and PIDNet-M trails
DDRNet by 0.0003, which is presented as a rank. A thesis that ranks five models owes an
answer to "is that difference real?", and until now the project had none: the aggregate
records keep only the dataset-level number, so nothing could be tested.

`evaluation/statistics.py` has carried `paired_comparison` and
`deterministic_bootstrap_interval` -- complete, tested, and never called -- for exactly
this. What was missing is per-frame scores. This scores every model on the *same* frames
and keeps them, so each pair of models is compared on identical images and the difference
is bootstrapped.

Pairing matters here: frames vary enormously in difficulty, and that variance is shared
between models. Comparing unpaired means would drown an 0.008 difference in it.

Note on the statistic: per-frame mIoU averages over the classes *present in that frame*,
which is not the same quantity as dataset mIoU accumulated over one global confusion
matrix. Both are reported. The paired test is on the per-frame series, and the record says
so rather than implying the dataset number was tested.
"""

from __future__ import annotations

import argparse
import itertools
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edgeguard.evaluation.statistics import deterministic_bootstrap_interval, paired_comparison
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
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="repeatable, as `pidnet_s=/path/to/pidnet_s.onnx`",
    )
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _frame_miou(matrix: np.ndarray) -> float:
    """mIoU over the classes this frame actually contains."""
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - np.diag(matrix)
    present = union > 0
    if not present.any():
        return float("nan")
    return float(np.mean(intersection[present] / union[present]))


def _dataset_miou(matrix: np.ndarray) -> float:
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - np.diag(matrix)
    return float(np.nanmean(np.where(union > 0, intersection / np.maximum(union, 1), np.nan)))


def main() -> int:
    args = _parser().parse_args()
    models: dict[str, Path] = {}
    for entry in args.model:
        name, _, path = str(entry).partition("=")
        if not path:
            raise ValueError(f"--model expects NAME=PATH, got {entry!r}")
        models[name] = Path(path).expanduser().resolve()

    images = sorted(args.image_root.rglob(f"*{IMAGE_SUFFIX}"))[: args.frames]
    masks = {path.name: path for path in args.mask_root.rglob(f"*{MASK_SUFFIX}")}
    if not images:
        raise ValueError(f"no frames under {args.image_root}")

    per_frame: dict[str, dict[str, float]] = {name: {} for name in models}
    totals: dict[str, np.ndarray] = {
        name: np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64) for name in models
    }
    for index, image_path in enumerate(images, start=1):
        identifier = image_path.name[: -len(IMAGE_SUFFIX)]
        mask_path = masks.get(f"{identifier}{MASK_SUFFIX}")
        if mask_path is None:
            continue
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        target = np.array(Image.open(mask_path))
        valid = target != IGNORE_LABEL
        truth = target[valid].astype(np.int64) * CLASS_COUNT
        for name, path in models.items():
            result = predict_onnx(image, path)
            matrix = np.bincount(
                truth + result.mask[valid].astype(np.int64), minlength=CLASS_COUNT**2
            ).reshape(CLASS_COUNT, CLASS_COUNT)
            totals[name] += matrix
            # Keyed by frame identity, which is what makes the comparison paired.
            per_frame[name][identifier] = _frame_miou(matrix)
        if index % 25 == 0:
            print(f"  {index}/{len(images)}", flush=True)

    shared = sorted(set.intersection(*(set(scores) for scores in per_frame.values())))
    if len(shared) < 2:
        raise ValueError("paired comparison needs at least two frames scored by every model")

    comparisons: list[dict[str, Any]] = []
    for left, right in itertools.combinations(sorted(models), 2):
        comparison = paired_comparison(
            {frame: per_frame[left][frame] for frame in shared},
            {frame: per_frame[right][frame] for frame in shared},
        )
        interval = comparison["bootstrap_interval"]
        comparisons.append(
            {
                "left": left,
                "right": right,
                "mean_difference": comparison["mean_difference"],
                "lower_95": interval["lower"],
                "upper_95": interval["upper"],
                # The interval excluding zero is the honest way to say this: a difference
                # whose interval spans zero is not a ranking, whatever the point estimates
                # look like side by side.
                "interval_excludes_zero": bool(interval["lower"] > 0 or interval["upper"] < 0),
                "significance_claim": False,
            }
        )

    record = {
        "schema_version": "1.0",
        "record_type": "edgeguard_paired_model_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frames": len(shared),
        "models": {
            name: {
                "model_sha256": sha256_file(path),
                "dataset_mIoU": _dataset_miou(totals[name]),
                "mean_per_frame_mIoU": float(np.mean([per_frame[name][frame] for frame in shared])),
                "per_frame_interval": deterministic_bootstrap_interval(
                    np.asarray([per_frame[name][frame] for frame in shared], dtype=np.float64)
                ),
            }
            for name, path in models.items()
        },
        "pairs": comparisons,
        "statistic": "per-frame mIoU over classes present in that frame",
        "pairing": "identical frames scored by every model",
        "scientific_measurement": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(record) + "\n", encoding="utf-8")

    print(f"\n{len(shared)} ortak kare · {len(models)} model")
    for name, summary in sorted(
        record["models"].items(), key=lambda item: -item[1]["mean_per_frame_mIoU"]
    ):
        interval = summary["per_frame_interval"]
        print(
            f"  {name:16s} veri kümesi {summary['dataset_mIoU']:.4f} · "
            f"kare-başına {summary['mean_per_frame_mIoU']:.4f} "
            f"[{interval['lower']:.4f}, {interval['upper']:.4f}]"
        )
    print("\n  eşleştirilmiş farklar (%95 bootstrap):")
    for pair in sorted(comparisons, key=lambda row: -abs(row["mean_difference"])):
        verdict = "ayrışıyor" if pair["interval_excludes_zero"] else "AYIRT EDİLEMEZ"
        print(
            f"  {pair['right']:16s} − {pair['left']:16s} "
            f"{pair['mean_difference']:+.4f} "
            f"[{pair['lower_95']:+.4f}, {pair['upper_95']:+.4f}]  {verdict}"
        )
    print(f"\nkayıt: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
