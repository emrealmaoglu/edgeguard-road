"""Measure class imbalance, and test whether it explains where the system fails.

Four measurements in this project land on the same classes -- pole, fence, traffic light,
person -- for per-class IoU, boundary agreement, calibration error and component
fragmentation. The explanation offered so far is output stride: a coarse grid resolves
thin structures badly. That is a mechanism, but there is a competing one that has to be
ruled out or admitted, because it predicts the same victims: those classes are also the
rarest, and a model sees far fewer of their pixels during training.

Both can be true and probably are. What is not acceptable is asserting one while the other
goes unmeasured. This measures the class distribution of the evaluation data and then
correlates pixel share against per-class IoU and per-class calibration error, so the
thesis can say how much of the pattern frequency accounts for.

Ground truth only -- no model runs here. The distribution is a property of the dataset.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from edgeguard.rescue.dataset import CITYSCAPES_CLASSES
from edgeguard.serialization import canonical_json

CLASS_COUNT = 19
IGNORE_LABEL = 255
MASK_SUFFIX = "_gtFine_labelTrainIds.png"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument(
        "--accuracy-record",
        type=Path,
        help="a `*_cityscapes_val.json` record; its `per_class_iou` is correlated against "
        "pixel share",
    )
    parser.add_argument(
        "--calibration-record",
        type=Path,
        help="a `calibration/*.json` record; its per-class ECE is correlated against pixel share",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _spearman(left: list[float], right: list[float]) -> float:
    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        ranks = [0.0] * len(values)
        for position, index in enumerate(order):
            ranks[index] = position + 1.0
        return ranks

    a, b = rank(left), rank(right)
    n = len(a)
    mean_a, mean_b = sum(a) / n, sum(b) / n
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    denominator = (sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)) ** 0.5
    return numerator / denominator if denominator else float("nan")


def main() -> int:
    args = _parser().parse_args()
    masks = sorted(args.mask_root.rglob(f"*{MASK_SUFFIX}"))[: args.frames]
    if not masks:
        raise ValueError(f"no label masks under {args.mask_root}")

    totals = np.zeros(CLASS_COUNT, dtype=np.int64)
    present = np.zeros(CLASS_COUNT, dtype=np.int64)
    for index, path in enumerate(masks, start=1):
        target = np.array(Image.open(path))
        valid = target[target != IGNORE_LABEL].astype(np.int64)
        counts = np.bincount(valid, minlength=CLASS_COUNT)[:CLASS_COUNT]
        totals += counts
        present += (counts > 0).astype(np.int64)
        if index % 100 == 0:
            print(f"  {index}/{len(masks)}", flush=True)

    labelled = int(totals.sum())
    share = totals / max(1, labelled)
    per_class = {
        CITYSCAPES_CLASSES[index]: {
            "pixels": int(totals[index]),
            "pixel_share": float(share[index]),
            "frames_present": int(present[index]),
            "frame_share": float(present[index] / len(masks)),
        }
        for index in range(CLASS_COUNT)
    }

    correlations: dict[str, float | None] = {}
    if args.accuracy_record and args.accuracy_record.is_file():
        record = json.loads(args.accuracy_record.read_text(encoding="utf-8"))
        iou = record.get("per_class_iou")
        if isinstance(iou, list) and len(iou) == CLASS_COUNT:
            correlations["pixel_share_vs_iou"] = _spearman(
                list(share), [float(value) for value in iou]
            )
    if args.calibration_record and args.calibration_record.is_file():
        record = json.loads(args.calibration_record.read_text(encoding="utf-8"))
        rows = record.get("per_class") or {}
        pairs = [
            (float(share[index]), float(rows[CITYSCAPES_CLASSES[index]]["ece"]))
            for index in range(CLASS_COUNT)
            if isinstance(rows.get(CITYSCAPES_CLASSES[index]), dict)
            and rows[CITYSCAPES_CLASSES[index]].get("ece") is not None
            and rows[CITYSCAPES_CLASSES[index]].get("counted")
        ]
        if len(pairs) > 2:
            correlations["pixel_share_vs_ece"] = _spearman(
                [p[0] for p in pairs], [p[1] for p in pairs]
            )
            correlations["classes_in_ece_correlation"] = float(len(pairs))

    ranked = sorted(per_class.items(), key=lambda item: item[1]["pixel_share"])
    result = {
        "schema_version": "1.0",
        "record_type": "edgeguard_class_distribution",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frames": len(masks),
        "labelled_pixels": labelled,
        "per_class": per_class,
        "rarest_five": [name for name, _ in ranked[:5]],
        "most_common_five": [name for name, _ in reversed(ranked[-5:])],
        "imbalance_ratio": float(share.max() / max(share.min(), 1e-12)),
        "correlations": correlations,
        "ground_truth_only": True,
        "scientific_measurement": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(result) + "\n", encoding="utf-8")

    print(f"\n{len(masks)} kare · {labelled:,} etiketli piksel")
    print(f"  dengesizlik oranı (en sık / en nadir): {result['imbalance_ratio']:.0f}×")
    print("\n  sınıf payına göre (en nadir üstte):")
    for name, row in ranked:
        print(
            f"    {name:14s} %{row['pixel_share'] * 100:6.3f} · "
            f"karelerin %{row['frame_share'] * 100:5.1f}'inde var"
        )
    if correlations:
        print("\n  Spearman:")
        for key, value in correlations.items():
            if value is not None and not key.startswith("classes_"):
                print(f"    {key:24s} {value:+.2f}")
    print(f"kayıt: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
