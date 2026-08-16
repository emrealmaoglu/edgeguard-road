"""Score object-level localisation, which pixel metrics can flatter or punish unfairly.

mIoU counts pixels, so a model that finds a third of one large obstacle and a model that
finds three separate thirds of it score the same, and a model that misses a small but
lethal object entirely is barely penalised. Research document 26 makes this argument and
prescribes component-level metrics; `evaluation/perception.py` has carried
`component_localization_metrics` -- complete, tested -- with no caller since it was
written, which is the same pattern `drivable_metrics` and `paired_comparison` were in.

What it answers that pixels cannot: what fraction of ground-truth objects the model finds
at all (coverage), how badly it splits one object into several (fragmentation), and how
often it merges distinct objects into one (merging). Fragmentation matters downstream --
the temporal tracker and the risk ranking both operate on components, so an object split
in two is two tracks competing for attention.

Scored on Cityscapes val for the ten attention classes, against ground truth at the same
resolution.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from edgeguard.evaluation.perception import component_localization_metrics
from edgeguard.rescue.inference import predict_onnx
from edgeguard.serialization import canonical_json, sha256_file

IMAGE_SUFFIX = "_leftImg8bit.png"
MASK_SUFFIX = "_gtFine_labelTrainIds.png"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument(
        "--minimum-area",
        type=int,
        default=64,
        help="components smaller than this are ignored on both sides, so the metric is "
        "not dominated by single-pixel speckle that no consumer would act on",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    images = sorted(args.image_root.rglob(f"*{IMAGE_SUFFIX}"))[: args.frames]
    masks = {path.name: path for path in args.mask_root.rglob(f"*{MASK_SUFFIX}")}
    if not images:
        raise ValueError(f"no frames under {args.image_root}")

    rows: list[dict[str, float]] = []
    for index, image_path in enumerate(images, start=1):
        identifier = image_path.name[: -len(IMAGE_SUFFIX)]
        mask_path = masks.get(f"{identifier}{MASK_SUFFIX}")
        if mask_path is None:
            continue
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        result = predict_onnx(image, args.model.resolve())
        target = np.array(Image.open(mask_path)).astype(np.int64)
        rows.append(
            component_localization_metrics(
                result.mask.astype(np.int64), target, minimum_area=args.minimum_area
            )
        )
        if index % 20 == 0:
            print(f"  {index}/{len(images)}", flush=True)

    def summarise(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    record = {
        "schema_version": "1.0",
        "record_type": "edgeguard_component_localization",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model_name or args.model.name,
        "model_sha256": sha256_file(args.model),
        "frames": len(rows),
        "minimum_area": args.minimum_area,
        "component_coverage": summarise("component_coverage"),
        "mean_best_component_iou": summarise("mean_best_component_iou"),
        "mean_fragments_per_ground_truth": summarise("mean_fragments_per_ground_truth"),
        "mean_ground_truths_per_prediction": summarise("mean_ground_truths_per_prediction"),
        "ground_truth_component_count": summarise("ground_truth_component_count"),
        "predicted_component_count": summarise("predicted_component_count"),
        # Named because the record must not be mistaken for a detector benchmark: these
        # are connected semantic regions, not instances, and two adjacent cars of the same
        # class are one component here.
        "instance_detection": False,
        "scientific_measurement": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(record) + "\n", encoding="utf-8")

    print(f"\n{record['model']} · {len(rows)} kare")
    print(f"  bileşen kapsama       {record['component_coverage']:.4f}")
    print(f"  en iyi bileşen IoU    {record['mean_best_component_iou']:.4f}")
    print(f"  GT başına parça       {record['mean_fragments_per_ground_truth']:.3f}")
    print(f"  tahmin başına GT      {record['mean_ground_truths_per_prediction']:.3f}")
    print(
        f"  bileşen sayısı        GT {record['ground_truth_component_count']:.1f} · "
        f"tahmin {record['predicted_component_count']:.1f}"
    )
    print(f"kayıt: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
