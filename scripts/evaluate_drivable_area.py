"""Score the drivable corridor against ground-truth road, per architecture.

The thesis claims the system identifies the drivable area. So far that claim rests on
pictures: `drivable_corridor.png` looks right. `edgeguard.evaluation.perception` has
carried a complete, tested metric suite for exactly this -- road IoU, boundary F1,
false-drivable rate, fragmentation -- with no caller, because it needed ground-truth road
masks and none were on hand. Cityscapes val supplies them.

The false-drivable rate is the one that matters for safety: it counts pixels the system
would drive into that are not road.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from edgeguard.evaluation.perception import drivable_metrics
from edgeguard.rescue.inference import predict_onnx
from edgeguard.rescue.perception import drivable_corridor_from_semantics
from edgeguard.serialization import canonical_json, sha256_file

IMAGE_SUFFIX = "_leftImg8bit.png"
MASK_SUFFIX = "_gtFine_labelTrainIds.png"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--minimum-drivable-area", type=int, default=64)
    # A common tolerance for every model, set to 8 because that is the deployment logit
    # stride of four of the five. SegFormer-B0 emits at stride 4, which is exactly why it
    # is expected to lead on boundaries -- so the tolerance stays fixed across models
    # rather than following each one, or the comparison would hide the difference it is
    # meant to expose.
    parser.add_argument("--output-stride", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    images = sorted(args.image_root.rglob(f"*{IMAGE_SUFFIX}"))[: args.frames]
    masks = {path.name: path for path in args.mask_root.rglob(f"*{MASK_SUFFIX}")}
    if not images:
        raise ValueError(f"no frames under {args.image_root}")

    road_rows: list[dict[str, float]] = []
    corridor_rows: list[dict[str, float]] = []
    for index, image_path in enumerate(images, start=1):
        identifier = image_path.name[: -len(IMAGE_SUFFIX)]
        mask_path = masks.get(f"{identifier}{MASK_SUFFIX}")
        if mask_path is None:
            continue
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        result = predict_onnx(image, args.model.resolve())
        target = np.array(Image.open(mask_path))
        road, corridor = drivable_corridor_from_semantics(
            result.mask.astype(np.int64), minimum_area=args.minimum_drivable_area
        )
        # Two questions, not one: how well the road class is segmented, and how well the
        # ego corridor -- the subset the vehicle would actually enter -- is carved from it.
        road_rows.append(
            drivable_metrics(
                road.astype(np.bool_), target.astype(np.int64), output_stride=args.output_stride
            )
        )
        corridor_rows.append(
            drivable_metrics(
                corridor.astype(np.bool_),
                target.astype(np.int64),
                output_stride=args.output_stride,
            )
        )
        if index % 25 == 0:
            print(f"  {index}/{len(images)}", flush=True)

    def summarise(rows: list[dict[str, float]]) -> dict[str, float]:
        return {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in rows[0]
            if isinstance(rows[0][key], int | float) and not isinstance(rows[0][key], bool)
        }

    record = {
        "schema_version": "1.0",
        "record_type": "edgeguard_drivable_area_evaluation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model_name or args.model.name,
        "model_sha256": sha256_file(args.model),
        "frames": len(road_rows),
        "road_mask": summarise(road_rows),
        "ego_corridor": summarise(corridor_rows),
        "output_stride": args.output_stride,
        "ignore_pixels_excluded": True,
        "scientific_measurement": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(record) + "\n", encoding="utf-8")

    print(f"\n{record['model']} · {len(road_rows)} kare")
    for name, summary in (
        ("yol maskesi", record["road_mask"]),
        ("ego koridor", record["ego_corridor"]),
    ):
        print(
            f"  {name:12s} IoU {summary['road_iou']:.4f} · sınır F1 "
            f"{summary['road_boundary_f1_tolerance_1px']:.4f} (1 px) / "
            f"{summary[f'road_boundary_f1_tolerance_{args.output_stride}px']:.4f} "
            f"({args.output_stride} px) · yanlış-sürülebilir "
            f"{summary['false_drivable_rate']:.4f}"
        )
    print(f"kayıt: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
