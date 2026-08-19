"""Measure both sides of reducing a model's logit stride before post-processing.

SegFormer-B0 emits logits at stride 4 (128x256) where the other architectures use stride 8
(64x128), and on the Jetson its frame is 193 ms slower than DDRNet's while its engine is
only 11 ms slower. The obvious conclusion -- that the gap is integration overhead and can
be recovered by downsampling the logits -- is a hypothesis, and it has two sides: how much
compute comes back, and how much accuracy it costs.

Both are measured here. Reporting only the speed-up would turn a trade into a free lunch.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from edgeguard.rescue.inference import predict_onnx
from edgeguard.rescue.perception import derive_perception
from edgeguard.rescue.visualization import confidence_entropy
from edgeguard.serialization import canonical_json, sha256_file

CLASS_COUNT = 19
IGNORE_LABEL = 255


def _mean_iou(matrix: np.ndarray) -> float:
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - np.diag(matrix)
    return float(np.nanmean(np.where(union > 0, intersection / np.maximum(union, 1), np.nan)))


def _downsample_logits(logits: np.ndarray, factor: int) -> np.ndarray:
    """Reduce logit resolution the way a stride-matched head would have emitted it."""
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(logits))[None]
    reduced = torch.nn.functional.interpolate(
        tensor,
        size=(max(1, logits.shape[1] // factor), max(1, logits.shape[2] // factor)),
        mode="bilinear",
        align_corners=False,
    )
    return reduced[0].numpy()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--timing-repeats", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    images = sorted(args.image_root.rglob("*_leftImg8bit.png"))[: args.frames]
    masks = {path.name: path for path in args.mask_root.rglob("*_gtFine_labelTrainIds.png")}
    if not images:
        raise ValueError(f"no frames under {args.image_root}")

    full = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    reduced = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    logit_shape: list[int] = []
    for index, image_path in enumerate(images, start=1):
        identifier = image_path.name[: -len("_leftImg8bit.png")]
        mask_path = masks.get(f"{identifier}_gtFine_labelTrainIds.png")
        if mask_path is None:
            continue
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        result = predict_onnx(image, args.model.resolve())
        if result.logits is None:
            raise RuntimeError("backend returned no logits")
        if not logit_shape:
            logit_shape = list(result.logits.shape)

        target = np.array(Image.open(mask_path))
        valid = target != IGNORE_LABEL
        flat = target[valid].astype(np.int64) * CLASS_COUNT
        full += np.bincount(
            flat + result.mask[valid].astype(np.int64), minlength=CLASS_COUNT**2
        ).reshape(CLASS_COUNT, CLASS_COUNT)

        small = np.argmax(_downsample_logits(result.logits, args.factor), axis=0).astype(np.uint8)
        upscaled = np.array(Image.fromarray(small).resize(image.size, Image.Resampling.NEAREST))
        reduced += np.bincount(
            flat + upscaled[valid].astype(np.int64), minlength=CLASS_COUNT**2
        ).reshape(CLASS_COUNT, CLASS_COUNT)
        if index % 20 == 0:
            print(f"  {index}/{len(images)}", flush=True)

    generator = np.random.default_rng(20260728)

    def time_postprocessing(height: int, width: int) -> float:
        logits = (generator.normal(size=(CLASS_COUNT, height, width)) * 3).astype(np.float32)
        coarse = np.argmax(logits[:, ::8, ::8], axis=0)
        mask = np.repeat(np.repeat(coarse, 8, 0), 8, 1)[:height, :width].astype(np.uint8)
        confidence, entropy = confidence_entropy(logits)
        derive_perception(mask, confidence, entropy)
        started = time.perf_counter()
        for _ in range(args.timing_repeats):
            confidence, entropy = confidence_entropy(logits)
            derive_perception(mask, confidence, entropy)
        return (time.perf_counter() - started) / args.timing_repeats * 1000.0

    height, width = logit_shape[1], logit_shape[2]
    at_full = time_postprocessing(height, width)
    at_reduced = time_postprocessing(max(1, height // args.factor), max(1, width // args.factor))

    record = {
        "schema_version": "1.0",
        "record_type": "edgeguard_logit_stride_tradeoff",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model_name or args.model.name,
        "model_sha256": sha256_file(args.model),
        "frames": len(images),
        "logit_shape": logit_shape,
        "downsample_factor": args.factor,
        "mIoU_full": _mean_iou(full),
        "mIoU_reduced": _mean_iou(reduced),
        "mIoU_delta": _mean_iou(reduced) - _mean_iou(full),
        "postprocess_ms_full": at_full,
        "postprocess_ms_reduced": at_reduced,
        "postprocess_speedup": at_full / at_reduced if at_reduced else None,
        # Timing is host-side and indicative of the ratio, not of Jetson wall clock; the
        # accuracy figures are exact for this split.
        "timing_host": "development machine, ratio only",
        "scientific_measurement": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(record) + "\n", encoding="utf-8")

    print(f"\n{record['model']} · logit {logit_shape[1]}x{logit_shape[2]} · {len(images)} kare")
    print(f"  mIoU tam        {record['mIoU_full']:.4f}")
    print(f"  mIoU indirgenmiş {record['mIoU_reduced']:.4f}   ({record['mIoU_delta']:+.4f})")
    print(
        f"  post-processing {at_full:.1f} ms -> {at_reduced:.1f} ms  ({at_full / at_reduced:.2f}x)"
    )
    print(f"kayıt: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
