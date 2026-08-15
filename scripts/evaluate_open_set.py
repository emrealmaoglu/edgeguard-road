"""Measure pixel-level open-set road-hazard detection against real anomaly labels.

`edgeguard.evaluation.ood` has carried a complete, tested metric suite (AUROC/AP/FPR95,
threshold selection and policies, bootstrap intervals, per-source breakdown) whose only
callers fed it synthetic random logits. This driver connects it to real data: a road
anomaly set whose pixels are labelled background/anomaly, scored by the uncertainty maps
the segmentation model already produces.

The model is never retrained or fine-tuned for this. Anomalies here are objects outside
the 19 Cityscapes training classes -- animals, rocks, cones, debris -- so a closed-set
segmentation model has genuinely never seen them, which is exactly the open-set condition
the metric is meant to characterise.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edgeguard.evaluation.ood import (
    bootstrap_ood_metrics,
    per_source_ood_metrics,
    pixel_ood_metrics,
    score_distribution,
    select_anomaly_threshold,
    threshold_policies,
)
from edgeguard.rescue.inference import predict_mmseg, predict_onnx
from edgeguard.rescue.shift import uncertainty_maps
from edgeguard.rescue.visualization import resize_scalar
from edgeguard.serialization import canonical_json, sha256_file

# RoadAnomaly ships `labels_semantic.png` with background 0 and anomaly 2; the metric
# suite's contract is ID 0 / anomaly 1 / ignore 255.
DATASET_ANOMALY_VALUE = 2
DATASET_BACKGROUND_VALUE = 0

# Each map's sign convention, normalised so that a higher score always means "more
# anomalous" -- the direction `pixel_ood_metrics` documents and validates against.
SCORE_DIRECTIONS: dict[str, float] = {
    "energy": 1.0,  # energy is -logsumexp, so it already rises on unfamiliar input
    "normalized_entropy": 1.0,
    "maximum_softmax_probability": -1.0,  # confidence falls as anomaly likelihood rises
    "maximum_logit": -1.0,
}

# Frames are named `<category><index>_<free text>`; the leading word is the hazard family
# (animals, obstacles, cones, ...), which makes a per-hazard breakdown free.
_CATEGORY = re.compile(r"^([a-zA-Z]+)")


def hazard_category(frame_name: str) -> str:
    match = _CATEGORY.match(frame_name)
    return match.group(1).lower() if match else "unknown"


def discover_frames(dataset_root: Path) -> list[tuple[str, Path, Path]]:
    """Pair every image with its per-pixel anomaly mask."""
    frames_root = dataset_root / "frames"
    if not frames_root.is_dir():
        raise ValueError(f"road anomaly root must contain frames/: {dataset_root}")
    discovered: list[tuple[str, Path, Path]] = []
    for label_dir in sorted(frames_root.glob("*.labels")):
        mask = label_dir / "labels_semantic.png"
        if not mask.is_file():
            continue
        stem = label_dir.name[: -len(".labels")]
        image = next(
            (
                candidate
                for suffix in (".jpg", ".jpeg", ".png", ".webp")
                if (candidate := frames_root / f"{stem}{suffix}").is_file()
            ),
            None,
        )
        if image is not None:
            discovered.append((stem, image, mask))
    if not discovered:
        raise ValueError(f"no image/mask pairs found under {frames_root}")
    return discovered


def load_anomaly_labels(mask_path: Path) -> np.ndarray:
    """Map the dataset's own encoding onto the metric suite's ID/anomaly contract."""
    with Image.open(mask_path) as opened:
        raw = np.array(opened)
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    labels = np.full(raw.shape, 255, dtype=np.int64)
    labels[raw == DATASET_BACKGROUND_VALUE] = 0
    labels[raw == DATASET_ANOMALY_VALUE] = 1
    return labels


def frame_scores(logits: np.ndarray, target_size: tuple[int, int]) -> dict[str, np.ndarray]:
    """Return every uncertainty map at label resolution, oriented anomaly-positive."""
    maps = uncertainty_maps(logits)
    scores: dict[str, np.ndarray] = {}
    for name, sign in SCORE_DIRECTIONS.items():
        value = maps[name]
        if value.shape != (target_size[1], target_size[0]):
            value = resize_scalar(value, target_size)
        scores[name] = (sign * value.astype(np.float64)).astype(np.float32)
    return scores


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help=".onnx graph or .pth checkpoint")
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--model-name", help="label used in the record; defaults to the filename")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=1024)
    parser.add_argument(
        "--pixels-per-frame",
        type=int,
        default=100_000,
        help=(
            "deterministic per-frame pixel subsample. Every frame contributes equally and "
            "the retained count is recorded, so the metrics stay reproducible without "
            "holding 55M pixels per score map in memory."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--target-tpr", type=float, default=0.95)
    parser.add_argument("--risk-budget-fpr", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    frames = discover_frames(dataset_root)
    generator = np.random.default_rng(args.seed)

    collected: dict[str, list[np.ndarray]] = {name: [] for name in SCORE_DIRECTIONS}
    labels_collected: list[np.ndarray] = []
    sources_collected: list[np.ndarray] = []
    per_frame: list[dict[str, Any]] = []

    for index, (stem, image_path, mask_path) in enumerate(frames, start=1):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        if args.model.suffix.lower() == ".onnx":
            result = predict_onnx(
                image, args.model.resolve(), input_size=(args.input_height, args.input_width)
            )
        else:
            if args.resolved_config is None:
                raise ValueError("a PyTorch checkpoint requires --resolved-config")
            result = predict_mmseg(
                image, args.resolved_config.resolve(), args.model.resolve(), device=args.device
            )
        if result.logits is None:
            raise RuntimeError(f"backend returned no logits for {stem}")

        labels = load_anomaly_labels(mask_path)
        if labels.shape != (image.height, image.width):
            raise ValueError(
                f"{stem}: mask {labels.shape} does not match image {(image.height, image.width)}"
            )
        scores = frame_scores(result.logits, (image.width, image.height))

        flat_labels = labels.reshape(-1)
        keep = np.flatnonzero(flat_labels != 255)
        if keep.size > args.pixels_per_frame:
            keep = generator.choice(keep, size=args.pixels_per_frame, replace=False)
            keep.sort()
        labels_collected.append(flat_labels[keep])
        # A bare `np.str_` dtype resolves to `<U1` and silently truncates every category
        # to its first letter, so "animals"/"obstacles" both collapse into useless keys.
        sources_collected.append(np.full(keep.size, hazard_category(stem), dtype="<U32"))
        for name, value in scores.items():
            collected[name].append(value.reshape(-1)[keep])

        anomaly_pixels = int(np.count_nonzero(flat_labels[keep] == 1))
        per_frame.append(
            {
                "frame": stem,
                "hazard_category": hazard_category(stem),
                "sampled_pixels": int(keep.size),
                "anomaly_pixels": anomaly_pixels,
                "anomaly_ratio": anomaly_pixels / keep.size if keep.size else None,
                "inference_latency_ms": result.latency_ms,
            }
        )
        print(
            f"  [{index:>3}/{len(frames)}] {stem[:48]:48s} anomali={anomaly_pixels:>6}/{keep.size}",
            flush=True,
        )

    labels_all = np.concatenate(labels_collected)
    sources_all = np.concatenate(sources_collected)

    results: dict[str, Any] = {}
    for name in SCORE_DIRECTIONS:
        scores_all = np.concatenate(collected[name]).astype(np.float32)
        metrics = pixel_ood_metrics(scores_all, labels_all)
        entry: dict[str, Any] = {"metrics": metrics.__dict__}
        entry["distribution"] = score_distribution(scores_all, labels_all)
        entry["threshold_at_target_tpr"] = select_anomaly_threshold(
            scores_all, labels_all, target_tpr=args.target_tpr
        )
        entry["threshold_policies"] = threshold_policies(
            scores_all,
            labels_all,
            fixed_threshold=float(np.median(scores_all)),
            risk_budget_fpr=args.risk_budget_fpr,
        )
        entry["bootstrap"] = bootstrap_ood_metrics(
            scores_all, labels_all, resamples=args.bootstrap_resamples, seed=args.seed
        )
        entry["per_hazard_category"] = per_source_ood_metrics(scores_all, labels_all, sources_all)
        results[name] = entry
        auroc = metrics.auroc
        average_precision = metrics.average_precision
        fpr = metrics.fpr_at_95_tpr
        print(
            f"{name:32s} AUROC={auroc:.4f}  AP={average_precision:.4f}  FPR95={fpr:.4f}"
            if auroc is not None and average_precision is not None and fpr is not None
            else f"{name:32s} tanımsız (tek sınıf)",
            flush=True,
        )

    record = {
        "schema_version": "1.0",
        "record_type": "edgeguard_open_set_pixel_evaluation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "dataset_frame_count": len(frames),
        "model": args.model_name or args.model.name,
        "model_sha256": sha256_file(args.model),
        "resolved_config": str(args.resolved_config) if args.resolved_config else None,
        "backend_device": args.device,
        "pixels_per_frame": args.pixels_per_frame,
        "seed": args.seed,
        "sampled_pixel_count": int(labels_all.size),
        "anomaly_pixel_count": int(np.count_nonzero(labels_all == 1)),
        "score_direction": "higher_means_more_anomalous",
        "scores": results,
        "frames": per_frame,
        # The model was never trained, fine-tuned or thresholded on this data; every
        # anomaly here is an object class absent from the 19 training classes.
        "model_trained_on_this_data": False,
        "scientific_status": "measured",
        "accepted_release": False,
        "sealed_test_data_opened": False,
    }
    (output_dir / "open_set_evaluation.json").write_text(
        canonical_json(record) + "\n", encoding="utf-8"
    )
    print(f"\nKayıt: {output_dir / 'open_set_evaluation.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
