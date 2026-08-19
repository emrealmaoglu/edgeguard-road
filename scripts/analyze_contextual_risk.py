"""Rank one frame's regions by explainable contextual risk.

`edgeguard.context.risk.contextual_risk` fuses seven weighted features into an
operational risk category with a per-feature explanation. It has had no callers: the
features it needs were spread across the segmentation mask, the perception regions and
the uncertainty maps, and nothing assembled them. This driver does.

Features that this frame cannot supply -- `detector_overlap` without a detector,
`temporal_persistence` without a previous frame -- are given **zero weight** rather than a
zero value. `contextual_risk` normalises by the summed weight, so a zero-valued feature
would quietly drag every score down and make an unmeasured signal look like a measured
absence of risk. Zero weight instead renormalises over the features actually present, and
the record names both sets.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edgeguard.context import RiskWeights, contextual_risk
from edgeguard.detection.contracts import Detection, box_mask_overlap
from edgeguard.rescue.inference import predict_mmseg, predict_onnx
from edgeguard.rescue.perception import derive_perception
from edgeguard.rescue.shift import uncertainty_maps
from edgeguard.rescue.visualization import resize_scalar
from edgeguard.serialization import canonical_json, sha256_file

# Weights when every feature is available. Not scientifically tuned -- `RiskWeights`
# says so itself -- but declared once here so a record can be reproduced from it.
DECLARED_WEIGHTS = {
    "anomaly_score": 0.30,
    "component_area": 0.10,
    "image_position": 0.15,
    "road_overlap": 0.15,
    "relative_proximity": 0.20,
    "detector_overlap": 0.05,
    "temporal_persistence": 0.05,
}


def _normalise_anomaly(energy: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Map an energy map onto [0,1] using this frame's own spread.

    Energy is unbounded and its scale shifts per architecture and per scene, so a fixed
    cut would not transfer. The 5th/95th percentiles of the frame give a stable, stated
    reference; the bounds travel in the record so a score can be traced back.
    """
    low = float(np.percentile(energy, 5))
    high = float(np.percentile(energy, 95))
    span = high - low
    if span <= 0.0:
        return np.zeros_like(energy, dtype=np.float32), {"low": low, "high": high}
    scaled = np.clip((energy - low) / span, 0.0, 1.0)
    return scaled.astype(np.float32), {"low": low, "high": high}


def _road_adjacency(region_mask: np.ndarray, road_mask: np.ndarray) -> float:
    """Fraction of the region's four-neighbourhood that is road.

    An obstacle sitting on the carriageway matters more than one on a wall, and touching
    is what distinguishes them -- so this measures contact, not containment.
    """
    grown = region_mask.copy()
    grown[1:, :] |= region_mask[:-1, :]
    grown[:-1, :] |= region_mask[1:, :]
    grown[:, 1:] |= region_mask[:, :-1]
    grown[:, :-1] |= region_mask[:, 1:]
    border = grown & ~region_mask
    if not border.any():
        return 0.0
    return float(np.count_nonzero(border & road_mask) / np.count_nonzero(border))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help=".onnx graph or .pth checkpoint")
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=1024)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--entropy-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-region-area", type=int, default=8)
    parser.add_argument(
        "--area-saturation-fraction",
        type=float,
        default=0.02,
        help="region area, as a fraction of the frame, at which component_area reaches 1.0",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        help=(
            "optional JSON list of {class_id, score, xyxy} boxes in image pixels. Without "
            "it `detector_overlap` carries zero weight instead of a zero value."
        ),
    )
    parser.add_argument(
        "--previous-regions",
        type=Path,
        help=(
            "optional contextual-risk record for the preceding frame. Without it "
            "`temporal_persistence` carries zero weight instead of a zero value."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_detections(path: Path | None) -> list[Detection] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Detection(
            class_id=int(row["class_id"]),
            score=float(row["score"]),
            xyxy=tuple(float(value) for value in row["xyxy"]),  # type: ignore[arg-type]
        )
        for row in payload
    ]


def _previous_centroids(path: Path | None) -> list[tuple[float, float, int]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        (float(row["centroid_xy"][0]), float(row["centroid_xy"][1]), int(row["class_id"]))
        for row in payload.get("regions", [])
    ]


def main() -> int:
    args = _parser().parse_args()
    with Image.open(args.image) as opened:
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
        raise RuntimeError("backend returned no logits, so no anomaly score can be formed")

    energy = uncertainty_maps(result.logits)["energy"]
    if energy.shape != result.mask.shape:
        energy = resize_scalar(energy, image.size)
    anomaly, anomaly_bounds = _normalise_anomaly(energy)

    perception = derive_perception(
        result.mask,
        result.confidence,
        result.entropy,
        confidence_threshold=args.confidence_threshold,
        entropy_threshold=args.entropy_threshold,
        minimum_region_area=args.minimum_region_area,
    )

    detections = _load_detections(args.detections)
    previous = _previous_centroids(args.previous_regions)
    weights = dict(DECLARED_WEIGHTS)
    absent: list[str] = []
    if detections is None:
        weights["detector_overlap"] = 0.0
        absent.append("detector_overlap")
    if not previous:
        weights["temporal_persistence"] = 0.0
        absent.append("temporal_persistence")
    risk_weights = RiskWeights(**weights)

    height, width = result.mask.shape
    saturation = max(1.0, args.area_saturation_fraction * height * width)
    # A region is "the same" as one in the previous frame when the nearest same-class
    # centroid is within this radius; frame-to-frame motion is small next to region size.
    persistence_radius = 0.05 * float(np.hypot(height, width))

    rows: list[dict[str, Any]] = []
    for region in perception.regions:
        region_mask = region.mask.astype(np.bool_)
        centroid_x, centroid_y = region.centroid_xy

        detector_overlap = 0.0
        if detections is not None:
            detector_overlap = max(
                (box_mask_overlap(detection, region_mask) for detection in detections),
                default=0.0,
            )

        temporal_persistence = 0.0
        if previous:
            same_class = [row for row in previous if row[2] == region.class_id]
            if same_class:
                nearest = min(
                    float(np.hypot(row[0] - centroid_x, row[1] - centroid_y)) for row in same_class
                )
                temporal_persistence = float(np.clip(1.0 - nearest / persistence_radius, 0.0, 1.0))

        distance = region.corridor_distance_pixels
        features = {
            "anomaly_score": float(np.mean(anomaly[region_mask])),
            "component_area": float(np.clip(region.area_pixels / saturation, 0.0, 1.0)),
            "image_position": float(region.bbox_xyxy[3] - 1) / max(1.0, height - 1),
            "road_overlap": _road_adjacency(region_mask, perception.road_mask.astype(np.bool_)),
            "relative_proximity": (
                0.0
                if distance is None or not np.isfinite(distance)
                else float(np.clip(1.0 - distance / max(1.0, height * 0.25), 0.0, 1.0))
            ),
            "detector_overlap": detector_overlap,
            "temporal_persistence": temporal_persistence,
        }
        assessment = contextual_risk(features, risk_weights)
        rows.append(
            {
                "region_id": region.region_id,
                "class_id": region.class_id,
                "class_name": region.class_name,
                "bbox_xyxy": list(region.bbox_xyxy),
                "centroid_xy": [centroid_x, centroid_y],
                "area_pixels": region.area_pixels,
                "corridor_distance_pixels": distance,
                "risk": assessment,
            }
        )

    rows.sort(key=lambda row: -float(row["risk"]["total_risk_score"]))
    record = {
        "schema_version": "1.0",
        "record_type": "edgeguard_contextual_risk_frame",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "image": str(args.image),
        "model": args.model.name,
        "model_sha256": sha256_file(args.model),
        "backend": result.backend,
        "declared_weights": DECLARED_WEIGHTS,
        "applied_weights": weights,
        "zero_weighted_features": absent,
        "anomaly_normalisation": {"method": "frame_percentile_5_95", **anomaly_bounds},
        "area_saturation_pixels": saturation,
        "persistence_radius_pixels": persistence_radius,
        "regions": rows,
        # The fusion is an ordered operational attention signal, not a probability that
        # anything physical will happen; `contextual_risk` states the same in every row.
        "calibrated_physical_risk_probability": False,
        "scientific_measurement": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(record) + "\n", encoding="utf-8")

    print(f"{'sıra':>4} {'sınıf':14s} {'risk':>6} {'seviye':8s} {'baskın etken':22s}")
    for index, row in enumerate(rows[:12], start=1):
        risk = row["risk"]
        print(
            f"{index:4d} {row['class_name']:14s} {risk['total_risk_score']:6.3f} "
            f"{risk['risk_category']:8s} {risk['explanation'][0]:22s}"
        )
    if absent:
        print(f"\nsıfır ağırlıklı (ölçülemedi): {', '.join(absent)}")
    print(f"kayıt: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
