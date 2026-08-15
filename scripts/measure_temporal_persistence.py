"""Measure what temporal persistence does to the contextual-risk ranking.

`temporal_persistence` is the seventh feature of the risk contract and the only one that
has never carried weight. Single-frame analysis cannot supply it, so `contextual_risk` is
given zero *weight* for it -- correct, but it leaves the feature declared and unmeasured,
and a thesis cannot claim a seven-feature fusion while one of the seven has never run.

A sequence supplies it. This drives consecutive frames through the same perception stack
the single-frame driver uses, associates regions across frames with
`TemporalPersistence` -- the real tracker, not the nearest-centroid proxy the one-frame
script falls back to -- and then scores every region twice: once with the feature at zero
weight, exactly as today, and once with it at its declared weight.

The comparison is the point. Two numbers come out of it: how much of what the system
flags is transient, and whether persistence would have demoted it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edgeguard.context import RiskWeights, contextual_risk
from edgeguard.evaluation.components import ComponentRecord
from edgeguard.rescue.inference import predict_onnx
from edgeguard.rescue.perception import derive_perception
from edgeguard.rescue.shift import uncertainty_maps
from edgeguard.rescue.visualization import resize_scalar
from edgeguard.serialization import canonical_json, sha256_file
from edgeguard.temporal import TemporalPersistence

# Same declared weights as `analyze_contextual_risk.py`; they are the contract, not a
# tunable, and the whole point here is to change only which of them is active.
DECLARED_WEIGHTS = {
    "anomaly_score": 0.30,
    "component_area": 0.10,
    "image_position": 0.15,
    "road_overlap": 0.15,
    "relative_proximity": 0.20,
    "detector_overlap": 0.05,
    "temporal_persistence": 0.05,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--minimum-region-area", type=int, default=64)
    parser.add_argument(
        "--persistence-saturation",
        type=int,
        default=5,
        help=(
            "frames a track must survive for the feature to reach 1.0. Declared, not "
            "fitted: at ~17 FPS this is roughly a third of a second of agreement."
        ),
    )
    parser.add_argument("--area-saturation-fraction", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _normalise_anomaly(energy: np.ndarray) -> np.ndarray:
    low, high = np.percentile(energy, 5), np.percentile(energy, 95)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(energy, dtype=np.float64)
    return np.clip((energy - low) / (high - low), 0.0, 1.0)


def _road_adjacency(region: np.ndarray, road: np.ndarray) -> float:
    grown = region.copy()
    grown[1:, :] |= region[:-1, :]
    grown[:-1, :] |= region[1:, :]
    grown[:, 1:] |= region[:, :-1]
    grown[:, :-1] |= region[:, 1:]
    border = grown & ~region
    total = int(np.count_nonzero(border))
    return float(np.count_nonzero(border & road) / total) if total else 0.0


def main() -> int:
    args = _parser().parse_args()
    images = sorted(args.image_root.rglob("*_leftImg8bit.png"))[: args.frames]
    if not images:
        raise ValueError(f"no frames under {args.image_root}")

    tracker = TemporalPersistence(missed_frame_tolerance=1)
    sequence = args.image_root.name
    # There is no detector in this pipeline, so `detector_overlap` is zero-weighted in
    # *both* scorings. Leaving it weighted with a zero value would drag every score down
    # equally, which is the exact error zero-weighting exists to prevent -- and here it
    # would also be noise on top of the one difference being measured.
    active = {**DECLARED_WEIGHTS, "detector_overlap": 0.0}
    with_weight = RiskWeights(**active)
    without = RiskWeights(**{**active, "temporal_persistence": 0.0})

    lifetimes: dict[int, int] = {}
    first_seen: dict[int, int] = {}
    rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(images):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        result = predict_onnx(image, args.model.resolve())
        if result.logits is None:
            raise RuntimeError("backend returned no logits, so no anomaly score can be formed")
        energy = uncertainty_maps(result.logits)["energy"]
        if energy.shape != result.mask.shape:
            # Energy comes back at logit resolution; the regions are at image
            # resolution. Same upsampling the single-frame driver uses.
            energy = resize_scalar(energy, image.size)
        anomaly = _normalise_anomaly(energy)
        perception = derive_perception(
            result.mask,
            result.confidence,
            result.entropy,
            minimum_region_area=args.minimum_region_area,
        )
        height, width = result.mask.shape
        saturation = max(1.0, args.area_saturation_fraction * height * width)
        road = perception.road_mask.astype(np.bool_)

        components = []
        for region in perception.regions:
            mask = region.mask.astype(np.bool_)
            components.append(
                ComponentRecord(
                    component_id=region.region_id,
                    area=region.area_pixels,
                    bbox_xyxy=region.bbox_xyxy,
                    centroid_xy=region.centroid_xy,
                    mean_score=float(np.mean(anomaly[mask])),
                    max_score=float(np.max(anomaly[mask])),
                    road_overlap=_road_adjacency(mask, road),
                )
            )
        tracked = {
            record["component_id"]: record
            for record in tracker.update(sequence, index, tuple(components))
        }

        for region in perception.regions:
            mask = region.mask.astype(np.bool_)
            track = tracked[region.region_id]
            track_id = int(track["track_id"])
            lifetimes[track_id] = int(track["persistence_count"])
            first_seen.setdefault(track_id, index)
            distance = region.corridor_distance_pixels
            features = {
                "anomaly_score": float(np.mean(anomaly[mask])),
                "component_area": float(np.clip(region.area_pixels / saturation, 0.0, 1.0)),
                "image_position": float(region.bbox_xyxy[3] - 1) / max(1.0, height - 1),
                "road_overlap": _road_adjacency(mask, road),
                "relative_proximity": (
                    0.0
                    if distance is None or not np.isfinite(distance)
                    else float(np.clip(1.0 - distance / max(1.0, height * 0.25), 0.0, 1.0))
                ),
                # No detector in this pipeline; held at zero weight in both scorings so it
                # cannot contribute to the difference being measured.
                "detector_overlap": 0.0,
                "temporal_persistence": float(
                    min(track["persistence_count"] / args.persistence_saturation, 1.0)
                ),
            }
            baseline = contextual_risk({**features, "temporal_persistence": 0.0}, without)
            informed = contextual_risk(features, with_weight)
            rows.append(
                {
                    "frame_index": index,
                    "track_id": int(track["track_id"]),
                    "class_name": region.class_name,
                    "event": track["event"],
                    "persistence_count": int(track["persistence_count"]),
                    "risk_without_persistence": float(baseline["total_risk_score"]),
                    "risk_with_persistence": float(informed["total_risk_score"]),
                    "category_without": baseline["risk_category"],
                    "category_with": informed["risk_category"],
                }
            )
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(images)}", flush=True)

    # A track that first appears in the last few frames cannot accumulate lifetime, so
    # counting it as transient would measure where the sequence was cut, not flicker.
    fair_deadline = len(images) - args.persistence_saturation
    judged = [track for track, start in first_seen.items() if start < fair_deadline]
    transient = [track for track in judged if lifetimes[track] <= 1]
    # Per frame, does adding persistence change which region is ranked first?
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_frame.setdefault(int(row["frame_index"]), []).append(row)
    leader_changes = 0
    for frame_rows in by_frame.values():
        without_leader = max(frame_rows, key=lambda r: r["risk_without_persistence"])["track_id"]
        with_leader = max(frame_rows, key=lambda r: r["risk_with_persistence"])["track_id"]
        leader_changes += int(without_leader != with_leader)

    demoted = [
        row for row in rows if row["category_without"] == "high" and row["category_with"] != "high"
    ]
    promoted = [
        row for row in rows if row["category_with"] == "high" and row["category_without"] != "high"
    ]

    record = {
        "schema_version": "1.0",
        "record_type": "edgeguard_temporal_persistence_measurement",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model_name or args.model.name,
        "model_sha256": sha256_file(args.model),
        "sequence": sequence,
        "frames": len(images),
        "region_observations": len(rows),
        "tracks": len(lifetimes),
        "tracks_with_a_fair_chance": len(judged),
        "transient_tracks": len(transient),
        "transient_track_fraction": len(transient) / max(1, len(judged)),
        "tracks_excluded_as_born_too_late": len(lifetimes) - len(judged),
        "mean_track_lifetime_frames": (
            float(np.mean([lifetimes[t] for t in judged])) if judged else 0.0
        ),
        "median_track_lifetime_frames": (
            float(np.median([lifetimes[t] for t in judged])) if judged else 0.0
        ),
        "persistence_saturation_frames": args.persistence_saturation,
        "frames_whose_top_region_changed": leader_changes,
        "top_region_change_fraction": leader_changes / max(1, len(by_frame)),
        "high_to_lower_observations": len(demoted),
        "lower_to_high_observations": len(promoted),
        "declared_weights": DECLARED_WEIGHTS,
        "detector_overlap_zero_weighted_in_both": True,
        "scientific_measurement": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(record) + "\n", encoding="utf-8")

    print(f"\n{record['model']} · {sequence} · {len(images)} kare")
    print(f"  bölge gözlemi          {record['region_observations']}")
    print(
        f"  iz (track)             {record['tracks']}"
        f"  (adil değerlendirilen: {record['tracks_with_a_fair_chance']})"
    )
    print(
        f"  tek karelik iz         {record['transient_tracks']} "
        f"({record['transient_track_fraction'] * 100:.1f}%)"
    )
    print(f"  ortalama iz ömrü       {record['mean_track_lifetime_frames']:.2f} kare")
    print(
        f"  1. sırası değişen kare {leader_changes} "
        f"({record['top_region_change_fraction'] * 100:.1f}%)"
    )
    print(f"  high -> daha düşük     {len(demoted)}")
    print(f"  daha düşük -> high     {len(promoted)}")
    print(f"kayıt: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
