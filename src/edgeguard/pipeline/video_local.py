"""Full real-codepath synthetic video perception pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from edgeguard.context import RiskWeights, contextual_risk
from edgeguard.demo.video_dashboard import headless_dashboard_smoke, render_evidence_frame
from edgeguard.detection.models import _build_yolo
from edgeguard.evaluation.components import connected_components
from edgeguard.models.semantic_local import _build_model, _native_logits_and_features
from edgeguard.scoring.uncertainty import (
    energy_anomaly_score,
    max_logit_anomaly_score,
    msp_anomaly_score,
    predictive_entropy,
)
from edgeguard.serialization import sha256_file, sha256_payload
from edgeguard.temporal import TemporalPersistence
from edgeguard.training.config import load_semantic_model_suite


def run_synthetic_video_pipeline(
    config_root: Path,
    mmseg_checkout: Path,
    output_root: Path,
    *,
    frame_count: int = 6,
    detector_failure_frame: int = 3,
) -> dict[str, Any]:
    """Run semantic, OOD, detector, temporal, risk, overlay, and restart paths."""
    if not 5 <= frame_count <= 10:
        raise ValueError("synthetic video validation requires 5..10 frames")
    torch = __import__("torch")
    from ultralytics.utils.ops import non_max_suppression

    output_root.mkdir(parents=True, exist_ok=False)
    overlay_root = output_root / "overlays"
    overlay_root.mkdir()
    spec = next(
        item
        for item in load_semantic_model_suite(config_root)
        if item.model_family.value == "fast_scnn"
    )
    torch.manual_seed(20260727)
    semantic = _build_model(spec, mmseg_checkout, torch).eval()
    detector = _build_yolo(torch).eval()
    generator = np.random.default_rng(20260727)
    frames = generator.integers(0, 256, size=(frame_count, 128, 256, 3), dtype=np.uint8)
    tracker = TemporalPersistence(missed_frame_tolerance=2)
    weights = RiskWeights(1, 1, 1, 1, 0, 1, 1)
    anomaly_head: Any = None
    anomaly_optimizer: Any = None
    report_frames: list[dict[str, Any]] = []
    detector_failures: list[dict[str, Any]] = []
    snapshot: dict[str, Any] | None = None
    frame_indices = list(range(frame_count))
    if frame_count > 4:
        frame_indices[4:] = [value + 1 for value in frame_indices[4:]]
    for position, (frame_index, frame) in enumerate(zip(frame_indices, frames, strict=True)):
        inputs = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            native, feature, feature_shapes = _native_logits_and_features(semantic, inputs, torch)
            aligned = torch.nn.functional.interpolate(
                native, size=(128, 256), mode="bilinear", align_corners=False
            )
        if anomaly_head is None:
            anomaly_head = torch.nn.Sequential(
                torch.nn.Conv2d(int(feature.shape[1]), 16, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 1, 1),
            )
            anomaly_optimizer = torch.optim.AdamW(anomaly_head.parameters(), lr=0.001)
            target = torch.zeros((1, 128, 256))
            target[:, 52:76, 100:140] = 1
            for _step in range(2):
                feature_logits = torch.nn.functional.interpolate(
                    anomaly_head(feature.detach()),
                    size=(128, 256),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(feature_logits, target)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("video feature-head loss is non-finite")
                anomaly_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                anomaly_optimizer.step()
        with torch.no_grad():
            feature_score = torch.sigmoid(
                torch.nn.functional.interpolate(
                    anomaly_head(feature),
                    size=(128, 256),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
            ).numpy()
        logits = aligned.numpy().astype(np.float32)
        semantic_mask = logits.argmax(axis=1)[0].astype(np.uint8)
        scores = {
            "msp": msp_anomaly_score(logits)[0],
            "entropy": predictive_entropy(logits)[0],
            "max_logit": max_logit_anomaly_score(logits)[0],
            "energy": energy_anomaly_score(logits)[0],
            "feature_head": feature_score[0].astype(np.float32),
        }
        threshold = float(np.quantile(scores["msp"], 0.9))
        road_mask = np.zeros((128, 256), dtype=np.bool_)
        road_mask[64:, :] = True
        components = connected_components(
            np.asarray(scores["msp"] >= threshold), scores["msp"], road_mask=road_mask
        )
        detections: list[dict[str, Any]] = []
        if position == detector_failure_frame:
            detector_failures.append(
                {
                    "frame_index": frame_index,
                    "classification": "bounded_detector_failure_injection",
                    "campaign_continued": True,
                }
            )
        else:
            with torch.no_grad():
                prediction = detector(
                    torch.nn.functional.interpolate(inputs, size=(128, 128), mode="bilinear")
                )[0]
            decoded = non_max_suppression(prediction, conf_thres=0.001, max_det=3)[0]
            for row in decoded:
                x1, y1, x2, y2, confidence, class_id = [float(value) for value in row]
                detections.append(
                    {
                        "box_xyxy": [2 * x1, y1, 2 * x2, y2],
                        "confidence": confidence,
                        "class_id": int(class_id),
                    }
                )
        tracks = list(tracker.update("synthetic-sequence", frame_index, components))
        if position == 2:
            snapshot = tracker.snapshot()
            recovered = TemporalPersistence(missed_frame_tolerance=2)
            recovered.restore(snapshot)
            tracker = recovered
        primary = components[0] if components else None
        persistence = max((int(item["persistence_count"]) for item in tracks), default=0)
        risk = contextual_risk(
            {
                "anomaly_score": min(1.0, float(primary.mean_score) if primary else 0.0),
                "component_area": min(1.0, (primary.area / (128 * 256)) if primary else 0.0),
                "image_position": 0.5,
                "road_overlap": float(primary.road_overlap) if primary else 0.0,
                "relative_proximity": 0.0,
                "detector_overlap": 1.0 if detections else 0.0,
                "temporal_persistence": min(1.0, persistence / 3),
            },
            weights,
        )
        overlay = render_evidence_frame(
            frame,
            semantic_mask,
            scores["msp"],
            components=[item.to_dict() for item in components],
            detections=detections,
            tracks=tracks,
            risk=risk,
        )
        overlay_path = overlay_root / f"frame-{frame_index:03d}.png"
        overlay.save(overlay_path, format="PNG", optimize=True)
        lineage = {
            "input_frame_sha256": sha256_payload(frame.tolist()),
            "native_logits_sha256": sha256_payload(native.detach().numpy().tolist()),
            "scores_sha256": sha256_payload(
                {name: sha256_payload(value.tolist()) for name, value in scores.items()}
            ),
            "components_sha256": sha256_payload([item.to_dict() for item in components]),
            "tracks_sha256": sha256_payload(tracks),
            "risk_sha256": sha256_payload(risk),
            "overlay_sha256": sha256_file(overlay_path),
        }
        report_frames.append(
            {
                "frame_index": frame_index,
                "native_logits_shape": list(native.shape),
                "aligned_logits_shape": list(aligned.shape),
                "feature_candidate_shapes": feature_shapes,
                "score_shapes": {name: list(value.shape) for name, value in scores.items()},
                "components": [item.to_dict() for item in components],
                "detections": detections,
                "tracks": tracks,
                "risk": risk,
                "overlay_relative_path": overlay_path.relative_to(output_root).as_posix(),
                "lineage": lineage,
            }
        )
    anomaly_checkpoint = output_root / "anomaly_head.pt"
    torch.save(
        {
            "model": anomaly_head.state_dict(),
            "optimizer": anomaly_optimizer.state_dict(),
            "identity": sha256_payload({"model": "fast_scnn", "features": "decode_input"}),
        },
        anomaly_checkpoint,
    )
    report = {
        "schema_version": "1.0",
        "record_type": "full_real_codepath_synthetic_video",
        "profile": "local_synthetic_validation",
        "semantic_model": "fast_scnn_random_weight_actual_mmseg",
        "detector_model": "yolo11n_random_weight_actual_ultralytics",
        "frames": report_frames,
        "detector_failures": detector_failures,
        "temporal_recovery_snapshot": snapshot,
        "anomaly_checkpoint_sha256": sha256_file(anomaly_checkpoint),
        "latency_includes_ui": False,
        "scientific_evidence": False,
    }
    report_path = output_root / "report.json"
    report_path.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    report["dashboard_smoke"] = headless_dashboard_smoke(
        overlay_root / f"frame-{frame_indices[0]:03d}.png",
        {
            "profile": report["profile"],
            "model_identity": report["semantic_model"],
            "frame_count": frame_count,
            "latency_includes_ui": False,
        },
    )
    report_path.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return report
