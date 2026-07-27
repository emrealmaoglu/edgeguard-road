"""Prerecorded-frame visualization with an optional thin Streamlit surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw


def render_evidence_frame(
    image: npt.NDArray[np.uint8],
    semantic_mask: npt.NDArray[np.integer],
    uncertainty: npt.NDArray[np.floating],
    *,
    components: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    risk: dict[str, Any],
) -> Image.Image:
    """Render one deterministic local-validation overlay outside latency timing."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("dashboard image must be HWC uint8 RGB")
    if semantic_mask.shape != image.shape[:2] or uncertainty.shape != image.shape[:2]:
        raise ValueError("dashboard masks must match the image geometry")
    base = Image.fromarray(image).convert("RGBA")
    semantic = np.zeros((*semantic_mask.shape, 4), dtype=np.uint8)
    semantic[..., 1] = np.asarray(semantic_mask % 19, dtype=np.uint8) * 11
    semantic[..., 3] = 55
    base.alpha_composite(Image.fromarray(semantic, mode="RGBA"))
    normalized = np.asarray(uncertainty, dtype=np.float32)
    normalized = (normalized - normalized.min()) / max(float(np.ptp(normalized)), 1e-6)
    heat = np.zeros((*uncertainty.shape, 4), dtype=np.uint8)
    heat[..., 0] = np.asarray(normalized * 255, dtype=np.uint8)
    heat[..., 3] = np.asarray(normalized * 100, dtype=np.uint8)
    base.alpha_composite(Image.fromarray(heat, mode="RGBA"))
    draw = ImageDraw.Draw(base)
    for component in components:
        draw.rectangle(tuple(component["bbox_xyxy"]), outline=(255, 200, 0, 255), width=2)
    for detection in detections:
        draw.rectangle(tuple(detection["box_xyxy"]), outline=(0, 220, 255, 255), width=2)
    for track in tracks:
        draw.text((4, 4 + 12 * int(track["track_id"])), f"T{track['track_id']}", fill="white")
    draw.text(
        (4, image.shape[0] - 14),
        f"risk={risk['risk_category']} {risk['total_risk_score']:.3f}",
        fill="white",
    )
    return base.convert("RGB")


def headless_dashboard_smoke(frame_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate dashboard artifacts without starting a web server."""
    if not frame_path.is_file():
        raise ValueError("dashboard smoke requires one rendered frame")
    with Image.open(frame_path) as image:
        image.verify()
    required = {"profile", "model_identity", "frame_count", "latency_includes_ui"}
    if not required <= set(payload) or payload["latency_includes_ui"] is not False:
        raise ValueError("dashboard payload contract is incomplete")
    return {
        "status": "passed",
        "frame_filename": frame_path.name,
        "profile": payload["profile"],
        "model_identity": payload["model_identity"],
        "frame_count": payload["frame_count"],
        "real_time_claim": False,
    }


def run_streamlit(report_path: Path) -> None:
    """Render an existing prerecorded report when Streamlit is explicitly launched."""
    import json

    import streamlit as st

    report = json.loads(report_path.read_text(encoding="utf-8"))
    st.title("EdgeGuard-Road prerecorded pipeline evidence")
    st.caption("NON-SCIENTIFIC PIPELINE VALIDATION — UI time is excluded from model latency.")
    st.json(
        {
            "profile": report["profile"],
            "semantic_model": report["semantic_model"],
            "detector_model": report["detector_model"],
            "frame_count": len(report["frames"]),
        }
    )
    frame_index = (
        st.slider("Frame", 0, len(report["frames"]) - 1, 0) if len(report["frames"]) > 1 else 0
    )
    frame = report["frames"][frame_index]
    frame_path = report_path.parent / frame["overlay_relative_path"]
    st.image(str(frame_path), caption=f"Frame {frame['frame_index']}")
    st.json({"risk": frame["risk"], "tracks": frame["tracks"]})
