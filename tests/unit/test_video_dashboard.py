from __future__ import annotations

from pathlib import Path

import numpy as np

from edgeguard.demo.video_dashboard import headless_dashboard_smoke, render_evidence_frame


def test_dashboard_overlay_and_headless_smoke(tmp_path: Path) -> None:
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    mask = np.zeros((32, 64), dtype=np.uint8)
    score = np.linspace(0, 1, 32 * 64, dtype=np.float32).reshape(32, 64)
    rendered = render_evidence_frame(
        image,
        mask,
        score,
        components=[{"bbox_xyxy": [4, 4, 10, 10]}],
        detections=[{"box_xyxy": [15, 5, 24, 16]}],
        tracks=[{"track_id": 1}],
        risk={"risk_category": "low", "total_risk_score": 0.2},
    )
    frame = tmp_path / "frame.png"
    rendered.save(frame)
    result = headless_dashboard_smoke(
        frame,
        {
            "profile": "local",
            "model_identity": "fixture",
            "frame_count": 1,
            "latency_includes_ui": False,
        },
    )
    assert result["status"] == "passed"
    assert result["real_time_claim"] is False
