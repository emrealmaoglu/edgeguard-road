from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image


def test_streamlit_prerecorded_dashboard_headless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    streamlit = pytest.importorskip("streamlit.testing.v1")
    overlay_root = tmp_path / "overlays"
    overlay_root.mkdir()
    Image.new("RGB", (64, 32), color=(10, 20, 30)).save(overlay_root / "frame-000.png")
    report = {
        "profile": "local_synthetic_validation",
        "semantic_model": "actual_random_weight_fixture",
        "detector_model": "actual_random_weight_fixture",
        "frames": [
            {
                "frame_index": 0,
                "overlay_relative_path": "overlays/frame-000.png",
                "risk": {"risk_category": "low", "total_risk_score": 0.1},
                "tracks": [],
            }
        ],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setenv("EDGEGUARD_VIDEO_REPORT", str(report_path))
    app = streamlit.AppTest.from_file("scripts/run_video_dashboard.py")
    app.run(timeout=10)
    assert not app.exception
    assert "EdgeGuard-Road" in app.title[0].value
