from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from edgeguard.rescue.visualization import InferenceResult
from scripts import predict


def test_predict_emits_phase_one_perception_contract(tmp_path: Path, monkeypatch: object) -> None:
    image_path = tmp_path / "input.png"
    model_path = tmp_path / "model.onnx"
    output = tmp_path / "output"
    Image.new("RGB", (16, 12), color=(20, 30, 40)).save(image_path)
    model_path.write_bytes(b"fixture")
    mask = np.full((12, 16), 2, dtype=np.uint8)
    mask[6:, 3:13] = 0
    mask[7:11, 7:10] = 13
    result = InferenceResult(
        mask=mask,
        confidence=np.full(mask.shape, 0.8, dtype=np.float32),
        entropy=np.full(mask.shape, 0.2, dtype=np.float32),
        latency_ms=12.5,
        backend="fixture",
        metadata={"fixture": True},
    )
    monkeypatch.setattr(predict, "predict_onnx", lambda *_args, **_kwargs: result)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            "--image",
            str(image_path),
            "--model",
            str(model_path),
            "--output-dir",
            str(output),
            "--emit-regions",
            "--emit-risk",
            "--minimum-region-area",
            "2",
            "--minimum-drivable-area",
            "4",
        ],
    )
    assert predict.main() == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["record_type"] == "edgeguard_perception_prediction"
    assert summary["physical_risk_probability"] is False
    assert summary["instance_detection"] is False
    assert summary["domain_shift_alert"] is None
    assert summary["regions"][0]["class_name"] == "car"
    assert (output / summary["outputs"]["drivable_mask"]).is_file()
    assert (output / summary["outputs"]["attention_map"]).is_file()
    assert (output / summary["regions"][0]["mask"]).is_file()
