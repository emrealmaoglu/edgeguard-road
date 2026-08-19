"""Tests for fail-closed accepted Streamlit bundle discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgeguard.demo.accepted_bundle import discover_accepted_demo_models
from edgeguard.serialization import sha256_file


def _bundle(tmp_path: Path, *, status: str = "accepted") -> tuple[Path, Path]:
    root = tmp_path / "release"
    root.mkdir()
    model = root / "model.onnx"
    model.write_bytes(b"fixture")
    manifest = root / "accepted_demo_bundle.json"
    manifest.write_text(
        json.dumps(
            {
                "record_type": "edgeguard_accepted_demo_bundle",
                "status": status,
                "models": [
                    {
                        "label": "Accepted SegFormer",
                        "backend": "onnx",
                        "datasets": "cityscapes+idd20k",
                        "model": {"path": model.name, "sha256": sha256_file(model)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return model, manifest


def test_accepted_bundle_discovery_verifies_model_hash(tmp_path: Path) -> None:
    model, _manifest = _bundle(tmp_path)
    records = discover_accepted_demo_models(tmp_path)
    assert records[0]["model"] == str(model)
    assert records[0]["scientific_status"] == "accepted"

    model.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="identity mismatch"):
        discover_accepted_demo_models(tmp_path)


def test_unaccepted_bundle_is_not_silently_promoted(tmp_path: Path) -> None:
    _bundle(tmp_path, status="measured")
    with pytest.raises(ValueError, match="accepted bundles"):
        discover_accepted_demo_models(tmp_path)
