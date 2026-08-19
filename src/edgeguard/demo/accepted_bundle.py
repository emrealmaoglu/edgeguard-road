"""Fail-closed discovery of accepted, hash-bound Streamlit demo bundles."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from edgeguard.serialization import sha256_file

MANIFEST_NAME = "accepted_demo_bundle.json"


def _artifact(root: Path, payload: object, field: str) -> Path:
    if not isinstance(payload, dict):
        raise ValueError(f"accepted demo model {field} artifact is invalid")
    relative = PurePosixPath(str(payload.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"accepted demo model {field} path is unsafe")
    path = root.joinpath(*relative.parts)
    if not path.is_file() or sha256_file(path) != payload.get("sha256"):
        raise ValueError(f"accepted demo model {field} identity mismatch")
    return path


def load_accepted_demo_bundle(manifest_path: Path) -> dict[str, Any]:
    """Validate one accepted bundle and resolve its selected-model records."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("record_type") != "edgeguard_accepted_demo_bundle":
        raise ValueError("invalid accepted demo bundle record type")
    if payload.get("status") != "accepted":
        raise ValueError("Streamlit default mode accepts only accepted bundles")
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("accepted demo bundle has no models")
    records: list[dict[str, str]] = []
    for model in models:
        if not isinstance(model, dict) or model.get("backend") not in {"onnx", "mmseg"}:
            raise ValueError("accepted demo bundle has an invalid model record")
        model_path = _artifact(manifest_path.parent, model.get("model"), "model")
        record = {
            "label": str(model.get("label") or model_path.stem),
            "backend": str(model["backend"]),
            "model": str(model_path),
            "datasets": str(model.get("datasets", "unknown")),
            "bundle_manifest": str(manifest_path),
            "scientific_status": "accepted",
        }
        if model["backend"] == "mmseg":
            record["config"] = str(_artifact(manifest_path.parent, model.get("config"), "config"))
        records.append(record)
    return {"manifest": payload, "records": records}


def discover_accepted_demo_models(root: Path) -> list[dict[str, str]]:
    """Discover only valid accepted bundle manifests under one explicit root."""
    if not root.is_dir():
        return []
    records: list[dict[str, str]] = []
    for manifest in sorted(root.glob(f"**/{MANIFEST_NAME}")):
        records.extend(load_accepted_demo_bundle(manifest)["records"])
    return records
