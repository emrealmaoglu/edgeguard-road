"""Deterministic, self-verifying deployment bundles for local fixture backends."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

import numpy as np

from edgeguard.serialization import canonical_json, sha256_file, sha256_payload

REQUIRED_MEMBERS = {
    "calibration.json",
    "manifest.json",
    "metadata.json",
    "model.onnx",
    "numerical_validation.json",
    "ontology.yaml",
    "postprocessing.json",
    "preprocessing.json",
    "runtime_requirements.json",
    "thresholds.json",
}


def _stable_zip_member(name: str, payload: bytes) -> tuple[ZipInfo, bytes]:
    info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info, payload


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (canonical_json(payload) + "\n").encode("utf-8")


def _require_placeholder(payload: dict[str, Any], name: str) -> None:
    if payload.get("status") != "non_scientific_placeholder" or payload.get("value") is not None:
        raise ValueError(f"{name} must remain an explicit non-scientific null placeholder")


def _self_contained_onnx_bytes(path: Path) -> bytes:
    """Load any ONNX external data and serialize one portable model member."""
    try:
        onnx = __import__("onnx")
        model = onnx.load_model(path, load_external_data=True)
        onnx.external_data_helper.convert_model_from_external_data(model)
        onnx.checker.check_model(model)
        return model.SerializeToString()
    except Exception as error:
        raise ValueError("deployment ONNX model or its external data is invalid") from error


def build_deployment_package(
    destination: Path,
    *,
    onnx_path: Path,
    metadata: dict[str, Any],
    preprocessing: dict[str, Any],
    postprocessing: dict[str, Any],
    ontology_path: Path,
    calibration: dict[str, Any],
    thresholds: dict[str, Any],
    runtime_requirements: dict[str, Any],
    numerical_validation: dict[str, Any],
) -> dict[str, Any]:
    """Build one deterministic package without embedding a checkpoint or private path."""
    if destination.exists():
        raise ValueError("deployment package overwrite is not permitted")
    if not onnx_path.is_file() or not ontology_path.is_file():
        raise ValueError("deployment package source artifact is missing")
    required_identity = {
        "model_family",
        "config_sha256",
        "checkpoint_sha256",
        "source_commit",
        "initialization",
    }
    if not required_identity <= set(metadata):
        raise ValueError("deployment metadata is missing model/config/checkpoint identity")
    if any(Path(str(value)).is_absolute() for value in metadata.values() if isinstance(value, str)):
        raise ValueError("deployment metadata cannot contain absolute paths")
    _require_placeholder(calibration, "calibration")
    _require_placeholder(thresholds, "thresholds")

    source_onnx_sha256 = sha256_file(onnx_path)
    model_bytes = _self_contained_onnx_bytes(onnx_path)
    ontology_bytes = ontology_path.read_bytes()
    normalized_metadata = {
        **metadata,
        "source_onnx_sha256": source_onnx_sha256,
        "model_artifact_sha256": __import__("hashlib").sha256(model_bytes).hexdigest(),
        "onnx_external_data_embedded": True,
        "checkpoint_artifact_included": False,
        "scientific_evidence": False,
    }
    members: dict[str, bytes] = {
        "calibration.json": _json_bytes(calibration),
        "metadata.json": _json_bytes(normalized_metadata),
        "model.onnx": model_bytes,
        "numerical_validation.json": _json_bytes(numerical_validation),
        "ontology.yaml": ontology_bytes,
        "postprocessing.json": _json_bytes(postprocessing),
        "preprocessing.json": _json_bytes(preprocessing),
        "runtime_requirements.json": _json_bytes(runtime_requirements),
        "thresholds.json": _json_bytes(thresholds),
    }
    files = [
        {
            "relative_path": name,
            "byte_size": len(payload),
            "sha256": __import__("hashlib").sha256(payload).hexdigest(),
        }
        for name, payload in sorted(members.items())
    ]
    manifest_base = {
        "schema_version": "1.0",
        "record_type": "edgeguard_deployment_package_manifest",
        "artifact_type": "onnx_deployment_fixture",
        "files": files,
        "identities": {
            "model_family": normalized_metadata["model_family"],
            "model_artifact_sha256": normalized_metadata["model_artifact_sha256"],
            "config_sha256": normalized_metadata["config_sha256"],
            "checkpoint_sha256": normalized_metadata["checkpoint_sha256"],
            "ontology_sha256": __import__("hashlib").sha256(ontology_bytes).hexdigest(),
            "preprocessing_sha256": sha256_payload(preprocessing),
            "postprocessing_sha256": sha256_payload(postprocessing),
        },
        "scientific_evidence": False,
    }
    manifest = {**manifest_base, "package_identity": sha256_payload(manifest_base)}
    members["manifest.json"] = _json_bytes(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in sorted(members.items()):
            info, content = _stable_zip_member(name, payload)
            archive.writestr(info, content)
    return {
        "status": "created",
        "filename": destination.name,
        "sha256": sha256_file(destination),
        "byte_size": destination.stat().st_size,
        "package_identity": manifest["package_identity"],
        "manifest": manifest,
    }


def _validated_members(archive: ZipFile) -> dict[str, bytes]:
    infos = archive.infolist()
    names = [item.filename for item in infos]
    if len(names) != len(set(names)):
        raise ValueError("deployment package contains duplicate members")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
            raise ValueError("deployment package contains an unsafe member")
    if set(names) != REQUIRED_MEMBERS:
        missing = sorted(REQUIRED_MEMBERS - set(names))
        extra = sorted(set(names) - REQUIRED_MEMBERS)
        raise ValueError(f"deployment package members mismatch: missing={missing}, extra={extra}")
    return {name: archive.read(name) for name in names}


def verify_deployment_package(
    package_path: Path,
    *,
    expected_identities: dict[str, str] | None = None,
    expected_ontology_sha256: str | None = None,
    expected_preprocessing_sha256: str | None = None,
    fixture_input: np.ndarray | None = None,
) -> dict[str, Any]:
    """Verify package integrity, identities, contracts, and optional ORT inference."""
    if not package_path.is_file():
        raise ValueError("deployment package is missing")
    try:
        with ZipFile(package_path) as archive:
            members = _validated_members(archive)
            if archive.testzip() is not None:
                raise ValueError("deployment package CRC validation failed")
    except BadZipFile as error:
        raise ValueError("deployment package is corrupt") from error
    manifest = json.loads(members["manifest.json"])
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("deployment package manifest file inventory is invalid")
    for item in files:
        name = item["relative_path"]
        payload = members.get(name)
        if payload is None:
            raise ValueError(f"deployment package manifest artifact is missing: {name}")
        digest = __import__("hashlib").sha256(payload).hexdigest()
        if digest != item["sha256"] or len(payload) != item["byte_size"]:
            raise ValueError(f"deployment package artifact hash mismatch: {name}")
    manifest_base = {key: value for key, value in manifest.items() if key != "package_identity"}
    if sha256_payload(manifest_base) != manifest.get("package_identity"):
        raise ValueError("deployment package identity mismatch")
    identities = manifest["identities"]
    for key, expected in (expected_identities or {}).items():
        if identities.get(key) != expected:
            raise ValueError(f"deployment package model/config identity mismatch: {key}")
    if (
        expected_ontology_sha256 is not None
        and identities["ontology_sha256"] != expected_ontology_sha256
    ):
        raise ValueError("deployment package ontology identity mismatch")
    if (
        expected_preprocessing_sha256 is not None
        and identities["preprocessing_sha256"] != expected_preprocessing_sha256
    ):
        raise ValueError("deployment package preprocessing identity mismatch")

    fixture: dict[str, Any] | None = None
    if fixture_input is not None:
        preprocessing = json.loads(members["preprocessing.json"])
        expected_shape = tuple(preprocessing["model_input"]["shape"])
        array = np.asarray(fixture_input, dtype=np.float32)
        if array.shape != expected_shape:
            raise ValueError("deployment fixture input shape does not match preprocessing")
        ort = __import__("onnxruntime")
        session = ort.InferenceSession(members["model.onnx"], providers=["CPUExecutionProvider"])
        outputs = session.run(None, {preprocessing["model_input"]["name"]: array})
        if not outputs or not all(np.isfinite(item).all() for item in outputs):
            raise ValueError("deployment package fixture inference produced invalid outputs")
        fixture = {
            "status": "passed",
            "output_names": [item.name for item in session.get_outputs()],
            "output_shapes": [list(item.shape) for item in outputs],
            "output_dtypes": [str(item.dtype) for item in outputs],
        }
    return {
        "status": "verified",
        "package_sha256": sha256_file(package_path),
        "package_identity": manifest["package_identity"],
        "identities": identities,
        "fixture_inference": fixture,
        "scientific_evidence": False,
    }
