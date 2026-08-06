"""Static ONNX and golden-vector handoff bundles built before Jetson execution."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

import numpy as np

from edgeguard.deployment.package import _self_contained_onnx_bytes
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload

REQUIRED_MEMBERS = {
    "artifact_index.json",
    "golden_input.npy",
    "golden_output.npy",
    "metadata.json",
    "model.onnx",
    "numerical_equivalence.json",
    "ontology.yaml",
    "preprocessing.json",
    "runtime_contract.json",
}
STATIC_INPUT_SHAPE = (1, 3, 512, 1024)


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (canonical_json(payload) + "\n").encode("utf-8")


def _member(name: str, payload: bytes) -> tuple[ZipInfo, bytes]:
    info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info, payload


def build_jetson_deployment_bundle(
    destination: Path,
    *,
    onnx_path: Path,
    ontology_path: Path,
    preprocessing: dict[str, Any],
    model_family: str,
    source_commit: str,
    config_sha256: str,
    checkpoint_sha256: str,
    golden_input: np.ndarray,
    golden_output: np.ndarray,
) -> dict[str, Any]:
    """Build a self-verifying bundle without a TensorRT engine or device claim."""
    if destination.exists():
        raise ValueError("Jetson deployment bundle overwrite is not permitted")
    if not onnx_path.is_file() or not ontology_path.is_file():
        raise FileNotFoundError("Jetson deployment source artifact is missing")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source_commit must be a full lowercase Git SHA")
    for name, value in (("config_sha256", config_sha256), ("checkpoint_sha256", checkpoint_sha256)):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256")
    model_input = preprocessing.get("model_input")
    if (
        not isinstance(model_input, dict)
        or tuple(model_input.get("shape", ())) != STATIC_INPUT_SHAPE
    ):
        raise ValueError("Jetson preprocessing must declare static 1x3x512x1024 input")
    if model_input.get("dtype") != "float32" or not model_input.get("name"):
        raise ValueError("Jetson preprocessing input name/dtype is incomplete")
    input_array = np.asarray(golden_input, dtype=np.float32)
    output_array = np.asarray(golden_output, dtype=np.float32)
    if input_array.shape != STATIC_INPUT_SHAPE or not np.isfinite(input_array).all():
        raise ValueError("golden input must be finite float32 1x3x512x1024")
    if (
        output_array.ndim != 4
        or output_array.shape[0] != 1
        or output_array.shape[1] != 19
        or not np.isfinite(output_array).all()
    ):
        raise ValueError("golden output must be finite NCHW logits for 19 classes")
    model_bytes = _self_contained_onnx_bytes(onnx_path)
    try:
        ort = __import__("onnxruntime")
        session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
        actual = session.run(None, {str(model_input["name"]): input_array})[0]
    except Exception as error:
        raise ValueError("Jetson ONNX golden-vector execution failed") from error
    if actual.shape != output_array.shape:
        raise ValueError("golden output shape does not match ONNX output")
    difference = np.abs(actual.astype(np.float64) - output_array.astype(np.float64))
    allclose = bool(np.allclose(actual, output_array, atol=1.0e-4, rtol=1.0e-4))
    if not allclose:
        raise ValueError("PyTorch/ONNX golden output equivalence failed")
    equivalence = {
        "schema_version": "2.0",
        "record_type": "edgeguard_onnx_golden_equivalence",
        "input_shape": list(input_array.shape),
        "output_shape": list(output_array.shape),
        "max_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "allclose_atol_1e_4_rtol_1e_4": allclose,
        "scientific_status": "measured",
    }
    metadata = {
        "schema_version": "2.0",
        "record_type": "edgeguard_jetson_deployment_metadata",
        "model_family": model_family,
        "source_commit": source_commit,
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "source_onnx_sha256": sha256_file(onnx_path),
        "static_input_shape": list(STATIC_INPUT_SHAPE),
        "class_count": 19,
        "tensorrt_engine_included": False,
        "jetson_benchmark_status": "not_run",
    }
    runtime_contract = {
        "schema_version": "2.0",
        "record_type": "edgeguard_jetson_runtime_contract",
        "assumed_target": "Jetson Orin Nano Super",
        "assumed_jetpack": "6.2",
        "assumed_l4t": "36.4.x",
        "engine_build_location": "target_jetson_only",
        "automatic_device_upgrade": False,
        "primary_power_mode": "25W",
        "benchmark": {"warmup_frames": 200, "minimum_frames": 5000, "minimum_seconds": 600},
        "acceptance": {"median_ms_max": 50.0, "p95_ms_max": 66.7, "fps_min": 20.0},
    }
    members = {
        "golden_input.npy": _npy_bytes(input_array),
        "golden_output.npy": _npy_bytes(output_array),
        "metadata.json": _json_bytes(metadata),
        "model.onnx": model_bytes,
        "numerical_equivalence.json": _json_bytes(equivalence),
        "ontology.yaml": ontology_path.read_bytes(),
        "preprocessing.json": _json_bytes(preprocessing),
        "runtime_contract.json": _json_bytes(runtime_contract),
    }
    index_base = {
        "schema_version": "2.0",
        "record_type": "edgeguard_jetson_bundle_artifact_index",
        "artifacts": {
            name: {"sha256": hashlib.sha256(payload).hexdigest(), "byte_size": len(payload)}
            for name, payload in sorted(members.items())
        },
        "identities": {
            "model_family": model_family,
            "source_commit": source_commit,
            "config_sha256": config_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "ontology_sha256": sha256_file(ontology_path),
            "preprocessing_sha256": sha256_payload(preprocessing),
        },
    }
    artifact_index = {**index_base, "bundle_identity": sha256_payload(index_base)}
    members["artifact_index.json"] = _json_bytes(artifact_index)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in sorted(members.items()):
            info, content = _member(name, payload)
            archive.writestr(info, content)
    return {
        "status": "created",
        "bundle": destination.name,
        "bundle_sha256": sha256_file(destination),
        "bundle_identity": artifact_index["bundle_identity"],
        "tensorrt_engine_included": False,
        "jetson_benchmark_status": "not_run",
    }


def verify_jetson_deployment_bundle(package_path: Path) -> dict[str, Any]:
    """Verify archive safety, hashes, identity, and recorded ONNX equivalence."""
    try:
        with ZipFile(package_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != REQUIRED_MEMBERS:
                raise ValueError("Jetson deployment bundle members mismatch")
            if any(
                PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts
                or len(PurePosixPath(name).parts) != 1
                for name in names
            ):
                raise ValueError("Jetson deployment bundle contains an unsafe member")
            if archive.testzip() is not None:
                raise ValueError("Jetson deployment bundle CRC validation failed")
            members = {name: archive.read(name) for name in names}
    except BadZipFile as error:
        raise ValueError("Jetson deployment bundle is corrupt") from error
    index = json.loads(members["artifact_index.json"])
    index_base = {key: value for key, value in index.items() if key != "bundle_identity"}
    if sha256_payload(index_base) != index.get("bundle_identity"):
        raise ValueError("Jetson deployment bundle identity mismatch")
    for name, identity in index["artifacts"].items():
        payload = members.get(name)
        if payload is None or hashlib.sha256(payload).hexdigest() != identity["sha256"]:
            raise ValueError(f"Jetson deployment artifact hash mismatch: {name}")
        if len(payload) != identity["byte_size"]:
            raise ValueError(f"Jetson deployment artifact size mismatch: {name}")
    equivalence = json.loads(members["numerical_equivalence.json"])
    metadata = json.loads(members["metadata.json"])
    if equivalence.get("allclose_atol_1e_4_rtol_1e_4") is not True:
        raise ValueError("Jetson deployment bundle lacks ONNX numerical acceptance")
    if metadata.get("tensorrt_engine_included") is not False:
        raise ValueError("Jetson deployment bundle must not include a TensorRT engine")
    return {
        "status": "verified",
        "bundle_sha256": sha256_file(package_path),
        "bundle_identity": index["bundle_identity"],
        "identities": index["identities"],
        "tensorrt_engine_included": False,
        "jetson_benchmark_status": "not_run",
    }
