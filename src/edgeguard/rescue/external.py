"""Sealed external inference packaging without accessing hidden ground truth."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edgeguard.rescue.inference import predict_mmseg, predict_onnx
from edgeguard.rescue.ledger import append_run_ledger
from edgeguard.rescue.multidomain import validate_dataset_manifest, verify_sealed_release
from edgeguard.serialization import canonical_json, sha256_file

CITYSCAPES_TRAIN_TO_LABEL_ID = np.asarray(
    [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
    dtype=np.uint8,
)


def _encode_submission_mask(mask: np.ndarray, encoding: str) -> np.ndarray:
    """Convert canonical predictions to the vendor-declared submission encoding."""
    canonical = np.asarray(mask, dtype=np.uint8)
    if bool((canonical > 18).any()):
        raise ValueError("prediction contains a non-canonical semantic class")
    if encoding == "canonical_train_ids":
        return canonical
    if encoding == "cityscapes_label_ids":
        return CITYSCAPES_TRAIN_TO_LABEL_ID[canonical]
    raise ValueError("unreviewed sealed submission encoding")


def package_external_predictions(
    *,
    manifest_path: Path,
    model_path: Path,
    output_dir: Path,
    sealed_release: Path,
    resolved_config: Path | None,
    device: str,
) -> dict[str, Any]:
    """Run a frozen model once and create a hash-recorded server submission archive."""
    manifest = validate_dataset_manifest(manifest_path)
    if manifest.get("sealed") is not True:
        raise ValueError("external prediction packaging is reserved for sealed manifests")
    verify_sealed_release(manifest_path, model_path, sealed_release)
    records = manifest["roles"].get("sealed_external_test")
    if not isinstance(records, list) or not records:
        raise ValueError("sealed manifest has no external test records")
    submission_encoding = str(manifest.get("submission_encoding", ""))
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite external package: {output_dir}")
    predictions = output_dir / "predictions"
    predictions.mkdir(parents=True)
    backend: Any = None
    if model_path.suffix.lower() == ".onnx":
        try:
            ort = __import__("onnxruntime")
        except ModuleNotFoundError as error:
            raise RuntimeError("ONNX Runtime is required for external packaging") from error
        backend = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    else:
        if resolved_config is None:
            raise ValueError("PyTorch external packaging requires --resolved-config")
        try:
            apis = __import__("mmseg.apis", fromlist=["init_model"])
        except ModuleNotFoundError as error:
            raise RuntimeError("MMSeg is required for checkpoint external packaging") from error
        backend = apis.init_model(str(resolved_config), str(model_path), device=device)
    output_records: list[dict[str, Any]] = []
    root = Path(manifest["dataset_root"])
    for record in records:
        submission_name = record.get("submission_name")
        if not isinstance(submission_name, str) or not submission_name.endswith(".png"):
            raise ValueError("every sealed record requires an official .png submission_name")
        with Image.open(root / str(record["image"])) as opened:
            image = opened.convert("RGB")
        if model_path.suffix.lower() == ".onnx":
            inference_result = predict_onnx(image, model_path, session=backend)
        else:
            assert resolved_config is not None
            inference_result = predict_mmseg(
                image, resolved_config, model_path, device=device, model=backend
            )
        target = predictions / submission_name
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = _encode_submission_mask(inference_result.mask, submission_encoding)
        Image.fromarray(encoded, mode="L").save(target)
        output_records.append(
            {
                "sample_id": record["sample_id"],
                "submission_name": submission_name,
                "prediction_sha256": sha256_file(target),
            }
        )
    archive_path = output_dir / f"{manifest['dataset_id']}_submission.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(predictions.glob("**/*.png")):
            archive.write(path, path.relative_to(predictions).as_posix())
    package = {
        "schema_version": "2.0",
        "record_type": "sealed_external_prediction_package",
        "dataset_id": manifest["dataset_id"],
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "model_sha256": sha256_file(model_path),
        "sealed_release_sha256": sha256_file(sealed_release),
        "prediction_count": len(output_records),
        "archive": archive_path.name,
        "archive_sha256": sha256_file(archive_path),
        "predictions": output_records,
        "contains_ground_truth": False,
        "submission_encoding": submission_encoding,
        "result_claim": "pending official evaluation server",
    }
    (output_dir / "package_manifest.json").write_text(
        canonical_json(package) + "\n", encoding="utf-8"
    )
    append_run_ledger(
        output_dir.parent / "run_ledger.jsonl",
        operation="sealed_external_prediction_package",
        result=package,
    )
    return package


def record_external_server_result(
    package_manifest: Path, server_result: Path, output_path: Path
) -> dict[str, Any]:
    """Bind a human-downloaded official score to the exact submitted archive."""
    package = json.loads(package_manifest.read_text(encoding="utf-8"))
    result = json.loads(server_result.read_text(encoding="utf-8"))
    if package.get("record_type") != "sealed_external_prediction_package":
        raise ValueError("invalid external package manifest")
    archive = package_manifest.parent / str(package["archive"])
    if not archive.is_file() or sha256_file(archive) != package.get("archive_sha256"):
        raise ValueError("external submission archive hash mismatch")
    required = ("metric_name", "metric_value", "submission_id", "submitted_at")
    if any(key not in result for key in required):
        raise ValueError("server result lacks required official result fields")
    metric_value = float(result["metric_value"])
    if not np.isfinite(metric_value):
        raise ValueError("official metric must be finite")
    record = {
        "schema_version": "2.0",
        "record_type": "sealed_external_server_result",
        "dataset_id": package["dataset_id"],
        "dataset_manifest_sha256": package["dataset_manifest_sha256"],
        "model_sha256": package["model_sha256"],
        "archive_sha256": package["archive_sha256"],
        "submission_id": str(result["submission_id"]),
        "submitted_at": str(result["submitted_at"]),
        "official_metric_name": str(result["metric_name"]),
        "official_metric_value": metric_value,
        "server_result_sha256": sha256_file(server_result),
        "not_relabelled_as_cityscapes19_mIoU": True,
        "post_result_tuning_forbidden": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_json(record) + "\n", encoding="utf-8")
    append_run_ledger(
        output_path.parent / "run_ledger.jsonl",
        operation="sealed_external_server_result",
        result=record,
    )
    return record
