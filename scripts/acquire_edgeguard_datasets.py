"""Acquire the small approved EdgeGuard dataset queue with dataset-level gates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from edgeguard.config import UniqueKeySafeLoader
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload
from edgeguard.telemetry.longrun import LiveCommandRunner, LongRunStatus, atomic_write_json

GENERATOR_SHA256_ENV_FIELDS = {
    "cityscapes_input_manifest_sha256": "cityscapes_input_manifest_sha256_env",
    "generator_config_sha256": "generator_config_sha256_env",
    "output_manifest_sha256": "output_manifest_sha256_env",
}


def load_queue(path: Path) -> dict[str, Any]:
    """Load the fixed queue and enforce the artifact-or-generator schema."""
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError("acquisition queue is missing or malformed") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "2.0"
        or payload.get("record_type") != "edgeguard_dataset_acquisition_queue"
    ):
        raise ValueError("invalid acquisition queue record")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("acquisition queue has no entries")
    dataset_ids: set[str] = set()
    prohibited = {"SMIYC RoadAnomaly21", "SMIYC RoadObstacle21"}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("dataset_id"), str):
            raise ValueError("acquisition queue contains an invalid dataset")
        dataset_id = entry["dataset_id"]
        if dataset_id in dataset_ids:
            raise ValueError("acquisition queue contains a duplicate dataset ID")
        dataset_ids.add(dataset_id)
        if any(name in canonical_json(entry) for name in prohibited):
            raise ValueError("sealed data cannot enter the acquisition queue")
        kind = entry.get("acquisition_kind")
        if kind == "required_artifacts":
            artifacts = entry.get("required_artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                raise ValueError("artifact acquisition requires at least one artifact")
            artifact_ids = [artifact.get("artifact_id") for artifact in artifacts]
            if any(not isinstance(value, str) for value in artifact_ids):
                raise ValueError("artifact IDs must be strings")
            if len(set(artifact_ids)) != len(artifact_ids):
                raise ValueError("artifact IDs must be unique within a dataset")
        elif kind == "pinned_generator":
            generator = entry.get("generator")
            if not isinstance(generator, dict) or set(GENERATOR_SHA256_ENV_FIELDS.values()) - set(
                generator
            ):
                raise ValueError("pinned generator identity is incomplete")
            for field in (
                "source_repository",
                "immutable_revision",
                "framework_identity",
            ):
                if not isinstance(generator.get(field), str) or not generator[field]:
                    raise ValueError("pinned generator source identity is incomplete")
            for field in (
                "progress_resume_required",
                "deterministic_output_manifest_required",
                "generated_data_receipt_required",
            ):
                if generator.get(field) is not True:
                    raise ValueError("pinned generator execution contract is incomplete")
            if "required_artifacts" in entry:
                raise ValueError("generated datasets cannot masquerade as archives")
        elif kind != "blocked":
            raise ValueError("unknown acquisition kind")
    return payload


def _safe_filename(value: str) -> str:
    path = Path(value)
    if path.name != value or value in {"", ".", ".."}:
        raise ValueError("runtime archive filename must be one safe path component")
    return value


def _runtime_value(record: dict[str, Any], field: str) -> str | None:
    variable = record.get(field)
    if not isinstance(variable, str):
        raise ValueError(f"acquisition record is missing {field}")
    return os.environ.get(variable)


def _download_command(url: str, partial: Path) -> tuple[str, ...]:
    if shutil.which("aria2c"):
        return (
            "aria2c",
            "--continue=true",
            "--max-tries=1",
            "--summary-interval=5",
            "--dir",
            str(partial.parent),
            "--out",
            partial.name,
            url,
        )
    if shutil.which("curl"):
        return (
            "curl",
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        )
    if shutil.which("wget"):
        return ("wget", "--continue", "--output-document", str(partial), url)
    raise RuntimeError("aria2c, curl, or wget is required for resumable acquisition")


def _copy_resumable(source: Path, partial: Path, status: LongRunStatus) -> None:
    total = source.stat().st_size
    initial = partial.stat().st_size if partial.exists() else 0
    completed = initial
    if completed > total:
        raise ValueError("destination partial file is larger than its source")
    started = time.monotonic()
    partial.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, partial.open("ab") as outgoing:
        incoming.seek(completed)
        while chunk := incoming.read(8 * 1024**2):
            outgoing.write(chunk)
            completed += len(chunk)
            elapsed = max(time.monotonic() - started, 1e-9)
            status.update(
                phase="drive-copy",
                completed=completed,
                total=total,
                speed_per_second=(completed - initial) / elapsed,
            )
        outgoing.flush()
        os.fsync(outgoing.fileno())
    status.update(completed=completed, total=total, force=True)


def _verify_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError("downloaded artifact byte size mismatch")
    if sha256_file(path) != expected_sha256:
        raise ValueError("downloaded artifact SHA-256 mismatch")


def acquire_artifact(
    dataset: dict[str, Any],
    artifact: dict[str, Any],
    drive_root: Path,
    *,
    mode: str | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    """Acquire one named required artifact without promoting the whole dataset."""
    dataset_id = str(dataset["dataset_id"])
    artifact_id = str(artifact["artifact_id"])
    runtime = {
        field: _runtime_value(artifact, field)
        for field in ("url_env", "filename_env", "sha256_env", "size_env")
    }
    missing = [artifact[field] for field, value in runtime.items() if not value]
    if missing:
        return {
            "dataset_id": dataset_id,
            "artifact_id": artifact_id,
            "artifact_status": "blocked_access",
            "missing_runtime_variables": missing,
        }
    url = str(runtime["url_env"])
    filename = _safe_filename(str(runtime["filename_env"]))
    expected_sha256 = str(runtime["sha256_env"])
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("expected artifact SHA-256 must be 64 lowercase hexadecimal characters")
    expected_size = int(str(runtime["size_env"]))
    if expected_size <= 0:
        raise ValueError("expected artifact byte size must be positive")
    selected_mode = mode or str(artifact["default_mode"])
    archive_directory = drive_root / "archives" / dataset_id
    manifest_directory = drive_root / "manifests" / dataset_id / "artifacts"
    final = archive_directory / filename
    receipt_path = manifest_directory / f"{artifact_id}.json"
    manifest_directory.mkdir(parents=True, exist_ok=True)
    status = LongRunStatus(manifest_directory / f"{artifact_id}.run_status.json")
    if final.is_file():
        _verify_file(final, expected_size, expected_sha256)
        status.complete(last_checkpoint=filename)
    else:
        archive_directory.mkdir(parents=True, exist_ok=True)
        if selected_mode == "persistent":
            partial = drive_root / "downloads" / "partial" / f"{filename}.part"
            partial.parent.mkdir(parents=True, exist_ok=True)
            command = _download_command(url, partial)
            runner = LiveCommandRunner(manifest_directory / "logs" / artifact_id, status)
            last_error: BaseException | None = None
            for attempt in range(1, retries + 1):
                try:
                    runner.run(
                        f"download-{dataset_id}-{artifact_id}-attempt-{attempt}",
                        command,
                        display_command=(*command[:-1], "<runtime-url>"),
                        stage_index=attempt,
                        stage_total=retries,
                        progress_probe=lambda: (
                            partial.stat().st_size if partial.exists() else 0,
                            expected_size,
                        ),
                        redact_values=(url,),
                    )
                    last_error = None
                    break
                except (OSError, subprocess.CalledProcessError) as error:
                    last_error = error
                    status.update(last_error=type(error).__name__, force=True)
                    if attempt < retries:
                        time.sleep(2 ** (attempt - 1))
            if last_error is not None:
                status.fail(last_error)
                raise RuntimeError("resumable acquisition exhausted retries") from last_error
            _verify_file(partial, expected_size, expected_sha256)
            os.replace(partial, final)
        elif selected_mode == "fast":
            local = Path("/content/edgeguard-downloads") / dataset_id / filename
            local.parent.mkdir(parents=True, exist_ok=True)
            runner = LiveCommandRunner(manifest_directory / "logs" / artifact_id, status)
            command = _download_command(url, local)
            runner.run(
                f"download-{dataset_id}-{artifact_id}",
                command,
                display_command=(*command[:-1], "<runtime-url>"),
                stage_index=1,
                stage_total=2,
                progress_probe=lambda: (
                    local.stat().st_size if local.exists() else 0,
                    expected_size,
                ),
                redact_values=(url,),
            )
            _verify_file(local, expected_size, expected_sha256)
            incoming = final.with_name(f".{final.name}.incoming")
            _copy_resumable(local, incoming, status)
            _verify_file(incoming, expected_size, expected_sha256)
            os.replace(incoming, final)
        else:
            raise ValueError("acquisition mode must be persistent or fast")
        status.complete(last_checkpoint=filename)
    receipt = {
        "schema_version": "1.0",
        "record_type": "edgeguard_dataset_artifact_acquisition_receipt",
        "dataset_id": dataset_id,
        "artifact_id": artifact_id,
        "source": artifact["source_reference"],
        "access_date": datetime.now(timezone.utc).date().isoformat(),
        "terms_status": dataset["terms_status"],
        "filename": filename,
        "byte_size": expected_size,
        "sha256": expected_sha256,
        "artifact_status": "verified",
        "mode": selected_mode,
    }
    atomic_write_json(receipt_path, receipt)
    return receipt


def evaluate_artifact_dataset_readiness(
    dataset: dict[str, Any], drive_root: Path
) -> dict[str, Any]:
    """Promote only when every declared artifact receipt and archive verifies."""
    dataset_id = str(dataset["dataset_id"])
    verified: list[dict[str, Any]] = []
    missing: list[str] = []
    for artifact in dataset["required_artifacts"]:
        artifact_id = str(artifact["artifact_id"])
        receipt_path = drive_root / "manifests" / dataset_id / "artifacts" / f"{artifact_id}.json"
        if not receipt_path.is_file():
            missing.append(artifact_id)
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        archive = drive_root / "archives" / dataset_id / str(receipt.get("filename", ""))
        try:
            _verify_file(archive, int(receipt["byte_size"]), str(receipt["sha256"]))
        except (KeyError, OSError, TypeError, ValueError):
            missing.append(artifact_id)
            continue
        if (
            receipt.get("artifact_id") != artifact_id
            or receipt.get("artifact_status") != "verified"
        ):
            missing.append(artifact_id)
            continue
        verified.append(
            {
                "artifact_id": artifact_id,
                "filename": receipt["filename"],
                "byte_size": receipt["byte_size"],
                "sha256": receipt["sha256"],
            }
        )
    if missing:
        return {
            "dataset_id": dataset_id,
            "dataset_status": "blocked_required_artifacts",
            "verified_artifact_ids": sorted(item["artifact_id"] for item in verified),
            "missing_or_invalid_artifact_ids": sorted(missing),
        }
    readiness = {
        "schema_version": "1.0",
        "record_type": "edgeguard_dataset_readiness_receipt",
        "dataset_id": dataset_id,
        "dataset_status": "ready",
        "required_artifacts": sorted(verified, key=lambda item: item["artifact_id"]),
    }
    readiness["readiness_sha256"] = sha256_payload(readiness)
    atomic_write_json(
        drive_root / "manifests" / dataset_id / "dataset_readiness_receipt.json",
        readiness,
    )
    return readiness


def generator_dataset_status(dataset: dict[str, Any], drive_root: Path) -> dict[str, Any]:
    """Report the pinned generator gate; never accept an arbitrary Static archive."""
    dataset_id = str(dataset["dataset_id"])
    generator = dataset["generator"]
    identities = {
        identity: _runtime_value(generator, environment_field)
        for identity, environment_field in GENERATOR_SHA256_ENV_FIELDS.items()
    }
    missing = [
        generator[GENERATOR_SHA256_ENV_FIELDS[identity]]
        for identity, value in identities.items()
        if not value
    ]
    base = {
        "dataset_id": dataset_id,
        "acquisition_kind": "pinned_generator",
        "source_repository": generator["source_repository"],
        "immutable_revision": generator["immutable_revision"],
    }
    if missing:
        return {**base, "dataset_status": "blocked_generator_inputs", "missing": missing}
    if any(
        len(str(value)) != 64
        or any(character not in "0123456789abcdef" for character in str(value))
        for value in identities.values()
    ):
        raise ValueError("generator identity hashes must be lowercase SHA-256 values")
    receipt_path = drive_root / "manifests" / dataset_id / "generated_data_receipt.json"
    if not receipt_path.is_file():
        return {
            **base,
            "dataset_status": "generator_ready_to_execute",
            "input_identities": identities,
            "progress_resume_required": True,
            "dataset_ready": False,
        }
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "source_repository": generator["source_repository"],
        "immutable_revision": generator["immutable_revision"],
        **identities,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("generated-data receipt identity mismatch")
    result = {
        **base,
        "dataset_status": "ready",
        "dataset_ready": True,
        "generated_data_receipt_sha256": sha256_file(receipt_path),
        "output_manifest_sha256": identities["output_manifest_sha256"],
    }
    atomic_write_json(
        drive_root / "manifests" / dataset_id / "dataset_readiness_receipt.json",
        result,
    )
    return result


def process_dataset(
    dataset: dict[str, Any], drive_root: Path, *, mode: str | None = None
) -> dict[str, Any]:
    """Process supplied artifacts independently, then evaluate dataset readiness."""
    kind = dataset["acquisition_kind"]
    if kind == "blocked":
        return {
            "dataset_id": dataset["dataset_id"],
            "dataset_status": "blocked_policy",
            "human_action": dataset["human_action"],
        }
    if kind == "pinned_generator":
        return generator_dataset_status(dataset, drive_root)
    artifact_results: list[dict[str, Any]] = []
    for artifact in dataset["required_artifacts"]:
        try:
            artifact_results.append(acquire_artifact(dataset, artifact, drive_root, mode=mode))
        except (OSError, RuntimeError, ValueError) as error:
            artifact_results.append(
                {
                    "dataset_id": dataset["dataset_id"],
                    "artifact_id": artifact["artifact_id"],
                    "artifact_status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                }
            )
    return {
        "dataset_id": dataset["dataset_id"],
        "artifact_results": artifact_results,
        "readiness": evaluate_artifact_dataset_readiness(dataset, drive_root),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-config", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--dataset-id")
    parser.add_argument("--mode", choices=("persistent", "fast"))
    parser.add_argument("--list", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    queue_payload = load_queue(args.queue_config)
    entries = queue_payload["entries"]
    if args.list:
        result = {
            "status": "queue_valid",
            "datasets": [
                {
                    "dataset_id": entry["dataset_id"],
                    "acquisition_kind": entry["acquisition_kind"],
                    "priority": entry["priority"],
                    "terms_status": entry["terms_status"],
                    "required_artifact_ids": [
                        artifact["artifact_id"] for artifact in entry.get("required_artifacts", [])
                    ],
                    "generator_revision": entry.get("generator", {}).get("immutable_revision"),
                    "human_action": entry["human_action"],
                }
                for entry in entries
            ],
            "sealed_exclusions": queue_payload["sealed_exclusions"],
        }
    else:
        selected = [entry for entry in entries if args.dataset_id in (None, entry["dataset_id"])]
        if args.dataset_id is not None and not selected:
            raise ValueError("unknown dataset ID")
        result = {
            "status": "queue_processed",
            "results": [
                process_dataset(entry, args.drive_root, mode=args.mode) for entry in selected
            ],
        }
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
