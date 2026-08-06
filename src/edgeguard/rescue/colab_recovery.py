"""Crash-safe, low-I/O recovery primitives for ephemeral Colab runtimes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import uuid
import zipfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from edgeguard.serialization import canonical_json, sha256_file, sha256_payload

COMPLETION_NAME = ".edgeguard-completion.json"
TARGETS = (
    "audit",
    "smoke",
    "pilot",
    "screening",
    "hpo",
    "final",
    "evaluate",
    "export",
    "report",
)
TRAINING_STAGES = ("smoke", "pilot", "screening", "final")
RUNTIME_TARGETS = frozenset(TARGETS[1:])


def utc_now() -> str:
    """Return a stable UTC timestamp for append-only evidence."""
    return datetime.now(timezone.utc).isoformat()


def _label(value: str, field: str) -> str:
    if not value or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value) is None:
        raise ValueError(f"{field} must use lowercase ASCII letters, digits, '.', '_' or '-'")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish a small JSON record last, leaving interrupted temporary files ignorable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = path.with_name(f".{path.name}.{uuid.uuid4().hex}.incoming")
    incoming.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(incoming, path)


def _copy_hash(source: Path, destination: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_size = 0
    before = source.stat()
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        while chunk := input_stream.read(8 * 1024**2):
            output_stream.write(chunk)
            digest.update(chunk)
            byte_size += len(chunk)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        destination.unlink(missing_ok=True)
        raise RuntimeError("source artifact changed while it was being published")
    return digest.hexdigest(), byte_size


def _receipt_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid recovery receipt: {path}") from error
    receipt_hash = payload.pop("receipt_sha256", None)
    if receipt_hash != sha256_payload(payload):
        raise ValueError(f"recovery receipt hash mismatch: {path}")
    return payload


def publish_recovery_file(
    source: Path,
    store_root: Path,
    *,
    artifact_id: str,
    campaign_id: str,
    project_commit: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish one immutable object and commit its generation pointer last."""
    artifact_id = _label(artifact_id, "artifact_id")
    campaign_id = _label(campaign_id, "campaign_id")
    if not source.is_file():
        raise FileNotFoundError(f"recovery source is missing: {source}")
    if re.fullmatch(r"[0-9a-f]{40}", project_commit) is None:
        raise ValueError("project_commit must be a full lowercase 40-character Git SHA")
    incoming_root = store_root / "objects" / ".incoming"
    incoming_root.mkdir(parents=True, exist_ok=True)
    incoming = incoming_root / uuid.uuid4().hex
    digest, byte_size = _copy_hash(source, incoming)
    object_path = store_root / "objects" / digest[:2] / digest
    object_path.parent.mkdir(parents=True, exist_ok=True)
    if object_path.exists():
        if object_path.stat().st_size != byte_size:
            incoming.unlink(missing_ok=True)
            raise ValueError("content-addressed object has an impossible size collision")
        incoming.unlink(missing_ok=True)
    else:
        os.replace(incoming, object_path)
    generation = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{digest[:12]}"
    receipt: dict[str, Any] = {
        "schema_version": "2.0",
        "record_type": "edgeguard_recovery_artifact",
        "artifact_id": artifact_id,
        "campaign_id": campaign_id,
        "project_commit": project_commit,
        "generation": generation,
        "created_at": utc_now(),
        "object": object_path.relative_to(store_root).as_posix(),
        "source_name": source.name,
        "byte_size": byte_size,
        "sha256": digest,
        "metadata": metadata or {},
    }
    receipt["receipt_sha256"] = sha256_payload(receipt)
    receipt_path = store_root / "receipts" / artifact_id / f"{generation}.json"
    atomic_json(receipt_path, receipt)
    pointer_path = store_root / "pointers" / f"{artifact_id}.json"
    previous: dict[str, Any] | None = None
    if pointer_path.is_file():
        try:
            current_pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            previous = current_pointer.get("current")
        except (OSError, json.JSONDecodeError):
            previous = None
    pointer = {
        "schema_version": "2.0",
        "record_type": "edgeguard_recovery_pointer",
        "artifact_id": artifact_id,
        "current": {
            "receipt": receipt_path.relative_to(store_root).as_posix(),
            "receipt_sha256": receipt["receipt_sha256"],
        },
        "previous": previous,
        "updated_at": utc_now(),
    }
    pointer["pointer_sha256"] = sha256_payload(pointer)
    atomic_json(pointer_path, pointer)
    return receipt


def _pointer_candidates(store_root: Path, artifact_id: str) -> list[Path]:
    pointer_path = store_root / "pointers" / f"{_label(artifact_id, 'artifact_id')}.json"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"recovery pointer is missing or invalid: {pointer_path}") from error
    pointer_hash = pointer.pop("pointer_sha256", None)
    if pointer_hash != sha256_payload(pointer):
        raise ValueError("recovery pointer hash mismatch")
    result: list[Path] = []
    for name in ("current", "previous"):
        row = pointer.get(name)
        if not isinstance(row, dict):
            continue
        relative = PurePosixPath(str(row.get("receipt", "")))
        if relative.is_absolute() or ".." in relative.parts:
            continue
        receipt = store_root.joinpath(*relative.parts)
        if receipt.is_file():
            result.append(receipt)
    return result


def restore_recovery_file(
    store_root: Path, *, artifact_id: str, destination: Path
) -> dict[str, Any]:
    """Restore current generation, falling back to previous, into an atomic local path."""
    errors: list[str] = []
    for receipt_path in _pointer_candidates(store_root, artifact_id):
        incoming: Path | None = None
        try:
            receipt = _receipt_payload(receipt_path)
            relative = PurePosixPath(str(receipt["object"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe recovery object path")
            source = store_root.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            incoming = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.incoming")
            digest, byte_size = _copy_hash(source, incoming)
            if digest != receipt["sha256"] or byte_size != int(receipt["byte_size"]):
                raise ValueError("recovery payload hash or size mismatch")
            os.replace(incoming, destination)
            return {
                "status": "restored_verified",
                "artifact_id": artifact_id,
                "generation": receipt["generation"],
                "receipt": str(receipt_path),
                "destination": str(destination),
                "sha256": digest,
                "metadata": receipt.get("metadata", {}),
                "campaign_id": receipt.get("campaign_id"),
                "project_commit": receipt.get("project_commit"),
            }
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            if incoming is not None:
                incoming.unlink(missing_ok=True)
            errors.append(f"{receipt_path}: {error}")
    raise ValueError("no valid recovery generation found; " + " | ".join(errors))


def create_state_archive(
    work_root: Path,
    output: Path,
    *,
    relative_paths: Iterable[str],
    compression: bool = True,
) -> dict[str, Any]:
    """Create a small local campaign-state archive without model or dataset payloads."""
    excluded_suffixes = {".pth", ".onnx", ".engine", ".plan"}
    excluded_parts = {"datasets", "canonical_masks", "checkpoints"}
    selected: list[tuple[Path, str]] = []
    for raw in sorted(set(relative_paths)):
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe state archive source: {raw}")
        source = work_root.joinpath(*relative.parts)
        if source.exists():
            selected.append((source, relative.as_posix()))
    output.parent.mkdir(parents=True, exist_ok=True)
    incoming = output.with_name(f".{output.name}.{uuid.uuid4().hex}.incoming")
    file_count = 0
    with tarfile.open(incoming, "w:gz" if compression else "w") as archive:
        for source, _arcname in selected:
            paths = [source] if source.is_file() else sorted(source.rglob("*"))
            for path in paths:
                if path.is_symlink():
                    raise ValueError(f"state source contains a symlink: {path}")
                if not path.is_file():
                    continue
                archived_relative = path.relative_to(work_root)
                if (
                    excluded_parts.intersection(archived_relative.parts)
                    or path.suffix in excluded_suffixes
                ):
                    continue
                if path.name == "last_checkpoint":
                    continue
                info = archive.gettarinfo(str(path), arcname=archived_relative.as_posix())
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
                file_count += 1
    os.replace(incoming, output)
    return {"path": str(output), "file_count": file_count, "sha256": sha256_file(output)}


def restore_state_archive(archive_path: Path, destination: Path) -> dict[str, Any]:
    """Validate a local state archive and atomically publish its extracted root."""
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("state restore destination must be empty")
    temporary = temporary_directory(destination.parent)
    try:
        with tarfile.open(archive_path, "r:*") as archive:
            members = archive.getmembers()
            for member in members:
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or member.issym()
                    or member.islnk()
                    or not member.isfile()
                ):
                    raise ValueError(f"unsafe state archive member: {member.name}")
            archive.extractall(temporary, filter="data")
        if destination.exists():
            destination.rmdir()
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"status": "restored_verified", "destination": str(destination)}


def write_completion_receipt(
    output_root: Path,
    *,
    artifact_type: str,
    required_paths: Iterable[str],
    inputs: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Commit stage completion only after all required output hashes are known."""
    if not output_root.is_dir():
        raise FileNotFoundError(f"completion root is missing: {output_root}")
    files: list[dict[str, Any]] = []
    for raw in sorted(set(required_paths)):
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe completion path: {raw}")
        path = output_root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"required completion output is missing: {path}")
        files.append(
            {
                "path": relative.as_posix(),
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError("completion receipt requires at least one output file")
    receipt: dict[str, Any] = {
        "schema_version": "2.0",
        "record_type": "edgeguard_stage_completion",
        "artifact_type": artifact_type,
        "completed_at": utc_now(),
        "files": files,
        "inputs": inputs or {},
        "metadata": metadata or {},
    }
    receipt["receipt_sha256"] = sha256_payload(receipt)
    atomic_json(output_root / COMPLETION_NAME, receipt)
    return receipt


def completion_is_valid(
    output_root: Path, *, expected_inputs: dict[str, str] | None = None
) -> bool:
    """Return true only when a completion receipt and every bound output still match."""
    receipt_path = output_root / COMPLETION_NAME
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_hash = receipt.pop("receipt_sha256")
        if receipt_hash != sha256_payload(receipt):
            return False
        if expected_inputs is not None and receipt.get("inputs") != expected_inputs:
            return False
        for row in receipt["files"]:
            relative = PurePosixPath(str(row["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                return False
            path = output_root.joinpath(*relative.parts)
            if (
                not path.is_file()
                or path.stat().st_size != int(row["byte_size"])
                or sha256_file(path) != row["sha256"]
            ):
                return False
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def quarantine_incomplete(output_root: Path) -> Path | None:
    """Move a partial stage aside so a clean idempotent retry cannot mistake it for success."""
    if not output_root.exists() or completion_is_valid(output_root):
        return None
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    quarantine = output_root.with_name(f"{output_root.name}.incomplete-{suffix}")
    os.replace(output_root, quarantine)
    return quarantine


def write_campaign_status(path: Path, **values: Any) -> dict[str, Any]:
    """Write the latest bounded heartbeat/status record to persistent storage."""
    payload = {
        "schema_version": "2.0",
        "record_type": "edgeguard_campaign_status",
        "updated_at": utc_now(),
        **values,
    }
    payload["status_sha256"] = sha256_payload(payload)
    atomic_json(path, payload)
    return payload


def package_interrupted_campaign(status_path: Path, failure_root: Path) -> dict[str, Any] | None:
    """Turn a previous running heartbeat into durable interruption evidence once."""
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status_hash = status.pop("status_sha256")
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    if status_hash != sha256_payload(status) or status.get("state") != "running":
        return None
    failure_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-session-reset"
    root = failure_root / failure_id
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": "2.0",
        "record_type": "edgeguard_interrupted_campaign",
        "failure_id": failure_id,
        "detected_at": utc_now(),
        "reason": "ephemeral_runtime_ended_while_campaign_status_was_running",
        "last_verified_status": {**status, "status_sha256": status_hash},
        "automatic_recovery": True,
    }
    report["report_sha256"] = sha256_payload(report)
    report_path = root / "failure.json"
    atomic_json(report_path, report)
    package = root / "failure-report.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, arcname="failure.json")
    write_campaign_status(
        status_path,
        state="interrupted_detected",
        previous_status_sha256=status_hash,
        recovery="automatic",
        failure_package=str(package),
    )
    return {
        "failure_id": failure_id,
        "report": str(report_path),
        "package": str(package),
        "package_sha256": sha256_file(package),
    }


def latest_checkpoint(run_dir: Path) -> Path:
    """Resolve MMEngine last_checkpoint metadata, never lexicographic filename order."""
    marker = run_dir / "last_checkpoint"
    if marker.is_file():
        raw = marker.read_text(encoding="utf-8").strip()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        if candidate.is_file() and candidate.parent.resolve() == run_dir.resolve():
            return candidate
    numbered: list[tuple[int, Path]] = []
    for path in run_dir.glob("iter_*.pth"):
        match = re.fullmatch(r"iter_(\d+)\.pth", path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    if numbered:
        return max(numbered, key=lambda row: row[0])[1]
    checkpoints = [path for path in run_dir.glob("*.pth") if path.is_file()]
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint found in {run_dir}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime_ns)


def planned_stages(target: str) -> tuple[str, ...]:
    """Return ordered campaign prerequisites for one notebook target."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of: {', '.join(TARGETS)}")
    order = ("audit", "smoke", "pilot", "screening", "hpo", "final", "evaluate", "export", "report")
    return order[: order.index(target) + 1]


def action_requirements(
    target: str, *, allow_final_data: bool = False, provisional_bdd: bool = False
) -> dict[str, Any]:
    """Resolve staging/runtime needs without touching Drive or the local runtime."""
    stages = planned_stages(target)
    datasets: list[str] = []
    datasets.extend(("cityscapes", "idd20k"))
    if provisional_bdd:
        datasets.append("bdd100k")
    if allow_final_data and target in {"evaluate", "export", "report"}:
        datasets.append("cityscapes_official_val")
    return {
        "target": target,
        "stages": list(stages),
        "training_stages": [stage for stage in stages if stage in TRAINING_STAGES],
        "datasets": datasets,
        "runtime_required": target in RUNTIME_TARGETS,
        "final_data_allowed": allow_final_data,
    }


def temporary_directory(parent: Path) -> Path:
    """Create a local temporary directory for verified atomic restore operations."""
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="edgeguard-", dir=parent))


def remove_stale_incoming(store_root: Path) -> int:
    """Remove only recognized orphaned incoming files from the recovery object store."""
    incoming = store_root / "objects" / ".incoming"
    if not incoming.is_dir():
        return 0
    count = 0
    for path in incoming.iterdir():
        if path.is_file():
            path.unlink()
            count += 1
        elif path.is_dir():
            shutil.rmtree(path)
            count += 1
    return count
