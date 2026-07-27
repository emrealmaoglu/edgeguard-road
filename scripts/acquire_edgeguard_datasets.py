"""Project-specific resumable acquisition queue for approved EdgeGuard datasets."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from edgeguard.config import UniqueKeySafeLoader
from edgeguard.serialization import canonical_json, sha256_file
from edgeguard.telemetry.longrun import LiveCommandRunner, LongRunStatus, atomic_write_json


def load_queue(path: Path) -> dict[str, Any]:
    """Load the fixed acquisition queue and reject sealed-development entries."""
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError("acquisition queue is missing or malformed") from error
    if (
        not isinstance(payload, dict)
        or payload.get("record_type") != "edgeguard_dataset_acquisition_queue"
    ):
        raise ValueError("invalid acquisition queue record")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("acquisition queue has no entries")
    identifiers: set[str] = set()
    prohibited = {"SMIYC RoadAnomaly21", "SMIYC RoadObstacle21", "Fishyscapes Lost & Found"}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("dataset_id"), str):
            raise ValueError("acquisition queue contains an invalid entry")
        dataset_id = entry["dataset_id"]
        if dataset_id in identifiers:
            raise ValueError("acquisition queue contains a duplicate dataset ID")
        identifiers.add(dataset_id)
        serialized = canonical_json(entry)
        if any(name in serialized for name in prohibited):
            raise ValueError("sealed or frozen-holdout data cannot enter the acquisition queue")
    return payload


def _safe_filename(value: str) -> str:
    path = Path(value)
    if path.name != value or value in {"", ".", ".."}:
        raise ValueError("runtime archive filename must be one safe path component")
    return value


def _runtime_value(entry: dict[str, Any], field: str) -> str | None:
    variable = entry.get(field)
    if not isinstance(variable, str):
        raise ValueError(f"queue entry is missing {field}")
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
    completed = partial.stat().st_size if partial.exists() else 0
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
                speed_per_second=(completed - (partial.stat().st_size - len(chunk))) / elapsed,
            )
        outgoing.flush()
        os.fsync(outgoing.fileno())
    status.update(completed=completed, total=total, force=True)


def _verify_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    if path.stat().st_size != expected_size:
        raise ValueError("downloaded archive byte size mismatch")
    if sha256_file(path) != expected_sha256:
        raise ValueError("downloaded archive SHA-256 mismatch")


def acquire_entry(
    entry: dict[str, Any],
    drive_root: Path,
    *,
    mode: str | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    """Acquire one approved runtime-supplied archive without printing credentials."""
    dataset_id = str(entry["dataset_id"])
    if entry.get("acquisition_status") == "blocked_until_dataset_choice":
        return {
            "dataset_id": dataset_id,
            "status": "blocked_policy",
            "human_action": entry["human_action"],
        }
    url = _runtime_value(entry, "url_env")
    filename_value = _runtime_value(entry, "filename_env")
    sha256_value = _runtime_value(entry, "sha256_env")
    size_value = _runtime_value(entry, "size_env")
    missing = [
        name
        for name, value in (
            (entry["url_env"], url),
            (entry["filename_env"], filename_value),
            (entry["sha256_env"], sha256_value),
            (entry["size_env"], size_value),
        )
        if not value
    ]
    if missing:
        return {
            "dataset_id": dataset_id,
            "status": "blocked_access",
            "missing_runtime_variables": missing,
            "human_action": entry["human_action"],
        }
    assert url is not None and filename_value is not None
    assert sha256_value is not None and size_value is not None
    filename = _safe_filename(filename_value)
    if len(sha256_value) != 64 or any(
        character not in "0123456789abcdef" for character in sha256_value
    ):
        raise ValueError("expected archive SHA-256 must be 64 lowercase hexadecimal characters")
    expected_size = int(size_value)
    if expected_size <= 0:
        raise ValueError("expected archive byte size must be positive")
    selected_mode = mode or str(entry["default_mode"])
    archive_directory = drive_root / "archives" / dataset_id
    manifest_directory = drive_root / "manifests" / dataset_id
    final = archive_directory / filename
    manifest_directory.mkdir(parents=True, exist_ok=True)
    status = LongRunStatus(manifest_directory / "run_status.json")
    if final.is_file():
        _verify_file(final, expected_size, sha256_value)
        status.complete(last_checkpoint=filename)
        return {"dataset_id": dataset_id, "status": "already_verified", "filename": filename}

    archive_directory.mkdir(parents=True, exist_ok=True)
    if selected_mode == "persistent":
        partial = drive_root / "downloads" / "partial" / f"{filename}.part"
        partial.parent.mkdir(parents=True, exist_ok=True)
        command = _download_command(url, partial)
        runner = LiveCommandRunner(manifest_directory / "logs", status)
        last_error: BaseException | None = None
        for attempt in range(1, retries + 1):
            try:
                runner.run(
                    f"download-{dataset_id}-attempt-{attempt}",
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
        _verify_file(partial, expected_size, sha256_value)
        os.replace(partial, final)
    elif selected_mode == "fast":
        local = Path("/content/edgeguard-downloads") / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        runner = LiveCommandRunner(manifest_directory / "logs", status)
        command = _download_command(url, local)
        runner.run(
            f"download-{dataset_id}",
            command,
            display_command=(*command[:-1], "<runtime-url>"),
            stage_index=1,
            stage_total=2,
            progress_probe=lambda: (local.stat().st_size if local.exists() else 0, expected_size),
            redact_values=(url,),
        )
        _verify_file(local, expected_size, sha256_value)
        incoming = final.with_name(f".{final.name}.incoming")
        _copy_resumable(local, incoming, status)
        _verify_file(incoming, expected_size, sha256_value)
        os.replace(incoming, final)
    else:
        raise ValueError("acquisition mode must be persistent or fast")

    receipt = {
        "schema_version": "1.0",
        "record_type": "edgeguard_dataset_acquisition_receipt",
        "dataset_id": dataset_id,
        "source": entry["source_reference"],
        "access_date": datetime.now(timezone.utc).date().isoformat(),
        "terms_status": entry["terms_status"],
        "filename": filename,
        "byte_size": expected_size,
        "sha256": sha256_value,
        "completion_status": "verified",
        "extraction_status": "not_extracted",
        "mode": selected_mode,
    }
    atomic_write_json(manifest_directory / "acquisition_receipt.json", receipt)
    status.complete(last_checkpoint=filename)
    return receipt


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
            "entries": [
                {
                    "dataset_id": entry["dataset_id"],
                    "priority": entry["priority"],
                    "terms_status": entry["terms_status"],
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
                acquire_entry(entry, args.drive_root, mode=args.mode) for entry in selected
            ],
        }
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
