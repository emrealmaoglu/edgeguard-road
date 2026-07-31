"""Bounded Colab campaign snapshots and human-review packages."""

from __future__ import annotations

import json
import os
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from edgeguard.serialization import canonical_json, sha256_file, sha256_payload

REVIEW_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".jpeg",
    ".jpg",
    ".log",
    ".md",
    ".pdf",
    ".png",
    ".txt",
    ".yaml",
    ".yml",
}
REVIEW_EXCLUDED_PARTS = {
    "canonical_masks",
    "checkpoints",
    "datasets",
    "engines",
    "onnx",
    "prepared",
}
DEFAULT_REVIEW_FILE_LIMIT = 25 * 1024**2


def _validate_label(label: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
    if not label or any(character not in allowed for character in label):
        raise ValueError("artifact label must use lowercase ASCII letters, digits, '-' or '_'")
    return label


def _iter_review_files(root: Path, maximum_file_bytes: int) -> list[Path]:
    selected: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"review source contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if REVIEW_EXCLUDED_PARTS.intersection(relative.parts):
            continue
        if path.suffix.lower() not in REVIEW_EXTENSIONS:
            continue
        if path.stat().st_size > maximum_file_bytes:
            continue
        selected.append(path)
    return selected


def create_review_package(
    source_root: Path,
    output_zip: Path,
    *,
    campaign_id: str,
    project_commit: str,
    maximum_file_bytes: int = DEFAULT_REVIEW_FILE_LIMIT,
) -> dict[str, Any]:
    """Create a deterministic, small review ZIP that cannot include model/data payloads."""
    if not source_root.is_dir():
        raise ValueError(f"review source root is missing: {source_root}")
    if len(project_commit) != 40:
        raise ValueError("project commit must be a 40-character Git commit")
    _validate_label(campaign_id)
    files = _iter_review_files(source_root, maximum_file_bytes)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    incoming = output_zip.with_name(f".{output_zip.name}.incoming")
    incoming.unlink(missing_ok=True)
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(incoming, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(source_root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            payload = path.read_bytes()
            archive.writestr(info, payload)
            rows.append({"path": relative, "bytes": len(payload), "sha256": sha256_file(path)})
        manifest = {
            "schema_version": "1.0",
            "record_type": "edgeguard_colab_review_manifest",
            "campaign_id": campaign_id,
            "project_commit": project_commit,
            "files": rows,
        }
        archive.writestr(
            zipfile.ZipInfo("REVIEW_MANIFEST.json", date_time=(1980, 1, 1, 0, 0, 0)),
            canonical_json(manifest) + "\n",
        )
    os.replace(incoming, output_zip)
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "edgeguard_colab_review_package",
        "campaign_id": campaign_id,
        "project_commit": project_commit,
        "path": str(output_zip),
        "file_count": len(rows),
        "byte_size": output_zip.stat().st_size,
        "sha256": sha256_file(output_zip),
    }
    receipt["receipt_sha256"] = sha256_payload(receipt)
    receipt_path = output_zip.with_suffix(output_zip.suffix + ".receipt.json")
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def _validated_sources(root: Path, relative_paths: Iterable[str]) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for raw in relative_paths:
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe snapshot source: {raw}")
        normalized = relative.as_posix()
        if normalized in seen:
            raise ValueError(f"duplicate snapshot source: {normalized}")
        seen.add(normalized)
        source = root.joinpath(*relative.parts)
        if source.is_symlink():
            raise ValueError(f"snapshot source is a symlink: {source}")
        if source.exists():
            rows.append((source, normalized))
    return rows


def create_campaign_snapshot(
    work_root: Path,
    snapshot: Path,
    *,
    relative_paths: Iterable[str],
    campaign_id: str,
    project_commit: str,
) -> dict[str, Any]:
    """Atomically snapshot only explicit campaign state, never the staged datasets."""
    _validate_label(campaign_id)
    if len(project_commit) != 40:
        raise ValueError("project commit must be a 40-character Git commit")
    sources = _validated_sources(work_root, relative_paths)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    incoming = snapshot.with_name(f".{snapshot.name}.incoming")
    incoming.unlink(missing_ok=True)
    with tarfile.open(incoming, "w:gz") as archive:
        for source, arcname in sources:
            members = [source] if source.is_file() else sorted(source.rglob("*"))
            for member in members:
                if member.is_symlink():
                    raise ValueError(f"snapshot source contains a symlink: {member}")
                if not member.is_file():
                    continue
                suffix = member.relative_to(source).as_posix() if source.is_dir() else ""
                target = PurePosixPath(arcname, suffix).as_posix()
                info = archive.gettarinfo(str(member), arcname=target)
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with member.open("rb") as stream:
                    archive.addfile(info, stream)
    receipt_path = snapshot.with_suffix(snapshot.suffix + ".receipt.json")
    previous = snapshot.with_name(f"{snapshot.stem}.previous{snapshot.suffix}")
    previous_receipt = previous.with_suffix(previous.suffix + ".receipt.json")
    if snapshot.exists():
        if not receipt_path.is_file():
            raise ValueError("existing snapshot has no receipt; inspect before replacing")
        os.replace(snapshot, previous)
        os.replace(receipt_path, previous_receipt)
    os.replace(incoming, snapshot)
    receipt: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "edgeguard_colab_campaign_snapshot",
        "campaign_id": campaign_id,
        "project_commit": project_commit,
        "path": str(snapshot),
        "relative_paths": [value for _, value in sources],
        "byte_size": snapshot.stat().st_size,
        "sha256": sha256_file(snapshot),
    }
    receipt["receipt_sha256"] = sha256_payload(receipt)
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def restore_campaign_snapshot(snapshot: Path, destination: Path) -> dict[str, Any]:
    """Verify and safely restore a campaign snapshot into an empty work root."""
    receipt_path = snapshot.with_suffix(snapshot.suffix + ".receipt.json")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("snapshot receipt is missing or invalid") from error
    receipt_hash = receipt.pop("receipt_sha256", None)
    if receipt_hash != sha256_payload(receipt):
        raise ValueError("snapshot receipt hash mismatch")
    if receipt.get("sha256") != sha256_file(snapshot):
        raise ValueError("snapshot hash mismatch")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("snapshot destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(snapshot, "r:gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise ValueError(f"unsafe snapshot member: {member.name}")
        archive.extractall(destination, filter="data")
    return {
        "status": "restored_verified",
        "campaign_id": receipt["campaign_id"],
        "project_commit": receipt["project_commit"],
        "destination": str(destination),
    }
