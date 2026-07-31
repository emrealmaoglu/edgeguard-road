"""Pack once in Drive and stage policy-selected Cityscapes training data to Colab."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from edgeguard.serialization import canonical_json, sha256_file
from edgeguard.telemetry.longrun import LongRunStatus, atomic_write_json, ensure_disk_space
from edgeguard.training.data import load_policy_selected_cityscapes_split


def _copy_with_progress(source: Path, partial: Path, status: LongRunStatus) -> None:
    total = source.stat().st_size
    initial = partial.stat().st_size if partial.exists() else 0
    if initial > total:
        raise ValueError("local bundle partial is larger than the Drive bundle")
    ensure_disk_space(partial.parent, total - initial)
    partial.parent.mkdir(parents=True, exist_ok=True)
    completed = initial
    started = time.monotonic()
    with source.open("rb") as incoming, partial.open("ab") as outgoing:
        incoming.seek(initial)
        while chunk := incoming.read(8 * 1024**2):
            outgoing.write(chunk)
            completed += len(chunk)
            elapsed = max(time.monotonic() - started, 1e-9)
            status.update(
                phase="copy-bundle-to-content",
                completed=completed,
                total=total,
                speed_per_second=(completed - initial) / elapsed,
            )
        outgoing.flush()
        os.fsync(outgoing.fileno())
    status.update(completed=completed, total=total, force=True)


def _bundle_paths(samples: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                relative
                for sample in samples
                for relative in (sample.image_relative_path, sample.train_id_relative_path)
            }
        )
    )


def _create_bundle(
    dataset_root: Path,
    paths: tuple[str, ...],
    incoming: Path,
    status: LongRunStatus,
) -> None:
    incoming.parent.mkdir(parents=True, exist_ok=True)
    with (
        incoming.open("xb") as raw_stream,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_stream, mtime=0, compresslevel=6
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        total = len(paths)
        started = time.monotonic()
        for index, relative in enumerate(paths, start=1):
            source = dataset_root / relative
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"Cityscapes bundle source is missing: {relative}")
            info = tarfile.TarInfo(relative)
            info.size = source.stat().st_size
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            with source.open("rb") as stream:
                archive.addfile(info, stream)
            elapsed = max(time.monotonic() - started, 1e-9)
            status.update(
                phase="create-drive-bundle",
                completed=index,
                total=total,
                speed_per_second=index / elapsed,
            )
    status.update(completed=len(paths), total=len(paths), force=True)


def _validate_members(archive: tarfile.TarFile, expected: set[str]) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    names: set[str] = set()
    for member in members:
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or member.issym()
            or member.islnk()
            or not member.isfile()
            or member.name in names
        ):
            raise ValueError(f"unsafe or duplicate Cityscapes bundle member: {member.name}")
        names.add(member.name)
    if names != expected:
        raise ValueError("Cityscapes bundle members differ from the policy-selected manifest")
    return members


def stage_cityscapes_training(
    *,
    dataset_root: Path,
    dataset_manifest_path: Path,
    split_policy_path: Path,
    drive_bundle_directory: Path,
    cache_directory: Path,
    staged_dataset_root: Path,
    allow_bundle_creation: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Create/reuse, copy, verify, and safely extract one deterministic bundle."""
    identity, samples = load_policy_selected_cityscapes_split(
        dataset_manifest_path, split_policy_path
    )
    paths = _bundle_paths(samples)
    bundle_name = (
        f"cityscapes-fine-{identity.dataset_manifest_sha256[:12]}-"
        f"{identity.split_manifest_sha256[:12]}.tar.gz"
    )
    receipt_path = drive_bundle_directory / f"{bundle_name}.receipt.json"
    bundle = drive_bundle_directory / bundle_name
    source_bytes = sum((dataset_root / relative).stat().st_size for relative in paths)
    bundle_reusable = bundle.is_file() and receipt_path.is_file()
    plan = {
        "schema_version": "1.0",
        "record_type": "cityscapes_training_staging_plan",
        "status": "ready" if bundle_reusable else "blocked_missing_reusable_bundle",
        "bundle_name": bundle_name,
        "bundle_reusable": bundle_reusable,
        "expected_download_bytes": 0,
        "expected_drive_write_bytes": 0 if bundle_reusable else source_bytes,
        "expected_local_staging_bytes": bundle.stat().st_size if bundle_reusable else source_bytes,
        "sample_count": len(samples),
        "file_count": len(paths),
    }
    if dry_run:
        return plan
    if not bundle_reusable and not allow_bundle_creation:
        raise ValueError(
            "verified Drive bundle is missing; creation requires explicit --create-bundle"
        )
    drive_bundle_directory.mkdir(parents=True, exist_ok=True)
    status = LongRunStatus(drive_bundle_directory / "run_status.json")
    if bundle.exists():
        if not receipt_path.is_file():
            raise ValueError("existing Drive bundle has no identity receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("sha256") != sha256_file(bundle):
            raise ValueError("existing Drive bundle SHA-256 mismatch")
    else:
        incoming = bundle.with_name(f".{bundle.name}.incoming")
        if incoming.exists():
            raise ValueError("stale Drive bundle incoming file must be inspected")
        _create_bundle(dataset_root, paths, incoming, status)
        bundle_sha256 = sha256_file(incoming)
        receipt = {
            "schema_version": "1.0",
            "record_type": "cityscapes_training_bundle_receipt",
            "filename": bundle_name,
            "byte_size": incoming.stat().st_size,
            "sha256": bundle_sha256,
            "dataset_manifest_sha256": identity.dataset_manifest_sha256,
            "split_manifest_sha256": identity.split_manifest_sha256,
            "file_count": len(paths),
        }
        os.replace(incoming, bundle)
        atomic_write_json(receipt_path, receipt)

    cache_directory.mkdir(parents=True, exist_ok=True)
    local_bundle = cache_directory / bundle_name
    partial = local_bundle.with_name(f"{local_bundle.name}.part")
    if not local_bundle.is_file():
        _copy_with_progress(bundle, partial, status)
        if sha256_file(partial) != receipt["sha256"]:
            raise ValueError("Colab-local bundle SHA-256 mismatch")
        os.replace(partial, local_bundle)
    elif sha256_file(local_bundle) != receipt["sha256"]:
        raise ValueError("existing Colab-local bundle SHA-256 mismatch")

    if staged_dataset_root.exists():
        if not staged_dataset_root.is_dir():
            raise ValueError("staged dataset destination is not a directory")
        if all((staged_dataset_root / relative).is_file() for relative in paths):
            status.complete(last_checkpoint=bundle_name)
            return {**receipt, "status": "already_staged", "sample_count": len(samples)}
        raise ValueError("staged dataset destination is partial or identity-mismatched")
    staging = staged_dataset_root.with_name(f".{staged_dataset_root.name}.incoming")
    if staging.exists():
        raise ValueError("stale extraction staging directory must be inspected")
    with tarfile.open(local_bundle, mode="r:gz") as archive:
        uncompressed_bytes = sum(int(member.size) for member in archive.getmembers())
    ensure_disk_space(staging.parent, uncompressed_bytes)
    staging.mkdir(parents=True)
    with tarfile.open(local_bundle, mode="r:gz") as archive:
        members = _validate_members(archive, set(paths))
        total = len(members)
        started = time.monotonic()
        for index, member in enumerate(members, start=1):
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Cityscapes bundle member has no file content")
            destination = staging / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024**2)
            elapsed = max(time.monotonic() - started, 1e-9)
            status.update(
                phase="extract-content-bundle",
                completed=index,
                total=total,
                speed_per_second=index / elapsed,
            )
    if not all((staging / relative).is_file() for relative in paths):
        raise ValueError("extracted Cityscapes cache failed sample validation")
    staged_dataset_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, staged_dataset_root)
    status.complete(last_checkpoint=bundle_name)
    return {**receipt, "status": "staged_verified", "sample_count": len(samples)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--split-policy-manifest", type=Path, required=True)
    parser.add_argument("--drive-bundle-directory", type=Path, required=True)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--staged-dataset-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--create-bundle", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = stage_cityscapes_training(
        dataset_root=args.dataset_root,
        dataset_manifest_path=args.dataset_manifest,
        split_policy_path=args.split_policy_manifest,
        drive_bundle_directory=args.drive_bundle_directory,
        cache_directory=args.cache_directory,
        staged_dataset_root=args.staged_dataset_root,
        allow_bundle_creation=args.create_bundle,
        dry_run=args.dry_run,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
