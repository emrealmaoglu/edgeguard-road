"""Read-only EdgeGuard storage inventory with a zero-redownload Cityscapes gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from edgeguard.serialization import canonical_json, sha256_file

AREA_PATHS = {
    "archives": "archives",
    "prepared_datasets": "datasets",
    "manifests": "manifests",
    "bundles": "datasets/cityscapes/fine/bundles",
    "checkpoints": "checkpoints",
    "experiments": "experiments",
}
CITYSCAPES_REQUIRED = (
    "datasets/cityscapes/fine/v1",
    "manifests/cityscapes/fine/v1/dataset_manifest.json",
    "manifests/cityscapes/fine/v1/group_summary.json",
    "manifests/cityscapes/fine/v1/split-policy-v1/policy_selected_split.json",
)


def _area(root: Path, relative: str, *, hash_small_files: bool) -> dict[str, Any]:
    path = root / relative
    files = (
        sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
        if path.is_dir()
        else []
    )
    total = sum(candidate.stat().st_size for candidate in files)
    records = []
    for candidate in files:
        record: dict[str, Any] = {
            "relative_path": candidate.relative_to(root).as_posix(),
            "byte_size": candidate.stat().st_size,
        }
        if hash_small_files and candidate.stat().st_size <= 10 * 1024**2:
            record["sha256"] = sha256_file(candidate)
        records.append(record)
    return {
        "relative_root": relative,
        "exists": path.exists(),
        "file_count": len(files),
        "total_bytes": total,
        "files": records,
    }


def inventory_edgeguard_storage(
    root: Path,
    *,
    hash_small_files: bool = False,
) -> dict[str, Any]:
    """Inspect metadata and sizes without writing or hashing large files by default."""
    resolved = root.resolve()
    areas = {
        name: _area(resolved, relative, hash_small_files=hash_small_files)
        for name, relative in AREA_PATHS.items()
    }
    all_files = {
        item["relative_path"]: item for area in areas.values() for item in area["files"]
    }.values()
    duplicate_keys: dict[tuple[str, int], list[str]] = defaultdict(list)
    for item in all_files:
        duplicate_keys[(Path(item["relative_path"]).name, int(item["byte_size"]))].append(
            item["relative_path"]
        )
    potential_duplicates = [
        {"filename": key[0], "byte_size": key[1], "relative_paths": sorted(paths)}
        for key, paths in sorted(duplicate_keys.items())
        if len(paths) > 1
    ]
    missing = [relative for relative in CITYSCAPES_REQUIRED if not (resolved / relative).exists()]
    bundle_area = areas["bundles"]
    bundles = [item for item in bundle_area["files"] if item["relative_path"].endswith(".tar.gz")]
    reusable_bundle = next(
        (
            item
            for item in bundles
            if (resolved / f"{item['relative_path']}.receipt.json").is_file()
        ),
        None,
    )
    cityscapes_reusable = not missing and reusable_bundle is not None
    return {
        "schema_version": "1.0",
        "record_type": "edgeguard_storage_inventory",
        "status": "reusable" if cityscapes_reusable else "blocked_missing_verified_assets",
        "root_basename": resolved.name,
        "exists_only_in_ephemeral_content": (
            resolved == Path("/content") or Path("/content") in resolved.parents
        ),
        "areas": areas,
        "missing_required_files": missing,
        "potential_duplicates": potential_duplicates,
        "cityscapes_fine": {
            "reusable": cityscapes_reusable,
            "verified_bundle_relative_path": (
                reusable_bundle["relative_path"] if reusable_bundle is not None else None
            ),
            "expected_local_staging_bytes": (
                reusable_bundle["byte_size"] if reusable_bundle is not None else None
            ),
            "expected_drive_write_bytes": 0,
            "expected_download_bytes": 0,
            "zero_redownload_policy": True,
            "automatic_large_bundle_creation": False,
        },
        "hash_policy": ("files_up_to_10MiB" if hash_small_files else "metadata_and_sizes_only"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--hash-small-files", action="store_true")
    parser.add_argument("--require-cityscapes-reusable", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = inventory_edgeguard_storage(
        args.external_root,
        hash_small_files=args.hash_small_files,
    )
    print(canonical_json(result))
    if args.require_cityscapes_reusable and not result["cityscapes_fine"]["reusable"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
