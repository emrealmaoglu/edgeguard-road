"""Rebuild Cityscapes Fine D/E split policy from prepared manifest evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from edgeguard.data.cityscapes_split_policy import build_diversity_split_policy
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--group-summary", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is missing or malformed") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def rebuild_cityscapes_splits(
    dataset_manifest_path: Path,
    group_summary_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Write one deterministic policy-selected split without touching dataset files."""
    if output_directory.exists():
        raise ValueError("split-policy output directory already exists; overwrite is not permitted")
    incoming = output_directory.with_name(f".{output_directory.name}.incoming")
    if incoming.exists():
        raise ValueError("stale split-policy incoming directory exists")
    dataset_manifest = _load_object(dataset_manifest_path, "dataset manifest")
    group_summary = _load_object(group_summary_path, "group summary")
    result = build_diversity_split_policy(dataset_manifest, group_summary)

    incoming.mkdir(parents=True)
    candidate_directory = incoming / "split_candidates"
    candidate_directory.mkdir()
    for candidate in result["candidates"]:
        candidate_id = candidate["candidate_id"]
        _write_json(
            candidate_directory / f"{candidate_id}.samples.json",
            {
                "schema_version": "1.0",
                "record_type": "cityscapes_fine_diversity_split_sample_manifest",
                "status": candidate["status"],
                "policy_version": candidate["policy_version"],
                "candidate_id": candidate_id,
                "candidate_sha256": candidate["candidate_sha256"],
                "samples": candidate["sample_manifest"],
            },
        )
        _write_json(
            candidate_directory / f"{candidate_id}.groups.json",
            {
                "schema_version": "1.0",
                "record_type": "cityscapes_fine_diversity_split_group_manifest",
                "status": candidate["status"],
                "policy_version": candidate["policy_version"],
                "candidate_id": candidate_id,
                "candidate_sha256": candidate["candidate_sha256"],
                "groups": candidate["group_manifest"],
            },
        )
    _write_json(incoming / "split_policy_result.json", result)
    _write_json(incoming / "policy_selected_split.json", result["selected_manifest"])
    identities: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_split_policy_file_identities",
        "files": [
            {
                "relative_path": path.relative_to(incoming).as_posix(),
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(incoming.rglob("*.json"))
        ],
    }
    identities["manifest_sha256"] = sha256_payload(identities)
    _write_json(incoming / "file_identities.json", identities)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    os.replace(incoming, output_directory)
    return {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_split_rebuild_completion",
        "status": "policy_selected",
        "policy_version": result["policy_version"],
        "policy_config_sha256": result["policy_config_sha256"],
        "candidate_id": result["selected_candidate_id"],
        "candidate_sha256": result["selected_candidate_sha256"],
        "dataset_manifest_sha256": result["dataset_manifest_sha256"],
        "ontology_sha256": result["ontology_sha256"],
        "selected_manifest_sha256": result["selected_manifest"]["manifest_sha256"],
        "output_relative_files": [
            path.relative_to(output_directory).as_posix()
            for path in sorted(output_directory.rglob("*"))
            if path.is_file()
        ],
    }


def main() -> int:
    args = _parser().parse_args()
    try:
        result = rebuild_cityscapes_splits(
            args.dataset_manifest,
            args.group_summary,
            args.output_directory,
        )
    except (OSError, ValueError) as error:
        message = str(error)
        for path in (args.dataset_manifest, args.group_summary, args.output_directory):
            message = message.replace(str(path), f"<{path.name or 'runtime-path'}>")
        print(
            canonical_json(
                {"status": "error", "error_type": type(error).__name__, "error": message}
            ),
            file=sys.stderr,
        )
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
