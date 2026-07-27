"""Verify a small semantic stack-probe evidence package without framework imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from edgeguard.serialization import canonical_json, sha256_file
from edgeguard.training.contracts import CheckpointMetadata, ExperimentRegistryRecord

EXPECTED_FILES = {
    "checkpoint_metadata.json",
    "config_receipt.json",
    "environment.json",
    "registry.jsonl",
    "stack_probe_summary.json",
}


def verify_package(path: Path, expected_sha256: str) -> dict[str, object]:
    """Verify hash, safe members, schemas, and five-model non-scientific status."""
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("semantic stack evidence SHA-256 mismatch")
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            if set(names) != EXPECTED_FILES or len(names) != len(set(names)):
                raise ValueError("semantic stack evidence member set mismatch")
            for name in names:
                member = PurePosixPath(name)
                if member.is_absolute() or ".." in member.parts or "\\" in name:
                    raise ValueError("unsafe semantic stack evidence member")
            summary = json.loads(archive.read("stack_probe_summary.json"))
            checkpoint = json.loads(archive.read("checkpoint_metadata.json"))
            registry_lines = archive.read("registry.jsonl").decode().splitlines()
    except (BadZipFile, KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("semantic stack evidence package is malformed") from error
    CheckpointMetadata.model_validate(checkpoint)
    records = [ExperimentRegistryRecord.model_validate_json(line) for line in registry_lines]
    if summary.get("model_count") != 5 or len(records) != 5:
        raise ValueError("semantic stack evidence must contain all five model probes")
    if summary.get("scientific_accuracy_evidence") is not False:
        raise ValueError("synthetic stack evidence must reject scientific claims")
    return {
        "schema_version": "1.0",
        "record_type": "semantic_stack_evidence_verification",
        "sha256": actual_sha256,
        "model_count": 5,
        "scientific_accuracy_evidence": False,
        "valid": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(canonical_json(verify_package(args.package, args.sha256)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
