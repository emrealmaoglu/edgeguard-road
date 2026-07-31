#!/usr/bin/env python3
"""Manage crash-safe Colab recovery artifacts and stage completion receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgeguard.rescue.colab_recovery import (
    action_requirements,
    completion_is_valid,
    create_state_archive,
    latest_checkpoint,
    package_interrupted_campaign,
    publish_recovery_file,
    quarantine_incomplete,
    remove_stale_incoming,
    restore_recovery_file,
    restore_state_archive,
    write_campaign_status,
    write_completion_receipt,
)
from edgeguard.serialization import canonical_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--target", required=True)
    plan.add_argument("--allow-final-data", action="store_true")
    plan.add_argument("--provisional-bdd", action="store_true")

    publish = commands.add_parser("publish")
    publish.add_argument("--source", type=Path, required=True)
    publish.add_argument("--store-root", type=Path, required=True)
    publish.add_argument("--artifact-id", required=True)
    publish.add_argument("--campaign-id", required=True)
    publish.add_argument("--project-commit", required=True)
    publish.add_argument("--metadata-json", default="{}")

    restore = commands.add_parser("restore")
    restore.add_argument("--store-root", type=Path, required=True)
    restore.add_argument("--artifact-id", required=True)
    restore.add_argument("--destination", type=Path, required=True)

    complete = commands.add_parser("complete")
    complete.add_argument("--output-root", type=Path, required=True)
    complete.add_argument("--artifact-type", required=True)
    complete.add_argument("--required-path", action="append", required=True)
    complete.add_argument("--input", action="append", default=[])
    complete.add_argument("--metadata-json", default="{}")

    check = commands.add_parser("check")
    check.add_argument("--output-root", type=Path, required=True)

    quarantine = commands.add_parser("quarantine")
    quarantine.add_argument("--output-root", type=Path, required=True)

    checkpoint = commands.add_parser("latest-checkpoint")
    checkpoint.add_argument("--run-dir", type=Path, required=True)

    status = commands.add_parser("status")
    status.add_argument("--output", type=Path, required=True)
    status.add_argument("--values-json", required=True)

    cleanup = commands.add_parser("cleanup-incoming")
    cleanup.add_argument("--store-root", type=Path, required=True)
    pack_state = commands.add_parser("pack-state")
    pack_state.add_argument("--work-root", type=Path, required=True)
    pack_state.add_argument("--output", type=Path, required=True)
    pack_state.add_argument("--include", action="append", required=True)
    pack_state.add_argument("--uncompressed", action="store_true")
    restore_state = commands.add_parser("restore-state")
    restore_state.add_argument("--archive", type=Path, required=True)
    restore_state.add_argument("--destination", type=Path, required=True)
    interrupted = commands.add_parser("package-interruption")
    interrupted.add_argument("--status", type=Path, required=True)
    interrupted.add_argument("--failure-root", type=Path, required=True)
    return parser


def _pairs(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or key in result:
            raise ValueError("--input values must be unique KEY=VALUE pairs")
        result[key] = item
    return result


def main() -> int:
    args = _parser().parse_args()
    if args.command == "plan":
        result = action_requirements(
            args.target,
            allow_final_data=args.allow_final_data,
            provisional_bdd=args.provisional_bdd,
        )
    elif args.command == "publish":
        result = publish_recovery_file(
            args.source.resolve(),
            args.store_root.resolve(),
            artifact_id=args.artifact_id,
            campaign_id=args.campaign_id,
            project_commit=args.project_commit,
            metadata=json.loads(args.metadata_json),
        )
    elif args.command == "restore":
        result = restore_recovery_file(
            args.store_root.resolve(),
            artifact_id=args.artifact_id,
            destination=args.destination.resolve(),
        )
    elif args.command == "complete":
        result = write_completion_receipt(
            args.output_root.resolve(),
            artifact_type=args.artifact_type,
            required_paths=args.required_path,
            inputs=_pairs(args.input),
            metadata=json.loads(args.metadata_json),
        )
    elif args.command == "check":
        result = {"complete": completion_is_valid(args.output_root.resolve())}
    elif args.command == "quarantine":
        path = quarantine_incomplete(args.output_root.resolve())
        result = {"quarantined": str(path) if path else None}
    elif args.command == "latest-checkpoint":
        result = {"checkpoint": str(latest_checkpoint(args.run_dir.resolve()))}
    elif args.command == "status":
        result = write_campaign_status(args.output.resolve(), **json.loads(args.values_json))
    elif args.command == "cleanup-incoming":
        result = {"removed": remove_stale_incoming(args.store_root.resolve())}
    elif args.command == "pack-state":
        result = create_state_archive(
            args.work_root.resolve(),
            args.output.resolve(),
            relative_paths=args.include,
            compression=not args.uncompressed,
        )
    elif args.command == "restore-state":
        result = restore_state_archive(args.archive.resolve(), args.destination.resolve())
    elif args.command == "package-interruption":
        result = package_interrupted_campaign(
            args.status.resolve(), args.failure_root.resolve()
        ) or {"status": "no_interrupted_campaign"}
    else:  # pragma: no cover - argparse enforces a command
        raise AssertionError(args.command)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
