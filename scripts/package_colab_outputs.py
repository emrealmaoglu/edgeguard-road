#!/usr/bin/env python3
"""Create or restore bounded EdgeGuard Colab output packages."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeguard.rescue.colab_artifacts import (
    create_campaign_snapshot,
    create_review_package,
    restore_campaign_snapshot,
)
from edgeguard.serialization import canonical_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    review = subparsers.add_parser("review")
    review.add_argument("--source-root", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--campaign-id", required=True)
    review.add_argument("--project-commit", required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--work-root", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--include", action="append", required=True)
    snapshot.add_argument("--campaign-id", required=True)
    snapshot.add_argument("--project-commit", required=True)
    restore = subparsers.add_parser("restore")
    restore.add_argument("--snapshot", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "review":
        result = create_review_package(
            args.source_root,
            args.output,
            campaign_id=args.campaign_id,
            project_commit=args.project_commit,
        )
    elif args.command == "snapshot":
        result = create_campaign_snapshot(
            args.work_root,
            args.output,
            relative_paths=args.include,
            campaign_id=args.campaign_id,
            project_commit=args.project_commit,
        )
    else:
        result = restore_campaign_snapshot(args.snapshot, args.destination)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
