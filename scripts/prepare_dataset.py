"""Prepare approved semantic datasets from immutable source archives."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeguard.data.preparation import prepare_dataset
from edgeguard.serialization import canonical_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("cityscapes", "bdd100k", "idd20k", "acdc"), required=True
    )
    parser.add_argument("--archive", action="append", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--ontology",
        type=Path,
        default=REPOSITORY_ROOT / "configs/dataset/semantic_ontology_v2.yaml",
    )
    parser.add_argument(
        "--source-profile", choices=("official", "kaggle_mirror"), default="official"
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--allow-fixture-count",
        action="store_true",
        help="tests only; marks non-official counts scientifically ineligible",
    )
    parser.add_argument(
        "--skip-pinned-archive-hash",
        action="store_true",
        help="fixtures only; official execution must retain pinned SHA-256 checks",
    )
    parser.add_argument("--idd-shard-root", type=Path)
    parser.add_argument("--idd-shard-size", type=int, default=500)
    return parser


def main() -> int:
    """Dispatch one fail-closed preparation operation."""
    args = _parser().parse_args()
    result = prepare_dataset(
        args.dataset,
        tuple(args.archive),
        args.destination.resolve(),
        source_profile=args.source_profile,
        allow_fixture_count=args.allow_fixture_count,
        verify_only=args.verify_only,
        verify_archive_hashes=not args.skip_pinned_archive_hash,
        ontology_path=args.ontology,
        idd_shard_root=args.idd_shard_root,
        idd_shard_size=args.idd_shard_size,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
