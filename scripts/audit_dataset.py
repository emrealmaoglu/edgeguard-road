"""Audit one approved semantic dataset and produce a leakage-safe manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from edgeguard.rescue.dataset import (
    SemanticSample,
    audit_cityscapes,
    validate_or_build_split,
    write_train_fit_statistics,
)
from edgeguard.rescue.multidomain import (
    audit_evaluation_dataset,
    audit_training_dataset,
    build_cityscapes_dataset_manifest,
    freeze_candidate_manifest,
    write_multidomain_statistics,
)
from edgeguard.serialization import canonical_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("cityscapes", "bdd100k", "idd20k", "acdc", "wilddash2", "muses", "kitti"),
        default="cityscapes",
    )
    parser.add_argument("--pairs-file", type=Path)
    parser.add_argument("--data-manifest", type=Path, action="append", default=[])
    parser.add_argument("--source-manifest", type=Path, action="append", default=[])
    parser.add_argument("--source-url")
    parser.add_argument("--license-id")
    parser.add_argument("--access-date")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="persistent catalog chunks used to resume long canonical-data audits",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--source-split",
        choices=("train", "val"),
        default="train",
        help="BDD/IDD train partitioning or withheld official validation audit",
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=REPOSITORY_ROOT / "configs/dataset/semantic_ontology_v2.yaml",
    )
    parser.add_argument(
        "--allow-fixture-count",
        action="store_true",
        help="test-only escape hatch; scientific audits require official counts",
    )
    parser.add_argument(
        "--freeze-approved",
        action="store_true",
        help="freeze the reviewed --split-manifest instead of auditing data",
    )
    return parser


def main() -> int:
    """Run the audit before allowing any scientific training."""
    args = _parser().parse_args()
    if args.data_manifest:
        result = write_multidomain_statistics(
            tuple(path.resolve() for path in args.data_manifest), args.output_root.resolve()
        )
        print(canonical_json(result))
        return 0
    if args.freeze_approved:
        if args.split_manifest is None:
            raise ValueError("manifest freeze requires --split-manifest")
        destination = args.output_root.resolve() / f"{args.dataset}.frozen.json"
        result = freeze_candidate_manifest(args.split_manifest.resolve(), destination)
        print(canonical_json(result))
        return 0
    if args.dataset_root is None:
        raise ValueError("dataset audit requires --dataset-root")
    if args.dataset in {"bdd100k", "idd20k"}:
        result = audit_training_dataset(
            args.dataset_root.resolve(),
            args.output_root.resolve(),
            dataset_id=args.dataset,
            ontology_path=args.ontology.resolve(),
            seed=args.seed,
            strict_count=not args.allow_fixture_count,
            source_split=args.source_split,
            source_manifests=tuple(path.resolve() for path in args.source_manifest),
            checkpoint_root=(args.checkpoint_root.resolve() if args.checkpoint_root else None),
        )
        print(canonical_json(result))
        return 0 if result["audit_passed"] else 2
    if args.dataset != "cityscapes":
        if not all((args.pairs_file, args.source_url, args.license_id, args.access_date)):
            raise ValueError(
                "evaluation-only audit requires --pairs-file, --source-url, --license-id, "
                "and --access-date"
            )
        result = audit_evaluation_dataset(
            args.dataset_root.resolve(),
            args.output_root.resolve(),
            dataset_id=args.dataset,
            pairs_file=args.pairs_file.resolve(),
            ontology_path=args.ontology.resolve(),
            source_url=args.source_url,
            license_id=args.license_id,
            access_date=args.access_date,
            source_manifests=tuple(path.resolve() for path in args.source_manifest),
        )
        print(canonical_json(result))
        return 0 if result["audit_passed"] else 2
    summary = audit_cityscapes(args.dataset_root.resolve(), args.output_root.resolve())
    samples = [SemanticSample(**record) for record in summary.pop("samples")]
    split_result: dict[str, object] | None = None
    role_statistics: dict[str, Any] | None = None
    if summary["audit_passed"]:
        manifest_path = args.output_root / "dataset_audit" / "CSF-SPLIT-D.json"
        split_result = validate_or_build_split(
            samples,
            manifest_path,
            seed=args.seed,
            existing_path=args.split_manifest,
        )
        role_statistics = write_train_fit_statistics(
            args.dataset_root.resolve(), manifest_path, args.output_root / "dataset_audit"
        )
        build_cityscapes_dataset_manifest(
            args.dataset_root.resolve(),
            manifest_path,
            args.output_root / "dataset_audit" / "dataset_manifest.candidate.json",
            audit_passed=True,
            frozen=args.split_manifest is not None,
        )
    result = {
        "schema_version": "1.0",
        "record_type": "semantic_first_audit_result",
        "audit": summary,
        "split": split_result,
        "role_statistics": role_statistics,
        "training_allowed": bool(
            summary["audit_passed"]
            and split_result is not None
            and role_statistics is not None
            and role_statistics["split_summary"]["all_roles_have_all_classes"]
        ),
    }
    print(canonical_json(result))
    return 0 if result["training_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
