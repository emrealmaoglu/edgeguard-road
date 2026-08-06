"""Plan or run the EdgeGuard semantic-cs-idd-v3 Colab production pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeguard.rescue.colab_pipeline import ALL_MODELS, TARGETS, ColabPipeline, PipelineInputs
from edgeguard.serialization import canonical_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"))
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument(
        "--execution-mode", choices=("production", "acceptance"), default="production"
    )
    parser.add_argument("--campaign-id", default="semantic-cs-idd-v3")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--mmseg-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--state-store-root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, action="append", required=True)
    parser.add_argument("--candidate-table", type=Path)
    parser.add_argument("--rare-classes-file", type=Path)
    parser.add_argument("--class-weights-file", type=Path)
    parser.add_argument("--authorization-policy", type=Path, required=True)
    parser.add_argument("--release-policy", type=Path)
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="staged data root; production all/evaluate requires cityscapes and idd20k",
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        action="append",
        default=[],
        help="frozen official_source_val manifest; evaluate requires Cityscapes and IDD20K",
    )
    return parser


def _data_roots(values: list[str]) -> tuple[tuple[str, Path], ...]:
    roots: list[tuple[str, Path]] = []
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or name not in {"cityscapes", "idd20k"}:
            raise ValueError("--data-root must be cityscapes=PATH or idd20k=PATH")
        roots.append((name, Path(raw_path).resolve()))
    return tuple(roots)


def main() -> int:
    args = _parser().parse_args()
    work_root = args.work_root.resolve()
    candidate_table = (
        args.candidate_table.resolve()
        if args.candidate_table
        else work_root / "reports/screening/candidate_table.json"
    )
    accepted_release = work_root / "accepted_release.json"
    evaluation_manifests = tuple(path.resolve() for path in args.evaluation_manifest) or (
        work_root / "manifests/official-validation/cityscapes.frozen.json",
        work_root / "manifests/official-validation/idd20k.frozen.json",
    )
    pipeline = ColabPipeline(
        PipelineInputs(
            project_root=args.project_root.resolve(),
            project_commit=args.project_commit,
            runtime_receipt=args.runtime_receipt.resolve(),
            mmseg_root=args.mmseg_root.resolve(),
            work_root=work_root,
            recovery_root=args.recovery_root.resolve(),
            config_path=args.config.resolve(),
            data_manifests=tuple(path.resolve() for path in args.data_manifest),
            candidate_table=candidate_table,
            final_models=ALL_MODELS,
            rare_classes_file=(
                args.rare_classes_file.resolve() if args.rare_classes_file else None
            ),
            class_weights_file=(
                args.class_weights_file.resolve() if args.class_weights_file else None
            ),
            accepted_release=accepted_release,
            evaluation_manifests=evaluation_manifests,
            authorization_policy=args.authorization_policy.resolve(),
            release_policy=(
                args.release_policy.resolve()
                if args.release_policy
                else args.authorization_policy.resolve()
            ),
            data_roots=_data_roots(args.data_root),
            campaign_id=args.campaign_id,
            execution_mode=args.execution_mode,
            state_store_root=(args.state_store_root.resolve() if args.state_store_root else None),
        )
    )
    result = pipeline.plan(args.target) if args.mode == "plan" else pipeline.run(args.target)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
