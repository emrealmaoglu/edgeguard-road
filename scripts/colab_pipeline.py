"""Plan or run the EdgeGuard semantic-cs-idd-v2 Colab pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeguard.rescue.colab_pipeline import TARGETS, ColabPipeline, PipelineInputs
from edgeguard.serialization import canonical_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run"))
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--mmseg-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, action="append", required=True)
    parser.add_argument("--candidate-table", type=Path)
    parser.add_argument("--final-model", action="append", default=[])
    parser.add_argument("--ablation-model")
    parser.add_argument("--rare-classes-file", type=Path)
    parser.add_argument("--class-weights-file", type=Path)
    parser.add_argument(
        "--accepted-release",
        type=Path,
        help="human-accepted, hash-bound final-model release required by evaluate/export/report",
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        action="append",
        default=[],
        help="frozen official_source_val manifest; evaluate requires Cityscapes and IDD20K",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    pipeline = ColabPipeline(
        PipelineInputs(
            project_root=args.project_root.resolve(),
            project_commit=args.project_commit,
            runtime_receipt=args.runtime_receipt.resolve(),
            mmseg_root=args.mmseg_root.resolve(),
            work_root=args.work_root.resolve(),
            recovery_root=args.recovery_root.resolve(),
            config_path=args.config.resolve(),
            data_manifests=tuple(path.resolve() for path in args.data_manifest),
            candidate_table=(args.candidate_table.resolve() if args.candidate_table else None),
            final_models=tuple(args.final_model),
            ablation_model=args.ablation_model,
            rare_classes_file=(
                args.rare_classes_file.resolve() if args.rare_classes_file else None
            ),
            class_weights_file=(
                args.class_weights_file.resolve() if args.class_weights_file else None
            ),
            accepted_release=(args.accepted_release.resolve() if args.accepted_release else None),
            evaluation_manifests=tuple(path.resolve() for path in args.evaluation_manifest),
        )
    )
    result = pipeline.plan(args.target) if args.mode == "plan" else pipeline.run(args.target)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
