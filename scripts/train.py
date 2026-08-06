"""Train one semantic-first model with the standard MMSeg Runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeguard.rescue.config import load_rescue_config
from edgeguard.rescue.hpo_runtime import run_hpo_study, select_hpo_models
from edgeguard.rescue.mmseg_runtime import default_mmseg_root, train_model
from edgeguard.serialization import canonical_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs/rescue/semantic_first.yaml"
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--stage", choices=("smoke", "pilot", "screening", "hpo", "final"), required=True
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--data-manifest", type=Path, action="append", default=[])
    parser.add_argument("--candidate-table", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mmseg-root", type=Path)
    parser.add_argument("--loss", choices=("ce", "median_frequency"), default="ce")
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--rare-classes-file", type=Path)
    parser.add_argument("--initialization", choices=("random", "pretrained"), default="random")
    parser.add_argument("--pretrained-manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device-batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--project-commit")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--scheduler", choices=("poly", "cosine"), default="poly")
    parser.add_argument("--warmup-ratio", type=float, choices=(0.01, 0.03, 0.05), default=0.03)
    parser.add_argument("--run-name")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--crop-height", type=int)
    parser.add_argument("--crop-width", type=int)
    parser.add_argument("--intentional-interrupt-step", type=int)
    parser.add_argument("--acceptance-test", action="store_true")
    return parser


def main() -> int:
    """Resolve the shared protocol and execute a bounded training stage."""
    args = _parser().parse_args()
    if (args.crop_height is None) != (args.crop_width is None):
        raise ValueError("crop override requires both --crop-height and --crop-width")
    protocol = load_rescue_config(args.config.resolve())
    manifests = tuple(path.resolve() for path in args.data_manifest)
    if args.stage == "hpo":
        if args.candidate_table is None or args.model is not None:
            raise ValueError("HPO requires --candidate-table and selects its two models itself")
        results = [
            run_hpo_study(
                protocol,
                model=model,
                manifests=manifests,
                mmseg_root=default_mmseg_root(args.mmseg_root),
                output_root=args.output_root.resolve(),
                rare_classes_file=(
                    args.rare_classes_file.resolve() if args.rare_classes_file else None
                ),
                recovery_root=args.recovery_root.resolve() if args.recovery_root else None,
                campaign_id=args.campaign_id,
                project_commit=args.project_commit,
                device_batch=args.device_batch,
                workers=args.workers,
                precision=args.precision,
                acceptance_test=args.acceptance_test,
            )
            for model in select_hpo_models(
                args.candidate_table.resolve(), protocol.datasets.training
            )
        ]
        print(canonical_json({"schema_version": "2.0", "hpo_results": results}))
        return 0
    if args.model is None:
        raise ValueError("non-HPO training requires --model")
    if not manifests and (args.dataset_root is None or args.split_manifest is None):
        raise ValueError(
            "provide repeated --data-manifest values or legacy --dataset-root/--split-manifest"
        )
    result = train_model(
        protocol,
        model_name=args.model,
        stage_name=args.stage,
        mmseg_root=default_mmseg_root(args.mmseg_root),
        dataset_root=args.dataset_root.resolve() if args.dataset_root else None,
        split_manifest=args.split_manifest.resolve() if args.split_manifest else None,
        output_root=args.output_root.resolve(),
        loss=args.loss,
        audit_report=args.audit_report.resolve() if args.audit_report else None,
        resume=args.resume,
        data_manifests=manifests,
        initialization=args.initialization,
        pretrained_manifest=(
            args.pretrained_manifest.resolve() if args.pretrained_manifest else None
        ),
        device_batch=args.device_batch,
        workers=args.workers,
        precision=args.precision,
        recovery_root=args.recovery_root.resolve() if args.recovery_root else None,
        campaign_id=args.campaign_id,
        project_commit=args.project_commit,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        run_name=args.run_name,
        max_steps_override=args.max_steps,
        crop_size_override=(
            (args.crop_height, args.crop_width)
            if args.crop_height is not None and args.crop_width is not None
            else None
        ),
        intentional_interrupt_optimizer_step=args.intentional_interrupt_step,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
