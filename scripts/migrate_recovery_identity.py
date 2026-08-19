"""Manually re-publish real Drive checkpoints under a new project_commit.

Thin CLI wrapper over `edgeguard.rescue.mmseg_runtime.migrate_recovery_identity`, which
also runs automatically inside `scripts/run_colab_master.py` before every production
run -- this script exists for manual/one-off use (e.g. verifying a past migration, or
migrating a model the automatic pass skipped). See that function's docstring for the
full verify-before-republish contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeguard.rescue.colab_pipeline import ALL_MODELS
from edgeguard.rescue.config import load_rescue_config
from edgeguard.rescue.mmseg_runtime import default_mmseg_root, migrate_recovery_identity
from edgeguard.rescue.multidomain import validate_dataset_manifest
from edgeguard.serialization import canonical_json

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--new-project-commit", required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs/rescue/semantic_first.yaml"
    )
    parser.add_argument("--mmseg-root", type=Path)
    parser.add_argument("--data-manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--model",
        action="append",
        choices=ALL_MODELS,
        help="repeatable; defaults to all five known models when omitted",
    )
    parser.add_argument("--stage", default="screening")
    parser.add_argument("--loss", default="ce")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="defaults to the frozen protocol's stage.max_steps when omitted",
    )
    parser.add_argument("--precision", default="bf16", choices=("fp32", "fp16", "bf16"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="without this, only verify and report -- publishes nothing",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    protocol = load_rescue_config(args.config.resolve())
    manifests = tuple(path.resolve() for path in args.data_manifest)
    datasets = [str(validate_dataset_manifest(manifest)["dataset_id"]) for manifest in manifests]
    mmseg_root = default_mmseg_root(args.mmseg_root)
    max_steps = args.max_steps
    if max_steps is None:
        stage_config = protocol.stages[args.stage]
        if stage_config.max_steps is None:
            raise ValueError(f"stage {args.stage!r} has no fixed max_steps; pass --max-steps")
        max_steps = stage_config.max_steps
    results = [
        migrate_recovery_identity(
            recovery_root=args.recovery_root.resolve(),
            campaign_id=args.campaign_id,
            new_project_commit=args.new_project_commit,
            protocol=protocol,
            mmseg_root=mmseg_root,
            manifests=manifests,
            datasets=datasets,
            model_name=model,
            stage_name=args.stage,
            loss=args.loss,
            max_steps=max_steps,
            precision=args.precision,
            execute=args.execute,
        )
        for model in (args.model or list(ALL_MODELS))
    ]
    print(
        canonical_json(
            {
                "schema_version": "1.0",
                "record_type": "edgeguard_recovery_identity_migration",
                "new_project_commit": args.new_project_commit,
                "executed": args.execute,
                "results": results,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
