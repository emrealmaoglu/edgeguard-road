"""Republish a real Drive-persisted checkpoint's recovery pointer under a new
project_commit, when nothing about the actual training configuration changed.

This project's Drive-backed recovery store binds every published checkpoint to an
immutable per-run "identity" (protocol hash, dataset manifest hashes, optimizer config,
step counts, and the exact git commit that produced it -- see
`edgeguard.rescue.mmseg_runtime.compute_run_identity`). A commit that only changes
orchestration code unrelated to any individual model's training recipe (e.g. which
models a phase loops over) still changes that commit's identity, and therefore makes
every previously-published checkpoint look "stale" to the resume check -- even though
nothing about how that model was actually trained changed.

This tool verifies that is really the case (recomputes the identity under BOTH commits
and checks that project_commit is the ONLY field that differs, and that the recomputed
old-commit identity matches the real one recorded at publish time) before republishing
the SAME checkpoint bytes under the new commit's identity. It never invents, re-trains,
or alters a single byte of the checkpoint -- it only re-labels an already-real artifact
once the training-equivalence claim is verified, and refuses outright if it is not.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from edgeguard.rescue.colab_recovery import (
    peek_recovery_metadata,
    publish_recovery_file,
    restore_recovery_file,
    temporary_directory,
)
from edgeguard.rescue.config import RescueConfig, load_rescue_config
from edgeguard.rescue.mmseg_runtime import compute_run_identity, default_mmseg_root
from edgeguard.rescue.multidomain import validate_dataset_manifest
from edgeguard.serialization import canonical_json, sha256_payload

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--old-project-commit", required=True)
    parser.add_argument("--new-project-commit", required=True)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs/rescue/semantic_first.yaml"
    )
    parser.add_argument("--mmseg-root", type=Path)
    parser.add_argument("--data-manifest", type=Path, action="append", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--stage", default="screening")
    parser.add_argument("--loss", default="ce")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--precision", default="bf16", choices=("fp32", "fp16", "bf16"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="without this, only verify and report -- publishes nothing",
    )
    return parser


def _datasets_for(manifests: tuple[Path, ...]) -> list[str]:
    return [str(validate_dataset_manifest(manifest)["dataset_id"]) for manifest in manifests]


def migrate_one(
    *,
    recovery_root: Path,
    campaign_id: str,
    old_project_commit: str,
    new_project_commit: str,
    protocol: RescueConfig,
    mmseg_root: Path,
    manifests: tuple[Path, ...],
    datasets: list[str],
    model_name: str,
    stage_name: str,
    loss: str,
    max_steps: int,
    precision: str,
    execute: bool,
) -> dict[str, Any]:
    """Verify then (optionally) migrate one model's recovery pointer across commits."""
    artifact_id = f"{stage_name}-{model_name}-{loss}".replace("_", "-")
    pointer_metadata = peek_recovery_metadata(recovery_root, artifact_id=artifact_id)
    if pointer_metadata is None:
        return {"model": model_name, "artifact_id": artifact_id, "status": "no_existing_pointer"}
    recorded_identity_sha256 = pointer_metadata.get("identity_sha256")

    def _identity(commit: str) -> dict[str, Any]:
        return compute_run_identity(
            protocol,
            model_name=model_name,
            stage_name=stage_name,
            mmseg_root=mmseg_root,
            loss=loss,
            audit_report=None,
            split_manifest=None,
            manifests=manifests,
            datasets=datasets,
            learning_rate=None,
            weight_decay=None,
            scheduler="poly",
            warmup_ratio=0.03,
            initialization="random",
            pretrained_manifest=None,
            precision=precision,
            max_steps=max_steps,
            scheduler_steps=max_steps,
            intentional_interrupt_optimizer_step=None,
            project_commit=commit,
        )

    old_identity = _identity(old_project_commit)
    old_identity_sha256 = sha256_payload(old_identity)
    if old_identity_sha256 != recorded_identity_sha256:
        return {
            "model": model_name,
            "artifact_id": artifact_id,
            "status": "verification_failed",
            "recorded_identity_sha256": recorded_identity_sha256,
            "recomputed_old_identity_sha256": old_identity_sha256,
            "reason": (
                "recomputed old-commit identity does not match the real recorded one -- "
                "something besides project_commit differs; refusing to migrate"
            ),
        }
    new_identity = _identity(new_project_commit)
    new_identity_sha256 = sha256_payload(new_identity)
    diff_keys = sorted(key for key in old_identity if old_identity[key] != new_identity[key])
    if diff_keys != ["project_commit"]:
        return {
            "model": model_name,
            "artifact_id": artifact_id,
            "status": "unexpected_diff",
            "diff_keys": diff_keys,
            "reason": "more than project_commit differs between commits; refusing to migrate",
        }
    result: dict[str, Any] = {
        "model": model_name,
        "artifact_id": artifact_id,
        "status": "verified_dry_run",
        "old_identity_sha256": old_identity_sha256,
        "new_identity_sha256": new_identity_sha256,
    }
    if not execute:
        return result
    tmp = temporary_directory(recovery_root.parent)
    try:
        destination = tmp / f"{artifact_id}.pth"
        restore_recovery_file(recovery_root, artifact_id=artifact_id, destination=destination)
        receipt = publish_recovery_file(
            destination,
            recovery_root,
            artifact_id=artifact_id,
            campaign_id=campaign_id,
            project_commit=new_project_commit,
            metadata={
                "identity_sha256": new_identity_sha256,
                "migrated_from_project_commit": old_project_commit,
                "migrated_from_identity_sha256": old_identity_sha256,
            },
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    result["status"] = "migrated"
    result["receipt_generation"] = receipt["generation"]
    return result


def main() -> int:
    args = _parser().parse_args()
    protocol = load_rescue_config(args.config.resolve())
    manifests = tuple(path.resolve() for path in args.data_manifest)
    datasets = _datasets_for(manifests)
    mmseg_root = default_mmseg_root(args.mmseg_root)
    results = [
        migrate_one(
            recovery_root=args.recovery_root.resolve(),
            campaign_id=args.campaign_id,
            old_project_commit=args.old_project_commit,
            new_project_commit=args.new_project_commit,
            protocol=protocol,
            mmseg_root=mmseg_root,
            manifests=manifests,
            datasets=datasets,
            model_name=model,
            stage_name=args.stage,
            loss=args.loss,
            max_steps=args.max_steps,
            precision=args.precision,
            execute=args.execute,
        )
        for model in args.model
    ]
    print(
        canonical_json(
            {
                "schema_version": "1.0",
                "record_type": "edgeguard_recovery_identity_migration",
                "old_project_commit": args.old_project_commit,
                "new_project_commit": args.new_project_commit,
                "executed": args.execute,
                "results": results,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
