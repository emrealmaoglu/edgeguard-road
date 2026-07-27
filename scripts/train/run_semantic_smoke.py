"""Run the bounded five-model EG-SEG-002 Cityscapes training-path smoke."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from edgeguard.serialization import canonical_json, sha256_file
from edgeguard.telemetry.longrun import LiveCommandRunner, LongRunStatus, atomic_write_json

MODEL_CONFIGS = (
    "fast_scnn.yaml",
    "bisenetv2.yaml",
    "pidnet_s.yaml",
    "ddrnet_23_slim.yaml",
    "segformer_b0.yaml",
)


def build_train_command(
    interpreter: Path,
    project_root: Path,
    *,
    config_root: Path,
    model_config: Path,
    mmseg_checkout: Path,
    project_commit: str,
    dataset_root: Path,
    dataset_manifest: Path,
    split_policy_manifest: Path,
    output_dir: Path,
    recovery_dir: Path,
    device_batch: int,
    gradient_accumulation: int,
    resume: bool = False,
) -> tuple[str, ...]:
    """Build one explicit random-initialized, identity-protected smoke command."""
    command = [
        str(interpreter),
        str(project_root / "scripts/train/train_semantic.py"),
        "train",
        "--config-root",
        str(config_root),
        "--model-config",
        str(model_config),
        "--mmseg-checkout",
        str(mmseg_checkout),
        "--project-root",
        str(project_root),
        "--project-commit",
        project_commit,
        "--dataset-root",
        str(dataset_root),
        "--dataset-manifest",
        str(dataset_manifest),
        "--split-policy-manifest",
        str(split_policy_manifest),
        "--output-dir",
        str(output_dir),
        "--precision",
        "fp16",
        "--recovery-sync-dir",
        str(recovery_dir),
        "--max-optimizer-steps",
        "100",
        "--validation-subset-size",
        "25",
        "--device-batch",
        str(device_batch),
        "--gradient-accumulation",
        str(gradient_accumulation),
        "--train-fit-fraction",
        "1.0",
        "--smoke-random-initialization",
    ]
    if resume:
        command.append("--resume")
    return tuple(command)


def _has_oom(log_directory: Path) -> bool:
    return any(
        "out of memory" in path.read_text(encoding="utf-8", errors="replace").lower()
        for path in log_directory.rglob("*.stderr.log")
    )


def _sync_small_evidence(source: Path, destination: Path) -> dict[str, Any]:
    incoming = destination.with_name(f".{destination.name}.incoming")
    if destination.exists() or incoming.exists():
        raise ValueError("smoke evidence destination already exists")
    incoming.mkdir(parents=True)
    files: list[dict[str, Any]] = []
    for item in sorted(source.rglob("*")):
        if not item.is_file() or item.suffix == ".pth" or item.stat().st_size > 50 * 1024**2:
            continue
        relative = item.relative_to(source)
        target = incoming / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        files.append(
            {
                "relative_path": relative.as_posix(),
                "byte_size": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    atomic_write_json(
        incoming / "sync_manifest.json",
        {
            "schema_version": "1.0",
            "record_type": "semantic_smoke_evidence_sync",
            "files": files,
        },
    )
    os.replace(incoming, destination)
    return {"file_count": len(files), "destination": destination.name}


def run_five_model_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Execute five bounded jobs, adapt once on OOM, and verify exact resume."""
    args.run_root.mkdir(parents=True, exist_ok=True)
    status = LongRunStatus(args.run_root / "run_status.json")
    log_root = args.run_root / "logs"
    results: list[dict[str, Any]] = []
    for index, filename in enumerate(MODEL_CONFIGS, start=1):
        model_name = Path(filename).stem
        model_result: dict[str, Any] = {"model": model_name, "status": "failed_smoke"}
        for attempt, (batch, accumulation) in enumerate(((2, 2), (1, 4)), start=1):
            output = args.run_root / model_name / f"attempt-{attempt}"
            recovery = (
                args.drive_root
                / "checkpoints/segmentation"
                / model_name
                / "recovery"
                / f"attempt-{attempt}"
            )
            command = build_train_command(
                args.interpreter,
                args.project_root,
                config_root=args.config_root,
                model_config=args.config_root / filename,
                mmseg_checkout=args.mmseg_checkout,
                project_commit=args.project_commit,
                dataset_root=args.dataset_root,
                dataset_manifest=args.dataset_manifest,
                split_policy_manifest=args.split_policy_manifest,
                output_dir=output,
                recovery_dir=recovery,
                device_batch=batch,
                gradient_accumulation=accumulation,
            )
            runner = LiveCommandRunner(log_root / model_name / f"attempt-{attempt}", status)
            try:
                runner.run(
                    f"train-{model_name}-batch-{batch}",
                    command,
                    stage_index=index,
                    stage_total=len(MODEL_CONFIGS),
                    cwd=args.project_root,
                )
                runner.run(
                    f"resume-{model_name}-batch-{batch}",
                    (*command, "--resume"),
                    stage_index=index,
                    stage_total=len(MODEL_CONFIGS),
                    cwd=args.project_root,
                )
                model_result = {
                    "model": model_name,
                    "status": "passed_smoke",
                    "device_batch": batch,
                    "gradient_accumulation": accumulation,
                    "optimizer_steps": 100,
                    "resume_verified": True,
                }
                break
            except subprocess.CalledProcessError as error:
                if attempt == 1 and _has_oom(log_root / model_name / f"attempt-{attempt}"):
                    model_result = {"model": model_name, "status": "retrying_after_oom"}
                    continue
                model_result = {
                    "model": model_name,
                    "status": "oom_after_adaptation" if _has_oom(log_root) else "failed_smoke",
                    "error": str(error)[:500],
                }
                break
        results.append(model_result)
    passed = sum(result["status"] == "passed_smoke" for result in results)
    summary = {
        "schema_version": "1.0",
        "record_type": "semantic_five_model_smoke_summary",
        "project_commit": args.project_commit,
        "models": results,
        "passed_model_count": passed,
        "status": "ready_for_common_screening" if passed >= 4 else "failed_smoke",
        "starts_screening": False,
    }
    atomic_write_json(args.run_root / "smoke_summary.json", summary)
    evidence_destination = args.drive_root / "experiments/segmentation/EG-SEG-002"
    summary["evidence_sync"] = _sync_small_evidence(args.run_root, evidence_destination)
    atomic_write_json(args.run_root / "smoke_summary.json", summary)
    if passed < 4:
        status.fail("fewer than four semantic models passed the bounded smoke")
        raise RuntimeError("five-model smoke did not meet the four-model promotion gate")
    status.complete(last_checkpoint="smoke_summary.json")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpreter", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--mmseg-checkout", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--split-policy-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("/content/edgeguard-runs/EG-SEG-002"))
    parser.add_argument("--drive-root", type=Path, required=True)
    return parser


def main() -> int:
    result = run_five_model_smoke(_parser().parse_args())
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
