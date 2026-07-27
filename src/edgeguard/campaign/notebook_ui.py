"""Compact notebook status views and self-contained failure review bundles."""

from __future__ import annotations

import shutil
import traceback
import zipfile
from pathlib import Path
from typing import Any

from edgeguard.serialization import canonical_json, sha256_file


def campaign_overview(
    *,
    exact_commit: str,
    campaign_id: str,
    current_stage: str,
    completed_stages: list[str],
    blocked_stages: list[str],
    environment: dict[str, Any],
    dataset_readiness: dict[str, Any],
    execution_plan: list[str],
) -> dict[str, Any]:
    """Return a compact, path-free notebook overview payload."""
    if len(exact_commit) != 40 or not campaign_id or not current_stage:
        raise ValueError("notebook overview requires exact campaign provenance")
    return {
        "exact_commit": exact_commit,
        "campaign_id": campaign_id,
        "current_stage": current_stage,
        "completed_stages": completed_stages,
        "blocked_stages": blocked_stages,
        "environment": environment,
        "dataset_readiness": dataset_readiness,
        "execution_plan": execution_plan,
    }


def training_status_row(
    *,
    model: str,
    epoch: int,
    optimizer_step: int,
    total_steps: int,
    loss: float,
    learning_rate: float,
    images_per_second: float,
    data_seconds: float,
    step_seconds: float,
    eta_seconds: float,
    allocated_bytes: int,
    reserved_bytes: int,
    last_checkpoint: str | None,
    last_recovery_sync: str | None,
) -> dict[str, Any]:
    """Validate one periodic training row suitable for notebook display or JSONL."""
    if not 0 <= optimizer_step <= total_steps or total_steps < 1:
        raise ValueError("training progress counts are invalid")
    return {
        "model": model,
        "epoch": epoch,
        "optimizer_step": optimizer_step,
        "completed_total": f"{optimizer_step}/{total_steps}",
        "percent": 100 * optimizer_step / total_steps,
        "loss": loss,
        "learning_rate": learning_rate,
        "images_per_second": images_per_second,
        "data_seconds": data_seconds,
        "step_seconds": step_seconds,
        "eta_seconds": eta_seconds,
        "gpu_allocated_bytes": allocated_bytes,
        "gpu_reserved_bytes": reserved_bytes,
        "last_checkpoint": last_checkpoint,
        "last_recovery_sync": last_recovery_sync,
    }


def stage_summary(
    *,
    result: str,
    artifacts: list[dict[str, Any]],
    metrics: dict[str, Any],
    warnings: list[str],
    failed_items: list[dict[str, Any]],
    next_eligible_stage: str | None,
    review_pack: str | None,
) -> dict[str, Any]:
    """Build the compact post-stage notebook view."""
    return {
        "result": result,
        "artifacts": artifacts,
        "metrics": metrics,
        "warnings": warnings,
        "failed_items": failed_items,
        "next_eligible_stage": next_eligible_stage,
        "review_pack": review_pack,
    }


def create_failure_bundle(
    output_root: Path,
    *,
    campaign_id: str,
    stage_id: str,
    error: BaseException,
    environment: dict[str, Any],
    campaign_state: dict[str, Any],
    artifact_identities: list[dict[str, Any]],
    recovery_status: dict[str, Any],
    log_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Create a sanitized diagnostic ZIP rather than exposing only a subprocess error."""
    safe_name = f"edgeguard-failure-{campaign_id}-{stage_id}"
    work = output_root / safe_name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    payload = {
        "error_type": type(error).__name__,
        "error": str(error)[:4000],
        "traceback": "".join(traceback.format_exception(error))[-12000:],
        "environment": environment,
        "campaign_state": campaign_state,
        "artifact_identities": artifact_identities,
        "recovery_status": recovery_status,
    }
    (work / "failure.json").write_text(canonical_json(payload) + "\n", encoding="utf-8")
    tails: dict[str, str] = {}
    for path in log_paths:
        if path.is_file():
            tails[path.name] = path.read_text(encoding="utf-8", errors="replace")[-12000:]
    (work / "log_tails.json").write_text(canonical_json(tails) + "\n", encoding="utf-8")
    archive = output_root / f"{safe_name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(work.iterdir()):
            bundle.write(path, path.name)
    shutil.rmtree(work)
    return {
        "path": archive.name,
        "sha256": sha256_file(archive),
        "byte_size": archive.stat().st_size,
    }
