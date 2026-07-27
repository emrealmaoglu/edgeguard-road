"""Campaign planning, execution, reuse, and bounded failure injection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from edgeguard.campaign.contracts import topological_stages
from edgeguard.campaign.stages import StageContext, execute_stage
from edgeguard.campaign.state import Campaign
from edgeguard.serialization import sha256_file
from edgeguard.telemetry.longrun import atomic_write_json


def campaign_plan(campaign: Campaign) -> dict[str, Any]:
    """Return the current deterministic plan with dependency state."""
    campaign.verify_identity()
    state = campaign.load_state()
    campaign.refresh_readiness(state)
    campaign.save_state(state)
    return {
        "campaign_id": state["campaign_id"],
        "stages": [
            {
                "stage_id": stage_id,
                "status": state["stages"][stage_id]["status"],
            }
            for stage_id in topological_stages()
        ],
    }


def run_campaign(
    campaign: Campaign,
    *,
    stop_after: str | None = None,
    interrupt_stage: str | None = None,
    mmseg_checkout: Path | None = None,
) -> dict[str, Any]:
    """Run eligible stages, reusing only hash-verified completed outputs."""
    with campaign.locked():
        manifest = campaign.verify_identity()
        state = campaign.load_state()
        campaign.refresh_readiness(state)
        campaign.save_state(state)
        executed: list[str] = []
        reused: list[str] = []
        for stage_id in topological_stages():
            record = state["stages"][stage_id]
            if record["status"] == "completed":
                campaign.verify_completed(state, stage_id)
                reused.append(stage_id)
                if stage_id == stop_after:
                    break
                continue
            if record["status"] == "failed":
                recovery = record.get("recovery_identity")
                recovery_path = campaign.root / "recovery" / f"{stage_id}.json"
                recovery_valid = (
                    recovery and recovery_path.is_file() and sha256_file(recovery_path) == recovery
                )
                if not recovery_valid:
                    raise ValueError(f"failed stage {stage_id} has no valid recovery state")
                record["status"] = "ready"
            campaign.refresh_readiness(state)
            if record["status"] != "ready":
                continue
            campaign.begin_stage(state, stage_id)
            context = StageContext(
                campaign_root=campaign.root,
                repository=campaign.repository,
                campaign_id=str(manifest["campaign_id"]),
                profile=str(manifest["profile"]),
                stage_id=stage_id,
                attempt=int(record["attempt_number"]),
                mmseg_checkout=mmseg_checkout,
            )
            try:
                receipt = execute_stage(context)
                if interrupt_stage == stage_id:
                    recovery_path = campaign.root / "recovery" / f"{stage_id}.json"
                    atomic_write_json(
                        recovery_path,
                        {
                            "stage_id": stage_id,
                            "receipt_sha256": sha256_file(receipt),
                            "attempt_number": record["attempt_number"],
                        },
                    )
                    identity = sha256_file(recovery_path)
                    campaign.fail_stage(
                        state,
                        stage_id,
                        "process_interrupted",
                        "bounded failure injection after recoverable stage output",
                        recovery_identity=identity,
                    )
                    break
                campaign.complete_stage(state, stage_id, receipt)
                executed.append(stage_id)
            except BaseException as error:
                if state["stages"][stage_id]["status"] == "running":
                    campaign.fail_stage(state, stage_id, type(error).__name__, str(error))
                raise
            if stage_id == stop_after:
                break
        return {
            "campaign_id": manifest["campaign_id"],
            "executed": executed,
            "reused": reused,
            "state": state,
        }


def status_summary(campaign: Campaign) -> dict[str, Any]:
    """Return compact current status, tolerating partial campaigns."""
    state = campaign.load_state()
    return {
        "campaign_id": state["campaign_id"],
        "revision": state["revision"],
        "stages": {
            stage_id: {
                "status": record["status"],
                "attempt_number": record["attempt_number"],
                "failure_classification": record["failure_classification"],
                "last_error": record["last_error"],
            }
            for stage_id, record in state["stages"].items()
        },
    }


def load_stage_receipts(campaign: Campaign) -> list[dict[str, Any]]:
    """Load all promoted stage receipts and fail on malformed JSON."""
    index = json.loads(campaign.artifact_index_path.read_text(encoding="utf-8"))
    receipts: list[dict[str, Any]] = []
    for item in index["artifacts"]:
        path = campaign.root / item["relative_path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError("artifact index contains a hash mismatch")
        receipts.append(json.loads(path.read_text(encoding="utf-8")))
    return receipts
