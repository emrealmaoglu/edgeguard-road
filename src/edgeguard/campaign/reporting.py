"""Small assistant review packs and deterministic thesis figure artifacts."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from PIL import Image, ImageDraw

from edgeguard.campaign.contracts import topological_stages
from edgeguard.campaign.figures import generate_thesis_figures
from edgeguard.campaign.runner import load_stage_receipts
from edgeguard.campaign.state import Campaign
from edgeguard.serialization import canonical_json, sha256_file
from edgeguard.telemetry.longrun import atomic_write_json

MAX_REVIEW_BYTES = 100 * 1024**2


def _zip_directory(source: Path, destination: Path) -> None:
    """Create a byte-stable ZIP with fixed timestamps and sorted members."""
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            info = ZipInfo(path.relative_to(source).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _metric_rows(receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for receipt in receipts:
        stage = receipt["stage_id"]
        if stage == "semantic_smoke":
            for model in receipt.get("models", []):
                for step, loss in enumerate(model["train_losses"], start=1):
                    rows.append(
                        {
                            "stage_id": stage,
                            "subject": model["model_family"],
                            "metric": "train_loss",
                            "step": step,
                            "value": loss,
                            "scientific_evidence": False,
                        }
                    )
                mean_iou = model["validation"].get("mean_iou")
                if mean_iou is not None:
                    rows.append(
                        {
                            "stage_id": stage,
                            "subject": model["model_family"],
                            "metric": "synthetic_validation_mean_iou",
                            "step": len(model["train_losses"]),
                            "value": mean_iou,
                            "scientific_evidence": False,
                        }
                    )
        elif stage == "zero_shot_ood":
            for method, values in receipt.get("summaries", {}).items():
                for metric in ("auroc", "average_precision", "fpr_at_95_tpr"):
                    value = values["metrics"].get(metric)
                    if value is not None:
                        rows.append(
                            {
                                "stage_id": stage,
                                "subject": method,
                                "metric": metric,
                                "step": 0,
                                "value": value,
                                "scientific_evidence": False,
                            }
                        )
        elif stage == "temperature_calibration":
            for phase in ("before", "after"):
                for metric in ("nll", "ece", "brier_score"):
                    rows.append(
                        {
                            "stage_id": stage,
                            "subject": phase,
                            "metric": metric,
                            "step": 0,
                            "value": receipt[phase][metric],
                            "scientific_evidence": False,
                        }
                    )
    return rows


def _write_metrics(directory: Path, rows: list[dict[str, Any]]) -> None:
    jsonl = "".join(canonical_json(row) + "\n" for row in rows)
    _write_text(directory / "metrics.jsonl", jsonl)
    buffer = io.StringIO()
    fields = ("stage_id", "subject", "metric", "step", "value", "scientific_evidence")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_text(directory / "metrics.csv", buffer.getvalue())


def _representative_image(campaign: Campaign, destination: Path) -> None:
    image = Image.new("RGB", (640, 180), color=(20, 28, 38))
    draw = ImageDraw.Draw(image)
    draw.text((24, 24), "EDGEGUARD-ROAD LOCAL-MINI", fill=(245, 190, 60))
    draw.text((24, 64), "NON-SCIENTIFIC PIPELINE VALIDATION", fill=(255, 255, 255))
    draw.text((24, 104), campaign.load_manifest()["campaign_id"], fill=(145, 200, 255))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)


def _decision_markdown(campaign: Campaign) -> str:
    lines = ["# Campaign decision log", ""]
    for line in campaign.decision_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        lines.append(f"- **{row['stage_id']} — {row['decision']}**: {row['rationale']}")
    if len(lines) == 2:
        lines.append("- No non-blocking decisions were recorded.")
    return "\n".join(lines) + "\n"


def _review_readme(
    campaign: Campaign, state: dict[str, Any], receipts: list[dict[str, Any]]
) -> str:
    completed = [
        stage for stage in topological_stages() if state["stages"][stage]["status"] == "completed"
    ]
    failed = [
        stage for stage in topological_stages() if state["stages"][stage]["status"] == "failed"
    ]
    pending = [
        stage for stage in topological_stages() if state["stages"][stage]["status"] != "completed"
    ]
    next_stage = pending[0] if pending else "human review of the local-mini evidence"
    return (
        "# EdgeGuard-Road assistant review pack\n\n"
        f"Campaign: `{campaign.load_manifest()['campaign_id']}`\n\n"
        "Classification: **NON-SCIENTIFIC PIPELINE VALIDATION**. No synthetic metric "
        "is a thesis performance result.\n\n"
        f"Executed stages: {', '.join(completed) or 'none'}.\n\n"
        f"Failed stages: {', '.join(failed) or 'none'}.\n\n"
        f"Next stage: {next_stage}.\n\n"
        "Inspect `pipeline_state.json`, `failures.json`, `metrics.csv`, "
        "`figure_index.json`, and the stage receipts. A human decision is required only "
        "where `executive_summary.json` lists one.\n"
    )


def _build_review_pack(campaign: Campaign) -> dict[str, Any]:
    manifest = campaign.load_manifest()
    state = campaign.load_state()
    receipts = load_stage_receipts(campaign)
    completed = [
        stage for stage in topological_stages() if state["stages"][stage]["status"] == "completed"
    ]
    stage_label = completed[-1] if completed else "initialized"
    build = campaign.root / "reports" / ".assistant-build"
    if build.exists():
        shutil = __import__("shutil")
        shutil.rmtree(build)
    for name in ("stage_receipts", "selected_log_tails", "representative_images"):
        (build / name).mkdir(parents=True, exist_ok=True)
    _write_text(build / "README_REVIEW.md", _review_readme(campaign, state, receipts))
    atomic_write_json(build / "campaign_manifest.json", manifest)
    atomic_write_json(build / "pipeline_state.json", state)
    atomic_write_json(
        build / "artifact_index.json",
        json.loads(campaign.artifact_index_path.read_text(encoding="utf-8")),
    )
    _write_text(build / "decision_log.md", _decision_markdown(campaign))
    failures = [
        {
            "stage_id": stage,
            "classification": record["failure_classification"],
            "last_error": record["last_error"],
        }
        for stage, record in state["stages"].items()
        if record["status"] == "failed"
    ]
    atomic_write_json(build / "failures.json", {"failures": failures})
    preflight = next((receipt for receipt in receipts if receipt["stage_id"] == "preflight"), {})
    atomic_write_json(
        build / "environment.json",
        {
            "python_version": preflight.get("python_version"),
            "machine": preflight.get("machine"),
            "environment_identity": state["stages"]["preflight"]["environment_identity"],
        },
    )
    for receipt in receipts:
        atomic_write_json(build / "stage_receipts" / f"{receipt['stage_id']}.json", receipt)
    rows = _metric_rows(receipts)
    _write_metrics(build, rows)
    _representative_image(campaign, build / "representative_images" / "local-mini-banner.png")
    figure_result = generate_thesis_figures(campaign, receipts=receipts)
    atomic_write_json(build / "figure_index.json", figure_result["figure_index"])
    summary = {
        "schema_version": "1.0",
        "record_type": "edgeguard_assistant_review_summary",
        "campaign_id": manifest["campaign_id"],
        "completed_stage_count": len(completed),
        "failed_stage_count": len(failures),
        "synthetic_pipeline_validation": manifest["profile"] in {"local-mini", "linux-cpu"},
        "scientific_results": False,
        "decision_requested": None if not failures else "inspect recorded stage failure",
        "exact_next_stage": (
            "human review of completed local-mini evidence"
            if len(completed) == len(state["stages"])
            else next(
                stage
                for stage in topological_stages()
                if state["stages"][stage]["status"] != "completed"
            )
        ),
    }
    atomic_write_json(build / "executive_summary.json", summary)
    destination = (
        campaign.root / "reports" / f"edgeguard-review-{manifest['campaign_id']}-{stage_label}.zip"
    )
    _zip_directory(build, destination)
    if destination.stat().st_size >= MAX_REVIEW_BYTES:
        raise ValueError("assistant review pack exceeds 100 MB")
    return {
        "audience": "assistant",
        "relative_path": destination.relative_to(campaign.root).as_posix(),
        "sha256": sha256_file(destination),
        "byte_size": destination.stat().st_size,
        "completed_stage_count": len(completed),
    }


def generate_report(campaign: Campaign, *, audience: str) -> dict[str, Any]:
    """Generate a partial-safe assistant pack or deterministic thesis figures."""
    campaign.verify_identity()
    if audience == "assistant":
        return _build_review_pack(campaign)
    receipts = load_stage_receipts(campaign)
    return generate_thesis_figures(campaign, receipts=receipts)
