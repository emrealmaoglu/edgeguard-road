"""Tests for the project-specific campaign source of truth."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from edgeguard.campaign.contracts import STAGE_DEPENDENCIES
from edgeguard.campaign.runner import campaign_plan, run_campaign, status_summary
from edgeguard.campaign.state import Campaign


def _repository(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=path, check=True, capture_output=True)
    return path


def test_campaign_init_contract_and_plan(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")
    campaign = Campaign(tmp_path / "campaign", repository)

    manifest = campaign.initialize(campaign_id="eg-test", profile="local-mini")
    plan = campaign_plan(campaign)

    assert manifest["campaign_id"] == "eg-test"
    assert [item["stage_id"] for item in plan["stages"]] == list(STAGE_DEPENDENCIES)
    assert plan["stages"][0]["status"] == "ready"
    assert {path.name for path in campaign.root.iterdir()} >= {
        "campaign_manifest.json",
        "pipeline_state.json",
        "artifact_index.json",
        "decisions.jsonl",
        "stages",
        "checkpoints",
        "reports",
        "recovery",
    }


def test_completed_stage_reuse_requires_valid_artifact(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")
    campaign = Campaign(tmp_path / "campaign", repository)
    campaign.initialize(campaign_id="eg-reuse", profile="local-mini")
    first = run_campaign(campaign, stop_after="preflight")
    second = run_campaign(campaign, stop_after="preflight")
    assert first["executed"] == ["preflight"]
    assert second["reused"] == ["preflight"]

    receipt = next((campaign.root / "stages" / "preflight").rglob("stage_receipt.json"))
    receipt.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or corrupt"):
        run_campaign(campaign, stop_after="preflight")


def test_interrupted_stage_resumes_with_compatible_recovery(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")
    campaign = Campaign(tmp_path / "campaign", repository)
    campaign.initialize(campaign_id="eg-resume", profile="local-mini")
    interrupted = run_campaign(
        campaign,
        stop_after="storage_inventory",
        interrupt_stage="storage_inventory",
    )
    assert interrupted["state"]["stages"]["storage_inventory"]["status"] == "failed"

    resumed = run_campaign(campaign, stop_after="storage_inventory")
    assert resumed["state"]["stages"]["storage_inventory"]["status"] == "completed"
    assert resumed["state"]["stages"]["storage_inventory"]["attempt_number"] == 2


def test_campaign_rejects_incompatible_commit_and_corrupt_state(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")
    campaign = Campaign(tmp_path / "campaign", repository)
    campaign.initialize(campaign_id="eg-identity", profile="local-mini")
    (repository / "second.txt").write_text("second\n", encoding="utf-8")
    subprocess.run(["git", "add", "second.txt"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-m", "second"], cwd=repository, check=True, capture_output=True
    )
    with pytest.raises(ValueError, match="Git commit"):
        campaign_plan(campaign)

    campaign.state_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        status_summary(campaign)


def test_decisions_are_append_only_jsonl(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repository")
    campaign = Campaign(tmp_path / "campaign", repository)
    campaign.initialize(campaign_id="eg-decisions", profile="local-mini")
    campaign.record_decision("preflight", "use local-mini", "bounded synthetic validation")
    campaign.record_decision("export_probe", "record failure", "onnx is optional")

    rows = [json.loads(line) for line in campaign.decision_path.read_text().splitlines()]
    assert [row["stage_id"] for row in rows] == ["preflight", "export_probe"]
