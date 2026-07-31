"""Integration coverage for the project-owned local readiness gate."""

import importlib.util
import json
from pathlib import Path

import pytest

from scripts.dev.validate_local_readiness import validate_local_readiness


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional local Torch")
def test_local_readiness_runs_without_content_drive_or_real_data(tmp_path: Path) -> None:
    workspace = tmp_path / "local-readiness"

    result = validate_local_readiness(workspace, profile="mac-cpu")

    assert result["status"] == "passed_with_platform_block"
    assert result["fixture_probe"]["checkpoint_resume"]["checkpoint_round_trip"] is True
    assert result["fixture_probe"]["loader_loss_metrics"]["per_class_iou_count"] == 19
    assert result["workspace_bytes_before_evidence_package"] < 5 * 1024**2
    assert result["framework_probe"]["requires_linux_cpu_gate"] is True
    assert len(result["phases"]) == 11
    path_a_plan = result["phases"]["installer-plan-generation"]["result"]["path_a"]
    assert any("python3.12" in value for command in path_a_plan for value in command)
    assert any("mmcv-lite==2.1.0" in command for command in path_a_plan)
    assert (workspace / "evidence/local-readiness-evidence.zip").is_file()
    persisted = json.loads((workspace / "evidence/readiness_receipt.json").read_text())
    assert str(tmp_path) not in json.dumps(persisted)
    assert "/content" not in json.dumps(persisted)
