from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgeguard.rescue.colab_recovery import (
    action_requirements,
    completion_is_valid,
    latest_checkpoint,
    package_interrupted_campaign,
    publish_recovery_file,
    quarantine_incomplete,
    restore_recovery_file,
    write_campaign_status,
    write_completion_receipt,
)

COMMIT = "a" * 40


def test_content_addressed_recovery_current_and_previous_fallback(tmp_path: Path) -> None:
    store = tmp_path / "drive"
    source = tmp_path / "checkpoint.pth"
    source.write_bytes(b"first")
    publish_recovery_file(
        source,
        store,
        artifact_id="screening-model",
        campaign_id="semantic-cs-idd-v1",
        project_commit=COMMIT,
        metadata={"optimizer_step": 500},
    )
    source.write_bytes(b"second")
    second = publish_recovery_file(
        source,
        store,
        artifact_id="screening-model",
        campaign_id="semantic-cs-idd-v1",
        project_commit=COMMIT,
        metadata={"optimizer_step": 1000},
    )
    current_object = store / second["object"]
    current_object.write_bytes(b"corrupt")
    destination = tmp_path / "fresh-content" / "checkpoint.pth"
    restored = restore_recovery_file(store, artifact_id="screening-model", destination=destination)
    assert destination.read_bytes() == b"first"
    assert restored["metadata"]["optimizer_step"] == 500


def test_completion_receipt_rejects_partial_and_changed_output(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "evaluation.json").write_text('{"mIoU": 0.5}\n', encoding="utf-8")
    assert completion_is_valid(output) is False
    write_completion_receipt(
        output,
        artifact_type="evaluation",
        required_paths=["evaluation.json"],
        inputs={"checkpoint": "b" * 64},
    )
    assert completion_is_valid(output) is True
    (output / "evaluation.json").write_text('{"mIoU": 0.6}\n', encoding="utf-8")
    assert completion_is_valid(output) is False
    quarantined = quarantine_incomplete(output)
    assert quarantined is not None and quarantined.is_dir()
    assert not output.exists()


def test_completion_receipt_rejects_changed_immutable_inputs(tmp_path: Path) -> None:
    output = tmp_path / "evaluation"
    output.mkdir()
    (output / "metrics.json").write_text("{}", encoding="utf-8")
    write_completion_receipt(
        output,
        artifact_type="evaluation",
        required_paths=["metrics.json"],
        inputs={"checkpoint_sha256": "a" * 64},
    )
    assert completion_is_valid(output, expected_inputs={"checkpoint_sha256": "a" * 64})
    assert not completion_is_valid(output, expected_inputs={"checkpoint_sha256": "b" * 64})


def test_latest_checkpoint_uses_marker_or_numeric_iteration(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    for name in ("iter_500.pth", "iter_1000.pth", "iter_9000.pth"):
        (run / name).write_bytes(name.encode())
    assert latest_checkpoint(run).name == "iter_9000.pth"
    (run / "last_checkpoint").write_text("iter_1000.pth\n", encoding="utf-8")
    assert latest_checkpoint(run).name == "iter_1000.pth"


def test_action_planner_is_run_all_and_dataset_conditional() -> None:
    report = action_requirements("report")
    assert report["datasets"] == ["cityscapes", "idd20k"]
    assert report["runtime_required"] is True
    hpo = action_requirements("hpo")
    assert hpo["datasets"] == ["cityscapes", "idd20k"]
    assert hpo["training_stages"] == ["smoke", "pilot", "screening"]
    assert hpo["runtime_required"] is True
    final = action_requirements("evaluate", allow_final_data=True)
    assert "cityscapes_official_val" in final["datasets"]
    with pytest.raises(ValueError, match="target must be"):
        action_requirements("external")


def test_recovery_pointer_is_committed_after_receipt(tmp_path: Path) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"payload")
    store = tmp_path / "store"
    receipt = publish_recovery_file(
        source,
        store,
        artifact_id="artifact",
        campaign_id="campaign",
        project_commit=COMMIT,
    )
    pointer = json.loads((store / "pointers/artifact.json").read_text(encoding="utf-8"))
    assert (store / pointer["current"]["receipt"]).is_file()
    assert receipt["sha256"] in str(store / receipt["object"])


def test_running_status_becomes_one_interruption_failure_package(tmp_path: Path) -> None:
    status = tmp_path / "campaign/state/status.json"
    write_campaign_status(status, state="running", stage="screening", optimizer_step=731)
    packaged = package_interrupted_campaign(status, tmp_path / "campaign/failures")
    assert packaged is not None
    assert Path(packaged["package"]).is_file()
    assert package_interrupted_campaign(status, tmp_path / "campaign/failures") is None
