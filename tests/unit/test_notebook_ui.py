import zipfile
from pathlib import Path

from edgeguard.campaign.notebook_ui import (
    campaign_overview,
    create_failure_bundle,
    stage_summary,
    training_status_row,
)


def test_notebook_compact_views() -> None:
    overview = campaign_overview(
        exact_commit="a" * 40,
        campaign_id="eg-test",
        current_stage="semantic_smoke",
        completed_stages=["preflight"],
        blocked_stages=[],
        environment={"gpu": "unavailable"},
        dataset_readiness={"ready": False},
        execution_plan=["preflight", "semantic_smoke"],
    )
    assert overview["exact_commit"] == "a" * 40
    row = training_status_row(
        model="fast_scnn",
        epoch=0,
        optimizer_step=2,
        total_steps=4,
        loss=1.0,
        learning_rate=0.001,
        images_per_second=2.0,
        data_seconds=0.1,
        step_seconds=0.5,
        eta_seconds=1.0,
        allocated_bytes=0,
        reserved_bytes=0,
        last_checkpoint=None,
        last_recovery_sync=None,
    )
    assert row["percent"] == 50
    summary = stage_summary(
        result="passed",
        artifacts=[],
        metrics={},
        warnings=[],
        failed_items=[],
        next_eligible_stage="export",
        review_pack=None,
    )
    assert summary["next_eligible_stage"] == "export"


def test_failure_bundle_contains_actionable_context(tmp_path: Path) -> None:
    log = tmp_path / "stdout.log"
    log.write_text("last useful line\n", encoding="utf-8")
    try:
        raise RuntimeError("bounded failure")
    except RuntimeError as error:
        result = create_failure_bundle(
            tmp_path,
            campaign_id="eg-test",
            stage_id="semantic",
            error=error,
            environment={"disk": "available"},
            campaign_state={"status": "failed"},
            artifact_identities=[],
            recovery_status={"compatible": False},
            log_paths=(log,),
        )
    with zipfile.ZipFile(tmp_path / result["path"]) as bundle:
        assert set(bundle.namelist()) == {"failure.json", "log_tails.json"}
        assert b"bounded failure" in bundle.read("failure.json")
