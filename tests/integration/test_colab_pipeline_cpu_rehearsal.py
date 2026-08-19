"""Run the real `ColabPipeline` orchestrator end to end on CPU, tiny data.

Five real bugs (a hardcoded AMP dtype, a wrong `Pad` transform keyword
argument, a bare-filename `last_checkpoint` marker breaking MMEngine's own
auto-resume, a `Pad` `size` argument passed in the wrong (h, w)/(w, h)
order, and a stale cross-commit Drive recovery pointer crashing the whole
campaign before any training step ran) were each discovered one at a time
on real Google Colab L4 GPU runs, one full Colab round-trip per bug. All
but the AMP dtype bug (CUDA-only) are fully CPU-reproducible and
device-independent; nothing in this repository drove the real orchestrator
(`ColabPipeline.run`), the real `EdgeGuardRecoveryHook` interrupt+resume
cycle, or the real `val_dataloader` build (`Compose()` with
`_evaluation_pipeline`) before this test existed. The closest thing,
`scripts/dev/run_campaign_notebook_harness.py`'s "claim-safe local cell
execution", stubs the entire training call behind a hardcoded
`{"scientific_status": "not_run"}` dict under
`EDGEGUARD_NOTEBOOK_LOCAL_TEST=1` and never touches any of this.

This test drives the same subprocess-spawning `ColabPipeline` production
code Colab itself runs (`_run_command`'s real `subprocess.Popen(...,
cwd=project_root)`, matching the exact cwd-vs-work_dir mismatch that broke
`last_checkpoint`), against tiny synthetic fixture manifests
(`tests/support/tiny_pipeline_fixture.py`), so this bug class is caught
here in seconds instead of on the next real Colab L4 run.

Requires the real pinned stack (torch/mmengine/mmcv/mmsegmentation) via
`EDGEGUARD_MMSEG_CHECKOUT`, matching the convention in
`test_mmseg_real_training_step.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from edgeguard.models.semantic_local import (
    local_mmseg_checkout_from_environment,
    semantic_environment_ready,
)
from edgeguard.rescue.colab_pipeline import CORE_MODELS, EXTENSION_MODELS

REPO_ROOT = Path(__file__).resolve().parents[2]

# Not `from tests.support...`: an installed `ultralytics` ships its own
# top-level `tests/` package in site-packages, which shadows this repo's
# `tests/` for dotted-package imports. Import `support` as a fresh
# top-level module from `tests/` directly instead, matching no existing
# precedent in this repo because none was ever needed before this file.
_TESTS_ROOT = str(REPO_ROOT / "tests")
if _TESTS_ROOT not in sys.path:
    sys.path.insert(0, _TESTS_ROOT)

_MMSEG_CHECKOUT = local_mmseg_checkout_from_environment()
_STACK_READY = _MMSEG_CHECKOUT is not None and semantic_environment_ready(_MMSEG_CHECKOUT)

pytestmark = pytest.mark.skipif(
    not _STACK_READY,
    reason=(
        "requires the pinned torch/mmengine/mmcv/mmsegmentation stack and "
        "EDGEGUARD_MMSEG_CHECKOUT; see requirements/colab-openmmlab.lock"
    ),
)


def _interruption_resumes(metrics: dict) -> list[dict]:
    return [command for command in metrics["commands"] if "interruption_resume" in command]


def test_smoke_target_trains_and_resumes_core_models_through_real_pipeline(
    tmp_path: Path,
) -> None:
    """Real preflight/restore/stage-data/canary/smoke, real subprocess per
    core model, real intentional-interruption-then-resume cycle for the
    campaign's single recovery self-test model (CORE_MODELS[0]) only.

    The self-test used to run once per model (5 deliberate crash+resume
    cycles per campaign); it now runs once per campaign
    (`PipelineInputs.recovery_self_test_model`, default `CORE_MODELS[0]`) --
    proving the shared interrupt+resume code path once on real hardware is
    sufficient evidence, and each extra cycle was itself a chance for the
    self-test's own resume leg to fail and take the whole campaign down with
    it (see `test_smoke_target_skips_stale_cross_commit_drive_recovery_checkpoint`'s
    sibling in this file for the failure-handling case). The `>=` assertion
    below guards against a future regression that shrinks the self-test
    silently dropping a model from actually training.
    """
    from support.tiny_pipeline_fixture import build_tiny_pipeline

    pipeline = build_tiny_pipeline(tmp_path, REPO_ROOT, mmseg_root=_MMSEG_CHECKOUT)  # type: ignore[arg-type]

    result = pipeline.run("smoke")

    assert result["completed"] == ["preflight", "restore", "stage-data", "canary", "smoke"]
    metrics = json.loads((pipeline.state_root / "smoke/metrics.json").read_text())
    resumed = _interruption_resumes(metrics)
    assert {command["label"] for command in resumed} == {f"{CORE_MODELS[0]}-ce-resume"}
    assert len(resumed) == 1
    # The self-test model's first attempt is deliberately interrupted before
    # _run_command can return, so only its "-resume" label is ever recorded;
    # every other core model trains straight through under its plain label.
    expected_labels = {f"{model}-ce" for model in CORE_MODELS if model != CORE_MODELS[0]}
    expected_labels.add(f"{CORE_MODELS[0]}-ce-resume")
    assert {command["label"] for command in metrics["commands"]} >= expected_labels
    for command in resumed:
        proof = command["interruption_resume"]
        assert proof["verified"] is True
        assert proof["optimizer_step"] == 25
        assert len(proof["checkpoint_sha256"]) == 64
    run_manifest = json.loads((pipeline.state_root / "smoke/run_manifest.json").read_text())
    assert run_manifest["scientific_status"] == "not_run"
    assert run_manifest["synthetic_or_smoke"] is True


def test_smoke_target_skips_stale_cross_commit_drive_recovery_checkpoint(
    tmp_path: Path,
) -> None:
    """A Drive recovery pointer left over from an earlier, incompatible commit
    must not crash the campaign on a fresh Colab session's first attempt.

    Reproduces the fifth real bug found on Colab L4: a brand-new session has
    an empty local work_dir (no local run_identity.json), so
    `ColabPipeline._run_training_phase` speculatively appends `--resume`
    purely because a Drive pointer for this artifact_id exists
    (`_recovery_pointer_exists` has zero identity awareness). `train_model()`
    then restored the stale checkpoint, found its `identity_sha256` did not
    match the current run (every commit changes `project_commit`, which is
    part of the identity), and hard-raised
    `ValueError("Drive recovery checkpoint belongs to a different immutable
    run")`, crashing the whole 5-model campaign before a single training
    step ran. The fix makes this speculative path degrade to a fresh run
    instead of raising; an explicit resume of a known *local* run must still
    raise (that check, `existing != identity` in `train_model()`, is
    untouched by this fix and has no CPU-rehearsal coverage gap to close
    here).
    """
    from support.tiny_pipeline_fixture import build_tiny_pipeline

    from edgeguard.rescue.colab_recovery import publish_recovery_file

    pipeline = build_tiny_pipeline(tmp_path, REPO_ROOT, mmseg_root=_MMSEG_CHECKOUT)  # type: ignore[arg-type]

    stale_source = tmp_path / "stale-checkpoint.pth"
    stale_source.write_bytes(b"stale-checkpoint-from-an-earlier-incompatible-commit")
    publish_recovery_file(
        stale_source,
        pipeline.inputs.recovery_root,
        artifact_id="smoke-segformer-b0-ce",
        campaign_id=pipeline.inputs.campaign_id,
        project_commit="0" * 40,
        metadata={"identity_sha256": "f" * 64},
    )

    result = pipeline.run("smoke")

    assert result["completed"] == ["preflight", "restore", "stage-data", "canary", "smoke"]
    run_dir = pipeline.inputs.work_root / "runs" / "smoke" / "segformer_b0" / "ce"
    assert not (run_dir / "recovered.pth").is_file()
    stale_marker = json.loads((run_dir / "stale_recovery_skipped.json").read_text())
    assert stale_marker["artifact_id"] == "smoke-segformer-b0-ce"
    assert stale_marker["found_identity_sha256"] == "f" * 64


def test_extension_smoke_trains_extension_models_without_repeating_the_recovery_self_test(
    tmp_path: Path,
) -> None:
    """Covers ddrnet_23_slim/bisenetv2 by calling the internal extension-smoke
    phase directly, skipping the real 2000-step pilot stage it would
    otherwise require — matching the existing "call a private phase method
    directly" convention already used in tests/unit/test_colab_pipeline.py.

    Neither extension model is `PipelineInputs.recovery_self_test_model`
    (default `CORE_MODELS[0]`, already exercised by `smoke` above), so
    neither should get an intentional-interrupt-then-resume cycle here —
    the self-test runs once per campaign, not once per model.
    """
    from support.tiny_pipeline_fixture import build_tiny_pipeline

    pipeline = build_tiny_pipeline(tmp_path, REPO_ROOT, mmseg_root=_MMSEG_CHECKOUT)  # type: ignore[arg-type]
    pipeline.run("smoke")

    pipeline._run_phase("extension-smoke")  # noqa: SLF001

    assert pipeline._phase_complete("extension-smoke")  # noqa: SLF001
    metrics = json.loads((pipeline.state_root / "extension-smoke/metrics.json").read_text())
    resumed = _interruption_resumes(metrics)
    assert resumed == []
    assert {command["label"] for command in metrics["commands"]} >= {
        f"{model}-ce" for model in EXTENSION_MODELS
    }
