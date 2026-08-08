"""Run the real `ColabPipeline` orchestrator end to end on CPU, tiny data.

Three real bugs (a hardcoded AMP dtype, a wrong `Pad` transform keyword
argument, a bare-filename `last_checkpoint` marker breaking MMEngine's own
auto-resume, and a fourth found *while building this test*: `Pad`'s `size`
argument passed in the wrong (h, w)/(w, h) order) were each discovered one
at a time on real Google Colab L4 GPU runs, one full Colab round-trip per
bug. All but the AMP dtype bug (CUDA-only) are fully CPU-reproducible and
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
    core model, real intentional-interruption-then-resume cycle."""
    from support.tiny_pipeline_fixture import build_tiny_pipeline

    pipeline = build_tiny_pipeline(tmp_path, REPO_ROOT, mmseg_root=_MMSEG_CHECKOUT)  # type: ignore[arg-type]

    result = pipeline.run("smoke")

    assert result["completed"] == ["preflight", "restore", "stage-data", "canary", "smoke"]
    metrics = json.loads((pipeline.state_root / "smoke/metrics.json").read_text())
    resumed = _interruption_resumes(metrics)
    assert {command["label"] for command in resumed} == {
        f"{model}-ce-resume" for model in CORE_MODELS
    }
    for command in resumed:
        proof = command["interruption_resume"]
        assert proof["verified"] is True
        assert proof["optimizer_step"] == 25
        assert len(proof["checkpoint_sha256"]) == 64
    run_manifest = json.loads((pipeline.state_root / "smoke/run_manifest.json").read_text())
    assert run_manifest["scientific_status"] == "not_run"
    assert run_manifest["synthetic_or_smoke"] is True


def test_extension_smoke_trains_and_resumes_extension_models_bypassing_pilot(
    tmp_path: Path,
) -> None:
    """Covers ddrnet_23_slim/bisenetv2 by calling the internal extension-smoke
    phase directly, skipping the real 2000-step pilot stage it would
    otherwise require — matching the existing "call a private phase method
    directly" convention already used in tests/unit/test_colab_pipeline.py.
    """
    from support.tiny_pipeline_fixture import build_tiny_pipeline

    pipeline = build_tiny_pipeline(tmp_path, REPO_ROOT, mmseg_root=_MMSEG_CHECKOUT)  # type: ignore[arg-type]
    pipeline.run("smoke")

    pipeline._run_phase("extension-smoke")  # noqa: SLF001

    assert pipeline._phase_complete("extension-smoke")  # noqa: SLF001
    metrics = json.loads((pipeline.state_root / "extension-smoke/metrics.json").read_text())
    resumed = _interruption_resumes(metrics)
    assert {command["label"] for command in resumed} == {
        f"{model}-ce-resume" for model in EXTENSION_MODELS
    }
    for command in resumed:
        proof = command["interruption_resume"]
        assert proof["verified"] is True
        assert proof["optimizer_step"] == 25
