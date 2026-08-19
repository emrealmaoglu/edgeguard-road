"""Regression test for the ``last_checkpoint`` marker format.

A real L4 Colab run completed an intentional-interruption smoke test (the
checkpoint saved fine, the process exited on purpose), then the resume
subprocess failed with ``FileNotFoundError: recovery_25.pth can not be
found.``. Root cause: ``EdgeGuardRecoveryHook.after_train_iter`` wrote only
the bare checkpoint filename into ``<work_dir>/last_checkpoint``. This
codebase's own readers (``latest_checkpoint`` in ``colab_recovery.py``)
resolve a relative marker against ``work_dir`` defensively, but MMEngine's
own built-in auto-resume (``Runner.load_or_resume`` ->
``find_latest_checkpoint``) does not: it returns the raw marker content
verbatim and resolves it relative to the process's current working
directory, which is the project root for every child process spawned by
``colab_pipeline.py``, not the run's ``work_dir``. The same bare-filename
bug existed in ``train_model``'s Drive cross-session recovery path
(``mmseg_runtime.py``), which is the mechanism this whole recovery system
exists to support.

Requires the real pinned stack (mmengine/mmcv/mmsegmentation) via
``EDGEGUARD_MMSEG_CHECKOUT``, matching the convention in
``test_mmseg_real_training_step.py``, since ``register_mmseg_components``
imports ``mmseg.datasets``/``mmseg.registry``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from edgeguard.models.semantic_local import (
    local_mmseg_checkout_from_environment,
    semantic_environment_ready,
)
from edgeguard.rescue.colab_recovery import latest_checkpoint
from edgeguard.rescue.mmseg_components import register_mmseg_components
from edgeguard.rescue.mmseg_runtime import install_mmcv_lite_guard

_MMSEG_CHECKOUT = local_mmseg_checkout_from_environment()
_STACK_READY = _MMSEG_CHECKOUT is not None and semantic_environment_ready(_MMSEG_CHECKOUT)

pytestmark = pytest.mark.skipif(
    not _STACK_READY,
    reason=(
        "requires the pinned torch/mmengine/mmcv/mmsegmentation stack and "
        "EDGEGUARD_MMSEG_CHECKOUT; see requirements/colab-openmmlab.lock"
    ),
)


class _FakeRunner:
    def __init__(self, work_dir: Path) -> None:
        self.iter = 0
        self.max_iters = 10
        self.work_dir = str(work_dir)
        self.epoch = 0

    def save_checkpoint(
        self,
        out_dir: str,
        *,
        filename: str,
        save_optimizer: bool,
        save_param_scheduler: bool,
        meta: dict[str, Any],
    ) -> None:
        del save_optimizer, save_param_scheduler, meta
        (Path(out_dir) / filename).write_bytes(b"fake-checkpoint")


def test_recovery_hook_marker_resolves_via_mmengines_own_auto_resume(tmp_path: Path) -> None:
    from mmengine.registry import HOOKS
    from mmengine.runner.checkpoint import find_latest_checkpoint

    install_mmcv_lite_guard()
    register_mmseg_components()
    hook_cls = HOOKS.get("EdgeGuardRecoveryHook")
    assert hook_cls is not None
    hook = hook_cls(
        store_root=str(tmp_path / "store"),
        artifact_id="test-artifact",
        campaign_id="test-campaign",
        project_commit="0" * 40,
        identity_sha256="a" * 64,
        accumulation=1,
        optimizer_interval=1,
        maximum_seconds=10_000,
    )
    work_dir = tmp_path / "run"
    work_dir.mkdir()
    runner = _FakeRunner(work_dir)

    hook.after_train_iter(runner, batch_idx=0)

    marker = work_dir / "last_checkpoint"
    assert marker.is_file()
    content = marker.read_text(encoding="utf-8").strip()
    assert Path(content).is_absolute(), "marker must hold an absolute path, not a bare filename"

    # This is the exact call MMEngine's Runner.load_or_resume makes on a
    # fresh process whose cwd is not work_dir; it must resolve to a real file.
    resume_from = find_latest_checkpoint(str(work_dir))
    assert resume_from is not None
    assert Path(resume_from).is_file()

    # This codebase's own defensive reader must keep agreeing with it.
    assert latest_checkpoint(work_dir) == Path(resume_from)
