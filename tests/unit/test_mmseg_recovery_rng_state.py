"""Regression test for the checkpoint RNG-state device mismatch.

A real L4 Colab run's `smoke`/`segformer_b0` intentional-interruption cycle
saved a checkpoint fine, then the resume subprocess crashed inside
`Runner.resume()` -> `call_hook('after_load_checkpoint')` with:

    TypeError: RNG state must be a torch.ByteTensor in EdgeGuardRecoveryHook

Root cause: `Runner.resume()` loads the checkpoint with
`map_location=get_device()` (`mmengine/runner/runner.py`), which moves every
tensor in the pickle -- including the RNG state
`EdgeGuardRecoveryHook.before_save_checkpoint` stores under
`checkpoint["edgeguard_recovery_state"]` -- onto that device. On Colab that
is CUDA; `torch.set_rng_state`/`torch.cuda.set_rng_state` require a CPU
uint8 tensor and reject anything else. mmengine 0.10.7 has no RNG
save/restore of its own (confirmed: `grep -rn rng` in the installed package
returns nothing), so this hook is genuinely new capability, not a
duplicate -- it must be made device-agnostic, not deleted.

This machine has no CUDA. `mmengine.device.get_device()` returns "mps" here
(Apple Silicon), which is itself a real foreign device relative to the CPU
tensor `torch.get_rng_state()` returns -- so simulating the move with
`.to("mps")` reproduces the *exact* real-world TypeError without any GPU.
On a CUDA-less, MPS-less CI box the test falls back to a wrong-dtype tensor,
which raises the identical error message and is the same class of bug.

Requires the real pinned stack (mmengine/mmcv/mmsegmentation) via
EDGEGUARD_MMSEG_CHECKOUT, matching the convention in
test_mmseg_recovery_checkpoint_marker.py.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from edgeguard.models.semantic_local import (
    local_mmseg_checkout_from_environment,
    semantic_environment_ready,
)
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


def _build_hook(tmp_path: Path) -> Any:
    from mmengine.registry import HOOKS

    from edgeguard.rescue.mmseg_components import register_mmseg_components

    register_mmseg_components()
    hook_cls = HOOKS.get("EdgeGuardRecoveryHook")
    assert hook_cls is not None
    return hook_cls(
        store_root=str(tmp_path / "store"),
        artifact_id="test-artifact",
        campaign_id="test-campaign",
        project_commit="0" * 40,
        identity_sha256="a" * 64,
        accumulation=1,
        optimizer_interval=1,
        maximum_seconds=10_000,
    )


def _foreign_copy(torch_module: Any, state: Any) -> Any:
    """Return `state` as it would come back through Runner.resume's
    map_location=get_device() on a Colab GPU box -- moved to whatever
    non-CPU device this machine actually has, or a wrong dtype otherwise."""
    if torch_module.cuda.is_available():
        return state.to("cuda")
    if torch_module.backends.mps.is_available():
        return state.to("mps")
    return state.to(torch_module.float32)


def test_foreign_device_rng_state_is_rejected_by_torch_before_the_fix() -> None:
    """Pin the premise: torch itself must still reject a non-CPU-ByteTensor
    RNG state. If a future torch version relaxes this, this test (not the
    fix's regression test) is the one that should start failing, so the fix
    below doesn't silently become a no-op."""
    import torch

    state = torch.get_rng_state()
    foreign = _foreign_copy(torch, state)
    with pytest.raises(TypeError, match="ByteTensor"):
        torch.set_rng_state(foreign)


def test_recovery_hook_restores_rng_state_saved_and_reloaded_across_a_foreign_device(
    tmp_path: Path,
) -> None:
    import torch

    install_mmcv_lite_guard()
    hook = _build_hook(tmp_path)

    original = torch.get_rng_state().clone()
    runner = SimpleNamespace(iter=0)
    checkpoint: dict[str, Any] = {}
    hook.before_save_checkpoint(runner, checkpoint)

    state = checkpoint["edgeguard_recovery_state"]
    assert state["torch_random"].device.type == "cpu"
    assert state["torch_random"].dtype == torch.uint8

    # Simulate what Runner.resume's map_location=get_device() does to this
    # tensor on real Colab hardware before after_load_checkpoint ever runs.
    state["torch_random"] = _foreign_copy(torch, state["torch_random"])
    if state.get("cuda_random") is not None:
        state["cuda_random"] = [_foreign_copy(torch, item) for item in state["cuda_random"]]

    # Move the RNG somewhere else first so restoring it is actually observable.
    torch.manual_seed(1234)
    assert not torch.equal(torch.get_rng_state(), original)

    hook.after_load_checkpoint(runner, checkpoint)

    assert torch.equal(torch.get_rng_state(), original)
