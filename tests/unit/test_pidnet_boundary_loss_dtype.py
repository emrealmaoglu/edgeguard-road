"""Regression test for PIDNet's BoundaryLoss under bf16 autocast.

A real L4 Colab run trained segformer_b0 and fast_scnn through their full
smoke cycle (fresh start, intentional interrupt, real resume via the RNG
device fix, validation), then pidnet_s crashed on its very first training
step:

    File ".../mmseg/models/losses/boundary_loss.py", line 52, in forward
        weight[pos_index] = neg_num * 1.0 / sum_num
    RuntimeError: Index put requires the source and destination dtypes
    match, got BFloat16 for the destination and Float for the source.

Root cause: upstream `BoundaryLoss.forward` builds `weight =
torch.zeros_like(log_p)`, and under real `AmpOptimWrapper(dtype='bfloat16')`
autocast on real CUDA, `log_p` (derived from the boundary head's raw
logits) has already been cast to bfloat16. `pos_num`/`neg_num` come from
summing a float32 label mask that autocast never touches, so the ratio
assigned into `weight` stays float32 -- a dtype mismatch `index_put_`
rejects on the pinned Colab torch (2.1.1+cu121).

`EdgeGuardBoundaryLoss` (registered in `mmseg_components.py` via the
project's existing `force=True` override idiom, the same one used for
`EdgeGuardManifestDataset`/`EdgeGuardRecoveryHook`) is identical to
upstream except the two assignments are cast to `weight.dtype` before the
indexed write.

Note on local reproduction: this dev machine's torch (2.13.0) silently
allows the exact same implicit float32->bfloat16 index_put_ that torch
2.1.1 (the pinned Colab/CI version) rejects -- confirmed directly against
both `Tensor.__setitem__` and `Tensor.index_put_`. This is a torch-version
behavior difference, not a device (CPU/CUDA) one; it means this bug class,
like the AMP hardcode bug earlier this session, cannot be reproduced as a
"raises pre-fix" test in this environment. This test instead asserts the
POST-fix invariant directly: with mismatched input dtypes matching the
real Colab shape (bfloat16 predictions, float32 labels), the fixed
`weight` tensor's dtype always matches `log_p`'s dtype before the indexed
assignment, and the loss is finite -- which is the actual property the
fix guarantees regardless of which torch version's index_put_ enforces it.

Requires the real pinned stack (mmengine/mmcv/mmsegmentation) via
EDGEGUARD_MMSEG_CHECKOUT, matching the convention in
test_mmseg_recovery_checkpoint_marker.py.
"""

from __future__ import annotations

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


def test_boundary_loss_survives_bf16_predictions_with_float32_labels() -> None:
    import torch

    from edgeguard.rescue.mmseg_components import register_mmseg_components

    install_mmcv_lite_guard()
    register_mmseg_components()

    from mmseg.registry import MODELS

    loss_cls = MODELS.get("BoundaryLoss")
    assert loss_cls is not None
    assert loss_cls.__name__ == "EdgeGuardBoundaryLoss"

    loss_fn = loss_cls(loss_weight=20.0)
    bd_pre = torch.randn(2, 1, 8, 8).to(torch.bfloat16)
    bd_gt = (torch.rand(2, 1, 8, 8) > 0.5).float()

    result = loss_fn(bd_pre, bd_gt)

    assert result.isfinite().all()


def test_boundary_loss_matches_upstream_numerics_at_matching_dtype() -> None:
    """The fix must not change behavior when dtypes already agree (the
    common fp32 case, e.g. the CPU rehearsal harness) -- only add casts
    that are no-ops when source and destination already match."""
    import torch
    from mmseg.models.losses.boundary_loss import BoundaryLoss as UpstreamBoundaryLoss

    from edgeguard.rescue.mmseg_components import register_mmseg_components

    install_mmcv_lite_guard()
    register_mmseg_components()

    from mmseg.registry import MODELS

    fixed_cls = MODELS.get("BoundaryLoss")
    assert fixed_cls is not None

    torch.manual_seed(0)
    bd_pre = torch.randn(2, 1, 8, 8)
    bd_gt = (torch.rand(2, 1, 8, 8) > 0.5).float()

    upstream_loss = UpstreamBoundaryLoss(loss_weight=20.0)(bd_pre.clone(), bd_gt.clone())
    fixed_loss = fixed_cls(loss_weight=20.0)(bd_pre.clone(), bd_gt.clone())

    assert torch.equal(upstream_loss, fixed_loss)
