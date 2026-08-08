"""Run one real optimizer-relevant step through every pinned MMSeg architecture.

This is deliberately not a fixture/mock test: it resolves each model's real
upstream MMSeg config the same way ``build_training_config`` does, builds the
real registered model, runs the real configured training pipeline over one
tiny on-disk image/mask pair, and calls the real ``model.loss()``. Earlier,
nothing in this repository exercised that path, which is exactly how a
missing ``GenerateEdge`` pipeline step for PIDNet-S (crashes on the first
optimizer step) and a weighted-cross-entropy override that silently no-oped
for OhemCrossEntropy-based decode heads both went undetected for months.

Requires the real pinned stack (torch/mmengine/mmcv/mmsegmentation) and a
local MMSegmentation checkout at the commit pinned by
``configs/rescue/semantic_first.yaml``, pointed to by
``EDGEGUARD_MMSEG_CHECKOUT``. Skipped entirely otherwise, matching the
existing convention in ``edgeguard.models.semantic_local``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from edgeguard.models.semantic_local import (
    local_mmseg_checkout_from_environment,
    semantic_environment_ready,
)
from edgeguard.rescue.config import load_rescue_config
from edgeguard.rescue.mmseg_runtime import build_training_config, install_mmcv_lite_guard
from edgeguard.serialization import sha256_file, sha256_payload

_MMSEG_CHECKOUT = local_mmseg_checkout_from_environment()
_STACK_READY = _MMSEG_CHECKOUT is not None and semantic_environment_ready(_MMSEG_CHECKOUT)

pytestmark = pytest.mark.skipif(
    not _STACK_READY,
    reason=(
        "requires the pinned torch/mmengine/mmcv/mmsegmentation stack and "
        "EDGEGUARD_MMSEG_CHECKOUT; see requirements/colab-openmmlab.lock"
    ),
)

_MODEL_NAMES = (
    "segformer_b0",
    "fast_scnn",
    "pidnet_s",
    "ddrnet_23_slim",
    "bisenetv2",
)


def _write_manifest(tmp_path: Path) -> Path:
    Image.fromarray((np.random.rand(64, 96, 3) * 255).astype(np.uint8)).save(tmp_path / "img.png")
    Image.fromarray(np.random.randint(0, 19, size=(64, 96)).astype(np.uint8)).save(
        tmp_path / "mask.png"
    )
    record = {
        "sample_id": "s0",
        "group_id": "g0",
        "image": "img.png",
        "mask": "mask.png",
        "canonical_mask": "mask.png",
    }
    payload: dict[str, Any] = {
        "schema_version": "2.0",
        "record_type": "edgeguard_dataset_manifest",
        "dataset_id": "cityscapes",
        "split_state": "frozen",
        "dataset_root": str(tmp_path),
        "prepared_root": str(tmp_path),
        "roles": {
            "train_fit": [record],
            "train_select": [dict(record, sample_id="s1", group_id="g1")],
        },
    }
    payload["manifest_sha256"] = sha256_payload(payload)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def _write_audit_report(tmp_path: Path, manifest_path: Path) -> Path:
    payload = {
        "source_role": "train_fit",
        "dataset_manifest_sha256s": [sha256_file(manifest_path)],
        "weights": [round(1.0 + 0.01 * index, 4) for index in range(19)],
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(payload), encoding="utf-8")
    return audit_path


def _resolved_model_config(
    *,
    model_name: str,
    work_dir: Path,
    manifest_path: Path,
    loss: str,
    audit_report: Path | None,
) -> Any:
    protocol = load_rescue_config(Path("configs/rescue/semantic_first.yaml"))
    return build_training_config(
        protocol,
        model_name=model_name,
        stage_name="smoke",
        mmseg_root=_MMSEG_CHECKOUT,
        dataset_root=None,
        split_manifest=None,
        work_dir=work_dir,
        loss=loss,
        audit_report=audit_report,
        resume=False,
        data_manifests=[manifest_path],
        initialization="random",
        precision="fp32",
    )


@pytest.mark.parametrize("model_name", _MODEL_NAMES)
def test_pidnet_pipeline_carries_generate_edge_only_for_pidnet(
    tmp_path: Path, model_name: str
) -> None:
    manifest_path = _write_manifest(tmp_path)
    cfg = _resolved_model_config(
        model_name=model_name,
        work_dir=tmp_path / "work",
        manifest_path=manifest_path,
        loss="ce",
        audit_report=None,
    )
    pipeline_types = [step["type"] for step in cfg.train_dataloader["dataset"]["pipeline"]]
    has_generate_edge = "GenerateEdge" in pipeline_types
    assert has_generate_edge == (model_name == "pidnet_s")


@pytest.mark.parametrize("model_name", _MODEL_NAMES)
def test_real_model_runs_one_real_loss_step(tmp_path: Path, model_name: str) -> None:
    """Every pinned architecture must accept its own configured pipeline output.

    This is the test that would have caught PIDNet-S's missing GenerateEdge
    step: it builds the real model from the real resolved config and calls
    the real ``model.loss()`` on data produced by the real training pipeline,
    instead of a hand-rolled generic cross-entropy over raw decode-head
    features (as the CPU mini-training harness in
    ``edgeguard.models.semantic_local`` does).
    """
    manifest_path = _write_manifest(tmp_path)
    cfg = _resolved_model_config(
        model_name=model_name,
        work_dir=tmp_path / "work",
        manifest_path=manifest_path,
        loss="ce",
        audit_report=None,
    )

    install_mmcv_lite_guard()
    import torch
    from mmengine.dataset import pseudo_collate
    from mmseg.registry import DATASETS, MODELS
    from mmseg.utils import register_all_modules

    from edgeguard.rescue.mmseg_components import register_mmseg_components

    register_all_modules(init_default_scope=True)
    register_mmseg_components()

    dataset = DATASETS.build(cfg.train_dataloader["dataset"])
    batch = pseudo_collate([dataset[0]])

    torch.manual_seed(0)
    model = MODELS.build(cfg.model)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    prepared = model.data_preprocessor(batch, True)
    losses = model.loss(prepared["inputs"], prepared["data_samples"])
    loss_terms = [
        value for key, value in losses.items() if "loss" in key and torch.is_tensor(value)
    ]
    assert loss_terms, f"{model_name} produced no loss terms"
    total = sum(loss_terms)
    assert bool(torch.isfinite(total)), f"{model_name} produced a non-finite loss"

    total.backward()
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ), f"{model_name} backward pass produced no finite gradients"


@pytest.mark.parametrize("model_name", _MODEL_NAMES)
def test_weighted_ce_ablation_changes_every_models_resolved_config(
    tmp_path: Path, model_name: str
) -> None:
    """SEG-06 (CE vs weighted-CE) must not silently no-op for any architecture.

    DDRNet-23-Slim's and PIDNet-S's dominant loss terms are OhemCrossEntropy,
    not CrossEntropyLoss; before the ``_set_num_classes_and_loss`` fix, the
    ``loss="ce"`` and ``loss="median_frequency"`` branches resolved to a
    byte-identical decode_head config for those two models.
    """
    manifest_path = _write_manifest(tmp_path)
    audit_report = _write_audit_report(tmp_path, manifest_path)

    ce_config = _resolved_model_config(
        model_name=model_name,
        work_dir=tmp_path / "work-ce",
        manifest_path=manifest_path,
        loss="ce",
        audit_report=None,
    )
    weighted_config = _resolved_model_config(
        model_name=model_name,
        work_dir=tmp_path / "work-weighted",
        manifest_path=manifest_path,
        loss="median_frequency",
        audit_report=audit_report,
    )

    assert dict(ce_config.model) != dict(weighted_config.model)
