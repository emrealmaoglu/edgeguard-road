"""Shared tiny-fixture builders for a real, CPU-only `ColabPipeline` rehearsal.

Generalizes two previously narrower, near-duplicate fixture helpers:

- ``tests/unit/test_mmseg_real_training_step.py::_write_manifest`` (single
  role, missing ``audit_passed``/``scientific_eligible``, so it could only
  ever drive ``build_training_config`` directly, never a real
  ``train_model()`` call).
- ``tests/unit/test_colab_pipeline.py::_write_staged_dataset_manifest``/``_pipeline``
  (missing the ``train_select`` role and the same two eligibility fields, and
  always paired with a mocked ``_run_command``, so it never drove a real
  training subprocess either).

Neither existing fixture is reusable as-is to drive a real
``ColabPipeline.run("smoke")`` against the real pinned MMSeg/MMEngine stack;
this module fills that specific gap.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from edgeguard.rescue.colab_pipeline import ColabPipeline, PipelineInputs
from edgeguard.serialization import sha256_payload

_STAGES_TEMPLATE = """stages:
  smoke:
    max_steps: {smoke_max_steps}
  pilot:
    max_steps: 2000
  screening:
    max_steps: 6000
  hpo:
    max_steps: 6000
  final:
    max_steps: 40000
"""


def write_tiny_dataset_manifest(
    root: Path,
    *,
    dataset_id: Literal["cityscapes", "idd20k"],
    image_size: tuple[int, int] = (256, 512),
) -> Path:
    """Write a real, schema-valid, ``train_model()``-eligible frozen manifest.

    Includes both ``train_fit`` and ``train_select`` roles (``train_model()``
    requires both) and sets ``audit_passed``/``scientific_eligible`` to
    ``True`` (also required, and absent from the narrower existing
    fixtures). Two tiny real on-disk PNGs per role.

    ``image_size`` must be at least as large as the rescue config's
    ``crop_size`` in both dimensions (see ``write_tiny_rescue_config``),
    matching real Cityscapes/IDD20K images (natively 2048x1024/1920x1080),
    which are always larger than any reasonable training crop — never
    smaller. The evaluation pipeline's ``EncoderDecoder`` postprocessing
    resizes predictions back down to each sample's original (``ori_shape``)
    resolution before scoring; a fixture image *smaller* than ``crop_size``
    (e.g. the 64x96 convention ``test_mmseg_real_training_step.py`` uses,
    which never runs evaluation) makes the resized-back prediction shape
    mismatch the still-padded-to-``crop_size`` ground truth mask, raising
    ``IndexError: The shape of the mask ... does not match the shape of the
    indexed tensor`` — a fixture-realism artifact, not a real training bug.
    """
    if image_size[0] < 1 or image_size[1] < 1:
        raise ValueError("image_size must be positive")
    root.mkdir(parents=True, exist_ok=True)

    def _record(role: str, index: int) -> dict[str, Any]:
        image_name = f"{role}-{index}-img.png"
        mask_name = f"{role}-{index}-mask.png"
        Image.fromarray((np.random.rand(*image_size, 3) * 255).astype(np.uint8)).save(
            root / image_name
        )
        Image.fromarray(np.random.randint(0, 19, size=image_size).astype(np.uint8)).save(
            root / mask_name
        )
        return {
            "sample_id": f"{dataset_id}-{role}-s{index}",
            "group_id": f"{dataset_id}-{role}-g{index}",
            "image": image_name,
            "mask": mask_name,
            "canonical_mask": mask_name,
        }

    payload: dict[str, Any] = {
        "schema_version": "2.0",
        "record_type": "edgeguard_dataset_manifest",
        "dataset_id": dataset_id,
        "split_state": "frozen",
        "audit_passed": True,
        "scientific_eligible": True,
        "dataset_root": str(root),
        "prepared_root": str(root),
        "roles": {
            "train_fit": [_record("train_fit", 0), _record("train_fit", 1)],
            "train_select": [_record("train_select", 0), _record("train_select", 1)],
        },
    }
    payload["manifest_sha256"] = sha256_payload(payload)
    manifest_path = root.parent / f"{dataset_id}.frozen.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def write_tiny_runtime_receipt(path: Path, *, project_commit: str) -> Path:
    """Write a runtime receipt satisfying ``ColabPipeline._verify_canary()``.

    The ``canary`` phase never runs real GPU/model code; it only checks
    these JSON fields already match a completed five-model probe.
    """
    payload = {
        "record_type": "semantic_hermetic_runtime_receipt",
        "runtime_profile": "py311-cu121",
        "project_commit": project_commit,
        "lock_sha256": {"lock": "a" * 64},
        "environment": {"cuda_available": False},
        "core_model_probe": {
            "model_count": 5,
            "fp16_finite_model_count": 5,
            "checkpoint_resume_verified": True,
            "checkpoint_resume_model_count": 5,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_tiny_rescue_config(
    path: Path,
    *,
    smoke_max_steps: int = 30,
    crop_size: tuple[int, int] = (256, 512),
    device_batch: int = 2,
    workers: int = 0,
) -> Path:
    """Copy the frozen ``models:``/``datasets:``/``hpo:`` sections of the real
    production protocol verbatim and shrink only the engineering knobs that
    are not scientifically frozen (see ``src/edgeguard/rescue/config.py``):
    ``crop_size``, ``device_batch``, ``gradient_accumulation``, ``workers``,
    and ``stages.smoke.max_steps``.

    ``device_batch`` defaults to 2, not 1: some backbones (e.g. Fast-SCNN's
    pyramid pooling module) pool down to a 1x1 spatial size internally, and
    ``BatchNorm`` raises ``ValueError: Expected more than 1 value per
    channel when training`` with a batch of exactly 1 at that resolution.

    ``smoke_max_steps`` must stay greater than 25: production-mode
    ``ColabPipeline._train_command`` hardcodes
    ``--intentional-interrupt-step 25`` for the ``smoke``/``extension-smoke``
    phases, and ``train_model()`` requires
    ``0 < intentional_interrupt_optimizer_step < resolved_max_steps``.
    """
    if smoke_max_steps <= 25:
        raise ValueError("smoke_max_steps must exceed the hardcoded interrupt step of 25")
    real_config = Path("configs/rescue/semantic_first.yaml").read_text(encoding="utf-8")
    lines = real_config.splitlines(keepends=True)
    header_end = next(index for index, line in enumerate(lines) if line.startswith("models:"))
    models_and_optimizer = "".join(lines[:header_end])
    models_end = next(index for index, line in enumerate(lines) if line.startswith("stages:"))
    models_section = "".join(lines[header_end:models_end])
    hpo_start = next(index for index, line in enumerate(lines) if line.startswith("hpo:"))
    hpo_and_datasets = "".join(lines[hpo_start:])
    config_text = (
        models_and_optimizer.replace(
            "crop_size: [512, 1024]", f"crop_size: [{crop_size[0]}, {crop_size[1]}]"
        )
        .replace("device_batch: 4", f"device_batch: {device_batch}")
        .replace("workers: 6", f"workers: {workers}")
        + models_section
        + _STAGES_TEMPLATE.format(smoke_max_steps=smoke_max_steps)
        + hpo_and_datasets
    )
    path.write_text(config_text, encoding="utf-8")
    return path


def current_git_head(project_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_tiny_pipeline(
    tmp_path: Path,
    project_root: Path,
    *,
    mmseg_root: Path,
    execution_mode: Literal["production"] = "production",
    crop_size: tuple[int, int] = (256, 512),
) -> ColabPipeline:
    """Assemble a real ``ColabPipeline`` wired to tiny fixture data end to end.

    ``execution_mode`` defaults to (and should stay) ``"production"``, not
    ``"acceptance"``: acceptance mode never adds
    ``--intentional-interrupt-step`` (see ``_train_command``'s mutually
    exclusive ``if``/``elif`` branches), so it cannot exercise the
    interrupt+resume path this rehearsal exists to cover.

    The same ``crop_size`` is passed to both ``write_tiny_rescue_config``
    and ``write_tiny_dataset_manifest`` (as ``image_size``) so fixture
    images always stay at least as large as the training crop — see
    ``write_tiny_dataset_manifest`` for why a smaller image breaks real
    evaluation.
    """
    project_commit = current_git_head(project_root)
    runtime_receipt = write_tiny_runtime_receipt(
        tmp_path / "runtime_receipt.json", project_commit=project_commit
    )
    config_path = write_tiny_rescue_config(tmp_path / "semantic_first.yaml", crop_size=crop_size)
    manifests = (
        write_tiny_dataset_manifest(
            tmp_path / "cityscapes-data", dataset_id="cityscapes", image_size=crop_size
        ),
        write_tiny_dataset_manifest(
            tmp_path / "idd20k-data", dataset_id="idd20k", image_size=crop_size
        ),
    )
    return ColabPipeline(
        PipelineInputs(
            project_root=project_root,
            project_commit=project_commit,
            runtime_receipt=runtime_receipt,
            mmseg_root=mmseg_root,
            work_root=tmp_path / "work",
            recovery_root=tmp_path / "drive-recovery",
            config_path=config_path,
            data_manifests=manifests,
            execution_mode=execution_mode,
        )
    )
