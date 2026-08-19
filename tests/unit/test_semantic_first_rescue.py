from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from edgeguard.rescue.config import load_rescue_config, model_by_name
from edgeguard.rescue.dataset import (
    EXPECTED_ROLE_COUNTS,
    SemanticSample,
    audit_cityscapes,
    build_split_manifest,
    median_frequency_weights,
    validate_split_manifest,
    write_train_fit_statistics,
)
from edgeguard.rescue.hpo_runtime import select_hpo_models
from edgeguard.rescue.inference import discover_demo_models, preprocess_image
from edgeguard.rescue.mmseg_runtime import (
    _acdc_dataset,
    build_training_config,
    materialize_role_file,
    resolve_auto_precision,
    validate_scientific_split,
)
from edgeguard.rescue.multidomain import build_cityscapes_official_validation_manifest
from edgeguard.rescue.reporting import build_evidence_report
from edgeguard.rescue.selection import select_top_two
from edgeguard.rescue.stress import build_stress_dataset
from edgeguard.rescue.visualization import (
    InferenceResult,
    calibrate_inference_result,
    confidence_entropy,
    overlay_mask,
    resize_mask,
)
from edgeguard.serialization import sha256_file


def _sample(index: int) -> SemanticSample:
    city = f"city{index % 20:02d}"
    identifier = f"{city}_{index:06d}_000019"
    return SemanticSample(
        sample_id=identifier,
        city=city,
        group_id=f"{city}_{index:06d}",
        image=f"leftImg8bit/train/{city}/{identifier}_leftImg8bit.png",
        mask=f"gtFine/train/{city}/{identifier}_gtFine_labelTrainIds.png",
    )


def test_rescue_config_is_frozen_to_five_step_based_models() -> None:
    config = load_rescue_config(Path("configs/rescue/semantic_first.yaml"))
    assert [model.name for model in config.models] == [
        "segformer_b0",
        "fast_scnn",
        "pidnet_s",
        "ddrnet_23_slim",
        "bisenetv2",
    ]
    assert config.effective_batch == 4
    assert config.stages["smoke"].max_steps == 50
    assert config.stages["pilot"].max_steps == 600
    assert config.stages["screening"].max_steps == 2_500
    assert config.hpo.trials_per_model == 3
    # Every pruning rung must sit strictly below the ceiling, or a trial would be
    # "pruned" at a budget it can never reach and the study would never terminate.
    assert config.hpo.pruning_steps[-1] < config.hpo.max_steps
    assert model_by_name(config, "pidnet_s").upstream_config.endswith("cityscapes.py")
    with pytest.raises(ValueError, match="unsupported model"):
        model_by_name(config, "unknown_model")


def test_split_manifest_has_exact_counts_and_no_group_leakage() -> None:
    samples = [_sample(index) for index in range(2975)]
    manifest = build_split_manifest(samples, seed=20260728)
    validate_split_manifest(manifest, samples)
    assert manifest["counts"] == EXPECTED_ROLE_COUNTS
    assert sum(len(records) for records in manifest["roles"].values()) == 2975


def test_split_rejects_wrong_dataset_size() -> None:
    with pytest.raises(ValueError, match="2,975"):
        build_split_manifest([_sample(0)], seed=1)


def test_median_frequency_weights_are_positive_and_mean_one() -> None:
    weights = median_frequency_weights(range(1, 20))
    assert len(weights) == 19
    assert min(weights) > 0
    assert np.mean(weights) == pytest.approx(1.0)


def test_cityscapes_audit_writes_expected_reports(tmp_path: Path) -> None:
    root = tmp_path / "cityscapes"
    for index in range(3):
        sample = _sample(index)
        image_path = root / sample.image
        mask_path = root / sample.mask
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        intensity = np.tile(np.arange(19, dtype=np.uint8) * 10, (3, 1)) + index
        image = np.stack((intensity, np.flip(intensity, axis=1), intensity), axis=-1)
        mask = np.tile(np.arange(19, dtype=np.uint8), (3, 1))
        Image.fromarray(image, mode="RGB").save(image_path)
        Image.fromarray(mask, mode="L").save(mask_path)
    output = tmp_path / "reports"
    summary = audit_cityscapes(root, output)
    assert summary["audit_passed"] is True
    assert summary["valid_pairs"] == 3
    report_root = output / "dataset_audit"
    assert (report_root / "summary.json").is_file()
    assert (report_root / "class_pixel_frequency.csv").is_file()
    assert (report_root / "class_image_frequency.csv").is_file()
    assert (report_root / "crop_survival.csv").is_file()
    assert (report_root / "class_cooccurrence.csv").is_file()
    assert (report_root / "near_duplicates.csv").is_file()
    assert json.loads((report_root / "class_weights.json").read_text())["weights"] is not None


def test_cityscapes_audit_blocks_black_and_all_ignore_samples(tmp_path: Path) -> None:
    root = tmp_path / "cityscapes"
    sample = _sample(0)
    image_path = root / sample.image
    mask_path = root / sample.mask
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 4), color=(0, 0, 0)).save(image_path)
    Image.fromarray(np.full((4, 8), 255, dtype=np.uint8), mode="L").save(mask_path)
    summary = audit_cityscapes(root, tmp_path / "reports")
    assert summary["audit_passed"] is False
    assert summary["black_image_count"] == 1
    assert summary["all_ignore_mask_count"] == 1


def test_cityscapes_official_val_manifest_is_separate_and_review_required(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cityscapes"
    identifier = "valcity_000001_000019"
    image_path = root / f"leftImg8bit/val/valcity/{identifier}_leftImg8bit.png"
    mask_path = root / f"gtFine/val/valcity/{identifier}_gtFine_labelTrainIds.png"
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    intensity = np.tile(np.arange(19, dtype=np.uint8) * 10, (3, 1))
    Image.fromarray(np.stack((intensity, intensity, intensity), axis=-1), mode="RGB").save(
        image_path
    )
    Image.fromarray(np.tile(np.arange(19, dtype=np.uint8), (3, 1)), mode="L").save(mask_path)
    summary = audit_cityscapes(root, tmp_path / "val-audit", split="val")
    candidate_path = tmp_path / "cityscapes-val.candidate.json"
    candidate = build_cityscapes_official_validation_manifest(
        root,
        summary,
        candidate_path,
        source_manifests=(),
        strict_count=False,
    )
    assert candidate["split_state"] == "candidate_requires_human_freeze"
    assert set(candidate["roles"]) == {"official_source_val"}
    assert candidate["scientific_eligible"] is True


def test_split_statistics_are_derived_for_every_role(tmp_path: Path) -> None:
    root = tmp_path / "cityscapes"
    sample = _sample(0)
    mask_path = root / sample.mask
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(np.tile(np.arange(19, dtype=np.uint8), (2, 1)), mode="L").save(mask_path)
    record = {
        "sample_id": sample.sample_id,
        "group_id": sample.group_id,
        "image": sample.image,
        "mask": sample.mask,
    }
    manifest = tmp_path / "CSF-SPLIT-D.json"
    manifest.write_text(
        json.dumps(
            {
                "roles": {
                    "train_fit": [record],
                    "train_select": [record],
                    "train_calibration": [record],
                }
            }
        ),
        encoding="utf-8",
    )
    report_root = tmp_path / "report"
    report_root.mkdir()
    result = write_train_fit_statistics(root, manifest, report_root)
    assert result["split_summary"]["all_roles_have_all_classes"] is True
    assert (report_root / "split_comparison.csv").is_file()


def test_scientific_runtime_requires_a_passing_audit_receipt(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="summary.json"):
        validate_scientific_split(tmp_path, tmp_path / "CSF-SPLIT-D.json")


def test_mmseg_role_file_uses_suffix_free_cityscapes_identifiers(tmp_path: Path) -> None:
    sample = _sample(0)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "roles": {
                    "train_fit": [
                        {
                            "sample_id": sample.sample_id,
                            "group_id": sample.group_id,
                            "image": sample.image,
                            "mask": sample.mask,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "train_fit.txt"
    assert materialize_role_file(manifest, "train_fit", destination) == 1
    assert destination.read_text().strip() == f"{sample.city}/{sample.sample_id}"


def test_all_models_receive_the_same_model_input_size(tmp_path: Path) -> None:
    pytest.importorskip("mmengine", reason="MMSeg config resolution is an optional integration")
    protocol = load_rescue_config(Path("configs/rescue/semantic_first.yaml"))
    sample = _sample(0)
    record = {
        "sample_id": sample.sample_id,
        "group_id": sample.group_id,
        "image": sample.image,
        "mask": sample.mask,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"roles": {"train_fit": [record], "train_select": [record]}},
        ),
        encoding="utf-8",
    )
    mmseg_root = tmp_path / "mmseg"
    fake_config = """
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(type='SegDataPreProcessor', size=(1024, 1024)),
    decode_head=dict(
        type='FakeHead',
        num_classes=150,
        loss_decode=dict(type='CrossEntropyLoss'),
    ),
)
default_hooks = dict(checkpoint=dict(type='CheckpointHook'))
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=6e-05, weight_decay=0.01))
"""
    for model in protocol.models:
        upstream = mmseg_root / model.upstream_config
        upstream.parent.mkdir(parents=True, exist_ok=True)
        upstream.write_text(fake_config, encoding="utf-8")
        cfg = build_training_config(
            protocol,
            model_name=model.name,
            stage_name="smoke",
            mmseg_root=mmseg_root,
            dataset_root=tmp_path / "cityscapes",
            split_manifest=manifest,
            work_dir=tmp_path / "work" / model.name,
            loss="ce",
            audit_report=None,
            resume=False,
        )
        assert tuple(cfg.model.data_preprocessor.size) == protocol.crop_size

    weights = tmp_path / "class_weights.json"
    weights.write_text(
        json.dumps(
            {
                "source_role": "train_fit_pending_split_filter",
                "weights": [1.0] * 19,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen train_fit"):
        build_training_config(
            protocol,
            model_name="fast_scnn",
            stage_name="smoke",
            mmseg_root=mmseg_root,
            dataset_root=tmp_path / "cityscapes",
            split_manifest=manifest,
            work_dir=tmp_path / "weighted-provisional",
            loss="median_frequency",
            audit_report=weights,
            resume=False,
        )
    weights.write_text(
        json.dumps(
            {
                "source_role": "train_fit",
                "split_manifest_sha256": sha256_file(manifest),
                "weights": [1.0] * 19,
            }
        ),
        encoding="utf-8",
    )
    weighted = build_training_config(
        protocol,
        model_name="fast_scnn",
        stage_name="smoke",
        mmseg_root=mmseg_root,
        dataset_root=tmp_path / "cityscapes",
        split_manifest=manifest,
        work_dir=tmp_path / "weighted-valid",
        loss="median_frequency",
        audit_report=weights,
        resume=False,
    )
    assert weighted.model.decode_head.loss_decode.class_weight == [1.0] * 19


def test_amp_optim_wrapper_shape_is_correct_for_every_precision(tmp_path: Path) -> None:
    """Config-level check for the AmpOptimWrapper branch (`build_training_config`,
    precision in {"fp16", "bf16"}).

    This branch is only reachable at *training* time via `resolve_auto_precision`,
    which returns "fp32" whenever `torch.cuda.is_available()` is False --
    meaning it is a dead branch in every CPU rehearsal run (this dev machine,
    and CI). That is exactly how the first real bug found this session (a
    hardcoded AMP dtype) reached a real Colab L4 GPU before being caught: no
    local run ever actually built or exercised an Amp config. Real bf16/fp16
    numerical behavior can only be observed on real CUDA hardware -- this
    test cannot substitute for that -- but the *shape* of the config
    `build_training_config` produces (which wrapper type, which dtype key,
    whether loss_scale is present) needs zero GPU and should never regress
    silently again.
    """
    pytest.importorskip("mmengine", reason="MMSeg config resolution is an optional integration")
    protocol = load_rescue_config(Path("configs/rescue/semantic_first.yaml"))
    sample = _sample(0)
    record = {
        "sample_id": sample.sample_id,
        "group_id": sample.group_id,
        "image": sample.image,
        "mask": sample.mask,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"roles": {"train_fit": [record], "train_select": [record]}}),
        encoding="utf-8",
    )
    mmseg_root = tmp_path / "mmseg"
    model = protocol.models[0]
    upstream = mmseg_root / model.upstream_config
    upstream.parent.mkdir(parents=True, exist_ok=True)
    upstream.write_text(
        "model = dict(type='EncoderDecoder', "
        "data_preprocessor=dict(type='SegDataPreProcessor', size=(1024, 1024)), "
        "decode_head=dict(type='FakeHead', num_classes=150, "
        "loss_decode=dict(type='CrossEntropyLoss')))\n"
        "default_hooks = dict(checkpoint=dict(type='CheckpointHook'))\n"
        "optim_wrapper = dict(type='OptimWrapper', "
        "optimizer=dict(type='AdamW', lr=6e-05, weight_decay=0.01))\n",
        encoding="utf-8",
    )

    def _build(precision: str, work_dir_name: str) -> Any:
        return build_training_config(
            protocol,
            model_name=model.name,
            stage_name="smoke",
            mmseg_root=mmseg_root,
            dataset_root=tmp_path / "cityscapes",
            split_manifest=manifest,
            work_dir=tmp_path / "work" / work_dir_name,
            loss="ce",
            audit_report=None,
            resume=False,
            precision=precision,
        )

    fp32 = _build("fp32", "fp32")
    assert fp32.optim_wrapper.type == "OptimWrapper"
    assert "dtype" not in fp32.optim_wrapper
    assert "loss_scale" not in fp32.optim_wrapper

    bf16 = _build("bf16", "bf16")
    assert bf16.optim_wrapper.type == "AmpOptimWrapper"
    assert bf16.optim_wrapper.dtype == "bfloat16"
    assert "loss_scale" not in bf16.optim_wrapper

    fp16 = _build("fp16", "fp16")
    assert fp16.optim_wrapper.type == "AmpOptimWrapper"
    assert fp16.optim_wrapper.dtype == "float16"
    assert fp16.optim_wrapper.loss_scale == "dynamic"

    with pytest.raises(ValueError, match="precision must be fp32, fp16, or bf16"):
        _build("auto", "invalid")


def test_resolved_config_survives_dump_and_reload_round_trip(tmp_path: Path) -> None:
    """`train_model()` writes `resolved.py` via `cfg.dump()` right after building the
    config, then `evaluate_model()` (used by the screening/final evidence phases) reloads
    it via `mmengine.Config.fromfile()`, which `eval()`s the dumped Python source. A real
    L4 run reached this exact reload path for the first time after all five models
    completed a full 6000-step screening run and crashed with
    `NameError: name 'inf' is not defined` -- `clip_grad.max_norm` was `float("inf")`,
    which mmengine's dumper serializes as the bare token `inf`, not a valid Python literal
    without `float(...)`/`math.inf` in scope. `error_if_nonfinite=True` is the real safety
    net (raises on NaN/Inf gradients); `max_norm` only needs to be large enough to never
    bind for any real gradient norm, so a large finite sentinel round-trips safely.
    """
    mmengine = pytest.importorskip(
        "mmengine", reason="MMSeg config resolution is an optional integration"
    )
    protocol = load_rescue_config(Path("configs/rescue/semantic_first.yaml"))
    sample = _sample(0)
    record = {
        "sample_id": sample.sample_id,
        "group_id": sample.group_id,
        "image": sample.image,
        "mask": sample.mask,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"roles": {"train_fit": [record], "train_select": [record]}}),
        encoding="utf-8",
    )
    mmseg_root = tmp_path / "mmseg"
    model = protocol.models[0]
    upstream = mmseg_root / model.upstream_config
    upstream.parent.mkdir(parents=True, exist_ok=True)
    upstream.write_text(
        "model = dict(type='EncoderDecoder', "
        "data_preprocessor=dict(type='SegDataPreProcessor', size=(1024, 1024)), "
        "decode_head=dict(type='FakeHead', num_classes=150, "
        "loss_decode=dict(type='CrossEntropyLoss')))\n"
        "default_hooks = dict(checkpoint=dict(type='CheckpointHook'))\n"
        "optim_wrapper = dict(type='OptimWrapper', "
        "optimizer=dict(type='AdamW', lr=6e-05, weight_decay=0.01))\n",
        encoding="utf-8",
    )
    cfg = build_training_config(
        protocol,
        model_name=model.name,
        stage_name="smoke",
        mmseg_root=mmseg_root,
        dataset_root=tmp_path / "cityscapes",
        split_manifest=manifest,
        work_dir=tmp_path / "work",
        loss="ce",
        audit_report=None,
        resume=False,
    )
    assert cfg.optim_wrapper.clip_grad.max_norm == pytest.approx(1e9)
    assert cfg.optim_wrapper.clip_grad.error_if_nonfinite is True

    resolved_path = tmp_path / "resolved.py"
    cfg.dump(str(resolved_path))
    dumped_source = resolved_path.read_text(encoding="utf-8")
    assert "inf" not in dumped_source.replace("clip_grad", "").replace("finite", "")

    reloaded = mmengine.Config.fromfile(str(resolved_path))
    assert reloaded.optim_wrapper.clip_grad.max_norm == pytest.approx(1e9)


def test_acdc_uses_the_original_condition_split_sequence_layout(tmp_path: Path) -> None:
    protocol = load_rescue_config(Path("configs/rescue/semantic_first.yaml"))
    (tmp_path / "rgb_anon/fog/val/sequence").mkdir(parents=True)
    (tmp_path / "gt/fog/val/sequence").mkdir(parents=True)
    dataset = _acdc_dataset(tmp_path, protocol, condition="fog")
    assert dataset["data_prefix"] == {
        "img_path": "rgb_anon/fog/val",
        "seg_map_path": "gt/fog/val",
    }
    with pytest.raises(FileNotFoundError, match="rgb_anon/night/val"):
        _acdc_dataset(tmp_path, protocol, condition="night")


class _FakeCuda:
    def __init__(self, *, available: bool, bf16_supported: bool) -> None:
        self._available = available
        self._bf16_supported = bf16_supported

    def is_available(self) -> bool:
        return self._available

    def is_bf16_supported(self) -> bool:
        return self._bf16_supported


class _FakeTorch:
    def __init__(self, *, available: bool, bf16_supported: bool) -> None:
        self.cuda = _FakeCuda(available=available, bf16_supported=bf16_supported)


def test_resolve_auto_precision_prefers_bf16_over_fp16_on_capable_hardware() -> None:
    # A real L4 (or any Ampere/Ada-class GPU) supports bf16; a stack-probe or
    # training call that hardcodes fp16 here would test a precision the real
    # "auto" policy never actually selects on this hardware.
    assert (
        resolve_auto_precision("auto", torch=_FakeTorch(available=True, bf16_supported=True))
        == "bf16"
    )


def test_resolve_auto_precision_falls_back_to_fp16_without_bf16_support() -> None:
    assert (
        resolve_auto_precision("auto", torch=_FakeTorch(available=True, bf16_supported=False))
        == "fp16"
    )


def test_resolve_auto_precision_uses_fp32_without_cuda() -> None:
    assert (
        resolve_auto_precision("auto", torch=_FakeTorch(available=False, bf16_supported=False))
        == "fp32"
    )


def test_resolve_auto_precision_passes_through_explicit_choices() -> None:
    torch_module = _FakeTorch(available=True, bf16_supported=True)
    assert resolve_auto_precision("fp32", torch=torch_module) == "fp32"
    assert resolve_auto_precision("fp16", torch=torch_module) == "fp16"
    assert resolve_auto_precision("bf16", torch=torch_module) == "bf16"


def test_visualization_and_preprocessing_contracts() -> None:
    logits = np.zeros((19, 2, 3), dtype=np.float32)
    confidence, entropy = confidence_entropy(logits)
    assert confidence == pytest.approx(np.full((2, 3), 1 / 19))
    assert entropy == pytest.approx(np.ones((2, 3)))
    image = Image.new("RGB", (6, 4), color=(10, 20, 30))
    tensor = preprocess_image(image, (2, 3))
    assert tensor.shape == (1, 3, 2, 3)
    mask = resize_mask(np.zeros((2, 3), dtype=np.uint8), image.size)
    assert overlay_mask(image, mask, 0.5).size == image.size
    raw = InferenceResult(
        mask=np.zeros((2, 3), dtype=np.uint8),
        confidence=np.ones((2, 3), dtype=np.float32),
        entropy=np.zeros((2, 3), dtype=np.float32),
        latency_ms=1.0,
        backend="fixture",
        metadata={},
        logits=np.zeros((19, 2, 3), dtype=np.float32),
    )
    calibrated = calibrate_inference_result(raw, 2.0)
    assert calibrated.metadata["calibrated"] is True
    assert calibrated.confidence == pytest.approx(np.full((2, 3), 1 / 19))


def test_demo_discovers_only_complete_model_records(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"graph")
    run = tmp_path / "pilot"
    run.mkdir()
    (run / "best.pth").write_bytes(b"checkpoint")
    assert len(discover_demo_models(tmp_path)) == 1
    (run / "resolved.py").write_text("model = {}\n", encoding="utf-8")
    assert len(discover_demo_models(tmp_path)) == 2


def test_top_two_selection_applies_accuracy_and_edge_rules() -> None:
    candidates = [
        {
            "model": "segformer_b0",
            "mIoU": 0.70,
            "onnx_validated": True,
            "onnx_median_latency_ms": 30,
            "onnx_bytes": 10_000,
        },
        {
            "model": "fast_scnn",
            "mIoU": 0.68,
            "onnx_validated": True,
            "onnx_median_latency_ms": 10,
            "onnx_bytes": 5_000,
        },
        {
            "model": "pidnet_s",
            "mIoU": 0.66,
            "onnx_validated": True,
            "onnx_median_latency_ms": 8,
            "onnx_bytes": 8_000,
        },
    ]
    result = select_top_two(candidates)
    assert result["scientific_candidate"] == "segformer_b0"
    assert result["edge_candidate"] == "fast_scnn"


def test_reporting_uses_only_present_evidence(tmp_path: Path) -> None:
    evaluations = tmp_path / "evaluations"
    exports = tmp_path / "exports"
    evaluations.mkdir()
    exports.mkdir()
    for model, miou, latency in (("segformer_b0", 0.7, 30.0), ("fast_scnn", 0.68, 10.0)):
        model_root = evaluations / model
        model_root.mkdir()
        for dataset in ("cityscapes", "bdd100k", "idd20k"):
            dataset_root = model_root / dataset
            dataset_root.mkdir()
            (dataset_root / "evaluation.json").write_text(
                json.dumps(
                    {
                        "model": model,
                        "dataset": dataset,
                        "role": "train_select",
                        "condition": None,
                        "metrics": {"mIoU": miou},
                        "rare_class_mIoU": miou - 0.1,
                        "reliability": None,
                    }
                ),
                encoding="utf-8",
            )
        (exports / f"{model}.validation.json").write_text(
            json.dumps(
                {
                    "shape_equal": True,
                    "allclose_atol_1e_4_rtol_1e_4": True,
                    "prediction_equivalent": True,
                    "argmax_agreement_ratio": 1.0,
                    "disagreeing_pixel_count": 0,
                    "mean_absolute_difference": 1.0e-6,
                    "onnx_bytes": 1000,
                    "onnxruntime_cpu": {"median_latency_ms": latency},
                }
            ),
            encoding="utf-8",
        )
    output = tmp_path / "report"
    result = build_evidence_report(evaluations, exports, output)
    assert result["selection_generated"] is True
    assert (output / "metrics_table.csv").is_file()
    assert json.loads((output / "top_two.json").read_text())["edge_candidate"] == "fast_scnn"


def test_semantic_first_master_notebook_is_valid_and_output_free() -> None:
    payload = json.loads(Path("notebooks/EdgeGuard_Master_Colab.ipynb").read_text())
    assert payload["nbformat"] == 4
    assert all(not cell.get("outputs") for cell in payload["cells"])
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"colab-cell-{index}", "exec")
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    pins = re.findall(r'EXPECTED_PROJECT_COMMIT = "([0-9a-f]{40})"', source)
    assert len(pins) == 1
    assert 'BRANCH = "stabilize/colab-v2"' in source
    assert 'CAMPAIGN_ID = "semantic-cs-idd-v3"' in source
    assert "scripts/run_colab_master.py" in source
    assert '"--execution-mode", "production"' in source
    assert "EdgeGuard_Jetson_Release.zip" in source
    assert "LOCAL_TEST_MODE" in source
    assert "scientific_status" in source and '"not_run"' in source
    assert "pip install -e" not in source


def test_synthetic_stress_fallback_preserves_labels_and_claim_boundary(tmp_path: Path) -> None:
    root = tmp_path / "cityscapes"
    identifier = "city_000001_000019"
    image_path = root / f"leftImg8bit/val/city/{identifier}_leftImg8bit.png"
    mask_path = root / f"gtFine/val/city/{identifier}_gtFine_labelTrainIds.png"
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    Image.new("RGB", (19, 2), color=(40, 50, 60)).save(image_path)
    Image.fromarray(np.tile(np.arange(19, dtype=np.uint8), (2, 1)), mode="L").save(mask_path)
    output = tmp_path / "stress"
    result = build_stress_dataset(root, output, condition="fog", severity=0.6)
    assert result["external_ood_evidence"] is False
    assert result["scientific_label"] == "synthetic robustness stress test"
    assert (output / mask_path.relative_to(root)).read_bytes() == mask_path.read_bytes()


def test_evaluation_weight_loading_rejects_a_model_that_kept_random_initialization(
    tmp_path: Path,
) -> None:
    """`evaluate_model()` builds its runner with `Runner.from_cfg()` and then drives
    `runner.model` directly so it can keep raw logits for calibration. mmengine only
    consumes `cfg.load_from` inside `Runner.train()`/`val()`/`test()`, none of which that
    path calls, so for the whole 2026-08-14 screening campaign every evaluation record
    measured PyTorch's default random initialisation: `pidnet_s` scored 35.48 mIoU in its
    own in-training validation loop and 1.93 mIoU in the evaluation record built from the
    identical `iter_2500.pth`. `load_evaluation_weights()` performs the load and then
    proves every float tensor landed, because mmengine's loader is non-strict and a silent
    key mismatch would reproduce exactly the same untrained-network measurement.
    """
    torch = pytest.importorskip("torch", reason="weight loading is a torch integration")
    pytest.importorskip("mmengine", reason="checkpoint loading is an mmengine integration")
    from edgeguard.rescue.mmseg_runtime import load_evaluation_weights

    torch.manual_seed(20260814)
    trained = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4))
    with torch.no_grad():
        for parameter in trained.parameters():
            parameter.add_(1.5)
    checkpoint = tmp_path / "iter_2500.pth"
    torch.save({"meta": {"iter": 2500}, "state_dict": trained.state_dict()}, checkpoint)

    torch.manual_seed(1)
    fresh = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4))
    weight = fresh[0].weight
    assert not torch.equal(weight, trained[0].weight)

    verified = load_evaluation_weights(fresh, checkpoint)
    assert verified > 0
    for name, tensor in trained.state_dict().items():
        if tensor.is_floating_point():
            assert torch.equal(fresh.state_dict()[name], tensor), name

    mismatched = tmp_path / "other-architecture.pth"
    torch.save({"state_dict": {"backbone.stem.weight": torch.ones(2, 2)}}, mismatched)
    with pytest.raises(RuntimeError, match="did not land in the model"):
        load_evaluation_weights(fresh, mismatched)


def test_candidate_table_records_why_a_trained_model_was_dropped(tmp_path: Path) -> None:
    """A model whose ONNX export disagrees with PyTorch is removed from the candidate
    table, and until now that removal left no trace: the campaign ran screening to
    completion and then died inside HPO with "requires two interpretable screening
    candidates" and nothing pointing at the export gate. The table now carries the reason,
    and `select_hpo_models()` repeats it in the failure.
    """
    evaluations = tmp_path / "evaluations"
    exports = tmp_path / "exports"
    evaluations.mkdir()
    exports.mkdir()
    for model, equivalent in (("pidnet_s", False), ("segformer_b0", False)):
        model_root = evaluations / model
        model_root.mkdir()
        for dataset in ("cityscapes", "idd20k"):
            dataset_root = model_root / dataset
            dataset_root.mkdir()
            (dataset_root / "evaluation.json").write_text(
                json.dumps(
                    {
                        "model": model,
                        "dataset": dataset,
                        "role": "train_select",
                        "condition": None,
                        "metrics": {"mIoU": 0.35},
                        "rare_class_mIoU": 0.2,
                        "reliability": None,
                    }
                ),
                encoding="utf-8",
            )
        (exports / f"{model}.validation.json").write_text(
            json.dumps(
                {
                    "shape_equal": True,
                    "allclose_atol_1e_4_rtol_1e_4": equivalent,
                    "prediction_equivalent": equivalent,
                    "argmax_agreement_ratio": 0.97,
                    "disagreeing_pixel_count": 61,
                    "max_absolute_difference": 0.0053,
                    "mean_absolute_difference": 0.0008,
                    "parity_device": "cpu",
                    "onnx_bytes": 1000,
                    "onnxruntime_cpu": {"median_latency_ms": 73.0},
                }
            ),
            encoding="utf-8",
        )
    output = tmp_path / "report"
    result = build_evidence_report(evaluations, exports, output)
    assert result["candidate_count"] == 0

    table = output / "candidate_table.json"
    payload = json.loads(table.read_text(encoding="utf-8"))
    assert [entry["model"] for entry in payload["rejected"]] == ["pidnet_s", "segformer_b0"]
    assert payload["rejected"][0]["max_absolute_difference"] == 0.0053

    with pytest.raises(ValueError, match=r"pidnet_s \(ONNX export does not reproduce"):
        select_hpo_models(table, expected_domains=("cityscapes", "idd20k"))


def test_skipping_frame_uncertainty_leaves_the_confusion_matrix_identical() -> None:
    """`uncertainty_maps` casts each image's 19x512x1024 logit volume to float64 and makes
    a dozen passes over it, which measured ~0.42 s/image on the real L4 screening run --
    roughly 85% of evaluation wall clock -- to produce `frame_uncertainty.json`, a file no
    reporting, selection, HPO or thesis-bundle code reads. Screening, HPO, selection and
    ablation therefore skip it. Those passes decide which model wins, so the metrics they
    compute must not move: the confusion matrix is built from `pred_sem_seg`/`gt_sem_seg`
    and the logit tensor is consulted only for its shape.
    """
    torch = pytest.importorskip("torch", reason="evidence collection is a torch integration")
    from edgeguard.rescue.mmseg_runtime import _collect_reporting_evidence

    class _Field:
        def __init__(self, data: Any) -> None:
            self.data = data

    class _Output:
        def __init__(self, index: int) -> None:
            torch.manual_seed(index)
            logits = torch.randn(19, 6, 10)
            self.seg_logits = _Field(logits)
            self.gt_sem_seg = _Field(torch.randint(0, 19, (1, 6, 10)))
            self.pred_sem_seg = _Field(logits.argmax(dim=0)[None])
            self.metainfo = {"img_path": f"/data/frame_{index:03d}.png"}

    class _Model:
        def eval(self) -> Any:
            return self

        def test_step(self, batch: Any) -> Any:
            return batch

    class _Runner:
        def __init__(self) -> None:
            self.model = _Model()
            self.test_dataloader = [[_Output(0), _Output(1)], [_Output(2)]]

    _, _, with_frames, frames = _collect_reporting_evidence(
        _Runner(), max_pixels=0, collect_frame_uncertainty=True
    )
    _, _, without_frames, skipped = _collect_reporting_evidence(
        _Runner(), max_pixels=0, collect_frame_uncertainty=False
    )

    assert with_frames == without_frames
    assert without_frames["mean_iou"] > 0.0
    assert frames is not None and len(frames) == 3
    assert frames[0]["sample_id"] == "frame_000"
    # `None` means "not collected"; an empty list would claim "collected, and there were
    # none", which is a different and false statement.
    assert skipped is None


def test_export_gate_measures_the_class_map_not_a_float_tolerance() -> None:
    """The deployment gate exists to catch a broken export -- wrong graph, wrong weights,
    wrong preprocessing -- before a model reaches HPO, selection or a release. It used to
    be `np.allclose(atol=1e-4, rtol=1e-4)` on raw logits, which is a proxy, and measuring
    the real exports on CPU on 2026-08-14 showed it is the wrong one: PIDNet-S produced a
    3.231e-03 worst-case logit delta against a 3.531e-05 mean, all of it at interpolation
    boundaries from its `align_corners=True` head, while assigning an identical class to
    every one of its 2,048 output pixels. That proxy would have silently dropped the
    campaign's strongest model (35.48 screening mIoU) out of the candidate table. The gate
    now asks what the Jetson runtime actually consumes -- the per-pixel argmax -- and
    keeps a mean-absolute ceiling so a systematically wrong graph still cannot pass.
    """
    from edgeguard.export.equivalence import semantic_onnx_export_accepted

    measured_pidnet = {
        "shape_equal": True,
        "allclose_atol_1e_4_rtol_1e_4": False,
        "prediction_equivalent": True,
        "argmax_agreement_ratio": 1.0,
        "disagreeing_pixel_count": 0,
        "max_absolute_difference": 3.230571746826172e-03,
        "mean_absolute_difference": 3.5311902172751565e-05,
        "parity_device": "cpu",
    }
    assert semantic_onnx_export_accepted(measured_pidnet) is True

    # A graph that disagrees about even one pixel's class is not deployable evidence.
    assert (
        semantic_onnx_export_accepted({**measured_pidnet, "prediction_equivalent": False}) is False
    )
    assert semantic_onnx_export_accepted({**measured_pidnet, "shape_equal": False}) is False
    # A systematically wrong graph moves the whole tensor, not a few boundary pixels, so
    # the mean ceiling still rejects it even if the argmax happened to survive.
    assert (
        semantic_onnx_export_accepted({**measured_pidnet, "mean_absolute_difference": 0.5}) is False
    )
    # A record from before the gate existed carries no verdict and must be rebuilt.
    assert semantic_onnx_export_accepted({"shape_equal": True}) is False


def test_a_class_map_equivalent_export_reaches_the_candidate_table(tmp_path: Path) -> None:
    """End-to-end companion to the gate unit test: the measured PIDNet-S record must
    produce a candidate rather than a rejection, because that is the failure mode that
    would have cost the campaign its best model.
    """
    evaluations = tmp_path / "evaluations"
    exports = tmp_path / "exports"
    evaluations.mkdir()
    exports.mkdir()
    model_root = evaluations / "pidnet_s"
    model_root.mkdir()
    for dataset in ("cityscapes", "idd20k"):
        dataset_root = model_root / dataset
        dataset_root.mkdir()
        (dataset_root / "evaluation.json").write_text(
            json.dumps(
                {
                    "model": "pidnet_s",
                    "dataset": dataset,
                    "role": "train_select",
                    "condition": None,
                    "metrics": {"mIoU": 0.3548},
                    "rare_class_mIoU": 0.11,
                    "reliability": None,
                }
            ),
            encoding="utf-8",
        )
    (exports / "pidnet_s.validation.json").write_text(
        json.dumps(
            {
                "shape_equal": True,
                "allclose_atol_1e_4_rtol_1e_4": False,
                "prediction_equivalent": True,
                "argmax_agreement_ratio": 1.0,
                "disagreeing_pixel_count": 0,
                "max_absolute_difference": 3.230571746826172e-03,
                "mean_absolute_difference": 3.5311902172751565e-05,
                "parity_device": "cpu",
                "onnx_bytes": 30568865,
                "onnxruntime_cpu": {"median_latency_ms": 73.47},
            }
        ),
        encoding="utf-8",
    )
    result = build_evidence_report(evaluations, exports, tmp_path / "report")
    assert result["candidate_count"] == 1
    payload = json.loads((tmp_path / "report" / "candidate_table.json").read_text())
    assert payload["rejected"] == []
    assert payload["candidates"][0]["model"] == "pidnet_s"
