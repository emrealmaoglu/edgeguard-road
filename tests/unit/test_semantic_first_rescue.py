from __future__ import annotations

import json
from pathlib import Path

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
from edgeguard.rescue.inference import discover_demo_models, preprocess_image
from edgeguard.rescue.mmseg_runtime import (
    _acdc_dataset,
    build_training_config,
    materialize_role_file,
    validate_scientific_split,
)
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
    assert config.stages["pilot"].max_steps == 2_000
    assert config.stages["screening"].max_steps == 6_000
    assert config.hpo.trials_per_model == 12
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


def test_semantic_first_notebook_is_valid_and_output_free() -> None:
    preflight = json.loads(Path("notebooks/EdgeGuard_Data_Preflight_Colab.ipynb").read_text())
    payload = json.loads(Path("notebooks/EdgeGuard_Road_Colab.ipynb").read_text())
    for notebook in (preflight, payload):
        assert notebook["nbformat"] == 4
        assert all(not cell.get("outputs") for cell in notebook["cells"])
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"colab-cell-{index}", "exec")
    preflight_source = "\n".join("".join(cell.get("source", [])) for cell in preflight["cells"])
    assert "scripts/prepare_colab_data.py" in preflight_source
    assert "scripts/prepare_dataset.py" in preflight_source
    assert "VERIFY_ARCHIVE_HASHES = True" in preflight_source
    assert "RUN_ARCHIVE_PREPARATION = False" in preflight_source
    assert "CREATE_BUNDLES = True" in preflight_source
    assert 'BDD_SOURCE_PROFILE = "kaggle_mirror"' in preflight_source
    assert 'SCIENTIFIC_SOURCE_DATASETS = ["cityscapes", "idd20k"]' in preflight_source
    assert "ColabFailureReporter" in preflight_source
    assert "run_logged_command" in preflight_source
    assert "cwd=PROJECT_ROOT" in preflight_source
    assert "persist_bootstrap_failure" in preflight_source
    assert "live_preparation_progress" in preflight_source
    assert "EDGEGUARD PROGRESS dataset=" in preflight_source
    assert "reuse_or_copy_archive" in preflight_source
    assert "Yerel cache SHA-256 doğrulanıyor" in preflight_source
    assert "shutil.rmtree(CACHE_ROOT)" not in preflight_source
    assert (
        'EXPECTED_PROJECT_COMMIT = "3134d3f1e6d3bf23ede14f7a29b0adbeb51e0e89"' in preflight_source
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "scripts/audit_dataset.py" in source
    assert "scripts/train.py" in source
    assert "scripts/evaluate.py" in source
    assert "scripts/predict.py" in source
    assert "scripts/export_onnx.py" in source
    assert "calibrate-shift" in source
    assert "evaluate-shift" in source
    assert "--emit-regions" in source
    assert "scripts/jetson/benchmark.py" in source
    assert "--local-root" in source
    assert "sync_work_snapshot" in source
    assert "RUN_TRAINING = False" in source
    assert 'CAMPAIGN_ID = "semantic-first-cs-idd-v1"' in source
    assert "bdd100k.frozen.json" not in source
    assert "runtime-compatibility-cascade" in source
    assert "failure-report.zip" in source
    assert "run_logged_command" in source
    assert "cwd=PROJECT_ROOT" in source
    assert 'EXPECTED_PROJECT_COMMIT = "3134d3f1e6d3bf23ede14f7a29b0adbeb51e0e89"' in source


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
