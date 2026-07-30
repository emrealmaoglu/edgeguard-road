from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from edgeguard.rescue.config import load_rescue_config
from edgeguard.rescue.external import _encode_submission_mask, record_external_server_result
from edgeguard.rescue.hpo_runtime import hpo_search_space, select_hpo_models
from edgeguard.rescue.ledger import append_run_ledger
from edgeguard.rescue.mmseg_runtime import _verified_pretrained_checkpoint
from edgeguard.rescue.multidomain import (
    _bounded_mean_one_weights,
    audit_evaluation_dataset,
    audit_training_dataset,
    domain_mixture_probabilities,
    freeze_candidate_manifest,
    load_semantic_ontology,
    map_source_mask,
    power_domain_indices,
    uniform_domain_indices,
    verify_sealed_release,
    write_multidomain_statistics,
)
from edgeguard.rescue.reliability import (
    fit_global_temperature_from_evidence,
    save_calibration_evidence,
)
from edgeguard.serialization import canonical_json, sha256_file

ONTOLOGY = Path("configs/dataset/semantic_ontology_v2.yaml")


def test_semantic_ontology_maps_only_exact_idd_classes() -> None:
    ontology = load_semantic_ontology(ONTOLOGY)
    source = np.asarray([[0, 3, 6, 11, 19, 23, 33, 255]], dtype=np.uint8)
    mapped = map_source_mask(source, "idd20k", ontology)
    assert mapped.tolist() == [[0, 1, 11, 255, 255, 255, 10, 255]]
    with pytest.raises(ValueError, match="unreviewed source IDs"):
        map_source_mask(np.asarray([[40]], dtype=np.uint8), "idd20k", ontology)


def test_uniform_domain_indices_do_not_follow_dataset_size() -> None:
    indices = uniform_domain_indices([2, 20, 200], total_size=30, seed=7)
    domains = [0 if value < 2 else 1 if value < 22 else 2 for value in indices]
    assert [domains.count(index) for index in range(3)] == [10, 10, 10]
    assert indices == uniform_domain_indices([2, 20, 200], total_size=30, seed=7)


def test_size_power_domain_ablation_has_explicit_extremes() -> None:
    lengths = [100, 400, 900]
    assert domain_mixture_probabilities(lengths, alpha=0.0) == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert domain_mixture_probabilities(lengths, alpha=1.0) == pytest.approx(
        [1 / 14, 4 / 14, 9 / 14]
    )
    indices = power_domain_indices(lengths, total_size=140, alpha=1.0, seed=17)
    domains = [0 if value < 100 else 1 if value < 500 else 2 for value in indices]
    assert [domains.count(index) for index in range(3)] == [10, 40, 90]
    assert indices == power_domain_indices(lengths, total_size=140, alpha=1.0, seed=17)
    with pytest.raises(ValueError, match="between 0 and 1"):
        domain_mixture_probabilities(lengths, alpha=1.1)


def _write_bdd_fixture(root: Path, *, split: str = "train", offset: int = 0) -> None:
    image_root = root / f"images/10k/{split}"
    mask_root = root / f"labels/sem_seg/masks/{split}"
    image_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    mask = np.tile(np.arange(19, dtype=np.uint8), (4, 1))
    for index in range(3):
        name = f"sequence{index}-frame"
        pixels = np.random.default_rng(1_000 + offset + index).integers(
            0, 256, size=(4, 19, 3), dtype=np.uint8
        )
        Image.fromarray(pixels, mode="RGB").save(image_root / f"{name}.jpg")
        Image.fromarray(mask, mode="L").save(mask_root / f"{name}.png")


def _write_idd_fixture(root: Path) -> None:
    image_root = root / "leftImg8bit/train/city"
    mask_root = root / "gtFine/train/city"
    image_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    source_ids = np.asarray(
        [0, 3, 29, 20, 21, 26, 25, 24, 32, 2, 33, 6, 8, 12, 13, 14, 17, 9, 10],
        dtype=np.uint8,
    )
    mask = np.tile(source_ids, (4, 1))
    for index in range(3):
        identifier = f"city_{index:06d}_000019"
        pixels = np.full((4, 19, 3), 70 + index * 20, dtype=np.uint8)
        pixels[:, :, (index + 1) % 3] += np.arange(19, dtype=np.uint8)
        Image.fromarray(pixels, mode="RGB").save(image_root / f"{identifier}_leftImg8bit.png")
        Image.fromarray(mask, mode="L").save(mask_root / f"{identifier}_gtFine_labelids.png")


def test_training_audits_freeze_and_pool_statistics(tmp_path: Path) -> None:
    bdd = tmp_path / "bdd"
    idd = tmp_path / "idd"
    _write_bdd_fixture(bdd)
    _write_idd_fixture(idd)
    audit_training_dataset(
        bdd,
        tmp_path / "audit-bdd",
        dataset_id="bdd100k",
        ontology_path=ONTOLOGY,
        seed=3,
        strict_count=False,
    )
    audit_training_dataset(
        idd,
        tmp_path / "audit-idd",
        dataset_id="idd20k",
        ontology_path=ONTOLOGY,
        seed=3,
        strict_count=False,
    )
    bdd_frozen = tmp_path / "bdd.frozen.json"
    idd_frozen = tmp_path / "idd.frozen.json"
    freeze_candidate_manifest(
        tmp_path / "audit-bdd/bdd100k_audit/dataset_manifest.candidate.json", bdd_frozen
    )
    freeze_candidate_manifest(
        tmp_path / "audit-idd/idd20k_audit/dataset_manifest.candidate.json", idd_frozen
    )
    assert json.loads(bdd_frozen.read_text())["scientific_eligible"] is False
    result = write_multidomain_statistics((bdd_frozen, idd_frozen), tmp_path / "statistics")
    weights = json.loads((tmp_path / "statistics/class_weights.json").read_text())["weights"]
    rare = json.loads((tmp_path / "statistics/rare_classes.json").read_text())
    assert result["duplicate_groups"] == 0
    assert len(rare["groups"]["rare"]) == 5
    assert min(weights) >= 0.5 and max(weights) <= 5.0
    assert np.mean(weights) == pytest.approx(1.0)
    assert result["figures_generated"] is True
    for relative in (
        "class_distribution.csv",
        "figures/class_distribution_by_domain.png",
        "figures/class_distribution_by_domain.pdf",
        "figures/pooled_imbalance_and_weights.png",
        "figures/source_split_sizes.pdf",
        "figures/source_domain_examples.png",
        "thesis_figures.json",
    ):
        assert (tmp_path / "statistics" / relative).is_file()
    report = json.loads((tmp_path / "statistics/thesis_figures.json").read_text())
    assert report["dataset_redistribution_authorized"] is False
    assert all(len(row["sha256"]) == 64 for row in report["files"])


def test_official_source_validation_is_separate_and_overlap_checked(tmp_path: Path) -> None:
    bdd = tmp_path / "bdd"
    _write_bdd_fixture(bdd)
    _write_bdd_fixture(bdd, split="val", offset=120)
    audit_training_dataset(
        bdd,
        tmp_path / "train-audit",
        dataset_id="bdd100k",
        ontology_path=ONTOLOGY,
        seed=3,
        strict_count=False,
    )
    frozen_train = tmp_path / "bdd-train.frozen.json"
    freeze_candidate_manifest(
        tmp_path / "train-audit/bdd100k_audit/dataset_manifest.candidate.json",
        frozen_train,
    )
    result = audit_training_dataset(
        bdd,
        tmp_path / "val-audit",
        dataset_id="bdd100k",
        ontology_path=ONTOLOGY,
        seed=3,
        strict_count=False,
        source_split="val",
        source_manifests=(frozen_train,),
    )
    candidate = json.loads(
        (tmp_path / "val-audit/bdd100k_val_audit/dataset_manifest.candidate.json").read_text()
    )
    assert result["source_split"] == "val"
    assert set(candidate["roles"]) == {"official_source_val"}
    assert result["exact_source_overlap_count"] == 0


def test_bounded_weights_reject_missing_classes() -> None:
    with pytest.raises(ValueError, match="all 19 classes"):
        _bounded_mean_one_weights(np.zeros(19, dtype=np.int64))


def test_sealed_external_manifest_requires_hash_bound_release(tmp_path: Path) -> None:
    root = tmp_path / "wilddash"
    root.mkdir()
    Image.new("RGB", (8, 4), color=(10, 20, 30)).save(root / "frame.jpg")
    pairs = tmp_path / "pairs.json"
    pairs.write_text(
        json.dumps(
            {
                "submission_encoding": "cityscapes_label_ids",
                "samples": [
                    {
                        "sample_id": "frame",
                        "group_id": "frame",
                        "image": "frame.jpg",
                        "submission_name": "frame.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    audit_evaluation_dataset(
        root,
        tmp_path / "audit",
        dataset_id="wilddash2",
        pairs_file=pairs,
        ontology_path=ONTOLOGY,
        source_url="https://www.wilddash.cc/",
        license_id="test-license-record",
        access_date="2026-07-28",
    )
    frozen = tmp_path / "wilddash.frozen.json"
    manifest = freeze_candidate_manifest(
        tmp_path / "audit/wilddash2_audit/dataset_manifest.candidate.json", frozen
    )
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fixture-model")
    with pytest.raises(PermissionError, match="sealed external"):
        verify_sealed_release(frozen, model, None)
    release = tmp_path / "release.json"
    release.write_text(
        canonical_json(
            {
                "record_type": "edgeguard_sealed_release",
                "manifest_sha256": manifest["manifest_sha256"],
                "checkpoint_sha256": sha256_file(model),
                "model_selection_frozen": True,
                "human_approved": True,
            }
        ),
        encoding="utf-8",
    )
    assert verify_sealed_release(frozen, model, release)["human_approved"] is True


def test_hpo_contract_and_candidate_selection(tmp_path: Path) -> None:
    protocol = load_rescue_config(Path("configs/rescue/semantic_first.yaml"))
    search = hpo_search_space(protocol)
    assert search["fixed"]["resolution"] == [512, 1024]
    assert search["fixed"]["loss"] == "ce"
    table = tmp_path / "candidates.json"
    table.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "model": model,
                        "domain_macro_mIoU": score,
                        "source_domain_mIoU": {
                            "cityscapes": score,
                            "bdd100k": score,
                            "idd20k": score,
                        },
                        "screening_valid": True,
                        "onnx_validated": True,
                    }
                    for model, score in (
                        ("fast_scnn", 0.61),
                        ("segformer_b0", 0.70),
                        ("pidnet_s", 0.66),
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    assert select_hpo_models(table) == ("segformer_b0", "pidnet_s")


def test_official_external_score_keeps_vendor_metric_name(tmp_path: Path) -> None:
    archive = tmp_path / "wilddash2_submission.zip"
    archive.write_bytes(b"submission")
    package = tmp_path / "package_manifest.json"
    package.write_text(
        json.dumps(
            {
                "record_type": "sealed_external_prediction_package",
                "dataset_id": "wilddash2",
                "dataset_manifest_sha256": "a" * 64,
                "model_sha256": "b" * 64,
                "archive": archive.name,
                "archive_sha256": sha256_file(archive),
            }
        ),
        encoding="utf-8",
    )
    server = tmp_path / "server.json"
    server.write_text(
        json.dumps(
            {
                "metric_name": "official_wilddash_score",
                "metric_value": 0.42,
                "submission_id": "submission-1",
                "submitted_at": "2026-07-28T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    result = record_external_server_result(package, server, tmp_path / "external.json")
    assert result["official_metric_name"] == "official_wilddash_score"
    assert result["not_relabelled_as_cityscapes19_mIoU"] is True


def test_wilddash_submission_uses_regular_cityscapes_label_ids() -> None:
    canonical = np.arange(19, dtype=np.uint8).reshape(1, 19)
    encoded = _encode_submission_mask(canonical, "cityscapes_label_ids")
    assert encoded.tolist() == [
        [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    ]


def test_pretrained_initialization_requires_approved_classification_provenance(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "backbone.pth"
    checkpoint.write_bytes(b"verified-backbone")
    manifest = tmp_path / "pretrained.json"
    payload = {
        "record_type": "edgeguard_pretrained_initialization",
        "model": "segformer_b0",
        "source_task": "image_classification",
        "human_approved": True,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_url": "https://example.invalid/model-card",
        "license_id": "fixture-license",
        "access_date": "2026-07-28",
    }
    manifest.write_text(canonical_json(payload), encoding="utf-8")
    assert _verified_pretrained_checkpoint(manifest, "segformer_b0") == checkpoint

    payload["source_task"] = "semantic_segmentation"
    manifest.write_text(canonical_json(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="classification task"):
        _verified_pretrained_checkpoint(manifest, "segformer_b0")


def test_pretrained_initialization_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "backbone.pth"
    checkpoint.write_bytes(b"verified-backbone")
    manifest = tmp_path / "pretrained.json"
    manifest.write_text(
        canonical_json(
            {
                "record_type": "edgeguard_pretrained_initialization",
                "model": "pidnet_s",
                "source_task": "image_classification",
                "human_approved": True,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": "0" * 64,
                "source_url": "https://example.invalid/model-card",
                "license_id": "fixture-license",
                "access_date": "2026-07-28",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="path/hash mismatch"):
        _verified_pretrained_checkpoint(manifest, "pidnet_s")


def test_run_ledger_is_append_only_and_hashes_each_result(tmp_path: Path) -> None:
    ledger = tmp_path / "run_ledger.jsonl"
    first = append_run_ledger(
        ledger,
        operation="fixture_one",
        result={"record_type": "fixture", "value": 1},
        repository=tmp_path,
    )
    second = append_run_ledger(
        ledger,
        operation="fixture_two",
        result={"record_type": "fixture", "value": 2},
        repository=tmp_path,
    )
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert rows == [first, second]
    assert rows[0]["result_sha256"] != rows[1]["result_sha256"]


def test_global_calibration_uses_equal_hash_bound_source_evidence(tmp_path: Path) -> None:
    logits = np.zeros((1, 19, 1, 30), dtype=np.float32)
    targets = np.arange(30, dtype=np.int64).reshape(1, 1, 30) % 19
    evidence = []
    for index, dataset in enumerate(("cityscapes", "bdd100k", "idd20k")):
        path = tmp_path / f"{dataset}.npz"
        save_calibration_evidence(
            path,
            logits=logits + index * 0.01,
            targets=targets,
            dataset_id=dataset,
            dataset_manifest_sha256=str(index + 1) * 64,
            checkpoint_sha256="a" * 64,
        )
        evidence.append(path)
    result = fit_global_temperature_from_evidence(evidence, tmp_path / "temperature.json", seed=7)
    assert result["equal_pixels_per_domain"] == 30
    assert set(result["dataset_manifest_sha256s"]) == {
        "cityscapes",
        "bdd100k",
        "idd20k",
    }
