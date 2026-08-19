"""Tests for the single bounded pre-Colab readiness gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import edgeguard.deployment.precolab as precolab
from edgeguard.deployment.precolab import CANONICAL_NOTEBOOKS, check_precolab_readiness


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _repository(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repository = tmp_path / "repository"
    notebook_root = repository / "notebooks"
    notebook_root.mkdir(parents=True)
    for name in CANONICAL_NOTEBOOKS:
        _write_json(
            notebook_root / name,
            {"cells": [{"cell_type": "markdown", "source": ["# Canonical\n"]}]},
        )
    _write_json(
        repository / "reports/local-final-audit/project_gap_matrix.json",
        {
            "records": [
                {"capability_id": "EG-CAP-01", "after": {"maturity": "contract_only"}},
                {
                    "capability_id": "EG-CAP-03",
                    "after": {"maturity": "requires_real_data"},
                },
                {
                    "capability_id": "EG-CAP-24",
                    "after": {"maturity": "local_end_to_end_validated"},
                },
                {
                    "capability_id": "EG-CAP-26",
                    "after": {"maturity": "requires_jetson"},
                },
            ]
        },
    )
    closure = tmp_path / "closure.json"
    _write_json(
        closure,
        {
            "git_commit": "a" * 40,
            "all_completed_or_reused": True,
            "stages": [{"stage": "reporting", "status": "completed"}],
        },
    )
    equivalence = tmp_path / "equivalence.json"
    _write_json(
        equivalence,
        {
            "models": [
                {"model_family": "pidnet_s", "classification": "bounded_documented_drift"},
                {
                    "model_family": "rt_detr_r18",
                    "classification": "bounded_documented_drift",
                },
            ]
        },
    )
    deployment = tmp_path / "deployment.json"
    _write_json(
        deployment,
        {"status": "verified", "fixture_inference": {"status": "passed"}},
    )
    return repository, closure, equivalence, deployment


def test_precolab_readiness_accepts_only_external_or_human_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, closure, equivalence, deployment = _repository(tmp_path)

    def fake_git(_repository: Path, *args: str) -> str:
        return "a" * 40 if args == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr(precolab, "_git", fake_git)
    report = check_precolab_readiness(
        repository,
        expected_commit="a" * 40,
        closure_summary_path=closure,
        equivalence_report_path=equivalence,
        deployment_validation_path=deployment,
        minimum_free_gib=0.001,
    )

    assert report["status"] == "passed"
    assert report["locally_testable_capability_remaining"] is False
    assert {item["gate"] for item in report["remaining_gates"]} == {
        "human_scientific_decision",
        "real_data",
        "jetson",
    }


def test_precolab_readiness_rejects_an_extra_active_notebook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, closure, equivalence, deployment = _repository(tmp_path)
    _write_json(
        repository / "notebooks/obsolete-active.ipynb",
        {"cells": [{"cell_type": "markdown", "source": ["# Old workflow\n"]}]},
    )
    monkeypatch.setattr(
        precolab,
        "_git",
        lambda _repository, *args: "a" * 40 if args == ("rev-parse", "HEAD") else "",
    )
    with pytest.raises(ValueError, match="canonical notebook set"):
        check_precolab_readiness(
            repository,
            expected_commit="a" * 40,
            closure_summary_path=closure,
            equivalence_report_path=equivalence,
            deployment_validation_path=deployment,
            minimum_free_gib=0.001,
        )


def test_package_fixture_selection_falls_back_to_available_validated_model(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    closure = tmp_path / "closure"
    for model_family in ("fast_scnn", "bisenetv2"):
        (repository / "configs/training/segmentation").mkdir(parents=True, exist_ok=True)
        (repository / f"configs/training/segmentation/{model_family}.yaml").write_text(
            "model: fixture\n", encoding="utf-8"
        )
        (closure / f"semantic/{model_family}").mkdir(parents=True, exist_ok=True)
        (closure / f"semantic/{model_family}/result.json").write_text("{}", encoding="utf-8")
    (closure / "semantic_onnx").mkdir(parents=True)
    (closure / "semantic_onnx/bisenetv2.onnx").write_bytes(b"fixture")
    semantic_onnx = {
        "results": [
            {
                "model_family": "fast_scnn",
                "allclose_atol_1e_4_rtol_1e_4": True,
            },
            {
                "model_family": "bisenetv2",
                "allclose_atol_1e_4_rtol_1e_4": True,
            },
        ]
    }

    selected, onnx_path, _result_path, numerical = precolab._select_package_fixture(
        repository, closure, semantic_onnx
    )

    assert selected == "bisenetv2"
    assert onnx_path.name == "bisenetv2.onnx"
    assert numerical["model_family"] == "bisenetv2"
