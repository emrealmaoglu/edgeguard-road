"""One bounded pre-Colab package, notebook, and readiness gate."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

from edgeguard.deployment.package import build_deployment_package, verify_deployment_package
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload

CANONICAL_NOTEBOOKS = (
    "00_campaign_control.ipynb",
    "10_semantic_campaign.ipynb",
    "20_ood_calibration_risk.ipynb",
    "30_detection_temporal_fusion.ipynb",
    "40_export_and_reporting.ipynb",
)
DEPRECATED_MARKER = "DEPRECATED — NON-CANONICAL"
PACKAGE_FIXTURE_PREFERENCE = (
    "fast_scnn",
    "bisenetv2",
    "ddrnet_23_slim",
    "segformer_b0",
    "pidnet_s",
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON mapping: {path.name}")
    return payload


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def active_campaign_notebooks(repository: Path) -> tuple[str, ...]:
    """Return notebooks not explicitly marked deprecated in their first Markdown cell."""
    active: list[str] = []
    for path in sorted((repository / "notebooks" / "colab").glob("*.ipynb")):
        notebook = _json(path)
        first = next(
            (cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "markdown"),
            None,
        )
        source = "" if first is None else "".join(first.get("source", []))
        if DEPRECATED_MARKER not in source:
            active.append(path.name)
    return tuple(active)


def check_precolab_readiness(
    repository: Path,
    *,
    expected_commit: str,
    closure_summary_path: Path,
    equivalence_report_path: Path,
    deployment_validation_path: Path,
    minimum_free_gib: float = 5.0,
) -> dict[str, Any]:
    """Fail unless all local gates are complete and only external/human gates remain."""
    commit = _git(repository, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise ValueError("pre-Colab checkout does not match the expected exact commit")
    if _git(repository, "status", "--porcelain=v1"):
        raise ValueError("pre-Colab checkout must be clean")
    active = active_campaign_notebooks(repository)
    if active != CANONICAL_NOTEBOOKS:
        raise ValueError(f"canonical notebook set mismatch: active={list(active)}")

    closure = _json(closure_summary_path)
    if closure.get("git_commit") != commit or not closure.get("all_completed_or_reused"):
        raise ValueError("local closure evidence is incomplete or belongs to another commit")
    if any(item.get("status") not in {"completed", "reused"} for item in closure["stages"]):
        raise ValueError("local closure contains an unresolved stage")

    equivalence = _json(equivalence_report_path)
    models = {item.get("model_family"): item for item in equivalence.get("models", [])}
    if set(models) != {"pidnet_s", "rt_detr_r18"}:
        raise ValueError("ONNX classification records are incomplete")
    allowed = {
        "fixed_and_equivalent",
        "bounded_documented_drift",
        "requires_cuda_or_tensorrt_validation",
    }
    if any(item.get("classification") not in allowed for item in models.values()):
        raise ValueError("ONNX classification record is invalid")

    deployment = _json(deployment_validation_path)
    if (
        deployment.get("status") != "verified"
        or (deployment.get("fixture_inference") or {}).get("status") != "passed"
    ):
        raise ValueError("deployment package fixture is not verified")

    audit = _json(repository / "reports" / "local-final-audit" / "project_gap_matrix.json")
    remaining: list[dict[str, str]] = []
    for row in audit.get("records", []):
        after = row["after"]
        maturity = after["maturity"]
        if maturity in {"absent", "surrogate_validated"}:
            raise ValueError(
                f"locally testable capability remains unresolved: {row['capability_id']}"
            )
        if maturity == "requires_real_data":
            gate = "real_data"
        elif maturity == "requires_cuda":
            gate = "cuda"
        elif maturity == "requires_jetson":
            gate = "jetson"
        elif maturity == "contract_only":
            gate = "human_scientific_decision"
        else:
            continue
        remaining.append({"capability_id": row["capability_id"], "gate": gate})

    free_bytes = shutil.disk_usage(repository).free
    required_bytes = int(minimum_free_gib * 1024**3)
    if free_bytes < required_bytes:
        raise ValueError("insufficient local disk for the bounded pre-Colab handoff")
    return {
        "schema_version": "1.0",
        "record_type": "edgeguard_pre_colab_readiness",
        "status": "passed",
        "git_commit": commit,
        "git_dirty": False,
        "canonical_notebooks": list(active),
        "local_closure_stage_count": len(closure["stages"]),
        "onnx_classifications": {
            name: item["classification"] for name, item in sorted(models.items())
        },
        "deployment_package_fixture": "verified",
        "minimum_free_gib": minimum_free_gib,
        "observed_free_bytes": free_bytes,
        "remaining_gates": remaining,
        "locally_testable_capability_remaining": False,
        "scientific_evidence": False,
    }


def _stable_zip(destination: Path, members: dict[str, bytes]) -> dict[str, Any]:
    with ZipFile(destination, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in sorted(members.items()):
            info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return {
        "filename": destination.name,
        "sha256": sha256_file(destination),
        "byte_size": destination.stat().st_size,
    }


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (canonical_json(payload) + "\n").encode("utf-8")


def _select_package_fixture(
    repository: Path, closure_root: Path, semantic_onnx: dict[str, Any]
) -> tuple[str, Path, Path, dict[str, Any]]:
    """Select the first preferred, numerically validated artifact actually produced."""
    results = {
        item.get("model_family"): item
        for item in semantic_onnx.get("results", [])
        if item.get("allclose_atol_1e_4_rtol_1e_4") is True
    }
    for model_family in PACKAGE_FIXTURE_PREFERENCE:
        onnx_path = closure_root / "semantic_onnx" / f"{model_family}.onnx"
        semantic_result_path = closure_root / "semantic" / model_family / "result.json"
        model_config = repository / "configs" / "training" / "segmentation" / f"{model_family}.yaml"
        if (
            model_family in results
            and onnx_path.is_file()
            and semantic_result_path.is_file()
            and model_config.is_file()
        ):
            return model_family, onnx_path, semantic_result_path, results[model_family]
    raise ValueError("no numerically validated semantic ONNX artifact is available for packaging")


def build_precolab_evidence(
    repository: Path,
    *,
    closure_root: Path,
    equivalence_report_path: Path,
    output_root: Path,
    expected_commit: str,
) -> dict[str, Any]:
    """Build and verify one actual random-weight ONNX package plus review ZIPs."""
    output_root.mkdir(parents=True, exist_ok=False)
    semantic_onnx = _json(closure_root / "semantic_onnx" / "report.json")
    model_family, onnx_path, semantic_result_path, numerical = _select_package_fixture(
        repository, closure_root, semantic_onnx
    )
    semantic_result = _json(semantic_result_path)
    model_config = repository / "configs" / "training" / "segmentation" / f"{model_family}.yaml"
    common_config = repository / "configs" / "training" / "segmentation" / "common_cityscapes.yaml"
    config_sha256 = sha256_payload(
        {"common": sha256_file(common_config), "model": sha256_file(model_config)}
    )
    preprocessing = {
        "raw_image": {"layout": "HWC", "dtype": "uint8", "channel_order": "RGB"},
        "resize": {"target_hw": [128, 256], "image_mode": "bilinear"},
        "normalization": {
            "scale": "divide_by_255",
            "mean_rgb": [0.485, 0.456, 0.406],
            "std_rgb": [0.229, 0.224, 0.225],
        },
        "model_input": {
            "name": "model_input",
            "shape": [1, 3, 128, 256],
            "dtype": "float32",
            "layout": "NCHW",
        },
    }
    postprocessing = {
        "onnx_output": "native_logits",
        "native_logits": {"layout": "NCHW", "dtype": "float32", "class_count": 19},
        "alignment": {"mode": "bilinear", "target": "analysis_grid", "align_corners": False},
        "semantic_mask": "argmax(aligned_logits,axis=1)",
        "softmax_in_onnx_graph": False,
    }
    metadata = {
        "model_family": model_family,
        "config_sha256": config_sha256,
        "checkpoint_sha256": semantic_result["checkpoint_sha256"],
        "source_commit": expected_commit,
        "initialization": "project_owned_random_no_download",
        "framework_commit": semantic_result["framework_commit"],
    }
    calibration = {
        "status": "non_scientific_placeholder",
        "value": None,
        "required_human_gate": "fit_only_on_train_calibration",
    }
    thresholds = {
        "status": "non_scientific_placeholder",
        "value": None,
        "raw_scores_are_probabilities": False,
        "required_human_gate": "development_protocol_and_holdout_boundary",
    }
    runtime = {
        "python": ">=3.10",
        "onnx_opset": 17,
        "onnxruntime": importlib.metadata.version("onnxruntime"),
        "provider": "CPUExecutionProvider",
        "tensorrt_required_for_jetson": True,
    }
    package_stem = model_family.replace("_", "-")
    package_path = output_root / f"{package_stem}-random-weight-fixture.deployment.zip"
    created = build_deployment_package(
        package_path,
        onnx_path=onnx_path,
        metadata=metadata,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        ontology_path=repository / "configs" / "dataset" / "ontology_v1.yaml",
        calibration=calibration,
        thresholds=thresholds,
        runtime_requirements=runtime,
        numerical_validation=numerical,
    )
    fixture = np.linspace(-1.0, 1.0, num=1 * 3 * 128 * 256, dtype=np.float32).reshape(
        1, 3, 128, 256
    )
    verified = verify_deployment_package(
        package_path,
        expected_identities={
            "model_family": model_family,
            "config_sha256": config_sha256,
            "checkpoint_sha256": semantic_result["checkpoint_sha256"],
        },
        expected_ontology_sha256=sha256_file(
            repository / "configs" / "dataset" / "ontology_v1.yaml"
        ),
        expected_preprocessing_sha256=sha256_payload(preprocessing),
        fixture_input=fixture,
    )
    validation_path = output_root / "deployment-validation.json"
    validation_path.write_text(canonical_json(verified) + "\n", encoding="utf-8")

    equivalence = _json(equivalence_report_path)
    onnx_zip = _stable_zip(
        output_root / "onnx-equivalence-report.zip",
        {"report.json": _json_bytes(equivalence)},
    )
    deployment_zip = _stable_zip(
        output_root / "deployment-package-validation.zip",
        {
            "deployment-validation.json": _json_bytes(verified),
            "manifest.json": _json_bytes(created["manifest"]),
            "README.md": (
                b"NON-SCIENTIFIC PIPELINE VALIDATION. The ONNX binary remains in the "
                b"external deployment fixture package.\n"
            ),
        },
    )
    readiness = check_precolab_readiness(
        repository,
        expected_commit=expected_commit,
        closure_summary_path=closure_root / "campaign_summary.json",
        equivalence_report_path=equivalence_report_path,
        deployment_validation_path=validation_path,
    )
    readiness_path = output_root / "precolab-readiness.json"
    readiness_path.write_text(canonical_json(readiness) + "\n", encoding="utf-8")
    runbook = (repository / "docs" / "canonical-colab-runbook.md").read_bytes()
    (output_root / "canonical-colab-runbook.md").write_bytes(runbook)
    final_zip = _stable_zip(
        output_root / "final-pre-colab-review.zip",
        {
            "campaign-summary.json": (closure_root / "campaign_summary.json").read_bytes(),
            "canonical-colab-runbook.md": runbook,
            "deployment-validation.json": _json_bytes(verified),
            "onnx-equivalence-summary.json": _json_bytes(equivalence),
            "precolab-readiness.json": _json_bytes(readiness),
        },
    )
    return {
        "status": "passed",
        "git_commit": expected_commit,
        "deployment_package": {
            key: created[key] for key in ("filename", "sha256", "byte_size", "package_identity")
        },
        "deployment_validation": verified,
        "onnx_equivalence_report": onnx_zip,
        "deployment_package_validation": deployment_zip,
        "final_pre_colab_review": final_zip,
        "readiness": readiness,
        "scientific_evidence": False,
    }
