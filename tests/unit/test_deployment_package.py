"""Deterministic deployment package creation and fail-closed verification."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from edgeguard.deployment.package import build_deployment_package, verify_deployment_package
from edgeguard.serialization import sha256_file, sha256_payload


def _tiny_onnx(path: Path, *, external_data: bool = False) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    tensor = onnx.TensorProto
    bias = onnx.numpy_helper.from_array(
        np.zeros((1, 3, 1, 1), dtype=np.float32), name="fixture_bias"
    )
    graph = helper.make_graph(
        [helper.make_node("Add", ["model_input", "fixture_bias"], ["native_logits"])],
        "fixture",
        [helper.make_tensor_value_info("model_input", tensor.FLOAT, [1, 3, 2, 2])],
        [helper.make_tensor_value_info("native_logits", tensor.FLOAT, [1, 3, 2, 2])],
        [bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    if external_data:
        onnx.save_model(
            model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{path.name}.data",
            size_threshold=0,
        )
    else:
        onnx.save(model, path)


def _inputs(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    model = tmp_path / "model.onnx"
    _tiny_onnx(model)
    ontology = tmp_path / "ontology.yaml"
    ontology.write_text("version: fixture-v1\nclasses: [road, anomaly]\n", encoding="utf-8")
    preprocessing = {
        "model_input": {"name": "model_input", "shape": [1, 3, 2, 2], "dtype": "float32"},
        "channel_order": "RGB",
    }
    values: dict[str, object] = {
        "onnx_path": model,
        "metadata": {
            "model_family": "fixture_semantic",
            "config_sha256": "1" * 64,
            "checkpoint_sha256": "2" * 64,
            "source_commit": "3" * 40,
            "initialization": "random_no_download",
        },
        "preprocessing": preprocessing,
        "postprocessing": {"output": "native_logits", "softmax_in_graph": False},
        "ontology_path": ontology,
        "calibration": {"status": "non_scientific_placeholder", "value": None},
        "thresholds": {"status": "non_scientific_placeholder", "value": None},
        "runtime_requirements": {"onnxruntime": "CPUExecutionProvider"},
        "numerical_validation": {"classification": "fixed_and_equivalent"},
    }
    return ontology, values


def _rewrite(
    source: Path, destination: Path, *, drop: str | None = None, mutate: str | None = None
) -> None:
    with ZipFile(source) as archive, ZipFile(destination, "w", compression=ZIP_DEFLATED) as output:
        for name in archive.namelist():
            if name == drop:
                continue
            payload = archive.read(name)
            if name == mutate:
                payload += b"corrupt"
            output.writestr(name, payload)


def test_deployment_package_creation_verification_and_fixture_inference(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    ontology, values = _inputs(tmp_path)
    package = tmp_path / "fixture.zip"
    created = build_deployment_package(package, **values)  # type: ignore[arg-type]
    preprocessing = values["preprocessing"]
    result = verify_deployment_package(
        package,
        expected_identities={"config_sha256": "1" * 64, "checkpoint_sha256": "2" * 64},
        expected_ontology_sha256=sha256_file(ontology),
        expected_preprocessing_sha256=sha256_payload(preprocessing),
        fixture_input=np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2),
    )

    assert created["status"] == "created"
    assert result["status"] == "verified"
    assert result["fixture_inference"]["status"] == "passed"
    assert result["fixture_inference"]["output_shapes"] == [[1, 3, 2, 2]]


def test_deployment_package_embeds_onnx_external_data(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    _ontology, values = _inputs(tmp_path)
    model = values["onnx_path"]
    assert isinstance(model, Path)
    _tiny_onnx(model, external_data=True)
    external_data = model.with_name(f"{model.name}.data")
    assert external_data.is_file()

    package = tmp_path / "fixture.zip"
    build_deployment_package(package, **values)  # type: ignore[arg-type]
    external_data.unlink()
    result = verify_deployment_package(
        package,
        fixture_input=np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2),
    )

    assert result["status"] == "verified"
    assert result["fixture_inference"]["status"] == "passed"


@pytest.mark.parametrize("member", ["model.onnx", "runtime_requirements.json"])
def test_deployment_package_rejects_missing_or_corrupted_artifact(
    tmp_path: Path, member: str
) -> None:
    _ontology, values = _inputs(tmp_path)
    package = tmp_path / "fixture.zip"
    build_deployment_package(package, **values)  # type: ignore[arg-type]
    broken = tmp_path / "broken.zip"
    if member == "runtime_requirements.json":
        _rewrite(package, broken, drop=member)
        expected = "members mismatch"
    else:
        _rewrite(package, broken, mutate=member)
        expected = "hash mismatch"
    with pytest.raises(ValueError, match=expected):
        verify_deployment_package(broken)


def test_deployment_package_rejects_identity_mismatches(tmp_path: Path) -> None:
    ontology, values = _inputs(tmp_path)
    package = tmp_path / "fixture.zip"
    build_deployment_package(package, **values)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="ontology"):
        verify_deployment_package(package, expected_ontology_sha256="f" * 64)
    with pytest.raises(ValueError, match="preprocessing"):
        verify_deployment_package(package, expected_preprocessing_sha256="e" * 64)
    with pytest.raises(ValueError, match="model/config"):
        verify_deployment_package(package, expected_identities={"config_sha256": "d" * 64})

    assert sha256_file(ontology) != "f" * 64


def test_deployment_package_refuses_invalid_placeholders_and_overwrite(tmp_path: Path) -> None:
    _ontology, values = _inputs(tmp_path)
    package = tmp_path / "fixture.zip"
    invalid = dict(values)
    invalid["thresholds"] = {"status": "selected", "value": 0.5}
    with pytest.raises(ValueError, match="non-scientific"):
        build_deployment_package(package, **invalid)  # type: ignore[arg-type]

    build_deployment_package(package, **values)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="overwrite"):
        build_deployment_package(package, **values)  # type: ignore[arg-type]
