"""Tests for the device-neutral Jetson deployment handoff bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edgeguard.deployment.jetson_bundle import (
    build_jetson_deployment_bundle,
    verify_jetson_deployment_bundle,
)


def _onnx(path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    tensor = onnx.TensorProto
    weights = onnx.numpy_helper.from_array(
        np.zeros((19, 3, 1, 1), dtype=np.float32), name="weights"
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["normalized_rgb", "weights"],
                ["native_logits"],
                strides=[8, 8],
            )
        ],
        "jetson-fixture",
        [helper.make_tensor_value_info("normalized_rgb", tensor.FLOAT, [1, 3, 512, 1024])],
        [helper.make_tensor_value_info("native_logits", tensor.FLOAT, [1, 19, 64, 128])],
        [weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 10)
    onnx.save(model, path)


def test_jetson_bundle_contains_golden_contract_without_engine(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    model = tmp_path / "model.onnx"
    _onnx(model)
    ontology = tmp_path / "ontology.yaml"
    ontology.write_text("classes: 19\n", encoding="utf-8")
    package = tmp_path / "jetson-deployment.zip"
    input_array = np.zeros((1, 3, 512, 1024), dtype=np.float32)
    output_array = np.zeros((1, 19, 64, 128), dtype=np.float32)

    created = build_jetson_deployment_bundle(
        package,
        onnx_path=model,
        ontology_path=ontology,
        preprocessing={
            "model_input": {
                "name": "normalized_rgb",
                "shape": [1, 3, 512, 1024],
                "dtype": "float32",
            },
            "channel_order": "RGB",
        },
        model_family="segformer_b0",
        source_commit="1" * 40,
        config_sha256="2" * 64,
        checkpoint_sha256="3" * 64,
        golden_input=input_array,
        golden_output=output_array,
    )
    verified = verify_jetson_deployment_bundle(package)

    assert created["tensorrt_engine_included"] is False
    assert verified["status"] == "verified"
    assert verified["jetson_benchmark_status"] == "not_run"


def test_jetson_bundle_rejects_dynamic_or_wrong_input_contract(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"not-reached")
    ontology = tmp_path / "ontology.yaml"
    ontology.write_text("classes: 19\n", encoding="utf-8")
    with pytest.raises(ValueError, match="static 1x3x512x1024"):
        build_jetson_deployment_bundle(
            tmp_path / "bundle.zip",
            onnx_path=model,
            ontology_path=ontology,
            preprocessing={
                "model_input": {
                    "name": "normalized_rgb",
                    "shape": [1, 3, -1, -1],
                    "dtype": "float32",
                }
            },
            model_family="segformer_b0",
            source_commit="1" * 40,
            config_sha256="2" * 64,
            checkpoint_sha256="3" * 64,
            golden_input=np.zeros((1, 3, 512, 1024), dtype=np.float32),
            golden_output=np.zeros((1, 19, 64, 128), dtype=np.float32),
        )
