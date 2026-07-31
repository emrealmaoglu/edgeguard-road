"""Actual random-weight MMSeg architecture ONNX feasibility probes."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from edgeguard.models.semantic_local import _build_model
from edgeguard.serialization import sha256_file
from edgeguard.training.config import load_semantic_model_suite


def probe_five_semantic_onnx(
    config_root: Path,
    mmseg_checkout: Path,
    output_root: Path,
    *,
    opset: int = 17,
) -> dict[str, Any]:
    """Export five actual random-weight decode heads and compare PyTorch with ORT."""
    if opset != 17:
        raise ValueError("local architecture probe is frozen to opset 17")
    torch = __import__("torch")
    onnx = __import__("onnx")
    ort = __import__("onnxruntime")
    output_root.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    class NativeWrapper(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model

        def forward(self, inputs: Any) -> Any:
            decoded = self.model.decode_head.forward(self.model.extract_feat(inputs))
            if isinstance(decoded, (list, tuple)):
                mapping = {"PIDHead": 1, "DDRHead": 0}
                name = type(self.model.decode_head).__name__
                if name not in mapping:
                    raise ValueError(f"unreviewed multi-output semantic decode head: {name}")
                return decoded[mapping[name]]
            return decoded

    for index, spec in enumerate(load_semantic_model_suite(config_root)):
        name = spec.model_family.value
        try:
            torch.manual_seed(20260727 + index)
            model = _build_model(spec, mmseg_checkout, torch).eval()
            wrapper = NativeWrapper(model).eval()
            inputs = torch.randn((1, 3, 128, 256), dtype=torch.float32)
            with torch.no_grad():
                expected = wrapper(inputs).detach().cpu().numpy()
            path = output_root / f"{name}.onnx"
            started = time.perf_counter()
            exporter = "legacy"
            try:
                torch.onnx.export(
                    wrapper,
                    inputs,
                    path,
                    input_names=["model_input"],
                    output_names=["native_logits"],
                    opset_version=opset,
                    do_constant_folding=True,
                    dynamo=False,
                )
            except Exception as legacy_error:
                exporter = "dynamo_bounded_fallback"
                try:
                    torch.onnx.export(
                        wrapper,
                        inputs,
                        path,
                        input_names=["model_input"],
                        output_names=["native_logits"],
                        opset_version=opset,
                        dynamo=True,
                    )
                except Exception as dynamo_error:
                    raise RuntimeError(
                        "legacy and bounded dynamo export failed; "
                        f"legacy={type(legacy_error).__name__}: {legacy_error}; "
                        f"dynamo={type(dynamo_error).__name__}: {dynamo_error}"
                    ) from dynamo_error
            export_seconds = time.perf_counter() - started
            graph = onnx.load(path)
            onnx.checker.check_model(graph)
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            actual = session.run(["native_logits"], {"model_input": inputs.numpy()})[0]
            difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
            operators = sorted({node.op_type for node in graph.graph.node})
            results.append(
                {
                    "model_family": name,
                    "architecture": spec.mmseg_config_relative_path,
                    "initialization": "random_no_download",
                    "exporter": exporter,
                    "opset": opset,
                    "input_shape": list(inputs.shape),
                    "pytorch_shape": list(expected.shape),
                    "onnxruntime_shape": list(actual.shape),
                    "shape_equal": expected.shape == actual.shape,
                    "max_absolute_difference": float(difference.max()),
                    "mean_absolute_difference": float(difference.mean()),
                    "allclose_atol_1e_4_rtol_1e_4": bool(
                        np.allclose(expected, actual, atol=1e-4, rtol=1e-4)
                    ),
                    "operator_inventory": operators,
                    "unsupported_operator_report": [],
                    "graph_node_count": len(graph.graph.node),
                    "onnx_byte_size": path.stat().st_size,
                    "onnx_sha256": sha256_file(path),
                    "export_seconds": export_seconds,
                    "scientific_evidence": False,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "model_family": name,
                    "classification": type(error).__name__,
                    "error": str(error)[:2000],
                    "requires_linux_confirmation": True,
                }
            )
    report = {
        "schema_version": "1.0",
        "record_type": "five_actual_semantic_onnx_probe",
        "results": results,
        "failures": failures,
        "scientific_evidence": False,
        "jetson_performance_evidence": False,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return report
