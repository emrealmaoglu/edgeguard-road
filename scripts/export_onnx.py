"""Export a frozen MMSeg checkpoint and verify PyTorch/ONNX Runtime agreement."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from edgeguard.rescue.ledger import append_run_ledger
from edgeguard.rescue.mmseg_runtime import install_mmcv_lite_guard
from edgeguard.serialization import canonical_json, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    return parser


def export_and_verify(args: argparse.Namespace) -> dict[str, Any]:
    """Perform static export, checker validation, equivalence, and CPU timing."""
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite ONNX artifact: {args.output}")
    if min(args.input_height, args.input_width, args.iterations) <= 0 or args.warmup < 0:
        raise ValueError("input size/iterations must be positive and warmup cannot be negative")
    install_mmcv_lite_guard()
    try:
        torch = __import__("torch")
        onnx = __import__("onnx")
        ort = __import__("onnxruntime")
        apis = __import__("mmseg.apis", fromlist=["init_model"])
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "install the canonical semantic and ONNX runtimes before export"
        ) from error
    model = apis.init_model(
        str(args.resolved_config.resolve()), str(args.checkpoint.resolve()), device=args.device
    ).eval()

    class NativeLogits(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self, semantic_model: Any) -> None:
            super().__init__()
            self.semantic_model = semantic_model

        def forward(self, inputs: Any) -> Any:
            decoded = self.semantic_model.decode_head.forward(
                self.semantic_model.extract_feat(inputs)
            )
            if isinstance(decoded, (list, tuple)):
                indices = {"PIDHead": 1, "DDRHead": 0}
                name = type(self.semantic_model.decode_head).__name__
                if name not in indices:
                    raise RuntimeError(f"unreviewed multi-output decode head: {name}")
                decoded = decoded[indices[name]]
            return decoded

    wrapper = NativeLogits(model).eval()
    torch.manual_seed(20260728)
    tensor = torch.randn(
        (1, 3, args.input_height, args.input_width), dtype=torch.float32, device=args.device
    )
    with torch.no_grad():
        expected = wrapper(tensor).detach().cpu().numpy()
    if expected.ndim != 4 or expected.shape[1] != 19:
        raise RuntimeError(f"export requires 19-channel NCHW logits, received {expected.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        tensor,
        args.output,
        input_names=["normalized_rgb"],
        output_names=["native_logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    feed = {"normalized_rgb": tensor.detach().cpu().numpy()}
    actual = session.run(["native_logits"], feed)[0]
    if expected.shape != actual.shape or not bool(np.isfinite(actual).all()):
        raise RuntimeError("ONNX output shape/finiteness check failed")
    difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    for _ in range(args.warmup):
        session.run(["native_logits"], feed)
    timings = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        session.run(["native_logits"], feed)
        timings.append((time.perf_counter() - started) * 1000.0)
    result = {
        "schema_version": "1.0",
        "record_type": "semantic_onnx_validation",
        "opset": args.opset,
        "input_name": "normalized_rgb",
        "input_shape": list(tensor.shape),
        "input_contract": "RGB float32 normalized by ImageNet mean/std outside graph",
        "output_name": "native_logits",
        "output_shape": list(actual.shape),
        "class_count": int(actual.shape[1]),
        "shape_equal": expected.shape == actual.shape,
        "max_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "allclose_atol_1e_4_rtol_1e_4": bool(
            np.allclose(expected, actual, atol=1.0e-4, rtol=1.0e-4)
        ),
        "onnx_sha256": sha256_file(args.output),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "onnx_bytes": args.output.stat().st_size,
        "onnxruntime_cpu": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "median_latency_ms": float(np.median(timings)),
            "p95_latency_ms": float(np.percentile(timings, 95)),
            "fps_from_median": float(1000.0 / np.median(timings)),
        },
        "jetson_evidence": False,
        "scientific_accuracy_evidence": False,
    }
    report_path = args.output.with_suffix(".validation.json")
    report_path.write_text(canonical_json(result) + "\n", encoding="utf-8")
    append_run_ledger(
        args.output.parent / "run_ledger.jsonl",
        operation="onnx_export_validation",
        result=result,
    )
    return result


def main() -> int:
    """Export and print the path-free validation record."""
    result = export_and_verify(_parser().parse_args())
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
