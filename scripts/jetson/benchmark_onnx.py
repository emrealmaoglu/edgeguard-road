"""Reviewed, non-privileged ONNX latency benchmark for manual Jetson execution."""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from edgeguard.serialization import canonical_json, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=1024)
    parser.add_argument("--telemetry-log", type=Path)
    return parser


def _telemetry_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "source": path.name,
        "line_count": len(lines),
        "raw_sha256": sha256_file(path),
        "interpretation": "manual_review_required",
        "note": "No power, temperature, or throttling value is inferred automatically.",
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Measure only the selected ONNX Runtime provider; never alter device state."""
    if min(args.warmup, args.iterations, args.input_height, args.input_width) <= 0:
        raise ValueError("benchmark counts and input dimensions must be positive")
    ort = __import__("onnxruntime")
    available = ort.get_available_providers()
    if args.provider not in available:
        raise RuntimeError(f"provider {args.provider!r} unavailable; found {available}")
    session = ort.InferenceSession(str(args.model), providers=[args.provider])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    rng = np.random.default_rng(20260728)
    tensor = rng.standard_normal((1, 3, args.input_height, args.input_width), dtype=np.float32)
    feed = {input_meta.name: tensor}
    for _ in range(args.warmup):
        session.run([output_meta.name], feed)
    timings: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        session.run([output_meta.name], feed)
        timings.append((time.perf_counter() - started) * 1000.0)
    median = float(np.median(timings))
    return {
        "schema_version": "1.0",
        "record_type": "jetson_onnx_benchmark",
        "model_sha256": sha256_file(args.model),
        "provider": args.provider,
        "available_providers": available,
        "input_shape": list(tensor.shape),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "median_latency_ms": median,
        "p95_latency_ms": float(np.percentile(timings, 95)),
        "fps_from_median": float(1000.0 / median),
        "system": {"platform": platform.platform(), "machine": platform.machine()},
        "telemetry": _telemetry_summary(args.telemetry_log),
        "preprocessing_included": False,
        "ui_included": False,
        "power_temperature_throttling_claim": False,
        "human_device_review_required": True,
    }


def main() -> int:
    """Write the immutable benchmark record after a manual device run."""
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark evidence: {args.output}")
    result = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(result) + "\n", encoding="utf-8")
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
