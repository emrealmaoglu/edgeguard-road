"""Sustained TensorRT FP16 semantic-pipeline benchmark for manual Jetson execution."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edgeguard.rescue.inference import preprocess_image
from edgeguard.rescue.perception import derive_perception
from edgeguard.rescue.visualization import confidence_entropy
from edgeguard.serialization import canonical_json, sha256_file

POWER_PATTERN = re.compile(r"\bVDD_IN\s+(?P<mw>[0-9]+)mW")
RAM_PATTERN = re.compile(r"\bRAM\s+(?P<used>[0-9]+)/(?P<total>[0-9]+)MB")
TEMPERATURE_PATTERN = re.compile(r"(?P<zone>[A-Za-z0-9_]+)@(?P<celsius>[0-9.]+)C")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--engine-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--telemetry-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--minimum-iterations", type=int, default=5000)
    parser.add_argument("--minimum-duration-seconds", type=float, default=600.0)
    parser.add_argument("--power-profile", choices=("25W", "MAXN_SUPER"), required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--entropy-threshold", type=float, default=0.5)
    return parser


def parse_tegrastats(path: Path) -> dict[str, Any]:
    """Parse only explicitly present RAM, temperature, power, and warning evidence."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    powers: list[float] = []
    ram_used: list[int] = []
    ram_total: list[int] = []
    temperatures: dict[str, list[float]] = {}
    warning_lines: list[str] = []
    for line in lines:
        power = POWER_PATTERN.search(line)
        if power:
            powers.append(int(power.group("mw")) / 1000.0)
        ram = RAM_PATTERN.search(line)
        if ram:
            ram_used.append(int(ram.group("used")))
            ram_total.append(int(ram.group("total")))
        for match in TEMPERATURE_PATTERN.finditer(line):
            temperatures.setdefault(match.group("zone"), []).append(float(match.group("celsius")))
        lowered = line.lower()
        if any(token in lowered for token in ("throttl", "over-current", "undervolt")):
            warning_lines.append(line)
    return {
        "source": path.name,
        "sha256": sha256_file(path),
        "line_count": len(lines),
        "mean_input_power_w": float(np.mean(powers)) if powers else None,
        "peak_input_power_w": float(np.max(powers)) if powers else None,
        "mean_ram_used_mib": float(np.mean(ram_used)) if ram_used else None,
        "peak_ram_used_mib": int(max(ram_used)) if ram_used else None,
        "ram_total_mib": int(max(ram_total)) if ram_total else None,
        "temperature_c": {
            zone: {"mean": float(np.mean(values)), "peak": float(np.max(values))}
            for zone, values in sorted(temperatures.items())
        },
        "throttling_warning_detected": bool(warning_lines),
        "warning_line_count": len(warning_lines),
        "telemetry_complete": bool(lines and powers and ram_used and temperatures),
    }


def realtime_acceptance(
    *, median_ms: float, p95_ms: float, fps: float, telemetry: dict[str, Any]
) -> dict[str, Any]:
    """Apply the thesis-frozen 25W real-time gate without inventing measurements."""
    checks = {
        "median_latency_le_50ms": median_ms <= 50.0,
        "p95_latency_le_66_7ms": p95_ms <= 66.7,
        "sustained_fps_ge_20": fps >= 20.0,
        "telemetry_complete": telemetry.get("telemetry_complete") is True,
        "no_telemetry_throttling_warning": (telemetry.get("throttling_warning_detected") is False),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "claim_scope": "25W_end_to_end_realtime" if all(checks.values()) else "gate_not_passed",
    }


class TensorRTTorchRunner:
    """TensorRT 10 runner backed by target-device CUDA torch tensors."""

    def __init__(self, engine_path: Path) -> None:
        try:
            self.torch = __import__("torch")
            self.trt = __import__("tensorrt")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Jetson benchmark requires target TensorRT and CUDA PyTorch"
            ) from error
        if not self.torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable on the target device")
        logger = self.trt.Logger(self.trt.Logger.WARNING)
        runtime = self.trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT could not deserialize the engine")
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)]
        inputs = [
            name
            for name in names
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT
        ]
        outputs = [name for name in names if name not in inputs]
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError("semantic engine must have exactly one input and one output")
        self.input_name = inputs[0]
        self.output_name = outputs[0]
        self.input_shape = tuple(
            int(value) for value in self.engine.get_tensor_shape(self.input_name)
        )
        if len(self.input_shape) != 4 or any(value <= 0 for value in self.input_shape):
            raise RuntimeError("benchmark requires one static NCHW input")
        self.context.set_input_shape(self.input_name, self.input_shape)
        self.output_shape = tuple(
            int(value) for value in self.context.get_tensor_shape(self.output_name)
        )
        if len(self.output_shape) != 4 or self.output_shape[1] != 19:
            raise RuntimeError("semantic engine output must be static N19HW logits")
        self.input = self.torch.empty(self.input_shape, device="cuda", dtype=self.torch.float32)
        self.output = self.torch.empty(self.output_shape, device="cuda", dtype=self.torch.float32)
        self.context.set_tensor_address(self.input_name, self.input.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output.data_ptr())

    def infer(self, tensor: np.ndarray) -> tuple[np.ndarray, float]:
        """Run one synchronized engine call and return CPU logits plus GPU latency."""
        source = self.torch.from_numpy(np.ascontiguousarray(tensor)).to("cuda")
        self.input.copy_(source)
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        stream = self.torch.cuda.current_stream()
        start.record(stream)
        if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")
        end.record(stream)
        end.synchronize()
        return self.output.detach().cpu().numpy(), float(start.elapsed_time(end))


def _images(root: Path) -> list[Path]:
    paths = sorted((*root.glob("**/*.png"), *root.glob("**/*.jpg"), *root.glob("**/*.jpeg")))
    if not paths:
        raise FileNotFoundError(f"no benchmark images found under {root}")
    return paths


def _power_mode_query() -> str | None:
    try:
        return subprocess.run(
            ["/usr/sbin/nvpmodel", "-q"], check=False, capture_output=True, text=True
        ).stdout[-2000:]
    except OSError:
        return None


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Measure pure engine and complete preprocess/inference/postprocess latency."""
    if args.warmup < 200:
        raise ValueError("scientific Jetson benchmark requires at least 200 warm-up frames")
    if args.minimum_iterations < 5000 or args.minimum_duration_seconds < 600.0:
        raise ValueError(
            "scientific Jetson benchmark thresholds cannot be below 5,000 frames/600 seconds"
        )
    manifest = json.loads(args.engine_manifest.read_text(encoding="utf-8"))
    if manifest.get("engine_sha256") != sha256_file(args.engine):
        raise ValueError("TensorRT engine does not match its build manifest")
    images = _images(args.image_root)
    runner = TensorRTTorchRunner(args.engine)
    input_height, input_width = runner.input_shape[2:]
    for index in range(args.warmup):
        with Image.open(images[index % len(images)]) as opened:
            tensor = preprocess_image(opened.convert("RGB"), (input_height, input_width))
        runner.infer(tensor)
    engine_ms: list[float] = []
    end_to_end_ms: list[float] = []
    started_campaign = time.monotonic()
    index = 0
    while index < args.minimum_iterations and (
        time.monotonic() - started_campaign < args.minimum_duration_seconds
    ):
        started = time.perf_counter()
        with Image.open(images[index % len(images)]) as opened:
            image = opened.convert("RGB")
            tensor = preprocess_image(image, (input_height, input_width))
        logits, pure_ms = runner.infer(tensor)
        confidence, entropy = confidence_entropy(logits[0])
        derive_perception(
            np.argmax(logits[0], axis=0).astype(np.uint8),
            confidence,
            entropy,
            confidence_threshold=args.confidence_threshold,
            entropy_threshold=args.entropy_threshold,
        )
        end_to_end_ms.append((time.perf_counter() - started) * 1000.0)
        engine_ms.append(pure_ms)
        index += 1
    elapsed = time.monotonic() - started_campaign
    if not end_to_end_ms:
        raise RuntimeError("benchmark produced no frames")
    telemetry = parse_tegrastats(args.telemetry_log)
    median = float(np.median(end_to_end_ms))
    p95 = float(np.percentile(end_to_end_ms, 95))
    fps = float(len(end_to_end_ms) / elapsed)
    mean_power = telemetry["mean_input_power_w"]
    result = {
        "schema_version": "1.0",
        "record_type": "edgeguard_jetson_tensorrt_sustained_benchmark",
        "engine_sha256": sha256_file(args.engine),
        "engine_manifest_sha256": sha256_file(args.engine_manifest),
        "backend": "tensorrt_fp16",
        "power_profile_declared": args.power_profile,
        "nvpmodel_query": _power_mode_query(),
        "warmup_frames": args.warmup,
        "measured_frames": len(end_to_end_ms),
        "elapsed_seconds": elapsed,
        "input_shape": list(runner.input_shape),
        "output_shape": list(runner.output_shape),
        "pure_engine_latency_ms": {
            "median": float(np.median(engine_ms)),
            "p95": float(np.percentile(engine_ms, 95)),
            "p99": float(np.percentile(engine_ms, 99)),
        },
        "end_to_end_latency_ms": {
            "median": median,
            "p95": p95,
            "p99": float(np.percentile(end_to_end_ms, 99)),
        },
        "sustained_fps": fps,
        "telemetry": telemetry,
        "joule_per_frame": float(mean_power / fps) if mean_power is not None else None,
        "system": {"platform": platform.platform(), "machine": platform.machine()},
        "preprocessing_included": True,
        "semantic_postprocessing_included": True,
        "ui_included": False,
        "power_mode_changed_by_script": False,
        "scientific_measurement": True,
    }
    result["realtime_acceptance"] = (
        realtime_acceptance(median_ms=median, p95_ms=p95, fps=fps, telemetry=telemetry)
        if args.power_profile == "25W"
        else {
            "passed": None,
            "checks": {},
            "claim_scope": "MAXN_SUPER_comparison_only",
        }
    )
    return result


def main() -> int:
    """Write non-overwriting target-device benchmark evidence."""
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
