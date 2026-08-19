"""Break the sustained-benchmark frame time into its individual stages.

`benchmark.py` reports one pure-engine number and one end-to-end number. On a real Orin
Nano Super the gap between them was 30x -- a 5.1 ms TensorRT engine inside a 151.7 ms
frame -- which says the accelerator is not what decides whether this pipeline runs in real
time. This profiles exactly the stages that benchmark's measured region contains, so the
gap can be attributed instead of guessed at.

Stage boundaries mirror `benchmark.benchmark`'s loop body one-for-one. Any divergence
would make the attribution describe a pipeline nobody runs.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

# Running this as `python scripts/jetson/<name>.py` puts *this directory* on the import
# path, not the repository root, so the shared runner below is not importable. The runbook
# documents exactly that invocation, so the script makes it work rather than asking the
# reader to know about PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edgeguard.rescue.inference import preprocess_image
from edgeguard.rescue.perception import derive_perception
from edgeguard.rescue.visualization import confidence_entropy
from edgeguard.serialization import canonical_json, sha256_file
from scripts.jetson.benchmark import TensorRTTorchRunner, _images

STAGES = (
    "image_open_and_decode",
    "preprocess",
    "engine_infer",
    "argmax",
    "confidence_entropy",
    "derive_perception",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--entropy-threshold", type=float, default=0.5)
    parser.add_argument(
        "--gpu-preprocess",
        action="store_true",
        help="letterbox and normalise on the GPU instead of the CPU; also reports how far "
        "the resulting prediction moves, since the two filters are not bit-identical",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    images = _images(args.image_root)
    runner = TensorRTTorchRunner(args.engine)
    input_height, input_width = runner.input_shape[2:]

    for index in range(args.warmup):
        with Image.open(images[index % len(images)]) as opened:
            tensor = preprocess_image(opened.convert("RGB"), (input_height, input_width))
        runner.infer(tensor)

    timings: dict[str, list[float]] = {name: [] for name in STAGES}
    agreement: list[float] = []
    resolutions: dict[str, int] = {}
    for index in range(args.frames):
        path = images[index % len(images)]

        start = time.perf_counter()
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        timings["image_open_and_decode"].append((time.perf_counter() - start) * 1000.0)
        resolutions[f"{image.width}x{image.height}"] = (
            resolutions.get(f"{image.width}x{image.height}", 0) + 1
        )

        rgb = np.asarray(image, dtype=np.uint8)
        if args.gpu_preprocess:
            start = time.perf_counter()
            runner.preprocess_on_device(rgb)
            runner.torch.cuda.synchronize()
            timings["preprocess"].append((time.perf_counter() - start) * 1000.0)
            logits, pure_ms = runner.infer_prepared()
        else:
            start = time.perf_counter()
            tensor = preprocess_image(image, (input_height, input_width))
            timings["preprocess"].append((time.perf_counter() - start) * 1000.0)
            logits, pure_ms = runner.infer(tensor)
        timings["engine_infer"].append(pure_ms)

        if args.gpu_preprocess:
            # A faster preprocess is only a win if the prediction survives it, so measure
            # the disagreement rather than assuming the filters match.
            reference, _ = runner.infer(preprocess_image(image, (input_height, input_width)))
            agreement.append(
                float(np.mean(np.argmax(logits[0], axis=0) == np.argmax(reference[0], axis=0)))
            )

        start = time.perf_counter()
        mask = np.argmax(logits[0], axis=0).astype(np.uint8)
        timings["argmax"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        confidence, entropy = confidence_entropy(logits[0])
        timings["confidence_entropy"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        derive_perception(
            mask,
            confidence,
            entropy,
            confidence_threshold=args.confidence_threshold,
            entropy_threshold=args.entropy_threshold,
        )
        timings["derive_perception"].append((time.perf_counter() - start) * 1000.0)

    stages = {
        name: {
            "median_ms": float(np.median(values)),
            "mean_ms": float(np.mean(values)),
            "p95_ms": float(np.percentile(values, 95)),
        }
        for name, values in timings.items()
    }
    total_median = sum(row["median_ms"] for row in stages.values())
    for row in stages.values():
        row["share_of_median_frame"] = row["median_ms"] / total_median if total_median else None

    record = {
        "schema_version": "1.0",
        "record_type": "edgeguard_jetson_pipeline_stage_profile",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": str(args.engine),
        "engine_sha256": sha256_file(args.engine),
        "input_shape": list(runner.input_shape),
        "frames": args.frames,
        "warmup_frames": args.warmup,
        "source_resolutions": resolutions,
        "stages": stages,
        "gpu_preprocess": args.gpu_preprocess,
        "cpu_preprocess_argmax_agreement": (float(np.mean(agreement)) if agreement else None),
        "summed_median_frame_ms": total_median,
        # Stage boundaries mirror `benchmark.py`'s measured region, but summing medians is
        # not the same statistic as the median of the whole frame, so this is an
        # attribution of where time goes, not a restatement of the benchmark's latency.
        "comparable_to_benchmark_end_to_end": False,
        "scientific_measurement": True,
    }
    args.output.write_text(canonical_json(record) + "\n", encoding="utf-8")

    print(f"{'stage':24s} {'medyan ms':>10} {'p95 ms':>10} {'pay':>7}")
    for name, row in sorted(stages.items(), key=lambda kv: -kv[1]["median_ms"]):
        share = row["share_of_median_frame"]
        print(
            f"{name:24s} {row['median_ms']:10.2f} {row['p95_ms']:10.2f} "
            f"{(share * 100 if share else 0):6.1f}%"
        )
    print(f"\n{'toplam (medyanlar)':24s} {total_median:10.2f}")
    if agreement:
        print(f"CPU on islemeye argmax uyumu: {np.mean(agreement) * 100:.4f}%")
    print(f"kayıt: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
