"""Measure how far frame-level uncertainty responds to weather and lighting shift.

The thesis abstract claims the system should notice "weather, lighting, road and traffic
conditions different from the training distribution". That is a claim about the
uncertainty signal, not about accuracy, so it can be measured without ground-truth masks:
corrupt a fixed set of frames and watch what the frame summary does.

The corruptions come from `rescue.stress._corrupt`, the same transforms the synthetic
stress split uses, so this measures the response to *those* transforms -- not to genuine
night or rain photography. Real adverse-condition imagery (ACDC) would be needed before
claiming anything stronger, and the record says `synthetic_corruption: true` for exactly
that reason.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from edgeguard.rescue.inference import predict_onnx
from edgeguard.rescue.shift import frame_uncertainty_summary, uncertainty_maps
from edgeguard.rescue.stress import _corrupt
from edgeguard.rescue.visualization import resize_scalar
from edgeguard.serialization import canonical_json, sha256_file

CONDITIONS = ("fog", "night", "rain", "snow")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help=".onnx graph")
    parser.add_argument("--frames", type=int, default=15)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="also descend into subdirectories; off by default so sibling label sets "
        "are not silently fed to the model as photographs",
    )
    parser.add_argument("--severity", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    # Non-recursive on purpose. Annotation sets sit beside the photographs -- RoadAnomaly
    # keeps per-frame masks in `<frame>.labels/` -- and a recursive sweep silently runs
    # inference on label images, which moves every number without failing anything.
    # Recursion is opt-in so that has to be a deliberate choice.
    search = args.image_root.rglob if args.recursive else args.image_root.glob
    paths = sorted({path for suffix in ("*.jpg", "*.jpeg", "*.png") for path in search(suffix)})[
        : args.frames
    ]
    if not paths:
        raise ValueError(f"no frames directly under {args.image_root}")

    measurements: dict[str, dict[str, float]] = {}
    for condition in ("clean", *CONDITIONS):
        entropies, energies, low_confidence = [], [], []
        for index, path in enumerate(paths):
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            if condition != "clean":
                image = _corrupt(
                    image, condition=condition, severity=args.severity, seed=args.seed + index
                )
            result = predict_onnx(
                image, args.model.resolve(), input_size=(args.input_height, args.input_width)
            )
            if result.logits is None:
                raise RuntimeError("backend returned no logits")
            energy = uncertainty_maps(result.logits)["energy"]
            if energy.shape != result.confidence.shape:
                energy = resize_scalar(energy, image.size)
            summary = frame_uncertainty_summary(result.confidence, result.entropy, energy=energy)
            entropies.append(summary["mean_normalized_entropy"])
            energies.append(summary["mean_energy"])
            low_confidence.append(summary["low_confidence_pixel_ratio"])
        measurements[condition] = {
            "mean_normalized_entropy": float(np.mean(entropies)),
            "mean_energy": float(np.mean(energies)),
            "low_confidence_pixel_ratio": float(np.mean(low_confidence)),
        }

    clean = measurements["clean"]
    for row in measurements.values():
        row["entropy_ratio_to_clean"] = (
            row["mean_normalized_entropy"] / clean["mean_normalized_entropy"]
            if clean["mean_normalized_entropy"]
            else None
        )

    record = {
        "schema_version": "1.0",
        "record_type": "edgeguard_synthetic_shift_response",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model.name,
        "model_sha256": sha256_file(args.model),
        "image_root": str(args.image_root),
        "frame_count": len(paths),
        "severity": args.severity,
        "seed": args.seed,
        "conditions": measurements,
        # These are algorithmic corruptions, not photographs taken in fog or at night, so
        # the result characterises the response to these transforms and nothing wider.
        "synthetic_corruption": True,
        "real_adverse_condition_imagery": False,
        "scientific_measurement": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(record) + "\n", encoding="utf-8")

    print(f"{'koşul':8s} {'ort.entropi':>12} {'artış':>7} {'düşük-güven px':>15}")
    for condition, row in measurements.items():
        print(
            f"{condition:8s} {row['mean_normalized_entropy']:12.4f} "
            f"{row['entropy_ratio_to_clean']:6.2f}x "
            f"{row['low_confidence_pixel_ratio'] * 100:14.2f}%"
        )
    print(f"kayıt: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
