"""Create segmentation, overlay, confidence, and entropy outputs for one image."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from edgeguard.rescue.inference import predict_mmseg, predict_onnx
from edgeguard.rescue.perception import attention_contract, derive_perception
from edgeguard.rescue.shift import (
    apply_shift_reference,
    frame_uncertainty_summary,
    load_shift_reference,
    uncertainty_maps,
)
from edgeguard.rescue.visualization import resize_scalar, save_perception_result, save_result
from edgeguard.serialization import canonical_json, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help=".onnx graph or .pth checkpoint")
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=1024)
    parser.add_argument("--opacity", type=float, default=0.55)
    parser.add_argument("--emit-regions", action="store_true")
    parser.add_argument("--emit-risk", action="store_true")
    parser.add_argument("--shift-reference", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--entropy-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-region-area", type=int, default=8)
    parser.add_argument("--minimum-drivable-area", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    """Dispatch to the model backend while keeping one output contract."""
    args = _parser().parse_args()
    with Image.open(args.image) as opened:
        image = opened.convert("RGB")
    if args.model.suffix.lower() == ".onnx":
        result = predict_onnx(
            image,
            args.model.resolve(),
            input_size=(args.input_height, args.input_width),
        )
    else:
        if args.resolved_config is None:
            raise ValueError("a PyTorch checkpoint requires --resolved-config")
        result = predict_mmseg(
            image,
            args.resolved_config.resolve(),
            args.model.resolve(),
            device=args.device,
        )
    output_dir = args.output_dir.resolve()
    outputs = save_result(image, result, output_dir, opacity=args.opacity)
    postprocess_started = time.perf_counter()
    energy = None
    uncertainty = None
    if result.logits is not None:
        uncertainty = uncertainty_maps(result.logits)
        energy = uncertainty["energy"]
        if energy.shape != result.mask.shape:
            energy = resize_scalar(energy, image.size)
    frame_summary = frame_uncertainty_summary(
        result.confidence,
        result.entropy,
        energy=energy,
        confidence_threshold=args.confidence_threshold,
    )
    shift_result: dict[str, object] | None = None
    if args.shift_reference is not None:
        reference = load_shift_reference(args.shift_reference.resolve())
        if args.model.suffix.lower() == ".onnx":
            validation_path = args.model.with_suffix(".validation.json")
            if not validation_path.is_file():
                raise ValueError("shift-aware ONNX inference requires its validation artifact")
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            model_checkpoint_sha256 = validation.get("checkpoint_sha256")
        else:
            model_checkpoint_sha256 = sha256_file(args.model)
        if model_checkpoint_sha256 != reference.get("checkpoint_sha256"):
            raise ValueError("shift reference does not match the selected model checkpoint")
        shift_result = apply_shift_reference(frame_summary, reference)
    regions: list[dict[str, object]] = []
    if args.emit_regions or args.emit_risk:
        perception = derive_perception(
            result.mask,
            result.confidence,
            result.entropy,
            confidence_threshold=args.confidence_threshold,
            entropy_threshold=args.entropy_threshold,
            minimum_region_area=args.minimum_region_area,
            minimum_drivable_area=args.minimum_drivable_area,
        )
        perception_outputs, regions = save_perception_result(image, perception, output_dir)
        outputs.update(perception_outputs)
    postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0
    summary = {
        "schema_version": "2.0",
        "record_type": "edgeguard_perception_prediction",
        "model": args.model.name,
        "backend": result.backend,
        "latency_ms": {
            "inference": result.latency_ms,
            "postprocess": postprocess_ms,
            "total": result.latency_ms + postprocess_ms,
        },
        "outputs": outputs,
        "frame_uncertainty": frame_summary,
        "domain_shift": shift_result,
        "domain_shift_alert": (
            shift_result["domain_shift_alert"] if shift_result is not None else None
        ),
        "regions": regions,
        "attention_contract": attention_contract() if args.emit_risk else None,
        "metadata": result.metadata,
        "scientific_benchmark": False,
        "instance_detection": False,
        "physical_risk_probability": False,
    }
    (output_dir / "summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
