"""Create segmentation, overlay, confidence, and entropy outputs for one image."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from edgeguard.rescue.inference import predict_mmseg, predict_onnx
from edgeguard.rescue.visualization import save_result
from edgeguard.serialization import canonical_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help=".onnx graph or .pth checkpoint")
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-height", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=1024)
    parser.add_argument("--opacity", type=float, default=0.55)
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
    outputs = save_result(image, result, args.output_dir.resolve(), opacity=args.opacity)
    summary = {
        "schema_version": "1.0",
        "record_type": "semantic_prediction",
        "backend": result.backend,
        "latency_ms": result.latency_ms,
        "outputs": outputs,
        "metadata": result.metadata,
        "scientific_benchmark": False,
    }
    (args.output_dir / "summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
