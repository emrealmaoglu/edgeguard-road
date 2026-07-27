"""Run the bounded production-architecture ONNX equivalence investigation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgeguard.export.equivalence import run_production_equivalence_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=Path("configs/training/segmentation"))
    parser.add_argument("--mmseg-checkout", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_production_equivalence_probe(
        args.config_root.resolve(), args.mmseg_checkout.resolve(), args.output_root.resolve()
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
