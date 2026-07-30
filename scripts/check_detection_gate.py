"""Evaluate the fail-closed RTMDet-Tiny phase-two gate from measured evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgeguard.rescue.detection_gate import evaluate_detection_gate
from edgeguard.serialization import canonical_json, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-one-evidence", type=Path, required=True)
    parser.add_argument("--jetson-benchmark", type=Path, required=True)
    parser.add_argument("--bdd-detection-provenance", type=Path, required=True)
    parser.add_argument("--remaining-gpu-hours", type=float, required=True)
    parser.add_argument("--remaining-calendar-days", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Write a new hash-bound phase-two decision without overwriting evidence."""
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite detection-gate evidence: {args.output}")
    inputs = (
        args.phase_one_evidence,
        args.jetson_benchmark,
        args.bdd_detection_provenance,
    )
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    result = evaluate_detection_gate(
        payloads[0],
        payloads[1],
        payloads[2],
        remaining_gpu_hours=args.remaining_gpu_hours,
        remaining_calendar_days=args.remaining_calendar_days,
    )
    result["input_sha256s"] = {path.name: sha256_file(path.resolve()) for path in inputs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(result) + "\n", encoding="utf-8")
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
