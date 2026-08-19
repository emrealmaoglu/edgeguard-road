"""Combine real training-log evidence with real per-class data frequency to support
model/class-scope decisions. See src/edgeguard/rescue/training_log_analysis.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeguard.rescue.training_log_analysis import build_analysis_report, write_analysis_report
from edgeguard.serialization import canonical_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-file",
        type=Path,
        action="append",
        required=True,
        help="real saved Colab training-log text (repeatable)",
    )
    parser.add_argument(
        "--class-weights",
        type=Path,
        required=True,
        help="Cityscapes class_weights.json from write_train_fit_statistics",
    )
    parser.add_argument(
        "--idd20k-summary",
        type=Path,
        help="IDD20K summary.json from audit_training_dataset (optional)",
    )
    parser.add_argument("--pixel-ratio-threshold", type=float, default=0.005)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    log_texts = {str(path): path.read_text(encoding="utf-8") for path in args.log_file}
    report = build_analysis_report(
        log_texts,
        class_weights_path=args.class_weights.resolve(),
        idd20k_summary_path=args.idd20k_summary.resolve() if args.idd20k_summary else None,
        pixel_ratio_threshold=args.pixel_ratio_threshold,
    )
    write_analysis_report(report, args.output_root.resolve())
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
