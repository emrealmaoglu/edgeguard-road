"""Evaluate a frozen semantic model on selection, calibration, ID, or ACDC data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgeguard.rescue.config import load_rescue_config
from edgeguard.rescue.external import package_external_predictions, record_external_server_result
from edgeguard.rescue.mmseg_runtime import evaluate_model
from edgeguard.rescue.multidomain import validate_dataset_manifest
from edgeguard.rescue.reliability import fit_global_temperature_from_evidence
from edgeguard.rescue.reporting import build_evidence_report
from edgeguard.rescue.selection import select_top_two
from edgeguard.rescue.shift import (
    apply_shift_reference,
    fit_shift_reference,
    frame_shift_metrics,
    load_shift_reference,
    save_shift_reference,
)
from edgeguard.rescue.stress import build_stress_dataset
from edgeguard.serialization import canonical_json, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="evaluate one frozen model")
    run.add_argument("--config", type=Path, default=Path("configs/rescue/semantic_first.yaml"))
    run.add_argument("--resolved-config", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument(
        "--dataset",
        choices=(
            "cityscapes",
            "bdd100k",
            "idd20k",
            "cityscapes_stress",
            "acdc",
            "wilddash2",
            "muses",
            "kitti",
        ),
        required=True,
    )
    run.add_argument("--dataset-root", type=Path)
    run.add_argument("--dataset-manifest", type=Path)
    run.add_argument("--sealed-release", type=Path)
    run.add_argument("--split-manifest", type=Path)
    run.add_argument(
        "--role",
        choices=(
            "train_select",
            "train_calibration",
            "official_val_common_eval",
            "domain_shift_val",
            "synthetic_stress_test",
            "sealed_external_test",
            "official_source_val",
        ),
        required=True,
    )
    run.add_argument("--condition", choices=("fog", "night", "rain", "snow"))
    run.add_argument("--output-dir", type=Path, required=True)
    calibration = run.add_mutually_exclusive_group()
    calibration.add_argument("--fit-temperature", action="store_true")
    calibration.add_argument("--temperature-file", type=Path)
    run.add_argument("--max-reliability-pixels", type=int, default=100_000)
    run.add_argument("--rare-classes-file", type=Path)
    run.add_argument("--save-calibration-evidence", type=Path)
    global_calibration = commands.add_parser(
        "calibrate-global", help="fit one equal-domain source calibration temperature"
    )
    global_calibration.add_argument("--evidence", type=Path, action="append", required=True)
    global_calibration.add_argument("--output", type=Path, required=True)
    global_calibration.add_argument("--seed", type=int, default=20260728)
    shift_calibration = commands.add_parser(
        "calibrate-shift", help="freeze source-only frame uncertainty thresholds"
    )
    shift_calibration.add_argument("--summary", type=Path, action="append", required=True)
    shift_calibration.add_argument("--checkpoint", type=Path, required=True)
    shift_calibration.add_argument("--data-manifest", type=Path, action="append", required=True)
    shift_calibration.add_argument("--quantile", type=float, default=0.95)
    shift_calibration.add_argument("--output", type=Path, required=True)
    shift_evaluation = commands.add_parser(
        "evaluate-shift", help="measure source-vs-external frame shift separation"
    )
    shift_evaluation.add_argument("--source-summary", type=Path, action="append", required=True)
    shift_evaluation.add_argument("--external-summary", type=Path, action="append", required=True)
    shift_evaluation.add_argument("--reference", type=Path, required=True)
    shift_evaluation.add_argument("--output", type=Path, required=True)
    package = commands.add_parser(
        "package-external", help="create one hash-bound WildDash/MUSES submission package"
    )
    package.add_argument("--dataset-manifest", type=Path, required=True)
    package.add_argument("--model", type=Path, required=True)
    package.add_argument("--resolved-config", type=Path)
    package.add_argument("--sealed-release", type=Path, required=True)
    package.add_argument("--device", default="cpu")
    package.add_argument("--output-dir", type=Path, required=True)
    external_result = commands.add_parser(
        "record-external", help="bind an official server score to the submitted archive"
    )
    external_result.add_argument("--package-manifest", type=Path, required=True)
    external_result.add_argument("--server-result", type=Path, required=True)
    external_result.add_argument("--output", type=Path, required=True)
    select = commands.add_parser("select", help="apply the frozen top-two rule")
    select.add_argument("--candidate-table", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    summarize = commands.add_parser("summarize", help="build measured report tables")
    summarize.add_argument("--evaluation-root", type=Path, required=True)
    summarize.add_argument("--export-root", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    stress = commands.add_parser("stress", help="build a clearly labeled synthetic fallback")
    stress.add_argument("--cityscapes-root", type=Path, required=True)
    stress.add_argument("--output-root", type=Path, required=True)
    stress.add_argument("--condition", choices=("fog", "night", "rain", "snow"), required=True)
    stress.add_argument("--severity", type=float, default=0.6)
    stress.add_argument("--limit", type=int)
    return parser


def main() -> int:
    """Run standard IoUMetric and optional reliability measurement."""
    args = _parser().parse_args()
    if args.command == "stress":
        result = build_stress_dataset(
            args.cityscapes_root.resolve(),
            args.output_root.resolve(),
            condition=args.condition,
            severity=args.severity,
            limit=args.limit,
        )
        print(canonical_json(result))
        return 0
    if args.command == "calibrate-global":
        result = fit_global_temperature_from_evidence(
            tuple(path.resolve() for path in args.evidence),
            args.output.resolve(),
            seed=args.seed,
        )
        print(canonical_json(result))
        return 0
    if args.command == "calibrate-shift":
        summaries: list[dict[str, float | None]] = []
        for path in args.summary:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload.get("frames") if isinstance(payload, dict) else None
            if not isinstance(records, list):
                raise ValueError("each shift summary file must contain a frames list")
            summaries.extend(records)
        manifest_hashes: dict[str, str] = {}
        for path in args.data_manifest:
            manifest = validate_dataset_manifest(path.resolve())
            dataset_id = str(manifest["dataset_id"])
            if dataset_id not in {"cityscapes", "bdd100k", "idd20k"}:
                raise ValueError("shift calibration accepts only source-domain manifests")
            if dataset_id in manifest_hashes:
                raise ValueError("shift calibration repeats a source domain")
            manifest_hashes[dataset_id] = sha256_file(path.resolve())
        result = fit_shift_reference(
            summaries,
            checkpoint_sha256=sha256_file(args.checkpoint.resolve()),
            dataset_manifest_sha256s=manifest_hashes,
            quantile=args.quantile,
        )
        save_shift_reference(args.output.resolve(), result)
        print(canonical_json(result))
        return 0
    if args.command == "evaluate-shift":

        def load_frames(paths: list[Path]) -> list[dict[str, float | None]]:
            frames: list[dict[str, float | None]] = []
            for path in paths:
                payload = json.loads(path.read_text(encoding="utf-8"))
                records = payload.get("frames") if isinstance(payload, dict) else None
                if not isinstance(records, list):
                    raise ValueError("each shift summary file must contain a frames list")
                frames.extend(records)
            return frames

        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite shift evidence: {args.output}")
        reference = load_shift_reference(args.reference.resolve())
        source_frames = load_frames(args.source_summary)
        external_frames = load_frames(args.external_summary)
        source_alerts = [
            bool(apply_shift_reference(row, reference)["domain_shift_alert"])
            for row in source_frames
        ]
        external_alerts = [
            bool(apply_shift_reference(row, reference)["domain_shift_alert"])
            for row in external_frames
        ]
        result = frame_shift_metrics(
            source_frames,
            external_frames,
            source_alerts=source_alerts,
            external_alerts=external_alerts,
        )
        result["shift_reference_sha256"] = sha256_file(args.reference.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(canonical_json(result) + "\n", encoding="utf-8")
        print(canonical_json(result))
        return 0
    if args.command == "package-external":
        result = package_external_predictions(
            manifest_path=args.dataset_manifest.resolve(),
            model_path=args.model.resolve(),
            output_dir=args.output_dir.resolve(),
            sealed_release=args.sealed_release.resolve(),
            resolved_config=args.resolved_config.resolve() if args.resolved_config else None,
            device=args.device,
        )
        print(canonical_json(result))
        return 0
    if args.command == "record-external":
        result = record_external_server_result(
            args.package_manifest.resolve(), args.server_result.resolve(), args.output.resolve()
        )
        print(canonical_json(result))
        return 0
    if args.command == "summarize":
        result = build_evidence_report(
            args.evaluation_root.resolve(), args.export_root.resolve(), args.output_dir.resolve()
        )
        print(canonical_json(result))
        return 0
    if args.command == "select":
        payload = json.loads(args.candidate_table.read_text(encoding="utf-8"))
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("candidate table must contain a candidates list")
        result = select_top_two(candidates)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(canonical_json(result) + "\n", encoding="utf-8")
        print(canonical_json(result))
        return 0
    if args.max_reliability_pixels <= 0:
        raise ValueError("--max-reliability-pixels must be positive")
    if args.dataset_root is None and args.dataset_manifest is None:
        raise ValueError("provide --dataset-root or --dataset-manifest")
    result = evaluate_model(
        load_rescue_config(args.config.resolve()),
        resolved_config=args.resolved_config.resolve(),
        checkpoint=args.checkpoint.resolve(),
        dataset=args.dataset,
        dataset_root=args.dataset_root.resolve() if args.dataset_root else None,
        split_manifest=args.split_manifest.resolve() if args.split_manifest else None,
        role=args.role,
        condition=args.condition,
        output_dir=args.output_dir.resolve(),
        fit_calibrator=args.fit_temperature,
        temperature_file=args.temperature_file.resolve() if args.temperature_file else None,
        max_reliability_pixels=args.max_reliability_pixels,
        rare_classes_file=(args.rare_classes_file.resolve() if args.rare_classes_file else None),
        dataset_manifest=(args.dataset_manifest.resolve() if args.dataset_manifest else None),
        sealed_release=args.sealed_release.resolve() if args.sealed_release else None,
        calibration_evidence_output=(
            args.save_calibration_evidence.resolve() if args.save_calibration_evidence else None
        ),
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
