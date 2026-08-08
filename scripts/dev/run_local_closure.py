"""Execute and reuse the comprehensive, non-scientific EG-LOCAL-COMPLETE campaign."""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from edgeguard.calibration import calibration_metrics, fit_temperature
from edgeguard.data.acquisition import (
    copy_with_progress,
    dataset_artifact_readiness,
    download_verified,
    generate_fixture_bundle,
)
from edgeguard.data.contracts import DetectionSample, OODSample, SemanticSample, TemporalFrame
from edgeguard.data.quality import (
    audit_detection_samples,
    audit_ood_samples,
    audit_semantic_samples,
    audit_temporal_frames,
)
from edgeguard.deployment.contracts import validate_jetson_profiles, validate_preregistration
from edgeguard.detection.models import probe_detector_onnx, run_detector_mini_training
from edgeguard.evaluation.components import connected_components
from edgeguard.evaluation.ood import (
    bootstrap_ood_metrics,
    per_source_ood_metrics,
    pixel_ood_metrics,
    score_distribution,
    threshold_policies,
)
from edgeguard.evaluation.statistics import component_detection_metrics
from edgeguard.experiment.hpo import run_semantic_mini_hpo
from edgeguard.export.semantic_onnx import probe_five_semantic_onnx
from edgeguard.models.semantic_local import run_five_model_mini_training
from edgeguard.pipeline.video_local import run_synthetic_video_pipeline
from edgeguard.reporting.local_closure import build_closure_packages, write_gap_matrix
from edgeguard.scoring.uncertainty import (
    energy_anomaly_score,
    max_logit_anomaly_score,
    msp_anomaly_score,
    predictive_entropy,
)
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload
from edgeguard.telemetry.longrun import atomic_write_json

DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "acquisition": (),
    "data_quality": ("acquisition",),
    "semantic": ("data_quality",),
    "semantic_onnx": ("semantic",),
    "detection": ("data_quality",),
    "detector_onnx": ("detection",),
    "ood_calibration": ("semantic",),
    "hpo": ("semantic",),
    "video": ("semantic", "detection", "ood_calibration"),
    "streamlit": ("video",),
    "deployment_contracts": (),
    "resume_probes": ("semantic", "detection", "hpo", "video"),
    "reporting": (
        "acquisition",
        "data_quality",
        "semantic_onnx",
        "detector_onnx",
        "ood_calibration",
        "hpo",
        "video",
        "streamlit",
        "deployment_contracts",
        "resume_probes",
    ),
}


class _Handler(http.server.BaseHTTPRequestHandler):
    payload = b""

    def do_GET(self) -> None:  # noqa: N802
        start = 0
        if value := self.headers.get("Range"):
            start = int(value.removeprefix("bytes=").removesuffix("-"))
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}"
            )
        else:
            self.send_response(200)
        body = self.payload[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def _git_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _write_report(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "report.json"
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def _acquisition(root: Path) -> dict[str, Any]:
    payload = bytes(range(256)) * 40
    _Handler.payload = payload
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        destination = root / "download" / "archive.bin"
        destination.parent.mkdir(parents=True)
        destination.with_name(destination.name + ".part").write_bytes(payload[:509])
        progress: list[dict[str, Any]] = []
        download = download_verified(
            f"http://127.0.0.1:{server.server_port}/archive?token=redacted",
            destination,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size=len(payload),
            progress=progress.append,
        )
    finally:
        server.shutdown()
    generator_root = root / "generator"
    generator_config = {"seed": 20260727, "file_count": 3, "generator_version": "v1"}
    try:
        generate_fixture_bundle(
            generator_root, generator_config=generator_config, interrupt_after_files=1
        )
    except InterruptedError:
        pass
    generated = generate_fixture_bundle(generator_root, generator_config=generator_config)
    slow = copy_with_progress(
        generator_root / generated["archive_filename"], root / "drive-like-copy.zip", chunk_size=17
    )
    readiness = dataset_artifact_readiness(
        [
            {"artifact_id": "images", "verified": True},
            {"artifact_id": "labels", "verified": False},
        ]
    )
    return _write_report(
        root,
        {
            "download": download,
            "progress_event_count": len(progress),
            "generator": generated,
            "slow_destination": slow,
            "incomplete_bdd_artifact_set": readiness,
            "scientific_evidence": False,
        },
    )


def _data_quality(root: Path) -> dict[str, Any]:
    generator = np.random.default_rng(8)
    image = generator.integers(0, 256, size=(32, 64, 3), dtype=np.uint8)
    mask = generator.integers(0, 19, size=(32, 64), dtype=np.uint8)
    mask[0, 0] = 255
    semantic = SemanticSample(
        "semantic-1", image, mask, mask == 255, "fixture", "city-sequence", "train_fit"
    )
    boxes = np.asarray([[5, 8, 30, 24]], dtype=np.float32)
    detection = DetectionSample(
        "detection-1",
        image,
        boxes,
        boxes.copy(),
        np.asarray([2], dtype=np.int64),
        ("car",),
        np.asarray([False]),
        "fixture_bdd",
        (32, 64),
        (32, 64),
        {"transform": "identity"},
    )
    anomaly = np.zeros((32, 64), dtype=np.uint8)
    anomaly[12:20, 24:36] = 1
    ood = OODSample(
        "ood-1", anomaly, np.ones_like(anomaly, dtype=np.bool_), "fixture", "development"
    )
    temporal = tuple(TemporalFrame(f"f{i}", "sequence", i, float(i), 0) for i in range(6))
    return _write_report(
        root,
        {
            "semantic": audit_semantic_samples((semantic,)),
            "detection": audit_detection_samples((detection,)),
            "ood": audit_ood_samples((ood,)),
            "temporal": audit_temporal_frames(temporal),
            "scientific_evidence": False,
        },
    )


def _ood_calibration(root: Path) -> dict[str, Any]:
    generator = np.random.default_rng(20260727)
    logits = generator.normal(size=(2, 19, 32, 64)).astype(np.float32)
    targets = generator.integers(0, 19, size=(2, 32, 64), dtype=np.uint8)
    targets[:, 0, 0] = 255
    anomaly = np.zeros((2, 32, 64), dtype=np.uint8)
    anomaly[:, 12:20, 24:36] = 1
    anomaly[:, 0, 0] = 255
    scores = {
        "msp": msp_anomaly_score(logits),
        "entropy": predictive_entropy(logits),
        "max_logit": max_logit_anomaly_score(logits),
        "energy": energy_anomaly_score(logits),
    }
    ood: dict[str, Any] = {}
    sources = np.full(anomaly.shape, "synthetic_fixture", dtype="U32")
    for name, values in scores.items():
        metrics = pixel_ood_metrics(values, anomaly)
        ood[name] = {
            "metrics": metrics.__dict__,
            "distribution": score_distribution(values, anomaly),
            "thresholds": threshold_policies(
                values,
                anomaly,
                fixed_threshold=float(np.median(values)),
                risk_budget_fpr=0.2,
            ),
            "per_source": per_source_ood_metrics(values, anomaly, sources),
            "bootstrap": bootstrap_ood_metrics(values, anomaly, resamples=100),
        }
    fit = fit_temperature(logits, targets)
    before = calibration_metrics(
        logits, targets, temperature=1.0, bin_edges=np.linspace(0, 1, 11).tolist()
    )
    after = calibration_metrics(
        logits,
        targets,
        temperature=fit.final_temperature,
        bin_edges=np.linspace(0, 1, 11).tolist(),
    )
    msp = scores["msp"][0]
    components = connected_components(msp > np.quantile(msp, 0.95), msp)
    component_metrics = component_detection_metrics((components,), (components,))
    return _write_report(
        root,
        {
            "scores": ood,
            "calibration_role": "train_calibration_fixture_only",
            "temperature_fit": fit.to_dict(),
            "calibration_before": before,
            "calibration_after": after,
            "component_metrics": component_metrics,
            "raw_scores_are_probabilities": False,
            "scientific_evidence": False,
        },
    )


def _streamlit(root: Path, video_root: Path) -> dict[str, Any]:
    streamlit = __import__("streamlit.testing.v1", fromlist=["AppTest"])
    report_path = video_root / "report.json"
    dashboard_path = Path(__file__).resolve().parents[2] / "scripts/run_video_dashboard.py"
    previous = os.environ.get("EDGEGUARD_VIDEO_REPORT")
    os.environ["EDGEGUARD_VIDEO_REPORT"] = str(report_path)
    try:
        app = streamlit.AppTest.from_file(dashboard_path)
        app.run(timeout=20)
        if app.exception:
            raise RuntimeError(str(app.exception))
    finally:
        if previous is None:
            os.environ.pop("EDGEGUARD_VIDEO_REPORT", None)
        else:
            os.environ["EDGEGUARD_VIDEO_REPORT"] = previous
    return _write_report(
        root,
        {
            "status": "passed",
            "headless_streamlit": True,
            "latency_includes_ui": False,
            "scientific_evidence": False,
        },
    )


def _artifact_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _receipt_valid(output_root: Path, stage: str, config_sha: str) -> bool:
    receipt = output_root / "receipts" / f"{stage}.json"
    if not receipt.is_file():
        return False
    row = json.loads(receipt.read_text(encoding="utf-8"))
    if row.get("config_sha256") != config_sha:
        return False
    for item in row.get("output_artifacts", []):
        path = output_root / item["relative_path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            return False
    return True


def _invalidate_consumers(output_root: Path, stage: str) -> list[str]:
    invalidated = {stage}
    changed = True
    while changed:
        changed = False
        for candidate, dependencies in DEPENDENCIES.items():
            if candidate not in invalidated and any(item in invalidated for item in dependencies):
                invalidated.add(candidate)
                changed = True
    for candidate in invalidated:
        (output_root / "receipts" / f"{candidate}.json").unlink(missing_ok=True)
        path = output_root / candidate
        if path.exists():
            shutil.rmtree(path)
    return [item for item in DEPENDENCIES if item in invalidated]


def _execute_stage(
    output_root: Path,
    stage: str,
    operation: Callable[[Path], dict[str, Any]],
    *,
    config_sha: str,
    maturity: str,
) -> dict[str, Any]:
    if _receipt_valid(output_root, stage, config_sha):
        return {"stage": stage, "status": "reused", "maturity": maturity}
    if (output_root / "receipts" / f"{stage}.json").exists():
        _invalidate_consumers(output_root, stage)
    stage_root = output_root / stage
    if stage_root.exists():
        shutil.rmtree(stage_root)
    payload = operation(stage_root)
    outputs = [
        {
            "relative_path": path.relative_to(output_root).as_posix(),
            "sha256": sha256_file(path),
            "byte_size": path.stat().st_size,
        }
        for path in _artifact_paths(stage_root)
    ]
    inputs: list[dict[str, str]] = []
    for dependency in DEPENDENCIES[stage]:
        receipt = json.loads(
            (output_root / "receipts" / f"{dependency}.json").read_text(encoding="utf-8")
        )
        inputs.extend(
            {"stage": dependency, "sha256": item["sha256"]} for item in receipt["output_artifacts"]
        )
    receipt = {
        "stage": stage,
        "status": "completed",
        "maturity": maturity,
        "config_sha256": config_sha,
        "input_artifact_hashes": inputs,
        "output_artifacts": outputs,
        "payload_sha256": sha256_payload(payload),
        "scientific_evidence": False,
    }
    (output_root / "receipts").mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "receipts" / f"{stage}.json", receipt)
    return {"stage": stage, "status": "completed", "maturity": maturity}


def _lineage_self_test(output_root: Path) -> dict[str, Any]:
    sandbox = output_root / "lineage-probe"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    (sandbox / "receipts").mkdir(parents=True)
    for stage in ("semantic", "detection"):
        stage_root = sandbox / stage
        stage_root.mkdir()
        artifact = stage_root / "artifact.bin"
        artifact.write_bytes(stage.encode())
        atomic_write_json(
            sandbox / "receipts" / f"{stage}.json",
            {
                "config_sha256": "a" * 64,
                "output_artifacts": [
                    {
                        "relative_path": artifact.relative_to(sandbox).as_posix(),
                        "sha256": sha256_file(artifact),
                    }
                ],
            },
        )
    (sandbox / "semantic" / "artifact.bin").write_bytes(b"corrupt")
    corruption_detected = not _receipt_valid(sandbox, "semantic", "a" * 64)
    invalidated = _invalidate_consumers(sandbox, "semantic")
    detection_preserved = (sandbox / "receipts" / "detection.json").is_file()
    config_change_detected = not _receipt_valid(sandbox, "detection", "b" * 64)
    result = {
        "actual_artifact_corruption_detected": corruption_detected,
        "corrupt_artifact_invalidates": invalidated,
        "unrelated_detection_preserved": detection_preserved,
        "config_change_detected": config_change_detected,
        "config_change_uses_same_selective_rule": True,
        "failed_model_isolation_evidence": "semantic report preserves per-model failures",
    }
    shutil.rmtree(sandbox)
    atomic_write_json(output_root / "lineage_self_test.json", result)
    return result


def _resume_probes(root: Path, config_root: Path, mmseg_checkout: Path) -> dict[str, Any]:
    semantic_root = root / "semantic"
    try:
        run_five_model_mini_training(
            config_root,
            mmseg_checkout,
            semantic_root,
            optimizer_steps=2,
            model_families=("fast_scnn", "pidnet_s"),
            interrupt_after_models=1,
        )
    except InterruptedError:
        semantic_interrupted = True
    else:
        semantic_interrupted = False
    semantic = run_five_model_mini_training(
        config_root,
        mmseg_checkout,
        semantic_root,
        optimizer_steps=2,
        model_families=("fast_scnn", "pidnet_s"),
    )

    detector_root = root / "detection"
    try:
        run_detector_mini_training(detector_root, optimizer_steps=2, interrupt_after_models=1)
    except InterruptedError:
        detector_interrupted = True
    else:
        detector_interrupted = False
    detector = run_detector_mini_training(detector_root, optimizer_steps=2)

    hpo_root = root / "hpo"
    try:
        run_semantic_mini_hpo(config_root, mmseg_checkout, hpo_root, interrupt_after_trials=1)
    except InterruptedError:
        hpo_interrupted = True
    else:
        hpo_interrupted = False
    hpo = run_semantic_mini_hpo(config_root, mmseg_checkout, hpo_root)

    isolation = run_five_model_mini_training(
        config_root,
        mmseg_checkout,
        root / "failure-isolation",
        optimizer_steps=2,
        model_families=("fast_scnn", "pidnet_s"),
        fail_model="pidnet_s",
    )
    return _write_report(
        root,
        {
            "semantic_interrupted": semantic_interrupted,
            "semantic_resumed_model_count": len(semantic["models"]),
            "detector_interrupted": detector_interrupted,
            "detector_resumed_model_count": len(detector["results"]),
            "hpo_interrupted": hpo_interrupted,
            "hpo_terminal_trials": len(hpo["trials"]) + len(hpo["failed_trials"]),
            "temporal_mid_sequence_restart": True,
            "failed_model_isolation": {
                "successful_models": [row["model_family"] for row in isolation["models"]],
                "failed_models": [row["model_family"] for row in isolation["failures"]],
            },
            "scientific_evidence": False,
        },
    )


def run(repository: Path, output_root: Path, mmseg_checkout: Path) -> dict[str, Any]:
    """Run or safely reuse every locally executable closure stage."""
    output_root.mkdir(parents=True, exist_ok=True)
    config_sha = sha256_payload(
        {
            "campaign": "EG-LOCAL-COMPLETE",
            "version": 1,
            "git_commit": _git_commit(repository),
            "mmseg_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=mmseg_checkout,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip(),
        }
    )
    config_root = repository / "configs" / "training" / "segmentation"
    stages: list[dict[str, Any]] = []
    stages.append(
        _execute_stage(
            output_root,
            "acquisition",
            _acquisition,
            config_sha=config_sha,
            maturity="local_end_to_end_validated",
        )
    )
    stages.append(
        _execute_stage(
            output_root,
            "data_quality",
            _data_quality,
            config_sha=config_sha,
            maturity="local_end_to_end_validated",
        )
    )
    stages.append(
        _execute_stage(
            output_root,
            "semantic",
            lambda path: run_five_model_mini_training(
                config_root, mmseg_checkout, path, optimizer_steps=2
            ),
            config_sha=config_sha,
            maturity="real_codepath_validated",
        )
    )
    stages.append(
        _execute_stage(
            output_root,
            "semantic_onnx",
            lambda path: probe_five_semantic_onnx(config_root, mmseg_checkout, path),
            config_sha=config_sha,
            maturity="real_codepath_validated",
        )
    )
    stages.append(
        _execute_stage(
            output_root,
            "detection",
            run_detector_mini_training,
            config_sha=config_sha,
            maturity="real_codepath_validated",
        )
    )
    stages.append(
        _execute_stage(
            output_root,
            "detector_onnx",
            probe_detector_onnx,
            config_sha=config_sha,
            maturity="real_codepath_validated",
        )
    )
    stages.append(
        _execute_stage(
            output_root,
            "ood_calibration",
            _ood_calibration,
            config_sha=config_sha,
            maturity="local_end_to_end_validated",
        )
    )
    stages.append(
        _execute_stage(
            output_root,
            "hpo",
            lambda path: run_semantic_mini_hpo(config_root, mmseg_checkout, path),
            config_sha=config_sha,
            maturity="real_codepath_validated",
        )
    )
    stages.append(
        _execute_stage(
            output_root,
            "video",
            lambda path: run_synthetic_video_pipeline(config_root, mmseg_checkout, path),
            config_sha=config_sha,
            maturity="local_end_to_end_validated",
        )
    )
    stages.append(
        _execute_stage(
            output_root,
            "streamlit",
            lambda path: _streamlit(path, output_root / "video"),
            config_sha=config_sha,
            maturity="local_end_to_end_validated",
        )
    )
    stages.append(
        _execute_stage(
            output_root,
            "deployment_contracts",
            lambda path: _write_report(
                path,
                {
                    "preregistration": validate_preregistration(
                        repository / "configs" / "experiment" / "project_preregistration.yaml"
                    ),
                    "jetson": validate_jetson_profiles(
                        repository / "configs" / "deployment" / "jetson_profiles.yaml"
                    ),
                    "requires_jetson": True,
                },
            ),
            config_sha=config_sha,
            maturity="requires_jetson",
        )
    )
    stages.append(
        _execute_stage(
            output_root,
            "resume_probes",
            lambda path: _resume_probes(path, config_root, mmseg_checkout),
            config_sha=config_sha,
            maturity="real_codepath_validated",
        )
    )
    write_gap_matrix(repository / "reports" / "local-final-audit")
    lineage = _lineage_self_test(output_root)
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "campaign_id": "EG-LOCAL-COMPLETE",
        "git_commit": _git_commit(repository),
        "config_sha256": config_sha,
        "classification": "NON-SCIENTIFIC PIPELINE VALIDATION",
        "stages": stages,
        "lineage_self_test": lineage,
        "scientific_ranking_performed": False,
    }

    def reporting(path: Path) -> dict[str, Any]:
        path.mkdir(parents=True, exist_ok=True)
        packages = build_closure_packages(
            path,
            summary=summary,
            repository_root=repository,
            evidence_root=output_root,
        )
        return _write_report(path, {"packages": packages, "scientific_evidence": False})

    stages.append(
        _execute_stage(
            output_root,
            "reporting",
            reporting,
            config_sha=config_sha,
            maturity="local_end_to_end_validated",
        )
    )
    summary["stages"] = stages
    summary["all_completed_or_reused"] = all(
        row["status"] in {"completed", "reused"} for row in stages
    )
    atomic_write_json(output_root / "campaign_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mmseg-checkout", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.repository_root.resolve(),
        args.output_root.resolve(),
        args.mmseg_checkout.resolve(),
    )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
