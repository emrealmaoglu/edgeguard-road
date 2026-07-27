"""Generate small non-scientific OOD/calibration evidence from project-owned data."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

from edgeguard.calibration import apply_temperature, calibration_metrics, fit_temperature
from edgeguard.data.cityscapes_train import load_train_ids
from edgeguard.scoring.uncertainty import (
    energy_anomaly_score,
    max_logit_anomaly_score,
    msp_anomaly_score,
    predictive_entropy,
    uncertainty_score_contract,
)
from edgeguard.serialization import canonical_json, sha256_array, sha256_file
from edgeguard.telemetry.longrun import atomic_write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dev.validate_local_readiness import (  # noqa: E402
    create_mini_cityscapes_fixture,
)

SEED = 20260727
BIN_EDGES = tuple(index / 10.0 for index in range(11))


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _synthetic_logits(targets: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    logits = rng.normal(0.0, 1.0, size=(targets.shape[0], 19, *targets.shape[1:])).astype(
        np.float32
    )
    for batch_index, target in enumerate(targets):
        rows, columns = np.where(target != 255)
        logits[batch_index, target[rows, columns], rows, columns] += np.float32(0.75)
    return np.asarray(logits * np.float32(3.0), dtype=np.float32)


def _statistics(score: np.ndarray) -> dict[str, Any]:
    return {
        "shape": [int(dimension) for dimension in score.shape],
        "dtype": str(score.dtype),
        "finite": bool(np.isfinite(score).all()),
        "minimum": float(np.min(score)),
        "maximum": float(np.max(score)),
        "mean": float(np.mean(score, dtype=np.float64)),
        "direction": "higher_means_more_anomalous",
        "anomaly_probability": False,
    }


def _package(output_dir: Path, names: tuple[str, ...]) -> Path:
    destination = output_dir / "ood-calibration-evidence.zip"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(names):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (output_dir / name).read_bytes())
    return destination


def generate_evidence(output_dir: Path) -> dict[str, Any]:
    """Create path-free synthetic scoring and calibration records plus one ZIP."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("OOD/calibration evidence output must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="edgeguard-ood-calibration-") as temporary:
        fixture_root = Path(temporary) / "fixture"
        dataset_manifest_path, group_summary_path = create_mini_cityscapes_fixture(fixture_root)
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        selected = sorted(dataset_manifest["samples"], key=lambda item: item["sample_id"])[:6]
        targets = np.stack(
            [load_train_ids(fixture_root / item["train_id_relative_path"]) for item in selected]
        ).astype(np.int64, copy=False)
        logits = _synthetic_logits(targets)
        original_logits = logits.copy()
        fit = fit_temperature(logits, targets, max_iterations=64)
        calibrated_logits = apply_temperature(logits, fit.final_temperature)
        before = calibration_metrics(
            logits,
            targets,
            temperature=1.0,
            bin_edges=BIN_EDGES,
        )
        after = calibration_metrics(
            logits,
            targets,
            temperature=fit.final_temperature,
            bin_edges=BIN_EDGES,
        )
        scores = {
            "msp": msp_anomaly_score(logits),
            "predictive_entropy": predictive_entropy(logits),
            "max_logit": max_logit_anomaly_score(logits),
            "energy_t1": energy_anomaly_score(logits, temperature=1.0),
        }
        scoring_contract = uncertainty_score_contract()
        calibration_record = {
            "schema_version": "1.0",
            "record_type": "synthetic_calibration_before_after",
            "fit": fit.to_dict(),
            "before": before,
            "after": after,
            "input_logits_preserved": bool(np.array_equal(logits, original_logits)),
            "calibrated_logits_sha256": sha256_array(calibrated_logits),
            "scientific_evidence": False,
        }
        score_statistics = {
            "schema_version": "1.0",
            "record_type": "synthetic_uncertainty_score_statistics",
            "score_grid": "synthetic_logits",
            "statistics": {name: _statistics(score) for name, score in scores.items()},
            "cross_method_normalization": "none",
            "scientific_evidence": False,
        }
        test_identity = {
            "schema_version": "1.0",
            "record_type": "ood_calibration_test_identity",
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain=v1")),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "device": "cpu",
            "fixture_name": "edgeguard_project_owned_mini_cityscapes",
            "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
            "group_summary_sha256": sha256_file(group_summary_path),
            "sample_ids": [item["sample_id"] for item in selected],
            "synthetic_seed": SEED,
            "logits_sha256": sha256_array(logits),
            "targets_sha256": sha256_array(targets),
            "scientific_evidence": False,
        }
    records = {
        "scoring_contract.json": scoring_contract,
        "calibration_before_after.json": calibration_record,
        "synthetic_score_statistics.json": score_statistics,
        "test_identity.json": test_identity,
    }
    for name, payload in records.items():
        atomic_write_json(output_dir / name, payload)
    package = _package(output_dir, tuple(records))
    completion = {
        "schema_version": "1.0",
        "record_type": "ood_calibration_evidence_completion",
        "status": "passed",
        "scientific_evidence": False,
        "evidence_package": package.name,
        "evidence_package_sha256": sha256_file(package),
        "files": {name: sha256_file(output_dir / name) for name in sorted(records)},
    }
    print(canonical_json(completion))
    return completion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate_evidence(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
