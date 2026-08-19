"""Tests for accepted-only thesis evidence collection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgeguard.reporting.thesis_bundle import build_thesis_bundle
from edgeguard.serialization import sha256_file


def _release(tmp_path: Path, *, status: str = "accepted") -> Path:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    metrics = evidence / "metrics.csv"
    metrics.write_text("model,mIoU\nsegformer_b0,0.61\n", encoding="utf-8")
    environment = evidence / "environment.json"
    environment.write_text('{"python":"3.11.13","torch":"2.1.1+cu121"}\n', encoding="utf-8")
    manifest = tmp_path / "accepted_release.json"
    manifest.write_text(
        json.dumps(
            {
                "record_type": "edgeguard_accepted_release",
                "release_id": "semantic-cs-idd-v2-rc1",
                "status": status,
                "artifacts": [
                    {
                        "path": "evidence/metrics.csv",
                        "sha256": sha256_file(metrics),
                        "run_id": "screening-segformer",
                        "claim_ids": ["claim-model-comparison"],
                        "coverage": "miou_latency_pareto",
                        "scientific_status": "accepted",
                    },
                    {
                        "path": "evidence/environment.json",
                        "sha256": sha256_file(environment),
                        "run_id": "screening-segformer",
                        "claim_ids": ["claim-method-runtime"],
                        "scientific_status": "accepted",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_thesis_bundle_keeps_sources_latex_and_claim_hashes(tmp_path: Path) -> None:
    manifest = _release(tmp_path)
    output = tmp_path / "thesis_bundle"
    result = build_thesis_bundle(manifest, output)

    assert result["scientific_status"] == "accepted"
    assert (output / "sources/evidence/metrics.csv").is_file()
    assert (output / "latex/evidence/metrics.tex").is_file()
    assert "screening-segformer" in (output / "thesis_index.md").read_text()
    assert result["coverage"]["miou_latency_pareto"] == "accepted"
    assert result["coverage"]["jetson_latency_power_thermal"] == "not_run"


def test_thesis_bundle_rejects_unaccepted_release(tmp_path: Path) -> None:
    manifest = _release(tmp_path, status="measured")
    with pytest.raises(ValueError, match="accepted release"):
        build_thesis_bundle(manifest, tmp_path / "thesis_bundle")
