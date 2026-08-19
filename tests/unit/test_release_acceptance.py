"""Tests for human-gated Colab release promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgeguard.rescue.release_acceptance import accept_release_candidate
from edgeguard.serialization import sha256_file


def _candidate(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "runs/final/segformer_b0/ce/iter_40000.pth"
    resolved = checkpoint.parent / "resolved.py"
    summary = checkpoint.parent / "summary.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    resolved.write_text("model = {}\n", encoding="utf-8")
    summary.write_text('{"status":"measured"}\n', encoding="utf-8")
    records = [
        (checkpoint, "final_checkpoint"),
        (resolved, "resolved_config"),
        (summary, "training_summary"),
    ]
    artifacts = [
        {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": sha256_file(path),
            "run_id": "semantic-cs-idd-v2-final-segformer_b0-ce",
            "kind": kind,
            "scientific_status": "measured",
        }
        for path, kind in records
    ]
    candidate = tmp_path / "accepted_release.candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "record_type": "edgeguard_release_candidate",
                "status": "candidate_requires_human_acceptance",
                "campaign_id": "semantic-cs-idd-v2",
                "project_commit": "a" * 40,
                "data_manifest_sha256": {"city.json": "b" * 64, "idd.json": "c" * 64},
                "models": [
                    {
                        "model": "segformer_b0",
                        "checkpoint": {
                            "path": artifacts[0]["path"],
                            "sha256": artifacts[0]["sha256"],
                        },
                        "resolved_config": {
                            "path": artifacts[1]["path"],
                            "sha256": artifacts[1]["sha256"],
                        },
                    }
                ],
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return candidate


def _receipt(tmp_path: Path, candidate: Path, *, approved: bool = True) -> Path:
    receipt = tmp_path / "release.review.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "record_type": "edgeguard_release_review_receipt",
                "decision": "accept_release",
                "human_approved": approved,
                "reviewer": "fixture-human-reviewer",
                "release_id": "semantic-cs-idd-v2-rc1",
                "campaign_id": "semantic-cs-idd-v2",
                "project_commit": "a" * 40,
                "candidate_release_sha256": sha256_file(candidate),
            }
        ),
        encoding="utf-8",
    )
    return receipt


def test_release_promotion_is_human_and_hash_gated(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    receipt = _receipt(tmp_path, candidate)
    output = tmp_path / "accepted_release.json"
    accepted = accept_release_candidate(candidate, receipt, output)
    assert accepted["status"] == "accepted"
    assert accepted["release_id"] == "semantic-cs-idd-v2-rc1"
    assert all(item["scientific_status"] == "accepted" for item in accepted["artifacts"])


def test_release_promotion_rejects_denial_and_candidate_tamper(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    denied = _receipt(tmp_path, candidate, approved=False)
    with pytest.raises(PermissionError, match="does not approve"):
        accept_release_candidate(candidate, denied, tmp_path / "accepted_release.json")

    receipt = _receipt(tmp_path, candidate)
    candidate.write_text(candidate.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate hash mismatch"):
        accept_release_candidate(candidate, receipt, tmp_path / "accepted_release.json")
