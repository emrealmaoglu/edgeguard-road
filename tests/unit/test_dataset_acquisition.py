"""Tests for the project-specific resumable acquisition queue."""

from __future__ import annotations

from pathlib import Path

from scripts.acquire_edgeguard_datasets import acquire_entry, load_queue

REPO_ROOT = Path(__file__).resolve().parents[2]
QUEUE = REPO_ROOT / "configs/dataset/acquisition_queue.yaml"


def test_queue_covers_only_approved_development_acquisitions() -> None:
    payload = load_queue(QUEUE)
    identifiers = {entry["dataset_id"] for entry in payload["entries"]}

    assert identifiers == {
        "bdd100k_detection",
        "fishyscapes_static",
        "cityscapes_coarse_trainextra",
        "temporal_selected",
        "demo_videos",
    }
    serialized_entries = str(payload["entries"])
    assert "RoadAnomaly21" not in serialized_entries
    assert "RoadObstacle21" not in serialized_entries
    assert "dataset_id': 'fishyscapes_lost" not in serialized_entries
    assert "SMIYC RoadAnomaly21" in payload["sealed_exclusions"]


def test_missing_runtime_access_is_a_structured_block_not_a_download(tmp_path: Path) -> None:
    entry = load_queue(QUEUE)["entries"][0]
    result = acquire_entry(entry, tmp_path)

    assert result["status"] == "blocked_access"
    assert result["human_action"]
    assert result["missing_runtime_variables"]


def test_temporal_entry_stays_blocked_even_without_runtime_access(tmp_path: Path) -> None:
    entries = load_queue(QUEUE)["entries"]
    temporal = next(entry for entry in entries if entry["dataset_id"] == "temporal_selected")

    result = acquire_entry(temporal, tmp_path)

    assert result == {
        "dataset_id": "temporal_selected",
        "status": "blocked_policy",
        "human_action": temporal["human_action"],
    }
