"""Opt-in validation for the human-approved real PIDNet-S checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from edgeguard.config import load_pidnet_spike_config
from edgeguard.models.pidnet_spike import verify_pidnet_checkpoint_layout

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = os.environ.get("EDGEGUARD_PIDNET_CHECKPOINT")
CHECKOUT = os.environ.get("EDGEGUARD_PIDNET_CHECKOUT")


@pytest.mark.skipif(
    not CHECKPOINT or not CHECKOUT,
    reason="real PIDNet checkpoint/check-out paths were not supplied",
)
def test_real_pidnet_checkpoint_matches_reviewed_layout() -> None:
    config = load_pidnet_spike_config(REPO_ROOT / "configs/pidnet_spike.yaml")

    report = verify_pidnet_checkpoint_layout(
        checkout=Path(CHECKOUT or ""),
        checkpoint_path=Path(CHECKPOINT or ""),
        expected_checkpoint_sha256=config.checkpoint.sha256,
        config=config,
    )

    assert report["raw_checkpoint_key_count"] == 481
    assert report["loaded_inference_key_count"] == 453
    assert report["excluded_auxiliary_group_counts"] == {
        "seghead_d.": 13,
        "seghead_p.": 13,
    }
    assert report["ignored_training_root_keys"] == [
        "sb_loss.criterion.weight",
        "sem_loss.criterion.weight",
    ]
    assert report["weights_only"] is True
    assert report["strict"] is True
    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []
