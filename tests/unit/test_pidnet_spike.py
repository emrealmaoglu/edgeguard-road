"""Unit tests for fixed-source PIDNet spike safeguards."""

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

from edgeguard.models.pidnet_spike import (
    PIDNetSpikeError,
    difference_summary,
    normalize_state_dict_keys,
    preprocess_pidnet_rgb,
    validate_state_dict_shapes,
    verify_checkpoint_file,
    verify_upstream_checkout,
)

UPSTREAM_URL = "https://github.com/XuJiacong/PIDNet.git"


def test_preprocess_pidnet_rgb_preserves_rgb_channel_order() -> None:
    raw = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)

    model_input = preprocess_pidnet_rgb(
        raw,
        height=1,
        width=2,
        pixel_scale=1.0 / 255.0,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    assert model_input.shape == (1, 3, 1, 2)
    np.testing.assert_allclose(model_input[0, :, 0, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(model_input[0, :, 0, 1], [0.0, 0.0, 1.0])


def test_checkpoint_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "PIDNet_S_Cityscapes_val.pt"
    checkpoint.write_bytes(b"checkpoint fixture")

    with pytest.raises(PIDNetSpikeError, match="SHA-256 mismatch"):
        verify_checkpoint_file(
            checkpoint,
            expected_filename=checkpoint.name,
            expected_sha256="0" * 64,
        )


def test_checkpoint_hash_and_filename_are_recorded(tmp_path: Path) -> None:
    checkpoint = tmp_path / "PIDNet_S_Cityscapes_val.pt"
    content = b"checkpoint fixture"
    checkpoint.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest()

    report = verify_checkpoint_file(
        checkpoint,
        expected_filename=checkpoint.name,
        expected_sha256=expected_hash,
    )

    assert report == {
        "filename": checkpoint.name,
        "sha256": expected_hash,
        "size_bytes": len(content),
    }


def test_state_dict_allows_only_exact_or_uniform_known_prefix() -> None:
    model_keys = {"stem.weight", "head.bias"}
    exact, exact_transform = normalize_state_dict_keys(
        {"stem.weight": 1, "head.bias": 2}, model_keys
    )
    prefixed, prefixed_transform = normalize_state_dict_keys(
        {"model.stem.weight": 1, "model.head.bias": 2}, model_keys
    )

    assert set(exact) == model_keys
    assert exact_transform == "none"
    assert set(prefixed) == model_keys
    assert prefixed_transform == "removed_uniform_prefix:model."


def test_state_dict_mismatch_is_rejected_without_partial_filtering() -> None:
    with pytest.raises(PIDNetSpikeError, match="refusing partial load"):
        normalize_state_dict_keys({"model.stem.weight": 1}, {"stem.weight", "head.bias"})


def test_state_dict_shape_mismatch_is_rejected() -> None:
    model_state = {"stem.weight": np.zeros((2, 3), dtype=np.float32)}

    with pytest.raises(PIDNetSpikeError, match="parameter shape mismatch"):
        validate_state_dict_shapes({"stem.weight": np.zeros((3, 2), dtype=np.float32)}, model_state)

    manifest = validate_state_dict_shapes(
        {"stem.weight": np.zeros((2, 3), dtype=np.float32)}, model_state
    )
    assert manifest == {"stem.weight": [2, 3]}


def test_upstream_checkout_requires_exact_clean_origin_and_commit(tmp_path: Path) -> None:
    checkout = tmp_path / "pidnet"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    (checkout / "source.txt").write_text("fixed source\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "source.txt"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "fixture"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "add", "origin", UPSTREAM_URL], check=True
    )
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    report = verify_upstream_checkout(
        checkout,
        expected_repository_url=UPSTREAM_URL,
        expected_commit=commit,
    )

    assert report["actual_commit"] == commit
    assert report["git_clean"] is True
    with pytest.raises(PIDNetSpikeError, match="commit mismatch"):
        verify_upstream_checkout(
            checkout,
            expected_repository_url=UPSTREAM_URL,
            expected_commit="0" * 40,
        )


def test_difference_summary_reports_exact_and_changed_arrays() -> None:
    first = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    exact = difference_summary(first, first.copy())
    changed = difference_summary(first, np.array([0.0, 2.0, 2.0], dtype=np.float32))

    assert exact["byte_equal"] is True
    assert exact["max_absolute_difference"] == 0.0
    assert changed["byte_equal"] is False
    assert changed["max_absolute_difference"] == 1.0
