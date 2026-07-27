"""Tests for deterministic group-atomic learning-curve fractions."""

from __future__ import annotations

from edgeguard.training.contracts import SemanticTrainingSample
from edgeguard.training.fractions import build_train_fit_fraction


def _sample(group: int, frame: int) -> SemanticTrainingSample:
    group_id = f"alpha_{group:06d}"
    sample_id = f"{group_id}_{frame:06d}"
    return SemanticTrainingSample(
        sample_id=sample_id,
        group_id=group_id,
        role="train_fit",
        image_relative_path=f"leftImg8bit/train/alpha/{sample_id}_leftImg8bit.png",
        train_id_relative_path=f"gtFine/train/trainIds/alpha/{sample_id}_gtFine_trainIds.png",
    )


def test_fraction_is_root_free_deterministic_and_group_atomic() -> None:
    samples = tuple(_sample(group, frame) for group in range(8) for frame in range(group % 3 + 1))
    first = build_train_fit_fraction(samples, 0.5, seed=1337, split_manifest_sha256="a" * 64)
    second = build_train_fit_fraction(
        tuple(reversed(samples)), 0.5, seed=1337, split_manifest_sha256="a" * 64
    )

    assert first == second
    assert first["manifest_sha256"] == second["manifest_sha256"]
    selected = set(first["selected_group_ids"])
    assert {
        sample.group_id for sample in samples if sample.sample_id in first["selected_sample_ids"]
    } == selected
    assert "/Users/" not in str(first) and "/content/" not in str(first)


def test_full_fraction_contains_every_train_fit_group() -> None:
    samples = tuple(_sample(group, 0) for group in range(4))
    manifest = build_train_fit_fraction(samples, 1.0, seed=1, split_manifest_sha256="b" * 64)

    assert manifest["selected_sample_count"] == len(samples)
    assert manifest["selected_group_count"] == 4
