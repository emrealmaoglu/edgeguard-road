"""Tests for minimal Cityscapes evaluation runner safeguards."""

from pathlib import Path

import numpy as np
import pytest

from edgeguard.data.cityscapes import CityscapesValSample
from edgeguard.evaluation.cityscapes_runner import (
    _NumericSummary,
    _prepare_output_dir,
    _select_samples,
    run_cityscapes_evaluation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sample(city: str, frame: int) -> CityscapesValSample:
    sample_id = f"{city}_000000_{frame:06d}"
    return CityscapesValSample(
        sample_id=sample_id,
        city=city,
        sequence="000000",
        frame=f"{frame:06d}",
        image_relative_path=f"leftImg8bit/val/{city}/{sample_id}_leftImg8bit.png",
        label_relative_path=f"gtFine/val/{city}/{sample_id}_gtFine_labelIds.png",
    )


def test_runner_rejects_non_empty_output_collision(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty"):
        _prepare_output_dir(output)


def test_runner_rejects_missing_dataset_root_before_model_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset root does not exist"):
        run_cityscapes_evaluation(
            config_path=REPO_ROOT / "configs/cityscapes_eval_local.yaml",
            dataset_root=tmp_path / "missing-dataset",
            checkpoint_path=tmp_path / "missing-checkpoint.pt",
            upstream_checkout=tmp_path / "missing-checkout",
            subset_size=1,
            subset_manifest_path=None,
            select_all=False,
            device="cpu",
            output_dir=tmp_path / "output",
        )


def test_runner_selection_manifest_is_deterministic_and_path_free() -> None:
    samples = [_sample("munster", 1), _sample("frankfurt", 1), _sample("lindau", 1)]

    first_samples, first_manifest = _select_samples(
        samples,
        subset_size=2,
        subset_manifest_path=None,
        select_all=False,
        strategy="city_round_robin_v1",
        config_sha256_value="a" * 64,
        checkpoint_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
    )
    second_samples, second_manifest = _select_samples(
        list(reversed(samples)),
        subset_size=2,
        subset_manifest_path=None,
        select_all=False,
        strategy="city_round_robin_v1",
        config_sha256_value="a" * 64,
        checkpoint_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
    )

    assert [sample.sample_id for sample in first_samples] == [
        "frankfurt_000000_000001",
        "lindau_000000_000001",
    ]
    assert first_samples == second_samples
    assert first_manifest == second_manifest
    assert "/Users/" not in str(first_manifest)


def test_numeric_summary_reports_only_min_max_mean() -> None:
    summary = _NumericSummary()

    summary.update(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    assert summary.result() == {"count": 4, "min": 1.0, "max": 4.0, "mean": 2.5}
