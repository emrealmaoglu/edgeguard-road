from __future__ import annotations

import numpy as np
import pytest

from edgeguard.rescue.perception import (
    attention_contract,
    derive_perception,
    drivable_corridor_from_semantics,
)
from edgeguard.rescue.shift import (
    apply_shift_reference,
    fit_shift_reference,
    frame_uncertainty_summary,
    uncertainty_maps,
)


def _scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.full((12, 16), 2, dtype=np.uint8)
    mask[6:, 3:13] = 0
    mask[0:2, 0:2] = 0  # disconnected road false positive
    mask[7:11, 7:10] = 13
    mask[5:8, 4:6] = 11
    confidence = np.full(mask.shape, 0.9, dtype=np.float32)
    entropy = np.full(mask.shape, 0.1, dtype=np.float32)
    entropy[5:8, 4:6] = 0.8
    confidence[5:8, 4:6] = 0.3
    return mask, confidence, entropy


def test_drivable_corridor_selects_only_bottom_center_road_component() -> None:
    mask, _confidence, _entropy = _scene()
    road, corridor = drivable_corridor_from_semantics(mask, minimum_area=4)
    assert road[0, 0]
    assert not corridor[0, 0]
    assert corridor[-1, mask.shape[1] // 2]
    assert np.count_nonzero(corridor) < np.count_nonzero(road)


def test_perception_regions_have_fixed_explainable_attention() -> None:
    mask, confidence, entropy = _scene()
    result = derive_perception(
        mask,
        confidence,
        entropy,
        confidence_threshold=0.5,
        entropy_threshold=0.5,
        minimum_region_area=2,
        minimum_drivable_area=4,
    )
    assert {region.class_name for region in result.regions} == {"person", "car"}
    person = next(region for region in result.regions if region.class_name == "person")
    assert person.attention_score == pytest.approx(sum(person.contributions.values()))
    assert person.contributions["class_importance"] == pytest.approx(0.35)
    assert person.attention_level in {"medium", "high"}
    assert result.unreliable_mask[5, 4]
    assert not result.unreliable_mask[0, 3]
    assert result.attention_map[5, 4] == pytest.approx(person.attention_score)
    assert person.to_dict()["instance_detection"] is False
    assert attention_contract()["physical_risk_probability"] is False


def test_uncertainty_maps_and_frame_summary_are_finite() -> None:
    logits = np.zeros((19, 2, 3), dtype=np.float32)
    logits[0] = 2.0
    maps = uncertainty_maps(logits)
    assert set(maps) == {
        "maximum_softmax_probability",
        "normalized_entropy",
        "maximum_logit",
        "energy",
    }
    assert all(np.isfinite(values).all() for values in maps.values())
    summary = frame_uncertainty_summary(
        maps["maximum_softmax_probability"],
        maps["normalized_entropy"],
        energy=maps["energy"],
    )
    assert summary["mean_energy"] is not None


def test_shift_reference_is_source_only_hash_bound_and_two_vote() -> None:
    rows = [
        {
            "mean_normalized_entropy": 0.1 + index * 0.01,
            "low_confidence_pixel_ratio": 0.05 + index * 0.01,
            "mean_energy": -3.0 + index * 0.1,
        }
        for index in range(10)
    ]
    reference = fit_shift_reference(
        rows,
        checkpoint_sha256="a" * 64,
        dataset_manifest_sha256s={
            "cityscapes": "1" * 64,
            "bdd100k": "2" * 64,
            "idd20k": "3" * 64,
        },
    )
    result = apply_shift_reference(
        {
            "mean_normalized_entropy": 0.9,
            "low_confidence_pixel_ratio": 0.8,
            "mean_energy": -5.0,
        },
        reference,
    )
    assert result["domain_shift_alert"] is True
    assert result["alert_votes"] == 2
    assert reference["external_data_used"] is False

    reference["quantile"] = 0.9
    with pytest.raises(ValueError, match="hash mismatch"):
        apply_shift_reference(rows[0], reference)
