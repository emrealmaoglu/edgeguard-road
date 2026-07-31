from __future__ import annotations

import pytest

from edgeguard.rescue.shift import frame_shift_metrics


def test_frame_shift_metrics_report_external_separation_and_alert_rates() -> None:
    source = [
        {
            "mean_normalized_entropy": 0.1,
            "low_confidence_pixel_ratio": 0.1,
            "negative_mean_maximum_logit": -4.0,
            "mean_energy": -5.0,
        },
        {
            "mean_normalized_entropy": 0.2,
            "low_confidence_pixel_ratio": 0.2,
            "negative_mean_maximum_logit": -3.0,
            "mean_energy": -4.0,
        },
    ]
    external = [
        {
            "mean_normalized_entropy": 0.8,
            "low_confidence_pixel_ratio": 0.7,
            "negative_mean_maximum_logit": -1.0,
            "mean_energy": -1.0,
        },
        {
            "mean_normalized_entropy": 0.9,
            "low_confidence_pixel_ratio": 0.8,
            "negative_mean_maximum_logit": 0.0,
            "mean_energy": 0.0,
        },
    ]

    result = frame_shift_metrics(
        source,
        external,
        source_alerts=[False, False],
        external_alerts=[True, True],
    )

    assert result["scores"]["mean_normalized_entropy"]["auroc"] == pytest.approx(1.0)
    assert result["scores"]["mean_energy"]["average_precision"] == pytest.approx(1.0)
    assert result["scores"]["negative_mean_maximum_logit"]["auroc"] == pytest.approx(1.0)
    assert result["source_alert_rate"] == pytest.approx(0.0)
    assert result["external_alert_rate"] == pytest.approx(1.0)
    assert result["external_threshold_tuning"] is False


def test_frame_shift_metrics_marks_optional_energy_as_missing() -> None:
    result = frame_shift_metrics(
        [{"mean_normalized_entropy": 0.1, "low_confidence_pixel_ratio": 0.1}],
        [{"mean_normalized_entropy": 0.9, "low_confidence_pixel_ratio": 0.9}],
    )

    assert result["scores"]["mean_energy"]["reason"] == "score_missing"
