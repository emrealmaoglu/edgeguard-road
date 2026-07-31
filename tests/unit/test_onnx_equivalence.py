"""Regression tests for fixed-threshold layer-aware ONNX classification."""

from __future__ import annotations

import numpy as np

from edgeguard.export.equivalence import (
    classify_pidnet,
    classify_rtdetr,
    tensor_error_statistics,
)


def _stat(name: str, *, close: bool) -> dict[str, object]:
    return {"tensor_name": name, "allclose": close}


def test_tensor_error_statistics_reports_percentiles_and_affected_values() -> None:
    expected = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    actual = np.array([0.0, 1.0, 2.001, 3.0], dtype=np.float32)
    result = tensor_error_statistics(expected, actual, tensor_name="native_logits")

    assert result["tensor_name"] == "native_logits"
    assert result["allclose"] is False
    assert result["affected_element_count"] == 1
    assert result["absolute_error"]["maximum"] > 0.0009
    assert result["absolute_error"]["p95"] > 0.0
    assert result["relative_error"]["denominator"] == "max(abs(pytorch),1e-08)"


def test_pidnet_classification_does_not_relax_tolerance() -> None:
    drift = [_stat("backbone", close=False), _stat("native", close=False)]
    assert classify_pidnet(drift, argmax_pixel_agreement=1.0) == "bounded_documented_drift"
    assert (
        classify_pidnet(drift, argmax_pixel_agreement=0.99)
        == "requires_cuda_or_tensorrt_validation"
    )


def test_rtdetr_classification_isolates_random_weight_topk_tie() -> None:
    stats = {
        "encoder_class": _stat("encoder_class", close=True),
        "encoder_coord_logits": _stat("encoder_coord_logits", close=True),
        "topk_logits": _stat("topk_logits", close=True),
        "topk_boxes": _stat("topk_boxes", close=False),
        "final_logits": _stat("final_logits", close=False),
    }
    assert classify_rtdetr(stats, topk_cutoff_tie_count=320) == "bounded_documented_drift"
    assert classify_rtdetr(stats, topk_cutoff_tie_count=1) == "requires_cuda_or_tensorrt_validation"
