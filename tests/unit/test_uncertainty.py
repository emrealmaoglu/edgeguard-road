"""Tests for MSP and predictive entropy on raw semantic logits."""

import numpy as np

from edgeguard.scoring.uncertainty import (
    msp_anomaly_score,
    predictive_entropy,
    semantic_mask,
    stable_softmax,
)


def test_equal_logits_have_expected_msp_and_entropy() -> None:
    logits = np.zeros((1, 4, 2, 3), dtype=np.float32)

    probabilities = stable_softmax(logits)
    msp = msp_anomaly_score(logits)
    entropy = predictive_entropy(logits)

    np.testing.assert_allclose(probabilities, 0.25, atol=1.0e-7)
    np.testing.assert_allclose(msp, 0.75, atol=1.0e-7)
    np.testing.assert_allclose(entropy, np.log(4.0), atol=1.0e-6)


def test_dominant_class_reduces_both_anomaly_scores() -> None:
    equal = np.zeros((1, 3, 1, 1), dtype=np.float32)
    dominant = equal.copy()
    dominant[:, 2] = 20.0

    assert float(msp_anomaly_score(dominant)[0, 0, 0]) < float(msp_anomaly_score(equal)[0, 0, 0])
    assert float(predictive_entropy(dominant)[0, 0, 0]) < float(predictive_entropy(equal)[0, 0, 0])
    assert semantic_mask(dominant).item() == 2


def test_extreme_finite_logits_produce_finite_scores() -> None:
    logits = np.array([[[[1.0e30]], [[-1.0e30]], [[0.0]]]], dtype=np.float32)

    msp = msp_anomaly_score(logits)
    entropy = predictive_entropy(logits)

    assert msp.dtype == np.float32
    assert entropy.dtype == np.float32
    assert np.isfinite(msp).all()
    assert np.isfinite(entropy).all()
    assert 0.0 <= float(msp.item()) <= 1.0
    assert float(entropy.item()) >= 0.0
