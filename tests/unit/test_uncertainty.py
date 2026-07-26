"""Tests for MSP and predictive entropy on raw semantic logits."""

import numpy as np
import pytest

from edgeguard.scoring.uncertainty import (
    energy_anomaly_score,
    max_logit_anomaly_score,
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


def test_maxlogit_and_energy_are_finite_with_expected_shape() -> None:
    logits = np.array(
        [[[[1000.0, -1000.0]], [[999.0, -999.0]], [[998.0, -998.0]]]],
        dtype=np.float32,
    )

    maxlogit = max_logit_anomaly_score(logits)
    energy = energy_anomaly_score(logits)

    assert maxlogit.shape == (1, 1, 2)
    assert energy.shape == (1, 1, 2)
    assert maxlogit.dtype == np.float32
    assert energy.dtype == np.float32
    assert np.isfinite(maxlogit).all()
    assert np.isfinite(energy).all()


def test_logit_shift_behavior_distinguishes_probability_and_logit_scores() -> None:
    logits = np.array([[[[1.0]], [[0.0]], [[-1.0]]]], dtype=np.float32)
    shifted = logits + np.float32(5.0)

    np.testing.assert_allclose(msp_anomaly_score(logits), msp_anomaly_score(shifted))
    np.testing.assert_allclose(predictive_entropy(logits), predictive_entropy(shifted))
    np.testing.assert_allclose(
        max_logit_anomaly_score(shifted),
        max_logit_anomaly_score(logits) - np.float32(5.0),
    )
    np.testing.assert_allclose(
        energy_anomaly_score(shifted),
        energy_anomaly_score(logits) - np.float32(5.0),
        rtol=1e-6,
    )


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_energy_rejects_invalid_temperature(temperature: float) -> None:
    logits = np.zeros((1, 2, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="positive finite"):
        energy_anomaly_score(logits, temperature=temperature)
