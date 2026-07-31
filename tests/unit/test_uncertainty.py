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
    uncertainty_score_contract,
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


@pytest.mark.parametrize(
    "logits, expected",
    [
        (np.zeros((2, 3, 4), dtype=np.float32), "rank 4"),
        (np.zeros((1, 0, 2, 2), dtype=np.float32), "positive"),
        (np.full((1, 2, 1, 1), np.nan, dtype=np.float32), "finite"),
        (np.full((1, 2, 1, 1), np.inf, dtype=np.float32), "finite"),
    ],
)
def test_scoring_rejects_invalid_layout_and_nonfinite_values(
    logits: np.ndarray, expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        msp_anomaly_score(logits)


def test_scoring_accepts_float64_and_entropy_respects_class_bound() -> None:
    logits = np.zeros((2, 5, 3, 4), dtype=np.float64)

    entropy = predictive_entropy(logits)

    assert entropy.shape == (2, 3, 4)
    assert np.isfinite(entropy).all()
    assert np.all(entropy >= 0.0)
    assert np.all(entropy <= np.log(5.0) + 1.0e-12)


def test_all_scores_point_toward_obviously_uncertain_example() -> None:
    confident = np.array([[[[10.0]], [[0.0]], [[-1.0]]]], dtype=np.float32)
    uncertain = np.full((1, 3, 1, 1), -2.0, dtype=np.float32)

    for scorer in (
        msp_anomaly_score,
        predictive_entropy,
        max_logit_anomaly_score,
        energy_anomaly_score,
    ):
        assert float(scorer(uncertain).item()) > float(scorer(confident).item())


def test_scoring_contract_rejects_probability_and_cross_method_normalization_claims() -> None:
    contract = uncertainty_score_contract()

    assert contract["calibrated_anomaly_probability"] is False
    assert contract["normalization_across_methods"] == "none"
    assert all(method["anomaly_probability"] is False for method in contract["methods"].values())


def test_torch_cpu_scoring_preserves_layout_and_stays_finite() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [[[[2.0, 0.0]], [[0.0, 0.0]], [[-1.0, 0.0]]]],
        dtype=torch.float32,
        device="cpu",
    )

    probabilities = stable_softmax(logits)
    scores = (
        msp_anomaly_score(logits),
        predictive_entropy(logits),
        max_logit_anomaly_score(logits),
        energy_anomaly_score(logits, temperature=1.5),
    )

    assert tuple(probabilities.shape) == (1, 3, 1, 2)
    assert probabilities.device.type == "cpu"
    assert semantic_mask(logits).dtype == torch.int64
    for score in scores:
        assert tuple(score.shape) == (1, 1, 2)
        assert score.device.type == "cpu"
        assert bool(torch.isfinite(score).all().item())
