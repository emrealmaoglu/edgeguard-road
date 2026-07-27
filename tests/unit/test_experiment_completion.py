from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from edgeguard.evaluation.components import ComponentRecord
from edgeguard.evaluation.statistics import (
    component_detection_metrics,
    deterministic_bootstrap_interval,
    paired_comparison,
)
from edgeguard.experiment import hpo


def test_bootstrap_is_deterministic_and_paired_requires_alignment() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert deterministic_bootstrap_interval(values) == deterministic_bootstrap_interval(values)
    result = paired_comparison({"a": 1.0, "b": 2.0}, {"a": 1.5, "b": 1.5})
    assert result["mean_difference"] == 0.0
    assert result["significance_claim"] is False
    with pytest.raises(ValueError, match="same"):
        paired_comparison({"a": 1.0, "b": 2.0}, {"a": 1.5, "c": 1.5})


def _component(identifier: int, box: tuple[int, int, int, int]) -> ComponentRecord:
    x1, y1, x2, y2 = box
    return ComponentRecord(identifier, (x2 - x1) * (y2 - y1), box, (0.0, 0.0), 1.0, 1.0, 1.0)


def test_component_detection_and_false_positive_metrics() -> None:
    target = _component(1, (0, 0, 4, 4))
    matched = _component(1, (1, 1, 4, 4))
    false_positive = _component(2, (10, 10, 12, 12))
    result = component_detection_metrics(((matched, false_positive),), ((target,),))
    assert result["component_detection_rate"] == 1.0
    assert result["false_positive_components_per_image"] == 1.0


def test_hpo_interruption_resume_duplicate_prevention_and_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = {"calls": 0}

    def fake_training(
        _config_root: Path,
        _mmseg_checkout: Path,
        output_root: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        counter["calls"] += 1
        output_root.mkdir(parents=True)
        model = str(tuple(kwargs["model_families"])[0])  # type: ignore[call-overload,index]
        return {
            "models": [
                {
                    "model_family": model,
                    "validation_loss": 1.0 + counter["calls"] / 10,
                    "step_seconds": [0.1, 0.2],
                    "exact_resume": True,
                    "native_output_shape": [1, 19, 4, 8],
                }
            ],
            "failures": [],
        }

    monkeypatch.setattr(hpo, "run_five_model_mini_training", fake_training)
    with pytest.raises(InterruptedError):
        hpo.run_semantic_mini_hpo(tmp_path, tmp_path, tmp_path / "hpo", interrupt_after_trials=2)
    state = json.loads((tmp_path / "hpo" / "hpo_state.json").read_text(encoding="utf-8"))
    assert len(state["completed_trial_ids"]) == 2
    report = hpo.run_semantic_mini_hpo(tmp_path, tmp_path, tmp_path / "hpo")
    assert len(report["trials"]) == 4
    assert len(report["top_k_proposal"]) == 2
    assert report["final_promotion_performed"] is False
    calls_after_resume = counter["calls"]
    hpo.run_semantic_mini_hpo(tmp_path, tmp_path, tmp_path / "hpo")
    assert counter["calls"] == calls_after_resume
