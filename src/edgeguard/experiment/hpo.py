"""Small deterministic HPO/resume/promotion engine for semantic mini probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from edgeguard.models.semantic_local import run_five_model_mini_training
from edgeguard.serialization import sha256_payload
from edgeguard.telemetry.longrun import atomic_write_json


def _trial_identity(model: str, parameters: dict[str, Any], seed: int) -> str:
    return sha256_payload(
        {
            "model_family": model,
            "parameters": parameters,
            "seed": seed,
            "resolution": [128, 256],
            "resolution_is_hpo_variable": False,
        }
    )


def _valid_space(parameters: dict[str, Any]) -> None:
    if parameters.get("optimizer") not in {"sgd", "adamw"}:
        raise ValueError("HPO optimizer must be sgd or adamw")
    learning_rate = parameters.get("learning_rate")
    weight_decay = parameters.get("weight_decay")
    if not isinstance(learning_rate, (int, float)) or not 1e-6 <= learning_rate <= 0.1:
        raise ValueError("HPO learning rate is outside the reviewed range")
    if not isinstance(weight_decay, (int, float)) or not 0 <= weight_decay <= 0.1:
        raise ValueError("HPO weight decay is outside the reviewed range")
    if parameters.get("crop_profile") != "fixed_aspect_ratio_baseline":
        raise ValueError("primary HPO cannot tune crop resolution")
    if parameters.get("scheduler") != "poly":
        raise ValueError("HPO scheduler must use the bounded poly baseline")
    if parameters.get("effective_batch") != 2:
        raise ValueError("local HPO effective batch must remain fixed at two")


def _pareto_front(rows: list[dict[str, Any]]) -> list[str]:
    front: list[str] = []
    valid = [row for row in rows if row["hard_gates_passed"]]
    for candidate in valid:
        dominated = any(
            other["validation_loss"] <= candidate["validation_loss"]
            and other["step_seconds_mean"] <= candidate["step_seconds_mean"]
            and (
                other["validation_loss"] < candidate["validation_loss"]
                or other["step_seconds_mean"] < candidate["step_seconds_mean"]
            )
            for other in valid
            if other is not candidate
        )
        if not dominated:
            front.append(str(candidate["trial_id"]))
    return sorted(front)


def run_semantic_mini_hpo(
    config_root: Path,
    mmseg_checkout: Path,
    output_root: Path,
    *,
    interrupt_after_trials: int | None = None,
) -> dict[str, Any]:
    """Run/reuse four actual two-model trials and emit a non-scientific proposal."""
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "hpo_state.json"
    state: dict[str, Any] = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {
            "schema_version": "1.0",
            "record_type": "semantic_mini_hpo_state",
            "completed_trial_ids": [],
            "failed_trials": [],
        }
    )
    parameters: tuple[dict[str, Any], ...] = (
        {
            "optimizer": "sgd",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "scheduler": "poly",
            "crop_profile": "fixed_aspect_ratio_baseline",
            "effective_batch": 2,
        },
        {
            "optimizer": "adamw",
            "learning_rate": 0.0005,
            "weight_decay": 0.01,
            "scheduler": "poly",
            "crop_profile": "fixed_aspect_ratio_baseline",
            "effective_batch": 2,
        },
    )
    schedule = [
        (model, params, 20260727) for model in ("fast_scnn", "pidnet_s") for params in parameters
    ]
    if len({_trial_identity(model, params, seed) for model, params, seed in schedule}) != len(
        schedule
    ):
        raise ValueError("HPO schedule contains duplicate trial identities")
    completed = set(state.get("completed_trial_ids", []))
    failed_ids = {
        str(item["trial_id"])
        for item in state.get("failed_trials", [])
        if isinstance(item, dict) and "trial_id" in item
    }
    executed = 0
    for model, params, seed in schedule:
        _valid_space(params)
        trial_id = _trial_identity(model, params, seed)
        result_path = output_root / "trials" / trial_id / "trial_result.json"
        if trial_id in completed:
            if not result_path.is_file():
                raise ValueError("completed HPO trial is missing its result")
            continue
        if trial_id in failed_ids:
            continue
        trial_root = result_path.parent / "run"
        try:
            result = run_five_model_mini_training(
                config_root,
                mmseg_checkout,
                trial_root,
                optimizer_steps=2,
                model_families=(model,),
                learning_rate=float(params["learning_rate"]),
                weight_decay=float(params["weight_decay"]),
                optimizer_name=str(params["optimizer"]),
            )
            if result["failures"] or len(result["models"]) != 1:
                raise RuntimeError("bounded actual HPO model path failed")
            model_result = result["models"][0]
            validation_loss = float(model_result["validation_loss"])
            gates = {
                "finite_validation_loss": bool(np.isfinite(validation_loss)),
                "exact_resume": bool(model_result["exact_resume"]),
                "nineteen_channel_logits": model_result["native_output_shape"][1] == 19,
                "export_feasibility_available": True,
            }
            row = {
                "trial_id": trial_id,
                "model_family": model,
                "parameters": params,
                "seed": seed,
                "validation_loss": validation_loss,
                "step_seconds_mean": float(np.mean(model_result["step_seconds"])),
                "hard_gates": gates,
                "hard_gates_passed": all(gates.values()),
                "scientific_evidence": False,
            }
            result_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(result_path, row)
            completed.add(trial_id)
            state["completed_trial_ids"] = sorted(completed)
            atomic_write_json(state_path, state)
            executed += 1
            if interrupt_after_trials is not None and executed >= interrupt_after_trials:
                raise InterruptedError("bounded HPO interruption injection")
        except InterruptedError:
            raise
        except Exception as error:
            state.setdefault("failed_trials", []).append(
                {
                    "trial_id": trial_id,
                    "classification": type(error).__name__,
                    "error": str(error)[:1000],
                }
            )
            failed_ids.add(trial_id)
            atomic_write_json(state_path, state)
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_root / "trials").glob("*/trial_result.json"))
    ]
    if len(rows) + len(failed_ids) != len(schedule):
        raise RuntimeError("mini HPO did not terminate every scheduled trial")
    rows.sort(
        key=lambda row: (not row["hard_gates_passed"], row["validation_loss"], row["trial_id"])
    )
    hard_gate_rows = [row for row in rows if row["hard_gates_passed"]]
    report = {
        "schema_version": "1.0",
        "record_type": "actual_semantic_mini_hpo_report",
        "trials": rows,
        "failed_trials": state.get("failed_trials", []),
        "pareto_front_trial_ids": _pareto_front(rows),
        "top_k_proposal": [row["trial_id"] for row in hard_gate_rows[:2]],
        "promotion_refused": not bool(hard_gate_rows),
        "final_promotion_performed": False,
        "human_scientific_decision_required": True,
        "scientific_evidence": False,
    }
    atomic_write_json(output_root / "report.json", report)
    return report
