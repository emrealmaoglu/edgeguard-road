"""Bounded Optuna orchestration for the two frozen semantic finalists."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from edgeguard.rescue.config import RescueConfig
from edgeguard.rescue.ledger import append_run_ledger
from edgeguard.rescue.mmseg_runtime import evaluate_model, train_model
from edgeguard.rescue.multidomain import validate_dataset_manifest
from edgeguard.serialization import canonical_json, sha256_file


def select_hpo_models(candidate_table: Path) -> tuple[str, str]:
    """Select the two highest domain-macro screening candidates deterministically."""
    payload = json.loads(candidate_table.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("HPO candidate table must contain a candidates list")
    ranked: list[tuple[float, float, str]] = []
    for record in candidates:
        model = str(record.get("model"))
        metric = record.get("domain_macro_mIoU", record.get("mIoU"))
        domain_values = record.get("source_domain_mIoU")
        if (
            metric is None
            or record.get("screening_valid") is not True
            or record.get("onnx_validated") is not True
            or not isinstance(domain_values, dict)
            or set(domain_values) != {"cityscapes", "bdd100k", "idd20k"}
        ):
            continue
        domain_macro = sum(float(value) for value in domain_values.values()) / 3.0
        if abs(float(metric) - domain_macro) > 1.0e-12:
            raise ValueError("candidate domain-macro mIoU does not match its three domains")
        rare = float(record.get("rare_class_mIoU") or -1.0)
        ranked.append((float(metric), rare, model))
    selected: list[str] = []
    remaining = ranked
    while remaining and len(selected) < 2:
        best_macro = max(item[0] for item in remaining)
        tie_group = [item for item in remaining if best_macro - item[0] <= 0.002]
        winner = sorted(tie_group, key=lambda item: (-item[1], -item[0], item[2]))[0]
        if winner[2] not in selected:
            selected.append(winner[2])
        remaining = [item for item in remaining if item[2] != winner[2]]
    if len(selected) < 2:
        raise ValueError("HPO requires two interpretable screening candidates")
    return selected[0], selected[1]


def hpo_search_space(protocol: RescueConfig) -> dict[str, Any]:
    """Expose the frozen search space for preregistration and tests."""
    return {
        "learning_rate": list(protocol.hpo.learning_rate),
        "weight_decay": list(protocol.hpo.weight_decay),
        "scheduler": list(protocol.hpo.schedulers),
        "warmup_ratio": list(protocol.hpo.warmup_ratios),
        "fixed": {
            "resolution": list(protocol.crop_size),
            "loss": "ce",
            "initialization": "random",
            "domain_sampling": "uniform",
        },
    }


def _metric(metrics: dict[str, Any], name: str) -> float:
    for key, value in metrics.items():
        if key == name or key.endswith(f"/{name}"):
            return float(value)
    raise ValueError(f"evaluation did not emit {name}")


def _snapshot(study: Any, path: Path, *, model: str, manifests: Sequence[Path]) -> None:
    rows = []
    for trial in study.trials:
        rows.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "intermediate_values": {
                    str(key): value for key, value in trial.intermediate_values.items()
                },
                "user_attrs": trial.user_attrs,
            }
        )
    payload = {
        "schema_version": "2.0",
        "record_type": "semantic_hpo_snapshot",
        "model": model,
        "dataset_manifest_sha256s": [sha256_file(item) for item in manifests],
        "direction": "maximize_domain_macro_mIoU",
        "trials": rows,
    }
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def run_hpo_study(
    protocol: RescueConfig,
    *,
    model: str,
    manifests: Sequence[Path],
    mmseg_root: Path,
    output_root: Path,
    rare_classes_file: Path | None = None,
) -> dict[str, Any]:
    """Run/resume one 12-trial TPE study with 1.5k/3k successive-halving rungs."""
    try:
        optuna = __import__("optuna")
    except ModuleNotFoundError as error:
        raise RuntimeError("HPO requires the rescue/colab Optuna dependency") from error
    manifest_payloads = [validate_dataset_manifest(path) for path in manifests]
    if {item["dataset_id"] for item in manifest_payloads} != set(protocol.datasets.training):
        raise ValueError("HPO requires all three frozen source-domain manifests")
    if rare_classes_file is None:
        raise ValueError("HPO requires the frozen pooled rare-classes artifact")
    rare_payload = json.loads(rare_classes_file.read_text(encoding="utf-8"))
    expected_manifest_hashes = sorted(sha256_file(path) for path in manifests)
    if (
        sorted(rare_payload.get("dataset_manifest_sha256s", [])) != expected_manifest_hashes
        or len(rare_payload.get("groups", {}).get("rare", [])) != 5
    ):
        raise ValueError("rare-classes artifact does not match the three HPO manifests")
    study_root = output_root / "hpo" / model
    study_root.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{study_root / 'study.sqlite3'}"
    study = optuna.create_study(
        study_name=f"edgeguard-{model}-multidomain-v1",
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=protocol.hpo.sampler_seed),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=protocol.hpo.pruning_steps[0], reduction_factor=2
        ),
        load_if_exists=True,
    )
    study.set_user_attr("search_space", hpo_search_space(protocol))
    study.set_user_attr("dataset_manifest_sha256s", [sha256_file(path) for path in manifests])
    for stale in [trial for trial in study.trials if trial.state.name == "RUNNING"]:
        study.tell(stale.number, state=optuna.trial.TrialState.FAIL)

    def objective(trial: Any) -> float:
        learning_rate = trial.suggest_float("learning_rate", *protocol.hpo.learning_rate, log=True)
        weight_decay = trial.suggest_float("weight_decay", *protocol.hpo.weight_decay, log=True)
        scheduler = trial.suggest_categorical("scheduler", list(protocol.hpo.schedulers))
        warmup_ratio = trial.suggest_categorical("warmup_ratio", list(protocol.hpo.warmup_ratios))
        duplicate = next(
            (
                previous
                for previous in study.trials
                if previous.number != trial.number
                and previous.state.name != "RUNNING"
                and previous.params == trial.params
            ),
            None,
        )
        if duplicate is not None:
            trial.set_user_attr("duplicate_of_trial", duplicate.number)
            raise optuna.TrialPruned(f"duplicate of trial {duplicate.number}")
        run_name = f"trial-{trial.number:03d}"
        rungs = (*protocol.hpo.pruning_steps, protocol.hpo.max_steps)
        last_macro = 0.0
        for rung_index, rung in enumerate(rungs):
            train_model(
                protocol,
                model_name=model,
                stage_name="hpo",
                mmseg_root=mmseg_root,
                dataset_root=None,
                split_manifest=None,
                output_root=output_root,
                loss="ce",
                audit_report=None,
                resume=rung_index > 0,
                data_manifests=manifests,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                scheduler=str(scheduler),
                warmup_ratio=float(warmup_ratio),
                run_name=run_name,
                max_steps_override=rung,
                scheduler_steps_override=protocol.hpo.max_steps,
            )
            run_dir = output_root / "hpo" / model / run_name
            checkpoints = sorted(run_dir.glob("*.pth"), key=lambda path: path.stat().st_mtime)
            if not checkpoints:
                raise RuntimeError("HPO rung produced no checkpoint")
            scores: list[float] = []
            rare_scores: list[float] = []
            for manifest, payload in zip(manifests, manifest_payloads, strict=True):
                evaluation_dir = (
                    study_root / "evaluations" / run_name / str(rung) / str(payload["dataset_id"])
                )
                result = evaluate_model(
                    protocol,
                    resolved_config=run_dir / "resolved.py",
                    checkpoint=checkpoints[-1],
                    dataset=str(payload["dataset_id"]),
                    dataset_root=None,
                    split_manifest=None,
                    role="train_select",
                    condition=None,
                    output_dir=evaluation_dir,
                    fit_calibrator=False,
                    temperature_file=None,
                    max_reliability_pixels=1,
                    rare_classes_file=(
                        rare_classes_file if rung == protocol.hpo.max_steps else None
                    ),
                    dataset_manifest=manifest,
                    collect_classwise=rung == protocol.hpo.max_steps,
                )
                scores.append(_metric(result["metrics"], "mIoU"))
                if result["rare_class_mIoU"] is not None:
                    rare_scores.append(float(result["rare_class_mIoU"]))
            last_macro = sum(scores) / len(scores)
            trial.set_user_attr(f"domain_scores_{rung}", scores)
            if rare_scores:
                trial.set_user_attr(f"rare_macro_{rung}", sum(rare_scores) / len(rare_scores))
            trial.report(last_macro, step=rung)
            _snapshot(study, study_root / "trials.snapshot.json", model=model, manifests=manifests)
            if rung != protocol.hpo.max_steps and trial.should_prune():
                raise optuna.TrialPruned(f"pruned at {rung} optimizer steps")
        return last_macro

    terminal = [trial for trial in study.trials if trial.state.name != "RUNNING"]
    remaining = max(0, protocol.hpo.trials_per_model - len(terminal))
    if remaining:
        study.optimize(objective, n_trials=remaining, catch=(RuntimeError, ValueError, OSError))
    _snapshot(study, study_root / "trials.snapshot.json", model=model, manifests=manifests)
    complete = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if not complete:
        raise RuntimeError("HPO finished without a complete trial")
    best_macro = max(float(trial.value) for trial in complete)
    tied = [trial for trial in complete if best_macro - float(trial.value) <= 0.002]
    best = sorted(
        tied,
        key=lambda trial: (
            -float(trial.user_attrs.get(f"rare_macro_{protocol.hpo.max_steps}", -1.0)),
            -float(trial.value),
            trial.number,
        ),
    )[0]
    result = {
        "schema_version": "2.0",
        "record_type": "semantic_hpo_result",
        "model": model,
        "terminal_trials": len([trial for trial in study.trials if trial.state.name != "RUNNING"]),
        "complete_trials": len(complete),
        "best_trial": best.number,
        "best_domain_macro_mIoU": float(best.value),
        "best_params": best.params,
        "human_config_freeze_required": True,
    }
    (study_root / "result.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
    append_run_ledger(output_root / "run_ledger.jsonl", operation="semantic_hpo", result=result)
    return result
