"""State and artifact contracts for the resumable Colab production campaign."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from edgeguard.rescue.colab_recovery import (
    atomic_json,
    create_state_archive,
    publish_recovery_file,
    restore_recovery_file,
)
from edgeguard.rescue.multidomain import validate_dataset_manifest
from edgeguard.rescue.release_acceptance import accept_release_candidate_by_policy
from edgeguard.rescue.selection import select_recommended_model
from edgeguard.serialization import sha256_file, sha256_payload

CAMPAIGN_ID = "semantic-cs-idd-v3"
CORE_MODELS = ("segformer_b0", "fast_scnn", "pidnet_s")
EXTENSION_MODELS = ("ddrnet_23_slim", "bisenetv2")
ALL_MODELS = CORE_MODELS + EXTENSION_MODELS
SCIENTIFIC_STATUSES = ("measured", "accepted", "failed", "not_run")
PHASES = (
    "preflight",
    "restore",
    "stage-data",
    "canary",
    "smoke",
    "pilot",
    "extension-smoke",
    "screening",
    "hpo",
    "final",
    "selection",
    "ablation",
    "accept",
    "validation-data",
    "evaluate",
    "export",
    "report",
    "package",
)
TARGETS = (
    "smoke",
    "pilot",
    "screening",
    "hpo",
    "final",
    "evaluate",
    "export",
    "report",
    "package",
    "all",
)


def phases_for_target(target: str) -> tuple[str, ...]:
    """Return the complete ordered prerequisite closure for one public target."""
    if target not in TARGETS:
        raise ValueError(f"unsupported Colab pipeline target: {target}")
    end = len(PHASES) - 1 if target == "all" else PHASES.index(target)
    return PHASES[: end + 1]


def models_for_phase(phase: str) -> tuple[str, ...]:
    """Apply the frozen 3→5 model gate without notebook-local branching."""
    if phase == "canary":
        return ALL_MODELS
    if phase in {"smoke", "pilot"}:
        return CORE_MODELS
    if phase == "extension-smoke":
        return EXTENSION_MODELS
    if phase == "screening":
        return ALL_MODELS
    return ()


@dataclass(frozen=True)
class PipelineInputs:
    """Immutable identities and paths required by the Colab v3 orchestrator."""

    project_root: Path
    project_commit: str
    runtime_receipt: Path
    mmseg_root: Path
    work_root: Path
    recovery_root: Path
    config_path: Path
    data_manifests: tuple[Path, ...]
    candidate_table: Path | None = None
    final_models: tuple[str, ...] = ALL_MODELS
    ablation_model: str | None = None
    rare_classes_file: Path | None = None
    class_weights_file: Path | None = None
    accepted_release: Path | None = None
    evaluation_manifests: tuple[Path, ...] = ()
    authorization_policy: Path | None = None
    release_policy: Path | None = None
    data_roots: tuple[tuple[str, Path], ...] = ()
    campaign_id: str = CAMPAIGN_ID
    execution_mode: str = "production"
    state_store_root: Path | None = None

    def validated(self) -> PipelineInputs:
        if len(self.project_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.project_commit
        ):
            raise ValueError("project_commit must be a full lowercase Git SHA")
        if len(self.data_manifests) != 2:
            raise ValueError("Colab v3 requires Cityscapes and IDD20K frozen manifests")
        if len(set(self.data_manifests)) != len(self.data_manifests):
            raise ValueError("data manifest paths must be distinct")
        unknown = set(self.final_models) - set(ALL_MODELS)
        if unknown:
            raise ValueError(f"unsupported final models: {sorted(unknown)}")
        if self.final_models != ALL_MODELS:
            raise ValueError("Colab v3 final training requires all five models in frozen order")
        if self.ablation_model is not None:
            raise ValueError("Colab v3 derives the ablation model from train_select evidence")
        if self.execution_mode not in {"production", "acceptance"}:
            raise ValueError("execution_mode must be production or acceptance")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.campaign_id) is None:
            raise ValueError("campaign_id is missing or unsafe")
        if self.data_roots and {name for name, _ in self.data_roots} != {
            "cityscapes",
            "idd20k",
        }:
            raise ValueError("data roots must contain Cityscapes and IDD20K exactly")
        return self

    def identity(self) -> dict[str, Any]:
        """Return a path-light identity used to reject unsafe resume attempts."""
        self.validated()
        required_files = (
            self.runtime_receipt,
            self.config_path,
            *self.data_manifests,
        )
        for path in required_files:
            if not path.is_file():
                raise FileNotFoundError(f"pipeline identity input is missing: {path}")
        runtime = json.loads(self.runtime_receipt.read_text(encoding="utf-8"))
        probe = runtime.get("core_model_probe")
        stable_probe = (
            {
                key: probe.get(key)
                for key in (
                    "model_count",
                    "fp16_finite_model_count",
                    "checkpoint_resume_verified",
                )
            }
            if isinstance(probe, dict)
            else None
        )
        runtime_identity = {
            "record_type": runtime.get("record_type"),
            "project_commit": runtime.get("project_commit"),
            "runtime_profile": runtime.get("runtime_profile"),
            "lock_sha256": runtime.get("lock_sha256"),
            "environment": runtime.get("environment"),
            "core_model_probe": stable_probe,
        }
        return {
            "campaign_id": self.campaign_id,
            "project_commit": self.project_commit,
            "runtime_identity_sha256": sha256_payload(runtime_identity),
            "config_sha256": sha256_file(self.config_path),
            "data_manifest_sha256": {
                path.name: sha256_file(path) for path in sorted(self.data_manifests)
            },
        }


@dataclass(frozen=True)
class AcceptedModelSource:
    """One accepted final checkpoint/config pair resolved from a release."""

    model: str
    checkpoint: Path
    resolved_config: Path


def _release_artifact(release_path: Path, payload: object, label: str) -> Path:
    if not isinstance(payload, dict):
        raise ValueError(f"accepted release {label} artifact is invalid")
    relative = PurePosixPath(str(payload.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"accepted release {label} artifact path is unsafe")
    path = release_path.parent.joinpath(*relative.parts)
    if not path.is_file() or sha256_file(path) != payload.get("sha256"):
        raise ValueError(f"accepted release {label} artifact identity mismatch")
    return path


class ColabPipeline:
    """Execute resumable training phases and publish uniform evidence contracts."""

    def __init__(self, inputs: PipelineInputs) -> None:
        self.inputs = inputs.validated()
        self.identity = inputs.identity()
        self.identity_sha256 = sha256_payload(self.identity)

    def _accepted_release(self) -> tuple[dict[str, Any], tuple[AcceptedModelSource, ...]]:
        release_path = self.inputs.accepted_release
        if release_path is None or not release_path.is_file():
            raise PermissionError(
                "evaluate/export/report are closed until --accepted-release is provided"
            )
        release = json.loads(release_path.read_text(encoding="utf-8"))
        if release.get("record_type") != "edgeguard_accepted_release":
            raise ValueError("invalid accepted release record type")
        if release.get("status") != "accepted":
            raise PermissionError("pipeline handoff requires an accepted release")
        if release.get("campaign_id") != self.inputs.campaign_id:
            raise ValueError("accepted release belongs to another campaign")
        if release.get("project_commit") != self.inputs.project_commit:
            raise ValueError("accepted release belongs to another project commit")
        if release.get("data_manifest_sha256") != self.identity["data_manifest_sha256"]:
            raise ValueError("accepted release data manifest identity mismatch")
        release_id = str(release.get("release_id", ""))
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", release_id) is None:
            raise ValueError("accepted release_id is missing or unsafe")
        artifacts = release.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("accepted release has no hash-bound artifacts")
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict) or artifact.get("scientific_status") != "accepted":
                raise ValueError("accepted release contains a non-accepted artifact")
            _release_artifact(release_path, artifact, f"artifact[{index}]")
        raw_models = release.get("models")
        if not isinstance(raw_models, list) or len(raw_models) != len(ALL_MODELS):
            raise ValueError("accepted release must contain exactly five final model sources")
        models: list[AcceptedModelSource] = []
        seen: set[str] = set()
        for index, record in enumerate(raw_models):
            if not isinstance(record, dict):
                raise ValueError("accepted release model record is invalid")
            model = str(record.get("model", ""))
            if model not in ALL_MODELS or model in seen:
                raise ValueError(f"accepted release model is unsupported or repeated: {model}")
            seen.add(model)
            models.append(
                AcceptedModelSource(
                    model=model,
                    checkpoint=_release_artifact(
                        release_path, record.get("checkpoint"), f"models[{index}].checkpoint"
                    ),
                    resolved_config=_release_artifact(
                        release_path,
                        record.get("resolved_config"),
                        f"models[{index}].resolved_config",
                    ),
                )
            )
        if tuple(item.model for item in models) != ALL_MODELS:
            raise ValueError(
                "accepted release model order differs from the frozen five-model order"
            )
        return release, tuple(models)

    def _accepted_evaluation_manifests(self) -> dict[str, Path]:
        if len(self.inputs.evaluation_manifests) != 2:
            raise PermissionError(
                "evaluate requires two explicit frozen --evaluation-manifest inputs"
            )
        manifests: dict[str, Path] = {}
        for path in self.inputs.evaluation_manifests:
            payload = validate_dataset_manifest(
                path,
                allowed_roles=("official_source_val",),
            )
            dataset = str(payload["dataset_id"])
            if dataset not in {"cityscapes", "idd20k"} or dataset in manifests:
                raise ValueError(
                    "evaluation manifests must contain distinct Cityscapes and IDD20K sources"
                )
            manifests[dataset] = path
        if set(manifests) != {"cityscapes", "idd20k"}:
            raise ValueError("evaluation requires Cityscapes and IDD20K official source manifests")
        return manifests

    def _record_release_verification(
        self,
        phase: str,
        release: dict[str, Any],
        models: tuple[AcceptedModelSource, ...],
    ) -> None:
        assert self.inputs.accepted_release is not None
        atomic_json(
            self._phase_root(phase) / "accepted_release_verification.json",
            {
                "schema_version": "2.0",
                "record_type": "edgeguard_accepted_release_verification",
                "phase": phase,
                "release_id": release["release_id"],
                "release_sha256": sha256_file(self.inputs.accepted_release),
                "project_commit": self.inputs.project_commit,
                "data_manifest_sha256": self.identity["data_manifest_sha256"],
                "models": [
                    {
                        "model": item.model,
                        "checkpoint_sha256": sha256_file(item.checkpoint),
                        "resolved_config_sha256": sha256_file(item.resolved_config),
                    }
                    for item in models
                ],
                "status": "accepted",
            },
        )

    @property
    def state_root(self) -> Path:
        return self.inputs.work_root / "pipeline-v3"

    def plan(self, target: str) -> dict[str, Any]:
        phases = phases_for_target(target)
        return {
            "schema_version": "2.0",
            "record_type": "edgeguard_colab_pipeline_plan",
            "campaign_id": self.inputs.campaign_id,
            "target": target,
            "identity_sha256": self.identity_sha256,
            "phases": [
                {
                    "phase": phase,
                    "models": list(models_for_phase(phase)),
                    "status": "completed" if self._phase_complete(phase) else "pending",
                }
                for phase in phases
            ],
        }

    def _phase_root(self, phase: str) -> Path:
        return self.state_root / phase

    def _completion_path(self, phase: str) -> Path:
        return self._phase_root(phase) / "completion.json"

    def _phase_identity_sha256(self, phase: str) -> str:
        identity = dict(self.identity)
        if phase in {
            "hpo",
            "final",
            "selection",
            "ablation",
            "accept",
            "validation-data",
            "evaluate",
            "export",
            "report",
            "package",
        }:
            identity["candidate_table_sha256"] = (
                sha256_file(self.inputs.candidate_table)
                if self.inputs.candidate_table is not None and self.inputs.candidate_table.is_file()
                else None
            )
        if phase in {
            "selection",
            "ablation",
            "accept",
            "validation-data",
            "evaluate",
            "export",
            "report",
            "package",
        }:
            identity["final_models"] = list(self.inputs.final_models)
            identity["class_weights_sha256"] = (
                sha256_file(self.inputs.class_weights_file)
                if self.inputs.class_weights_file is not None
                and self.inputs.class_weights_file.is_file()
                else None
            )
            selection = self.inputs.work_root / "reports/selection/recommended_model.json"
            identity["recommended_selection_sha256"] = (
                sha256_file(selection) if selection.is_file() else None
            )
        if phase in {"validation-data", "evaluate", "export", "report", "package"}:
            identity["accepted_release_sha256"] = (
                sha256_file(self.inputs.accepted_release)
                if self.inputs.accepted_release is not None
                and self.inputs.accepted_release.is_file()
                else None
            )
            identity["evaluation_manifest_sha256"] = {
                path.name: sha256_file(path)
                for path in sorted(self.inputs.evaluation_manifests)
                if path.is_file()
            }
        if phase in {"accept", "validation-data", "evaluate", "export", "report", "package"}:
            identity["release_policy_sha256"] = (
                sha256_file(self.inputs.release_policy)
                if self.inputs.release_policy is not None and self.inputs.release_policy.is_file()
                else None
            )
        return sha256_payload(identity)

    def _restore_accepted_release_checkpoints(self) -> list[dict[str, Any]]:
        release_path = self.inputs.accepted_release
        if release_path is None or not release_path.is_file():
            return []
        release = json.loads(release_path.read_text(encoding="utf-8"))
        if (
            release.get("record_type") != "edgeguard_accepted_release"
            or release.get("status") != "accepted"
            or release.get("campaign_id") != self.inputs.campaign_id
            or release.get("project_commit") != self.inputs.project_commit
            or release.get("data_manifest_sha256") != self.identity["data_manifest_sha256"]
        ):
            raise ValueError("accepted release identity is invalid during restore")
        records = release.get("models")
        if not isinstance(records, list) or not records:
            raise ValueError("accepted release has no restorable model records")
        restored: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict) or record.get("model") not in ALL_MODELS:
                raise ValueError("accepted release contains an invalid restorable model")
            model = str(record["model"])
            checkpoint = record.get("checkpoint")
            if not isinstance(checkpoint, dict):
                raise ValueError("accepted release checkpoint record is invalid")
            relative = PurePosixPath(str(checkpoint.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("accepted release checkpoint path is unsafe")
            destination = release_path.parent.joinpath(*relative.parts)
            expected_sha = checkpoint.get("sha256")
            if destination.is_file() and sha256_file(destination) == expected_sha:
                continue
            destination.unlink(missing_ok=True)
            result = restore_recovery_file(
                self.inputs.recovery_root,
                artifact_id=f"final-{model.replace('_', '-')}-ce",
                destination=destination,
            )
            if sha256_file(destination) != expected_sha:
                destination.unlink(missing_ok=True)
                raise ValueError("restored final checkpoint does not match accepted release")
            destination.parent.joinpath("last_checkpoint").write_text(
                destination.name + "\n", encoding="utf-8"
            )
            restored.append(
                {
                    "label": f"restore-{model}",
                    "return_code": 0,
                    "artifact_id": result["artifact_id"],
                    "generation": result["generation"],
                    "sha256": expected_sha,
                }
            )
        return restored

    def _phase_complete(self, phase: str) -> bool:
        completion = self._completion_path(phase)
        if not completion.is_file():
            return False
        try:
            payload = json.loads(completion.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if (
            payload.get("identity_sha256") != self._phase_identity_sha256(phase)
            or payload.get("phase") != phase
            or payload.get("status") != "completed"
        ):
            return False
        index = self._phase_root(phase) / "artifact_index.json"
        if not index.is_file():
            return False
        if payload.get("artifact_index_sha256") != sha256_file(index):
            return False
        try:
            artifact_index = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        artifacts = artifact_index.get("artifacts")
        if not isinstance(artifacts, dict):
            return False
        for relative, identity in artifacts.items():
            candidate = self._phase_root(phase) / str(relative)
            if not isinstance(identity, dict) or not candidate.is_file():
                return False
            if identity.get("sha256") != sha256_file(candidate):
                return False
            if identity.get("byte_size") != candidate.stat().st_size:
                return False
        return True

    def _write_run_contracts(
        self,
        phase: str,
        *,
        elapsed_seconds: float,
        scientific_status: Literal["measured", "accepted", "failed", "not_run"],
        command_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        root = self._phase_root(phase)
        root.mkdir(parents=True, exist_ok=True)
        runtime = json.loads(self.inputs.runtime_receipt.read_text(encoding="utf-8"))
        phase_models = list(models_for_phase(phase))
        if phase in {"final", "selection", "accept", "validation-data", "package"}:
            phase_models = list(ALL_MODELS)
        if phase == "ablation":
            phase_models = [self._recommended_model()]
        release_verification = root / "accepted_release_verification.json"
        if release_verification.is_file():
            verified = json.loads(release_verification.read_text(encoding="utf-8"))
            phase_models = [str(record["model"]) for record in verified.get("models", [])]
        run_manifest = {
            "schema_version": "2.0",
            "record_type": "edgeguard_run_manifest",
            "run_id": f"{self.inputs.campaign_id}-{phase}",
            "campaign_id": self.inputs.campaign_id,
            "stage": phase,
            "models": phase_models,
            "git_commit": self.inputs.project_commit,
            "config_sha256": self.identity["config_sha256"],
            "data_manifest_sha256": self.identity["data_manifest_sha256"],
            "environment_sha256": self.identity["runtime_identity_sha256"],
            "seed": 20260728,
            "scientific_status": scientific_status,
            "synthetic_or_smoke": phase in {"smoke", "extension-smoke"},
        }
        environment = {
            "schema_version": "2.0",
            "record_type": "edgeguard_environment",
            "runtime_profile": runtime.get("runtime_profile"),
            "lock_sha256": runtime.get("lock_sha256"),
            "environment": runtime.get("environment"),
        }
        metric_rows: list[dict[str, Any]] = []
        class_rows: list[tuple[Any, ...]] = []
        evaluation_root = self.inputs.work_root / "evaluation" / phase
        for path in sorted(evaluation_root.glob("**/evaluation.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("scientific_evidence") is not True:
                continue
            metrics_payload = payload.get("metrics")
            classwise = payload.get("classwise_metrics")
            if not isinstance(metrics_payload, dict) or not isinstance(classwise, dict):
                continue
            metric_rows.append(
                {
                    "model": payload.get("model"),
                    "dataset": payload.get("dataset"),
                    "mIoU": metrics_payload.get("mIoU"),
                    "rare_class_mIoU": payload.get("rare_class_mIoU"),
                    "source_sha256": sha256_file(path),
                }
            )
            per_class = classwise.get("per_class_iou")
            if isinstance(per_class, list):
                class_rows.extend(
                    (
                        payload.get("model"),
                        payload.get("dataset"),
                        class_id,
                        value,
                        "measured",
                    )
                    for class_id, value in enumerate(per_class)
                )
        metrics = {
            "schema_version": "2.0",
            "record_type": "edgeguard_metrics",
            "scientific_status": "measured" if metric_rows else "not_run",
            "measurements": metric_rows,
            "reason": None if metric_rows else "no accepted aggregation input was produced",
            "commands": command_results,
        }
        atomic_json(root / "run_manifest.json", run_manifest)
        atomic_json(root / "environment.json", environment)
        atomic_json(root / "metrics.json", metrics)
        per_class = root / "per_class.csv"
        with per_class.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("model", "dataset", "class_id", "iou", "scientific_status"))
            writer.writerows(class_rows)
        actual_phase = "smoke" if phase == "extension-smoke" else phase
        training_seconds = 0.0
        peak_gpu_memory_bytes = 0
        for summary_path in sorted(
            (self.inputs.work_root / "runs" / actual_phase).glob("*/*/summary.json")
        ):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            training_seconds += float(summary.get("elapsed_seconds") or 0.0)
            environment_payload = summary.get("environment")
            if isinstance(environment_payload, dict):
                peak_gpu_memory_bytes = max(
                    peak_gpu_memory_bytes,
                    int(environment_payload.get("peak_gpu_memory_bytes") or 0),
                )
        with (root / "resource.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "elapsed_seconds",
                    "accelerator_hours",
                    "peak_gpu_memory_mb",
                    "interruption_cost_seconds",
                )
            )
            writer.writerow(
                (
                    f"{elapsed_seconds:.6f}",
                    f"{training_seconds / 3600:.9f}",
                    f"{peak_gpu_memory_bytes / 1024**2:.3f}" if peak_gpu_memory_bytes else "",
                    "0",
                )
            )
        indexed_paths = tuple(
            path
            for path in sorted(item for item in root.rglob("*") if item.is_file())
            if path.name not in {"artifact_index.json", "completion.json", "failure.json"}
        )
        artifact_index = {
            "schema_version": "2.0",
            "record_type": "edgeguard_artifact_index",
            "campaign_id": self.inputs.campaign_id,
            "phase": phase,
            "scientific_status": scientific_status,
            "artifacts": {
                path.relative_to(root).as_posix(): {
                    "sha256": sha256_file(path),
                    "byte_size": path.stat().st_size,
                }
                for path in indexed_paths
            },
        }
        atomic_json(root / "artifact_index.json", artifact_index)
        return artifact_index

    def _complete_phase(
        self,
        phase: str,
        *,
        elapsed_seconds: float,
        scientific_status: Literal["measured", "accepted", "failed", "not_run"],
        command_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        root = self._phase_root(phase)
        (root / "failure.json").unlink(missing_ok=True)
        self._write_run_contracts(
            phase,
            elapsed_seconds=elapsed_seconds,
            scientific_status=scientific_status,
            command_results=command_results,
        )
        completion = {
            "schema_version": "2.0",
            "record_type": "edgeguard_colab_phase_completion",
            "campaign_id": self.inputs.campaign_id,
            "phase": phase,
            "status": "completed",
            "identity_sha256": self._phase_identity_sha256(phase),
            "artifact_index_sha256": sha256_file(root / "artifact_index.json"),
        }
        atomic_json(self._completion_path(phase), completion)
        self._publish_campaign_state(phase)
        return completion

    def _publish_campaign_state(self, phase: str) -> None:
        """Publish small resumable state after every completed phase."""
        if self.inputs.state_store_root is None:
            return
        archive = self.inputs.work_root.parent / "edgeguard-campaign-state-v3.tar.gz"
        relative_paths = (
            "accepted_release.candidate.json",
            "accepted_release.json",
            "calibration",
            "evaluation",
            "exports",
            "ledger",
            "manifests",
            "multidomain-statistics",
            "pipeline-v3",
            "reports/screening",
            "reports/selection",
            "reviews",
            "runs",
        )
        create_state_archive(
            self.inputs.work_root,
            archive,
            relative_paths=relative_paths,
            compression=True,
        )
        publish_recovery_file(
            archive,
            self.inputs.state_store_root,
            artifact_id="campaign-state",
            campaign_id=self.inputs.campaign_id,
            project_commit=self.inputs.project_commit,
            metadata={"phase": phase, "target": "all"},
        )

    def _verify_preflight(self) -> None:
        head = subprocess.run(
            ["git", "-C", str(self.inputs.project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != self.inputs.project_commit:
            raise ValueError("checked-out project commit does not match the pipeline identity")
        runtime = json.loads(self.inputs.runtime_receipt.read_text(encoding="utf-8"))
        if runtime.get("project_commit") != self.inputs.project_commit:
            raise ValueError("runtime receipt belongs to another project commit")
        if runtime.get("record_type") != "semantic_hermetic_runtime_receipt":
            raise ValueError("runtime receipt is not the hermetic Colab v2 contract")
        if not self.inputs.mmseg_root.is_dir():
            raise FileNotFoundError("verified MMSegmentation checkout is missing")

    def _verify_canary(self) -> None:
        runtime = json.loads(self.inputs.runtime_receipt.read_text(encoding="utf-8"))
        probe = runtime.get("core_model_probe")
        if not isinstance(probe, dict) or (
            probe.get("model_count") != len(ALL_MODELS)
            or probe.get("checkpoint_resume_verified") is not True
            or probe.get("checkpoint_resume_model_count") != len(ALL_MODELS)
            or probe.get("fp16_finite_model_count") != len(ALL_MODELS)
        ):
            raise ValueError("runtime receipt lacks the complete five-model FP32/AMP canary")

    def _train_command(
        self,
        phase: str,
        model: str | None,
        *,
        device_batch: int | None = None,
        loss: str = "ce",
        run_name: str | None = None,
        crop_size: tuple[int, int] | None = None,
    ) -> list[str]:
        training_stage = (
            "smoke" if phase == "extension-smoke" else "final" if phase == "ablation" else phase
        )
        command = [
            str(Path(sys_executable())),
            str(self.inputs.project_root / "scripts/train.py"),
            "--config",
            str(self.inputs.config_path),
            "--stage",
            training_stage,
            "--output-root",
            str(self.inputs.work_root / "runs"),
            "--mmseg-root",
            str(self.inputs.mmseg_root),
            "--recovery-root",
            str(self.inputs.recovery_root),
            "--campaign-id",
            self.inputs.campaign_id,
            "--project-commit",
            self.inputs.project_commit,
            "--loss",
            loss,
        ]
        for manifest in self.inputs.data_manifests:
            command.extend(("--data-manifest", str(manifest)))
        if model is not None:
            command.extend(("--model", model))
        if device_batch is not None:
            command.extend(("--device-batch", str(device_batch)))
        if run_name is not None:
            command.extend(("--run-name", run_name))
        if crop_size is not None:
            command.extend(("--crop-height", str(crop_size[0]), "--crop-width", str(crop_size[1])))
        if self.inputs.execution_mode == "acceptance":
            command.extend(("--max-steps", "2", "--workers", "0", "--precision", "fp32"))
            if phase == "hpo":
                command.append("--acceptance-test")
            if device_batch is None:
                command.extend(("--device-batch", "1"))
        elif phase in {"smoke", "extension-smoke"}:
            command.extend(("--intentional-interrupt-step", "25"))
        if loss == "median_frequency":
            if self.inputs.class_weights_file is None:
                raise ValueError("weighted-loss command requires class weights")
            command.extend(("--audit-report", str(self.inputs.class_weights_file)))
        if phase in {"final", "ablation"} and model is not None:
            hpo_result = self.inputs.work_root / "runs" / "hpo" / model / "result.json"
            if hpo_result.is_file():
                params = json.loads(hpo_result.read_text(encoding="utf-8")).get("best_params")
                if not isinstance(params, dict):
                    raise ValueError(f"HPO result has no best_params: {hpo_result}")
                command.extend(
                    (
                        "--learning-rate",
                        str(params["learning_rate"]),
                        "--weight-decay",
                        str(params["weight_decay"]),
                        "--scheduler",
                        str(params["scheduler"]),
                        "--warmup-ratio",
                        str(params["warmup_ratio"]),
                    )
                )
        return command

    def _run_command(self, phase: str, label: str, command: list[str]) -> dict[str, Any]:
        log_root = self._phase_root(phase) / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"{label}.combined.log"
        tail: deque[str] = deque(maxlen=300)
        with log_path.open("w", encoding="utf-8") as log_stream:
            process = subprocess.Popen(
                command,
                cwd=self.inputs.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_stream.write(line)
                log_stream.flush()
                tail.append(line)
                print(line, end="", flush=True)
            return_code = process.wait()
        output_tail = "".join(tail)
        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code,
                command,
                output=output_tail,
                stderr=output_tail,
            )
        return {
            "label": label,
            "return_code": 0,
            "command": command,
            "log": log_path.relative_to(self._phase_root(phase)).as_posix(),
        }

    def _recovery_artifact_id(self, phase: str, model: str, suffix: str) -> str:
        actual_phase = (
            "smoke" if phase == "extension-smoke" else "final" if phase == "ablation" else phase
        )
        return f"{actual_phase}-{model}-{suffix}".replace("_", "-")

    def _recovery_pointer_exists(self, phase: str, model: str, suffix: str) -> bool:
        artifact_id = self._recovery_artifact_id(phase, model, suffix)
        return (self.inputs.recovery_root / "pointers" / f"{artifact_id}.json").is_file()

    def _prepare_oom_retry(
        self, phase: str, model: str, suffix: str, run_dir: Path, command: list[str]
    ) -> list[str]:
        retry = list(command)
        if self._recovery_pointer_exists(phase, model, suffix):
            if "--resume" not in retry:
                retry.append("--resume")
            return retry
        if run_dir.exists():
            quarantine = run_dir.with_name(f"{run_dir.name}.oom-attempt-{time.time_ns()}")
            run_dir.rename(quarantine)
        return [value for value in retry if value != "--resume"]

    def _run_directory(self, phase: str, model: str, loss: str = "ce") -> Path:
        actual_phase = (
            "smoke" if phase == "extension-smoke" else "final" if phase == "ablation" else phase
        )
        return self.inputs.work_root / "runs" / actual_phase / model / loss

    def _checkpoint(self, phase: str, model: str, loss: str = "ce") -> Path:
        run_dir = self._run_directory(phase, model, loss)
        pointer = run_dir / "last_checkpoint"
        checkpoint = (
            run_dir / pointer.read_text(encoding="utf-8").strip()
            if pointer.is_file()
            else run_dir / "missing.pth"
        )
        if not checkpoint.is_file():
            identity_path = run_dir / "run_identity.json"
            if not identity_path.is_file():
                raise FileNotFoundError(f"training checkpoint identity is missing: {run_dir}")
            artifact_id = self._recovery_artifact_id(phase, model, loss)
            checkpoint = run_dir / "recovered.pth"
            result = restore_recovery_file(
                self.inputs.recovery_root,
                artifact_id=artifact_id,
                destination=checkpoint,
            )
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if result.get("metadata", {}).get("identity_sha256") != sha256_payload(identity):
                checkpoint.unlink(missing_ok=True)
                raise ValueError("Drive recovery checkpoint has another immutable run identity")
            pointer.write_text(checkpoint.name + "\n", encoding="utf-8")
        return checkpoint

    def _screening_evidence(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        evaluation_root = self.inputs.work_root / "evaluation/screening"
        export_root = self.inputs.work_root / "exports/screening"
        for model in ALL_MODELS:
            run_dir = self._run_directory("screening", model)
            checkpoint = self._checkpoint("screening", model)
            resolved = run_dir / "resolved.py"
            for manifest in self.inputs.data_manifests:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                dataset = str(payload.get("dataset_id", ""))
                if dataset not in {"cityscapes", "idd20k"}:
                    raise ValueError("screening evidence accepts only Cityscapes and IDD20K")
                output = evaluation_root / model / dataset
                evidence = output / "evaluation.json"
                if evidence.is_file():
                    recorded = json.loads(evidence.read_text(encoding="utf-8"))
                    if recorded.get("checkpoint_sha256") != sha256_file(checkpoint):
                        raise ValueError(
                            "existing screening evaluation belongs to another checkpoint"
                        )
                    continue
                if output.exists() and any(output.iterdir()):
                    raise ValueError(f"incomplete screening evaluation requires review: {output}")
                command = [
                    sys_executable(),
                    str(self.inputs.project_root / "scripts/evaluate.py"),
                    "run",
                    "--resolved-config",
                    str(resolved),
                    "--checkpoint",
                    str(checkpoint),
                    "--dataset",
                    dataset,
                    "--dataset-manifest",
                    str(manifest),
                    "--role",
                    "train_select",
                    "--output-dir",
                    str(output),
                ]
                if self.inputs.rare_classes_file is not None:
                    command.extend(("--rare-classes-file", str(self.inputs.rare_classes_file)))
                results.append(
                    self._run_command("screening", f"evaluate-{model}-{dataset}", command)
                )
            export_dir = export_root / model
            onnx = export_dir / f"{model}.onnx"
            validation = onnx.with_suffix(".validation.json")
            if validation.is_file() and onnx.is_file():
                recorded = json.loads(validation.read_text(encoding="utf-8"))
                if recorded.get("checkpoint_sha256") != sha256_file(checkpoint) or recorded.get(
                    "onnx_sha256"
                ) != sha256_file(onnx):
                    raise ValueError("existing screening ONNX belongs to another checkpoint")
            else:
                if export_dir.exists() and any(export_dir.iterdir()):
                    raise ValueError(f"incomplete screening export requires review: {export_dir}")
                results.append(
                    self._run_command(
                        "screening",
                        f"export-{model}",
                        [
                            sys_executable(),
                            str(self.inputs.project_root / "scripts/export_onnx.py"),
                            "--resolved-config",
                            str(resolved),
                            "--checkpoint",
                            str(checkpoint),
                            "--output",
                            str(onnx),
                            "--device",
                            "cuda",
                            "--warmup",
                            "5",
                            "--iterations",
                            "20",
                        ],
                    )
                )
        report = self.inputs.work_root / "reports/screening"
        candidate_table = report / "candidate_table.json"
        if not candidate_table.is_file():
            if report.exists() and any(report.iterdir()):
                raise ValueError(f"incomplete screening report requires review: {report}")
            command = [
                sys_executable(),
                str(self.inputs.project_root / "scripts/evaluate.py"),
                "summarize",
                "--evaluation-root",
                str(evaluation_root),
                "--export-root",
                str(export_root),
                "--output-dir",
                str(report),
                "--expected-dataset",
                "cityscapes",
                "--expected-dataset",
                "idd20k",
            ]
            results.append(self._run_command("screening", "screening-report", command))
        return results

    def _selection_evidence(self) -> list[dict[str, Any]]:
        """Evaluate all final CE checkpoints only on frozen train-select roles."""
        results: list[dict[str, Any]] = []
        evaluation_root = self.inputs.work_root / "evaluation/selection"
        export_root = self.inputs.work_root / "exports/selection"
        for model in ALL_MODELS:
            run_dir = self._run_directory("final", model)
            checkpoint = self._checkpoint("final", model)
            resolved = run_dir / "resolved.py"
            for manifest in self.inputs.data_manifests:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                dataset = str(payload.get("dataset_id", ""))
                output = evaluation_root / model / dataset
                evidence = output / "evaluation.json"
                if evidence.is_file():
                    recorded = json.loads(evidence.read_text(encoding="utf-8"))
                    if (
                        recorded.get("checkpoint_sha256") != sha256_file(checkpoint)
                        or recorded.get("role") != "train_select"
                    ):
                        raise ValueError("existing final selection evidence identity mismatch")
                    continue
                if output.exists() and any(output.iterdir()):
                    raise ValueError(f"incomplete final selection evaluation: {output}")
                command = [
                    sys_executable(),
                    str(self.inputs.project_root / "scripts/evaluate.py"),
                    "run",
                    "--resolved-config",
                    str(resolved),
                    "--checkpoint",
                    str(checkpoint),
                    "--dataset",
                    dataset,
                    "--dataset-manifest",
                    str(manifest),
                    "--role",
                    "train_select",
                    "--output-dir",
                    str(output),
                ]
                if self.inputs.rare_classes_file is not None:
                    command.extend(("--rare-classes-file", str(self.inputs.rare_classes_file)))
                results.append(
                    self._run_command("selection", f"evaluate-{model}-{dataset}", command)
                )
            export_dir = export_root / model
            onnx = export_dir / f"{model}.onnx"
            validation = onnx.with_suffix(".validation.json")
            if validation.is_file() and onnx.is_file():
                recorded = json.loads(validation.read_text(encoding="utf-8"))
                if recorded.get("checkpoint_sha256") != sha256_file(checkpoint) or recorded.get(
                    "onnx_sha256"
                ) != sha256_file(onnx):
                    raise ValueError("existing final selection ONNX identity mismatch")
            else:
                if export_dir.exists() and any(export_dir.iterdir()):
                    raise ValueError(f"incomplete final selection export: {export_dir}")
                results.append(
                    self._run_command(
                        "selection",
                        f"export-{model}",
                        [
                            sys_executable(),
                            str(self.inputs.project_root / "scripts/export_onnx.py"),
                            "--resolved-config",
                            str(resolved),
                            "--checkpoint",
                            str(checkpoint),
                            "--output",
                            str(onnx),
                            "--device",
                            "cuda",
                            "--warmup",
                            "5",
                            "--iterations",
                            "20",
                        ],
                    )
                )
        report = self.inputs.work_root / "reports/selection"
        candidate_table = report / "candidate_table.json"
        if not candidate_table.is_file():
            if report.exists() and any(report.iterdir()):
                raise ValueError(f"incomplete final selection report: {report}")
            results.append(
                self._run_command(
                    "selection",
                    "selection-report",
                    [
                        sys_executable(),
                        str(self.inputs.project_root / "scripts/evaluate.py"),
                        "summarize",
                        "--evaluation-root",
                        str(evaluation_root),
                        "--export-root",
                        str(export_root),
                        "--output-dir",
                        str(report),
                        "--expected-dataset",
                        "cityscapes",
                        "--expected-dataset",
                        "idd20k",
                    ],
                )
            )
        payload = json.loads(candidate_table.read_text(encoding="utf-8"))
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("final selection candidate table is invalid")
        selection = select_recommended_model(candidates, expected_models=ALL_MODELS)
        atomic_json(report / "recommended_model.json", selection)
        return results

    def _run_training_phase(self, phase: str) -> list[dict[str, Any]]:
        if phase == "hpo":
            if self.inputs.candidate_table is None or not self.inputs.candidate_table.is_file():
                raise FileNotFoundError("HPO requires the frozen screening candidate table")
            command = self._train_command(phase, None)
            command.extend(("--candidate-table", str(self.inputs.candidate_table)))
            if self.inputs.rare_classes_file is not None:
                command.extend(("--rare-classes-file", str(self.inputs.rare_classes_file)))
            return [self._run_command(phase, "hpo-top-two", command)]
        models = models_for_phase(phase)
        if phase == "final":
            models = self.inputs.final_models
        results: list[dict[str, Any]] = []
        for model in models:
            for loss in ("ce",):
                label = f"{model}-{loss}"
                command = self._train_command(phase, model, loss=loss)
                run_dir = self._run_directory(phase, model, loss)
                if (run_dir / "run_identity.json").is_file() or self._recovery_pointer_exists(
                    phase, model, loss
                ):
                    command.append("--resume")
                try:
                    results.append(self._run_command(phase, label, command))
                except subprocess.CalledProcessError as error:
                    stderr = str(error.stderr or "").lower()
                    intentional = "edgeguard_intentional_interruption" in stderr
                    if intentional:
                        resume_command = list(command)
                        if "--resume" not in resume_command:
                            resume_command.append("--resume")
                        result = self._run_command(
                            phase, f"{label}-intentional-resume", resume_command
                        )
                        marker = run_dir / ".intentional-interruption-complete.json"
                        if not marker.is_file():
                            raise RuntimeError(
                                "intentional interruption marker is missing"
                            ) from error
                        proof = json.loads(marker.read_text(encoding="utf-8"))
                        result["interruption_resume"] = {
                            "verified": True,
                            "optimizer_step": proof["optimizer_step"],
                            "checkpoint_sha256": proof["checkpoint_sha256"],
                        }
                        results.append(result)
                        continue
                    if "out of memory" not in stderr and "cuda oom" not in stderr:
                        raise
                    retry = self._train_command(phase, model, device_batch=2, loss=loss)
                    retry = self._prepare_oom_retry(phase, model, loss, run_dir, retry)
                    try:
                        result = self._run_command(phase, f"{label}-oom-retry", retry)
                    except subprocess.CalledProcessError as retry_error:
                        retry_stderr = str(retry_error.stderr or "").lower()
                        if "edgeguard_intentional_interruption" not in retry_stderr:
                            raise
                        result = self._run_command(phase, f"{label}-oom-retry-resume", retry)
                    result["oom_recovery"] = {
                        "attempts": 1,
                        "device_batch": 2,
                        "effective_batch": 4,
                    }
                    results.append(result)
        if phase == "screening":
            results.extend(self._screening_evidence())
        return results

    def _recommended_model(self) -> str:
        selection_path = self.inputs.work_root / "reports/selection/recommended_model.json"
        if not selection_path.is_file():
            raise FileNotFoundError("recommended-model selection is missing")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        model = str(selection.get("recommended_model", ""))
        if model not in ALL_MODELS or selection.get("selection_role") != "train_select":
            raise ValueError("recommended-model selection is invalid")
        return model

    def _run_ablation_phase(self) -> list[dict[str, Any]]:
        """Run weighted-loss and low-resolution ablations for the selected model."""
        if self.inputs.class_weights_file is None or not self.inputs.class_weights_file.is_file():
            raise FileNotFoundError("ablation requires frozen class weights")
        model = self._recommended_model()
        variants = (
            ("median_frequency", "median_frequency", None),
            ("ce-256x512", "ce", (256, 512)),
        )
        results: list[dict[str, Any]] = []
        for run_name, loss, crop_size in variants:
            run_dir = self._run_directory("ablation", model, run_name)
            command = self._train_command(
                "ablation",
                model,
                loss=loss,
                run_name=run_name,
                crop_size=crop_size,
            )
            if (run_dir / "run_identity.json").is_file() or self._recovery_pointer_exists(
                "ablation", model, run_name
            ):
                command.append("--resume")
            try:
                results.append(self._run_command("ablation", f"train-{run_name}", command))
            except subprocess.CalledProcessError as error:
                stderr = str(error.stderr or "").lower()
                if "out of memory" not in stderr and "cuda oom" not in stderr:
                    raise
                retry = self._train_command(
                    "ablation",
                    model,
                    device_batch=2,
                    loss=loss,
                    run_name=run_name,
                    crop_size=crop_size,
                )
                retry = self._prepare_oom_retry("ablation", model, run_name, run_dir, retry)
                results.append(self._run_command("ablation", f"train-{run_name}-oom-retry", retry))
            checkpoint = self._checkpoint("ablation", model, run_name)
            for manifest in self.inputs.data_manifests:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                dataset = str(payload["dataset_id"])
                output = self.inputs.work_root / "evaluation/ablation" / model / run_name / dataset
                evidence = output / "evaluation.json"
                if evidence.is_file():
                    recorded = json.loads(evidence.read_text(encoding="utf-8"))
                    if recorded.get("checkpoint_sha256") != sha256_file(checkpoint):
                        raise ValueError("ablation evaluation checkpoint identity mismatch")
                    continue
                command = [
                    sys_executable(),
                    str(self.inputs.project_root / "scripts/evaluate.py"),
                    "run",
                    "--resolved-config",
                    str(run_dir / "resolved.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--dataset",
                    dataset,
                    "--dataset-manifest",
                    str(manifest),
                    "--role",
                    "train_select",
                    "--output-dir",
                    str(output),
                ]
                if self.inputs.rare_classes_file is not None:
                    command.extend(("--rare-classes-file", str(self.inputs.rare_classes_file)))
                results.append(
                    self._run_command("ablation", f"evaluate-{run_name}-{dataset}", command)
                )
        return results

    def _write_release_candidate(self) -> Path:
        """Write the deterministic five-model handoff for policy acceptance."""
        models: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []

        def relative_artifact(path: Path, *, run_id: str, kind: str) -> dict[str, Any]:
            try:
                relative = path.resolve().relative_to(self.inputs.work_root.resolve())
            except ValueError as error:
                raise ValueError("release candidate artifact is outside work_root") from error
            return {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "run_id": run_id,
                "kind": kind,
                "scientific_status": "measured",
            }

        for model in self.inputs.final_models:
            run_dir = self._run_directory("final", model)
            checkpoint = self._checkpoint("final", model)
            resolved = run_dir / "resolved.py"
            summary = run_dir / "summary.json"
            if not resolved.is_file() or not summary.is_file():
                raise FileNotFoundError(f"final release source is incomplete: {run_dir}")
            run_id = f"{self.inputs.campaign_id}-final-{model}-ce"
            checkpoint_record = relative_artifact(
                checkpoint, run_id=run_id, kind="final_checkpoint"
            )
            config_record = relative_artifact(resolved, run_id=run_id, kind="resolved_config")
            artifacts.extend(
                (
                    checkpoint_record,
                    config_record,
                    relative_artifact(summary, run_id=run_id, kind="training_summary"),
                )
            )
            models.append(
                {
                    "model": model,
                    "checkpoint": {
                        "path": checkpoint_record["path"],
                        "sha256": checkpoint_record["sha256"],
                    },
                    "resolved_config": {
                        "path": config_record["path"],
                        "sha256": config_record["sha256"],
                    },
                }
            )
        selection_path = self.inputs.work_root / "reports/selection/recommended_model.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection_record = relative_artifact(
            selection_path,
            run_id=f"{self.inputs.campaign_id}-selection",
            kind="recommended_model_selection",
        )
        artifacts.append(selection_record)
        selection_sources = [
            self.inputs.work_root / "reports/selection/candidate_table.json",
            *sorted((self.inputs.work_root / "evaluation/selection").glob("**/evaluation.json")),
            *sorted((self.inputs.work_root / "exports/selection").glob("**/*.validation.json")),
        ]
        for path in selection_sources:
            artifacts.append(
                relative_artifact(
                    path,
                    run_id=f"{self.inputs.campaign_id}-selection",
                    kind="selection_source_evidence",
                )
            )
        recommended = str(selection["recommended_model"])
        for run_name in ("median_frequency", "ce-256x512"):
            run_dir = self._run_directory("ablation", recommended, run_name)
            for path, kind in (
                (self._checkpoint("ablation", recommended, run_name), "ablation_checkpoint"),
                (run_dir / "resolved.py", "ablation_resolved_config"),
                (run_dir / "summary.json", "ablation_training_summary"),
            ):
                artifacts.append(
                    relative_artifact(
                        path,
                        run_id=f"{self.inputs.campaign_id}-ablation-{recommended}-{run_name}",
                        kind=kind,
                    )
                )
        candidate = {
            "schema_version": "3.0",
            "record_type": "edgeguard_release_candidate",
            "status": "candidate_requires_policy_acceptance",
            "campaign_id": self.inputs.campaign_id,
            "project_commit": self.inputs.project_commit,
            "data_manifest_sha256": self.identity["data_manifest_sha256"],
            "models": models,
            "artifacts": artifacts,
            "selection_sha256": sha256_file(selection_path),
            "recommended_model": recommended,
        }
        destination = self.inputs.work_root / "accepted_release.candidate.json"
        atomic_json(destination, candidate)
        return destination

    def _run_accept_phase(self) -> list[dict[str, Any]]:
        """Create and accept the exact release under the committed owner policy."""
        if self.inputs.release_policy is None or not self.inputs.release_policy.is_file():
            raise FileNotFoundError("automatic acceptance requires the release policy")
        if self.inputs.accepted_release is None:
            raise ValueError("automatic acceptance requires an output release path")
        candidate = self._write_release_candidate()
        selection = self.inputs.work_root / "reports/selection/recommended_model.json"
        accepted = accept_release_candidate_by_policy(
            candidate,
            self.inputs.release_policy,
            selection,
            self.inputs.accepted_release,
        )
        return [
            {
                "label": "policy-accept-release",
                "return_code": 0,
                "release_id": accepted["release_id"],
                "recommended_model": accepted["recommended_model"],
            }
        ]

    def _evaluation_manifest_outputs(self) -> dict[str, Path]:
        outputs = {
            path.name.removesuffix(".frozen.json"): path
            for path in self.inputs.evaluation_manifests
        }
        if set(outputs) != {"cityscapes", "idd20k"}:
            raise ValueError(
                "evaluation manifest filenames must be cityscapes.frozen.json and "
                "idd20k.frozen.json"
            )
        return outputs

    def _run_validation_data_phase(self) -> list[dict[str, Any]]:
        """Open and audit official source validation only after release acceptance."""
        self._accepted_release()
        if (
            self.inputs.authorization_policy is None
            or not self.inputs.authorization_policy.is_file()
        ):
            raise FileNotFoundError("official validation requires the authorization policy")
        roots = dict(self.inputs.data_roots)
        if set(roots) != {"cityscapes", "idd20k"}:
            raise ValueError("official validation requires both staged dataset roots")
        outputs = self._evaluation_manifest_outputs()
        results: list[dict[str, Any]] = []
        audit_base = self.inputs.work_root / "audit/official-validation"
        for dataset in ("cityscapes", "idd20k"):
            frozen = outputs[dataset]
            if frozen.is_file():
                validate_dataset_manifest(
                    frozen,
                    allowed_roles=("official_source_val",),
                )
                continue
            audit_root = audit_base / dataset
            report_name = "dataset_audit" if dataset == "cityscapes" else "idd20k_val_audit"
            candidate = audit_root / report_name / "dataset_manifest.candidate.json"
            report_root = audit_root / report_name
            if report_root.exists() and not candidate.is_file():
                quarantine = audit_root / f"{report_name}.incomplete-{time.time_ns()}"
                report_root.rename(quarantine)
            if not candidate.is_file():
                command = [
                    sys_executable(),
                    str(self.inputs.project_root / "scripts/audit_dataset.py"),
                    "--dataset",
                    dataset,
                    "--dataset-root",
                    str(roots[dataset]),
                    "--output-root",
                    str(audit_root),
                    "--source-split",
                    "val",
                ]
                for source in self.inputs.data_manifests:
                    command.extend(("--source-manifest", str(source)))
                results.append(
                    self._run_command("validation-data", f"audit-{dataset}-val", command)
                )
            freeze = [
                sys_executable(),
                str(self.inputs.project_root / "scripts/audit_dataset.py"),
                "--dataset",
                dataset,
                "--output-root",
                str(frozen.parent),
                "--split-manifest",
                str(candidate),
                "--freeze-approved",
                "--authorization-policy",
                str(self.inputs.authorization_policy),
                "--campaign-id",
                self.inputs.campaign_id,
                "--project-commit",
                self.inputs.project_commit,
            ]
            results.append(self._run_command("validation-data", f"freeze-{dataset}-val", freeze))
            validate_dataset_manifest(frozen, allowed_roles=("official_source_val",))
        return results

    def _run_evaluate_phase(
        self,
        release: dict[str, Any],
        models: tuple[AcceptedModelSource, ...],
    ) -> list[dict[str, Any]]:
        manifests = self._accepted_evaluation_manifests()
        results: list[dict[str, Any]] = []
        release_id = str(release["release_id"])
        for source in models:
            calibration_files: list[Path] = []
            for manifest in self.inputs.data_manifests:
                training_manifest = json.loads(manifest.read_text(encoding="utf-8"))
                dataset = str(training_manifest["dataset_id"])
                calibration_root = (
                    self.inputs.work_root
                    / "evaluation/calibration"
                    / release_id
                    / source.model
                    / dataset
                )
                calibration_evidence = calibration_root / "calibration-evidence.npz"
                calibration_files.append(calibration_evidence)
                if calibration_evidence.is_file():
                    continue
                if calibration_root.exists() and any(calibration_root.iterdir()):
                    raise ValueError(f"incomplete calibration evidence: {calibration_root}")
                command = [
                    sys_executable(),
                    str(self.inputs.project_root / "scripts/evaluate.py"),
                    "run",
                    "--resolved-config",
                    str(source.resolved_config),
                    "--checkpoint",
                    str(source.checkpoint),
                    "--dataset",
                    dataset,
                    "--dataset-manifest",
                    str(manifest),
                    "--role",
                    "train_calibration",
                    "--save-calibration-evidence",
                    str(calibration_evidence),
                    "--output-dir",
                    str(calibration_root),
                ]
                results.append(
                    self._run_command("evaluate", f"calibration-{source.model}-{dataset}", command)
                )
            temperature = (
                self.inputs.work_root
                / "calibration"
                / release_id
                / source.model
                / "global-temperature.json"
            )
            if not temperature.is_file():
                command = [
                    sys_executable(),
                    str(self.inputs.project_root / "scripts/evaluate.py"),
                    "calibrate-global",
                    "--output",
                    str(temperature),
                ]
                for evidence in calibration_files:
                    command.extend(("--evidence", str(evidence)))
                results.append(
                    self._run_command("evaluate", f"temperature-{source.model}", command)
                )
            for dataset, manifest in sorted(manifests.items()):
                output = (
                    self.inputs.work_root
                    / "evaluation/evaluate"
                    / release_id
                    / source.model
                    / dataset
                )
                evidence = output / "evaluation.json"
                if evidence.is_file():
                    payload = json.loads(evidence.read_text(encoding="utf-8"))
                    if (
                        payload.get("checkpoint_sha256") != sha256_file(source.checkpoint)
                        or payload.get("dataset_manifest_sha256") != sha256_file(manifest)
                        or payload.get("role") != "official_source_val"
                    ):
                        raise ValueError("existing accepted-release evaluation identity mismatch")
                    continue
                if output.exists() and any(output.iterdir()):
                    raise ValueError(
                        f"incomplete accepted-release evaluation requires review: {output}"
                    )
                command = [
                    sys_executable(),
                    str(self.inputs.project_root / "scripts/evaluate.py"),
                    "run",
                    "--resolved-config",
                    str(source.resolved_config),
                    "--checkpoint",
                    str(source.checkpoint),
                    "--dataset",
                    dataset,
                    "--dataset-manifest",
                    str(manifest),
                    "--role",
                    "official_source_val",
                    "--temperature-file",
                    str(temperature),
                    "--output-dir",
                    str(output),
                ]
                if self.inputs.rare_classes_file is not None:
                    command.extend(("--rare-classes-file", str(self.inputs.rare_classes_file)))
                results.append(self._run_command("evaluate", f"{source.model}-{dataset}", command))
        return results

    def _run_export_phase(
        self,
        release: dict[str, Any],
        models: tuple[AcceptedModelSource, ...],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        release_id = str(release["release_id"])
        for source in models:
            output = (
                self.inputs.work_root
                / "exports/export"
                / release_id
                / source.model
                / f"{source.model}.onnx"
            )
            validation = output.with_suffix(".validation.json")
            if output.is_file() and validation.is_file():
                payload = json.loads(validation.read_text(encoding="utf-8"))
                if (
                    payload.get("checkpoint_sha256") != sha256_file(source.checkpoint)
                    or payload.get("onnx_sha256") != sha256_file(output)
                    or payload.get("allclose_atol_1e_4_rtol_1e_4") is not True
                ):
                    raise ValueError("existing accepted-release ONNX identity mismatch")
                continue
            if output.parent.exists() and any(output.parent.iterdir()):
                raise ValueError(
                    f"incomplete accepted-release export requires review: {output.parent}"
                )
            selection = self.inputs.work_root / "exports/selection" / source.model
            selection_onnx = selection / f"{source.model}.onnx"
            selection_validation = selection_onnx.with_suffix(".validation.json")
            selection_payload = json.loads(selection_validation.read_text(encoding="utf-8"))
            if (
                selection_payload.get("checkpoint_sha256") != sha256_file(source.checkpoint)
                or selection_payload.get("onnx_sha256") != sha256_file(selection_onnx)
                or selection_payload.get("allclose_atol_1e_4_rtol_1e_4") is not True
            ):
                raise ValueError("selection ONNX cannot be reused for accepted export")
            output.parent.mkdir(parents=True)
            copied: list[str] = []
            for selection_source in sorted(selection.glob(f"{source.model}.*")):
                destination = output.parent / selection_source.name
                shutil.copy2(selection_source, destination)
                if sha256_file(destination) != sha256_file(selection_source):
                    raise OSError("accepted export copy hash mismatch")
                copied.append(destination.name)
            results.append(
                {
                    "label": source.model,
                    "return_code": 0,
                    "reused_hash_verified_selection_export": copied,
                }
            )
        return results

    def _run_report_phase(self, release: dict[str, Any]) -> list[dict[str, Any]]:
        assert self.inputs.accepted_release is not None
        output = self.inputs.work_root / "reports/report" / str(release["release_id"])
        manifest = output / "bundle_manifest.json"
        archive = output.with_suffix(".zip")
        gallery = self.inputs.work_root / "reports/gallery" / str(release["release_id"])
        results: list[dict[str, Any]] = []
        if not (gallery / "gallery_manifest.json").is_file():
            command = [
                sys_executable(),
                str(self.inputs.project_root / "scripts/generate_release_gallery.py"),
                "--accepted-release",
                str(self.inputs.accepted_release),
                "--work-root",
                str(self.inputs.work_root),
                "--output-root",
                str(gallery),
            ]
            for evaluation_manifest in self.inputs.evaluation_manifests:
                command.extend(("--evaluation-manifest", str(evaluation_manifest)))
            results.append(self._run_command("report", "measured-gallery", command))
        if manifest.is_file() and archive.is_file():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("source_release_sha256") != sha256_file(self.inputs.accepted_release):
                raise ValueError("existing thesis report belongs to another accepted release")
        elif output.exists() or archive.exists():
            raise ValueError(f"incomplete thesis report requires review: {output}")
        else:
            results.append(
                self._run_command(
                    "report",
                    "thesis-bundle",
                    [
                        sys_executable(),
                        str(self.inputs.project_root / "scripts/build_thesis_bundle.py"),
                        "--accepted-release",
                        str(self.inputs.accepted_release),
                        "--output-root",
                        str(output),
                        "--evidence-root",
                        str(self.inputs.work_root),
                    ],
                )
            )
        atomic_json(
            self._phase_root("report") / "report_delivery_verification.json",
            {
                "release_id": release["release_id"],
                "bundle_manifest_sha256": sha256_file(manifest),
                "thesis_archive_sha256": sha256_file(archive),
                "gallery_manifest_sha256": sha256_file(gallery / "gallery_manifest.json"),
            },
        )
        return results

    def _run_package_phase(self, release: dict[str, Any]) -> list[dict[str, Any]]:
        assert self.inputs.accepted_release is not None
        output = self.inputs.work_root / "deliveries" / str(release["release_id"])
        result = self._run_command(
            "package",
            "final-delivery",
            [
                sys_executable(),
                str(self.inputs.project_root / "scripts/package_colab_release.py"),
                "--accepted-release",
                str(self.inputs.accepted_release),
                "--work-root",
                str(self.inputs.work_root),
                "--project-root",
                str(self.inputs.project_root),
                "--output-root",
                str(output),
            ],
        )
        index = json.loads((output / "release_index.json").read_text(encoding="utf-8"))
        atomic_json(
            self._phase_root("package") / "delivery_verification.json",
            {
                "release_id": release["release_id"],
                "release_index_sha256": sha256_file(output / "release_index.json"),
                "packages": index["packages"],
            },
        )
        return [result]

    def _run_phase(self, phase: str) -> dict[str, Any]:
        started = time.perf_counter()
        if phase == "preflight":
            self._verify_preflight()
            commands: list[dict[str, Any]] = []
        elif phase == "restore":
            commands = self._restore_accepted_release_checkpoints()
        elif phase == "stage-data":
            commands = []
            for manifest in self.inputs.data_manifests:
                if not manifest.is_file():
                    raise FileNotFoundError(f"frozen local data manifest is missing: {manifest}")
        elif phase == "canary":
            self._verify_canary()
            commands = []
        elif phase in {"smoke", "pilot", "extension-smoke", "screening", "hpo", "final"}:
            commands = self._run_training_phase(phase)
        elif phase == "selection":
            commands = self._selection_evidence()
        elif phase == "ablation":
            commands = self._run_ablation_phase()
        elif phase == "accept":
            commands = self._run_accept_phase()
        elif phase == "validation-data":
            commands = self._run_validation_data_phase()
        elif phase in {"evaluate", "export", "report", "package"}:
            release, models = self._accepted_release()
            self._record_release_verification(phase, release, models)
            if phase == "evaluate":
                commands = self._run_evaluate_phase(release, models)
            elif phase == "export":
                commands = self._run_export_phase(release, models)
            elif phase == "report":
                commands = self._run_report_phase(release)
            else:
                commands = self._run_package_phase(release)
        else:
            raise ValueError(f"unsupported internal pipeline phase: {phase}")
        elapsed = time.perf_counter() - started
        scientific_status: Literal["measured", "accepted", "failed", "not_run"] = (
            "not_run"
            if phase
            in {
                "preflight",
                "restore",
                "stage-data",
                "canary",
                "smoke",
                "extension-smoke",
            }
            else "accepted"
            if phase in {"accept", "validation-data", "report", "package"}
            else "measured"
        )
        return self._complete_phase(
            phase,
            elapsed_seconds=elapsed,
            scientific_status=scientific_status,
            command_results=commands,
        )

    def run(self, target: str) -> dict[str, Any]:
        """Run missing prerequisites, skipping only hash-verified completions."""
        completed: list[str] = []
        skipped: list[str] = []
        for phase in phases_for_target(target):
            if self._phase_complete(phase):
                skipped.append(phase)
                continue
            try:
                self._run_phase(phase)
                completed.append(phase)
            except BaseException as error:
                failure = {
                    "schema_version": "2.0",
                    "record_type": "edgeguard_pipeline_failure",
                    "campaign_id": self.inputs.campaign_id,
                    "phase": phase,
                    "status": "failed",
                    "identity_sha256": self._phase_identity_sha256(phase),
                    "error_type": type(error).__name__,
                    "message": str(error)[:2000],
                    "safe_restart": f"rerun target {target}; verified phases will be skipped",
                }
                atomic_json(self._phase_root(phase) / "failure.json", failure)
                raise
        return {
            "schema_version": "2.0",
            "record_type": "edgeguard_colab_pipeline_result",
            "campaign_id": self.inputs.campaign_id,
            "target": target,
            "identity_sha256": self.identity_sha256,
            "completed": completed,
            "skipped_verified": skipped,
        }


def sys_executable() -> str:
    """Resolve the orchestrator interpreter without importing platform state elsewhere."""
    return sys.executable
