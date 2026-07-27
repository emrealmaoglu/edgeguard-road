"""Deterministic experiment, checkpoint, and resume identities."""

from __future__ import annotations

from typing import Any, Literal

from edgeguard.serialization import sha256_payload
from edgeguard.training.contracts import (
    CheckpointMetadata,
    DatasetIdentity,
    ProjectStatus,
    SemanticCommonConfig,
    SemanticExperimentContract,
    SemanticFrameworkConfig,
    SemanticModelConfig,
)


def build_experiment_contract(
    framework: SemanticFrameworkConfig,
    common: SemanticCommonConfig,
    model: SemanticModelConfig,
    *,
    dataset: DatasetIdentity,
    git_commit: str,
    environment: dict[str, Any],
    precision_mode: Literal["fp32", "fp16", "bf16"] = "fp32",
    status: ProjectStatus = ProjectStatus.IMPLEMENTED,
) -> SemanticExperimentContract:
    """Resolve deterministic scientific inputs without runtime paths or timestamps."""
    framework_sha = sha256_payload(framework.model_dump(mode="json"))
    common_sha = sha256_payload(common.model_dump(mode="json"))
    model_sha = sha256_payload(model.model_dump(mode="json"))
    config_payload = {
        "common": common.model_dump(mode="json"),
        "model": model.model_dump(mode="json"),
    }
    config_sha = sha256_payload(config_payload)
    is_synthetic = dataset.kind == "synthetic_stack_fixture"
    initialization_type = (
        model.initialization.stack_probe if is_synthetic else model.initialization.project_training
    )
    initialization_sha = None if is_synthetic else model.initialization.source.sha256
    fingerprint_payload = {
        "config_sha256": config_sha,
        "dataset": dataset.model_dump(mode="json"),
        "framework_identity_sha256": framework_sha,
        "git_commit": git_commit,
        "initialization_checkpoint_sha256": initialization_sha,
        "ontology_sha256": common.ontology_sha256,
        "precision_mode": precision_mode,
    }
    return SemanticExperimentContract(
        experiment_id=model.experiment_id,
        model_family=model.model_family,
        framework_identity_sha256=framework_sha,
        model_config_sha256=model_sha,
        common_config_sha256=common_sha,
        config_sha256=config_sha,
        experiment_fingerprint=sha256_payload(fingerprint_payload),
        initialization_type=initialization_type,
        initialization_checkpoint_sha256=initialization_sha,
        ontology_version=common.ontology_version,
        ontology_sha256=common.ontology_sha256,
        dataset=dataset,
        training_seed=common.seed,
        input_crop=common.input_crop,
        effective_global_batch=common.effective_global_batch,
        device_batch=common.device_batch,
        gradient_accumulation=common.gradient_accumulation,
        optimizer=common.optimizer,
        scheduler=common.scheduler,
        training_epochs=common.training_epochs,
        optimizer_steps=common.planned_optimizer_steps,
        augmentations=common.augmentations,
        precision_mode=precision_mode,
        loss_configuration=common.loss,
        auxiliary_loss_configuration=common.auxiliary_loss,
        checkpoint_interval=common.checkpoint.interval_optimizer_steps,
        validation_interval=common.checkpoint.validation_interval_epochs,
        git_commit=git_commit,
        git_dirty=False,
        environment=environment,
        status=status,
    )


def validate_resume_identity(
    expected: SemanticExperimentContract, checkpoint: CheckpointMetadata
) -> None:
    """Reject any checkpoint whose immutable training identity differs."""
    expected_values = {
        "experiment_id": expected.experiment_id,
        "config_sha256": expected.config_sha256,
        "experiment_fingerprint": expected.experiment_fingerprint,
        "dataset_manifest_sha256": expected.dataset.dataset_manifest_sha256,
        "split_manifest_sha256": expected.dataset.split_manifest_sha256,
        "initialization_checkpoint_sha256": expected.initialization_checkpoint_sha256,
        "model_family": expected.model_family,
        "framework_identity_sha256": expected.framework_identity_sha256,
        "git_commit": expected.git_commit,
        "precision_mode": expected.precision_mode,
        "seed": expected.training_seed,
    }
    actual = checkpoint.model_dump()
    mismatches = [key for key, value in expected_values.items() if actual[key] != value]
    if mismatches:
        raise ValueError(f"checkpoint resume identity mismatch: {', '.join(sorted(mismatches))}")
