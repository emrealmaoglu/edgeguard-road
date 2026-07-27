"""Typed, dependency-light contracts for semantic training experiments."""

from __future__ import annotations

from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from edgeguard.config import StrictConfigModel

SHA256_PATTERN = r"^[0-9a-f]{64}$"
COMMIT_PATTERN = r"^[0-9a-f]{40}$"
EXPERIMENT_ID_PATTERN = r"^EGX-SEG-[A-Z0-9][A-Z0-9-]*$"


class ProjectStatus(str, Enum):
    """Project-wide implementation and measurement status vocabulary."""

    PLANNED = "planned"
    IMPLEMENTED = "implemented"
    LOCALLY_TESTED = "locally_tested"
    COLAB_MEASURED = "colab_measured"
    JETSON_MEASURED = "jetson_measured"
    HUMAN_ACCEPTED = "human_accepted"
    BLOCKED = "blocked"
    FAILED = "failed"


class ModelFamily(str, Enum):
    """The five semantic candidates in the common laboratory."""

    FAST_SCNN = "fast_scnn"
    BISENET_V2 = "bisenetv2"
    PIDNET_S = "pidnet_s"
    DDRNET_23_SLIM = "ddrnet_23_slim"
    SEGFORMER_B0 = "segformer_b0"


class SemanticFrameworkConfig(StrictConfigModel):
    """Pinned framework proposal whose runtime compatibility needs a Colab probe."""

    schema_version: Literal["1.0"]
    record_type: Literal["semantic_framework_config"]
    backend: Literal["mmsegmentation"]
    repository_url: Literal["https://github.com/open-mmlab/mmsegmentation.git"]
    release: Literal["v1.2.2"]
    commit: str = Field(pattern=COMMIT_PATTERN)
    mmengine_version: str = Field(pattern=r"^0\.[0-9]+\.[0-9]+$")
    mmcv_version: str = Field(pattern=r"^2\.[0-9]+\.[0-9]+$")
    openmim_version: str = Field(pattern=r"^0\.[0-9]+\.[0-9]+$")
    supported_mmengine: Literal[">=0.5.0,<1.0.0"]
    supported_mmcv: Literal[">=2.0.0rc4,<2.2.0"]
    torch_resolution: Literal["colab_runtime_compatible"]
    cuda_wheel_policy: Literal["runtime_resolved_no_hardcoded_wheel"]
    install_scope: Literal["isolated_colab_runtime_only"]
    source_license: Literal["Apache-2.0"]
    status: Literal["proposal_pending_colab_probe"]


class DatasetRoleContract(StrictConfigModel):
    """Immutable role boundary consumed from EG-DATA-002 outputs."""

    train_role: Literal["train_fit"]
    selection_role: Literal["train_select"]
    calibration_role: Literal["train_calibration"]
    common_evaluation_role: Literal["official_val_common_eval"]
    routine_model_selection_roles: tuple[Literal["train_select"]]
    forbidden_routine_roles: tuple[
        Literal["train_calibration"], Literal["official_val_common_eval"]
    ]
    group_identity: Literal["city+sequence"]
    selected_split_required_for_training: Literal[True]


class AugmentationConfig(StrictConfigModel):
    """Common baseline augmentation family; later additions require an ablation."""

    random_scale: tuple[float, float]
    crop: tuple[Literal[512], Literal[1024]]
    horizontal_flip_probability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    color_jitter_brightness: float = Field(ge=0.0, le=0.5, allow_inf_nan=False)
    color_jitter_contrast: float = Field(ge=0.0, le=0.5, allow_inf_nan=False)
    color_jitter_saturation: float = Field(ge=0.0, le=0.5, allow_inf_nan=False)
    color_jitter_hue: float = Field(ge=0.0, le=0.25, allow_inf_nan=False)
    validation_preprocessing: Literal["deterministic"]
    categorical_resize: Literal["nearest"]

    @model_validator(mode="after")
    def validate_scale_range(self) -> AugmentationConfig:
        """Require the approved ordered random-scale interval."""
        if self.random_scale != (0.5, 2.0):
            raise ValueError("baseline random_scale must remain [0.5, 2.0]")
        return self


class OptimizerConfig(StrictConfigModel):
    """Human-reviewable common optimizer proposal."""

    name: Literal["AdamW", "SGD"]
    learning_rate: float = Field(gt=0.0, allow_inf_nan=False)
    weight_decay: float = Field(ge=0.0, allow_inf_nan=False)
    momentum: float | None = Field(default=None, gt=0.0, lt=1.0, allow_inf_nan=False)


class SchedulerConfig(StrictConfigModel):
    """Bounded schedule used by a resolved experiment."""

    name: Literal["poly"]
    power: float = Field(gt=0.0, allow_inf_nan=False)
    warmup_steps: int = Field(ge=0)


class CheckpointPolicy(StrictConfigModel):
    """Interruption-safe checkpoint policy without a recovery platform."""

    interval_optimizer_steps: int = Field(gt=0)
    validation_interval_epochs: int = Field(gt=0)
    keep_last: Literal[True]
    keep_best: Literal[True]
    recovery_interval_optimizer_steps: int = Field(gt=0)
    refuse_nonempty_output: Literal[True]
    exact_identity_resume: Literal[True]


class SemanticCommonConfig(StrictConfigModel):
    """Path-free common training policy shared by the five model specs."""

    schema_version: Literal["1.0"]
    record_type: Literal["semantic_common_config"]
    project_name: Literal["edgeguard-road"]
    experiment_family: Literal["EGX-SEG-STACK"]
    ontology_version: Literal["edgeguard-ontology-v1"]
    ontology_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_roles: DatasetRoleContract
    seed: int = Field(ge=0, le=2**32 - 1)
    input_crop: tuple[Literal[512], Literal[1024]]
    device_batch: int = Field(gt=0)
    gradient_accumulation: int = Field(gt=0)
    effective_global_batch: int = Field(gt=0)
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training_epochs: int = Field(gt=0)
    planned_optimizer_steps: int = Field(gt=0)
    augmentations: AugmentationConfig
    precision_modes: tuple[Literal["fp32", "fp16", "bf16"], ...]
    loss: Literal["CrossEntropyLoss"]
    auxiliary_loss: Literal["CrossEntropyLoss"]
    ignore_index: Literal[255]
    checkpoint: CheckpointPolicy
    synthetic_fixture_identity: Literal["synthetic-semantic-stack-v1"]
    status: Literal["implemented"]

    @model_validator(mode="after")
    def validate_batch_and_precision(self) -> SemanticCommonConfig:
        """Keep effective batch explicit and FP32 as the correctness baseline."""
        if self.effective_global_batch != self.device_batch * self.gradient_accumulation:
            raise ValueError("effective_global_batch must equal device_batch * accumulation")
        if not self.precision_modes or self.precision_modes[0] != "fp32":
            raise ValueError("fp32 must be the first precision mode")
        if len(set(self.precision_modes)) != len(self.precision_modes):
            raise ValueError("precision_modes must be unique")
        return self


class PretrainedSource(StrictConfigModel):
    """Exact pretrained identity, or an explicit unresolved human input."""

    status: Literal["not_applicable", "unresolved_human_input", "resolved"]
    source: str | None = None
    license: str | None = None
    filename_or_model_id: str | None = None
    revision: str | None = None
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    access_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    redistribution_rule: str | None = None

    @model_validator(mode="after")
    def validate_resolution(self) -> PretrainedSource:
        """Never allow a partially specified source to masquerade as resolved."""
        values = (
            self.source,
            self.license,
            self.filename_or_model_id,
            self.revision,
            self.redistribution_rule,
        )
        if self.status == "resolved" and (
            any(value is None for value in values) or self.sha256 is None
        ):
            raise ValueError("resolved pretrained sources require complete provenance")
        if self.status != "resolved" and any(value is not None for value in (*values, self.sha256)):
            raise ValueError("unresolved or inapplicable pretrained sources must remain empty")
        return self


class InitializationPolicy(StrictConfigModel):
    """Separate no-download stack-probe initialization from future training policy."""

    stack_probe: Literal["random"]
    project_training: Literal["random", "pretrained"]
    source: PretrainedSource
    notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source_policy(self) -> InitializationPolicy:
        """Require exact provenance before a pretrained project run can start."""
        if self.project_training == "random" and self.source.status != "not_applicable":
            raise ValueError("random project training must use a not_applicable source")
        if self.project_training == "pretrained" and self.source.status == "not_applicable":
            raise ValueError("pretrained project training requires a source decision")
        return self


class LogitsContractConfig(StrictConfigModel):
    """Minimum direct-logit handoff expected from each framework model."""

    output_kind: Literal["native_logits"]
    direct_pre_softmax: Literal[True]
    layout: Literal["NCHW"]
    dtype: Literal["float32"]
    class_count: Literal[19]
    native_resolution: Literal["model_defined_record_at_runtime"]
    alignment_mode: Literal["bilinear"]
    alignment_target: Literal["edgeguard_common_grid"]
    align_corners: bool


class SemanticModelConfig(StrictConfigModel):
    """One model-specific spec over the common semantic laboratory."""

    schema_version: Literal["1.0"]
    record_type: Literal["semantic_model_config"]
    experiment_id: str = Field(pattern=EXPERIMENT_ID_PATTERN)
    model_family: ModelFamily
    architecture: str = Field(min_length=1)
    backbone: str = Field(min_length=1)
    decode_head: str = Field(min_length=1)
    auxiliary_heads: tuple[str, ...]
    mmseg_config_relative_path: str
    num_classes: Literal[19]
    ignore_index: Literal[255]
    normalization: Literal["ImageNet RGB mean/std"]
    baseline_crop: tuple[Literal[512], Literal[1024]]
    initialization: InitializationPolicy
    logits: LogitsContractConfig
    export_risk_notes: str = Field(min_length=1)
    expected_training_memory_notes: str = Field(min_length=1)

    @field_validator("mmseg_config_relative_path")
    @classmethod
    def require_safe_mmseg_path(cls, value: str) -> str:
        """Keep the official config reference checkout-relative."""
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not value.startswith("configs/"):
            raise ValueError("mmseg config path must be a safe checkout-relative path")
        return value


class DatasetIdentity(StrictConfigModel):
    """Real selected manifests or a clearly non-scientific synthetic fixture."""

    kind: Literal["real_selected_split", "synthetic_stack_fixture"]
    dataset_manifest_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    split_manifest_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    synthetic_fixture_identity: str | None = None

    @model_validator(mode="after")
    def validate_identity_kind(self) -> DatasetIdentity:
        """Prevent a synthetic probe from impersonating a real selected split."""
        if self.kind == "real_selected_split":
            if self.dataset_manifest_sha256 is None or self.split_manifest_sha256 is None:
                raise ValueError("real training requires dataset and selected-split hashes")
            if self.synthetic_fixture_identity is not None:
                raise ValueError("real training cannot carry a synthetic fixture identity")
        elif (
            self.synthetic_fixture_identity is None
            or self.dataset_manifest_sha256 is not None
            or self.split_manifest_sha256 is not None
        ):
            raise ValueError("synthetic probes require only a fixture identity")
        return self


class PolicySelectedSplit(StrictConfigModel):
    """Cryptographic identity for one automatically selected diversity split."""

    schema_version: Literal["1.0"]
    record_type: Literal["cityscapes_fine_policy_selected_split"]
    status: Literal["policy_selected"]
    policy_version: Literal["cityscapes-diversity-policy-v1"]
    policy_config: dict[str, Any]
    policy_config_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_id: Literal["CSF-SPLIT-D", "CSF-SPLIT-E"]
    candidate_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    ontology_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate: dict[str, Any]
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)


class SemanticTrainingSample(StrictConfigModel):
    """One root-free Cityscapes sample assigned by an accepted group split."""

    sample_id: str = Field(pattern=r"^[a-z]+_\d{6}_\d{6}$")
    group_id: str = Field(pattern=r"^[a-z]+_\d{6}$")
    role: Literal["train_fit", "train_select", "train_calibration"]
    image_relative_path: str
    train_id_relative_path: str

    @field_validator("image_relative_path", "train_id_relative_path")
    @classmethod
    def require_root_relative_sample_path(cls, value: str) -> str:
        """Reject absolute roots, traversal, and non-POSIX separators."""
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("training sample paths must be dataset-root-relative")
        return value


class SemanticExperimentContract(StrictConfigModel):
    """Resolved path-free experiment identity and training policy."""

    schema_version: Literal["1.0"] = "1.0"
    record_type: Literal["semantic_experiment_contract"] = "semantic_experiment_contract"
    experiment_id: str = Field(pattern=EXPERIMENT_ID_PATTERN)
    model_family: ModelFamily
    framework_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    model_config_sha256: str = Field(pattern=SHA256_PATTERN)
    common_config_sha256: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    experiment_fingerprint: str = Field(pattern=SHA256_PATTERN)
    initialization_type: Literal["random", "pretrained"]
    initialization_checkpoint_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    ontology_version: Literal["edgeguard-ontology-v1"]
    ontology_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset: DatasetIdentity
    training_seed: int = Field(ge=0, le=2**32 - 1)
    input_crop: tuple[int, int]
    effective_global_batch: int = Field(gt=0)
    device_batch: int = Field(gt=0)
    gradient_accumulation: int = Field(gt=0)
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training_epochs: int = Field(gt=0)
    optimizer_steps: int = Field(gt=0)
    augmentations: AugmentationConfig
    precision_mode: Literal["fp32", "fp16", "bf16"]
    loss_configuration: str
    auxiliary_loss_configuration: str
    checkpoint_interval: int = Field(gt=0)
    validation_interval: int = Field(gt=0)
    git_commit: str = Field(pattern=COMMIT_PATTERN)
    git_dirty: Literal[False]
    environment: dict[str, Any]
    status: ProjectStatus


class CheckpointMetadata(StrictConfigModel):
    """Exact-resume identity stored beside a framework checkpoint."""

    schema_version: Literal["1.0"] = "1.0"
    record_type: Literal["semantic_checkpoint_metadata"] = "semantic_checkpoint_metadata"
    experiment_id: str = Field(pattern=EXPERIMENT_ID_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    experiment_fingerprint: str = Field(pattern=SHA256_PATTERN)
    dataset_manifest_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    split_manifest_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    initialization_checkpoint_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    model_family: ModelFamily
    framework_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=COMMIT_PATTERN)
    precision_mode: Literal["fp32", "fp16", "bf16"]
    seed: int = Field(ge=0, le=2**32 - 1)
    epoch: int = Field(ge=0)
    optimizer_step: int = Field(ge=0)
    best_metric: float | None = Field(default=None, allow_inf_nan=False)
    last_metric: float | None = Field(default=None, allow_inf_nan=False)
    contains_optimizer_state: Literal[True]
    contains_scheduler_state: Literal[True]
    contains_amp_scaler_state: bool


class GeneralizationGapInputs(StrictConfigModel):
    """Loss and selection-quality inputs retained for later gap analysis."""

    train_loss: float = Field(ge=0.0, allow_inf_nan=False)
    train_select_loss: float = Field(ge=0.0, allow_inf_nan=False)
    train_select_miou: float = Field(ge=0.0, le=100.0, allow_inf_nan=False)


class ValidationIntervalRecord(StrictConfigModel):
    """Required semantic training evidence written after each validation interval."""

    schema_version: Literal["1.0"] = "1.0"
    record_type: Literal["semantic_validation_interval"] = "semantic_validation_interval"
    epoch: int = Field(gt=0)
    optimizer_step: int = Field(gt=0)
    train_loss: float = Field(ge=0.0, allow_inf_nan=False)
    train_select_loss: float = Field(ge=0.0, allow_inf_nan=False)
    train_select_miou: float = Field(ge=0.0, le=100.0, allow_inf_nan=False)
    per_class_iou: tuple[float | None, ...]
    learning_rate: float = Field(gt=0.0, allow_inf_nan=False)
    metric_scale: Literal["percent_0_100"] = "percent_0_100"
    generalization_gap_inputs: GeneralizationGapInputs

    @field_validator("per_class_iou")
    @classmethod
    def require_nineteen_class_ious(
        cls, values: tuple[float | None, ...]
    ) -> tuple[float | None, ...]:
        """Require one finite percentage or explicit absence for every class."""
        if len(values) != 19:
            raise ValueError("per-class IoU must contain exactly 19 entries")
        if any(value is not None and not 0.0 <= value <= 100.0 for value in values):
            raise ValueError("per-class IoU values must be percentages in [0, 100]")
        return values


class ExperimentRegistryRecord(StrictConfigModel):
    """Small root-relative JSONL registry row for one run identity."""

    schema_version: Literal["1.0"] = "1.0"
    record_type: Literal["semantic_experiment_registry_record"] = (
        "semantic_experiment_registry_record"
    )
    experiment_id: str = Field(pattern=EXPERIMENT_ID_PATTERN)
    status: ProjectStatus
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=COMMIT_PATTERN)
    git_dirty: Literal[False]
    framework_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_manifest_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    split_manifest_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    initialization_checkpoint_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    seed: int = Field(ge=0, le=2**32 - 1)
    runtime: dict[str, Any]
    final_metrics: dict[str, float | int | None]
    last_metrics: dict[str, float | int | None]
    artifact_paths: tuple[str, ...]
    failure_summary: str | None = None

    @field_validator("artifact_paths")
    @classmethod
    def require_root_relative_artifacts(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Reject private absolute roots and parent traversal in registry entries."""
        for value in values:
            path = PurePosixPath(value)
            if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
                raise ValueError(f"artifact path must be external-root-relative: {value!r}")
        return values
