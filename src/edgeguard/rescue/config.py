"""Validated configuration for the semantic-first delivery path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """One supported MMSeg model and its upstream configuration."""

    name: str
    upstream_config: str


@dataclass(frozen=True)
class StageConfig:
    """A bounded training stage expressed in steps or epochs."""

    name: str
    epochs: int | None
    max_steps: int | None

    def __post_init__(self) -> None:
        if (self.epochs is None) == (self.max_steps is None):
            raise ValueError(f"stage {self.name!r} must define exactly one of epochs/max_steps")
        value = self.epochs if self.epochs is not None else self.max_steps
        if value is None or value <= 0:
            raise ValueError(f"stage {self.name!r} budget must be positive")


@dataclass(frozen=True)
class HPOConfig:
    """Bounded, reproducible search contract for the two finalists."""

    trials_per_model: int
    max_steps: int
    pruning_steps: tuple[int, ...]
    learning_rate: tuple[float, float]
    weight_decay: tuple[float, float]
    schedulers: tuple[str, ...]
    warmup_ratios: tuple[float, ...]
    sampler_seed: int


@dataclass(frozen=True)
class DatasetProtocol:
    """Approved source and sealed-evaluation dataset roles."""

    training: tuple[str, ...]
    domain_sampling: str
    external_sealed: tuple[str, ...]
    external_fallback: str


@dataclass(frozen=True)
class RescueConfig:
    """Decision-complete shared protocol for all semantic experiments."""

    schema_version: str
    seed: int
    num_classes: int
    ignore_index: int
    crop_size: tuple[int, int]
    base_scale: tuple[int, int]
    device_batch: int
    gradient_accumulation: int
    workers: int
    learning_rate: float
    weight_decay: float
    models: tuple[ModelConfig, ...]
    stages: dict[str, StageConfig]
    hpo: HPOConfig
    datasets: DatasetProtocol

    @property
    def effective_batch(self) -> int:
        """Return the common effective global batch for a single device."""
        return self.device_batch * self.gradient_accumulation


def _pair(value: Any, field: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must contain exactly two integers")
    pair = (int(value[0]), int(value[1]))
    if min(pair) <= 0:
        raise ValueError(f"{field} values must be positive")
    return pair


def load_rescue_config(path: Path) -> RescueConfig:
    """Load the small semantic-first YAML contract and reject ambiguous values."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("rescue config must be a YAML mapping")
    models_payload = payload.get("models")
    stages_payload = payload.get("stages")
    if not isinstance(models_payload, list) or not isinstance(stages_payload, dict):
        raise ValueError("rescue config must define models and stages")
    models = tuple(
        ModelConfig(name=str(item["name"]), upstream_config=str(item["upstream_config"]))
        for item in models_payload
    )
    if len(models) != 5 or len({model.name for model in models}) != 5:
        raise ValueError("multi-domain protocol requires exactly five unique models")
    stages = {
        str(name): StageConfig(
            name=str(name),
            epochs=int(item["epochs"]) if "epochs" in item else None,
            max_steps=int(item["max_steps"]) if "max_steps" in item else None,
        )
        for name, item in stages_payload.items()
    }
    required_stages = {"smoke", "pilot", "screening", "hpo", "final"}
    if set(stages) != required_stages:
        raise ValueError(f"stages must be exactly {sorted(required_stages)}")
    config = RescueConfig(
        schema_version=str(payload["schema_version"]),
        seed=int(payload["seed"]),
        num_classes=int(payload["num_classes"]),
        ignore_index=int(payload["ignore_index"]),
        crop_size=_pair(payload["crop_size"], "crop_size"),
        base_scale=_pair(payload["base_scale"], "base_scale"),
        device_batch=int(payload["device_batch"]),
        gradient_accumulation=int(payload["gradient_accumulation"]),
        workers=int(payload["workers"]),
        learning_rate=float(payload["optimizer"]["learning_rate"]),
        weight_decay=float(payload["optimizer"]["weight_decay"]),
        models=models,
        stages=stages,
        hpo=HPOConfig(
            trials_per_model=int(payload["hpo"]["trials_per_model"]),
            max_steps=int(payload["hpo"]["max_steps"]),
            pruning_steps=tuple(int(value) for value in payload["hpo"]["pruning_steps"]),
            learning_rate=tuple(float(value) for value in payload["hpo"]["learning_rate"]),  # type: ignore[arg-type]
            weight_decay=tuple(float(value) for value in payload["hpo"]["weight_decay"]),  # type: ignore[arg-type]
            schedulers=tuple(str(value) for value in payload["hpo"]["schedulers"]),
            warmup_ratios=tuple(float(value) for value in payload["hpo"]["warmup_ratios"]),
            sampler_seed=int(payload["hpo"]["sampler_seed"]),
        ),
        datasets=DatasetProtocol(
            training=tuple(str(value) for value in payload["datasets"]["training"]),
            domain_sampling=str(payload["datasets"]["domain_sampling"]),
            external_sealed=tuple(str(value) for value in payload["datasets"]["external_sealed"]),
            external_fallback=str(payload["datasets"]["external_fallback"]),
        ),
    )
    if config.schema_version != "1.0":
        raise ValueError("unsupported rescue config schema version")
    if config.num_classes != 19 or config.ignore_index != 255:
        raise ValueError("semantic-first protocol is frozen to Cityscapes 19/ignore 255")
    if min(config.device_batch, config.gradient_accumulation) <= 0 or config.workers < 0:
        raise ValueError("batch values must be positive and workers cannot be negative")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("optimizer values are invalid")
    if any(stage.epochs is not None for stage in config.stages.values()):
        raise ValueError("multi-domain stages must use comparable optimizer-step budgets")
    if config.hpo.trials_per_model != 12 or config.hpo.max_steps != 6_000:
        raise ValueError("HPO budget must remain frozen at 12 trials and 6,000 steps")
    if config.hpo.pruning_steps != (1_500, 3_000):
        raise ValueError("HPO pruning rungs must remain 1,500 and 3,000 steps")
    if config.hpo.schedulers != ("poly", "cosine"):
        raise ValueError("HPO scheduler choices must be poly and cosine")
    if config.datasets.domain_sampling != "uniform":
        raise ValueError("multi-domain sampling must remain domain-uniform")
    if config.datasets.training != ("cityscapes", "idd20k"):
        raise ValueError(
            "the active scientific campaign is frozen to Cityscapes and IDD20K; "
            "provisional BDD100K cannot enter model selection"
        )
    return config


def model_by_name(config: RescueConfig, name: str) -> ModelConfig:
    """Return one configured model or fail with a user-facing supported list."""
    for model in config.models:
        if model.name == name:
            return model
    supported = ", ".join(model.name for model in config.models)
    raise ValueError(f"unsupported model {name!r}; choose one of: {supported}")
