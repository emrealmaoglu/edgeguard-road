"""Stable stage graph and profile contracts for EdgeGuard campaigns."""

from __future__ import annotations

from dataclasses import dataclass

STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "preflight": (),
    "storage_inventory": ("preflight",),
    "dataset_prepare": ("storage_inventory",),
    "semantic_compatibility": ("dataset_prepare",),
    "semantic_smoke": ("semantic_compatibility",),
    "semantic_screening": ("semantic_smoke",),
    "semantic_medium": ("semantic_screening",),
    "semantic_hpo": ("semantic_medium",),
    "semantic_final": ("semantic_hpo",),
    "zero_shot_ood": ("semantic_smoke",),
    "temperature_calibration": ("zero_shot_ood",),
    "trainable_anomaly_head": ("temperature_calibration",),
    "detection_smoke": ("dataset_prepare",),
    "detection_training": ("detection_smoke",),
    "contextual_risk": ("zero_shot_ood", "detection_training"),
    "temporal_fusion": ("contextual_risk",),
    "export_probe": ("semantic_smoke",),
    "final_evaluation": (
        "semantic_final",
        "trainable_anomaly_head",
        "temporal_fusion",
        "export_probe",
    ),
    "report_generation": ("final_evaluation",),
}

STAGE_STATUSES = frozenset(
    {"pending", "ready", "running", "completed", "failed", "blocked", "skipped"}
)


@dataclass(frozen=True)
class CampaignProfile:
    """Execution scale without duplicating scientific logic."""

    name: str
    synthetic: bool
    optimizer_steps: int
    image_height: int
    image_width: int
    scientific_execution: bool


PROFILES = {
    "local-mini": CampaignProfile("local-mini", True, 3, 32, 64, False),
    "linux-cpu": CampaignProfile("linux-cpu", True, 3, 32, 64, False),
    "colab": CampaignProfile("colab", False, 100, 512, 1024, True),
    "jetson": CampaignProfile("jetson", False, 0, 512, 1024, True),
}


def topological_stages() -> tuple[str, ...]:
    """Return the reviewed, deterministic execution order."""
    return tuple(STAGE_DEPENDENCIES)
