"""Project-specific ontology validation for EdgeGuard-Road."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from edgeguard.config import StrictConfigModel, UniqueKeySafeLoader

NamespaceName = Literal[
    "semantic_cityscapes19",
    "known_detection10",
    "ood_binary",
    "risk_operational",
]

_EXPECTED_NAMES: dict[NamespaceName, tuple[str, ...]] = {
    "semantic_cityscapes19": (
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic_light",
        "traffic_sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ),
    "known_detection10": (
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "traffic_light",
        "traffic_sign",
    ),
    "ood_binary": ("id", "anomaly"),
    "risk_operational": ("low", "medium", "high"),
}


class OntologyClass(StrictConfigModel):
    """One class inside a project namespace."""

    id: int = Field(ge=0)
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")


class OntologyNamespace(StrictConfigModel):
    """One independent class namespace with optional ignore identifier."""

    ignore_id: int | None = Field(default=None, ge=0)
    classes: tuple[OntologyClass, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_classes(self) -> OntologyNamespace:
        """Reject duplicate IDs/names and ignore-ID collisions."""
        ids = [item.id for item in self.classes]
        names = [item.name for item in self.classes]
        if len(ids) != len(set(ids)):
            raise ValueError("ontology namespace contains duplicate project IDs")
        if len(names) != len(set(names)):
            raise ValueError("ontology namespace contains duplicate project names")
        if self.ignore_id is not None and self.ignore_id in ids:
            raise ValueError("ontology ignore ID collides with a project class ID")
        return self


class SourceClassPolicy(StrictConfigModel):
    """Fail-closed policy for source classes not listed in mappings."""

    unmapped_action: Literal["reject"]
    ignored_source_classes: tuple[str, ...]
    unsupported_source_classes: tuple[str, ...]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_policy_names(self) -> SourceClassPolicy:
        """Keep ignored and unsupported source declarations unambiguous."""
        combined = [name.casefold() for name in self.ignored_source_classes]
        combined.extend(name.casefold() for name in self.unsupported_source_classes)
        if len(combined) != len(set(combined)):
            raise ValueError("source class policy contains duplicate class names")
        return self


class SourceMapping(StrictConfigModel):
    """One explicit source-to-project detection mapping."""

    source_dataset: Literal["BDD100K"]
    source_task: Literal["object_detection"]
    source_id: int | None = Field(default=None, ge=0)
    source_name: str = Field(min_length=1)
    project_namespace: NamespaceName
    project_id: int = Field(ge=0)
    project_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    action: Literal["map"]
    reason: str = Field(min_length=1)
    mapping_version: Literal["bdd100k-detection-to-known_detection10-v1"]


class DeferredSource(StrictConfigModel):
    """A source whose real label specification must be reviewed first."""

    source_dataset: Literal["Mapillary Vistas"]
    source_task: Literal["semantic_segmentation"]
    action: Literal["human_review_required"]
    reason: str = Field(min_length=1)


class ProjectOntology(StrictConfigModel):
    """Complete EdgeGuard-Road ontology source of truth."""

    schema_version: Literal["1.0"]
    ontology_version: Literal["edgeguard-ontology-v1"]
    ontology_status: Literal["provisional"]
    semantic_cityscapes19: OntologyNamespace
    known_detection10: OntologyNamespace
    ood_binary: OntologyNamespace
    risk_operational: OntologyNamespace
    source_class_policy: SourceClassPolicy
    source_mappings: tuple[SourceMapping, ...] = Field(min_length=1)
    deferred_sources: tuple[DeferredSource, ...]

    @model_validator(mode="after")
    def validate_project_contract(self) -> ProjectOntology:
        """Require exact namespaces, complete BDD mappings, and path-free values."""
        namespaces: dict[NamespaceName, OntologyNamespace] = {
            "semantic_cityscapes19": self.semantic_cityscapes19,
            "known_detection10": self.known_detection10,
            "ood_binary": self.ood_binary,
            "risk_operational": self.risk_operational,
        }
        for name, expected_names in _EXPECTED_NAMES.items():
            namespace = namespaces[name]
            sorted_classes = sorted(namespace.classes, key=lambda item: item.id)
            actual = tuple(item.name for item in sorted_classes)
            expected_ids = tuple(range(len(expected_names)))
            actual_ids = tuple(item.id for item in sorted_classes)
            if actual_ids != expected_ids or actual != expected_names:
                raise ValueError(f"{name} must preserve the exact project ID/name contract")

        if self.semantic_cityscapes19.ignore_id != 255 or self.ood_binary.ignore_id != 255:
            raise ValueError("semantic and OOD namespaces must use ignore ID 255")
        if (
            self.known_detection10.ignore_id is not None
            or self.risk_operational.ignore_id is not None
        ):
            raise ValueError("detection and risk namespaces do not define an ignore ID")

        seen_sources: set[tuple[str, str, str, str]] = set()
        mapped_source_names: set[str] = set()
        for mapping in self.source_mappings:
            key = (
                mapping.source_dataset.casefold(),
                mapping.source_task.casefold(),
                mapping.source_name.casefold(),
                mapping.mapping_version,
            )
            if key in seen_sources:
                raise ValueError("ontology contains duplicate source mappings")
            seen_sources.add(key)
            mapped_source_names.add(mapping.source_name)
            target = namespaces[mapping.project_namespace]
            targets = {item.id: item.name for item in target.classes}
            if targets.get(mapping.project_id) != mapping.project_name:
                raise ValueError("source mapping target ID/name does not match its namespace")
            if mapping.project_namespace != "known_detection10":
                raise ValueError("BDD100K detection mappings must target known_detection10")

        expected_sources = {
            "pedestrian",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motor",
            "bike",
            "traffic light",
            "traffic sign",
        }
        if mapped_source_names != expected_sources:
            raise ValueError("BDD100K detection mappings must cover the exact reviewed names")

        for value in _iter_strings(self.model_dump(mode="json")):
            if value.startswith(("/", "~/")) or re.match(r"^[A-Za-z]:[\\/]", value):
                raise ValueError("ontology must not contain absolute or user-relative paths")
        return self


def _iter_strings(value: Any) -> list[str]:
    """Collect strings from JSON-compatible model data for a path-safety check."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            result.extend(_iter_strings(key))
            result.extend(_iter_strings(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_iter_strings(item))
        return result
    return []


def load_project_ontology(path: Path) -> ProjectOntology:
    """Load the project ontology with safe YAML and duplicate-key rejection."""
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.load(stream, Loader=UniqueKeySafeLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"Ontology must be a YAML mapping: {path}")
    return ProjectOntology.model_validate(payload)
