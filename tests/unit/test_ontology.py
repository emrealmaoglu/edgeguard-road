from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from edgeguard.config import DuplicateConfigKeyError
from edgeguard.data.ontology import load_project_ontology
from edgeguard.serialization import sha256_payload

ONTOLOGY_PATH = Path("configs/dataset/ontology_v1.yaml")


def _payload() -> dict[str, Any]:
    loaded = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write_payload(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "ontology.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_project_ontology_loads_complete_namespaces_and_bdd_mappings() -> None:
    ontology = load_project_ontology(ONTOLOGY_PATH)

    assert ontology.ontology_version == "edgeguard-ontology-v1"
    assert ontology.ontology_status == "provisional"
    assert len(ontology.semantic_cityscapes19.classes) == 19
    assert len(ontology.known_detection10.classes) == 10
    assert len(ontology.ood_binary.classes) == 2
    assert len(ontology.risk_operational.classes) == 3
    assert len(ontology.source_mappings) == 10
    assert ontology.source_class_policy.unmapped_action == "reject"
    assert ontology.source_class_policy.ignored_source_classes == ()
    assert ontology.source_class_policy.unsupported_source_classes == ()
    assert {mapping.source_name: mapping.project_name for mapping in ontology.source_mappings} == {
        "pedestrian": "person",
        "rider": "rider",
        "car": "car",
        "truck": "truck",
        "bus": "bus",
        "train": "train",
        "motor": "motorcycle",
        "bike": "bicycle",
        "traffic light": "traffic_light",
        "traffic sign": "traffic_sign",
    }
    assert all(mapping.source_id is None for mapping in ontology.source_mappings)


def test_project_ontology_has_stable_canonical_payload_hash() -> None:
    first = load_project_ontology(ONTOLOGY_PATH)
    second = load_project_ontology(ONTOLOGY_PATH)

    assert sha256_payload(first.model_dump(mode="json")) == sha256_payload(
        second.model_dump(mode="json")
    )


@pytest.mark.parametrize("field", ["id", "name"])
def test_project_ontology_rejects_duplicate_project_classes(tmp_path: Path, field: str) -> None:
    payload = _payload()
    classes = payload["known_detection10"]["classes"]
    classes[1][field] = classes[0][field]

    expected_message = "duplicate project IDs" if field == "id" else "duplicate project names"
    with pytest.raises(ValidationError, match=expected_message):
        load_project_ontology(_write_payload(tmp_path, payload))


def test_project_ontology_rejects_unknown_mapping_action(tmp_path: Path) -> None:
    payload = _payload()
    payload["source_mappings"][0]["action"] = "drop"

    with pytest.raises(ValidationError, match="Input should be 'map'"):
        load_project_ontology(_write_payload(tmp_path, payload))


def test_project_ontology_rejects_duplicate_source_mapping(tmp_path: Path) -> None:
    payload = _payload()
    payload["source_mappings"].append(deepcopy(payload["source_mappings"][0]))

    with pytest.raises(ValidationError, match="duplicate source mappings"):
        load_project_ontology(_write_payload(tmp_path, payload))


def test_project_ontology_rejects_inconsistent_mapping_target(tmp_path: Path) -> None:
    payload = _payload()
    payload["source_mappings"][0]["project_name"] = "rider"

    with pytest.raises(ValidationError, match="target ID/name"):
        load_project_ontology(_write_payload(tmp_path, payload))


def test_project_ontology_rejects_absolute_paths(tmp_path: Path) -> None:
    payload = _payload()
    payload["source_mappings"][0]["reason"] = "/absolute/source"

    with pytest.raises(ValidationError, match="must not contain absolute"):
        load_project_ontology(_write_payload(tmp_path, payload))


def test_project_ontology_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    text = ONTOLOGY_PATH.read_text(encoding="utf-8")
    path = tmp_path / "ontology.yaml"
    path.write_text(
        text.replace('schema_version: "1.0"', 'schema_version: "1.0"\nschema_version: "1.0"'),
        encoding="utf-8",
    )

    with pytest.raises(DuplicateConfigKeyError, match="Duplicate YAML key"):
        load_project_ontology(path)
