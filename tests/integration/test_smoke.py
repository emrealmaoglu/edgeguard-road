"""Integration tests for the synthetic smoke vertical slice."""

import json
from pathlib import Path

from edgeguard.cli import main
from edgeguard.config import load_smoke_config
from edgeguard.contracts import ArtifactManifest
from edgeguard.smoke import build_smoke_result

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = REPO_ROOT / "configs/smoke.yaml"


def test_normal_runs_have_unique_ids_and_stable_fingerprint() -> None:
    config = load_smoke_config(SMOKE_CONFIG)

    first = build_smoke_result(config, config_path=SMOKE_CONFIG, deterministic=False)
    second = build_smoke_result(config, config_path=SMOKE_CONFIG, deterministic=False)

    assert first.metadata.run_id != second.metadata.run_id
    assert first.metadata.experiment_fingerprint == second.metadata.experiment_fingerprint
    assert first.scientific_payload == second.scientific_payload


def test_deterministic_smoke_is_repeatable() -> None:
    config = load_smoke_config(SMOKE_CONFIG)

    first = build_smoke_result(config, config_path=SMOKE_CONFIG, deterministic=True)
    second = build_smoke_result(config, config_path=SMOKE_CONFIG, deterministic=True)

    assert first == second
    assert first.metadata.execution_mode == "deterministic_smoke"


def test_cli_writes_one_valid_jsonl_record(tmp_path: Path) -> None:
    output = tmp_path / "result.jsonl"

    exit_code = main(
        ["smoke", "--config", str(SMOKE_CONFIG), "--output", str(output), "--deterministic"]
    )

    assert exit_code == 0
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["record_type"] == "smoke_result"
    assert payload["scientific_payload"]["anomaly_shape"] == [1, 32, 64]


def test_example_artifact_manifest_matches_schema() -> None:
    payload = json.loads(
        (REPO_ROOT / "artifacts/manifest.example.json").read_text(encoding="utf-8")
    )

    manifest = ArtifactManifest.model_validate(payload)

    assert manifest.git_state.value == "unborn"
    assert manifest.git_commit is None
