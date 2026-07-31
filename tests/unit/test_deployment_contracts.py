from pathlib import Path

from edgeguard.deployment.contracts import validate_jetson_profiles, validate_preregistration

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_experiment_preregistration_preserves_scientific_boundaries() -> None:
    result = validate_preregistration(
        PROJECT_ROOT / "configs" / "experiment" / "project_preregistration.yaml"
    )
    assert result["status"] == "valid"
    assert result["payload"]["allowed_holdout_inspections"] == 1


def test_jetson_profiles_remain_unassigned_and_int8_gated() -> None:
    result = validate_jetson_profiles(
        PROJECT_ROOT / "configs" / "deployment" / "jetson_profiles.yaml"
    )
    assert result["status"] == "valid"
    assert result["payload"]["assignment_gate"].endswith("jetson_measurement")
