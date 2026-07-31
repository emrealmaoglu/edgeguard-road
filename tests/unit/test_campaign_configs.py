"""Static contracts for local and future campaign execution profiles."""

from pathlib import Path

import yaml

from edgeguard.config import UniqueKeySafeLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, object]:
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeySafeLoader)
    assert isinstance(payload, dict)
    return payload


def test_campaign_profiles_scale_execution_not_stage_logic() -> None:
    profiles = {
        path.stem: _load(path)
        for path in sorted((PROJECT_ROOT / "configs/campaign").glob("*.yaml"))
    }
    assert set(profiles) == {"local-mini", "linux-cpu", "colab", "jetson"}
    assert {profile["stage_graph"] for profile in profiles.values()} == {"edgeguard-bounded-v1"}
    assert profiles["local-mini"]["scientific_execution"] is False
    assert profiles["colab"]["runtime_paths"] == "supplied_at_runtime"
    assert profiles["jetson"]["engine_policy"] == "build_on_target_only"


def test_semantic_phase_order_and_random_final_contract() -> None:
    suite = _load(PROJECT_ROOT / "configs/training/semantic_campaign_phases.yaml")
    phases = suite["phases"]
    assert isinstance(phases, list)
    assert [phase["id"] for phase in phases] == [
        "smoke",
        "short_screening",
        "medium",
        "limited_hpo",
        "final",
    ]
    assert phases[-1]["minimum_random_initialization_runs"] == 1
    assert suite["official_cityscapes_val_routine_tuning"] is False


def test_detection_and_export_contracts_are_download_free_and_target_safe() -> None:
    detection = _load(PROJECT_ROOT / "configs/training/detection_campaign.yaml")
    pipeline = _load(PROJECT_ROOT / "configs/pipeline/risk_temporal_export.yaml")
    assert detection["detector_families"] == ["yolo11n", "rt_detr_r18"]
    assert detection["pretrained_weights_required_for_local_mini"] is False
    assert pipeline["export"]["local_tensorrt"] is False
    assert pipeline["export"]["jetson_engine_built_on_target"] is True
