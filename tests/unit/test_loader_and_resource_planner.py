from __future__ import annotations

import pytest

from edgeguard.runtime_planner import resource_plan
from edgeguard.training.loader import (
    benchmark_fixture_loader,
    build_fixture_loader,
    recommended_worker_count,
)


def test_loader_mixed_sources_corrupt_recovery_incomplete_batch_and_seed() -> None:
    pytest.importorskip("torch")
    first = list(build_fixture_loader(sample_count=5, batch_size=2, corrupt_index=1, seed=4))
    second = list(build_fixture_loader(sample_count=5, batch_size=2, corrupt_index=1, seed=4))
    assert [item["sample_ids"] for item in first] == [item["sample_ids"] for item in second]
    assert sum(len(item["sample_ids"]) for item in first) == 4
    assert any(item["corrupt_sample_ids"] for item in first)
    assert len(first[-1]["sample_ids"]) <= 2
    assert {source for item in first for source in item["sources"]} == {
        "synthetic-a",
        "synthetic-b",
    }
    assert all(item["images"].shape[0] <= 2 for item in first)


def test_worker_policy_and_loader_benchmark() -> None:
    pytest.importorskip("torch")
    assert recommended_worker_count(cpu_count=8, system="Darwin") == 0
    assert recommended_worker_count(cpu_count=8, system="Linux") == 4
    report = benchmark_fixture_loader(workers=0, batches=2)
    assert report["images"] == 4
    assert report["images_per_second"] > 0


def test_resource_planner_preserves_batch_and_requires_measurement() -> None:
    profile = {
        "preferred_accelerator": "T4",
        "device_batch": 2,
        "gradient_accumulation": 4,
        "effective_batch": 8,
        "workers": 2,
        "prefetch": 2,
    }
    unmeasured = resource_plan(profile)
    assert unmeasured["estimated_wall_time_seconds"] is None
    measured = resource_plan(
        profile, observed={"images_per_second": 10.0, "planned_samples": 1000.0}
    )
    assert measured["estimated_wall_time_seconds"] == 100.0
    invalid = dict(profile, effective_batch=9)
    with pytest.raises(ValueError, match="effective batch"):
        resource_plan(invalid)
