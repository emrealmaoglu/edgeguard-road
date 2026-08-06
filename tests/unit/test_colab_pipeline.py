"""Tests for the semantic-cs-idd-v3 Colab orchestrator contract."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

import edgeguard.rescue.colab_pipeline as colab_pipeline_module
from edgeguard.rescue.colab_pipeline import (
    ALL_MODELS,
    CORE_MODELS,
    EXTENSION_MODELS,
    ColabPipeline,
    PipelineInputs,
    models_for_phase,
    phases_for_target,
)
from edgeguard.rescue.colab_recovery import publish_recovery_file
from edgeguard.serialization import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pipeline(tmp_path: Path, *, final_models: tuple[str, ...] = ALL_MODELS) -> ColabPipeline:
    tmp_path.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runtime = tmp_path / "runtime_receipt.json"
    runtime.write_text(
        json.dumps(
            {
                "record_type": "semantic_hermetic_runtime_receipt",
                "runtime_profile": "py311-cu121",
                "project_commit": commit,
                "lock_sha256": {"lock": "a" * 64},
                "environment": {"cuda_available": True},
                "core_model_probe": {
                    "model_count": 5,
                    "fp16_finite_model_count": 5,
                    "checkpoint_resume_verified": True,
                    "checkpoint_resume_model_count": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    manifests = (tmp_path / "cityscapes.frozen.json", tmp_path / "idd20k.frozen.json")
    for manifest in manifests:
        manifest.write_text('{"status":"accepted"}\n', encoding="utf-8")
    config = tmp_path / "semantic_first.yaml"
    config.write_text("seed: 20260728\n", encoding="utf-8")
    mmseg = tmp_path / "mmsegmentation"
    mmseg.mkdir()
    return ColabPipeline(
        PipelineInputs(
            project_root=REPO_ROOT,
            project_commit=commit,
            runtime_receipt=runtime,
            mmseg_root=mmseg,
            work_root=tmp_path / "work",
            recovery_root=tmp_path / "drive-recovery",
            config_path=config,
            data_manifests=manifests,
            final_models=final_models,
        )
    )


def test_target_closure_and_five_model_gate_are_frozen() -> None:
    assert phases_for_target("pilot") == (
        "preflight",
        "restore",
        "stage-data",
        "canary",
        "smoke",
        "pilot",
    )
    assert phases_for_target("all")[-3:] == ("export", "report", "package")
    assert "extension-smoke" in phases_for_target("screening")
    assert models_for_phase("canary") == ALL_MODELS
    assert models_for_phase("smoke") == CORE_MODELS
    assert models_for_phase("pilot") == CORE_MODELS
    assert models_for_phase("extension-smoke") == EXTENSION_MODELS
    assert models_for_phase("screening") == ALL_MODELS


def test_pipeline_rejects_any_partial_or_reordered_final_model_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="all five models"):
        _pipeline(tmp_path / "partial", final_models=CORE_MODELS)
    with pytest.raises(ValueError, match="all five models"):
        _pipeline(tmp_path / "reordered", final_models=tuple(reversed(ALL_MODELS)))


def _accepted_release(tmp_path: Path, pipeline: ColabPipeline) -> Path:
    release_root = tmp_path / "release"
    release_root.mkdir()
    evidence = release_root / "metrics.json"
    evidence.write_text('{"mIoU":0.6}\n', encoding="utf-8")
    models: list[dict[str, object]] = []
    for model in ALL_MODELS:
        checkpoint = release_root / f"{model}.pth"
        checkpoint.write_bytes(model.encode())
        resolved = release_root / f"{model}.py"
        resolved.write_text("model = {}\n", encoding="utf-8")
        models.append(
            {
                "model": model,
                "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint)},
                "resolved_config": {"path": resolved.name, "sha256": sha256_file(resolved)},
            }
        )
    release = release_root / "accepted_release.json"
    release.write_text(
        json.dumps(
            {
                "record_type": "edgeguard_accepted_release",
                "release_id": "semantic-cs-idd-v3-fixture",
                "status": "accepted",
                "campaign_id": "semantic-cs-idd-v3",
                "project_commit": pipeline.inputs.project_commit,
                "data_manifest_sha256": pipeline.identity["data_manifest_sha256"],
                "models": models,
                "artifacts": [
                    {
                        "path": evidence.name,
                        "sha256": sha256_file(evidence),
                        "scientific_status": "accepted",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return release


def test_downstream_phases_require_exact_hash_verified_five_model_release(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(tmp_path)
    with pytest.raises(PermissionError, match="--accepted-release"):
        pipeline._accepted_release()  # noqa: SLF001

    release = _accepted_release(tmp_path, pipeline)
    accepted = ColabPipeline(replace(pipeline.inputs, accepted_release=release))
    payload, models = accepted._accepted_release()  # noqa: SLF001
    assert payload["release_id"] == "semantic-cs-idd-v3-fixture"
    assert tuple(model.model for model in models) == ALL_MODELS

    models[0].checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact identity mismatch"):
        accepted._accepted_release()  # noqa: SLF001


def test_restore_recovers_all_hash_bound_release_checkpoints(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    release = _accepted_release(tmp_path, pipeline)
    accepted = ColabPipeline(replace(pipeline.inputs, accepted_release=release))
    _, models = accepted._accepted_release()  # noqa: SLF001
    expected = {source.model: source.checkpoint.read_bytes() for source in models}
    for source in models:
        publish_recovery_file(
            source.checkpoint,
            accepted.inputs.recovery_root,
            artifact_id=f"final-{source.model.replace('_', '-')}-ce",
            campaign_id="semantic-cs-idd-v3",
            project_commit=accepted.inputs.project_commit,
        )
        source.checkpoint.unlink()
    restored = accepted._restore_accepted_release_checkpoints()  # noqa: SLF001
    assert len(restored) == 5
    assert {source.model: source.checkpoint.read_bytes() for source in models} == expected


def test_preflight_data_and_canary_resume_only_with_verified_artifacts(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    for phase in ("preflight", "restore", "stage-data", "canary"):
        pipeline._run_phase(phase)  # noqa: SLF001
        assert pipeline._phase_complete(phase)  # noqa: SLF001

    metrics = pipeline.state_root / "stage-data/metrics.json"
    metrics.write_text("{}\n", encoding="utf-8")
    assert not pipeline._phase_complete("stage-data")  # noqa: SLF001


def test_smoke_artifacts_are_never_scientific_results(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline._complete_phase(  # noqa: SLF001
        "smoke", elapsed_seconds=1.0, scientific_status="not_run", command_results=[]
    )
    manifest = json.loads((pipeline.state_root / "smoke/run_manifest.json").read_text())
    index = json.loads((pipeline.state_root / "smoke/artifact_index.json").read_text())
    assert manifest["synthetic_or_smoke"] is True
    assert manifest["scientific_status"] == "not_run"
    assert index["scientific_status"] == "not_run"


def test_smoke_command_enforces_cut_and_acceptance_command_stays_short(tmp_path: Path) -> None:
    production = _pipeline(tmp_path)
    command = production._train_command("smoke", "segformer_b0")  # noqa: SLF001
    assert command[-2:] == ["--intentional-interrupt-step", "25"]

    acceptance = ColabPipeline(replace(production.inputs, execution_mode="acceptance"))
    command = acceptance._train_command("final", "segformer_b0")  # noqa: SLF001
    assert command[command.index("--max-steps") + 1] == "2"
    assert command[command.index("--precision") + 1] == "fp32"


def test_oom_retry_interruption_resumes_with_same_proven_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pipeline = _pipeline(tmp_path)
    model = "segformer_b0"
    run_dir = pipeline._run_directory("smoke", model)  # noqa: SLF001
    calls: list[list[str]] = []

    monkeypatch.setattr(colab_pipeline_module, "models_for_phase", lambda phase: (model,))

    def fake_run_command(phase: str, label: str, command: list[str]) -> dict[str, object]:
        calls.append(list(command))
        if len(calls) == 1:
            raise subprocess.CalledProcessError(
                1, command, output="CUDA out of memory", stderr="CUDA out of memory"
            )
        if len(calls) == 2:
            run_dir.mkdir(parents=True)
            (run_dir / ".intentional-interruption-complete.json").write_text(
                json.dumps(
                    {
                        "record_type": "edgeguard_intentional_interruption",
                        "optimizer_step": 25,
                        "checkpoint_sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
            raise subprocess.CalledProcessError(
                1,
                command,
                output="EDGEGUARD_INTENTIONAL_INTERRUPTION",
                stderr="EDGEGUARD_INTENTIONAL_INTERRUPTION",
            )
        return {"label": label, "return_code": 0, "command": command}

    monkeypatch.setattr(pipeline, "_run_command", fake_run_command)
    results = pipeline._run_training_phase("smoke")  # noqa: SLF001

    assert len(calls) == 3
    assert "--device-batch" in calls[1]
    assert calls[1][calls[1].index("--device-batch") + 1] == "2"
    assert "--resume" not in calls[1]
    assert calls[2][-1] == "--resume"
    assert results[0]["oom_recovery"] == {
        "attempts": 1,
        "device_batch": 2,
        "effective_batch": 4,
    }
    assert results[0]["interruption_resume"] == {
        "verified": True,
        "optimizer_step": 25,
        "checkpoint_sha256": "a" * 64,
    }


def test_acceptance_mode_never_writes_measured_or_accepted_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    production = _pipeline(tmp_path)
    pipeline = ColabPipeline(replace(production.inputs, execution_mode="acceptance"))
    monkeypatch.setattr(pipeline, "_run_training_phase", lambda phase: [])

    result = pipeline._run_phase("pilot")  # noqa: SLF001

    assert result["status"] == "completed"
    manifest = json.loads((pipeline.state_root / "pilot/run_manifest.json").read_text())
    assert manifest["scientific_status"] == "not_run"
