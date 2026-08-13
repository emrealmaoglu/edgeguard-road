"""Tests for the semantic-cs-idd-v3 Colab orchestrator contract."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import edgeguard.rescue.colab_pipeline as colab_pipeline_module
from edgeguard.rescue.colab_pipeline import (
    ABLATION_MAX_STEPS,
    ALL_MODELS,
    CORE_MODELS,
    EXTENSION_MODELS,
    ColabPipeline,
    PipelineInputs,
    models_for_phase,
    phases_for_target,
)
from edgeguard.rescue.colab_recovery import publish_recovery_file
from edgeguard.serialization import sha256_file, sha256_payload

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_staged_dataset_manifest(root: Path, *, dataset_id: str) -> Path:
    """Write a real, schema-valid manifest whose referenced files exist on disk."""
    root.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(root / "img.png")
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(root / "mask.png")
    record = {
        "sample_id": f"{dataset_id}-s0",
        "group_id": f"{dataset_id}-g0",
        "image": "img.png",
        "mask": "mask.png",
        "canonical_mask": "mask.png",
    }
    payload: dict[str, Any] = {
        "schema_version": "2.0",
        "record_type": "edgeguard_dataset_manifest",
        "dataset_id": dataset_id,
        "split_state": "frozen",
        "dataset_root": str(root),
        "prepared_root": str(root),
        "roles": {"train_fit": [record]},
    }
    payload["manifest_sha256"] = sha256_payload(payload)
    manifest_path = root.parent / f"{dataset_id}.frozen.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def _pipeline(
    tmp_path: Path,
    *,
    screening_models: tuple[str, ...] = ALL_MODELS,
    final_models: tuple[str, ...] = ALL_MODELS,
    pretrained_manifest_root: Path | None = None,
) -> ColabPipeline:
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
    manifests = (
        _write_staged_dataset_manifest(tmp_path / "cityscapes-data", dataset_id="cityscapes"),
        _write_staged_dataset_manifest(tmp_path / "idd20k-data", dataset_id="idd20k"),
    )
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
            screening_models=screening_models,
            final_models=final_models,
            pretrained_manifest_root=pretrained_manifest_root,
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


def test_pipeline_accepts_an_evidence_selected_final_model_subset(tmp_path: Path) -> None:
    subset = (CORE_MODELS[0], CORE_MODELS[2], EXTENSION_MODELS[0])
    pipeline = _pipeline(tmp_path / "subset", final_models=subset)
    assert pipeline.inputs.final_models == subset


def test_pipeline_rejects_reordered_final_model_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen five-model relative order"):
        _pipeline(tmp_path / "reordered", final_models=tuple(reversed(ALL_MODELS)))


def test_pipeline_rejects_empty_final_model_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _pipeline(tmp_path / "empty", final_models=())


def test_pipeline_rejects_duplicate_final_models(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not contain duplicates"):
        _pipeline(tmp_path / "dup", final_models=(CORE_MODELS[0], CORE_MODELS[0]))


def test_pipeline_accepts_an_evidence_selected_screening_model_subset(tmp_path: Path) -> None:
    subset = tuple(model for model in ALL_MODELS if model != "bisenetv2")
    pipeline = _pipeline(tmp_path / "screening-subset", screening_models=subset)
    assert pipeline.inputs.screening_models == subset
    assert pipeline.inputs.final_models == ALL_MODELS


def test_pipeline_rejects_reordered_screening_model_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen five-model relative order"):
        _pipeline(tmp_path / "screening-reordered", screening_models=tuple(reversed(ALL_MODELS)))


def test_pipeline_rejects_empty_screening_model_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _pipeline(tmp_path / "screening-empty", screening_models=())


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


def test_recovery_self_test_runs_once_per_campaign_not_once_per_model(tmp_path: Path) -> None:
    """The intentional-interrupt self-test only fires for
    PipelineInputs.recovery_self_test_model (default CORE_MODELS[0]).
    Every other model's smoke/extension-smoke command must train without
    ever deliberately crashing itself -- five deliberate crashes per
    campaign is what let bugs 3/5/6 each take the whole campaign down.
    """
    pipeline = _pipeline(tmp_path)
    assert pipeline.inputs.recovery_self_test_model == CORE_MODELS[0]

    self_test_command = pipeline._train_command("smoke", CORE_MODELS[0])  # noqa: SLF001
    assert "--intentional-interrupt-step" in self_test_command

    for model in CORE_MODELS[1:]:
        command = pipeline._train_command("smoke", model)  # noqa: SLF001
        assert "--intentional-interrupt-step" not in command

    for model in EXTENSION_MODELS:
        command = pipeline._train_command("extension-smoke", model)  # noqa: SLF001
        assert "--intentional-interrupt-step" not in command


def test_failed_recovery_self_test_restarts_the_model_instead_of_killing_the_campaign(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces the sixth real bug's failure MODE (not the RNG bug itself):
    the self-test's own resume leg raises. The campaign must record durable
    evidence and restart the model from scratch, not propagate the error and
    kill the other four models.
    """
    pipeline = _pipeline(tmp_path)
    model = CORE_MODELS[0]
    run_dir = pipeline._run_directory("smoke", model)  # noqa: SLF001
    calls: list[list[str]] = []

    monkeypatch.setattr(colab_pipeline_module, "models_for_phase", lambda phase: (model,))

    def fake_run_command(phase: str, label: str, command: list[str]) -> dict[str, object]:
        calls.append(list(command))
        if len(calls) == 1:
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
        if len(calls) == 2:
            # The self-test's own resume leg fails on real hardware.
            raise subprocess.CalledProcessError(
                1, command, output="RNG state must be a torch.ByteTensor", stderr="TypeError"
            )
        return {"label": label, "return_code": 0, "command": command}

    monkeypatch.setattr(pipeline, "_run_command", fake_run_command)
    results = pipeline._run_training_phase("smoke")  # noqa: SLF001

    assert len(calls) == 3
    restart_command = calls[2]
    assert "--resume" not in restart_command
    assert "--intentional-interrupt-step" not in restart_command
    assert results[0]["interruption_resume"]["verified"] is False
    assert results[0]["interruption_resume"]["outcome"] == "restarted_without_resume"

    quarantined = sorted(run_dir.parent.glob(f"{run_dir.name}.recovery-self-test-*"))
    assert len(quarantined) == 1
    failure_record = json.loads((quarantined[0] / "recovery_self_test_failure.json").read_text())
    assert failure_record["record_type"] == "edgeguard_recovery_self_test_failure"
    phase_root_failure = pipeline._phase_root("smoke") / "recovery_self_test_failure.json"  # noqa: SLF001
    assert phase_root_failure.is_file()


def test_pilot_refuses_to_start_after_an_unresolved_recovery_self_test_failure(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(tmp_path)
    failure_path = pipeline.state_root / "smoke" / "recovery_self_test_failure.json"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(
        json.dumps({"record_type": "edgeguard_recovery_self_test_failure"}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="recovery self-test failed"):
        pipeline._run_phase("pilot")  # noqa: SLF001


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


def test_train_command_requests_pretrained_initialisation_only_where_a_manifest_exists(
    tmp_path: Path,
) -> None:
    """Models with a committed manifest transfer ImageNet weights; the rest stay random.

    `fast_scnn` and `bisenetv2` declare no `init_cfg` of type `Pretrained` upstream, so
    there is nothing to transfer and asking for `--initialization pretrained` would make
    `train_model` fail loudly rather than train. Absence of a manifest must therefore stay
    a silent, legitimate fall back to random initialisation.
    """
    manifest_root = tmp_path / "pretrained"
    manifest_root.mkdir(parents=True)
    (manifest_root / "segformer_b0.json").write_text("{}", encoding="utf-8")
    pipeline = _pipeline(tmp_path / "case", pretrained_manifest_root=manifest_root)

    with_manifest = pipeline._train_command("screening", "segformer_b0")
    assert "--initialization" in with_manifest
    assert with_manifest[with_manifest.index("--initialization") + 1] == "pretrained"
    manifest_index = with_manifest.index("--pretrained-manifest") + 1
    assert with_manifest[manifest_index] == str(manifest_root / "segformer_b0.json")

    without_manifest = pipeline._train_command("screening", "fast_scnn")
    assert "--initialization" not in without_manifest
    assert "--pretrained-manifest" not in without_manifest


def test_train_command_omits_initialisation_when_no_manifest_root_is_configured(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(tmp_path / "no-root")
    command = pipeline._train_command("screening", "segformer_b0")
    assert "--initialization" not in command


def test_ablation_runs_on_its_own_shorter_step_budget(tmp_path: Path) -> None:
    """Ablations are directional evidence, so they must not inherit the final budget."""
    pipeline = _pipeline(tmp_path / "ablation")
    command = pipeline._train_command("ablation", "segformer_b0", max_steps=ABLATION_MAX_STEPS)
    assert command[command.index("--max-steps") + 1] == str(ABLATION_MAX_STEPS)
    assert ABLATION_MAX_STEPS < 10_000
