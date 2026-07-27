"""Static command-contract tests for the five-model training smoke."""

from pathlib import Path

from scripts.train.run_semantic_smoke import MODEL_CONFIGS, build_train_command


def test_smoke_command_is_bounded_random_initialized_and_resume_protected(tmp_path: Path) -> None:
    command = build_train_command(
        tmp_path / "python",
        tmp_path / "project",
        config_root=tmp_path / "configs",
        model_config=tmp_path / "fast_scnn.yaml",
        mmseg_checkout=tmp_path / "mmseg",
        project_commit="a" * 40,
        dataset_root=tmp_path / "dataset",
        dataset_manifest=tmp_path / "dataset.json",
        split_policy_manifest=tmp_path / "split.json",
        output_dir=tmp_path / "run",
        recovery_dir=tmp_path / "recovery",
        device_batch=2,
        gradient_accumulation=2,
    )
    serialized = " ".join(command)

    assert len(MODEL_CONFIGS) == 5
    assert "--max-optimizer-steps 100" in serialized
    assert "--validation-subset-size 25" in serialized
    assert "--smoke-random-initialization" in command
    assert "--recovery-sync-dir" in command
    assert "--resume" not in command
