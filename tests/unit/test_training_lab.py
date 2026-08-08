"""Dependency-light tests for the pinned semantic training laboratory."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from edgeguard.data.cityscapes_split_policy import POLICY_CONFIG, POLICY_VERSION
from edgeguard.runtime import RuntimePathContract
from edgeguard.serialization import sha256_payload
from edgeguard.telemetry.longrun import LongRunStatus
from edgeguard.training.config import (
    load_semantic_common_config,
    load_semantic_framework_config,
    load_semantic_model_suite,
)
from edgeguard.training.contracts import (
    CheckpointMetadata,
    DatasetIdentity,
    ExperimentRegistryRecord,
    ModelFamily,
    ProjectStatus,
    ValidationIntervalRecord,
)
from edgeguard.training.data import (
    load_policy_selected_cityscapes_split,
    samples_for_training_role,
)
from edgeguard.training.identity import build_experiment_contract, validate_resume_identity
from edgeguard.training.logits import validate_native_logits_tensor
from edgeguard.training.registry import append_registry, load_registry
from scripts.train.install_semantic_stack import (
    BootstrapError,
    _environment_probe,
    _record_failure,
    _resolve_uv_executable,
    _validate_lock_contract,
    build_hermetic_commands,
    install_hermetic_runtime,
    repair_owned_path,
)
from scripts.train.train_semantic import (
    _atomic_sync_checkpoint,
    _validation_interval_record,
    run_stack_probe,
    validate_configs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs/training/segmentation"
COMMIT = "1" * 40
SHA = "2" * 64


def test_environment_probe_names_the_first_failed_import_and_preserves_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "scripts.train.install_semantic_stack.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="probe stdout",
            stderr="exact loader failure",
        ),
    )
    with pytest.raises(RuntimeError, match="exact loader failure") as captured:
        _environment_probe(tmp_path / "python", tmp_path / "mmseg")
    assert "failed for cv2" in str(captured.value)


def test_environment_probe_preserves_cuda_initialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls <= 13:
            return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="ok", stderr="")
        return subprocess.CompletedProcess(
            args=args[0], returncode=1, stdout="", stderr="undefined symbol: ncclCommRegister"
        )

    monkeypatch.setattr("scripts.train.install_semantic_stack.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="ncclCommRegister") as captured:
        _environment_probe(tmp_path / "python", tmp_path / "mmseg")
    assert "CUDA initialization failed" in str(captured.value)


def test_verified_bootstrap_receipt_skips_second_dependency_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    paths = RuntimePathContract.from_workspace(workspace)
    interpreter = paths.runtime_root / "bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    uv = workspace / "bootstrap/local/bin/uv"
    uv.parent.mkdir(parents=True)
    uv.touch()
    config = load_semantic_framework_config(CONFIG_ROOT / "framework_mmseg.yaml")
    lock_paths = (
        REPO_ROOT / config.lockfile,
        REPO_ROOT / "requirements/colab-openmmlab.lock",
    )
    from edgeguard.serialization import sha256_file

    bootstrap_receipt = workspace / "bootstrap-receipt.json"
    bootstrap_receipt.write_text(
        json.dumps(
            {
                "record_type": "edgeguard_stdlib_colab_bootstrap",
                "status": "completed",
                "project_commit": COMMIT,
                "interpreter": str(interpreter),
                "mmseg_commit": config.commit,
                "lock_sha256": {path.name: sha256_file(path) for path in lock_paths},
                "uv": {"version": "0.8.8", "path": str(uv)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.train.install_semantic_stack.repair_owned_path", lambda **kwargs: []
    )

    def fake_execute(commands: tuple[tuple[str, ...], ...], **kwargs: object) -> dict[str, object]:
        assert commands == ()
        return {
            "runtime_profile": "py311-cu121",
            "commands": [],
            "core_model_probe": {"evidence_package": "probe.zip"},
        }

    monkeypatch.setattr("scripts.train.install_semantic_stack._execute_runtime", fake_execute)

    def fake_package(evidence_root: Path, receipt: dict[str, object]) -> Path:
        package = evidence_root / "runtime.zip"
        package.parent.mkdir(parents=True, exist_ok=True)
        package.write_bytes(b"runtime")
        return package

    monkeypatch.setattr("scripts.train.install_semantic_stack._evidence_zip", fake_package)
    result = install_hermetic_runtime(
        CONFIG_ROOT / "framework_mmseg.yaml",
        paths=paths,
        project_root=REPO_ROOT,
        project_commit=COMMIT,
        config_root=CONFIG_ROOT,
        bootstrap_receipt=bootstrap_receipt,
    )
    assert result["commands"] == []
    assert result["preinstalled_bootstrap_verified"] is True


def _suite() -> tuple[object, object, tuple[object, ...]]:
    return (
        load_semantic_framework_config(CONFIG_ROOT / "framework_mmseg.yaml"),
        load_semantic_common_config(CONFIG_ROOT / "common_cityscapes.yaml"),
        load_semantic_model_suite(CONFIG_ROOT),
    )


def _contract() -> object:
    framework, common, models = _suite()
    return build_experiment_contract(
        framework,
        common,
        models[0],
        dataset=DatasetIdentity(
            kind="synthetic_stack_fixture",
            synthetic_fixture_identity="synthetic-semantic-stack-v1",
        ),
        git_commit=COMMIT,
        environment={"runtime": "synthetic-unit"},
    )


def test_five_model_configs_have_exact_contracts_and_unique_identities() -> None:
    framework, common, models = _suite()

    assert framework.commit == "c685fe6767c4cadf6b051983ca6208f1b9d1ccb8"
    assert framework.mmengine_version == "0.10.7"
    assert framework.mmcv_version == "2.1.0"
    assert {model.model_family for model in models} == set(ModelFamily)
    assert len({model.experiment_id for model in models}) == 5
    assert all(model.num_classes == 19 for model in models)
    assert all(model.ignore_index == 255 for model in models)
    assert all(model.baseline_crop == (512, 1024) for model in models)
    assert all(model.logits.direct_pre_softmax for model in models)
    assert common.dataset_roles.routine_model_selection_roles == ("train_select",)
    assert common.dataset_roles.forbidden_routine_roles == (
        "train_calibration",
        "official_val_common_eval",
    )


def test_initialization_policies_do_not_fabricate_pretrained_identities() -> None:
    _framework, _common, models = _suite()
    by_family = {model.model_family: model for model in models}

    assert by_family[ModelFamily.FAST_SCNN].initialization.project_training == "random"
    assert by_family[ModelFamily.FAST_SCNN].initialization.source.status == "not_applicable"
    assert all(model.initialization.stack_probe == "random" for model in models)
    for family in (ModelFamily.PIDNET_S, ModelFamily.DDRNET_23_SLIM, ModelFamily.SEGFORMER_B0):
        source = by_family[family].initialization.source
        assert source.status == "unresolved_human_input"
        assert source.sha256 is None


def test_training_config_and_hashes_are_deterministic() -> None:
    first = validate_configs(CONFIG_ROOT)
    second = validate_configs(CONFIG_ROOT)

    assert first == second
    assert sha256_payload(first) == sha256_payload(second)
    assert first["scientific_evidence"] is False
    assert len(first["models"]) == 5


@pytest.mark.parametrize("status", list(ProjectStatus))
def test_project_status_vocabulary_is_accepted(status: ProjectStatus) -> None:
    contract = _contract()
    record = ExperimentRegistryRecord(
        experiment_id=contract.experiment_id,
        status=status,
        config_sha256=contract.config_sha256,
        git_commit=COMMIT,
        git_dirty=False,
        framework_identity_sha256=contract.framework_identity_sha256,
        dataset_manifest_sha256=None,
        split_manifest_sha256=None,
        initialization_checkpoint_sha256=None,
        seed=contract.training_seed,
        runtime={"synthetic_stack_probe": True},
        final_metrics={},
        last_metrics={},
        artifact_paths=("experiments/segmentation/stack-probe/",),
    )

    assert record.status is status


def test_experiment_id_and_root_free_artifacts_fail_closed() -> None:
    contract = _contract()
    payload = {
        "experiment_id": "invalid",
        "status": "implemented",
        "config_sha256": contract.config_sha256,
        "git_commit": COMMIT,
        "git_dirty": False,
        "framework_identity_sha256": contract.framework_identity_sha256,
        "seed": contract.training_seed,
        "runtime": {},
        "final_metrics": {},
        "last_metrics": {},
        "artifact_paths": ["/private/result"],
    }

    with pytest.raises(ValidationError):
        ExperimentRegistryRecord.model_validate(payload)


def test_registry_append_round_trip_and_collision_refusal(tmp_path: Path) -> None:
    contract = _contract()
    path = tmp_path / "registry.jsonl"
    record = ExperimentRegistryRecord(
        experiment_id=contract.experiment_id,
        status=ProjectStatus.LOCALLY_TESTED,
        config_sha256=contract.config_sha256,
        git_commit=COMMIT,
        git_dirty=False,
        framework_identity_sha256=contract.framework_identity_sha256,
        dataset_manifest_sha256=None,
        split_manifest_sha256=None,
        initialization_checkpoint_sha256=None,
        seed=contract.training_seed,
        runtime={"synthetic_stack_probe": True},
        final_metrics={},
        last_metrics={},
        artifact_paths=("experiments/segmentation/stack-probe/",),
    )

    append_registry(path, record)

    assert load_registry(path) == (record,)
    assert json.loads(path.read_text(encoding="utf-8"))["experiment_id"] == record.experiment_id
    with pytest.raises(ValueError, match="collision"):
        append_registry(path, record)


def test_exact_resume_rejects_identity_mismatch() -> None:
    contract = _contract()
    metadata = CheckpointMetadata(
        experiment_id=contract.experiment_id,
        config_sha256=contract.config_sha256,
        experiment_fingerprint=contract.experiment_fingerprint,
        dataset_manifest_sha256=None,
        split_manifest_sha256=None,
        initialization_checkpoint_sha256=None,
        model_family=contract.model_family,
        framework_identity_sha256=contract.framework_identity_sha256,
        git_commit=contract.git_commit,
        precision_mode=contract.precision_mode,
        seed=contract.training_seed,
        epoch=0,
        optimizer_step=1,
        best_metric=None,
        last_metric=None,
        contains_optimizer_state=True,
        contains_scheduler_state=True,
        contains_amp_scaler_state=False,
    )

    validate_resume_identity(contract, metadata)
    with pytest.raises(ValueError, match="config_sha256"):
        validate_resume_identity(contract, metadata.model_copy(update={"config_sha256": SHA}))


def test_synthetic_identity_cannot_carry_real_manifest_hashes() -> None:
    with pytest.raises(ValidationError):
        DatasetIdentity(
            kind="synthetic_stack_fixture",
            dataset_manifest_sha256=SHA,
            synthetic_fixture_identity="synthetic-semantic-stack-v1",
        )


def test_native_logits_requires_predicate_approved_19_class_tensor() -> None:
    logits = np.zeros((1, 19, 8, 16), dtype=np.float32)
    assert validate_native_logits_tensor(
        logits, is_tensor=lambda value: isinstance(value, np.ndarray)
    ) == (1, 19, 8, 16)

    class TensorLike:
        shape = (1, 19, 8, 16)
        dtype = np.dtype(np.float32)

    with pytest.raises(ValueError, match="real framework tensor"):
        validate_native_logits_tensor(TensorLike(), is_tensor=lambda _value: False)


def test_hermetic_install_is_hash_locked_and_has_no_fallback(tmp_path: Path) -> None:
    framework = load_semantic_framework_config(CONFIG_ROOT / "framework_mmseg.yaml")
    commands = build_hermetic_commands(
        framework,
        tmp_path / "mmsegmentation",
        uv_executable=tmp_path / "resolved-tools/uv",
        project_root=REPO_ROOT,
        runtime_root=tmp_path / "runtime",
    )
    serialized = "\n".join(" ".join(command) for command in commands)

    assert "openmim" not in serialized.lower()
    assert f"{tmp_path}/resolved-tools/uv python install 3.11.13" in serialized
    assert "--system-site-packages" not in serialized
    assert "--upgrade-strategy" not in serialized
    assert "pip sync" in serialized
    assert "--strict --require-hashes" in serialized
    assert "colab-py311-cu121.lock" in serialized
    assert "colab-openmmlab.lock" in serialized
    assert framework.commit in serialized
    assert "download.openmmlab.com" not in serialized


def _write_valid_openmmlab_lock(path: Path) -> None:
    path.write_text(
        "mmengine==0.10.7 \\\n"
        "    --hash=sha256:262ac976a925562f78cd5fd14dd1bc9b680ed0aa81f0d85b723ef782f99c54ee\n"
        "mmcv-lite==2.1.0 \\\n"
        "    --hash=sha256:1d9913c35f793de4a3a022b93cecb712e1e7262eb4704eb8cd15e623dd375000\n",
        encoding="utf-8",
    )


def test_colab_locks_reject_gui_opencv(tmp_path: Path) -> None:
    main = tmp_path / "main.lock"
    openmmlab = tmp_path / "openmmlab.lock"
    _write_valid_openmmlab_lock(openmmlab)
    main.write_text(
        "opencv-python-headless==4.10.0.84\nrich==15.0.0\nsetuptools==80.9.0\n",
        encoding="utf-8",
    )
    _validate_lock_contract(main, openmmlab)

    main.write_text(
        "opencv-python==4.11.0.86\nopencv-python-headless==4.10.0.84\n"
        "rich==15.0.0\nsetuptools==80.9.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="GUI opencv-python"):
        _validate_lock_contract(main, openmmlab)


def test_colab_locks_require_openmmlab_no_deps_partition(tmp_path: Path) -> None:
    main = tmp_path / "main.lock"
    openmmlab = tmp_path / "openmmlab.lock"
    _write_valid_openmmlab_lock(openmmlab)
    main.write_text(
        "opencv-python-headless==4.10.0.84\nrich==15.0.0\nsetuptools==80.9.0\nmmengine==0.10.7\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only in the OpenMMLab lock"):
        _validate_lock_contract(main, openmmlab)


def test_uv_resolution_uses_the_hosted_path_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "hosted-bin" / "uv"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nprintf 'uv 0.8.8\\n'\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr("shutil.which", lambda name: str(executable) if name == "uv" else None)

    class NoBootstrapRunner:
        def run(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("an existing hosted uv executable must not be bootstrapped again")

    resolved, version, receipts = _resolve_uv_executable(  # type: ignore[arg-type]
        NoBootstrapRunner(),
        which=lambda name: str(executable) if name == "uv" else None,
    )

    assert resolved == executable.resolve()
    assert version == "0.8.8"
    assert receipts == []


def test_uv_absent_is_resolved_from_interpreter_scripts_after_install(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()

    class InstallRunner:
        def run(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            executable = scripts / "uv"
            executable.write_text("placeholder", encoding="utf-8")
            executable.chmod(0o755)
            return {"stage": "bootstrap-uv", "return_code": 0}

    resolved, version, receipts = _resolve_uv_executable(  # type: ignore[arg-type]
        InstallRunner(),
        which=lambda _name: None,
        scripts_directory=scripts,
        version_probe=lambda _path: "uv 0.8.8",
    )

    assert resolved == (scripts / "uv").resolve()
    assert version == "0.8.8"
    assert receipts == [{"stage": "bootstrap-uv", "return_code": 0}]


def test_uv_install_failure_is_classified(tmp_path: Path) -> None:
    class FailingRunner:
        def run(self, *_args: object, **_kwargs: object) -> None:
            raise subprocess.CalledProcessError(1, ["pip", "install", "uv"])

    with pytest.raises(BootstrapError, match="installation command failed") as captured:
        _resolve_uv_executable(  # type: ignore[arg-type]
            FailingRunner(), which=lambda _name: None, scripts_directory=tmp_path
        )
    assert captured.value.stage == "uv_install"
    assert captured.value.classification == "uv_install_failed"


def test_uv_install_without_resolvable_executable_is_rejected(tmp_path: Path) -> None:
    class SuccessfulRunner:
        def run(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"return_code": 0}

    with pytest.raises(BootstrapError, match="no uv executable") as captured:
        _resolve_uv_executable(  # type: ignore[arg-type]
            SuccessfulRunner(), which=lambda _name: None, scripts_directory=tmp_path
        )
    assert captured.value.classification == "uv_executable_not_found"


def test_uv_unexpected_host_version_bootstraps_private_exact_version(tmp_path: Path) -> None:
    executable = tmp_path / "host" / "uv"
    executable.parent.mkdir()
    executable.write_text("placeholder", encoding="utf-8")
    executable.chmod(0o755)
    bootstrap_root = tmp_path / "private"
    private_uv = bootstrap_root / "bin/uv"

    class BootstrapRunner:
        def run(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            private_uv.parent.mkdir(parents=True, exist_ok=True)
            private_uv.write_text("placeholder", encoding="utf-8")
            private_uv.chmod(0o755)
            return {"stage": "bootstrap-uv", "return_code": 0}

    resolved, version, receipts = _resolve_uv_executable(  # type: ignore[arg-type]
        BootstrapRunner(),
        which=lambda _name: str(executable),
        bootstrap_root=bootstrap_root,
        version_probe=lambda path: "uv 0.11.19" if path == executable else "uv 0.8.8",
    )
    assert resolved == private_uv.resolve()
    assert version == "0.8.8"
    assert receipts == [{"stage": "bootstrap-uv", "return_code": 0}]

    executable.chmod(0o644)
    private_uv.unlink()
    resolved, version, _ = _resolve_uv_executable(  # type: ignore[arg-type]
        BootstrapRunner(),
        which=lambda _name: str(executable),
        bootstrap_root=bootstrap_root,
        version_probe=lambda _path: "uv 0.8.8",
    )
    assert resolved == private_uv.resolve()
    assert version == "0.8.8"


def test_uv_private_bootstrap_accepts_colab_local_bin_prefix(tmp_path: Path) -> None:
    bootstrap_root = tmp_path / "private"
    private_uv = bootstrap_root / "local/bin/uv"

    class BootstrapRunner:
        def run(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            private_uv.parent.mkdir(parents=True)
            private_uv.write_text("placeholder", encoding="utf-8")
            private_uv.chmod(0o755)
            return {"stage": "bootstrap-uv", "return_code": 0}

    resolved, version, receipts = _resolve_uv_executable(  # type: ignore[arg-type]
        BootstrapRunner(),
        which=lambda _name: None,
        bootstrap_root=bootstrap_root,
        version_probe=lambda _path: "uv 0.8.8",
    )
    assert resolved == private_uv.resolve()
    assert version == "0.8.8"
    assert receipts == [{"stage": "bootstrap-uv", "return_code": 0}]


def test_uv_newer_version_with_platform_suffix_is_privately_replaced(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "host-uv"
    executable.write_text("placeholder", encoding="utf-8")
    executable.chmod(0o755)
    scripts = tmp_path / "private-bin"
    scripts.mkdir()

    class BootstrapRunner:
        def run(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            private = scripts / "uv"
            private.write_text("placeholder", encoding="utf-8")
            private.chmod(0o755)
            return {"return_code": 0}

    resolved, version, _ = _resolve_uv_executable(  # type: ignore[arg-type]
        BootstrapRunner(),
        which=lambda _name: str(executable),
        scripts_directory=scripts,
        version_probe=lambda path: (
            "uv 0.11.19 (x86_64-unknown-linux-gnu)"
            if path == executable
            else "uv 0.8.8 (x86_64-unknown-linux-gnu)"
        ),
    )
    assert resolved == (scripts / "uv").resolve()
    assert version == "0.8.8"


def test_bootstrap_failure_writes_terminal_receipt_and_logs(tmp_path: Path) -> None:
    paths = RuntimePathContract.from_workspace(tmp_path / "workspace")
    status = LongRunStatus(paths.evidence_root / "run_status.json")
    error = BootstrapError("uv_resolution", "uv_executable_not_found", "missing uv")

    _record_failure(
        error,
        paths=paths,
        status=status,
        failed_stage="uv_bootstrap",
    )

    persisted = json.loads((paths.evidence_root / "run_status.json").read_text())
    failure = json.loads((paths.evidence_root / "failure.json").read_text())
    assert persisted["status"] == "failed"
    assert persisted["phase"] == "uv_resolution"
    assert failure["failure_classification"] == "uv_executable_not_found"
    assert failure["failed_stage"] == "uv_resolution"


def test_bounded_runtime_repair_removes_only_recognizable_owned_targets(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "edgeguard-runtime"
    runtime.mkdir()
    (runtime / "pyvenv.cfg").write_text("incomplete", encoding="utf-8")
    checkout = tmp_path / "mmsegmentation"
    checkout.mkdir()
    probe = tmp_path / "hermetic-core-model-probe"
    probe.mkdir()

    actions = repair_owned_path(
        runtime_root=runtime,
        checkout=checkout,
        probe=probe,
        expected_commit="a" * 40,
    )

    assert {action["action"] for action in actions} == {
        "removed_incomplete_environment",
        "removed_incomplete_checkout",
        "removed_incomplete_probe",
    }
    assert not runtime.exists() and not checkout.exists() and not probe.exists()

    unrelated = tmp_path / "edgeguard-runtime"
    unrelated.mkdir()
    (unrelated / "user-data.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="unrecognized"):
        repair_owned_path(
            runtime_root=unrelated,
            checkout=tmp_path / "mmsegmentation",
            probe=tmp_path / "hermetic-core-model-probe",
            expected_commit="a" * 40,
        )
    assert (unrelated / "user-data.txt").is_file()


def test_policy_selected_handoff_preserves_roles_and_excludes_calibration_from_selection(
    tmp_path: Path,
) -> None:
    dataset = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_train_dataset_manifest",
        "samples": [
            {
                "sample_id": "alpha_000000_000001",
                "group_id": "alpha_000000",
                "image_relative_path": (
                    "leftImg8bit/train/alpha/alpha_000000_000001_leftImg8bit.png"
                ),
                "train_id_relative_path": (
                    "gtFine/train/trainIds/alpha/alpha_000000_000001_gtFine_trainIds.png"
                ),
            },
            {
                "sample_id": "beta_000000_000001",
                "group_id": "beta_000000",
                "image_relative_path": (
                    "leftImg8bit/train/beta/beta_000000_000001_leftImg8bit.png"
                ),
                "train_id_relative_path": (
                    "gtFine/train/trainIds/beta/beta_000000_000001_gtFine_trainIds.png"
                ),
            },
            {
                "sample_id": "gamma_000000_000001",
                "group_id": "gamma_000000",
                "image_relative_path": (
                    "leftImg8bit/train/gamma/gamma_000000_000001_leftImg8bit.png"
                ),
                "train_id_relative_path": (
                    "gtFine/train/trainIds/gamma/gamma_000000_000001_gtFine_trainIds.png"
                ),
            },
        ],
        "image_count": 3,
        "ontology_sha256": SHA,
    }
    dataset["manifest_sha256"] = sha256_payload(dataset)
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    candidate = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_diversity_split_candidate",
        "candidate_id": "CSF-SPLIT-D",
        "status": "policy_candidate",
        "hard_constraints_passed": True,
        "sample_manifest": [
            {
                "sample_id": "alpha_000000_000001",
                "group_id": "alpha_000000",
                "role": "train_fit",
            },
            {
                "sample_id": "beta_000000_000001",
                "group_id": "beta_000000",
                "role": "train_select",
            },
            {
                "sample_id": "gamma_000000_000001",
                "group_id": "gamma_000000",
                "role": "train_calibration",
            },
        ],
    }
    candidate["candidate_sha256"] = sha256_payload(candidate)
    split = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_policy_selected_split",
        "status": "policy_selected",
        "policy_version": POLICY_VERSION,
        "policy_config": POLICY_CONFIG,
        "policy_config_sha256": sha256_payload(POLICY_CONFIG),
        "candidate_id": "CSF-SPLIT-D",
        "candidate_sha256": candidate["candidate_sha256"],
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "ontology_sha256": SHA,
        "candidate": candidate,
    }
    split["manifest_sha256"] = sha256_payload(split)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")

    identity, samples = load_policy_selected_cityscapes_split(dataset_path, split_path)

    assert identity.kind == "real_selected_split"
    assert [sample.sample_id for sample in samples_for_training_role(samples, "train_fit")] == [
        "alpha_000000_000001"
    ]
    assert [sample.sample_id for sample in samples_for_training_role(samples, "train_select")] == [
        "beta_000000_000001"
    ]
    with pytest.raises(ValueError, match="no samples"):
        samples_for_training_role(samples, "official_val_common_eval")  # type: ignore[arg-type]


def test_policy_selected_contract_rejects_unselected_candidate() -> None:
    with pytest.raises(ValidationError):
        from edgeguard.training.contracts import PolicySelectedSplit

        PolicySelectedSplit.model_validate(
            {
                "schema_version": "1.0",
                "record_type": "cityscapes_fine_policy_selected_split",
                "status": "candidate_not_human_approved",
            }
        )


def test_validation_interval_requires_all_scientific_training_signals() -> None:
    metrics = {
        "train_select_loss": 0.7,
        "mIoU": 62.5,
        **{f"edgeguard_iou_{class_id:02d}": float(class_id) for class_id in range(19)},
    }

    record = _validation_interval_record(
        epoch=2,
        optimizer_step=100,
        train_losses=[0.9, 0.8],
        metrics=metrics,
        learning_rate=0.001,
    )

    assert isinstance(record, ValidationIntervalRecord)
    assert record.train_loss == pytest.approx(0.85)
    assert record.train_select_loss == 0.7
    assert record.train_select_miou == 62.5
    assert record.per_class_iou == tuple(float(class_id) for class_id in range(19))
    assert record.learning_rate == 0.001
    assert record.generalization_gap_inputs.train_loss == pytest.approx(0.85)


def test_recovery_sync_is_atomic_verified_and_path_free(tmp_path: Path) -> None:
    work = tmp_path / "work"
    sync = tmp_path / "sync"
    work.mkdir()
    checkpoint = work / "epoch_1.pth"
    checkpoint.write_bytes(b"checkpoint-state")
    (work / "checkpoint_metadata.json").write_text("{}\n", encoding="utf-8")
    (work / "last_checkpoint").write_text(str(checkpoint), encoding="utf-8")

    receipt = _atomic_sync_checkpoint(work, sync)

    assert {item["filename"] for item in receipt["files"]} == {
        "checkpoint_metadata.json",
        "epoch_1.pth",
    }
    assert not list(sync.glob("*.incoming"))
    assert str(tmp_path) not in json.dumps(receipt)


def test_stack_probe_refuses_nonempty_output_before_framework_import(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.txt").write_text("collision", encoding="utf-8")

    with pytest.raises(ValueError, match="absent or empty"):
        run_stack_probe(
            CONFIG_ROOT,
            tmp_path / "missing-mmseg",
            output,
            REPO_ROOT,
            COMMIT,
        )
