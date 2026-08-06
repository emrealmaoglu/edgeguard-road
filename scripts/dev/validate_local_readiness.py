"""Run the local-first EdgeGuard readiness gate without real datasets or Colab."""

from __future__ import annotations

import argparse
import importlib.util
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image

from edgeguard.data.cityscapes_train import load_train_ids
from edgeguard.evaluation.semantic import SemanticConfusionMatrix
from edgeguard.runtime import RuntimePathContract
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload
from edgeguard.telemetry.longrun import LongRunStatus, atomic_write_json
from edgeguard.training.data import load_policy_selected_cityscapes_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rebuild_cityscapes_splits import rebuild_cityscapes_splits  # noqa: E402
from scripts.stage_cityscapes_training import stage_cityscapes_training  # noqa: E402
from scripts.train.install_semantic_stack import (  # noqa: E402
    BootstrapError,
    _resolve_uv_executable,
    build_hermetic_commands,
)
from scripts.train.train_semantic import run_stack_probe, validate_configs  # noqa: E402

CITY_NAMES = (
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
)
SAMPLE_COUNT = 180


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _save_png(path: Path, array: np.ndarray, *, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode=mode).save(path, format="PNG", compress_level=9)


def create_mini_cityscapes_fixture(root: Path) -> tuple[Path, Path]:
    """Create a tiny deterministic, project-owned 19-class training fixture."""
    if root.exists():
        raise ValueError("mini-Cityscapes fixture root must not already exist")
    root.mkdir(parents=True)
    samples: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    height, width = 32, 64
    base_mask = (np.arange(height * width, dtype=np.uint16).reshape(height, width) % 19).astype(
        np.uint8
    )
    base_mask[0, 0] = 255
    valid = base_mask != 255
    pixel_counts = [int(np.count_nonzero(base_mask == class_id)) for class_id in range(19)]
    presence_counts = [1] * 19
    for index in range(SAMPLE_COUNT):
        city = CITY_NAMES[index % len(CITY_NAMES)]
        sequence = f"{index:06d}"
        frame = "000000"
        sample_id = f"{city}_{sequence}_{frame}"
        group_id = f"{city}_{sequence}"
        image_relative = f"leftImg8bit/train/{city}/{sample_id}_leftImg8bit.png"
        mask_relative = f"gtFine/train/trainIds/{city}/{sample_id}_gtFine_trainIds.png"
        x = np.arange(width, dtype=np.uint16)[None, :]
        y = np.arange(height, dtype=np.uint16)[:, None]
        image = np.stack(
            (
                np.broadcast_to((x + index) % 256, (height, width)),
                np.broadcast_to((y * 3 + index * 5) % 256, (height, width)),
                ((x + y) * 7 + index * 11) % 256,
            ),
            axis=2,
        ).astype(np.uint8)
        mask = np.roll(base_mask, shift=index % 19, axis=1)
        image_path = root / image_relative
        mask_path = root / mask_relative
        _save_png(image_path, image, mode="RGB")
        _save_png(mask_path, mask, mode="L")
        samples.append(
            {
                "sample_id": sample_id,
                "city": city,
                "sequence": sequence,
                "frame": frame,
                "group_id": group_id,
                "image_relative_path": image_relative,
                "train_id_relative_path": mask_relative,
                "image_sha256": sha256_file(image_path),
                "train_id_sha256": sha256_file(mask_path),
                "shape": [height, width],
            }
        )
        groups.append(
            {
                "group_id": group_id,
                "city": city,
                "sequence": sequence,
                "sample_count": 1,
                "valid_pixel_count": int(np.count_nonzero(valid)),
                "ignored_pixel_count": 1,
                "class_pixel_counts": pixel_counts,
                "class_presence_counts": presence_counts,
            }
        )
    ontology_sha256 = sha256_payload(
        {"semantic_ids": list(range(19)), "ignore_index": 255, "fixture": True}
    )
    dataset: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_train_dataset_manifest",
        "dataset_name": "edgeguard_project_owned_mini_cityscapes",
        "dataset_role": "synthetic_fixture_only",
        "scientific_evidence": False,
        "samples": samples,
        "image_count": len(samples),
        "ontology_sha256": ontology_sha256,
    }
    dataset["manifest_sha256"] = sha256_payload(dataset)
    group_summary: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "cityscapes_fine_train_group_summary",
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "group_count": len(groups),
        "groups": groups,
    }
    group_summary["manifest_sha256"] = sha256_payload(group_summary)
    manifest_root = root.parent / "manifests"
    manifest_root.mkdir()
    dataset_path = manifest_root / "dataset_manifest.json"
    group_path = manifest_root / "group_summary.json"
    dataset_path.write_text(canonical_json(dataset) + "\n", encoding="utf-8")
    group_path.write_text(canonical_json(group_summary) + "\n", encoding="utf-8")
    return dataset_path, group_path


class _SimulationRunner:
    def __init__(self, scripts_directory: Path, *, fail: bool = False) -> None:
        self.scripts_directory = scripts_directory
        self.fail = fail

    def run(self, *_args: object, **_kwargs: object) -> dict[str, Any]:
        if self.fail:
            raise subprocess.CalledProcessError(1, ["pip", "install", "uv"])
        executable = self.scripts_directory / "uv"
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text("simulated", encoding="utf-8")
        executable.chmod(0o755)
        return {"stage": "bootstrap-uv", "return_code": 0}


def run_uv_bootstrap_simulations(root: Path) -> dict[str, Any]:
    """Exercise every bootstrap branch without modifying the hosted interpreter."""
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    existing = scripts / "uv-existing"
    existing.write_text("simulated", encoding="utf-8")
    existing.chmod(0o755)
    passed: list[str] = []
    _resolve_uv_executable(
        _SimulationRunner(scripts),  # type: ignore[arg-type]
        which=lambda _name: str(existing),
        scripts_directory=scripts,
        version_probe=lambda _path: "uv 0.8.8",
    )
    passed.append("already_available")
    installed = root / "installed-scripts"
    _resolve_uv_executable(
        _SimulationRunner(installed),  # type: ignore[arg-type]
        which=lambda _name: None,
        scripts_directory=installed,
        version_probe=lambda _path: "uv 0.8.8",
    )
    passed.append("absent_then_installed")
    replacement = root / "replacement-scripts"
    _resolve_uv_executable(
        _SimulationRunner(replacement),  # type: ignore[arg-type]
        which=lambda _name: str(existing),
        scripts_directory=replacement,
        version_probe=lambda path: "uv 9.9.9" if path == existing else "uv 0.8.8",
    )
    passed.append("unexpected_host_version_replaced")
    cases = (
        (
            "install_command_failure",
            lambda: _resolve_uv_executable(
                _SimulationRunner(root / "failed", fail=True),  # type: ignore[arg-type]
                which=lambda _name: None,
                scripts_directory=root / "failed",
            ),
            "uv_install_failed",
        ),
        (
            "resolution_failure",
            lambda: _resolve_uv_executable(
                _SimulationRunner(root / "missing"),  # type: ignore[arg-type]
                which=lambda _name: None,
                scripts_directory=root / "empty-scripts",
            ),
            "uv_executable_not_found",
        ),
        (
            "private_version_mismatch",
            lambda: _resolve_uv_executable(
                _SimulationRunner(root / "wrong-private"),  # type: ignore[arg-type]
                which=lambda _name: None,
                scripts_directory=root / "wrong-private",
                version_probe=lambda _path: "uv 9.9.9",
            ),
            "uv_version_mismatch",
        ),
    )
    for name, operation, expected in cases:
        try:
            operation()
        except BootstrapError as error:
            if error.classification != expected:
                raise
            passed.append(name)
        else:
            raise AssertionError(f"bootstrap simulation did not fail: {name}")
    invalid = root / "non-executable-uv"
    invalid.write_text("simulated", encoding="utf-8")
    non_executable_replacement = root / "non-executable-replacement"
    _resolve_uv_executable(
        _SimulationRunner(non_executable_replacement),  # type: ignore[arg-type]
        which=lambda _name: str(invalid),
        scripts_directory=non_executable_replacement,
        version_probe=lambda _path: "uv 0.8.8",
    )
    passed.append("non_executable_host_replaced")
    return {"status": "passed", "cases": passed}


def _fixture_loader_loss_metrics_probe(
    dataset_root: Path,
    dataset_manifest: Path,
    split_manifest: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Iterate, augment, compute loss/metrics, and save a local checkpoint."""
    import torch

    identity, samples = load_policy_selected_cityscapes_split(dataset_manifest, split_manifest)
    selected = tuple(sorted(samples, key=lambda sample: sample.sample_id)[:8])
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for index, sample in enumerate(selected):
        with Image.open(dataset_root / sample.image_relative_path) as encoded:
            image = np.asarray(encoded.convert("RGB"), dtype=np.uint8).copy()
        mask = load_train_ids(dataset_root / sample.train_id_relative_path)
        if image.shape[:2] != mask.shape:
            raise ValueError("mini fixture image/mask geometry mismatch")
        if index % 2:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        images.append(np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)))
        masks.append(mask.astype(np.int64))
    inputs = torch.from_numpy(np.stack(images))
    targets = torch.from_numpy(np.stack(masks))
    torch.manual_seed(20260727)
    model = torch.nn.Conv2d(3, 19, kernel_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=255)
    if not bool(torch.isfinite(loss)):
        raise ValueError("mini fixture loss is non-finite")
    loss.backward()
    optimizer.step()
    scheduler.step()
    prediction = logits.detach().argmax(dim=1).numpy().astype(np.int64)
    accumulator = SemanticConfusionMatrix()
    accumulator.update(prediction, targets.numpy())
    metrics = accumulator.result()
    checkpoint = output_root / "mini-checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "identity": {
                "dataset_manifest_sha256": identity.dataset_manifest_sha256,
                "split_manifest_sha256": identity.split_manifest_sha256,
            },
        },
        checkpoint,
    )
    return {
        "status": "passed",
        "sample_count": len(selected),
        "batch_shape": list(inputs.shape),
        "logits_shape": list(logits.shape),
        "loss_finite": True,
        "metrics_record_type": metrics["record_type"],
        "per_class_iou_count": len(metrics["per_class_iou"]),
        "checkpoint_saved": True,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "scientific_evidence": False,
    }


def _fixture_checkpoint_resume_probe(
    dataset_manifest: Path,
    split_manifest: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Restore the exact model, optimizer, scheduler, and dataset identity."""
    import torch

    identity, _samples = load_policy_selected_cityscapes_split(dataset_manifest, split_manifest)
    checkpoint = output_root / "mini-checkpoint.pt"
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    resumed_model = torch.nn.Conv2d(3, 19, kernel_size=1)
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.01, momentum=0.9)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=1, gamma=0.9)
    resumed_model.load_state_dict(restored["model"], strict=True)
    resumed_optimizer.load_state_dict(restored["optimizer"])
    resumed_scheduler.load_state_dict(restored["scheduler"])
    if restored["identity"]["split_manifest_sha256"] != identity.split_manifest_sha256:
        raise ValueError("mini fixture checkpoint split identity mismatch")
    return {
        "status": "passed",
        "checkpoint_round_trip": True,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "scientific_evidence": False,
    }


def _package_evidence(evidence_root: Path) -> Path:
    package = evidence_root / "local-readiness-evidence.zip"
    with ZipFile(package, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sorted(evidence_root.rglob("*.json")):
            if source == package:
                continue
            info = ZipInfo(source.relative_to(evidence_root).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())
    return package


def _path_free(value: Any, *, workspace_root: Path, project_root: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _path_free(item, workspace_root=workspace_root, project_root=project_root)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _path_free(item, workspace_root=workspace_root, project_root=project_root)
            for item in value
        ]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        return value.replace(str(workspace_root), "<workspace>").replace(
            str(project_root), "<repository-root>"
        )
    return value


def validate_local_readiness(
    workspace_root: Path,
    *,
    profile: str,
    mmseg_checkout: Path | None = None,
) -> dict[str, Any]:
    """Execute the local readiness phases and produce a small evidence package."""
    if workspace_root.exists() and any(workspace_root.iterdir()):
        raise ValueError("local readiness workspace must be absent or empty")
    paths = RuntimePathContract.from_workspace(workspace_root).validated(forbid_content=True)
    paths.evidence_root.mkdir(parents=True, exist_ok=True)
    status = LongRunStatus(paths.evidence_root / "run_status.json", heartbeat_seconds=1.0)
    project_root = PROJECT_ROOT
    config_root = project_root / "configs/training/segmentation"
    project_commit = _git(project_root, "rev-parse", "HEAD")
    phases: dict[str, Any] = {}
    started = time.monotonic()

    def phase(name: str, index: int, operation: Any) -> Any:
        phase_started = time.monotonic()
        status.update(
            status="running",
            phase=name,
            phase_index=index,
            phase_total=11,
            completed=index - 1,
            total=11,
            speed_per_second=(index - 1) / max(time.monotonic() - started, 1e-9),
            force=True,
        )
        result = operation()
        phases[name] = {
            "result": result,
            "elapsed_seconds": time.monotonic() - phase_started,
        }
        return result

    try:
        config_receipt = phase(
            "repository-and-config-validation",
            1,
            lambda: {
                "repository_root_name": project_root.name,
                "git_commit": project_commit,
                "config": validate_configs(config_root),
            },
        )
        phase("runtime-path-contract", 2, lambda: paths.receipt())

        def installer_plans() -> dict[str, Any]:
            framework = __import__(
                "edgeguard.training.config", fromlist=["load_semantic_framework_config"]
            ).load_semantic_framework_config(config_root / "framework_mmseg.yaml")
            uv = Path("uv-resolved-at-runtime")
            return {
                "hermetic_runtime": [
                    list(command)
                    for command in build_hermetic_commands(
                        framework,
                        paths.checkout_root / "mmsegmentation",
                        uv_executable=uv,
                        project_root=project_root,
                        runtime_root=paths.runtime_root,
                    )
                ],
            }

        phase("installer-plan-generation", 3, installer_plans)
        phase(
            "uv-bootstrap-simulations",
            4,
            lambda: run_uv_bootstrap_simulations(paths.cache_root / "uv-simulations"),
        )
        fixture_root = paths.data_root / "mini-cityscapes"
        dataset_manifest, group_summary = phase(
            "mini-cityscapes-fixture",
            5,
            lambda: create_mini_cityscapes_fixture(fixture_root),
        )
        split_root = paths.data_root / "split-policy"
        split_result = phase(
            "split-policy-rebuild",
            6,
            lambda: rebuild_cityscapes_splits(dataset_manifest, group_summary, split_root),
        )
        split_manifest = split_root / "policy_selected_split.json"
        staged_root = paths.data_root / "staged-mini-cityscapes"
        phase(
            "dataset-staging",
            7,
            lambda: stage_cityscapes_training(
                dataset_root=fixture_root,
                dataset_manifest_path=dataset_manifest,
                split_policy_path=split_manifest,
                drive_bundle_directory=paths.cache_root / "fixture-bundles",
                cache_directory=paths.cache_root / "staging",
                staged_dataset_root=staged_root,
                allow_bundle_creation=True,
            ),
        )
        fixture_loader_probe = phase(
            "dataloader-loss-and-metrics",
            8,
            lambda: _fixture_loader_loss_metrics_probe(
                staged_root,
                dataset_manifest,
                split_manifest,
                paths.cache_root / "fixture-probe",
            ),
        )
        fixture_resume_probe = phase(
            "checkpoint-save-and-resume",
            9,
            lambda: _fixture_checkpoint_resume_probe(
                dataset_manifest,
                split_manifest,
                paths.cache_root / "fixture-probe",
            ),
        )

        def framework_probe() -> dict[str, Any]:
            available = all(
                importlib.util.find_spec(name) is not None
                for name in ("torch", "mmengine", "mmcv", "mmseg")
            )
            if not available or mmseg_checkout is None:
                return {
                    "status": "platform_blocked",
                    "reason": "local MMEngine/MMCV-lite/MMSeg stack or checkout unavailable",
                    "requires_linux_cpu_gate": True,
                }
            output = paths.evidence_root / "five-model-cpu-probe"
            return run_stack_probe(
                config_root,
                mmseg_checkout,
                output,
                project_root,
                project_commit,
                device_name="cpu",
                allow_dirty_project=profile == "mac-cpu",
            )

        framework_result = phase("optional-five-model-cpu-probe", 10, framework_probe)
        bytes_used = sum(
            path.stat().st_size for path in workspace_root.rglob("*") if path.is_file()
        )
        receipt: dict[str, Any] = {
            "schema_version": "1.0",
            "record_type": "edgeguard_local_readiness_receipt",
            "status": (
                "passed"
                if framework_result.get("status") != "platform_blocked"
                else "passed_with_platform_block"
            ),
            "profile": profile,
            "project_commit": project_commit,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "runtime_contract": paths.receipt(),
            "config_identity": config_receipt["config"]["framework_identity_sha256"],
            "split_identity": split_result["selected_manifest_sha256"],
            "fixture_probe": {
                "loader_loss_metrics": fixture_loader_probe,
                "checkpoint_resume": fixture_resume_probe,
            },
            "framework_probe": framework_result,
            "workspace_bytes_before_evidence_package": bytes_used,
            "phases": phases,
            "scientific_evidence": False,
        }
        receipt["phases"] = _path_free(
            phases,
            workspace_root=workspace_root.resolve(),
            project_root=project_root,
        )
        atomic_write_json(paths.evidence_root / "readiness_receipt.json", receipt)
        package = phase("evidence-packaging", 11, lambda: _package_evidence(paths.evidence_root))
        receipt["evidence_package"] = package.name
        receipt["evidence_package_sha256"] = sha256_file(package)
        receipt["phases"] = _path_free(
            phases,
            workspace_root=workspace_root.resolve(),
            project_root=project_root,
        )
        atomic_write_json(paths.evidence_root / "readiness_receipt.json", receipt)
        status.update(
            completed=11,
            total=11,
            speed_per_second=11 / max(time.monotonic() - started, 1e-9),
            force=True,
        )
        status.complete(last_checkpoint=package.name)
        return receipt
    except BaseException as error:
        status.fail(error)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--profile", choices=("mac-cpu", "linux-cpu"), required=True)
    parser.add_argument("--mmseg-checkout", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = validate_local_readiness(
        args.workspace_root,
        profile=args.profile,
        mmseg_checkout=args.mmseg_checkout,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
