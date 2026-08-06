"""Runtime-registered MMSeg components for explicit manifests and domain balance."""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from edgeguard.rescue.colab_recovery import (
    latest_checkpoint,
    publish_recovery_file,
    write_campaign_status,
)
from edgeguard.rescue.multidomain import uniform_domain_indices, validate_dataset_manifest

_REGISTERED = False


def register_mmseg_components() -> None:
    """Register optional components only after the CUDA/MMSeg stack is available."""
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        torch_data = __import__("torch.utils.data", fromlist=["Sampler"])
        mmengine_dist = __import__("mmengine.dist", fromlist=["get_dist_info", "sync_random_seed"])
        mmengine_registry = __import__("mmengine.registry", fromlist=["DATA_SAMPLERS"])
        mmseg_datasets = __import__("mmseg.datasets", fromlist=["BaseSegDataset"])
        mmseg_registry = __import__("mmseg.registry", fromlist=["DATASETS"])
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "MMSeg runtime is required to register multi-domain components"
        ) from error

    base_seg_dataset: Any = mmseg_datasets.BaseSegDataset
    sampler_base: Any = torch_data.Sampler

    @mmseg_registry.DATASETS.register_module(force=True)
    class EdgeGuardManifestDataset(base_seg_dataset):
        """MMSeg dataset backed by explicit image/mask pairs in a frozen manifest."""

        METAINFO = {
            "classes": (
                "road",
                "sidewalk",
                "building",
                "wall",
                "fence",
                "pole",
                "traffic light",
                "traffic sign",
                "vegetation",
                "terrain",
                "sky",
                "person",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle",
            )
        }

        def __init__(self, *, manifest_path: str, role: str, **kwargs: Any) -> None:
            self.edgeguard_manifest_path = Path(manifest_path)
            self.edgeguard_role = role
            super().__init__(img_suffix="", seg_map_suffix="", **kwargs)

        def load_data_list(self) -> list[dict[str, Any]]:
            payload = validate_dataset_manifest(self.edgeguard_manifest_path)
            records = payload["roles"].get(self.edgeguard_role)
            if not isinstance(records, list):
                raise ValueError(f"manifest has no role {self.edgeguard_role!r}")
            dataset_root = Path(payload["dataset_root"])
            prepared_root = Path(payload["prepared_root"])
            data_list: list[dict[str, Any]] = []
            for record in records:
                canonical = record.get("canonical_mask")
                if canonical is None:
                    raise ValueError("scientific segmentation record has no canonical mask")
                if payload["dataset_id"] == "idd20k":
                    mask_path = prepared_root / str(canonical)
                else:
                    mask_path = dataset_root / str(canonical)
                data_list.append(
                    {
                        "img_path": str(dataset_root / str(record["image"])),
                        "seg_map_path": str(mask_path),
                        "label_map": None,
                        "reduce_zero_label": False,
                        "seg_fields": [],
                        "dataset_id": payload["dataset_id"],
                        "sample_id": record["sample_id"],
                    }
                )
            return data_list

    @mmengine_registry.DATA_SAMPLERS.register_module(force=True)
    class EdgeGuardDomainBalancedSampler(sampler_base):
        """Distributed sampler assigning equal expected probability to each domain."""

        def __init__(
            self,
            dataset: Any,
            *,
            shuffle: bool = True,
            seed: int | None = None,
            round_up: bool = True,
        ) -> None:
            if not shuffle:
                raise ValueError("domain-balanced training sampler requires shuffle=True")
            datasets = getattr(dataset, "datasets", None)
            if not isinstance(datasets, (list, tuple)) or len(datasets) < 2:
                raise ValueError("domain-balanced sampler requires a multi-dataset concat wrapper")
            self.lengths = [len(item) for item in datasets]
            self.rank, self.world_size = mmengine_dist.get_dist_info()
            self.seed = int(mmengine_dist.sync_random_seed(seed))
            self.epoch = 0
            global_size = max(sum(self.lengths), self.world_size)
            if round_up:
                global_size = (
                    (global_size + self.world_size - 1) // self.world_size
                ) * self.world_size
            self.global_size = global_size
            self.num_samples = global_size // self.world_size
            self.resume_offset = 0

        def __iter__(self) -> Any:
            indices = uniform_domain_indices(
                self.lengths,
                total_size=self.global_size,
                seed=self.seed,
                epoch=self.epoch,
            )
            local = indices[self.rank : self.global_size : self.world_size]
            offset = self.resume_offset
            self.resume_offset = 0
            return iter(local[offset:])

        def __len__(self) -> int:
            return self.num_samples

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def set_resume_position(self, dataloader_iteration: int, batch_size: int) -> None:
            """Reconstruct sampler epoch/offset from the optimizer checkpoint iteration."""
            if dataloader_iteration < 0 or batch_size <= 0:
                raise ValueError("sampler resume position is invalid")
            consumed = dataloader_iteration * batch_size
            self.epoch = consumed // self.num_samples
            self.resume_offset = consumed % self.num_samples

    mmengine_hooks = __import__("mmengine.hooks", fromlist=["Hook"])
    hooks_registry = __import__("mmengine.registry", fromlist=["HOOKS"])
    hook_base: Any = mmengine_hooks.Hook

    @hooks_registry.HOOKS.register_module(force=True)
    class EdgeGuardRecoveryHook(hook_base):
        """Persist a resumable checkpoint without trusting the Colab VM lifetime."""

        priority = "LOWEST"

        def __init__(
            self,
            *,
            store_root: str,
            artifact_id: str,
            campaign_id: str,
            project_commit: str,
            identity_sha256: str,
            accumulation: int,
            status_path: str | None = None,
            optimizer_interval: int = 500,
            maximum_seconds: int = 600,
            status_seconds: int = 300,
            intentional_interrupt_optimizer_step: int | None = None,
        ) -> None:
            self.store_root = Path(store_root)
            self.artifact_id = artifact_id
            self.campaign_id = campaign_id
            self.project_commit = project_commit
            self.identity_sha256 = identity_sha256
            self.accumulation = accumulation
            self.status_path = Path(status_path) if status_path else None
            self.iteration_interval = optimizer_interval * accumulation
            self.maximum_seconds = maximum_seconds
            self.status_seconds = status_seconds
            self.started = time.monotonic()
            self.last_publish = time.monotonic()
            self.last_status = 0.0
            self.intentional_interrupt_optimizer_step = intentional_interrupt_optimizer_step

        def before_train(self, runner: Any) -> None:
            sampler: Any = getattr(runner.train_dataloader, "sampler", None)
            if int(runner.iter) > 0 and hasattr(sampler, "set_resume_position"):
                sampler.set_resume_position(
                    int(runner.iter), int(runner.train_dataloader.batch_size)
                )

        def before_save_checkpoint(self, runner: Any, checkpoint: dict[str, Any]) -> None:
            torch = __import__("torch")
            checkpoint["edgeguard_recovery_state"] = {
                "python_random": random.getstate(),
                "numpy_random": np.random.get_state(),
                "torch_random": torch.get_rng_state(),
                "cuda_random": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
                "dataloader_iteration": int(runner.iter + 1),
                "accumulation": self.accumulation,
            }

        def after_load_checkpoint(self, runner: Any, checkpoint: dict[str, Any]) -> None:
            del runner
            state = checkpoint.get("edgeguard_recovery_state")
            if not isinstance(state, dict):
                return
            torch = __import__("torch")
            random.setstate(state["python_random"])
            np.random.set_state(state["numpy_random"])
            torch.set_rng_state(state["torch_random"])
            if torch.cuda.is_available() and state.get("cuda_random") is not None:
                torch.cuda.set_rng_state_all(state["cuda_random"])

        def after_train_iter(
            self,
            runner: Any,
            batch_idx: int,
            data_batch: Any = None,
            outputs: Any = None,
        ) -> None:
            del batch_idx, data_batch, outputs
            iteration = int(runner.iter + 1)
            now = time.monotonic()
            if self.status_path is not None and (
                now - self.last_status >= self.status_seconds or iteration >= int(runner.max_iters)
            ):
                torch = __import__("torch")
                elapsed = max(now - self.started, 1.0e-9)
                rate = iteration / elapsed
                remaining = max(0, int(runner.max_iters) - iteration)
                usage = shutil.disk_usage(runner.work_dir)
                write_campaign_status(
                    self.status_path,
                    state="running",
                    artifact_id=self.artifact_id,
                    dataloader_iteration=iteration,
                    optimizer_step=iteration // self.accumulation,
                    optimizer_step_target=int(runner.max_iters) // self.accumulation,
                    iterations_per_second=rate,
                    eta_seconds=remaining / rate if rate else None,
                    disk_free_bytes=usage.free,
                    gpu_memory_allocated_bytes=(
                        int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else None
                    ),
                    gpu_memory_peak_bytes=(
                        int(torch.cuda.max_memory_allocated())
                        if torch.cuda.is_available()
                        else None
                    ),
                    last_heartbeat_monotonic_seconds=now,
                )
                self.last_status = now
            due_interval = iteration % self.iteration_interval == 0
            due_time = (
                now - self.last_publish >= self.maximum_seconds
                and iteration % self.accumulation == 0
            )
            due_final = iteration >= int(runner.max_iters)
            interrupt_marker = (
                Path(runner.work_dir) / ".intentional-interruption-complete.json"
            )
            optimizer_step = iteration // self.accumulation
            due_interrupt = (
                self.intentional_interrupt_optimizer_step is not None
                and optimizer_step == self.intentional_interrupt_optimizer_step
                and not interrupt_marker.exists()
            )
            if not (due_interval or due_time or due_final or due_interrupt):
                return
            marker = Path(runner.work_dir) / "last_checkpoint"
            current: Path | None = None
            if marker.is_file():
                try:
                    current = latest_checkpoint(Path(runner.work_dir))
                except FileNotFoundError:
                    current = None
            if current is None or not current.name.endswith(f"_{iteration}.pth"):
                filename = f"recovery_{iteration}.pth"
                runner.save_checkpoint(
                    runner.work_dir,
                    filename=filename,
                    save_optimizer=True,
                    save_param_scheduler=True,
                    meta={"iter": iteration, "epoch": int(runner.epoch)},
                )
                marker.write_text(filename + "\n", encoding="utf-8")
                current = Path(runner.work_dir) / filename
            receipt = publish_recovery_file(
                current,
                self.store_root,
                artifact_id=self.artifact_id,
                campaign_id=self.campaign_id,
                project_commit=self.project_commit,
                metadata={
                    "identity_sha256": self.identity_sha256,
                    "dataloader_iteration": iteration,
                    "optimizer_step": iteration // self.accumulation,
                    "accumulation": self.accumulation,
                },
            )
            self.last_publish = time.monotonic()
            if self.status_path is not None:
                write_campaign_status(
                    self.status_path,
                    state="running" if not due_final else "training_complete_pending_receipt",
                    artifact_id=self.artifact_id,
                    dataloader_iteration=iteration,
                    optimizer_step=iteration // self.accumulation,
                    last_checkpoint=current.name,
                    last_checkpoint_sha256=receipt["sha256"],
                    recovery_generation=receipt["generation"],
                )
            if (
                self.intentional_interrupt_optimizer_step is not None
                and optimizer_step == self.intentional_interrupt_optimizer_step
                and not interrupt_marker.exists()
            ):
                interrupt_marker.write_text(
                    json.dumps(
                        {
                            "record_type": "edgeguard_intentional_interruption",
                            "optimizer_step": optimizer_step,
                            "checkpoint_sha256": receipt["sha256"],
                            "recovery_generation": receipt["generation"],
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n",
                    encoding="utf-8",
                )
                raise RuntimeError(
                    "EDGEGUARD_INTENTIONAL_INTERRUPTION_AFTER_VERIFIED_CHECKPOINT"
                )
            recovery_files = sorted(
                Path(runner.work_dir).glob("recovery_*.pth"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            for stale in recovery_files[2:]:
                stale.unlink(missing_ok=True)

    @hooks_registry.HOOKS.register_module(force=True)
    class EdgeGuardMetricsHistoryHook(hook_base):
        """Append sparse optimizer-step loss history for thesis plots and diagnostics."""

        priority = "LOW"

        def __init__(self, *, accumulation: int, optimizer_interval: int = 50) -> None:
            self.accumulation = accumulation
            self.iteration_interval = optimizer_interval * accumulation
            self.seen: set[int] = set()

        def before_train(self, runner: Any) -> None:
            path = Path(runner.work_dir) / "metrics_history.jsonl"
            if not path.is_file():
                return
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    self.seen.add(int(json.loads(line)["optimizer_step"]))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue

        def after_train_iter(
            self,
            runner: Any,
            batch_idx: int,
            data_batch: Any = None,
            outputs: Any = None,
        ) -> None:
            del batch_idx, data_batch
            iteration = int(runner.iter + 1)
            if iteration % self.accumulation:
                return
            optimizer_step = iteration // self.accumulation
            if (
                iteration % self.iteration_interval
                and iteration < int(runner.max_iters)
            ) or optimizer_step in self.seen:
                return
            scalars: dict[str, float] = {}
            if isinstance(outputs, dict):
                for name, value in outputs.items():
                    try:
                        scalar = float(value.detach().cpu().item())
                    except (AttributeError, TypeError, ValueError, RuntimeError):
                        continue
                    if np.isfinite(scalar):
                        scalars[str(name)] = scalar
            record = {
                "optimizer_step": optimizer_step,
                "dataloader_iteration": iteration,
                "scalars": scalars,
            }
            with (Path(runner.work_dir) / "metrics_history.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            self.seen.add(optimizer_step)

    _REGISTERED = True
