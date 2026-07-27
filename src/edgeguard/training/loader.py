"""Source-aware mini dataloaders and bounded I/O benchmarking."""

from __future__ import annotations

import platform
import time
from typing import Any

import numpy as np


def recommended_worker_count(*, cpu_count: int | None = None, system: str | None = None) -> int:
    """Select a conservative local/Colab worker count without assuming GPU speed."""
    count = cpu_count if cpu_count is not None else __import__("os").cpu_count() or 1
    name = system or platform.system()
    return 0 if name == "Darwin" else max(0, min(4, count - 1))


class SourceAwareFixtureDataset:
    """Deterministic semantic fixtures with source/sequence metadata and corruption."""

    def __init__(self, *, sample_count: int = 5, corrupt_index: int | None = None) -> None:
        invalid_corrupt_index = corrupt_index is not None and not 0 <= corrupt_index < sample_count
        if sample_count < 1 or invalid_corrupt_index:
            raise ValueError("fixture dataset size/corrupt index is invalid")
        self.sample_count = sample_count
        self.corrupt_index = corrupt_index

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index == self.corrupt_index:
            return {"corrupt": True, "sample_id": f"sample-{index:03d}"}
        generator = np.random.default_rng(1000 + index)
        return {
            "corrupt": False,
            "sample_id": f"sample-{index:03d}",
            "source": "synthetic-a" if index % 2 == 0 else "synthetic-b",
            "sequence_id": f"sequence-{index // 3}",
            "frame_index": index % 3,
            "image": generator.normal(size=(3, 32, 64)).astype(np.float32),
            "target": generator.integers(0, 19, size=(32, 64), dtype=np.int64),
        }


def source_aware_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Filter explicit corrupt records and collate mixed source/sequence identities."""
    torch = __import__("torch")
    valid = [item for item in batch if not item.get("corrupt")]
    if not valid:
        raise ValueError("all records in one loader batch are corrupt")
    return {
        "images": torch.from_numpy(np.stack([item["image"] for item in valid])),
        "targets": torch.from_numpy(np.stack([item["target"] for item in valid])),
        "sample_ids": [item["sample_id"] for item in valid],
        "sources": [item["source"] for item in valid],
        "sequence_ids": [item["sequence_id"] for item in valid],
        "frame_indices": [item["frame_index"] for item in valid],
        "corrupt_sample_ids": [item["sample_id"] for item in batch if item.get("corrupt")],
    }


def build_fixture_loader(
    *,
    sample_count: int = 5,
    batch_size: int = 2,
    workers: int = 0,
    seed: int = 20260727,
    corrupt_index: int | None = None,
    pin_memory: bool = False,
) -> Any:
    """Build an actual Torch loader with deterministic worker and sampler seeds."""
    torch = __import__("torch")
    torch_data = __import__("torch.utils.data", fromlist=["DataLoader"])

    def worker_init(worker_id: int) -> None:
        np.random.seed((seed + worker_id) % 2**32)

    return torch_data.DataLoader(
        SourceAwareFixtureDataset(sample_count=sample_count, corrupt_index=corrupt_index),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        pin_memory=pin_memory,
        collate_fn=source_aware_collate,
        worker_init_fn=worker_init if workers > 0 else None,
        generator=torch.Generator().manual_seed(seed),
    )


def benchmark_fixture_loader(*, workers: int, batches: int = 3) -> dict[str, Any]:
    """Measure the actual mini loader path without claiming Colab throughput."""
    if batches < 1:
        raise ValueError("loader benchmark batch count must be positive")
    loader = build_fixture_loader(sample_count=batches * 2, workers=workers)
    started = time.perf_counter()
    images = 0
    batch_times: list[float] = []
    previous = started
    for index, batch in enumerate(loader):
        now = time.perf_counter()
        batch_times.append(now - previous)
        images += int(batch["images"].shape[0])
        previous = now
        if index + 1 >= batches:
            break
    elapsed = time.perf_counter() - started
    return {
        "workers": workers,
        "persistent_workers": workers > 0,
        "prefetch": 2 if workers > 0 else None,
        "pin_memory": False,
        "batches": batches,
        "images": images,
        "elapsed_seconds": elapsed,
        "images_per_second": images / max(elapsed, 1e-9),
        "mean_data_wait_seconds": float(np.mean(batch_times)),
        "scientific_evidence": False,
        "scope": "local_synthetic_loader_path_only",
    }
