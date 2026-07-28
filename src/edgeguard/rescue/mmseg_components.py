"""Runtime-registered MMSeg components for explicit manifests and domain balance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

        def __iter__(self) -> Any:
            indices = uniform_domain_indices(
                self.lengths,
                total_size=self.global_size,
                seed=self.seed,
                epoch=self.epoch,
            )
            return iter(indices[self.rank : self.global_size : self.world_size])

        def __len__(self) -> int:
            return self.num_samples

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

    _REGISTERED = True
