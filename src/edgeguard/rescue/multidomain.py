"""Fail-closed multi-domain manifests, label mapping, and uniform sampling."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError

from edgeguard.rescue.dataset import CITYSCAPES_CLASSES, _difference_hash
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload

TRAINING_DATASETS = ("cityscapes", "bdd100k", "idd20k")
EVALUATION_DATASETS = ("acdc", "wilddash2", "muses", "kitti")
SUPPORTED_DATASETS = TRAINING_DATASETS + EVALUATION_DATASETS
SEALED_DATASETS = ("wilddash2", "muses", "kitti")
ROLE_RATIOS = {"train_fit": 0.80, "train_select": 0.15, "train_calibration": 0.05}
EXPECTED_TRAIN_COUNTS = {"cityscapes": 2_975, "bdd100k": 7_000, "idd20k": 14_027}
EXPECTED_VAL_COUNTS = {"cityscapes": 500, "bdd100k": 1_000, "idd20k": 2_036}


@dataclass(frozen=True)
class DomainSample:
    """One root-relative semantic pair and its leakage group."""

    sample_id: str
    dataset_id: str
    group_id: str
    image: str
    mask: str | None
    canonical_mask: str | None
    condition: str | None = None
    city: str | None = None


def load_semantic_ontology(path: Path) -> dict[str, Any]:
    """Load and strictly validate the multi-domain Cityscapes19 mapping."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "2.0":
        raise ValueError("semantic ontology must use schema 2.0")
    classes = payload.get("classes")
    expected = [name.replace(" ", "_") for name in CITYSCAPES_CLASSES]
    if not isinstance(classes, list):
        raise ValueError("semantic ontology classes must be a list")
    actual = [str(row.get("name")) for row in sorted(classes, key=lambda row: row["id"])]
    if actual != expected or [int(row["id"]) for row in classes] != list(range(19)):
        raise ValueError("semantic ontology must preserve the exact Cityscapes19 contract")
    sources = payload.get("sources")
    if not isinstance(sources, dict) or not set(SUPPORTED_DATASETS).issubset(sources):
        raise ValueError("semantic ontology does not cover every approved dataset")
    idd = sources["idd20k"]
    mapping = {int(key): int(value) for key, value in idd.get("map", {}).items()}
    ignored = {int(value) for value in idd.get("ignore_source_ids", [])}
    if set(mapping) & ignored or set(mapping) | ignored != set(range(40)) | {255}:
        raise ValueError("IDD mapping must classify every source ID exactly once")
    if any(value not in range(19) for value in mapping.values()):
        raise ValueError("IDD mapping contains a non-canonical target")
    return payload


def map_source_mask(mask: np.ndarray, dataset_id: str, ontology: dict[str, Any]) -> np.ndarray:
    """Map a native label image to Cityscapes19 without treating unknowns as background."""
    if mask.ndim != 2:
        raise ValueError("semantic masks must be single-channel")
    unique = {int(value) for value in np.unique(mask)}
    if dataset_id != "idd20k":
        invalid = unique - set(range(19)) - {255}
        if invalid:
            raise ValueError(f"{dataset_id} mask contains unknown train IDs: {sorted(invalid)}")
        return mask.astype(np.uint8, copy=True)
    source = ontology["sources"]["idd20k"]
    mapping = {int(key): int(value) for key, value in source["map"].items()}
    ignored = {int(value) for value in source["ignore_source_ids"]}
    unknown = unique - set(mapping) - ignored
    if unknown:
        raise ValueError(f"IDD mask contains unreviewed source IDs: {sorted(unknown)}")
    result = np.full(mask.shape, 255, dtype=np.uint8)
    for source_id, target_id in mapping.items():
        result[mask == source_id] = target_id
    return result


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _image_perceptual_hash(path: Path) -> int:
    with Image.open(path) as image:
        return _difference_hash(image)


def _bdd_samples(root: Path, split: str) -> list[DomainSample]:
    image_root = root / "images" / "10k" / split
    mask_root = root / "labels" / "sem_seg" / "masks" / split
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError("BDD100K requires images/10k and labels/sem_seg/masks")
    masks = {path.stem: path for path in mask_root.glob("*.png")}
    result: list[DomainSample] = []
    for image in sorted((*image_root.glob("*.jpg"), *image_root.glob("*.png"))):
        mask = masks.get(image.stem)
        if mask is None:
            raise ValueError(f"BDD100K image has no semantic mask: {image.name}")
        sequence = image.stem.split("-")[0]
        result.append(
            DomainSample(
                sample_id=image.stem,
                dataset_id="bdd100k",
                group_id=f"bdd100k:{sequence}",
                image=_relative(image, root),
                mask=_relative(mask, root),
                canonical_mask=_relative(mask, root),
            )
        )
    if not result:
        raise FileNotFoundError(f"no BDD100K semantic pairs found in {image_root}")
    return result


def _idd_samples(root: Path, split: str) -> list[DomainSample]:
    image_root = root / "leftImg8bit" / split
    mask_root = root / "gtFine" / split
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError("IDD20K requires leftImg8bit and gtFine split directories")
    masks: dict[str, Path] = {}
    canonical_masks: dict[str, Path] = {}

    def identity(path: Path, base_root: Path, suffix: str) -> str:
        relative = path.relative_to(base_root)
        if len(relative.parts) != 2:
            raise ValueError(f"IDD sample path is malformed: {relative.as_posix()}")
        basename = relative.name.removesuffix(suffix)
        if not basename or basename == relative.name:
            raise ValueError(f"IDD sample suffix is malformed: {relative.as_posix()}")
        return f"{split}/{relative.parent.as_posix()}/{basename}"

    for suffix in ("_gtFine_labelids.png", "_gtFine_labelIds.png"):
        for path in mask_root.glob(f"**/*{suffix}"):
            identifier = identity(path, mask_root, suffix)
            if identifier in masks:
                raise ValueError(f"IDD sample has multiple raw source-ID masks: {identifier}")
            masks[identifier] = path
    for path in mask_root.glob("**/*_gtFine_labelTrainIds.png"):
        identifier = identity(path, mask_root, "_gtFine_labelTrainIds.png")
        if identifier in canonical_masks:
            raise ValueError(f"IDD sample has multiple canonical masks: {identifier}")
        canonical_masks[identifier] = path
    result: list[DomainSample] = []
    images = sorted(
        (
            *image_root.glob("**/*_leftImg8bit.png"),
            *image_root.glob("**/*_leftImg8bit.jpg"),
            *image_root.glob("**/*_leftImg8bit.jpeg"),
        )
    )
    for image in images:
        image_suffix = next(
            suffix
            for suffix in ("_leftImg8bit.png", "_leftImg8bit.jpg", "_leftImg8bit.jpeg")
            if image.name.lower().endswith(suffix.lower())
        )
        identifier = identity(image, image_root, image_suffix)
        mask = masks.get(identifier)
        canonical_mask = canonical_masks.get(identifier)
        if mask is None and canonical_mask is None:
            raise ValueError(f"IDD image lacks source-ID or canonical mask: {image.name}")
        selected_mask = mask or canonical_mask
        assert selected_mask is not None
        sequence = image.parent.relative_to(image_root).as_posix()
        city = image.parent.name
        result.append(
            DomainSample(
                sample_id=identifier,
                dataset_id="idd20k",
                group_id=f"idd20k:{sequence}",
                image=_relative(image, root),
                mask=_relative(selected_mask, root),
                canonical_mask=(
                    _relative(canonical_mask, root) if canonical_mask is not None else None
                ),
                city=city,
            )
        )
    if not result:
        raise FileNotFoundError(f"no IDD20K semantic pairs found in {image_root}")
    return result


def discover_domain_samples(
    root: Path, dataset_id: str, *, split: str = "train"
) -> list[DomainSample]:
    """Discover an approved dataset layout without guessing unsupported variants."""
    if dataset_id == "bdd100k":
        return _bdd_samples(root, split)
    if dataset_id == "idd20k":
        return _idd_samples(root, split)
    raise ValueError("generic discovery currently supports BDD100K and IDD20K training data")


def _split_groups(samples: Sequence[DomainSample], seed: int) -> dict[str, list[DomainSample]]:
    groups: defaultdict[str, list[DomainSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_id].append(sample)
    ordered = sorted(
        groups.values(),
        key=lambda group: hashlib.sha256(f"{seed}:{group[0].group_id}".encode()).hexdigest(),
    )
    targets = {role: len(samples) * ratio for role, ratio in ROLE_RATIOS.items()}
    roles: dict[str, list[DomainSample]] = {role: [] for role in ROLE_RATIOS}
    for group in ordered:
        role = max(
            ROLE_RATIOS,
            key=lambda candidate: (
                targets[candidate] - len(roles[candidate]),
                ROLE_RATIOS[candidate],
            ),
        )
        roles[role].extend(sorted(group, key=lambda sample: sample.sample_id))
    return roles


def _manifest_hash(payload: dict[str, Any]) -> str:
    return sha256_payload(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )


def _near_duplicate_pairs(
    records: Sequence[tuple[str, int]], *, maximum_distance: int = 2
) -> list[dict[str, Any]]:
    """Find dHash candidates efficiently by exact-match pigeonhole bands."""
    bands = ((0, 22), (22, 43), (43, 64))
    buckets: defaultdict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
    pairs: dict[tuple[str, str], int] = {}
    for sample_id, value in records:
        candidates: dict[str, int] = {}
        for band_id, (start, end) in enumerate(bands):
            width = end - start
            key = (band_id, (value >> start) & ((1 << width) - 1))
            for other_id, other_value in buckets[key]:
                candidates[other_id] = other_value
        for other_id, other_value in candidates.items():
            distance = (value ^ other_value).bit_count()
            if distance <= maximum_distance:
                left, right = sorted((sample_id, other_id))
                pair = (left, right)
                pairs[pair] = distance
        for band_id, (start, end) in enumerate(bands):
            width = end - start
            key = (band_id, (value >> start) & ((1 << width) - 1))
            buckets[key].append((sample_id, value))
    return [
        {"sample_id_a": left, "sample_id_b": right, "hamming_distance": distance}
        for (left, right), distance in sorted(pairs.items())
    ]


def validate_dataset_manifest(
    manifest_path: Path,
    *,
    allowed_roles: Iterable[str] | None = None,
    require_frozen: bool = True,
) -> dict[str, Any]:
    """Validate hash, role separation, paths, and freeze/sealed state."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "2.0"
        or payload.get("record_type") != "edgeguard_dataset_manifest"
    ):
        raise ValueError("dataset manifest must use EdgeGuard schema 2.0")
    if payload.get("manifest_sha256") != _manifest_hash(payload):
        raise ValueError("dataset manifest hash mismatch")
    dataset_id = str(payload.get("dataset_id"))
    if dataset_id not in SUPPORTED_DATASETS:
        raise ValueError("dataset manifest names an unsupported dataset")
    if require_frozen and payload.get("split_state") != "frozen":
        raise ValueError("scientific use requires a human-frozen dataset manifest")
    roles = payload.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("dataset manifest roles must be a mapping")
    if allowed_roles is not None and not set(roles).issubset(set(allowed_roles)):
        raise ValueError("dataset manifest contains an unexpected scientific role")
    seen_samples: set[str] = set()
    group_roles: dict[str, str] = {}
    for role, records in roles.items():
        if not isinstance(records, list):
            raise ValueError(f"manifest role {role} is not a list")
        for record in records:
            sample_id = str(record["sample_id"])
            group_id = str(record["group_id"])
            if sample_id in seen_samples:
                raise ValueError(f"sample appears in multiple roles: {sample_id}")
            seen_samples.add(sample_id)
            previous = group_roles.setdefault(group_id, role)
            if previous != role:
                raise ValueError(f"sequence leakage across roles: {group_id}")
            for field in ("image", "mask", "canonical_mask"):
                value = record.get(field)
                if value is not None and Path(str(value)).is_absolute():
                    raise ValueError(f"manifest record field {field} must remain root-relative")
    return payload


def validate_manifest_review_receipt(
    receipt_path: Path,
    *,
    candidate_path: Path,
    dataset_id: str,
    campaign_id: str,
    project_commit: str,
) -> dict[str, Any]:
    """Verify the explicit human decision that authorizes one candidate freeze."""
    if not receipt_path.is_file():
        raise PermissionError("manifest freeze requires a human review receipt")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "2.0"
        or payload.get("record_type") != "edgeguard_manifest_review_receipt"
    ):
        raise ValueError("invalid manifest review receipt")
    if payload.get("decision") != "freeze_approved" or payload.get("human_approved") is not True:
        raise PermissionError("manifest review receipt does not approve freezing")
    if payload.get("dataset_id") != dataset_id:
        raise ValueError("manifest review receipt names another dataset")
    if payload.get("campaign_id") != campaign_id:
        raise ValueError("manifest review receipt belongs to another campaign")
    if payload.get("project_commit") != project_commit:
        raise ValueError("manifest review receipt belongs to another project commit")
    if payload.get("candidate_manifest_sha256") != sha256_file(candidate_path):
        raise ValueError("manifest review receipt candidate hash mismatch")
    reviewer = payload.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("manifest review receipt must identify the human reviewer")
    return payload


def freeze_candidate_manifest(
    candidate_path: Path,
    output_path: Path,
    *,
    review_receipt_path: Path | None = None,
    campaign_id: str | None = None,
    project_commit: str | None = None,
) -> dict[str, Any]:
    """Create an immutable frozen copy only after a hash-bound human review receipt."""
    payload = validate_dataset_manifest(candidate_path, require_frozen=False)
    if payload.get("split_state") != "candidate_requires_human_freeze":
        raise ValueError("only a candidate manifest can be frozen")
    if review_receipt_path is None or campaign_id is None or project_commit is None:
        raise PermissionError(
            "manifest freeze requires a review receipt, campaign, and project commit"
        )
    review = validate_manifest_review_receipt(
        review_receipt_path,
        candidate_path=candidate_path,
        dataset_id=str(payload["dataset_id"]),
        campaign_id=campaign_id,
        project_commit=project_commit,
    )
    payload["split_state"] = "frozen"
    payload["human_freeze_approved"] = True
    payload["campaign_id"] = campaign_id
    payload["project_commit"] = project_commit
    payload["approved_candidate_sha256"] = sha256_file(candidate_path)
    payload["human_review_receipt_sha256"] = sha256_file(review_receipt_path)
    payload["human_reviewer"] = review["reviewer"]
    payload["manifest_sha256"] = _manifest_hash(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def freeze_candidate_manifest_by_policy(
    candidate_path: Path,
    output_path: Path,
    *,
    policy_path: Path,
    campaign_id: str,
    project_commit: str,
) -> dict[str, Any]:
    """Freeze an exact source candidate or post-release validation under owner policy."""
    payload = validate_dataset_manifest(candidate_path, require_frozen=False)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if (
        policy.get("schema_version") != "1.0"
        or policy.get("record_type") != "edgeguard_owner_authorization_policy"
        or policy.get("decision") != "preauthorize_exact_pipeline"
        or policy.get("owner_approved") is not True
        or policy.get("campaign_id") != campaign_id
    ):
        raise PermissionError("dataset policy is not valid for this campaign")
    if payload.get("split_state") != "candidate_requires_human_freeze":
        raise ValueError("only a candidate manifest can be policy-frozen")
    dataset_id = str(payload["dataset_id"])
    if dataset_id not in set(policy.get("scientific_sources", [])):
        raise PermissionError("dataset is outside the owner-authorized source set")
    roles = set(payload.get("roles", {}))
    if roles == {"official_source_val"}:
        if policy.get("official_source_validation_allowed_after_acceptance") is not True:
            raise PermissionError("official source validation is not authorized")
        if payload.get("source_split") != "val" or payload.get("scientific_eligible") is not True:
            raise ValueError("official validation candidate is incomplete or ineligible")
        approval_scope = "post_acceptance_official_source_validation"
    else:
        expected = policy.get("training_manifest_candidates", {}).get(dataset_id)
        if not isinstance(expected, dict):
            raise PermissionError("training candidate has no exact policy identity")
        if sha256_file(candidate_path) != expected.get("file_sha256"):
            raise ValueError("training candidate hash differs from owner authorization")
        valid_count = sum(int(value) for value in payload.get("counts", {}).values())
        if valid_count != int(expected.get("expected_valid_samples", -1)):
            raise ValueError("training candidate count differs from owner authorization")
        excluded = payload.get("excluded_samples", [])
        if not isinstance(excluded, list) or len(excluded) != int(
            expected.get("expected_quarantined_samples", -1)
        ):
            raise ValueError("training candidate quarantine differs from owner authorization")
        approval_scope = "exact_training_manifest_candidate"
    payload["split_state"] = "frozen"
    payload["human_freeze_approved"] = True
    payload["approval_method"] = "owner_preauthorized_policy"
    payload["approval_scope"] = approval_scope
    payload["campaign_id"] = campaign_id
    payload["project_commit"] = project_commit
    payload["approved_candidate_sha256"] = sha256_file(candidate_path)
    payload["authorization_policy_sha256"] = sha256_file(policy_path)
    payload["human_reviewer"] = policy.get("authorized_by")
    payload["manifest_sha256"] = _manifest_hash(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def build_cityscapes_dataset_manifest(
    dataset_root: Path,
    split_manifest: Path,
    output_path: Path,
    *,
    audit_passed: bool,
    frozen: bool,
) -> dict[str, Any]:
    """Wrap the existing reviewed Cityscapes split in the common manifest schema."""
    legacy = json.loads(split_manifest.read_text(encoding="utf-8"))
    roles = legacy.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("Cityscapes split has no roles")
    normalized: dict[str, list[dict[str, Any]]] = {}
    for role, records in roles.items():
        normalized[role] = []
        for record in records:
            normalized[role].append(
                {
                    **record,
                    "dataset_id": "cityscapes",
                    "canonical_mask": record["mask"],
                    "condition": None,
                    "image_sha256": sha256_file(dataset_root / str(record["image"])),
                    "perceptual_hash": (
                        f"{_image_perceptual_hash(dataset_root / str(record['image'])):016x}"
                    ),
                }
            )
    payload: dict[str, Any] = {
        "schema_version": "2.0",
        "record_type": "edgeguard_dataset_manifest",
        "dataset_id": "cityscapes",
        "dataset_root": str(dataset_root.resolve()),
        "prepared_root": str(output_path.parent.resolve()),
        "ontology_sha256": sha256_file(
            Path(__file__).parents[3] / "configs/dataset/semantic_ontology_v2.yaml"
        ),
        "mapping_version": "cityscapes-trainids-v1",
        "source_split_sha256": sha256_file(split_manifest),
        "seed": legacy.get("seed"),
        "split_state": "frozen" if frozen else "candidate_requires_human_freeze",
        "human_freeze_approved": frozen,
        "sealed": False,
        "audit_passed": audit_passed,
        "scientific_eligible": audit_passed
        and sum(len(records) for records in normalized.values())
        == EXPECTED_TRAIN_COUNTS["cityscapes"],
        "roles": normalized,
        "counts": {role: len(records) for role, records in normalized.items()},
        "official_validation_role": "frozen_source_domain_final_only",
    }
    payload["manifest_sha256"] = _manifest_hash(payload)
    output_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def build_cityscapes_official_validation_manifest(
    dataset_root: Path,
    audit_summary: dict[str, Any],
    output_path: Path,
    *,
    source_manifests: Sequence[Path],
    strict_count: bool = True,
) -> dict[str, Any]:
    """Build a review-required Cityscapes val manifest isolated from training roles."""
    if (
        audit_summary.get("record_type") != "cityscapes_dataset_audit"
        or audit_summary.get("source_split") != "val"
        or audit_summary.get("audit_passed") is not True
    ):
        raise ValueError("Cityscapes official validation requires a passing val audit")
    samples = audit_summary.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Cityscapes val audit has no valid samples")
    training_ids: set[str] = set()
    training_groups: set[str] = set()
    training_image_hashes: set[str] = set()
    source_hashes: list[str] = []
    for manifest_path in source_manifests:
        manifest = validate_dataset_manifest(manifest_path)
        source_hashes.append(sha256_file(manifest_path))
        for source_records in manifest["roles"].values():
            for record in source_records:
                training_ids.add(str(record["sample_id"]))
                training_groups.add(str(record["group_id"]))
                if record.get("image_sha256"):
                    training_image_hashes.add(str(record["image_sha256"]))
    records: list[dict[str, Any]] = []
    for sample in samples:
        image = str(sample["image"])
        image_path = dataset_root / image
        image_sha = sha256_file(image_path)
        sample_id = str(sample["sample_id"])
        group_id = str(sample["group_id"])
        if (
            sample_id in training_ids
            or group_id in training_groups
            or image_sha in training_image_hashes
        ):
            raise ValueError("Cityscapes official validation overlaps a training source")
        records.append(
            {
                "sample_id": sample_id,
                "dataset_id": "cityscapes",
                "group_id": group_id,
                "image": image,
                "mask": str(sample["mask"]),
                "canonical_mask": str(sample["mask"]),
                "condition": None,
                "city": sample.get("city"),
                "image_sha256": image_sha,
                "perceptual_hash": f"{_image_perceptual_hash(image_path):016x}",
            }
        )
    expected = EXPECTED_VAL_COUNTS["cityscapes"]
    eligible = len(records) == expected if strict_count else bool(records)
    payload: dict[str, Any] = {
        "schema_version": "2.0",
        "record_type": "edgeguard_dataset_manifest",
        "dataset_id": "cityscapes",
        "dataset_root": str(dataset_root.resolve()),
        "prepared_root": str(output_path.parent.resolve()),
        "ontology_sha256": sha256_file(
            Path(__file__).parents[3] / "configs/dataset/semantic_ontology_v2.yaml"
        ),
        "mapping_version": "cityscapes-trainids-v1",
        "source_split": "val",
        "source_manifest_sha256s": sorted(source_hashes),
        "audit_summary_sha256": sha256_payload(audit_summary),
        "split_state": "candidate_requires_human_freeze",
        "human_freeze_approved": False,
        "sealed": True,
        "audit_passed": True,
        "scientific_eligible": eligible,
        "roles": {"official_source_val": records},
        "counts": {"official_source_val": len(records)},
        "official_validation_role": "frozen_source_domain_final_only",
    }
    payload["manifest_sha256"] = _manifest_hash(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def audit_training_dataset(
    root: Path,
    output_root: Path,
    *,
    dataset_id: str,
    ontology_path: Path,
    seed: int,
    strict_count: bool = True,
    source_split: str = "train",
    source_manifests: Sequence[Path] = (),
    checkpoint_root: Path | None = None,
    quarantine_invalid_source_samples: bool = False,
    maximum_quarantine_rate: float = 0.001,
) -> dict[str, Any]:
    """Audit BDD/IDD train or withheld official validation data."""
    if dataset_id not in {"bdd100k", "idd20k"}:
        raise ValueError("multi-domain training audit supports bdd100k or idd20k")
    if source_split not in {"train", "val"}:
        raise ValueError("BDD/IDD source split must be train or val")
    if source_split == "val" and not source_manifests:
        raise ValueError("official source validation audit requires frozen training manifests")
    ontology = load_semantic_ontology(ontology_path)
    preparation_receipt_path = root / "preparation_receipt.json"
    if preparation_receipt_path.is_file():
        preparation_receipt = json.loads(preparation_receipt_path.read_text(encoding="utf-8"))
        if preparation_receipt.get("dataset_id") != dataset_id:
            raise ValueError("dataset preparation receipt names a different dataset")
        preparation_scientific_eligible = bool(
            preparation_receipt.get("scientific_eligible", False)
        )
        source_profile = str(preparation_receipt.get("source_profile", "unknown"))
        preparation_receipt_sha256 = sha256_file(preparation_receipt_path)
    else:
        preparation_scientific_eligible = True
        source_profile = "legacy_prepared_root_without_receipt"
        preparation_receipt_sha256 = None
    samples = discover_domain_samples(root, dataset_id, split=source_split)
    suffix = "audit" if source_split == "train" else "val_audit"
    report_root = output_root / f"{dataset_id}_{suffix}"
    report_root.mkdir(parents=True, exist_ok=False)
    canonical_root = report_root / "canonical_masks"
    canonical_masks_in_dataset = dataset_id == "idd20k" and all(
        sample.canonical_mask is not None for sample in samples
    )
    catalog_chunk_size = 250
    catalog_identity = sha256_payload(
        {
            "dataset_id": dataset_id,
            "source_split": source_split,
            "dataset_root": str(root.resolve()),
            "ontology_sha256": sha256_file(ontology_path),
            "preparation_receipt_sha256": preparation_receipt_sha256,
            "sample_ids": [sample.sample_id for sample in samples],
        }
    )
    catalog_dir = (
        checkpoint_root / dataset_id / source_split / catalog_identity
        if checkpoint_root is not None and canonical_masks_in_dataset
        else None
    )
    cached_rows: dict[str, dict[str, Any]] = {}
    if catalog_dir is not None:
        catalog_dir.mkdir(parents=True, exist_ok=True)
        for chunk_number, offset in enumerate(range(0, len(samples), catalog_chunk_size)):
            expected_ids = [
                sample.sample_id for sample in samples[offset : offset + catalog_chunk_size]
            ]
            chunk_path = catalog_dir / f"chunk-{chunk_number:04d}.json"
            if not chunk_path.is_file():
                continue
            try:
                chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
                receipt_hash = chunk.pop("receipt_sha256")
                rows = chunk["records"]
                valid = (
                    receipt_hash == sha256_payload(chunk)
                    and chunk.get("catalog_identity") == catalog_identity
                    and [str(row["sample_id"]) for row in rows] == expected_ids
                )
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                valid = False
                rows = []
            if valid:
                for row in rows:
                    cached_rows[str(row["sample_id"])] = row
    if not 0.0 <= maximum_quarantine_rate <= 0.01:
        raise ValueError("maximum quarantine rate must be between 0 and 0.01")
    invalid: list[dict[str, Any]] = []
    exact_hashes: defaultdict[str, list[str]] = defaultdict(list)
    sample_hashes: dict[str, str] = {}
    sample_perceptual_hashes: dict[str, int] = {}
    sample_pixel_counts: dict[str, list[int]] = {}
    perceptual: list[tuple[str, str, int]] = []
    pixel_counts = np.zeros(19, dtype=np.int64)
    ignore_pixels = 0
    total_pixels = 0
    audited: list[DomainSample] = []
    new_rows: dict[str, dict[str, Any]] = {}

    def accept_row(sample: DomainSample, row: dict[str, Any]) -> None:
        nonlocal ignore_pixels, total_pixels
        counts = np.asarray(row["class_pixel_counts"], dtype=np.int64)
        if counts.shape != (19,) or bool((counts < 0).any()) or not bool(counts.any()):
            raise ValueError("cached audit histogram is invalid")
        image_sha = str(row["image_sha256"])
        perceptual_hash = int(str(row["perceptual_hash"]), 16)
        pixel_counts[:] += counts
        ignore_pixels += int(row["ignore_pixels"])
        total_pixels += int(row["total_pixels"])
        exact_hashes[image_sha].append(sample.sample_id)
        sample_hashes[sample.sample_id] = image_sha
        sample_perceptual_hashes[sample.sample_id] = perceptual_hash
        perceptual.append((sample.sample_id, sample.group_id, perceptual_hash))
        sample_pixel_counts[sample.sample_id] = [int(value) for value in counts]
        audited.append(
            DomainSample(**{**asdict(sample), "canonical_mask": str(row["canonical_mask"])})
        )

    for sample_index, sample in enumerate(samples):
        assert sample.mask is not None
        cached = cached_rows.get(sample.sample_id)
        if cached is not None:
            accept_row(sample, cached)
            continue
        try:
            with Image.open(root / sample.image) as image:
                image.load()
                image_size = image.size
                perceptual_hash = _difference_hash(image)
            with Image.open(root / sample.mask) as mask_image:
                native = np.asarray(mask_image)
            canonical_only = dataset_id == "idd20k" and sample.mask == sample.canonical_mask
            canonical = (
                map_source_mask(native, "cityscapes", ontology)
                if canonical_only
                else map_source_mask(native, dataset_id, ontology)
            )
            sample_ignore_pixels = int(np.count_nonzero(canonical == 255))
            if canonical.shape != (image_size[1], image_size[0]):
                raise ValueError("image/mask geometry mismatch")
            valid_counts = np.bincount(canonical[canonical != 255], minlength=19)[:19]
            if not bool(valid_counts.any()):
                raise ValueError("canonical mask contains no usable class")
            image_sha = sha256_file(root / sample.image)
            if dataset_id == "idd20k" and not canonical_masks_in_dataset:
                canonical_path = canonical_root / f"{sample.sample_id}.png"
                canonical_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(canonical, mode="L").save(canonical_path)
                canonical_relative = canonical_path.relative_to(report_root).as_posix()
            else:
                existing_canonical = sample.canonical_mask
                assert existing_canonical is not None
                with Image.open(root / existing_canonical) as canonical_image:
                    prepared_canonical = np.asarray(canonical_image, dtype=np.uint8)
                if not np.array_equal(prepared_canonical, canonical):
                    raise ValueError("prepared canonical mask differs from frozen ontology mapping")
                canonical_relative = existing_canonical
            row = {
                "sample_id": sample.sample_id,
                "canonical_mask": canonical_relative,
                "image_sha256": image_sha,
                "perceptual_hash": f"{perceptual_hash:016x}",
                "class_pixel_counts": [int(value) for value in valid_counts],
                "ignore_pixels": sample_ignore_pixels,
                "total_pixels": int(canonical.size),
            }
            new_rows[sample.sample_id] = row
            accept_row(sample, row)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            message = str(error)
            if message == "canonical mask contains no usable class":
                error_code = "no_usable_canonical_class"
            elif message == "image/mask geometry mismatch":
                error_code = "geometry_mismatch"
            elif isinstance(error, (OSError, UnidentifiedImageError)):
                error_code = "decode_or_io_error"
            else:
                error_code = "contract_violation"
            invalid.append(
                {
                    "sample_id": sample.sample_id,
                    "group_id": sample.group_id,
                    "image": sample.image,
                    "mask": sample.mask,
                    "error_type": type(error).__name__,
                    "error_code": error_code,
                    "error": message,
                }
            )
        chunk_end = (sample_index + 1) % catalog_chunk_size == 0 or sample_index + 1 == len(samples)
        if catalog_dir is not None and chunk_end:
            offset = (sample_index // catalog_chunk_size) * catalog_chunk_size
            chunk_samples = samples[offset : sample_index + 1]
            rows = [
                cached_rows.get(item.sample_id) or new_rows.get(item.sample_id)
                for item in chunk_samples
            ]
            if all(row is not None for row in rows):
                chunk_number = sample_index // catalog_chunk_size
                chunk_payload = {
                    "schema_version": "2.0",
                    "record_type": "edgeguard_audit_catalog_chunk",
                    "catalog_identity": catalog_identity,
                    "chunk_number": chunk_number,
                    "records": rows,
                }
                chunk_payload["receipt_sha256"] = sha256_payload(chunk_payload)
                chunk_path = catalog_dir / f"chunk-{chunk_number:04d}.json"
                incoming = chunk_path.with_name(f".{chunk_path.name}.incoming")
                incoming.write_text(canonical_json(chunk_payload) + "\n", encoding="utf-8")
                incoming.replace(chunk_path)
                print(
                    canonical_json(
                        {
                            "phase": "audit-catalog",
                            "dataset": dataset_id,
                            "completed": sample_index + 1,
                            "total": len(samples),
                        }
                    ),
                    flush=True,
                )
    duplicates = [ids for ids in exact_hashes.values() if len(ids) > 1]
    expected = (
        EXPECTED_TRAIN_COUNTS[dataset_id]
        if source_split == "train"
        else EXPECTED_VAL_COUNTS[dataset_id]
    )
    official_inventory_ok = len(samples) == expected if strict_count else bool(samples)
    accounted_for = len(audited) + len(invalid) == len(samples)
    allowed_quarantine_codes = {
        "no_usable_canonical_class",
        "geometry_mismatch",
        "decode_or_io_error",
    }
    quarantine_limit = max(1, int(expected * maximum_quarantine_rate))
    quarantine_accepted = bool(
        quarantine_invalid_source_samples
        and source_split == "train"
        and invalid
        and len(invalid) <= quarantine_limit
        and all(row["error_code"] in allowed_quarantine_codes for row in invalid)
        and official_inventory_ok
        and accounted_for
    )
    invalid_gate_ok = not invalid or quarantine_accepted
    count_ok = official_inventory_ok and accounted_for
    source_hashes: set[str] = set()
    source_perceptual: list[tuple[str, int]] = []
    for manifest_path in source_manifests:
        source = validate_dataset_manifest(manifest_path, require_frozen=False)
        if source["dataset_id"] not in TRAINING_DATASETS:
            raise ValueError("source overlap audit accepts only training-domain manifests")
        if source.get("split_state") not in {"candidate_requires_human_freeze", "frozen"}:
            raise ValueError(
                "source overlap audit requires a reviewed candidate or frozen manifest"
            )
        if source.get("audit_passed") is not True:
            raise ValueError("source overlap audit requires an audit-passed source manifest")
        source_root = Path(source["dataset_root"])
        for source_records in source["roles"].values():
            for record in source_records:
                source_hashes.add(
                    str(
                        record.get("image_sha256")
                        or sha256_file(source_root / str(record["image"]))
                    )
                )
                value = record.get("perceptual_hash")
                if value is not None:
                    source_perceptual.append(
                        (
                            f"source:{source['dataset_id']}:{record['sample_id']}",
                            int(str(value), 16),
                        )
                    )
    exact_source_overlap = sorted(
        sample_id for sample_id, value in sample_hashes.items() if value in source_hashes
    )
    audited_perceptual = [
        (f"audited:{sample_id}", value) for sample_id, value in sample_perceptual_hashes.items()
    ]
    cross_near_pairs = [
        pair
        for pair in _near_duplicate_pairs((*source_perceptual, *audited_perceptual))
        if {str(pair["sample_id_a"]).split(":", 1)[0], str(pair["sample_id_b"]).split(":", 1)[0]}
        == {"source", "audited"}
    ]
    audit_passed = (
        invalid_gate_ok
        and not duplicates
        and not exact_source_overlap
        and not cross_near_pairs
        and count_ok
        and bool((pixel_counts > 0).any())
    )
    if source_split == "train":
        roles = _split_groups(audited, seed) if audit_passed else {role: [] for role in ROLE_RATIOS}
    else:
        roles = {"official_source_val": audited if audit_passed else []}
    payload: dict[str, Any] = {
        "schema_version": "2.0",
        "record_type": "edgeguard_dataset_manifest",
        "dataset_id": dataset_id,
        "dataset_root": str(root.resolve()),
        "prepared_root": str(
            root.resolve() if canonical_masks_in_dataset else report_root.resolve()
        ),
        "ontology_sha256": sha256_file(ontology_path),
        "mapping_version": ontology["sources"][dataset_id]["mapping_version"],
        "source_profile": source_profile,
        "preparation_receipt_sha256": preparation_receipt_sha256,
        "seed": seed,
        "source_split": source_split,
        "source_manifest_sha256s": sorted(sha256_file(path) for path in source_manifests),
        "split_state": "candidate_requires_human_freeze",
        "human_freeze_approved": False,
        "sealed": False,
        "audit_passed": audit_passed,
        "scientific_eligible": (audit_passed and strict_count and preparation_scientific_eligible),
        "official_train_count_required": expected,
        "official_inventory_count": len(samples),
        "valid_sample_count": len(audited),
        "excluded_samples": invalid if quarantine_accepted else [],
        "data_quality_policy": {
            "policy_id": "source-defect-quarantine-v1",
            "requested": quarantine_invalid_source_samples,
            "accepted": quarantine_accepted,
            "maximum_rate": maximum_quarantine_rate,
            "maximum_count": quarantine_limit,
            "allowed_error_codes": sorted(allowed_quarantine_codes),
            "fail_closed_error_codes": sorted(
                {str(row["error_code"]) for row in invalid} - allowed_quarantine_codes
            ),
        },
        "roles": {
            role: [
                {
                    **asdict(sample),
                    "image_sha256": sample_hashes[sample.sample_id],
                    "perceptual_hash": f"{sample_perceptual_hashes[sample.sample_id]:016x}",
                    "class_pixel_counts": sample_pixel_counts[sample.sample_id],
                }
                for sample in records
            ]
            for role, records in roles.items()
        },
        "official_validation_role": "frozen_source_domain_final_only",
        "counts": {role: len(records) for role, records in roles.items()},
    }
    payload["manifest_sha256"] = _manifest_hash(payload)
    manifest_path = report_root / "dataset_manifest.candidate.json"
    manifest_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "2.0",
        "record_type": "multi_domain_dataset_audit",
        "dataset_id": dataset_id,
        "source_split": source_split,
        "discovered_pairs": len(samples),
        "valid_pairs": len(audited),
        "expected_pairs": expected,
        "strict_count": strict_count,
        "source_profile": source_profile,
        "preparation_scientific_eligible": preparation_scientific_eligible,
        "invalid_count": len(invalid),
        "invalid_error_codes": {
            code: sum(row["error_code"] == code for row in invalid)
            for code in sorted({str(row["error_code"]) for row in invalid})
        },
        "quarantine_requested": quarantine_invalid_source_samples,
        "quarantine_accepted": quarantine_accepted,
        "quarantine_limit": quarantine_limit,
        "exact_duplicate_groups": len(duplicates),
        "perceptual_hashes_computed": len(perceptual),
        "near_duplicate_candidate_pairs": len(
            _near_duplicate_pairs([(sample_id, value) for sample_id, _, value in perceptual])
        ),
        "exact_source_overlap_count": len(exact_source_overlap),
        "near_source_overlap_count": len(cross_near_pairs),
        "class_pixel_counts": [int(value) for value in pixel_counts],
        "ignore_pixels": ignore_pixels,
        "total_pixels": total_pixels,
        "ignore_pixel_ratio": (float(ignore_pixels / total_pixels) if total_pixels else None),
        "audit_passed": audit_passed,
        "candidate_manifest": manifest_path.name,
        "training_allowed": False,
        "evaluation_allowed": False,
        "reason": "candidate manifest requires explicit freeze after review",
    }
    (report_root / "summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
    (report_root / "invalid_samples.json").write_text(
        canonical_json({"records": invalid}) + "\n", encoding="utf-8"
    )
    return summary


def audit_evaluation_dataset(
    root: Path,
    output_root: Path,
    *,
    dataset_id: str,
    pairs_file: Path,
    ontology_path: Path,
    source_url: str,
    license_id: str,
    access_date: str,
    source_manifests: Sequence[Path] = (),
) -> dict[str, Any]:
    """Audit an exact external pair list without inferring an unstable vendor layout."""
    if dataset_id not in EVALUATION_DATASETS:
        raise ValueError("external audit requires ACDC, WildDash2, MUSES, or KITTI")
    ontology = load_semantic_ontology(ontology_path)
    raw = json.loads(pairs_file.read_text(encoding="utf-8"))
    records = raw.get("samples") if isinstance(raw, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("external pairs file must contain a non-empty samples list")
    submission_encoding = raw.get("submission_encoding")
    if dataset_id in SEALED_DATASETS and submission_encoding not in {
        "canonical_train_ids",
        "cityscapes_label_ids",
    }:
        raise ValueError("sealed pairs must declare the official submission encoding")
    if dataset_id == "wilddash2" and submission_encoding != "cityscapes_label_ids":
        raise ValueError("WildDash2 submissions require regular Cityscapes label IDs")
    report_root = output_root / f"{dataset_id}_audit"
    report_root.mkdir(parents=True, exist_ok=False)
    samples: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    image_hashes: defaultdict[str, list[str]] = defaultdict(list)
    for row in records:
        try:
            sample_id = str(row["sample_id"])
            image_relative = Path(str(row["image"]))
            mask_value = row.get("mask")
            mask_relative = Path(str(mask_value)) if mask_value is not None else None
            if image_relative.is_absolute() or (mask_relative and mask_relative.is_absolute()):
                raise ValueError("external pair paths must be root-relative")
            image_path = root / image_relative
            with Image.open(image_path) as image:
                image.load()
                image_size = image.size
                perceptual_hash = _difference_hash(image)
            mask_sha: str | None = None
            if mask_relative is not None:
                with Image.open(root / mask_relative) as mask_image:
                    mask = np.asarray(mask_image)
                canonical = map_source_mask(mask, dataset_id, ontology)
                if canonical.shape != (image_size[1], image_size[0]):
                    raise ValueError("external image/mask geometry mismatch")
                mask_sha = sha256_file(root / mask_relative)
            elif dataset_id not in {"wilddash2", "muses"}:
                raise ValueError(f"{dataset_id} evaluation record requires a mask")
            image_sha = sha256_file(image_path)
            image_hashes[image_sha].append(sample_id)
            samples.append(
                {
                    "sample_id": sample_id,
                    "dataset_id": dataset_id,
                    "group_id": str(row.get("group_id", f"{dataset_id}:{sample_id}")),
                    "image": image_relative.as_posix(),
                    "mask": mask_relative.as_posix() if mask_relative else None,
                    "canonical_mask": mask_relative.as_posix() if mask_relative else None,
                    "condition": row.get("condition"),
                    "city": row.get("city"),
                    "image_sha256": image_sha,
                    "perceptual_hash": f"{perceptual_hash:016x}",
                    "mask_sha256": mask_sha,
                    "submission_name": row.get("submission_name"),
                }
            )
        except (KeyError, OSError, UnidentifiedImageError, ValueError) as error:
            invalid.append({"sample_id": str(row.get("sample_id", "unknown")), "error": str(error)})
    duplicates = [ids for ids in image_hashes.values() if len(ids) > 1]
    source_hashes: set[str] = set()
    source_perceptual: list[tuple[str, int]] = []
    for manifest_path in source_manifests:
        source = validate_dataset_manifest(manifest_path)
        if source["dataset_id"] not in TRAINING_DATASETS:
            raise ValueError("external overlap audit accepts only source training manifests")
        source_root = Path(source["dataset_root"])
        for role_records in source["roles"].values():
            for record in role_records:
                identity = f"source:{source['dataset_id']}:{record['sample_id']}"
                source_hashes.add(
                    str(
                        record.get("image_sha256")
                        or sha256_file(source_root / str(record["image"]))
                    )
                )
                perceptual_value = record.get("perceptual_hash")
                if perceptual_value is not None:
                    source_perceptual.append((identity, int(str(perceptual_value), 16)))
    excluded_ids = {
        str(sample["sample_id"])
        for sample in samples
        if str(sample["image_sha256"]) in source_hashes
    }
    external_perceptual = [
        (f"external:{sample['sample_id']}", int(str(sample["perceptual_hash"]), 16))
        for sample in samples
    ]
    near_pairs = _near_duplicate_pairs((*source_perceptual, *external_perceptual))
    for pair in near_pairs:
        values = (str(pair["sample_id_a"]), str(pair["sample_id_b"]))
        if any(value.startswith("source:") for value in values):
            excluded_ids.update(
                value.removeprefix("external:") for value in values if value.startswith("external:")
            )
    excluded = [sample for sample in samples if str(sample["sample_id"]) in excluded_ids]
    samples = [sample for sample in samples if str(sample["sample_id"]) not in excluded_ids]
    audit_passed = not invalid and not duplicates and bool(samples)
    role = "domain_shift_val" if dataset_id == "acdc" else "sealed_external_test"
    payload: dict[str, Any] = {
        "schema_version": "2.0",
        "record_type": "edgeguard_dataset_manifest",
        "dataset_id": dataset_id,
        "dataset_root": str(root.resolve()),
        "prepared_root": str(report_root.resolve()),
        "ontology_sha256": sha256_file(ontology_path),
        "mapping_version": ontology["sources"][dataset_id]["mapping_version"],
        "source_url": source_url,
        "license_id": license_id,
        "access_date": access_date,
        "source_pairs_sha256": sha256_file(pairs_file),
        "submission_encoding": submission_encoding,
        "source_manifest_sha256s": sorted(sha256_file(path) for path in source_manifests),
        "split_state": "candidate_requires_human_freeze",
        "human_freeze_approved": False,
        "sealed": dataset_id in SEALED_DATASETS,
        "audit_passed": audit_passed,
        "roles": {role: samples if audit_passed else []},
        "counts": {role: len(samples) if audit_passed else 0},
        "excluded_source_overlap": excluded,
    }
    payload["manifest_sha256"] = _manifest_hash(payload)
    manifest_path = report_root / "dataset_manifest.candidate.json"
    manifest_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "2.0",
        "record_type": "external_dataset_audit",
        "dataset_id": dataset_id,
        "sealed": payload["sealed"],
        "listed_samples": len(records),
        "valid_samples": len(samples),
        "invalid_count": len(invalid),
        "duplicate_groups": len(duplicates),
        "excluded_source_overlap_count": len(excluded),
        "audit_passed": audit_passed,
        "candidate_manifest": manifest_path.name,
        "evaluation_allowed": False,
        "reason": "candidate manifest requires freeze; sealed data additionally requires release",
    }
    (report_root / "summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
    (report_root / "invalid_samples.json").write_text(
        canonical_json({"records": invalid}) + "\n", encoding="utf-8"
    )
    return summary


def _bounded_mean_one_weights(counts: np.ndarray) -> list[float]:
    """Project median-frequency weights onto the frozen [0.5, 5.0] mean-one box."""
    if counts.shape != (19,) or bool((counts <= 0).any()):
        raise ValueError("all 19 classes require positive pooled train_fit counts")
    frequencies = counts.astype(np.float64) / float(counts.sum())
    raw = np.median(frequencies) / frequencies
    low, high = 0.0, 100.0
    for _ in range(100):
        scale = (low + high) / 2.0
        if float(np.clip(raw * scale, 0.5, 5.0).mean()) < 1.0:
            low = scale
        else:
            high = scale
    weights = np.clip(raw * ((low + high) / 2.0), 0.5, 5.0)
    return [float(value) for value in weights]


def write_multidomain_statistics(
    manifest_paths: Sequence[Path], output_root: Path
) -> dict[str, Any]:
    """Freeze pooled train-fit frequencies, rare five, and duplicate evidence."""
    if not manifest_paths:
        raise ValueError("at least one frozen training manifest is required")
    manifests = [validate_dataset_manifest(path) for path in manifest_paths]
    dataset_ids = [str(payload["dataset_id"]) for payload in manifests]
    if len(dataset_ids) != len(set(dataset_ids)) or not set(dataset_ids).issubset(
        TRAINING_DATASETS
    ):
        raise ValueError("statistics require unique approved training domains")
    pixel_counts = np.zeros(19, dtype=np.int64)
    image_hashes: defaultdict[str, list[str]] = defaultdict(list)
    perceptual_records: list[tuple[str, int]] = []
    per_domain_counts: dict[str, list[int]] = {}
    for payload in manifests:
        root = Path(payload["dataset_root"])
        prepared = Path(payload["prepared_root"])
        domain_counts = np.zeros(19, dtype=np.int64)
        for role_records in payload["roles"].values():
            for record in role_records:
                identity = f"{payload['dataset_id']}:{record['sample_id']}"
                image_sha = record.get("image_sha256") or sha256_file(root / str(record["image"]))
                image_hashes[str(image_sha)].append(identity)
                perceptual_value = record.get("perceptual_hash")
                if perceptual_value is not None:
                    perceptual_records.append((identity, int(str(perceptual_value), 16)))
        records = payload["roles"].get("train_fit")
        if not isinstance(records, list) or not records:
            raise ValueError("each training manifest needs a non-empty train_fit role")
        for record in records:
            cached_counts = record.get("class_pixel_counts")
            if isinstance(cached_counts, list) and len(cached_counts) == 19:
                sample_counts = np.asarray(cached_counts, dtype=np.int64)
                if bool((sample_counts < 0).any()):
                    raise ValueError("cached canonical histogram contains a negative count")
                domain_counts += sample_counts
                continue
            canonical = record.get("canonical_mask")
            if canonical is None:
                raise ValueError("train_fit record has no canonical mask")
            mask_path = (
                prepared / str(canonical)
                if payload["dataset_id"] == "idd20k"
                else root / str(canonical)
            )
            with Image.open(mask_path) as image:
                mask = np.asarray(image, dtype=np.uint8)
            invalid = {int(value) for value in np.unique(mask)} - set(range(19)) - {255}
            if invalid:
                raise ValueError(f"canonical mask contains invalid labels: {sorted(invalid)}")
            domain_counts += np.bincount(mask[mask != 255], minlength=19)[:19]
        pixel_counts += domain_counts
        per_domain_counts[str(payload["dataset_id"])] = [int(value) for value in domain_counts]
    if any(len(values) > 1 for values in image_hashes.values()):
        raise ValueError("exact duplicate images cross training domains or split records")
    near_duplicates = _near_duplicate_pairs(perceptual_records)
    weights = _bounded_mean_one_weights(pixel_counts)
    rare_ids = sorted(range(19), key=lambda index: (int(pixel_counts[index]), index))[:5]
    output_root.mkdir(parents=True, exist_ok=False)
    hashes = sorted(sha256_file(path) for path in manifest_paths)
    weights_payload = {
        "schema_version": "2.0",
        "record_type": "multi_domain_class_weights",
        "source_role": "train_fit",
        "dataset_manifest_sha256s": hashes,
        "method": "pooled_median_frequency_bounded_mean_one",
        "minimum": 0.5,
        "maximum": 5.0,
        "pixel_counts": [int(value) for value in pixel_counts],
        "per_domain_pixel_counts": per_domain_counts,
        "weights": weights,
    }
    rare_payload = {
        "schema_version": "2.0",
        "record_type": "multi_domain_rare_classes",
        "source_role": "train_fit",
        "dataset_manifest_sha256s": hashes,
        "method": "bottom_five_pooled_train_fit_pixel_frequency",
        "groups": {"rare": rare_ids},
        "class_names": list(CITYSCAPES_CLASSES),
    }
    (output_root / "class_weights.json").write_text(
        canonical_json(weights_payload) + "\n", encoding="utf-8"
    )
    (output_root / "rare_classes.json").write_text(
        canonical_json(rare_payload) + "\n", encoding="utf-8"
    )
    from edgeguard.reporting.dataset_figures import build_dataset_figures

    figure_report = build_dataset_figures(
        manifests,
        per_domain_counts=per_domain_counts,
        pooled_counts=[int(value) for value in pixel_counts],
        weights=weights,
        rare_ids=rare_ids,
        output_root=output_root,
    )
    result = {
        "schema_version": "2.0",
        "record_type": "multi_domain_statistics",
        "datasets": dataset_ids,
        "dataset_manifest_sha256s": hashes,
        "rare_class_ids": rare_ids,
        "duplicate_groups": 0,
        "near_duplicate_candidate_pairs": len(near_duplicates),
        "figures_generated": figure_report["figures_generated"],
    }
    (output_root / "near_duplicates.json").write_text(
        canonical_json({"records": near_duplicates}) + "\n", encoding="utf-8"
    )
    (output_root / "summary.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result


def uniform_domain_indices(
    lengths: Sequence[int], *, total_size: int, seed: int, epoch: int = 0
) -> list[int]:
    """Return deterministic global indices with equal expected mass per domain."""
    if not lengths or any(length <= 0 for length in lengths) or total_size <= 0:
        raise ValueError("domain lengths and total_size must be positive")
    offsets = np.cumsum((0, *lengths[:-1]))
    rng = np.random.default_rng(seed + epoch)
    domain_order = np.arange(total_size, dtype=np.int64) % len(lengths)
    rng.shuffle(domain_order)
    result: list[int] = []
    per_domain_draws = [0] * len(lengths)
    permutations = [rng.permutation(length).tolist() for length in lengths]
    for domain in domain_order:
        domain_id = int(domain)
        draw = per_domain_draws[domain_id]
        if draw and draw % lengths[domain_id] == 0:
            permutations[domain_id] = rng.permutation(lengths[domain_id]).tolist()
        local = permutations[domain_id][draw % lengths[domain_id]]
        result.append(int(offsets[domain_id]) + int(local))
        per_domain_draws[domain_id] += 1
    return result


def iter_uniform_domain_indices(
    lengths: Sequence[int], *, total_size: int, seed: int, epoch: int = 0
) -> Iterator[int]:
    """Yield the deterministic balanced sequence for sampler adapters."""
    yield from uniform_domain_indices(lengths, total_size=total_size, seed=seed, epoch=epoch)


def domain_mixture_probabilities(lengths: Sequence[int], *, alpha: float) -> list[float]:
    """Return size-power domain probabilities for a controlled data ablation."""
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("domain lengths must be positive")
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("domain mixture alpha must be finite and between 0 and 1")
    weights = np.power(np.asarray(lengths, dtype=np.float64), alpha)
    probabilities = weights / weights.sum()
    return [float(value) for value in probabilities]


def power_domain_indices(
    lengths: Sequence[int],
    *,
    total_size: int,
    alpha: float,
    seed: int,
    epoch: int = 0,
) -> list[int]:
    """Draw deterministic indices with exact largest-remainder size-power quotas."""
    if total_size <= 0:
        raise ValueError("domain mixture total_size must be positive")
    probabilities = np.asarray(domain_mixture_probabilities(lengths, alpha=alpha), dtype=np.float64)
    ideal = probabilities * total_size
    quotas = np.floor(ideal).astype(np.int64)
    remaining = int(total_size - int(quotas.sum()))
    remainder_order = sorted(
        range(len(lengths)), key=lambda index: (-(ideal[index] - quotas[index]), index)
    )
    for domain_id in remainder_order[:remaining]:
        quotas[domain_id] += 1
    rng = np.random.default_rng(seed + epoch)
    domain_order = np.concatenate(
        [np.full(int(quota), index, dtype=np.int64) for index, quota in enumerate(quotas)]
    )
    rng.shuffle(domain_order)
    offsets = np.cumsum((0, *lengths[:-1]))
    permutations = [rng.permutation(length).tolist() for length in lengths]
    draws = [0] * len(lengths)
    result: list[int] = []
    for domain in domain_order:
        domain_id = int(domain)
        draw = draws[domain_id]
        if draw and draw % lengths[domain_id] == 0:
            permutations[domain_id] = rng.permutation(lengths[domain_id]).tolist()
        result.append(
            int(offsets[domain_id]) + int(permutations[domain_id][draw % lengths[domain_id]])
        )
        draws[domain_id] += 1
    return result


def verify_sealed_release(
    manifest_path: Path, checkpoint: Path, release_path: Path | None
) -> dict[str, Any]:
    """Require a hash-bound human release before external inference."""
    manifest = validate_dataset_manifest(manifest_path)
    if manifest.get("dataset_id") not in SEALED_DATASETS or not manifest.get("sealed"):
        return {"sealed": False, "release_required": False}
    if release_path is None or not release_path.is_file():
        raise PermissionError("sealed external evaluation requires --sealed-release")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    required = {
        "record_type": "edgeguard_sealed_release",
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_selection_frozen": True,
        "human_approved": True,
    }
    if any(release.get(key) != value for key, value in required.items()):
        raise PermissionError("sealed release does not match the frozen manifest/checkpoint")
    return release
