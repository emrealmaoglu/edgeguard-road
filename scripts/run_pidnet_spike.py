"""Run the isolated PIDNet-S single-image vertical slice."""

from __future__ import annotations

import argparse
import socket
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image
from pydantic import ValidationError

from edgeguard.config import config_sha256, load_pidnet_spike_config
from edgeguard.contracts import validate_pipeline_shapes
from edgeguard.data.single_image import (
    build_upstream_sample_manifest,
    load_rgb_image,
    write_single_image_manifest,
)
from edgeguard.models.pidnet_spike import (
    PIDNetSpikeError,
    difference_summary,
    preprocess_pidnet_rgb,
    run_pidnet_forward,
    verify_upstream_checkout,
)
from edgeguard.provenance import detect_git_provenance, experiment_fingerprint
from edgeguard.scoring.uncertainty import (
    msp_anomaly_score,
    predictive_entropy,
    semantic_mask,
)
from edgeguard.serialization import canonical_json, sha256_array, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pidnet_spike.yaml"))
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-access-date", required=True, help="download date: YYYY-MM-DD")
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument(
        "--sample-access-date", required=True, help="checkout access date: YYYY-MM-DD"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dev/pidnet_spike"))
    return parser


def _shape(array: npt.NDArray[Any]) -> list[int]:
    return [int(dimension) for dimension in array.shape]


def _array_record(array: npt.NDArray[Any]) -> dict[str, Any]:
    return {
        "shape": _shape(array),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "sha256": sha256_array(array),
    }


def _visualize_map(array: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    minimum = np.min(array)
    maximum = np.max(array)
    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.uint8)
    normalized = (array - minimum) / (maximum - minimum)
    return np.round(normalized * np.float32(255.0)).astype(np.uint8)


def _write_artifacts(
    output_dir: Path,
    *,
    image_manifest: dict[str, Any],
    metadata: dict[str, Any],
    native_logits: npt.NDArray[np.float32],
    aligned_logits: npt.NDArray[np.float32],
    mask: npt.NDArray[np.int64],
    msp: npt.NDArray[np.float32],
    entropy: npt.NDArray[np.float32],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "image_manifest.json"
    tensor_path = output_dir / "tensors.npz"
    metadata_path = output_dir / "metadata.json"
    mask_path = output_dir / "semantic_mask.png"
    msp_path = output_dir / "msp.png"
    entropy_path = output_dir / "predictive_entropy.png"

    write_single_image_manifest(image_manifest, manifest_path)
    np.savez_compressed(
        tensor_path,
        native_logits=native_logits,
        aligned_logits=aligned_logits,
        semantic_mask=mask,
        msp=msp,
        predictive_entropy=entropy,
    )
    Image.fromarray(mask[0].astype(np.uint8)).save(mask_path)
    Image.fromarray(_visualize_map(msp[0])).save(msp_path)
    Image.fromarray(_visualize_map(entropy[0])).save(entropy_path)

    metadata["artifact_files"] = {
        "image_manifest": manifest_path.name,
        "tensors": tensor_path.name,
        "tensors_sha256": sha256_file(tensor_path),
        "semantic_mask_visualization": mask_path.name,
        "msp_visualization": msp_path.name,
        "predictive_entropy_visualization": entropy_path.name,
    }
    metadata_path.write_text(canonical_json(metadata) + "\n", encoding="utf-8")
    return {
        "image_manifest": str(manifest_path),
        "metadata": str(metadata_path),
        "tensors": str(tensor_path),
        "semantic_mask": str(mask_path),
        "msp": str(msp_path),
        "predictive_entropy": str(entropy_path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one approved image without downloading any external resource."""
    config = load_pidnet_spike_config(args.config)
    if (
        len(args.checkpoint_access_date) != 10
        or args.checkpoint_access_date[4] != "-"
        or args.checkpoint_access_date[7] != "-"
    ):
        raise PIDNetSpikeError("--checkpoint-access-date must use YYYY-MM-DD")
    try:
        checkpoint_access_date = date.fromisoformat(args.checkpoint_access_date)
    except ValueError as error:
        raise PIDNetSpikeError("--checkpoint-access-date must use YYYY-MM-DD") from error
    if (
        len(args.sample_access_date) != 10
        or args.sample_access_date[4] != "-"
        or args.sample_access_date[7] != "-"
    ):
        raise PIDNetSpikeError("--sample-access-date must use YYYY-MM-DD")
    try:
        sample_access_date = date.fromisoformat(args.sample_access_date)
    except ValueError as error:
        raise PIDNetSpikeError("--sample-access-date must use YYYY-MM-DD") from error

    verify_upstream_checkout(
        args.upstream_checkout,
        expected_repository_url=config.upstream.repository_url,
        expected_commit=config.upstream.commit,
    )
    selected_sample = config.sample.primary
    sample_kind = "primary"
    image_path = args.upstream_checkout / selected_sample.relative_path
    if not image_path.is_file():
        selected_sample = config.sample.fallback
        sample_kind = "fallback"
        image_path = args.upstream_checkout / selected_sample.relative_path
    if not image_path.is_file():
        raise PIDNetSpikeError("neither approved upstream sample exists in the verified checkout")
    image_manifest = build_upstream_sample_manifest(
        image_path,
        checkout_root=args.upstream_checkout,
        sample_id=f"pidnet-upstream-sample-{sample_kind}",
        upstream_repository=config.upstream.repository_url,
        upstream_commit=config.upstream.commit,
        source_access_date=sample_access_date.isoformat(),
        expected_relative_path=selected_sample.relative_path,
        expected_filename=selected_sample.filename,
        expected_sha256=selected_sample.sha256,
        expected_shape=selected_sample.original_shape,
    )
    raw_image = load_rgb_image(image_path)
    model_input = preprocess_pidnet_rgb(
        raw_image,
        height=config.input.height,
        width=config.input.width,
        pixel_scale=config.preprocess.pixel_scale,
        mean=config.preprocess.mean,
        std=config.preprocess.std,
    )
    forward = run_pidnet_forward(
        model_input,
        checkout=args.upstream_checkout,
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        config=config,
    )

    mask = semantic_mask(forward.aligned_logits)
    msp = msp_anomaly_score(forward.aligned_logits)
    entropy = predictive_entropy(forward.aligned_logits)
    repeated_mask = semantic_mask(forward.repeated_aligned_logits)
    repeated_msp = msp_anomaly_score(forward.repeated_aligned_logits)
    repeated_entropy = predictive_entropy(forward.repeated_aligned_logits)
    validate_pipeline_shapes(model_input, forward.aligned_logits, msp)
    validate_pipeline_shapes(model_input, forward.aligned_logits, entropy)

    checkpoint_sha256 = forward.checkpoint_load_report["sha256"]
    config_digest = config_sha256(config)
    git = detect_git_provenance(args.config.resolve().parent)
    fingerprint = experiment_fingerprint(
        config_sha256=config_digest,
        contract_version=config.contract_version,
        pipeline_name=config.pipeline_name,
        backend=config.model.backend,
        scorer="msp+predictive_entropy",
        git=git,
        dataset_manifest_sha256=image_manifest["manifest_sha256"],
        model_artifact_sha256=checkpoint_sha256,
    )
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "pidnet_single_image_spike",
        "claim_scope": "engineering_plumbing_only",
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname() or "unknown-host",
        "command": list(sys.argv),
        "config_sha256": config_digest,
        "experiment_fingerprint": fingerprint,
        "git_commit": git.commit,
        "git_state": git.state.value,
        "git_dirty": git.dirty,
        "device": forward.device,
        "image_manifest_sha256": image_manifest["manifest_sha256"],
        "sample_selection": sample_kind,
        "upstream": forward.checkpoint_load_report["upstream"],
        "checkpoint": {
            **forward.checkpoint_load_report,
            "repository_page": config.checkpoint.repository_page,
            "official_file_url": config.checkpoint.official_file_url,
            "official_collection_url": config.checkpoint.official_collection_url,
            "source_reference_access_date": config.checkpoint.source_reference_access_date,
            "file_access_date": checkpoint_access_date.isoformat(),
            "license_status": config.checkpoint.license_status,
            "permitted_use": config.checkpoint.permitted_use,
        },
        "model_input": _array_record(model_input),
        "native_logits": _array_record(forward.native_logits),
        "aligned_logits": _array_record(forward.aligned_logits),
        "native_logits_shape": _shape(forward.native_logits),
        "native_logits_sha256": sha256_array(forward.native_logits),
        "aligned_logits_shape": _shape(forward.aligned_logits),
        "aligned_logits_sha256": sha256_array(forward.aligned_logits),
        "alignment_mode": config.alignment.mode,
        "alignment_target": config.alignment.target,
        "align_corners": config.alignment.align_corners,
        "semantic_mask": _array_record(mask),
        "msp": _array_record(msp),
        "predictive_entropy": _array_record(entropy),
        "semantic_mask_logits_kind": "aligned_logits",
        "msp_logits_kind": "aligned_logits",
        "entropy_logits_kind": "aligned_logits",
        "score_direction": "higher_means_more_anomalous",
        "score_claim": "not_anomaly_probability",
        "repeatability": {
            "mask_equal": bool(np.array_equal(mask, repeated_mask)),
            "native_logits": difference_summary(
                forward.native_logits, forward.repeated_native_logits
            ),
            "aligned_logits": difference_summary(
                forward.aligned_logits, forward.repeated_aligned_logits
            ),
            "msp": difference_summary(msp, repeated_msp),
            "predictive_entropy": difference_summary(entropy, repeated_entropy),
        },
    }
    paths = _write_artifacts(
        args.output_dir,
        image_manifest=image_manifest,
        metadata=metadata,
        native_logits=forward.native_logits,
        aligned_logits=forward.aligned_logits,
        mask=mask,
        msp=msp,
        entropy=entropy,
    )
    return {"status": "ok", "artifacts": paths, "metadata": metadata}


def main() -> int:
    """Parse arguments and emit one canonical success or failure record."""
    try:
        result = run(_parser().parse_args())
    except (OSError, ValueError, ValidationError) as error:
        print(
            canonical_json(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ),
            file=sys.stderr,
        )
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
