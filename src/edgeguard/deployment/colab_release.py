"""Build deterministic five-model Colab delivery archives."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

from edgeguard.deployment.jetson_bundle import (
    build_jetson_deployment_bundle,
    verify_jetson_deployment_bundle,
)
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload


def _release_artifact(root: Path, payload: object, label: str) -> Path:
    if not isinstance(payload, dict):
        raise ValueError(f"release {label} artifact is invalid")
    relative = PurePosixPath(str(payload.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"release {label} path is unsafe")
    path = root.joinpath(*relative.parts)
    if not path.is_file() or sha256_file(path) != payload.get("sha256"):
        raise ValueError(f"release {label} identity mismatch")
    return path


def _zip_members(destination: Path, members: dict[str, Path | bytes]) -> None:
    with ZipFile(destination, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, source in sorted(members.items()):
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("delivery archive member is unsafe")
            payload = source if isinstance(source, bytes) else source.read_bytes()
            info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"delivery archive CRC verification failed: {destination.name}")
        if set(archive.namelist()) != set(members):
            raise ValueError(f"delivery archive member verification failed: {destination.name}")
        for name, source in members.items():
            expected = source if isinstance(source, bytes) else source.read_bytes()
            if hashlib.sha256(archive.read(name)).digest() != hashlib.sha256(expected).digest():
                raise ValueError(f"delivery archive payload verification failed: {name}")


def build_colab_release_packages(
    *,
    accepted_release: Path,
    work_root: Path,
    project_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Package five verified models for Jetson handoff, thesis, and Streamlit."""
    release = json.loads(accepted_release.read_text(encoding="utf-8"))
    if release.get("record_type") != "edgeguard_accepted_release" or release.get(
        "status"
    ) != "accepted":
        raise ValueError("Colab packaging requires an accepted release")
    if output_root.exists():
        index_path = output_root / "release_index.json"
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            if payload.get("accepted_release_sha256") == sha256_file(accepted_release):
                packages = payload.get("packages")
                if not isinstance(packages, dict) or not packages:
                    raise ValueError("existing delivery index has no packages")
                for name, identity in packages.items():
                    path = output_root / str(name)
                    if (
                        not isinstance(identity, dict)
                        or not path.is_file()
                        or sha256_file(path) != identity.get("sha256")
                        or path.stat().st_size != identity.get("byte_size")
                    ):
                        raise ValueError("existing delivery package identity mismatch")
                return payload
        raise FileExistsError("delivery output exists with another or incomplete identity")
    output_root.mkdir(parents=True)
    release_id = str(release["release_id"])
    recommendation = work_root / "reports/selection/recommended_model.json"
    selection = json.loads(recommendation.read_text(encoding="utf-8"))
    if selection.get("recommended_model") != release.get("recommended_model"):
        raise ValueError("accepted release recommendation identity mismatch")
    models = release.get("models")
    if not isinstance(models, list) or len(models) != 5:
        raise ValueError("Jetson release requires five accepted models")

    model_members: dict[str, Path | bytes] = {}
    demo_models: list[dict[str, Any]] = []
    model_index: list[dict[str, Any]] = []
    ontology = project_root / "configs/dataset/semantic_ontology_v2.yaml"
    preprocessing = {
        "schema_version": "2.0",
        "color_order": "RGB",
        "resize": {"height": 512, "width": 1024, "interpolation": "bilinear"},
        "normalization": {
            "mean": [123.675, 116.28, 103.53],
            "std": [58.395, 57.12, 57.375],
            "pixel_scale": 1.0,
        },
        "model_input": {
            "name": "normalized_rgb",
            "shape": [1, 3, 512, 1024],
            "dtype": "float32",
        },
    }
    for record in models:
        if not isinstance(record, dict):
            raise ValueError("accepted release model record is invalid")
        model = str(record["model"])
        checkpoint = _release_artifact(
            accepted_release.parent, record.get("checkpoint"), f"{model}.checkpoint"
        )
        config = _release_artifact(
            accepted_release.parent, record.get("resolved_config"), f"{model}.config"
        )
        export_root = work_root / "exports/export" / release_id / model
        onnx = export_root / f"{model}.onnx"
        validation_path = onnx.with_suffix(".validation.json")
        golden_input_path = onnx.with_suffix(".golden-input.npy")
        golden_output_path = onnx.with_suffix(".golden-output.npy")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if (
            validation.get("onnx_sha256") != sha256_file(onnx)
            or validation.get("checkpoint_sha256") != sha256_file(checkpoint)
            or validation.get("golden_input_sha256") != sha256_file(golden_input_path)
            or validation.get("golden_output_sha256") != sha256_file(golden_output_path)
            or validation.get("allclose_atol_1e_4_rtol_1e_4") is not True
        ):
            raise ValueError(f"{model} export lacks hash-bound numerical acceptance")
        deployment = output_root / f".{model}.deployment.zip"
        build_colab = build_jetson_deployment_bundle(
            deployment,
            onnx_path=onnx,
            ontology_path=ontology,
            preprocessing=preprocessing,
            model_family=model,
            source_commit=str(release["project_commit"]),
            config_sha256=sha256_file(config),
            checkpoint_sha256=sha256_file(checkpoint),
            golden_input=np.load(golden_input_path, allow_pickle=False),
            golden_output=np.load(golden_output_path, allow_pickle=False),
        )
        verify_colab = verify_jetson_deployment_bundle(deployment)
        if build_colab["bundle_identity"] != verify_colab["bundle_identity"]:
            raise ValueError(f"{model} deployment bundle verification changed identity")
        prefix = f"models/{model}"
        model_members[f"{prefix}/deployment.zip"] = deployment
        model_members[f"{prefix}/checkpoint.pth"] = checkpoint
        model_members[f"{prefix}/resolved.py"] = config
        model_members[f"{prefix}/model.onnx"] = onnx
        model_members[f"{prefix}/golden-input.npy"] = golden_input_path
        model_members[f"{prefix}/golden-output.npy"] = golden_output_path
        model_members[f"{prefix}/onnx_validation.json"] = validation_path
        demo_models.append(
            {
                "label": model,
                "backend": "onnx",
                "model": {
                    "path": f"models/{model}.onnx",
                    "sha256": sha256_file(onnx),
                },
                "datasets": "cityscapes+idd20k",
            }
        )
        model_index.append(
            {
                "model": model,
                "checkpoint_sha256": sha256_file(checkpoint),
                "onnx_sha256": sha256_file(onnx),
                "deployment_bundle_sha256": sha256_file(deployment),
                "recommended": model == release["recommended_model"],
                "tensorrt_engine_included": False,
            }
        )

    jetson_members = {
        **model_members,
        "recommended_model.json": recommendation,
        "accepted_release.json": accepted_release,
        "ontology/semantic_ontology_v2.yaml": ontology,
        "preprocessing.json": (canonical_json(preprocessing) + "\n").encode(),
        "jetson/build_tensorrt.py": project_root / "scripts/jetson/build_tensorrt.py",
        "jetson/benchmark.py": project_root / "scripts/jetson/benchmark.py",
    }
    jetson_index = {
        "schema_version": "1.0",
        "record_type": "edgeguard_jetson_multi_model_release",
        "release_id": release_id,
        "recommended_model": release["recommended_model"],
        "models": model_index,
        "tensorrt_engine_included": False,
        "jetson_measurements": "not_run",
        "artifacts": {
            name: {
                "sha256": hashlib.sha256(
                    source if isinstance(source, bytes) else source.read_bytes()
                ).hexdigest(),
                "byte_size": len(source)
                if isinstance(source, bytes)
                else source.stat().st_size,
            }
            for name, source in sorted(jetson_members.items())
        },
    }
    jetson_members["artifact_index.json"] = (canonical_json(jetson_index) + "\n").encode()
    jetson_archive = output_root / "EdgeGuard_Jetson_Release.zip"
    _zip_members(jetson_archive, jetson_members)

    demo_members: dict[str, Path | bytes] = {}
    for record in models:
        model = str(record["model"])
        onnx = work_root / "exports/export" / release_id / model / f"{model}.onnx"
        demo_members[f"models/{model}.onnx"] = onnx
        demo_members[f"models/{model}.validation.json"] = onnx.with_suffix(
            ".validation.json"
        )
    demo_manifest = {
        "schema_version": "1.0",
        "record_type": "edgeguard_accepted_demo_bundle",
        "status": "accepted",
        "release_id": release_id,
        "recommended_model": release["recommended_model"],
        "models": demo_models,
    }
    demo_members["accepted_demo_bundle.json"] = (canonical_json(demo_manifest) + "\n").encode()
    demo_members["app.py"] = project_root / "app.py"
    demo_members["recommended_model.json"] = recommendation
    demo_members["accepted_release.json"] = accepted_release
    demo_members["comparison/candidate_table.json"] = (
        work_root / "reports/selection/candidate_table.json"
    )
    candidate_payload = json.loads(
        (work_root / "reports/selection/candidate_table.json").read_text(encoding="utf-8")
    )
    candidate_rows = candidate_payload.get("candidates")
    if not isinstance(candidate_rows, list) or len(candidate_rows) != 5:
        raise ValueError("Streamlit comparison requires five measured candidates")
    comparison = io.StringIO()
    writer = csv.DictWriter(
        comparison,
        fieldnames=(
            "model",
            "domain_macro_mIoU",
            "rare_class_mIoU",
            "onnx_median_latency_ms",
            "onnx_bytes",
        ),
    )
    writer.writeheader()
    writer.writerows(
        {name: row.get(name) for name in writer.fieldnames} for row in candidate_rows
    )
    demo_members["comparison/metrics_table.csv"] = comparison.getvalue().encode()
    final_resource = work_root / "pipeline-v3/final/resource.csv"
    if final_resource.is_file():
        demo_members["comparison/resource.csv"] = final_resource
    demo_members["ontology/semantic_ontology_v2.yaml"] = ontology
    demo_members["README.md"] = bytes(
        "# EdgeGuard accepted Streamlit demo\n\n"
        "Install the project demo dependencies, then run from this extracted directory:\n\n"
        "```bash\n"
        "EDGEGUARD_ACCEPTED_BUNDLE_ROOT=. PYTHONPATH=src streamlit run app.py\n"
        "```\n\n"
        "Only the selected model is held in memory. Jetson measurements remain `not_run` "
        "until a real device benchmark is attached.\n",
        "utf-8",
    )
    for source in sorted((project_root / "src/edgeguard").rglob("*.py")):
        demo_members[f"src/{source.relative_to(project_root / 'src').as_posix()}"] = source
    gallery_root = work_root / "reports/gallery" / release_id
    for source in sorted(path for path in gallery_root.rglob("*") if path.is_file()):
        demo_members[f"examples/{source.relative_to(gallery_root).as_posix()}"] = source
    calibration_root = work_root / "calibration" / release_id
    for source in sorted(calibration_root.glob("**/*.json")):
        relative = source.relative_to(calibration_root).as_posix()
        demo_members[f"calibration/{relative}"] = source
    demo_members["jetson/benchmark_status.json"] = (
        canonical_json(
            {
                "record_type": "edgeguard_jetson_benchmark_status",
                "status": "not_run",
                "reason": "requires measurement on the target Jetson device",
            }
        )
        + "\n"
    ).encode()
    streamlit_archive = output_root / "EdgeGuard_Streamlit_Demo.zip"
    _zip_members(streamlit_archive, demo_members)

    thesis_source = work_root / "reports/report" / f"{release_id}.zip"
    if not thesis_source.is_file():
        raise FileNotFoundError("thesis bundle was not produced before delivery packaging")
    thesis_archive = output_root / "EdgeGuard_Thesis_Bundle.zip"
    shutil.copy2(thesis_source, thesis_archive)
    packages = {
        path.name: {"sha256": sha256_file(path), "byte_size": path.stat().st_size}
        for path in (jetson_archive, thesis_archive, streamlit_archive)
    }
    release_index = {
        "schema_version": "1.0",
        "record_type": "edgeguard_colab_release_index",
        "release_id": release_id,
        "recommended_model": release["recommended_model"],
        "accepted_release_sha256": sha256_file(accepted_release),
        "packages": packages,
        "release_identity": sha256_payload(packages),
    }
    (output_root / "release_index.json").write_text(
        canonical_json(release_index) + "\n", encoding="utf-8"
    )
    for path in output_root.glob(".*.deployment.zip"):
        path.unlink()
    return release_index
