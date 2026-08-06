#!/usr/bin/env python3
"""Build the two output-free semantic-first Colab delivery notebooks."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
NOTEBOOK_ROOT = ROOT / "notebooks"


def _source(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def _cell(cell_type: str, text: str) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": _source(text),
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def _metadata(name: str) -> dict[str, Any]:
    return {
        "colab": {"name": name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }


def _write(path: Path, cells: list[dict[str, Any]]) -> None:
    payload = {
        "cells": cells,
        "metadata": _metadata(path.name),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _pin_delivery(cells: list[dict[str, Any]], *, branch: str, project_commit: str) -> None:
    for cell in cells:
        source = "".join(cell["source"])
        source = re.sub(r'(?m)^BRANCH = "[^"]+"$', f'BRANCH = "{branch}"', source)
        source = re.sub(
            r'(?m)^EXPECTED_PROJECT_COMMIT = "[0-9a-f]{40}"$',
            f'EXPECTED_PROJECT_COMMIT = "{project_commit}"',
            source,
        )
        cell["source"] = _source(source)


def build_preflight_notebook(*, branch: str, project_commit: str) -> None:
    cells = [
        _cell(
            "markdown",
            """
# EdgeGuard · Drive veri ön-hazırlığı

Bu notebook veri indirme yetkisi vermez ve lisans kabulünü otomatikleştirmez. Resmî paketleri Drive'a yerleştirdikten sonra klasör düzenini denetler ve her veri setini tek, SHA-256 bağlı `.tar` dosyasına dönüştürür. Eğitim notebook'u Drive'daki binlerce küçük dosyayı okumak yerine bu paketleri `/content` alanına taşır.

Çekirdek eğitim için yalnız **Cityscapes Fine + IDD20K Part I/II** gerekir. Kaggle BDD yalnız mühendislik verisidir; ACDC ve kapalı external setler model dondurulmadan açılmaz. Tekrar çalıştırma doğrulanmış bundle ve hash receipt'lerini yeniden üretmez.
""",
        ),
        _cell(
            "code",
            """
import os
import sys
from pathlib import Path

LOCAL_TEST_MODE = os.environ.get("EDGEGUARD_NOTEBOOK_LOCAL_TEST") == "1"
if LOCAL_TEST_MODE:
    PROJECT_ROOT = Path(os.environ.get("EDGEGUARD_PROJECT_ROOT", Path.cwd())).resolve()
    DRIVE_ROOT = Path(os.environ["EDGEGUARD_TEST_DRIVE_ROOT"]).resolve()
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
else:
    from google.colab import drive

    drive.mount("/content/drive")
    PROJECT_ROOT = Path("/content/edgeguard-road")
    DRIVE_ROOT = Path("/content/drive/MyDrive")

REPOSITORY = "https://github.com/emrealmaoglu/edgeguard-road.git"
BRANCH = "stabilize/colab-v2"
EXPECTED_PROJECT_COMMIT = "5cc578cb9f15aa7a560108840f3055ae2f4e4733"
SCIENTIFIC_SOURCE_DATASETS = ["cityscapes", "idd20k"]
PROVISIONAL_ENGINEERING_DATASETS = ["bdd100k"]
OPTIONAL_FINAL_DATASETS = []  # Model/protokol freeze sonrası ör. ["acdc"]
PREFLIGHT_TARGET = "scientific_sources"
DEEP_VERIFY_ARCHIVES = False  # İlk hazırlık/final provenance dışında False kalır.
RUN_ARCHIVE_PREPARATION = not LOCAL_TEST_MODE  # Run all yalnız eksik bilimsel bundle'ları üretir.
BDD_SOURCE_PROFILE = "kaggle_mirror"  # Drive'daki bdd100k.zip; yalnız audit/smoke kanıtıdır.
PREPARE_PROVISIONAL_BDD = False
DATASETS_TO_BUNDLE = ["cityscapes", "idd20k"] + OPTIONAL_FINAL_DATASETS
if PREPARE_PROVISIONAL_BDD:
    DATASETS_TO_BUNDLE.append("bdd100k")
CREATE_BUNDLES = True  # Hazırlanan yerel kökten doğrudan tek Drive tar üretir.
REPLACE_BUNDLES = False  # Yalnız kaynak klasörü bilinçli değiştiyse True yapın.
REUSE_VERIFIED_LEGACY = False  # Legacy train korunur; canonical bundle resmî val'i de içerir.
REPAIR_STALE_EPHEMERAL_PREPARATION = True  # Yalnız iki sabit /content çalışma kökünü temizler.
DOWNLOAD_LATEST_FAILURE_REPORT = False  # Hata sonrası son hücreyi bununla yeniden çalıştırın.


def persist_bootstrap_failure(notebook, stage, error):
    import json
    import re
    import traceback
    import uuid
    from datetime import datetime, timezone
    from zipfile import ZIP_DEFLATED, ZipFile

    failed_at = datetime.now(timezone.utc)
    failure_id = f"{failed_at.strftime('%Y%m%dT%H%M%S.%fZ')}-{stage}-{uuid.uuid4().hex[:8]}"
    root = DRIVE_ROOT / "EdgeGuard/failures/bootstrap" / failure_id
    root.mkdir(parents=True, exist_ok=False)
    rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    rendered = re.sub(r"(?i)(token|password|secret|api[_-]?key)=\\S+", r"\\1=<redacted>", rendered)
    payload = {"record_type": "edgeguard_colab_bootstrap_failure", "failure_id": failure_id, "failed_at": failed_at.isoformat(), "notebook": notebook, "stage": stage, "error_type": type(error).__name__, "traceback": rendered}
    report = root / "failure.json"
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    package = root / "failure-report.zip"
    with ZipFile(package, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(report, arcname="failure.json")
    print("EDGEGUARD BOOTSTRAP FAILURE:", package)
    return package
""",
        ),
        _cell(
            "code",
            """
import subprocess


def run_bootstrap_command(command):
    completed = subprocess.run(command, capture_output=True, text=True)
    output = "\\n".join(
        value.strip() for value in (completed.stdout, completed.stderr) if value.strip()
    )
    if output:
        print(output)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Bootstrap command failed with exit code {completed.returncode}: {command}\\n"
            + output[-8000:]
        )
    return completed


try:
    if LOCAL_TEST_MODE:
        print("LOCAL_TEST_MODE: Drive mount, clone ve paket kurulumu atlandı.")
    elif not (PROJECT_ROOT / ".git").is_dir():
        run_bootstrap_command(["git", "clone", "--branch", BRANCH, REPOSITORY, str(PROJECT_ROOT)])
    else:
        run_bootstrap_command(["git", "-C", str(PROJECT_ROOT), "fetch", "origin", BRANCH])
        run_bootstrap_command(["git", "-C", str(PROJECT_ROOT), "checkout", BRANCH])
        run_bootstrap_command(["git", "-C", str(PROJECT_ROOT), "pull", "--ff-only"])
except BaseException as error:
    persist_bootstrap_failure("EdgeGuard_Data_Preflight_Colab.ipynb", "git-clone-or-update", error)
    raise
if not LOCAL_TEST_MODE:
    if EXPECTED_PROJECT_COMMIT:
        run_bootstrap_command(
            ["git", "-C", str(PROJECT_ROOT), "checkout", EXPECTED_PROJECT_COMMIT]
        )
PROJECT_COMMIT = subprocess.run(
    ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
# A Colab kernel can survive a notebook Git update. Purge modules loaded from the
# previous checkout so imports below always match PROJECT_COMMIT.
for loaded_module in list(sys.modules):
    if loaded_module == "edgeguard" or loaded_module.startswith("edgeguard."):
        del sys.modules[loaded_module]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from edgeguard.rescue.colab_failures import (  # noqa: E402
    ColabFailureReporter,
    run_logged_command,
)
FAILURE_REPORTER = ColabFailureReporter(
    DRIVE_ROOT / "EdgeGuard/failures/data-preflight" / PROJECT_COMMIT,
    notebook="EdgeGuard_Data_Preflight_Colab.ipynb",
    project_commit=PROJECT_COMMIT,
    context={"branch": BRANCH, "local_test_mode": LOCAL_TEST_MODE},
)
COMMAND_LOG_ROOT = Path(os.environ.get("EDGEGUARD_TEST_CONTENT_ROOT", "/content")) / "edgeguard-command-logs"
FAILURE_REPORTER.add_diagnostic_root("manifests", DRIVE_ROOT / "EdgeGuard/manifests")
FAILURE_REPORTER.add_diagnostic_root("command-logs", COMMAND_LOG_ROOT)
FAILURE_REPORTER.install_ipython_hook()


def run_colab_command(command, *, check=True):
    command_env = os.environ.copy()
    project_src = str(PROJECT_ROOT / "src")
    command_env["PYTHONPATH"] = project_src + os.pathsep + command_env.get("PYTHONPATH", "")
    return run_logged_command(
        command,
        log_root=COMMAND_LOG_ROOT,
        stage=FAILURE_REPORTER.stage,
        check=check,
        cwd=PROJECT_ROOT,
        env=command_env,
    )


if not LOCAL_TEST_MODE:
    FAILURE_REPORTER.set_stage("project-install")
    run_colab_command([sys.executable, "-m", "pip", "install", "-e", str(PROJECT_ROOT)])
""",
        ),
        _cell(
            "code",
            """
# Drive klasörlerini oluştur, erişim talimatlarını ve eksikleri tek raporda göster.
import json

from edgeguard.rescue.colab_data import load_colab_data_access

FAILURE_REPORTER.set_stage("drive-inventory-and-hashing")
PREFLIGHT_REPORT = DRIVE_ROOT / "EdgeGuard/manifests/colab-data-inventory.json"
inventory_plan = load_colab_data_access(PROJECT_ROOT / "configs/dataset/colab_data_access_v1.yaml")
inventory_command = [
    sys.executable,
    str(PROJECT_ROOT / "scripts/prepare_colab_data.py"),
    "--drive-root",
    str(DRIVE_ROOT),
    "--output",
    str(PREFLIGHT_REPORT),
    "inventory",
]
if DEEP_VERIFY_ARCHIVES:
    inventory_command.append("--hash-archives")
run_colab_command(inventory_command)
inventory = json.loads(PREFLIGHT_REPORT.read_text())
for row in inventory["datasets"]:
    print("\\n", row["dataset_id"], "=>", row["state"], "|", row["activation_phase"])
    print("resmî kaynak:", row["official_url"])
    print("işlem:", row["instructions"])
    if row["missing_required_paths"]:
        print("eksik hazır yollar:", row["missing_required_paths"])
    for package in row["packages"]:
        print(
            "paket:", package["filename"],
            "Drive'da:", package["present"],
            "konum:", package["location_profile"],
            "hash:", package["hash_status"],
        )
        if package.get("hash_error"):
            print("hash okuma uyarısı:", package["hash_error"])
    for package in row.get("engineering_packages", []):
        print(
            "mühendislik paketi:", package["filename"],
            "Drive'da:", package["present"],
            "profil:", package["source_profile"],
            "bilimsel:", package["scientific_eligible"],
            "hash:", package["hash_status"],
        )
        if package.get("hash_error"):
            print("hash okuma uyarısı:", package["hash_error"])
    if row.get("legacy_compatibility"):
        print("legacy uyumluluk:", row["legacy_compatibility"])
""",
        ),
        _cell(
            "markdown",
            """
## Arşivden güvenli hazırlama

Notebook hem yeni `MyDrive/EdgeGuard/archives/<dataset_id>/` düzenini hem de mevcut `MyDrive/EdgeGuard/private_inputs/` klasörünü salt-okunur girdi olarak tanır. Giriş bilgisi, cookie veya geçici indirme URL'sini notebook'a yazmayın. Datasetleri elle açmayın. `RUN_ARCHIVE_PREPARATION=True` olduğunda notebook arşivleri sırayla `/content` alanına kopyalar, hash doğrular, güvenli biçimde hazırlar ve doğrudan tek dosyalı Drive bundle üretir.

Mevcut `private_inputs/bdd100k.zip` Kaggle kaynağıdır. Bundle smoke/plumbing ve veri kataloğu için hazırlanır fakat bilimsel manifest olamaz. Ana bilimsel kaynaklar bu nedenle Cityscapes + IDD20K olarak ayarlanmıştır; resmî iki BDD paketi daha sonra gelirse BDD yeniden ana karşılaştırmaya alınabilir.

IDD polygon JSON etiketleri pinned AutoNUE source-ID sözleşmesiyle maskeye çevrilir; Part II JPG görüntüleri korunur. Native polygon ve source-ID maskeler ayrı kalır.
""",
        ),
        _cell(
            "code",
            """
# Arşivleri dataset bazında hazırla; büyük ağaçları Drive'a küçük dosyalar hâlinde yazma.
import shutil
import time

from edgeguard.rescue.colab_data import copy_archive_to_local, preparation_disk_budget
from edgeguard.serialization import canonical_json, sha256_file

FAILURE_REPORTER.set_stage("dataset-preparation-and-bundling")
CONTENT_ROOT = Path(os.environ.get("EDGEGUARD_TEST_CONTENT_ROOT", "/content"))
PREPARE_ROOT = CONTENT_ROOT / "edgeguard-prepare"
CACHE_ROOT = CONTENT_ROOT / "edgeguard-archive-cache"
archive_root = DRIVE_ROOT / "EdgeGuard/archives"
IDD_SHARD_ROOT = DRIVE_ROOT / "EdgeGuard/prepared/v2/idd20k/shards"


class preparation_phase:
    # prepare_dataset emits structured counters itself. Recursively walking the growing
    # tree every minute only competes with extraction and mask rendering for disk I/O.
    def __init__(self, dataset, phase):
        self.dataset = dataset
        self.phase = phase
        self.started = 0.0

    def __enter__(self):
        self.started = time.monotonic()
        print(
            f"EDGEGUARD PHASE dataset={self.dataset} phase={self.phase['value']} started",
            flush=True,
        )

    def __exit__(self, error_type, error, traceback):
        print(
            f"EDGEGUARD PHASE dataset={self.dataset} phase={self.phase['value']} "
            f"elapsed_min={(time.monotonic() - self.started) / 60:.1f} "
            f"complete={error is None}",
            flush=True,
        )


def _cache_receipt_path(local_archive):
    return local_archive.with_name(local_archive.name + ".copy-receipt.json")


def reuse_or_copy_archive(source, local, source_record):
    # Reuse only a same-source, same-size, hash-verified ephemeral archive copy.
    receipt_path = _cache_receipt_path(local)
    expected_sha256 = source_record.get("sha256") or source_record.get("published_sha256")
    reusable = False
    if local.is_file() and not local.is_symlink():
        try:
            same_size = local.stat().st_size == source.stat().st_size
            if receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                reusable = (
                    receipt.get("source") == str(source.resolve())
                    and int(receipt.get("byte_size", -1)) == source.stat().st_size
                    and same_size
                    and receipt.get("expected_sha256") == expected_sha256
                    and receipt.get("copied_sha256") == expected_sha256
                )
            else:
                # A cache produced by the previous notebook has no sidecar. A pinned
                # or freshly inventoried digest is sufficient to adopt it safely.
                reusable = same_size and expected_sha256 is not None
            if reusable and expected_sha256 and not receipt_path.is_file():
                print("Yerel cache SHA-256 doğrulanıyor:", local.name, flush=True)
                reusable = sha256_file(local) == expected_sha256
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            reusable = False
    if reusable:
        print("Doğrulanmış /content arşiv cache'i yeniden kullanılıyor:", local.name)
        return {"source": str(source), "destination": str(local), "byte_size": local.stat().st_size, "status": "reused"}
    local.unlink(missing_ok=True)
    receipt_path.unlink(missing_ok=True)
    print("Drive arşivi /content alanına kopyalanıyor:", source.name)
    copy_receipt = copy_archive_to_local(source, local, attempts=3)
    copied_sha256 = copy_receipt.get("sha256") or copy_receipt.get("copied_sha256")
    if copied_sha256 is None:
        # Defensive compatibility for a helper retained in a long-lived Colab kernel.
        # The completed /content copy is reused; the 11+ GiB Drive transfer is not repeated.
        print("Kopya receipt alanı eski; yerel arşiv SHA-256 doğrulanıyor:", local.name)
        copied_sha256 = sha256_file(local)
    if expected_sha256 is not None and copied_sha256 != expected_sha256:
        local.unlink(missing_ok=True)
        raise RuntimeError(f"Copied archive SHA-256 mismatch: {source.name}")
    cache_receipt = {
        "source": str(source.resolve()),
        "byte_size": source.stat().st_size,
        "expected_sha256": expected_sha256,
        "copied_sha256": copied_sha256,
    }
    receipt_path.write_text(canonical_json(cache_receipt) + "\\n", encoding="utf-8")
    return {**copy_receipt, "status": "copied"}


if RUN_ARCHIVE_PREPARATION:
    if PREPARE_ROOT.exists():
        if not REPAIR_STALE_EPHEMERAL_PREPARATION:
            raise RuntimeError("Stale preparation root found; inspect before retrying")
        if PREPARE_ROOT.is_symlink() or PREPARE_ROOT.name != "edgeguard-prepare":
            raise RuntimeError(f"Refusing unsafe preparation cleanup: {PREPARE_ROOT}")
        shutil.rmtree(PREPARE_ROOT)
    if CACHE_ROOT.is_symlink() or CACHE_ROOT.name != "edgeguard-archive-cache":
        raise RuntimeError(f"Refusing unsafe archive cache: {CACHE_ROOT}")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS_TO_BUNDLE:
        row = next(item for item in inventory["datasets"] if item["dataset_id"] == dataset)
        if dataset == "idd20k" and (IDD_SHARD_ROOT / "idd20k.shards.json").is_file():
            print("idd20k doğrulanmış shard index ile yeniden kullanılacak.")
            continue
        legacy = row.get("legacy_compatibility") or {}
        if REUSE_VERIFIED_LEGACY and legacy.get("usable_for_training_staging"):
            print(dataset, "mevcut doğrulanmış legacy bundle ile yeniden kullanılacak.")
            continue
        if dataset == "bdd100k" and BDD_SOURCE_PROFILE == "kaggle_mirror" and row.get("ineligible_smoke_bundle_usable"):
            print(dataset, "mevcut provisional mirror bundle ile yeniden kullanılacak.")
            continue
        if row.get("canonical_bundle_usable") and (
            dataset != "bdd100k" or BDD_SOURCE_PROFILE == "official"
        ):
            print(dataset, "mevcut canonical bundle ile yeniden kullanılacak.")
            continue
        if dataset == "bdd100k" and BDD_SOURCE_PROFILE == "kaggle_mirror":
            engineering = [
                item for item in row["engineering_packages"]
                if item["source_profile"] == "kaggle_mirror"
            ]
            source_records = engineering
        else:
            source_records = row["packages"]
        sources = [Path(item["path"]) for item in source_records]
        missing = [str(path) for path in sources if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing archives: " + ", ".join(missing))
        budget = preparation_disk_budget(inventory_plan, tuple(sources), CONTENT_ROOT)
        print(dataset, "hazırlık disk kapısı:", budget)
        dataset_cache = CACHE_ROOT / dataset
        dataset_cache.mkdir(parents=True, exist_ok=True)
        local_archives = []
        for source, source_record in zip(sources, source_records, strict=True):
            local = dataset_cache / source.name
            copy_receipt = reuse_or_copy_archive(source, local, source_record)
            print("Arşiv kopyası tamamlandı:", copy_receipt)
            local_archives.append(local)
        prepared = PREPARE_ROOT / dataset
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/prepare_dataset.py"),
            "--dataset", dataset,
            "--destination", str(prepared),
            "--ontology", str(PROJECT_ROOT / "configs/dataset/semantic_ontology_v2.yaml"),
        ]
        for archive in local_archives:
            command.extend(["--archive", str(archive)])
        if dataset == "bdd100k":
            command.extend(["--source-profile", BDD_SOURCE_PROFILE])
        if dataset == "idd20k":
            command.extend(["--idd-shard-root", str(IDD_SHARD_ROOT), "--idd-shard-size", "500"])
        phase = {"value": "archive-verify-extract-map"}
        with preparation_phase(dataset, phase):
            run_colab_command(command)
            if CREATE_BUNDLES and dataset != "idd20k":
                phase["value"] = "bundle-write-and-hash"
                bundle = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/prepare_colab_data.py"),
                    "--drive-root", str(DRIVE_ROOT),
                    "bundle", "--dataset", dataset,
                    "--source-root", str(prepared),
                ]
                if REPLACE_BUNDLES:
                    bundle.append("--replace")
                run_colab_command(bundle)
        shutil.rmtree(dataset_cache)
        shutil.rmtree(prepared)
    if CACHE_ROOT.is_dir():
        remaining_cache = sorted(path.name for path in CACHE_ROOT.iterdir())
        if remaining_cache:
            print("Yeniden deneme için korunan yerel arşiv cache'leri:", remaining_cache)
        else:
            CACHE_ROOT.rmdir()
    if PREPARE_ROOT.is_dir():
        PREPARE_ROOT.rmdir()
    post_inventory_command = [value for value in inventory_command if value != "--hash-archives"]
    run_colab_command(post_inventory_command)
    inventory = json.loads(PREFLIGHT_REPORT.read_text())
else:
    print("RUN_ARCHIVE_PREPARATION=False: arşiv yükleme ve inventory incelemesi bekleniyor.")
""",
        ),
        _cell(
            "code",
            """
# Eğitim notebook'una geçiş kapısı: bilimsel ve provisional durumlar ayrı raporlanır.
FAILURE_REPORTER.set_stage("preflight-readiness-gate")
bundle_root = DRIVE_ROOT / "EdgeGuard/bundles"
missing = []
for dataset in SCIENTIFIC_SOURCE_DATASETS:
    row = next(item for item in inventory["datasets"] if item["dataset_id"] == dataset)
    legacy = row.get("legacy_compatibility") or {}
    if dataset == "idd20k" and (DRIVE_ROOT / "EdgeGuard/prepared/v2/idd20k/shards/idd20k.shards.json").is_file():
        continue
    if row.get("canonical_bundle_usable") or legacy.get("usable_for_training_staging"):
        continue
    for suffix in (".prepared.tar", ".prepared.tar.receipt.json"):
        candidate = bundle_root / f"{dataset}{suffix}"
        if not candidate.is_file():
            missing.append(str(candidate))
if missing:
    print("Bilimsel eğitim öncesi eksikler:\\n- " + "\\n- ".join(missing))
else:
    print("BİLİMSEL VERİ KAPISI GEÇTİ — Cityscapes + IDD20K hazır.")
bdd_row = next(item for item in inventory["datasets"] if item["dataset_id"] == "bdd100k")
print("BDD provisional mirror bundle:", bdd_row.get("ineligible_smoke_bundle_usable", False))
print("BDD resmî bilimsel bundle:", bdd_row.get("canonical_bundle_usable", False))
print("Hata raporu kökü:", FAILURE_REPORTER.output_root)
if DOWNLOAD_LATEST_FAILURE_REPORT:
    latest_failure = FAILURE_REPORTER.latest_package()
    if latest_failure is None:
        raise RuntimeError("İndirilecek hata paketi bulunamadı")
    if not LOCAL_TEST_MODE:
        from google.colab import files

        files.download(str(latest_failure))
    print("Hata paketi:", latest_failure)
""",
        ),
    ]
    _pin_delivery(cells, branch=branch, project_commit=project_commit)
    _write(NOTEBOOK_ROOT / "EdgeGuard_Data_Preflight_Colab.ipynb", cells)


def build_training_notebook(*, branch: str, project_commit: str) -> None:
    existing = json.loads((NOTEBOOK_ROOT / "EdgeGuard_Road_Colab.ipynb").read_text())

    def find_cell(*markers: str) -> dict[str, Any]:
        return next(
            cell
            for cell in existing["cells"]
            if any(marker in "".join(cell["source"]) for marker in markers)
        )

    final_protocol = find_cell("FINAL_PROTOCOL_CODE =")
    setup_stack = find_cell("Pinned compatibility cascade", "One pinned hermetic runtime")
    audit = find_cell("Audit is the hard scientific gate", "dataset-audit-and-freeze")
    training = find_cell(
        "Five equal-protocol random-init runs",
        "resumable-training",
        "resumable-python-orchestrator",
    )
    evaluation = find_cell(
        "Frozen evaluation and ONNX export",
        "resumable-evaluation-export-hpo",
        "pipeline-artifact-review",
    )
    protocol_text = "".join(final_protocol["source"])
    protocol_text = protocol_text.replace(
        'source_val_manifests = {\n        "bdd100k": WORK_ROOT / "manifests/official-validation/bdd100k.frozen.json",\n        "idd20k": WORK_ROOT / "manifests/official-validation/idd20k.frozen.json",\n    }',
        'source_val_manifests = {dataset: WORK_ROOT / "manifests/official-validation" / f"{dataset}.frozen.json" for dataset in SECONDARY_SCIENTIFIC_DATASETS}',
    )
    protocol_text = protocol_text.replace(
        'zip(("cityscapes", "bdd100k", "idd20k"), DATA_MANIFESTS, strict=True)',
        "zip(SCIENTIFIC_SOURCE_DATASETS, DATA_MANIFESTS, strict=True)",
    )
    protocol_text = protocol_text.replace("subprocess.run(", "run_colab_command(")
    protocol_text = protocol_text.replace(
        'checkpoints = sorted(run_dir.glob("*.pth"))\n\n        if not checkpoints:\n\n            raise RuntimeError(f"Missing final checkpoint for {model}")\n\n        checkpoint = checkpoints[-1]',
        "checkpoint = latest_checkpoint(run_dir)",
    )
    protocol_text = '''# ruff: noqa: E501
FINAL_PROTOCOL_CODE = r"""# Final-only calibration and withheld evaluation with hash-bound receipts.
if RUN_FINAL_EVALUATION:
    if not 1 <= len(FINAL_MODELS) <= 2:
        raise RuntimeError("Freeze one or two FINAL_MODELS before final evaluation")
    source_val_manifests = {
        dataset: WORK_ROOT / "manifests/official-validation" / f"{dataset}.frozen.json"
        for dataset in SECONDARY_SCIENTIFIC_DATASETS
    }
    for model in FINAL_MODELS:
        run_dir = RUN_ROOT / "final" / model / "ce"
        checkpoint = ensure_checkpoint("final", model)
        checkpoint_sha = sha256_file(checkpoint)
        evidence = []
        for dataset, manifest in zip(SCIENTIFIC_SOURCE_DATASETS, DATA_MANIFESTS, strict=True):
            target = WORK_ROOT / "evaluation/calibration" / model / dataset
            evidence_path = target / "calibration-evidence.npz"
            calibration_inputs = {
                "checkpoint_sha256": checkpoint_sha,
                "dataset_manifest_sha256": sha256_file(manifest),
            }
            if not completion_is_valid(target, expected_inputs=calibration_inputs):
                quarantine_incomplete(target)
                run_colab_command([
                    str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/evaluate.py"), "run",
                    "--resolved-config", str(run_dir / "resolved.py"),
                    "--checkpoint", str(checkpoint), "--dataset", dataset,
                    "--dataset-manifest", str(manifest), "--role", "train_calibration",
                    "--save-calibration-evidence", str(evidence_path),
                    "--output-dir", str(target),
                ])
                write_completion_receipt(
                    target, artifact_type="source_calibration_evidence",
                    required_paths=["calibration-evidence.npz", "evaluation.json", "frame_uncertainty.json"],
                    inputs=calibration_inputs, metadata={"model": model, "dataset": dataset},
                )
            evidence.append(evidence_path)
        calibration_root = WORK_ROOT / "calibration" / model
        temperature = calibration_root / "global-temperature.json"
        temperature_inputs = {
            f"evidence_{index}_sha256": sha256_file(path)
            for index, path in enumerate(evidence)
        }
        if not completion_is_valid(calibration_root, expected_inputs=temperature_inputs):
            quarantine_incomplete(calibration_root)
            calibration_root.mkdir(parents=True, exist_ok=True)
            command = [
                str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/evaluate.py"),
                "calibrate-global", "--output", str(temperature),
            ]
            for item in evidence:
                command.extend(["--evidence", str(item)])
            run_colab_command(command)
            write_completion_receipt(
                calibration_root, artifact_type="global_temperature",
                required_paths=["global-temperature.json"], inputs=temperature_inputs,
                metadata={"model": model, "source_domains": SCIENTIFIC_SOURCE_DATASETS},
            )

        city_output = WORK_ROOT / "evaluation/final" / model / "cityscapes"
        city_inputs = {
            "checkpoint_sha256": checkpoint_sha,
            "temperature_sha256": sha256_file(temperature),
            "preparation": preparation_identity(CITYSCAPES_ROOT),
        }
        if not completion_is_valid(city_output, expected_inputs=city_inputs):
            quarantine_incomplete(city_output)
            run_colab_command([
                str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/evaluate.py"), "run",
                "--resolved-config", str(run_dir / "resolved.py"),
                "--checkpoint", str(checkpoint), "--dataset", "cityscapes",
                "--dataset-root", str(CITYSCAPES_ROOT), "--role", "official_val_common_eval",
                "--temperature-file", str(temperature), "--rare-classes-file", str(RARE),
                "--output-dir", str(city_output),
            ])
            write_completion_receipt(
                city_output, artifact_type="official_source_evaluation",
                required_paths=["evaluation.json", "frame_uncertainty.json"],
                inputs=city_inputs, metadata={"model": model, "dataset": "cityscapes"},
            )
        for dataset, manifest in source_val_manifests.items():
            if not manifest.is_file():
                raise RuntimeError(f"Missing frozen official validation manifest: {manifest}")
            target = WORK_ROOT / "evaluation/final" / model / dataset
            final_inputs = {
                "checkpoint_sha256": checkpoint_sha,
                "temperature_sha256": sha256_file(temperature),
                "dataset_manifest_sha256": sha256_file(manifest),
            }
            if completion_is_valid(target, expected_inputs=final_inputs):
                continue
            quarantine_incomplete(target)
            run_colab_command([
                str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/evaluate.py"), "run",
                "--resolved-config", str(run_dir / "resolved.py"),
                "--checkpoint", str(checkpoint), "--dataset", dataset,
                "--dataset-manifest", str(manifest), "--role", "official_source_val",
                "--temperature-file", str(temperature), "--rare-classes-file", str(RARE),
                "--output-dir", str(target),
            ])
            write_completion_receipt(
                target, artifact_type="official_source_evaluation",
                required_paths=["evaluation.json", "frame_uncertainty.json"],
                inputs=final_inputs, metadata={"model": model, "dataset": dataset},
            )
        if RUN_ACDC:
            if not ALLOW_FINAL_DATA:
                raise RuntimeError("ACDC is sealed until ALLOW_FINAL_DATA=True")
            for condition in ("fog", "night", "rain", "snow"):
                target = WORK_ROOT / "evaluation/acdc" / model / condition
                external_inputs = {
                    "checkpoint_sha256": checkpoint_sha,
                    "temperature_sha256": sha256_file(temperature),
                    "condition": condition,
                }
                if completion_is_valid(target, expected_inputs=external_inputs):
                    continue
                quarantine_incomplete(target)
                run_colab_command([
                    str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/evaluate.py"), "run",
                    "--resolved-config", str(run_dir / "resolved.py"),
                    "--checkpoint", str(checkpoint), "--dataset", "acdc",
                    "--dataset-root", str(ACDC_ROOT), "--role", "domain_shift_val",
                    "--condition", condition, "--temperature-file", str(temperature),
                    "--rare-classes-file", str(RARE), "--output-dir", str(target),
                ])
                write_completion_receipt(
                    target, artifact_type="acdc_domain_shift_evaluation",
                    required_paths=["evaluation.json", "frame_uncertainty.json"],
                    inputs=external_inputs, metadata={"model": model, "condition": condition},
                )
        sync_work_snapshot(f"final-evaluation-{model}")

if RUN_SEALED_PACKAGE:
    if len(FINAL_MODELS) != 1 or not SEALED_MANIFEST.is_file() or not SEALED_RELEASE.is_file():
        raise RuntimeError("Sealed packaging requires one frozen model, manifest, and release")
    model = FINAL_MODELS[0]
    export_dir = WORK_ROOT / "exports/final" / model
    onnx_model = export_dir / f"{model}.onnx"
    output = WORK_ROOT / "external-package" / model
    sealed_inputs = {
        "onnx_sha256": sha256_file(onnx_model),
        "manifest_sha256": sha256_file(SEALED_MANIFEST),
        "release_sha256": sha256_file(SEALED_RELEASE),
    }
    if not completion_is_valid(output, expected_inputs=sealed_inputs):
        quarantine_incomplete(output)
        run_colab_command([
            str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/evaluate.py"), "package-external",
            "--dataset-manifest", str(SEALED_MANIFEST), "--model", str(onnx_model),
            "--sealed-release", str(SEALED_RELEASE), "--output-dir", str(output),
        ])
        write_completion_receipt(
            output, artifact_type="sealed_external_package",
            required_paths=["package_manifest.json"], inputs=sealed_inputs, metadata={"model": model},
        )
    sync_work_snapshot("sealed-external-package")
"""
'''
    final_protocol["source"] = _source(
        '''FINAL_PROTOCOL_CODE = r"""# Accepted-release evaluation/export/report is owned by
# scripts/colab_pipeline.py.
print("Accepted-release sonrası fazlar sürümlenmiş Python orchestrator tarafından işlendi.")
"""'''
    )
    setup_text = """# One pinned hermetic runtime; no automatic compatibility fallback.
FAILURE_REPORTER.set_stage("runtime-hermetic-install")
if not RUNTIME_REQUIRED:
    RUNTIME_PYTHON = Path(sys.executable)
    MMSEG_ROOT = PROJECT_ROOT
    print("Bu hedef CUDA/MMCV/MMSeg gerektirmiyor; ağır runtime kurulumu atlandı.")
elif LOCAL_TEST_MODE:
    RUNTIME_PYTHON = Path(sys.executable)
    MMSEG_ROOT = PROJECT_ROOT
    print("LOCAL_TEST_MODE: CUDA compatibility installation skipped.")
else:
    import shutil

    import torch

    from edgeguard.rescue.colab_performance import environment_signature

    runtime_cache_identity = environment_signature(torch)
    runtime_cache_identity["framework_config_sha256"] = sha256_file(
        PROJECT_ROOT / "configs/training/segmentation/framework_mmseg.yaml"
    )
    runtime_cache_key = sha256_payload(runtime_cache_identity)
    runtime_cache_store = DRIVE_ROOT / "EdgeGuard/runtime_cache/v2"
    runtime_cache_artifact = f"uv-cache-{runtime_cache_key[:20]}"
    runtime_cache_archive = CONTENT_ROOT / "edgeguard-runtime-cache.tar"
    runtime_cache_root = CONTENT_ROOT / "edgeguard-cache"
    runtime_cache_pointer = runtime_cache_store / "pointers" / f"{runtime_cache_artifact}.json"
    cache_was_restored = False
    if runtime_cache_pointer.is_file() and not runtime_cache_root.exists():
        run_colab_command(
            [sys.executable, str(PROJECT_ROOT / "scripts/manage_colab_recovery.py"),
             "restore", "--store-root", str(runtime_cache_store),
             "--artifact-id", runtime_cache_artifact,
             "--destination", str(runtime_cache_archive)],
        )
        run_colab_command(
            [sys.executable, str(PROJECT_ROOT / "scripts/manage_colab_recovery.py"),
             "restore-state", "--archive", str(runtime_cache_archive),
             "--destination", str(runtime_cache_root)],
        )
        cache_was_restored = True
        print("Doğrulanmış runtime wheel/source cache /content alanına geri yüklendi.")

    compatibility_evidence = CONTENT_ROOT / "edgeguard-evidence"
    compatibility_logs = CONTENT_ROOT / "edgeguard-logs"
    drive_runtime_evidence = CAMPAIGN_ROOT / "runtime-compatibility"

    def persist_compatibility_evidence() -> None:
        drive_runtime_evidence.mkdir(parents=True, exist_ok=True)
        roots = ((compatibility_evidence, "evidence"), (compatibility_logs, "logs"))
        for root, label in roots:
            if not root.is_dir():
                continue
            for source in sorted(root.rglob("*")):
                if not source.is_file() or source.stat().st_size > 5 * 1024**2:
                    continue
                if source.suffix not in {".json", ".zip", ".log"}:
                    continue
                target = drive_runtime_evidence / label / source.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    install = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train/install_semantic_stack.py"),
        "--config", str(PROJECT_ROOT / "configs/training/segmentation/framework_mmseg.yaml"),
        "--project-root", str(PROJECT_ROOT),
        "--project-commit", PROJECT_COMMIT,
        "--config-root", str(PROJECT_ROOT / "configs/training/segmentation"),
        "--runtime-root", str(CONTENT_ROOT / "edgeguard-runtime"),
        "--checkout-root", str(CONTENT_ROOT / "edgeguard-checkouts"),
        "--evidence-root", str(CONTENT_ROOT / "edgeguard-evidence"),
        "--log-root", str(CONTENT_ROOT / "edgeguard-logs"),
        "--cache-root", str(CONTENT_ROOT / "edgeguard-cache"),
        "--data-root", str(CITYSCAPES_ROOT),
        "--execute",
    ]
    try:
        run_colab_command(install)
    except BaseException:
        persist_compatibility_evidence()
        raise
    if not cache_was_restored and (runtime_cache_root / "uv").is_dir():
        run_colab_command(
            [sys.executable, str(PROJECT_ROOT / "scripts/manage_colab_recovery.py"),
             "pack-state", "--work-root", str(runtime_cache_root),
             "--output", str(runtime_cache_archive), "--include", "uv", "--uncompressed"],
        )
        run_colab_command(
            [sys.executable, str(PROJECT_ROOT / "scripts/manage_colab_recovery.py"),
             "publish", "--source", str(runtime_cache_archive),
             "--store-root", str(runtime_cache_store),
             "--artifact-id", runtime_cache_artifact,
             "--campaign-id", CAMPAIGN_ID, "--project-commit", PROJECT_COMMIT,
             "--metadata-json", json.dumps(runtime_cache_identity)],
        )
    runtime_report = CONTENT_ROOT / "edgeguard-evidence/resolved-runtime.json"
    run_colab_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/resolve_colab_runtime.py"),
            "--receipt", str(CONTENT_ROOT / "edgeguard-evidence/runtime_receipt.json"),
            "--project-commit", PROJECT_COMMIT,
            "--output", str(runtime_report),
        ],
    )
    runtime = json.loads(runtime_report.read_text())
    RUNTIME_PYTHON = Path(runtime["interpreter"])
    MMSEG_ROOT = Path(runtime["mmseg_root"])
    persist_compatibility_evidence()
    work_runtime_evidence = WORK_ROOT / "reports/runtime-compatibility"
    work_runtime_evidence.mkdir(parents=True, exist_ok=True)
    for source in (compatibility_evidence / "runtime_receipt.json", runtime_report):
        shutil.copy2(source, work_runtime_evidence / source.name)
    training_profile = WORK_ROOT / "reports/runtime-compatibility/training-profile.json"
    run_colab_command(
        [str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/resolve_training_profile.py"),
         "--output", str(training_profile)],
    )
"""
    setup_stack["source"] = _source(setup_text)
    audit_text = "".join(audit["source"])
    if audit_text.startswith('FAILURE_REPORTER.set_stage("dataset-audit-and-freeze")'):
        audit_text = audit_text.split("\n", 1)[1]
    if audit_text.startswith("if LOCAL_TEST_MODE:"):
        audit_text = audit_text.split("\nelse:\n", 1)[1]
        audit_text = "\n".join(
            line[4:] if line.startswith("    ") else line for line in audit_text.splitlines()
        )
    audit_text = audit_text.replace(
        "if RUN_MULTIDOMAIN_AUDIT:\n",
        "if RUN_MULTIDOMAIN_AUDIT:\n",
    )
    audit_text = audit_text.replace(
        '                    str(CITYSCAPES_ROOT),\n                    "--output-root",',
        '                    str(DATASET_ROOTS[dataset]),\n                    "--output-root",',
    )
    audit_text = (
        "if LOCAL_TEST_MODE:\n"
        "    AUDIT_ROOT = WORK_ROOT / 'audit/cityscapes'\n"
        "    MANIFEST_ROOT = WORK_ROOT / 'manifests'\n"
        "    DATA_MANIFESTS = [MANIFEST_ROOT / f'{dataset}.frozen.json' for dataset in "
        "SCIENTIFIC_SOURCE_DATASETS]\n"
        "    STATS_ROOT = WORK_ROOT / 'multidomain-statistics'\n"
        "    WEIGHTS = STATS_ROOT / 'class_weights.json'\n"
        "    RARE = STATS_ROOT / 'rare_classes.json'\n"
        "    print('LOCAL_TEST_MODE: real dataset audit skipped.')\n"
        "else:\n" + "\n".join(f"    {line}" for line in audit_text.splitlines())
    )
    audit_text = 'FAILURE_REPORTER.set_stage("dataset-audit-and-freeze")\n' + audit_text
    audit_text = audit_text.replace(
        'for dataset, root in (("bdd100k", BDD100K_ROOT), ("idd20k", IDD20K_ROOT)):',
        "provisional_audits = PROVISIONAL_ENGINEERING_DATASETS if STAGE_PROVISIONAL_BDD else []\n        for dataset in [*provisional_audits, *SECONDARY_SCIENTIFIC_DATASETS]:\n            root = DATASET_ROOTS[dataset]",
        1,
    )
    audit_text = audit_text.replace(
        'candidates = {\n            "cityscapes": AUDIT_ROOT / "dataset_audit/dataset_manifest.candidate.json",\n            "bdd100k": WORK_ROOT / "audit/bdd100k/bdd100k_audit/dataset_manifest.candidate.json",\n            "idd20k": WORK_ROOT / "audit/idd20k/idd20k_audit/dataset_manifest.candidate.json",\n        }',
        'candidates = {"cityscapes": AUDIT_ROOT / "dataset_audit/dataset_manifest.candidate.json"}\n        for dataset in SECONDARY_SCIENTIFIC_DATASETS:\n            candidates[dataset] = WORK_ROOT / "audit" / dataset / f"{dataset}_audit/dataset_manifest.candidate.json"',
    )
    audit_text = audit_text.replace(
        'for dataset in ("cityscapes", "bdd100k", "idd20k")',
        "for dataset in SCIENTIFIC_SOURCE_DATASETS",
    )
    audit_text = audit_text.replace(
        'for dataset, root in (("bdd100k", BDD100K_ROOT), ("idd20k", IDD20K_ROOT)):',
        "for dataset in SECONDARY_SCIENTIFIC_DATASETS:\n            root = DATASET_ROOTS[dataset]",
    )
    audit_text = audit_text.replace(
        'for dataset in ("bdd100k", "idd20k"):',
        "for dataset in SECONDARY_SCIENTIFIC_DATASETS:",
    )
    audit_text = audit_text.replace(
        "for dataset in [*PROVISIONAL_ENGINEERING_DATASETS, *SECONDARY_SCIENTIFIC_DATASETS]:",
        "provisional_audits = PROVISIONAL_ENGINEERING_DATASETS if STAGE_PROVISIONAL_BDD else []\n        for dataset in [*provisional_audits, *SECONDARY_SCIENTIFIC_DATASETS]:",
    )
    audit_text = audit_text.replace(
        'if not (AUDIT_ROOT / "dataset_audit/summary.json").is_file():',
        'cityscapes_audit = AUDIT_ROOT / "dataset_audit"\n    if not completion_is_valid(cityscapes_audit):\n        quarantine_incomplete(cityscapes_audit)',
    )
    audit_text = audit_text.replace(
        "            check=True,\n        )\n\n    audit = json.loads",
        '            check=True,\n        )\n        write_completion_receipt(\n            cityscapes_audit, artifact_type="cityscapes_audit",\n            required_paths=["summary.json", "dataset_manifest.candidate.json", "CSF-SPLIT-D.json"],\n            metadata={"dataset": "cityscapes"},\n        )\n\n    audit = json.loads',
        1,
    )
    audit_text = audit_text.replace(
        '            if not (destination / f"{dataset}_audit/summary.json").is_file():',
        '            report_root = destination / f"{dataset}_audit"\n            if not completion_is_valid(report_root):\n                quarantine_incomplete(report_root)',
    )
    audit_text = audit_text.replace(
        "                    check=True,\n                )\n\n    if RUN_FREEZE:",
        '                    check=True,\n                )\n                write_completion_receipt(\n                    report_root, artifact_type="source_domain_audit",\n                    required_paths=["summary.json", "dataset_manifest.candidate.json"],\n                    metadata={"dataset": dataset},\n                )\n        sync_work_snapshot("dataset-audit")\n\n    if RUN_FREEZE:',
        1,
    )
    audit_text = audit_text.replace("subprocess.run(", "run_colab_command(")
    audit_text = """FAILURE_REPORTER.set_stage("dataset-audit-and-freeze")
AUDIT_ROOT = WORK_ROOT / "audit/cityscapes"
MANIFEST_ROOT = WORK_ROOT / "manifests"
DATA_MANIFESTS = [MANIFEST_ROOT / f"{dataset}.frozen.json" for dataset in SCIENTIFIC_SOURCE_DATASETS]
STATS_ROOT = WORK_ROOT / "multidomain-statistics"
WEIGHTS = STATS_ROOT / "class_weights.json"
RARE = STATS_ROOT / "rare_classes.json"

if LOCAL_TEST_MODE:
    print("LOCAL_TEST_MODE: real dataset audit skipped.")
elif "audit" not in PLANNED_STAGES:
    print("Review hedefi: dataset staging ve audit atlandı.")
else:
    def preparation_identity(dataset_root):
        receipt = dataset_root / "preparation_receipt.json"
        return sha256_file(receipt) if receipt.is_file() else "legacy-without-receipt"

    cityscapes_audit = AUDIT_ROOT / "dataset_audit"
    city_inputs = {"preparation": preparation_identity(CITYSCAPES_ROOT)}
    if not completion_is_valid(cityscapes_audit, expected_inputs=city_inputs):
        quarantine_incomplete(cityscapes_audit)
        run_colab_command([
            str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/audit_dataset.py"),
            "--dataset-root", str(CITYSCAPES_ROOT), "--output-root", str(AUDIT_ROOT),
        ])
        write_completion_receipt(
            cityscapes_audit, artifact_type="cityscapes_audit",
            required_paths=["summary.json", "dataset_manifest.candidate.json", "CSF-SPLIT-D.json"],
            inputs=city_inputs, metadata={"dataset": "cityscapes"},
        )
        sync_work_snapshot("audit-cityscapes")
    audit = json.loads((cityscapes_audit / "summary.json").read_text())
    if not audit["audit_passed"]:
        raise RuntimeError("Cityscapes audit failed; scientific training is blocked")

    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    if RUN_MULTIDOMAIN_AUDIT:
        source_manifest_candidates = {
            "cityscapes": cityscapes_audit / "dataset_manifest.candidate.json"
        }
        provisional = PROVISIONAL_ENGINEERING_DATASETS if STAGE_PROVISIONAL_BDD else []
        for dataset in [*provisional, *SECONDARY_SCIENTIFIC_DATASETS]:
            dataset_root = DATASET_ROOTS[dataset]
            destination = WORK_ROOT / "audit" / dataset
            report_root = destination / f"{dataset}_audit"
            audit_inputs = {
                "preparation": preparation_identity(dataset_root),
                "ontology": sha256_file(PROJECT_ROOT / "configs/dataset/semantic_ontology_v2.yaml"),
            }
            audit_inputs.update({
                f"source_manifest:{source_dataset}": sha256_file(source_manifest)
                for source_dataset, source_manifest in source_manifest_candidates.items()
            })
            if completion_is_valid(report_root, expected_inputs=audit_inputs):
                if dataset in SECONDARY_SCIENTIFIC_DATASETS:
                    source_manifest_candidates[dataset] = (
                        report_root / "dataset_manifest.candidate.json"
                    )
                continue
            quarantine_incomplete(report_root)
            command = [
                str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/audit_dataset.py"),
                "--dataset", dataset, "--dataset-root", str(dataset_root),
                "--output-root", str(destination),
            ]
            for source_manifest in source_manifest_candidates.values():
                command.extend(["--source-manifest", str(source_manifest)])
            if dataset == "idd20k":
                command.extend([
                    "--checkpoint-root",
                    str(CAMPAIGN_ROOT / "state/audit-catalog"),
                    "--quarantine-invalid-source-samples",
                ])
            audit_process = run_colab_command(command, check=False)
            if audit_process.returncode not in {0, 2}:
                raise RuntimeError(
                    f"{dataset} audit infrastructure failed with exit code "
                    f"{audit_process.returncode}"
                )
            if audit_process.returncode == 2:
                review_root = CAMPAIGN_ROOT / "reports/data-review" / dataset
                review_root.mkdir(parents=True, exist_ok=True)
                for name in ("summary.json", "invalid_samples.json", "dataset_manifest.candidate.json"):
                    source = report_root / name
                    if source.is_file():
                        shutil.copy2(source, review_root / name)
                review_status = {
                    "status": "data_review_required",
                    "dataset": dataset,
                    "report_root": str(review_root),
                    "message": (
                        "Audit found a fail-closed data contract violation. Training was not "
                        "started; review artifacts are persistent on Drive."
                    ),
                }
                (review_root / "status.json").write_text(
                    canonical_json(review_status) + "\\n", encoding="utf-8"
                )
                sync_work_snapshot(f"audit-review-{dataset}")
                raise RuntimeError(canonical_json(review_status))
            write_completion_receipt(
                report_root, artifact_type="source_domain_audit",
                required_paths=[
                    "summary.json", "invalid_samples.json", "dataset_manifest.candidate.json"
                ],
                inputs=audit_inputs, metadata={"dataset": dataset},
            )
            if dataset in SECONDARY_SCIENTIFIC_DATASETS:
                source_manifest_candidates[dataset] = (
                    report_root / "dataset_manifest.candidate.json"
                )
            sync_work_snapshot(f"audit-{dataset}")

    if RUN_FREEZE:
        candidates = {"cityscapes": cityscapes_audit / "dataset_manifest.candidate.json"}
        for dataset in SECONDARY_SCIENTIFIC_DATASETS:
            candidates[dataset] = (
                WORK_ROOT / "audit" / dataset / f"{dataset}_audit/dataset_manifest.candidate.json"
            )
        for dataset, candidate in candidates.items():
            if not candidate.is_file():
                raise RuntimeError(f"Missing reviewed audit candidate for {dataset}")
            review_receipt = MANIFEST_REVIEW_RECEIPT_ROOT / f"{dataset}.review.json"
            if not review_receipt.is_file():
                raise PermissionError(
                    "Manifest freeze is closed until a human review receipt exists: "
                    f"{review_receipt}"
                )
            frozen = MANIFEST_ROOT / f"{dataset}.frozen.json"
            reuse = False
            if frozen.is_file():
                frozen_payload = json.loads(frozen.read_text())
                reuse = (
                    frozen_payload.get("approved_candidate_sha256") == sha256_file(candidate)
                    and frozen_payload.get("human_review_receipt_sha256")
                    == sha256_file(review_receipt)
                    and frozen_payload.get("project_commit") == PROJECT_COMMIT
                    and frozen_payload.get("campaign_id") == CAMPAIGN_ID
                )
            if not reuse:
                if frozen.exists():
                    quarantine = frozen.with_name(f"{frozen.name}.stale-{PROJECT_COMMIT[:12]}")
                    frozen.replace(quarantine)
                run_colab_command([
                    str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/audit_dataset.py"),
                    "--dataset", dataset, "--output-root", str(MANIFEST_ROOT),
                    "--split-manifest", str(candidate), "--freeze-approved",
                    "--review-receipt", str(review_receipt),
                    "--campaign-id", CAMPAIGN_ID, "--project-commit", PROJECT_COMMIT,
                ])
        sync_work_snapshot("frozen-source-manifests")

    if RUN_SOURCE_VALIDATION_AUDIT:
        if not all(path.is_file() for path in DATA_MANIFESTS):
            raise RuntimeError("Official source validation requires all frozen source manifests")
        completed_final_runs = [
            path for path in (WORK_ROOT / "runs/final").glob("*/ce")
            if completion_is_valid(path)
        ]
        if len(completed_final_runs) < 3:
            raise RuntimeError(
                "Official validation remains sealed until CAMPAIGN_TARGET='final' completes"
            )
        manifest_set = sha256_payload(sorted(sha256_file(path) for path in DATA_MANIFESTS))
        for dataset in OFFICIAL_VALIDATION_DATASETS:
            destination = WORK_ROOT / "audit" / f"{dataset}-val"
            report_root = (
                destination / "dataset_audit"
                if dataset == "cityscapes"
                else destination / f"{dataset}_val_audit"
            )
            validation_inputs = {
                "training_manifests": manifest_set,
                "preparation": preparation_identity(DATASET_ROOTS[dataset]),
            }
            if completion_is_valid(report_root, expected_inputs=validation_inputs):
                continue
            quarantine_incomplete(report_root)
            command = [
                str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/audit_dataset.py"),
                "--dataset", dataset, "--source-split", "val",
                "--dataset-root", str(DATASET_ROOTS[dataset]),
                "--output-root", str(destination),
            ]
            if dataset != "cityscapes":
                command.extend([
                    "--checkpoint-root", str(CAMPAIGN_ROOT / "state/audit-catalog")
                ])
            for manifest in DATA_MANIFESTS:
                command.extend(["--source-manifest", str(manifest)])
            run_colab_command(command)
            write_completion_receipt(
                report_root, artifact_type="official_source_validation_audit",
                required_paths=["summary.json", "dataset_manifest.candidate.json"],
                inputs=validation_inputs, metadata={"dataset": dataset},
            )
            sync_work_snapshot(f"official-validation-audit-{dataset}")

    if RUN_FREEZE_SOURCE_VALIDATION:
        validation_root = MANIFEST_ROOT / "official-validation"
        for dataset in OFFICIAL_VALIDATION_DATASETS:
            report_name = "dataset_audit" if dataset == "cityscapes" else f"{dataset}_val_audit"
            candidate = WORK_ROOT / "audit" / f"{dataset}-val" / report_name / "dataset_manifest.candidate.json"
            frozen = validation_root / f"{dataset}.frozen.json"
            review_receipt = MANIFEST_REVIEW_RECEIPT_ROOT / f"{dataset}-official-val.review.json"
            if not candidate.is_file() or not review_receipt.is_file():
                raise PermissionError(
                    "Official validation freeze requires candidate and review receipt for "
                    f"{dataset}"
                )
            reuse = False
            if frozen.is_file():
                frozen_payload = json.loads(frozen.read_text())
                reuse = (
                    frozen_payload.get("approved_candidate_sha256") == sha256_file(candidate)
                    and frozen_payload.get("human_review_receipt_sha256")
                    == sha256_file(review_receipt)
                    and frozen_payload.get("project_commit") == PROJECT_COMMIT
                )
            if not reuse:
                if frozen.exists():
                    quarantine = frozen.with_name(f"{frozen.name}.stale-{PROJECT_COMMIT[:12]}")
                    frozen.replace(quarantine)
                run_colab_command([
                    str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/audit_dataset.py"),
                    "--dataset", dataset, "--output-root", str(validation_root),
                    "--split-manifest", str(candidate), "--freeze-approved",
                    "--review-receipt", str(review_receipt),
                    "--campaign-id", CAMPAIGN_ID, "--project-commit", PROJECT_COMMIT,
                ])
        sync_work_snapshot("frozen-official-validation-manifests")

    if all(path.is_file() for path in DATA_MANIFESTS):
        stats_inputs = {
            "dataset_manifest_set": sha256_payload(
                sorted(sha256_file(path) for path in DATA_MANIFESTS)
            )
        }
        if not completion_is_valid(STATS_ROOT, expected_inputs=stats_inputs):
            quarantine_incomplete(STATS_ROOT)
            command = [
                str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/audit_dataset.py"),
                "--output-root", str(STATS_ROOT),
            ]
            for manifest in DATA_MANIFESTS:
                command.extend(["--data-manifest", str(manifest)])
            run_colab_command(command)
            write_completion_receipt(
                STATS_ROOT, artifact_type="multidomain_statistics",
                required_paths=["class_weights.json", "rare_classes.json", "summary.json"],
                inputs=stats_inputs,
            )
            sync_work_snapshot("multidomain-statistics")
    print(json.dumps(audit, indent=2))
"""
    audit["source"] = _source(audit_text)

    training_text = """FAILURE_REPORTER.set_stage("resumable-training")
RUN_ROOT = WORK_ROOT / "runs"
RESOURCE_OVERRIDE_ROOT = WORK_ROOT / "reports/resource-overrides"


def resolved_device_batch(stage, model):
    path = RESOURCE_OVERRIDE_ROOT / f"{stage}-{model}.json"
    if path.is_file():
        return int(json.loads(path.read_text())["device_batch"])
    return int(RESOURCE_PROFILE["device_batch"])


def run_training_with_oom_retry(command, *, stage, model, run_dir):
    device_index = command.index("--device-batch") + 1
    current_batch = int(command[device_index])
    try:
        run_colab_command(command)
        return current_batch
    except RuntimeError as error:
        message = str(error).lower()
        if "out of memory" not in message or current_batch <= 1:
            raise
        fallback = current_batch // 2
        if 4 % fallback:
            fallback = 1
        quarantine_incomplete(run_dir)
        retry_command = [value for value in command if value != "--resume"]
        retry_index = retry_command.index("--device-batch") + 1
        retry_command[retry_index] = str(fallback)
        RESOURCE_OVERRIDE_ROOT.mkdir(parents=True, exist_ok=True)
        override = RESOURCE_OVERRIDE_ROOT / f"{stage}-{model}.json"
        override.write_text(
            json.dumps({
                "stage": stage, "model": model, "device_batch": fallback,
                "gradient_accumulation": 4 // fallback,
                "effective_batch": 4, "reason": "single_automatic_cuda_oom_retry",
            }, sort_keys=True) + "\\n",
            encoding="utf-8",
        )
        print(f"CUDA OOM: device batch {current_batch} -> {fallback}; effective batch remains 4.")
        run_colab_command(retry_command)
        return fallback

if TRAINING_STAGES:
    if not all(path.is_file() for path in DATA_MANIFESTS):
        raise RuntimeError("All reviewed/frozen scientific source manifests are required")
    profile_path = WORK_ROOT / "reports/runtime-compatibility/training-profile.json"
    if not profile_path.is_file():
        raise RuntimeError("Training target requires a measured runtime training profile")
    RESOURCE_PROFILE = json.loads(profile_path.read_text())["training"]
    for stage in [item for item in TRAINING_STAGES if item != "final"]:
        stage_models = RUN_MODELS
        for model in stage_models:
            FAILURE_REPORTER.set_stage(f"training-{stage}-{model}")
            run_dir = RUN_ROOT / stage / model / "ce"
            training_precision = "fp32" if stage == "smoke" else RESOURCE_PROFILE["precision"]
            training_device_batch = resolved_device_batch(stage, model)
            training_inputs = {
                "project_commit": PROJECT_COMMIT,
                "protocol_sha256": sha256_file(PROJECT_ROOT / "configs/rescue/semantic_first.yaml"),
                "dataset_manifest_set_sha256": sha256_payload(
                    sorted(sha256_file(path) for path in DATA_MANIFESTS)
                ),
                "resource_profile_sha256": sha256_payload(RESOURCE_PROFILE),
                "precision": training_precision,
                "device_batch": str(training_device_batch),
            }
            if completion_is_valid(run_dir, expected_inputs=training_inputs):
                print("Tamamlanmış eğitim atlandı:", stage, model)
                continue
            quarantine_incomplete(run_dir)
            command = [
                str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/train.py"),
                "--config", str(PROJECT_ROOT / "configs/rescue/semantic_first.yaml"),
                "--model", model, "--stage", stage, "--output-root", str(RUN_ROOT),
                "--mmseg-root", str(MMSEG_ROOT), "--loss", "ce",
                "--device-batch", str(training_device_batch),
                "--workers", str(RESOURCE_PROFILE["workers"]),
                "--precision", training_precision,
                "--recovery-root", str(RECOVERY_ROOT),
                "--campaign-id", CAMPAIGN_ID, "--project-commit", PROJECT_COMMIT,
            ]
            for manifest in DATA_MANIFESTS:
                command.extend(["--data-manifest", str(manifest)])
            recovery_pointer = RECOVERY_ROOT / "pointers" / f"{stage}-{model.replace('_', '-')}-ce.json"
            if AUTO_RESUME and ((run_dir / "run_identity.json").is_file() or recovery_pointer.is_file()):
                command.append("--resume")
            used_device_batch = run_training_with_oom_retry(
                command, stage=stage, model=model, run_dir=run_dir
            )
            training_inputs["device_batch"] = str(used_device_batch)
            write_completion_receipt(
                run_dir,
                artifact_type="semantic_training",
                required_paths=["run_identity.json", "resolved.py", "summary.json"],
                inputs=training_inputs,
                metadata={"stage": stage, "model": model, "resource_profile": RESOURCE_PROFILE},
            )
            sync_work_snapshot(f"training-{stage}-{model}")
else:
    print("Bu hedef GPU eğitimi gerektirmiyor.")
"""
    training_text = """FAILURE_REPORTER.set_stage("resumable-python-orchestrator")
RUN_ROOT = WORK_ROOT / "runs"


def ensure_checkpoint(stage, model):
    return latest_checkpoint(RUN_ROOT / stage / model / "ce")


PIPELINE_TARGETS = {"smoke", "pilot", "screening", "hpo", "final", "evaluate", "export", "report"}
if LOCAL_TEST_MODE:
    print("LOCAL_TEST_MODE: hermetic GPU pipeline execution skipped.")
elif CAMPAIGN_TARGET in PIPELINE_TARGETS:
    if RUN_ACCEPT_RELEASE:
        if ACCEPTED_RELEASE.is_file():
            print("Existing accepted release will be hash-verified by the orchestrator.")
        elif not RELEASE_CANDIDATE.is_file() or not RELEASE_REVIEW_RECEIPT.is_file():
            raise PermissionError("Release acceptance requires candidate and human review receipt")
        else:
            run_colab_command([
                str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/accept_colab_release.py"),
                "--candidate", str(RELEASE_CANDIDATE),
                "--review-receipt", str(RELEASE_REVIEW_RECEIPT),
                "--output", str(ACCEPTED_RELEASE),
            ])
    if not all(path.is_file() for path in DATA_MANIFESTS):
        raise RuntimeError("All reviewed/frozen scientific source manifests are required")
    pipeline_command = [
        str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/colab_pipeline.py"), "run",
        "--target", CAMPAIGN_TARGET,
        "--project-root", str(PROJECT_ROOT), "--project-commit", PROJECT_COMMIT,
        "--runtime-receipt", str(CONTENT_ROOT / "edgeguard-evidence/runtime_receipt.json"),
        "--mmseg-root", str(MMSEG_ROOT), "--work-root", str(WORK_ROOT),
        "--recovery-root", str(RECOVERY_ROOT),
        "--config", str(PROJECT_ROOT / "configs/rescue/semantic_first.yaml"),
        "--rare-classes-file", str(RARE), "--class-weights-file", str(WEIGHTS),
    ]
    for manifest in DATA_MANIFESTS:
        pipeline_command.extend(["--data-manifest", str(manifest)])
    candidate_table = WORK_ROOT / "reports/screening/candidate_table.json"
    if CAMPAIGN_TARGET in {"hpo", "final", "evaluate", "export", "report"}:
        if not candidate_table.is_file():
            raise RuntimeError("HPO/final target requires the reviewed screening candidate table")
        pipeline_command.extend(["--candidate-table", str(candidate_table)])
    if CAMPAIGN_TARGET in {"final", "evaluate", "export", "report"}:
        if len(FINAL_MODELS) != 3:
            raise RuntimeError("Final and accepted-release targets require three frozen finalists")
        for model in FINAL_MODELS:
            pipeline_command.extend(["--final-model", model])
        pipeline_command.extend(["--ablation-model", FINAL_MODELS[0]])
    if CAMPAIGN_TARGET in {"evaluate", "export", "report"}:
        if not ACCEPTED_RELEASE.is_file():
            raise PermissionError("Accepted-release targets require ACCEPTED_RELEASE")
        pipeline_command.extend(["--accepted-release", str(ACCEPTED_RELEASE)])
        validation_root = WORK_ROOT / "manifests/official-validation"
        for dataset in OFFICIAL_VALIDATION_DATASETS:
            manifest = validation_root / f"{dataset}.frozen.json"
            if not manifest.is_file():
                raise PermissionError(
                    f"Accepted-release targets require frozen official validation: {manifest}"
                )
            pipeline_command.extend(["--evaluation-manifest", str(manifest)])
    run_colab_command(pipeline_command)
    sync_work_snapshot(f"pipeline-{CAMPAIGN_TARGET}")
else:
    print("Bu hedef GPU eğitim orchestrator'ını gerektirmiyor.")
"""
    training["source"] = _source(training_text)

    evaluation_text = """FAILURE_REPORTER.set_stage("resumable-evaluation-export-hpo")

def ensure_checkpoint(stage, model):
    run_dir = RUN_ROOT / stage / model / "ce"
    try:
        return latest_checkpoint(run_dir)
    except FileNotFoundError:
        destination = run_dir / "recovered.pth"
        artifact_id = f"{stage}-{model.replace('_', '-')}-ce"
        run_colab_command(
            [sys.executable, str(PROJECT_ROOT / "scripts/manage_colab_recovery.py"),
             "restore", "--store-root", str(RECOVERY_ROOT), "--artifact-id", artifact_id,
             "--destination", str(destination)],
        )
        (run_dir / "last_checkpoint").write_text("recovered.pth\\n", encoding="utf-8")
        return destination


for stage in TRAINING_STAGES:
    stage_models = FINAL_MODELS if stage == "final" else RUN_MODELS
    for model in stage_models:
        run_dir = RUN_ROOT / stage / model / "ce"
        if not completion_is_valid(run_dir):
            continue
        checkpoint = ensure_checkpoint(stage, model)
        for dataset, manifest in zip(SCIENTIFIC_SOURCE_DATASETS, DATA_MANIFESTS, strict=True):
            target = WORK_ROOT / "evaluation" / stage / model / dataset
            evaluation_inputs = {
                "checkpoint_sha256": sha256_file(checkpoint),
                "dataset_manifest_sha256": sha256_file(manifest),
            }
            if completion_is_valid(target, expected_inputs=evaluation_inputs):
                continue
            quarantine_incomplete(target)
            run_colab_command(
                [str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/evaluate.py"), "run",
                 "--resolved-config", str(run_dir / "resolved.py"),
                 "--checkpoint", str(checkpoint), "--dataset", dataset,
                 "--dataset-manifest", str(manifest), "--role", "train_select",
                 "--rare-classes-file", str(RARE), "--output-dir", str(target)],
            )
            write_completion_receipt(
                target, artifact_type="semantic_evaluation", required_paths=["evaluation.json"],
                inputs=evaluation_inputs,
                metadata={"stage": stage, "model": model, "dataset": dataset},
            )

        if stage not in {"pilot", "screening", "final"}:
            continue
        export_dir = WORK_ROOT / "exports" / stage / model
        onnx_path = export_dir / f"{model}.onnx"
        artifact_id = f"onnx-{stage}-{model.replace('_', '-')}"
        if not onnx_path.is_file() and (RECOVERY_ROOT / "pointers" / f"{artifact_id}.json").is_file():
            run_colab_command(
                [sys.executable, str(PROJECT_ROOT / "scripts/manage_colab_recovery.py"),
                 "restore", "--store-root", str(RECOVERY_ROOT), "--artifact-id", artifact_id,
                 "--destination", str(onnx_path)],
            )
        export_inputs = {"checkpoint_sha256": sha256_file(checkpoint)}
        if not completion_is_valid(export_dir, expected_inputs=export_inputs):
            quarantine_incomplete(export_dir)
            run_colab_command(
                [str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/export_onnx.py"),
                 "--resolved-config", str(run_dir / "resolved.py"),
                 "--checkpoint", str(checkpoint), "--output", str(onnx_path),
                 "--device", "cuda", "--warmup", "5", "--iterations", "20"],
            )
            run_colab_command(
                [sys.executable, str(PROJECT_ROOT / "scripts/manage_colab_recovery.py"),
                 "publish", "--source", str(onnx_path), "--store-root", str(RECOVERY_ROOT),
                 "--artifact-id", artifact_id, "--campaign-id", CAMPAIGN_ID,
                 "--project-commit", PROJECT_COMMIT,
                 "--metadata-json", json.dumps({"stage": stage, "model": model})],
            )
            write_completion_receipt(
                export_dir, artifact_type="onnx_export",
                required_paths=[f"{model}.onnx", f"{model}.validation.json"],
                inputs=export_inputs,
                metadata={"stage": stage, "model": model, "benchmark": "screening_5_20"},
            )
    sync_work_snapshot(f"evaluation-export-{stage}")

screening_evidence = list((WORK_ROOT / "evaluation/screening").glob("**/evaluation.json"))
export_evidence = list((WORK_ROOT / "exports/screening").glob("**/*.validation.json"))
report_dir = WORK_ROOT / "reports/screening"
if len(screening_evidence) >= 2 * len(SCIENTIFIC_SOURCE_DATASETS) and len(export_evidence) >= 2:
    if not completion_is_valid(report_dir):
        quarantine_incomplete(report_dir)
        command = [
            str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/evaluate.py"), "summarize",
            "--evaluation-root", str(WORK_ROOT / "evaluation/screening"),
            "--export-root", str(WORK_ROOT / "exports/screening"),
            "--output-dir", str(report_dir),
        ]
        for dataset in SCIENTIFIC_SOURCE_DATASETS:
            command.extend(["--expected-dataset", dataset])
        run_colab_command(command)
        write_completion_receipt(
            report_dir, artifact_type="screening_report",
            required_paths=["candidate_table.json", "metrics_table.csv", "week3_summary.md"],
            metadata={"source_domains": SCIENTIFIC_SOURCE_DATASETS},
        )
        sync_work_snapshot("screening-report")

if RUN_HPO:
    candidate_table = report_dir / "candidate_table.json"
    if not completion_is_valid(report_dir):
        raise RuntimeError("HPO requires a verified screening report")
    hpo_inputs = {
        "project_commit": PROJECT_COMMIT,
        "protocol_sha256": sha256_file(PROJECT_ROOT / "configs/rescue/semantic_first.yaml"),
        "candidate_table_sha256": sha256_file(candidate_table),
        "rare_classes_sha256": sha256_file(RARE),
        "dataset_manifest_set_sha256": sha256_payload(
            sorted(sha256_file(path) for path in DATA_MANIFESTS)
        ),
        "resource_profile_sha256": sha256_payload(RESOURCE_PROFILE),
    }
    completed_hpo_roots = [
        path.parent
        for path in (RUN_ROOT / "hpo").glob("*/result.json")
        if completion_is_valid(path.parent, expected_inputs=hpo_inputs)
    ]
    if len(completed_hpo_roots) != 2:
        hpo_device_batch = min(
            resolved_device_batch("screening", model) for model in RUN_MODELS
        )
        command = [
            str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/train.py"), "--stage", "hpo",
            "--candidate-table", str(candidate_table), "--output-root", str(RUN_ROOT),
            "--mmseg-root", str(MMSEG_ROOT), "--rare-classes-file", str(RARE),
            "--device-batch", str(hpo_device_batch),
            "--workers", str(RESOURCE_PROFILE["workers"]),
            "--precision", RESOURCE_PROFILE["precision"],
            "--recovery-root", str(RECOVERY_ROOT), "--campaign-id", CAMPAIGN_ID,
            "--project-commit", PROJECT_COMMIT,
        ]
        for manifest in DATA_MANIFESTS:
            command.extend(["--data-manifest", str(manifest)])
        run_colab_command(command)
        for result_path in (RUN_ROOT / "hpo").glob("*/result.json"):
            write_completion_receipt(
                result_path.parent,
                artifact_type="hpo_study",
                required_paths=["result.json", "trials.snapshot.json", "study.sqlite3"],
                inputs=hpo_inputs,
                metadata={"model": json.loads(result_path.read_text())["model"]},
            )
        sync_work_snapshot("hpo-rung-or-study-complete")
    completed_hpo_roots = [
        path.parent
        for path in (RUN_ROOT / "hpo").glob("*/result.json")
        if completion_is_valid(path.parent, expected_inputs=hpo_inputs)
    ]
    if len(completed_hpo_roots) != 2:
        raise RuntimeError("HPO requires exactly two verified completion receipts")
    if not FINAL_MODELS:
        hpo_results = [
            json.loads((root / "result.json").read_text()) for root in completed_hpo_roots
        ]
        FINAL_MODELS = [
            row["model"]
            for row in sorted(
                hpo_results,
                key=lambda row: (-float(row["best_domain_macro_mIoU"]), row["model"]),
            )
        ]
        print("HPO ile dondurulan finalistler:", FINAL_MODELS)
"""
    evaluation_text = """FAILURE_REPORTER.set_stage("pipeline-artifact-review")
pipeline_state = WORK_ROOT / "pipeline-v2"
if pipeline_state.is_dir():
    completed = sorted(path.parent.name for path in pipeline_state.glob("*/completion.json"))
    print("Hash-doğrulanmış pipeline fazları:", completed)
else:
    print("Henüz bir v2 pipeline fazı tamamlanmadı.")
"""
    evaluation["source"] = _source(evaluation_text)

    final_training_text = """FAILURE_REPORTER.set_stage("resumable-final-training-and-export")
if "final" in TRAINING_STAGES:
    if len(FINAL_MODELS) != 2:
        raise RuntimeError("Final training requires the two completed HPO finalists")
    hpo_models = {
        json.loads(path.read_text())["model"]
        for path in (RUN_ROOT / "hpo").glob("*/result.json")
    }
    if set(FINAL_MODELS) != hpo_models:
        raise RuntimeError("FINAL_MODELS must exactly match the two frozen HPO results")
    for model in FINAL_MODELS:
        FAILURE_REPORTER.set_stage(f"training-final-{model}")
        run_dir = RUN_ROOT / "final" / model / "ce"
        hpo_result_path = RUN_ROOT / "hpo" / model / "result.json"
        hpo_result = json.loads(hpo_result_path.read_text())
        best_params = hpo_result["best_params"]
        final_device_batch = resolved_device_batch("final", model)
        final_inputs = {
            "project_commit": PROJECT_COMMIT,
            "protocol_sha256": sha256_file(PROJECT_ROOT / "configs/rescue/semantic_first.yaml"),
            "dataset_manifest_set_sha256": sha256_payload(
                sorted(sha256_file(path) for path in DATA_MANIFESTS)
            ),
            "resource_profile_sha256": sha256_payload(RESOURCE_PROFILE),
            "precision": RESOURCE_PROFILE["precision"],
            "hpo_result_sha256": sha256_file(hpo_result_path),
            "device_batch": str(final_device_batch),
        }
        if not completion_is_valid(run_dir, expected_inputs=final_inputs):
            quarantine_incomplete(run_dir)
            command = [
                str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/train.py"),
                "--config", str(PROJECT_ROOT / "configs/rescue/semantic_first.yaml"),
                "--model", model, "--stage", "final", "--output-root", str(RUN_ROOT),
                "--mmseg-root", str(MMSEG_ROOT), "--loss", "ce",
                "--device-batch", str(final_device_batch),
                "--workers", str(RESOURCE_PROFILE["workers"]),
                "--precision", RESOURCE_PROFILE["precision"],
                "--recovery-root", str(RECOVERY_ROOT),
                "--campaign-id", CAMPAIGN_ID, "--project-commit", PROJECT_COMMIT,
                "--learning-rate", str(best_params["learning_rate"]),
                "--weight-decay", str(best_params["weight_decay"]),
                "--scheduler", str(best_params["scheduler"]),
                "--warmup-ratio", str(best_params["warmup_ratio"]),
            ]
            for manifest in DATA_MANIFESTS:
                command.extend(["--data-manifest", str(manifest)])
            recovery_pointer = RECOVERY_ROOT / "pointers" / f"final-{model.replace('_', '-')}-ce.json"
            if AUTO_RESUME and recovery_pointer.is_file():
                command.append("--resume")
            used_device_batch = run_training_with_oom_retry(
                command, stage="final", model=model, run_dir=run_dir
            )
            final_inputs["device_batch"] = str(used_device_batch)
            write_completion_receipt(
                run_dir, artifact_type="semantic_final_training",
                required_paths=["run_identity.json", "resolved.py", "summary.json"],
                inputs=final_inputs, metadata={"model": model, "resource_profile": RESOURCE_PROFILE},
            )
            sync_work_snapshot(f"training-final-{model}")

        checkpoint = ensure_checkpoint("final", model)
        export_dir = WORK_ROOT / "exports/final" / model
        onnx_path = export_dir / f"{model}.onnx"
        export_inputs = {"checkpoint_sha256": sha256_file(checkpoint)}
        artifact_id = f"onnx-final-{model.replace('_', '-')}"
        if not onnx_path.is_file() and (RECOVERY_ROOT / "pointers" / f"{artifact_id}.json").is_file():
            run_colab_command([
                sys.executable, str(PROJECT_ROOT / "scripts/manage_colab_recovery.py"),
                "restore", "--store-root", str(RECOVERY_ROOT), "--artifact-id", artifact_id,
                "--destination", str(onnx_path),
            ])
        if not completion_is_valid(export_dir, expected_inputs=export_inputs):
            quarantine_incomplete(export_dir)
            run_colab_command([
                str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/export_onnx.py"),
                "--resolved-config", str(run_dir / "resolved.py"),
                "--checkpoint", str(checkpoint), "--output", str(onnx_path),
                "--device", "cuda", "--warmup", "5", "--iterations", "20",
            ])
            run_colab_command([
                sys.executable, str(PROJECT_ROOT / "scripts/manage_colab_recovery.py"),
                "publish", "--source", str(onnx_path), "--store-root", str(RECOVERY_ROOT),
                "--artifact-id", artifact_id, "--campaign-id", CAMPAIGN_ID,
                "--project-commit", PROJECT_COMMIT,
                "--metadata-json", json.dumps({"stage": "final", "model": model}),
            ])
            write_completion_receipt(
                export_dir, artifact_type="final_onnx_export",
                required_paths=[f"{model}.onnx", f"{model}.validation.json"],
                inputs=export_inputs, metadata={"model": model, "benchmark": "equivalence_5_20"},
            )
            sync_work_snapshot(f"final-onnx-{model}")

    # Class-imbalance ablation is isolated from HPO and run only on the scientific finalist.
    weighted_model = FINAL_MODELS[0]
    hpo_result_path = RUN_ROOT / "hpo" / weighted_model / "result.json"
    best_params = json.loads(hpo_result_path.read_text())["best_params"]
    weighted_run = RUN_ROOT / "final" / weighted_model / "median_frequency"
    weighted_device_batch = resolved_device_batch("final", weighted_model)
    weighted_inputs = {
        "project_commit": PROJECT_COMMIT,
        "hpo_result_sha256": sha256_file(hpo_result_path),
        "class_weights_sha256": sha256_file(WEIGHTS),
        "dataset_manifest_set_sha256": sha256_payload(
            sorted(sha256_file(path) for path in DATA_MANIFESTS)
        ),
        "resource_profile_sha256": sha256_payload(RESOURCE_PROFILE),
        "device_batch": str(weighted_device_batch),
    }
    if not completion_is_valid(weighted_run, expected_inputs=weighted_inputs):
        quarantine_incomplete(weighted_run)
        command = [
            str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/train.py"),
            "--config", str(PROJECT_ROOT / "configs/rescue/semantic_first.yaml"),
            "--model", weighted_model, "--stage", "final", "--output-root", str(RUN_ROOT),
            "--mmseg-root", str(MMSEG_ROOT), "--loss", "median_frequency",
            "--audit-report", str(WEIGHTS),
            "--device-batch", str(weighted_device_batch),
            "--workers", str(RESOURCE_PROFILE["workers"]),
            "--precision", RESOURCE_PROFILE["precision"],
            "--recovery-root", str(RECOVERY_ROOT),
            "--campaign-id", CAMPAIGN_ID, "--project-commit", PROJECT_COMMIT,
            "--learning-rate", str(best_params["learning_rate"]),
            "--weight-decay", str(best_params["weight_decay"]),
            "--scheduler", str(best_params["scheduler"]),
            "--warmup-ratio", str(best_params["warmup_ratio"]),
        ]
        for manifest in DATA_MANIFESTS:
            command.extend(["--data-manifest", str(manifest)])
        pointer = RECOVERY_ROOT / "pointers" / f"final-{weighted_model.replace('_', '-')}-median-frequency.json"
        if AUTO_RESUME and pointer.is_file():
            command.append("--resume")
        used_device_batch = run_training_with_oom_retry(
            command, stage="final", model=weighted_model, run_dir=weighted_run
        )
        weighted_inputs["device_batch"] = str(used_device_batch)
        write_completion_receipt(
            weighted_run, artifact_type="class_imbalance_ablation_training",
            required_paths=["run_identity.json", "resolved.py", "summary.json"],
            inputs=weighted_inputs,
            metadata={"model": weighted_model, "loss": "median_frequency"},
        )
        sync_work_snapshot(f"weighted-final-{weighted_model}")

    try:
        weighted_checkpoint = latest_checkpoint(weighted_run)
    except FileNotFoundError:
        weighted_checkpoint = weighted_run / "recovered.pth"
        run_colab_command([
            sys.executable, str(PROJECT_ROOT / "scripts/manage_colab_recovery.py"),
            "restore", "--store-root", str(RECOVERY_ROOT),
            "--artifact-id", f"final-{weighted_model.replace('_', '-')}-median-frequency",
            "--destination", str(weighted_checkpoint),
        ])
        (weighted_run / "last_checkpoint").write_text("recovered.pth\\n", encoding="utf-8")
    ablation_runs = {
        "ce": (RUN_ROOT / "final" / weighted_model / "ce", ensure_checkpoint("final", weighted_model)),
        "median_frequency": (weighted_run, weighted_checkpoint),
    }
    for loss_name, (ablation_run, ablation_checkpoint) in ablation_runs.items():
        for dataset, manifest in zip(SCIENTIFIC_SOURCE_DATASETS, DATA_MANIFESTS, strict=True):
            target = WORK_ROOT / "evaluation/class-imbalance" / weighted_model / loss_name / dataset
            ablation_inputs = {
                "checkpoint_sha256": sha256_file(ablation_checkpoint),
                "dataset_manifest_sha256": sha256_file(manifest),
            }
            if completion_is_valid(target, expected_inputs=ablation_inputs):
                continue
            quarantine_incomplete(target)
            run_colab_command([
                str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/evaluate.py"), "run",
                "--resolved-config", str(ablation_run / "resolved.py"),
                "--checkpoint", str(ablation_checkpoint), "--dataset", dataset,
                "--dataset-manifest", str(manifest), "--role", "train_select",
                "--rare-classes-file", str(RARE), "--output-dir", str(target),
            ])
            write_completion_receipt(
                target, artifact_type="class_imbalance_ablation_evaluation",
                required_paths=["evaluation.json", "frame_uncertainty.json"],
                inputs=ablation_inputs,
                metadata={"model": weighted_model, "loss": loss_name, "dataset": dataset},
            )
    sync_work_snapshot("class-imbalance-ablation")
"""
    final_training_text = """FAILURE_REPORTER.set_stage("final-training-owned-by-orchestrator")
if CAMPAIGN_TARGET == "final":
    print(
        "Final eğitim, HPO parametreleri ve CE/weighted-CE ablation "
        "orchestrator tarafından işlendi."
    )
"""
    final_training = _cell("code", final_training_text)

    cells = [
        _cell(
            "markdown",
            """
# EdgeGuard-Road · oturum kaybına dayanıklı bilimsel kampanya

Tek kontrol hücresinde hedefi seçip **Run all** kullanın. Yeni Colab oturumu `/content` alanını sıfırdan kurar, Drive'daki doğrulanmış küçük kampanya durumunu ve son checkpoint generation'ını geri yükler, yalnız eksik aşamaları çalıştırır. Gerçek veri olmadan bilimsel metrik üretmez.
""",
        ),
        _cell(
            "code",
            """
import json
import os
import sys
from pathlib import Path

LOCAL_TEST_MODE = os.environ.get("EDGEGUARD_NOTEBOOK_LOCAL_TEST") == "1"
if LOCAL_TEST_MODE:
    PROJECT_ROOT = Path(os.environ.get("EDGEGUARD_PROJECT_ROOT", Path.cwd())).resolve()
    DRIVE_ROOT = Path(os.environ["EDGEGUARD_TEST_DRIVE_ROOT"]).resolve()
    CONTENT_ROOT = Path(os.environ["EDGEGUARD_TEST_CONTENT_ROOT"]).resolve()
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    CONTENT_ROOT.mkdir(parents=True, exist_ok=True)
else:
    from google.colab import drive

    drive.mount("/content/drive")
    PROJECT_ROOT = Path("/content/edgeguard-road")
    DRIVE_ROOT = Path("/content/drive/MyDrive")
    CONTENT_ROOT = Path("/content")

REPOSITORY = "https://github.com/emrealmaoglu/edgeguard-road.git"
BRANCH = "stabilize/colab-v2"
EXPECTED_PROJECT_COMMIT = "5cc578cb9f15aa7a560108840f3055ae2f4e4733"
LOCAL_DATA_ROOT = CONTENT_ROOT / "edgeguard-data"
CITYSCAPES_ROOT = LOCAL_DATA_ROOT / "cityscapes"
BDD100K_ROOT = LOCAL_DATA_ROOT / "bdd100k"
IDD20K_ROOT = LOCAL_DATA_ROOT / "idd20k"
ACDC_ROOT = LOCAL_DATA_ROOT / "acdc"
DATASET_ROOTS = {
    "cityscapes": CITYSCAPES_ROOT,
    "bdd100k": BDD100K_ROOT,
    "idd20k": IDD20K_ROOT,
    "acdc": ACDC_ROOT,
}
SCIENTIFIC_SOURCE_DATASETS = ["cityscapes", "idd20k"]
SECONDARY_SCIENTIFIC_DATASETS = [dataset for dataset in SCIENTIFIC_SOURCE_DATASETS if dataset != "cityscapes"]
OFFICIAL_VALIDATION_DATASETS = list(SCIENTIFIC_SOURCE_DATASETS)
PROVISIONAL_ENGINEERING_DATASETS = ["bdd100k"]
STAGE_PROVISIONAL_BDD = False  # Yalnız açık mühendislik audit'i için True.
OPTIONAL_EVALUATION_DATASETS = []  # Model freeze sonrası ör. ["acdc"]
WORK_ROOT = CONTENT_ROOT / "edgeguard-work"
CAMPAIGN_ID = "semantic-cs-idd-v2"
CAMPAIGN_TARGET = "audit"  # audit|smoke|pilot|screening|hpo|final|evaluate|export|report
RUN_STAGE = CAMPAIGN_TARGET  # Legacy report naming; execution uses TRAINING_STAGES.
AUTO_RESUME = True
DEEP_VERIFY_ARCHIVES = False
ALLOW_FINAL_DATA = False
CORE_MODELS = ["segformer_b0", "fast_scnn", "pidnet_s"]
EXTENSION_MODELS = ["ddrnet_23_slim", "bisenetv2"]
RUN_MODELS = CORE_MODELS if CAMPAIGN_TARGET in {"smoke", "pilot"} else CORE_MODELS + EXTENSION_MODELS
ALLOW_INELIGIBLE_BDD_SMOKE = True  # Provisional audit için; DATA_MANIFESTS listesine girmez.
RUN_DATA_STAGING = not LOCAL_TEST_MODE
RUN_MULTIDOMAIN_AUDIT = not LOCAL_TEST_MODE
RUN_FREEZE = False  # Cityscapes + IDD candidate manifestleri incelendikten sonra True.
RUN_SOURCE_VALIDATION_AUDIT = False
RUN_FREEZE_SOURCE_VALIDATION = False
MANIFEST_REVIEW_RECEIPT_ROOT = WORK_ROOT / "reviews/manifest-freeze"
RELEASE_CANDIDATE = WORK_ROOT / "accepted_release.candidate.json"
RELEASE_REVIEW_RECEIPT = WORK_ROOT / "reviews/release.review.json"
ACCEPTED_RELEASE = WORK_ROOT / "accepted_release.json"
RUN_ACCEPT_RELEASE = False  # İnsan review receipt'i hazırlandıktan sonra açıkça True.
FINAL_MODELS = []
RUN_ACDC = False
RUN_SHIFT_CALIBRATION = False
RUN_PERCEPTION_PREVIEW = False
PREVIEW_IMAGE = Path("/content/preview.png")
CREATE_REVIEW_PACKAGE = True
DOWNLOAD_REVIEW_PACKAGE = False  # True: küçük rapor/şekil ZIP'i tarayıcıya indirir.
DOWNLOAD_LATEST_FAILURE_REPORT = False  # Hata sonrası son hücreyi bununla yeniden çalıştırın.
RUN_SEALED_PACKAGE = False
SEALED_MANIFEST = WORK_ROOT / "manifests/wilddash2.frozen.json"
SEALED_RELEASE = WORK_ROOT / "manifests/wilddash2.release.json"


def persist_bootstrap_failure(notebook, stage, error):
    import json
    import re
    import traceback
    import uuid
    from datetime import datetime, timezone
    from zipfile import ZIP_DEFLATED, ZipFile

    failed_at = datetime.now(timezone.utc)
    failure_id = f"{failed_at.strftime('%Y%m%dT%H%M%S.%fZ')}-{stage}-{uuid.uuid4().hex[:8]}"
    root = DRIVE_ROOT / "EdgeGuard/failures/bootstrap" / failure_id
    root.mkdir(parents=True, exist_ok=False)
    rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    rendered = re.sub(r"(?i)(token|password|secret|api[_-]?key)=\\S+", r"\\1=<redacted>", rendered)
    payload = {"record_type": "edgeguard_colab_bootstrap_failure", "failure_id": failure_id, "failed_at": failed_at.isoformat(), "notebook": notebook, "stage": stage, "error_type": type(error).__name__, "traceback": rendered}
    report = root / "failure.json"
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    package = root / "failure-report.zip"
    with ZipFile(package, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(report, arcname="failure.json")
    print("EDGEGUARD BOOTSTRAP FAILURE:", package)
    return package
""",
        ),
        final_protocol,
        _cell(
            "code",
            """
import subprocess


def run_bootstrap_command(command):
    completed = subprocess.run(command, capture_output=True, text=True)
    output = "\\n".join(
        value.strip() for value in (completed.stdout, completed.stderr) if value.strip()
    )
    if output:
        print(output)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Bootstrap command failed with exit code {completed.returncode}: {command}\\n"
            + output[-8000:]
        )
    return completed


try:
    if LOCAL_TEST_MODE:
        print("LOCAL_TEST_MODE: mount, clone ve kurulum atlandı.")
    elif not (PROJECT_ROOT / ".git").is_dir():
        run_bootstrap_command(["git", "clone", "--branch", BRANCH, REPOSITORY, str(PROJECT_ROOT)])
    else:
        run_bootstrap_command(["git", "-C", str(PROJECT_ROOT), "fetch", "origin", BRANCH])
        run_bootstrap_command(["git", "-C", str(PROJECT_ROOT), "checkout", BRANCH])
        run_bootstrap_command(["git", "-C", str(PROJECT_ROOT), "pull", "--ff-only"])
except BaseException as error:
    persist_bootstrap_failure("EdgeGuard_Road_Colab.ipynb", "git-clone-or-update", error)
    raise
if not LOCAL_TEST_MODE and EXPECTED_PROJECT_COMMIT:
    run_bootstrap_command(
        ["git", "-C", str(PROJECT_ROOT), "checkout", EXPECTED_PROJECT_COMMIT]
    )
PROJECT_COMMIT = subprocess.run(
    ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if EXPECTED_PROJECT_COMMIT and PROJECT_COMMIT != EXPECTED_PROJECT_COMMIT:
    if not LOCAL_TEST_MODE:
        raise RuntimeError("Project commit does not match EXPECTED_PROJECT_COMMIT")
    print(
        "LOCAL_TEST_MODE: sabit Colab commit checkout edilmedi; "
        f"yerel HEAD={PROJECT_COMMIT[:12]}, beklenen={EXPECTED_PROJECT_COMMIT[:12]}."
    )
# Never retain imports from a prior checkout in a reused Colab kernel.
for loaded_module in list(sys.modules):
    if loaded_module == "edgeguard" or loaded_module.startswith("edgeguard."):
        del sys.modules[loaded_module]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from edgeguard.rescue.colab_failures import (  # noqa: E402
    ColabFailureReporter,
    run_logged_command,
)
from edgeguard.rescue.colab_recovery import (  # noqa: E402
    action_requirements,
    completion_is_valid,
    latest_checkpoint,
    quarantine_incomplete,
    write_completion_receipt,
)
from edgeguard.serialization import canonical_json, sha256_file, sha256_payload  # noqa: E402

ACTION_PLAN = action_requirements(
    CAMPAIGN_TARGET,
    allow_final_data=ALLOW_FINAL_DATA,
    provisional_bdd=STAGE_PROVISIONAL_BDD,
)
if RUN_ACDC and ALLOW_FINAL_DATA and "acdc" not in ACTION_PLAN["datasets"]:
    ACTION_PLAN["datasets"].append("acdc")
PLANNED_STAGES = ACTION_PLAN["stages"]
TRAINING_STAGES = ACTION_PLAN["training_stages"]
RUNTIME_REQUIRED = ACTION_PLAN["runtime_required"]
RUN_HPO = "hpo" in PLANNED_STAGES
RUN_FINAL_EVALUATION = "evaluate" in PLANNED_STAGES
if CAMPAIGN_TARGET in {"evaluate", "export", "report"} and not ALLOW_FINAL_DATA:
    raise RuntimeError(
        "Source/external final data is sealed. First complete CAMPAIGN_TARGET='final'; "
        "then provide ACCEPTED_RELEASE and set ALLOW_FINAL_DATA=True."
    )
RUN_MULTIDOMAIN_AUDIT = RUN_MULTIDOMAIN_AUDIT and "audit" in PLANNED_STAGES
RUN_DATA_STAGING = RUN_DATA_STAGING and bool(ACTION_PLAN["datasets"])
if (TRAINING_STAGES or RUN_HPO or RUN_FINAL_EVALUATION) and RUN_FREEZE:
    print("RUN_FREEZE açık: yalnız hash-bağlı insan review receipt'leri kullanılacak.")
print("EDGEGUARD ACTION PLAN:", ACTION_PLAN)

FAILURE_REPORTER = ColabFailureReporter(
    DRIVE_ROOT / "EdgeGuard/failures" / CAMPAIGN_ID / PROJECT_COMMIT,
    notebook="EdgeGuard_Road_Colab.ipynb",
    project_commit=PROJECT_COMMIT,
    context={"branch": BRANCH, "campaign_id": CAMPAIGN_ID, "local_test_mode": LOCAL_TEST_MODE},
)
FAILURE_REPORTER.add_diagnostic_root("runtime-evidence", CONTENT_ROOT / "edgeguard-evidence")
FAILURE_REPORTER.add_diagnostic_root("runtime-logs", CONTENT_ROOT / "edgeguard-logs")
COMMAND_LOG_ROOT = CONTENT_ROOT / "edgeguard-command-logs"
FAILURE_REPORTER.add_diagnostic_root("command-logs", COMMAND_LOG_ROOT)
FAILURE_REPORTER.add_diagnostic_root("work", WORK_ROOT)
FAILURE_REPORTER.install_ipython_hook()


def run_colab_command(command, *, check=True):
    command_env = os.environ.copy()
    project_src = str(PROJECT_ROOT / "src")
    command_env["PYTHONPATH"] = project_src + os.pathsep + command_env.get("PYTHONPATH", "")
    command_env["UV_CACHE_DIR"] = str(CONTENT_ROOT / "edgeguard-cache/uv")
    return run_logged_command(
        command,
        log_root=COMMAND_LOG_ROOT,
        stage=FAILURE_REPORTER.stage,
        check=check,
        cwd=PROJECT_ROOT,
        env=command_env,
    )


if not LOCAL_TEST_MODE:
    FAILURE_REPORTER.set_stage("project-install")
    run_colab_command([sys.executable, "-m", "pip", "install", "-e", f"{PROJECT_ROOT}[colab]"])
print({"project_commit": PROJECT_COMMIT, "python": sys.version})
""",
        ),
        _cell(
            "code",
            """
# Önce küçük kampanya durumunu geri yükle, sonra yalnız hedefin gerektirdiği datasetleri stage et.
FAILURE_REPORTER.set_stage("campaign-state-restore-and-conditional-staging")
CAMPAIGN_ROOT = DRIVE_ROOT / "EdgeGuard/campaigns" / CAMPAIGN_ID
RECOVERY_ROOT = CAMPAIGN_ROOT / "recovery/v2"
STATUS_PATH = CAMPAIGN_ROOT / "state/status.json"
run_colab_command(
    [sys.executable, str(PROJECT_ROOT / "scripts/manage_colab_recovery.py"),
     "package-interruption", "--status", str(STATUS_PATH),
     "--failure-root", str(CAMPAIGN_ROOT / "failures")],
)
LOCAL_STATE_ARCHIVE = CONTENT_ROOT / "edgeguard-campaign-state.tar.gz"
STATE_INCLUDE = [
    "accepted_release.candidate.json",
    "accepted_release.json",
    "audit",
    "calibration",
    "evaluation",
    "external-package",
    "ledger",
    "manifests",
    "multidomain-statistics",
    "preview",
    "reports",
    "reviews",
    "runs",
]
run_colab_command(
    [sys.executable, str(PROJECT_ROOT / "scripts/manage_colab_recovery.py"),
     "cleanup-incoming", "--store-root", str(RECOVERY_ROOT)],
)
state_pointer = RECOVERY_ROOT / "pointers/campaign-state.json"
if AUTO_RESUME and not WORK_ROOT.exists() and state_pointer.is_file():
    restore_state = [
        sys.executable, str(PROJECT_ROOT / "scripts/manage_colab_recovery.py"),
        "restore", "--store-root", str(RECOVERY_ROOT),
        "--artifact-id", "campaign-state", "--destination", str(LOCAL_STATE_ARCHIVE),
    ]
    restored = run_colab_command(restore_state, check=False)
    if restored.returncode == 0:
        run_colab_command(
            [sys.executable, str(PROJECT_ROOT / "scripts/manage_colab_recovery.py"),
             "restore-state", "--archive", str(LOCAL_STATE_ARCHIVE),
             "--destination", str(WORK_ROOT)],
        )
        print("Drive campaign state restored and verified.")
    else:
        print("Doğrulanmış önceki campaign state yok; temiz kampanya başlatılıyor.")
elif AUTO_RESUME and not WORK_ROOT.exists():
    print("İlk kampanya oturumu: geri yüklenecek state generation yok.")

if RUN_DATA_STAGING and ACTION_PLAN["datasets"]:
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts/prepare_colab_data.py"),
        "--drive-root", str(DRIVE_ROOT), "stage", "--local-root", str(LOCAL_DATA_ROOT),
    ]
    datasets_to_stage = [
        dataset for dataset in ACTION_PLAN["datasets"] if dataset in DATASET_ROOTS
    ]
    for dataset in datasets_to_stage:
        command.extend(["--dataset", dataset])
    if STAGE_PROVISIONAL_BDD:
        command.append("--allow-ineligible")
    run_colab_command(command)


def sync_work_snapshot(label: str) -> None:
    # Only small state is compressed. Checkpoints/ONNX use immutable recovery objects.
    if not WORK_ROOT.exists():
        return
    pack = [
        sys.executable, str(PROJECT_ROOT / "scripts/manage_colab_recovery.py"),
        "pack-state", "--work-root", str(WORK_ROOT), "--output", str(LOCAL_STATE_ARCHIVE),
    ]
    for relative in STATE_INCLUDE:
        pack.extend(["--include", relative])
    run_colab_command(pack)
    run_colab_command(
        [sys.executable, str(PROJECT_ROOT / "scripts/manage_colab_recovery.py"),
         "publish", "--source", str(LOCAL_STATE_ARCHIVE),
         "--store-root", str(RECOVERY_ROOT), "--artifact-id", "campaign-state",
         "--campaign-id", CAMPAIGN_ID, "--project-commit", PROJECT_COMMIT,
         "--metadata-json", json.dumps({"label": label, "target": CAMPAIGN_TARGET})],
    )
    run_colab_command(
        [sys.executable, str(PROJECT_ROOT / "scripts/manage_colab_recovery.py"),
         "status", "--output", str(STATUS_PATH),
         "--values-json", json.dumps({"state": "checkpointed", "stage": label,
                                      "target": CAMPAIGN_TARGET, "project_commit": PROJECT_COMMIT})],
    )
    print("Drive state/checkpoint generation updated:", label)
""",
        ),
        setup_stack,
        audit,
        training,
        evaluation,
        final_training,
        _cell(
            "code",
            'FAILURE_REPORTER.set_stage("final-calibration-and-evaluation")\nexec(FINAL_PROTOCOL_CODE)',
        ),
        _cell(
            "code",
            """
# Temperature sonrası source-only shift referansı; external veri eşik üretimine giremez.
FAILURE_REPORTER.set_stage("shift-calibration-and-perception-preview")
if RUN_SHIFT_CALIBRATION:
    for model in FINAL_MODELS:
        run_dir = RUN_ROOT / "final" / model / "ce"
        checkpoint = latest_checkpoint(run_dir)
        temperature = WORK_ROOT / "calibration" / model / "global-temperature.json"
        if not temperature.is_file():
            raise RuntimeError(f"Missing global temperature for {model}")
        source_summaries = []
        for dataset, manifest in zip(SCIENTIFIC_SOURCE_DATASETS, DATA_MANIFESTS, strict=True):
            target = WORK_ROOT / "evaluation/shift-calibration" / model / dataset
            if not target.exists():
                run_colab_command([str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/evaluate.py"), "run", "--resolved-config", str(run_dir / "resolved.py"), "--checkpoint", str(checkpoint), "--dataset", dataset, "--dataset-manifest", str(manifest), "--role", "train_calibration", "--temperature-file", str(temperature), "--output-dir", str(target)])
            source_summaries.append(target / "frame_uncertainty.json")
        shift_reference = WORK_ROOT / "calibration" / model / "source-shift-reference.json"
        if not shift_reference.exists():
            command = [str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/evaluate.py"), "calibrate-shift", "--checkpoint", str(checkpoint), "--output", str(shift_reference)]
            for summary in source_summaries:
                command.extend(["--summary", str(summary)])
            for manifest in DATA_MANIFESTS:
                command.extend(["--data-manifest", str(manifest)])
            run_colab_command(command)
        if RUN_ACDC:
            source_final = [WORK_ROOT / "evaluation/final" / model / dataset / "frame_uncertainty.json" for dataset in SCIENTIFIC_SOURCE_DATASETS]
            for condition in ("fog", "night", "rain", "snow"):
                external_summary = WORK_ROOT / "evaluation/acdc" / model / condition / "frame_uncertainty.json"
                output = WORK_ROOT / "evaluation/shift" / model / f"acdc-{condition}.json"
                if external_summary.is_file() and all(path.is_file() for path in source_final) and not output.exists():
                    command = [str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/evaluate.py"), "evaluate-shift", "--reference", str(shift_reference), "--external-summary", str(external_summary), "--output", str(output)]
                    for summary in source_final:
                        command.extend(["--source-summary", str(summary)])
                    run_colab_command(command)
        if RUN_PERCEPTION_PREVIEW:
            onnx_model = WORK_ROOT / "exports/final" / model / f"{model}.onnx"
            if not PREVIEW_IMAGE.is_file() or not onnx_model.is_file():
                raise RuntimeError("Perception preview requires PREVIEW_IMAGE and final ONNX")
            preview = WORK_ROOT / "preview" / model
            if not preview.exists():
                run_colab_command([str(RUNTIME_PYTHON), str(PROJECT_ROOT / "scripts/predict.py"), "--image", str(PREVIEW_IMAGE), "--model", str(onnx_model), "--output-dir", str(preview), "--emit-regions", "--emit-risk", "--shift-reference", str(shift_reference)])
""",
        ),
        _cell(
            "code",
            """
FAILURE_REPORTER.set_stage("output-snapshot-and-review-package")
sync_work_snapshot("cell-complete")
if CREATE_REVIEW_PACKAGE and WORK_ROOT.is_dir():
    review_root = DRIVE_ROOT / "EdgeGuard/review_packages"
    review_root.mkdir(parents=True, exist_ok=True)
    review_zip = review_root / f"{CAMPAIGN_ID}-{PROJECT_COMMIT[:12]}-{RUN_STAGE}-review.zip"
    run_colab_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/package_colab_outputs.py"),
            "review",
            "--source-root", str(WORK_ROOT),
            "--output", str(review_zip),
            "--campaign-id", CAMPAIGN_ID,
            "--project-commit", PROJECT_COMMIT,
        ],
    )
    print("İnceleme paketi:", review_zip)
    if DOWNLOAD_REVIEW_PACKAGE and not LOCAL_TEST_MODE:
        from google.colab import files

        files.download(str(review_zip))
print("Demo command: streamlit run", PROJECT_ROOT / "app.py")
print("Jetson handoff: scripts/jetson/build_tensorrt.py then scripts/jetson/benchmark.py")
print("Hata raporu kökü:", FAILURE_REPORTER.output_root)
if DOWNLOAD_LATEST_FAILURE_REPORT:
    latest_failure = FAILURE_REPORTER.latest_package()
    if latest_failure is None:
        raise RuntimeError("İndirilecek hata paketi bulunamadı")
    if not LOCAL_TEST_MODE:
        from google.colab import files

        files.download(str(latest_failure))
    print("Hata paketi:", latest_failure)
""",
        ),
    ]
    _pin_delivery(cells, branch=branch, project_commit=project_commit)
    _write(NOTEBOOK_ROOT / "EdgeGuard_Road_Colab.ipynb", cells)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", default="stabilize/colab-v2")
    parser.add_argument("--project-commit")
    args = parser.parse_args()
    project_commit = (
        args.project_commit
        or subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if re.fullmatch(r"[0-9a-f]{40}", project_commit) is None:
        raise ValueError("notebook project commit must be a full lowercase Git SHA")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", args.branch) is None:
        raise ValueError("notebook delivery branch is invalid")
    build_preflight_notebook(branch=args.branch, project_commit=project_commit)
    build_training_notebook(branch=args.branch, project_commit=project_commit)
    subprocess.run(
        [
            "ruff",
            "format",
            str(NOTEBOOK_ROOT / "EdgeGuard_Data_Preflight_Colab.ipynb"),
            str(NOTEBOOK_ROOT / "EdgeGuard_Road_Colab.ipynb"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
