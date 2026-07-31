#!/usr/bin/env python3
"""Build the two output-free semantic-first Colab delivery notebooks."""

# ruff: noqa: E501

from __future__ import annotations

import json
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


def build_preflight_notebook() -> None:
    cells = [
        _cell(
            "markdown",
            """
# EdgeGuard · Drive veri ön-hazırlığı

Bu notebook veri indirme yetkisi vermez ve lisans kabulünü otomatikleştirmez. Resmî paketleri Drive'a yerleştirdikten sonra klasör düzenini denetler ve her veri setini tek, SHA-256 bağlı `.tar` dosyasına dönüştürür. Eğitim notebook'u Drive'daki binlerce küçük dosyayı okumak yerine bu paketleri `/content` alanına taşır.

Çekirdek eğitim için yalnız **Cityscapes Fine + BDD100K 10K Semantic + IDD20K Part I/II** gerekir. ACDC ve kapalı external setler model dondurulmadan indirilmez.
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
BRANCH = "rescue/semantic-first"
EXPECTED_PROJECT_COMMIT = "3134d3f1e6d3bf23ede14f7a29b0adbeb51e0e89"
SCIENTIFIC_SOURCE_DATASETS = ["cityscapes", "idd20k"]
PROVISIONAL_ENGINEERING_DATASETS = ["bdd100k"]
OPTIONAL_FINAL_DATASETS = []  # Model/protokol freeze sonrası ör. ["acdc"]
DATASETS_TO_BUNDLE = ["cityscapes", "bdd100k", "idd20k"] + OPTIONAL_FINAL_DATASETS
VERIFY_ARCHIVE_HASHES = True  # Resmî arşivleri bir kez SHA-256/MD5 ile kaydeder.
RUN_ARCHIVE_PREPARATION = False  # Arşivler Drive'a yüklendikten sonra bir kez True yapın.
BDD_SOURCE_PROFILE = "kaggle_mirror"  # Drive'daki bdd100k.zip; yalnız audit/smoke kanıtıdır.
CREATE_BUNDLES = True  # Hazırlanan yerel kökten doğrudan tek Drive tar üretir.
REPLACE_BUNDLES = False  # Yalnız kaynak klasörü bilinçli değiştiyse True yapın.
REUSE_VERIFIED_LEGACY = True  # Mevcut hash-bağlı Cityscapes bundle'ını yeniden kullanır.
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
if VERIFY_ARCHIVE_HASHES:
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
import contextlib
import shutil
import threading
import time

from edgeguard.rescue.colab_data import copy_archive_to_local, preparation_disk_budget
from edgeguard.serialization import canonical_json, sha256_file

FAILURE_REPORTER.set_stage("dataset-preparation-and-bundling")
CONTENT_ROOT = Path(os.environ.get("EDGEGUARD_TEST_CONTENT_ROOT", "/content"))
PREPARE_ROOT = CONTENT_ROOT / "edgeguard-prepare"
CACHE_ROOT = CONTENT_ROOT / "edgeguard-archive-cache"
archive_root = DRIVE_ROOT / "EdgeGuard/archives"


def _tree_progress(roots):
    files = 0
    byte_size = 0
    for root in roots:
        if not root.exists():
            continue
        for directory, _subdirectories, filenames in os.walk(root):
            for filename in filenames:
                try:
                    byte_size += (Path(directory) / filename).stat().st_size
                    files += 1
                except OSError:
                    continue
    return files, byte_size


@contextlib.contextmanager
def live_preparation_progress(dataset, phase, roots, interval_seconds=60):
    # Print bounded liveness evidence while a quiet archive subprocess runs.
    stopped = threading.Event()
    started = time.monotonic()

    def report():
        while not stopped.wait(interval_seconds):
            files, byte_size = _tree_progress(roots)
            free = shutil.disk_usage(CONTENT_ROOT).free
            print(
                f"EDGEGUARD PROGRESS dataset={dataset} phase={phase['value']} "
                f"elapsed_min={(time.monotonic() - started) / 60:.1f} "
                f"files={files} bytes={byte_size} free_bytes={free}",
                flush=True,
            )

    print(
        f"{dataset}: otomatik canlı durum satırı her {interval_seconds} saniyede yazılacak. "
        "Dosya/byte sayısı artmıyorsa aynı komutu yeniden başlatmayın.",
        flush=True,
    )
    print(
        "Gerekirse Colab terminalinde kontrol edin: "
        "ps -eo pid,etime,%cpu,%mem,stat,cmd | grep '[p]repare_dataset.py'; "
        f"du -sh {PREPARE_ROOT} {CACHE_ROOT}; df -h {CONTENT_ROOT}",
        flush=True,
    )
    worker = threading.Thread(target=report, name=f"edgeguard-progress-{dataset}", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join(timeout=2)
        files, byte_size = _tree_progress(roots)
        print(
            f"EDGEGUARD PROGRESS dataset={dataset} phase={phase['value']} "
            f"elapsed_min={(time.monotonic() - started) / 60:.1f} "
            f"files={files} bytes={byte_size} final=True",
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
    cache_receipt = {
        "source": str(source.resolve()),
        "byte_size": source.stat().st_size,
        "expected_sha256": expected_sha256,
        "copied_sha256": copy_receipt["sha256"],
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
        phase = {"value": "archive-verify-extract-map"}
        progress_roots = (dataset_cache, PREPARE_ROOT / f".{dataset}.incoming", prepared)
        with live_preparation_progress(dataset, phase, progress_roots):
            run_colab_command(command)
            if CREATE_BUNDLES:
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
    run_colab_command(inventory_command)
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
    _write(NOTEBOOK_ROOT / "EdgeGuard_Data_Preflight_Colab.ipynb", cells)


def build_training_notebook() -> None:
    existing = json.loads((NOTEBOOK_ROOT / "EdgeGuard_Road_Colab.ipynb").read_text())

    def find_cell(marker: str) -> dict[str, Any]:
        return next(cell for cell in existing["cells"] if marker in "".join(cell["source"]))

    final_protocol = find_cell("FINAL_PROTOCOL_CODE =")
    setup_stack = find_cell("Pinned compatibility cascade")
    audit = find_cell("Audit is the hard scientific gate")
    training = find_cell("Five equal-protocol random-init runs")
    evaluation = find_cell("Frozen evaluation and ONNX export")
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
    final_protocol["source"] = _source(protocol_text)
    setup_text = """# Pinned compatibility cascade; runtime comes from the verified receipt.
FAILURE_REPORTER.set_stage("runtime-compatibility-cascade")
if LOCAL_TEST_MODE:
    RUNTIME_PYTHON = Path(sys.executable)
    MMSEG_ROOT = PROJECT_ROOT
    print("LOCAL_TEST_MODE: CUDA compatibility installation skipped.")
else:
    import shutil

    compatibility_evidence = CONTENT_ROOT / "edgeguard-evidence"
    compatibility_logs = CONTENT_ROOT / "edgeguard-logs"
    drive_runtime_evidence = SNAPSHOT_ROOT / "runtime-compatibility"

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
        "--runtime-current-root", str(CONTENT_ROOT / "edgeguard-runtime-current"),
        "--runtime-py311-root", str(CONTENT_ROOT / "edgeguard-runtime-py311"),
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
    runtime_report = CONTENT_ROOT / "edgeguard-evidence/resolved-runtime.json"
    run_colab_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/resolve_colab_runtime.py"),
            "--receipt", str(CONTENT_ROOT / "edgeguard-evidence/compatibility_receipt.json"),
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
    for source in (compatibility_evidence / "compatibility_receipt.json", runtime_report):
        shutil.copy2(source, work_runtime_evidence / source.name)
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
    audit_text = audit_text.replace("subprocess.run(", "run_colab_command(")
    audit["source"] = _source(audit_text)

    training_text = "".join(training["source"])
    if not training_text.startswith("FAILURE_REPORTER.set_stage"):
        training_text = 'FAILURE_REPORTER.set_stage(f"training-{RUN_STAGE}")\n' + training_text
    training_text = training_text.replace(
        'raise RuntimeError("Three reviewed/frozen source manifests are required")',
        'raise RuntimeError("All reviewed/frozen scientific source manifests are required")',
    )
    if "sync_work_snapshot" not in training_text:
        training_text = training_text.replace(
            "        run_colab_command(command)\n",
            "        run_colab_command(command)\n"
            '        sync_work_snapshot(f"{RUN_STAGE}-{model}")\n',
        )
    training_text = training_text.replace("subprocess.run(", "run_colab_command(")
    training["source"] = _source(training_text)

    evaluation_text = "".join(evaluation["source"])
    if not evaluation_text.startswith("FAILURE_REPORTER.set_stage"):
        evaluation_text = (
            'FAILURE_REPORTER.set_stage(f"evaluation-export-{RUN_STAGE}")\n' + evaluation_text
        )
    evaluation_text = evaluation_text.replace(
        'zip(("cityscapes", "bdd100k", "idd20k"), DATA_MANIFESTS, strict=True)',
        "zip(SCIENTIFIC_SOURCE_DATASETS, DATA_MANIFESTS, strict=True)",
    )
    evaluation_text = evaluation_text.replace(
        "len(screening_evidence) >= 6",
        "len(screening_evidence) >= 2 * len(SCIENTIFIC_SOURCE_DATASETS)",
    )
    evaluation_text = evaluation_text.replace("subprocess.run(", "run_colab_command(")
    evaluation["source"] = _source(evaluation_text)

    cells = [
        _cell(
            "markdown",
            """
# EdgeGuard-Road · 200 GiB güvenli eğitim akışı

Akış: Drive paket doğrulama → `/content` staging → audit/split → smoke/pilot/screening → evaluation → ONNX → tek dosyalı Drive snapshot. Gerçek veri olmadan bilimsel metrik üretmez. Önce `EdgeGuard_Data_Preflight_Colab.ipynb` kapısını geçirin.
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
BRANCH = "rescue/semantic-first"
EXPECTED_PROJECT_COMMIT = "3134d3f1e6d3bf23ede14f7a29b0adbeb51e0e89"
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
PROVISIONAL_ENGINEERING_DATASETS = ["bdd100k"]
STAGE_PROVISIONAL_BDD = True  # False yapılırsa Cityscapes+IDD bilimsel akışı yine çalışır.
OPTIONAL_EVALUATION_DATASETS = []  # Model freeze sonrası ör. ["acdc"]
WORK_ROOT = CONTENT_ROOT / "edgeguard-work"
CAMPAIGN_ID = "semantic-first-cs-idd-v1"  # BDD-mirror içermeyen bilimsel kampanya kimliği.
RUN_STAGE = "smoke"  # smoke, pilot, screening, final
RUN_MODELS = ["segformer_b0", "fast_scnn", "pidnet_s", "ddrnet_23_slim", "bisenetv2"]
ALLOW_INELIGIBLE_BDD_SMOKE = True  # Provisional audit için; DATA_MANIFESTS listesine girmez.
RUN_DATA_STAGING = not LOCAL_TEST_MODE
RESTORE_LATEST_SNAPSHOT = True
RUN_MULTIDOMAIN_AUDIT = not LOCAL_TEST_MODE
RUN_FREEZE = False  # Cityscapes + IDD candidate manifestleri incelendikten sonra True.
RUN_SOURCE_VALIDATION_AUDIT = False
RUN_FREEZE_SOURCE_VALIDATION = False
RUN_TRAINING = False
RUN_HPO = False
FINAL_MODELS = []
RUN_FINAL_EVALUATION = False
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from edgeguard.rescue.colab_failures import (  # noqa: E402
    ColabFailureReporter,
    run_logged_command,
)

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
# Tek dosyalı paketleri sırayla kopyalar; SHA-256 ve 175/200 GiB kapıları zorunludur.
FAILURE_REPORTER.set_stage("data-staging-and-snapshot-restore")
if RUN_DATA_STAGING:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/prepare_colab_data.py"),
        "--drive-root",
        str(DRIVE_ROOT),
        "stage",
        "--local-root",
        str(LOCAL_DATA_ROOT),
    ]
    datasets_to_stage = SCIENTIFIC_SOURCE_DATASETS + OPTIONAL_EVALUATION_DATASETS
    if STAGE_PROVISIONAL_BDD:
        datasets_to_stage += PROVISIONAL_ENGINEERING_DATASETS
    for dataset in datasets_to_stage:
        command.extend(["--dataset", dataset])
    if ALLOW_INELIGIBLE_BDD_SMOKE:
        command.append("--allow-ineligible")
    run_colab_command(command)

SNAPSHOT_ROOT = DRIVE_ROOT / "EdgeGuard/campaigns" / CAMPAIGN_ID / PROJECT_COMMIT
SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
SNAPSHOT = SNAPSHOT_ROOT / "campaign.latest.tar.gz"
SNAPSHOT_INCLUDE = [
    "audit",
    "calibration",
    "evaluation",
    "exports",
    "external-package",
    "ledger",
    "manifests",
    "multidomain-statistics",
    "preview",
    "reports",
    "runs",
]

if RESTORE_LATEST_SNAPSHOT and SNAPSHOT.is_file() and not WORK_ROOT.exists():
    run_colab_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/package_colab_outputs.py"),
            "restore",
            "--snapshot", str(SNAPSHOT),
            "--destination", str(WORK_ROOT),
        ],
    )


def sync_work_snapshot(label: str) -> None:
    if not WORK_ROOT.exists():
        return
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/package_colab_outputs.py"),
        "snapshot",
        "--work-root", str(WORK_ROOT),
        "--output", str(SNAPSHOT),
        "--campaign-id", CAMPAIGN_ID,
        "--project-commit", PROJECT_COMMIT,
    ]
    for relative in SNAPSHOT_INCLUDE:
        command.extend(["--include", relative])
    run_colab_command(command)
    print("Drive snapshot updated:", label, SNAPSHOT)
""",
        ),
        setup_stack,
        audit,
        training,
        evaluation,
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
        checkpoints = sorted(run_dir.glob("*.pth"))
        if not checkpoints:
            raise RuntimeError(f"Missing final checkpoint for {model}")
        checkpoint = checkpoints[-1]
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
            onnx_model = WORK_ROOT / "exports/final" / f"{model}.onnx"
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
    review_root = DRIVE_ROOT / "EdgeGuard/downloads"
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
    _write(NOTEBOOK_ROOT / "EdgeGuard_Road_Colab.ipynb", cells)


def main() -> int:
    build_preflight_notebook()
    build_training_notebook()
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
