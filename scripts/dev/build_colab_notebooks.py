#!/usr/bin/env python3
"""Build the two output-free semantic-first Colab delivery notebooks."""

# ruff: noqa: E501

from __future__ import annotations

import json
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
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")

REPOSITORY = "https://github.com/emrealmaoglu/edgeguard-road.git"
BRANCH = "rescue/semantic-first"
PROJECT_ROOT = Path("/content/edgeguard-road")
DRIVE_ROOT = Path("/content/drive/MyDrive")
ACTIVE_SOURCE_DATASETS = ["cityscapes", "bdd100k", "idd20k"]
OPTIONAL_FINAL_DATASETS = []  # Model/protokol freeze sonrası ör. ["acdc"]
DATASETS_TO_BUNDLE = ACTIVE_SOURCE_DATASETS + OPTIONAL_FINAL_DATASETS
VERIFY_ARCHIVE_HASHES = True  # Resmî arşivleri bir kez SHA-256/MD5 ile kaydeder.
CREATE_BUNDLES = False  # Hazır klasörler denetlendikten sonra True yapın.
REPLACE_BUNDLES = False  # Yalnız kaynak klasörü bilinçli değiştiyse True yapın.
""",
        ),
        _cell(
            "code",
            """
import subprocess
import sys

if not (PROJECT_ROOT / ".git").is_dir():
    subprocess.run(["git", "clone", "--branch", BRANCH, REPOSITORY, str(PROJECT_ROOT)], check=True)
else:
    subprocess.run(["git", "-C", str(PROJECT_ROOT), "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(PROJECT_ROOT), "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", str(PROJECT_ROOT), "pull", "--ff-only"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(PROJECT_ROOT)], check=True)
""",
        ),
        _cell(
            "code",
            """
# Drive klasörlerini oluştur, erişim talimatlarını ve eksikleri tek raporda göster.
import json

PREFLIGHT_REPORT = DRIVE_ROOT / "EdgeGuard/manifests/colab-data-inventory.json"
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
subprocess.run(inventory_command, check=True)
inventory = json.loads(PREFLIGHT_REPORT.read_text())
for row in inventory["datasets"]:
    print("\\n", row["dataset_id"], "=>", row["state"], "|", row["activation_phase"])
    print("resmî kaynak:", row["official_url"])
    print("işlem:", row["instructions"])
    if row["missing_required_paths"]:
        print("eksik hazır yollar:", row["missing_required_paths"])
    for package in row["packages"]:
        print("paket:", package["filename"], "Drive'da:", package["present"])
""",
        ),
        _cell(
            "markdown",
            """
## Manuel hazırlama hedefi

Resmî paketleri `MyDrive/EdgeGuard/archives/<dataset_id>/` altında saklayın; giriş bilgisi, cookie veya geçici indirme URL'sini notebook'a yazmayın. Paketleri açtıktan sonra aşağıdaki hazır kökleri oluşturun:

- `MyDrive/EdgeGuard/datasets/cityscapes/{leftImg8bit,gtFine}`
- `MyDrive/EdgeGuard/datasets/bdd100k/{images/10k,labels/sem_seg/masks}`
- `MyDrive/EdgeGuard/datasets/idd20k/{leftImg8bit,gtFine}` — Part I ve Part II aynı köke açılmalı.

Notebook yalnız klasörlerin varlığını değil, sonraki bilimsel audit'in beklediği train/val alt yollarını da kontrol eder. Arşiv adlarını değiştirmeyin; BDD paketlerinde yayımlanmış MD5 değerleri erişim planında kayıtlıdır. Bilinmeyen veya üçüncü taraf ayna kullanmayın.
""",
        ),
        _cell(
            "code",
            """
# Hazır kökler geçtikten sonra tek dosyalı, hash-bağlı Drive paketlerini üret.
if CREATE_BUNDLES:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/prepare_colab_data.py"),
        "--drive-root",
        str(DRIVE_ROOT),
        "bundle",
    ]
    for dataset in DATASETS_TO_BUNDLE:
        command.extend(["--dataset", dataset])
    if REPLACE_BUNDLES:
        command.append("--replace")
    subprocess.run(command, check=True)
else:
    print("CREATE_BUNDLES=False: manuel indirme/açma ve inventory incelemesi bekleniyor.")
""",
        ),
        _cell(
            "code",
            """
# Eğitim notebook'una geçiş kapısı: üç receipt ve üç tar dosyası birlikte bulunmalı.
bundle_root = DRIVE_ROOT / "EdgeGuard/bundles"
missing = []
for dataset in ACTIVE_SOURCE_DATASETS:
    for suffix in (".prepared.tar", ".prepared.tar.receipt.json"):
        candidate = bundle_root / f"{dataset}{suffix}"
        if not candidate.is_file():
            missing.append(str(candidate))
if missing:
    print("Eğitim öncesi eksikler:\\n- " + "\\n- ".join(missing))
else:
    print("VERİ HAZIRLIK KAPISI GEÇTİ — EdgeGuard_Road_Colab.ipynb açılabilir.")
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
    audit_text = "".join(audit["source"])
    audit_text = audit_text.replace(
        "if RUN_MULTIDOMAIN_AUDIT:\n",
        "if RUN_MULTIDOMAIN_AUDIT:\n",
    )
    audit_text = audit_text.replace(
        '                    str(CITYSCAPES_ROOT),\n                    "--output-root",',
        '                    str(DATASET_ROOTS[dataset]),\n                    "--output-root",',
    )
    audit["source"] = _source(audit_text)

    training_text = "".join(training["source"])
    if "sync_work_snapshot" not in training_text:
        training_text = training_text.replace(
            "        subprocess.run(command, check=True)\n",
            "        subprocess.run(command, check=True)\n"
            '        sync_work_snapshot(f"{RUN_STAGE}-{model}")\n',
        )
    training["source"] = _source(training_text)

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
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")

REPOSITORY = "https://github.com/emrealmaoglu/edgeguard-road.git"
BRANCH = "rescue/semantic-first"
PROJECT_ROOT = Path("/content/edgeguard-road")
DRIVE_ROOT = Path("/content/drive/MyDrive")
LOCAL_DATA_ROOT = Path("/content/edgeguard-data")
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
ACTIVE_SOURCE_DATASETS = ["cityscapes", "bdd100k", "idd20k"]
OPTIONAL_EVALUATION_DATASETS = []  # Model freeze sonrası ör. ["acdc"]
WORK_ROOT = Path("/content/edgeguard-work")
CAMPAIGN_ID = "semantic-first-v1"  # Aynı kampanyayı sürdürürken değiştirmeyin.
RUN_STAGE = "smoke"  # smoke, pilot, screening, final
RUN_MODELS = ["segformer_b0", "fast_scnn", "pidnet_s", "ddrnet_23_slim", "bisenetv2"]
RUN_DATA_STAGING = True
RESTORE_LATEST_SNAPSHOT = True
RUN_MULTIDOMAIN_AUDIT = True
RUN_FREEZE = False  # Üç candidate manifest insan tarafından incelendikten sonra True.
RUN_SOURCE_VALIDATION_AUDIT = False
RUN_FREEZE_SOURCE_VALIDATION = False
RUN_TRAINING = False
RUN_HPO = False
FINAL_MODELS = []
RUN_FINAL_EVALUATION = False
RUN_ACDC = False
RUN_SEALED_PACKAGE = False
SEALED_MANIFEST = WORK_ROOT / "manifests/wilddash2.frozen.json"
SEALED_RELEASE = WORK_ROOT / "manifests/wilddash2.release.json"
""",
        ),
        final_protocol,
        _cell(
            "code",
            """
import subprocess
import sys

if not (PROJECT_ROOT / ".git").is_dir():
    subprocess.run(["git", "clone", "--branch", BRANCH, REPOSITORY, str(PROJECT_ROOT)], check=True)
else:
    subprocess.run(["git", "-C", str(PROJECT_ROOT), "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(PROJECT_ROOT), "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", str(PROJECT_ROOT), "pull", "--ff-only"], check=True)
PROJECT_COMMIT = subprocess.run(
    ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
subprocess.run([sys.executable, "-m", "pip", "install", "-e", f"{PROJECT_ROOT}[colab]"], check=True)
print({"project_commit": PROJECT_COMMIT, "python": sys.version})
""",
        ),
        _cell(
            "code",
            """
# Tek dosyalı paketleri sırayla kopyalar; SHA-256 ve 175/200 GiB kapıları zorunludur.
import hashlib
import json
import os
import shutil
import tarfile
import time

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
    for dataset in ACTIVE_SOURCE_DATASETS + OPTIONAL_EVALUATION_DATASETS:
        command.extend(["--dataset", dataset])
    subprocess.run(command, check=True)

SNAPSHOT_ROOT = DRIVE_ROOT / "EdgeGuard/artifacts" / CAMPAIGN_ID / PROJECT_COMMIT
SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)


def _safe_restore(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            parts = Path(member.name).parts
            if (
                member.name.startswith("/")
                or ".." in parts
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
            ):
                raise RuntimeError(f"Unsafe snapshot member: {member.name}")
        archive.extractall(destination)


def _sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


if RESTORE_LATEST_SNAPSHOT and not WORK_ROOT.exists():
    snapshots = sorted(SNAPSHOT_ROOT.glob("*.tar.gz"))
    if snapshots:
        latest = snapshots[-1]
        receipt = json.loads(latest.with_suffix(latest.suffix + ".json").read_text())
        digest = _sha256_stream(latest)
        if digest != receipt["sha256"]:
            raise RuntimeError("Drive snapshot SHA-256 mismatch")
        _safe_restore(latest, Path("/content"))
        print("Snapshot restored:", latest.name)


def sync_work_snapshot(label: str) -> None:
    if not WORK_ROOT.exists():
        return
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    local_archive = Path("/content") / f"edgeguard-{timestamp}-{label}.tar.gz"
    with tarfile.open(local_archive, "w:gz") as archive:
        archive.add(WORK_ROOT, arcname=WORK_ROOT.name, recursive=True)
    digest = _sha256_stream(local_archive)
    destination = SNAPSHOT_ROOT / local_archive.name
    incoming = destination.with_name(f".{destination.name}.incoming")
    shutil.copyfile(local_archive, incoming)
    os.replace(incoming, destination)
    receipt = {"sha256": digest, "project_commit": PROJECT_COMMIT, "label": label}
    destination.with_suffix(destination.suffix + ".json").write_text(json.dumps(receipt))
    local_archive.unlink()
    print("Drive snapshot:", destination.name)
""",
        ),
        setup_stack,
        audit,
        training,
        evaluation,
        _cell(
            "code",
            """
exec(FINAL_PROTOCOL_CODE)
sync_work_snapshot("cell-complete")
print("Demo command: streamlit run", PROJECT_ROOT / "app.py")
""",
        ),
    ]
    _write(NOTEBOOK_ROOT / "EdgeGuard_Road_Colab.ipynb", cells)


def main() -> int:
    build_preflight_notebook()
    build_training_notebook()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
