#!/usr/bin/env python3
"""Build the single output-free EdgeGuard master Colab notebook."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
NOTEBOOK = ROOT / "notebooks/EdgeGuard_Master_Colab.ipynb"
PHASE_NOTEBOOK_ROOT = ROOT / "notebooks/phases"


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


PHASE_SECTIONS: tuple[tuple[str, str], ...] = (
    ("smoke", "Duman testi (50 adım, kasıtlı kesinti + devam kanıtı)"),
    ("pilot", "Pilot (600 adım)"),
    (
        "screening",
        "Tarama (2.500 adım) — kontrol noktası: mIoU'lar ~0,35/0,35/0,20 civarı olmalı ve "
        "`candidate_table.json` üç aday içermeli",
    ),
    ("hpo", "Hiperparametre optimizasyonu (3 deneme × 2 model)"),
    ("final", "Final eğitim (10.000 adım × 3 model) — en uzun faz"),
    ("evaluate", "Seçim, ablation, kabul ve resmî kaynak değerlendirmesi"),
    ("export", "ONNX ihracı"),
    ("report", "Tez figürleri ve tabloları"),
    ("package", "Teslimat paketleri"),
)


def _prelude_cells(*, branch: str, project_commit: str) -> list[dict[str, Any]]:
    """Return the setup every notebook repeats: mount, pinned checkout, helpers, status."""
    if re.fullmatch(r"[0-9a-f]{40}", project_commit) is None:
        raise ValueError("project_commit must be a full lowercase Git SHA")
    return [
        _cell(
            "markdown",
            """
# EdgeGuard · Tek Notebook Üretim Pipeline'ı

Colab'da **L4 GPU** ve **Yüksek RAM** seçin, ardından yalnızca **Çalışma zamanı → Tümünü çalıştır** deyin. Notebook; Cityscapes + IDD20K verisini Drive'daki doğrulanmış paketlerden yerel diske alır, beş modeli canary/smoke/pilot/screening/HPO/final aşamalarından geçirir, seçim ve ablation'ları tamamlar, resmî kaynak değerlendirmesini kabul sonrasında açar ve Jetson/tez/Streamlit paketlerini Drive'a yazar.

Oturum kapanırsa yeni L4 + Yüksek RAM oturumunda aynı notebook için yeniden **Tümünü çalıştır** deyin. Hash-doğrulanmış aşamalar atlanır; eksik eğitim Drive checkpoint'inden devam eder. TensorRT engine Colab'da üretilmez; gerçek Jetson üzerinde oluşturulur. Jetson ölçümleri gelene kadar `not_run` kalır.

Kod çekilir çekilmez, ana kampanyadan bağımsız ve onu asla durdurmayan küçük bir adım `private_inputs/` klasöründeki her ham arşivi (isim bazlı kısayol yok, hepsi aynı derinlikte) tarar; boyut/görüntü sayısı/çözünürlük/format ve etiket dosyalarında gerçek ölçülmüş piksel-sınıf histogramlarını `dataset_inventory.json`/`.md` olarak Drive'a yazar ve zip halinde indirir.
""",
        ),
        _cell(
            "code",
            f"""
import json
import os
import shutil
import subprocess
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

LOCAL_TEST_MODE = os.environ.get("EDGEGUARD_NOTEBOOK_LOCAL_TEST") == "1"
REPOSITORY = "https://github.com/emrealmaoglu/edgeguard-road.git"
BRANCH = {json.dumps(branch)}
EXPECTED_PROJECT_COMMIT = {json.dumps(project_commit)}
CAMPAIGN_ID = "semantic-cs-idd-v3"
CONTENT_ROOT = Path(os.environ.get("EDGEGUARD_TEST_CONTENT_ROOT", "/content")).resolve()
PROJECT_ROOT = (
    Path(os.environ.get("EDGEGUARD_PROJECT_ROOT", Path.cwd())).resolve()
    if LOCAL_TEST_MODE
    else CONTENT_ROOT / "edgeguard-road"
)
DRIVE_ROOT = (
    Path(os.environ["EDGEGUARD_TEST_DRIVE_ROOT"]).resolve()
    if LOCAL_TEST_MODE
    else Path("/content/drive/MyDrive")
)
RESULT_PATH = CONTENT_ROOT / "edgeguard-master-result.json"
NOTEBOOK_LOG = CONTENT_ROOT / "edgeguard-notebook.log"
MASTER_CHILD_LOG = CONTENT_ROOT / "edgeguard-master-child.log"
MASTER_CHILD_FAILURE = CONTENT_ROOT / "edgeguard-master-child-failure.json"
MASTER_STAGE = CONTENT_ROOT / "edgeguard-master-stage.json"
BOOTSTRAP_FAILURE = CONTENT_ROOT / "bootstrap-failure.json"
EXECUTION_MODE_ARGS = ["--execution-mode", "production"]
AUTO_DOWNLOAD_JETSON_RELEASE = True


def persist_failure(stage, error):
    rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    if LOCAL_TEST_MODE:
        print(f"LOCAL CONTRACT FAILURE · {{stage}} · {{type(error).__name__}}")
        return None
    root = DRIVE_ROOT / "EdgeGuard/failures" / CAMPAIGN_ID / "master-notebook"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    identifier = timestamp + "-" + uuid.uuid4().hex[:8]
    report_root = root / identifier
    report_root.mkdir()
    child_failure = None
    if MASTER_CHILD_FAILURE.is_file():
        try:
            child_failure = json.loads(MASTER_CHILD_FAILURE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            child_failure = None
    payload = {{
        "schema_version": "1.0",
        "record_type": "edgeguard_master_notebook_failure",
        "campaign_id": CAMPAIGN_ID,
        "stage": stage,
        "project_commit": EXPECTED_PROJECT_COMMIT,
        "error_type": type(error).__name__,
        "message": str(error)[:2000],
        "traceback": rendered[-20000:],
        "root_failure": child_failure,
        "safe_restart": "Select L4 + High-RAM and run all again; verified state is retained.",
    }}
    report = report_root / "failure.json"
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    package = report_root / "failure-report.zip"
    with ZipFile(package, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(report, arcname="failure.json")
        for diagnostic in (MASTER_CHILD_FAILURE, MASTER_STAGE, BOOTSTRAP_FAILURE):
            if diagnostic.is_file():
                archive.write(diagnostic, arcname=diagnostic.name)
        for diagnostic in (NOTEBOOK_LOG, MASTER_CHILD_LOG):
            if diagnostic.is_file():
                tail_path = report_root / f"{{diagnostic.stem}}-tail.log"
                with diagnostic.open("rb") as source:
                    source.seek(0, 2)
                    source.seek(max(0, source.tell() - 256000))
                    tail_path.write_bytes(source.read())
                archive.write(tail_path, arcname=tail_path.name)
    print("Hata raporu Drive'a yazıldı:", package)
    if child_failure:
        print("GERÇEK DURMA AŞAMASI:", child_failure.get("stage"))
        print("GERÇEK HATA:", child_failure.get("message"))
    return package


if not LOCAL_TEST_MODE:
    from google.colab import drive

    drive.mount("/content/drive")
else:
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    CONTENT_ROOT.mkdir(parents=True, exist_ok=True)
    print("LOCAL_TEST_MODE: Drive, ağ, GPU ve eğitim işlemleri güvenle atlandı.")
""",
        ),
        _cell(
            "code",
            """
def run_visible(command, *, cwd=None, env=None):
    print("Çalıştırılıyor:", " ".join(str(value) for value in command), flush=True)
    NOTEBOOK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_LOG.open("a", encoding="utf-8") as sink:
        sink.write("\\nCOMMAND: " + " ".join(str(value) for value in command) + "\\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        tail = []
        tail_size = 0
        for line in process.stdout:
            print(line, end="", flush=True)
            sink.write(line)
            tail.append(line)
            tail_size += len(line)
            while tail_size > 24000 and len(tail) > 1:
                tail_size -= len(tail.pop(0))
        return_code = process.wait()
        sink.write(f"RETURN_CODE: {return_code}\\n")
    if return_code:
        raise RuntimeError(f"Komut {return_code} koduyla durdu: {command}\\n" + "".join(tail))


try:
    if LOCAL_TEST_MODE:
        checked_commit = EXPECTED_PROJECT_COMMIT
    else:
        if PROJECT_ROOT.exists() and (
            PROJECT_ROOT.is_symlink() or PROJECT_ROOT.resolve() != Path("/content/edgeguard-road")
        ):
            raise RuntimeError("Güvenli olmayan checkout yolu reddedildi")
        if PROJECT_ROOT.exists() and not (PROJECT_ROOT / ".git").is_dir():
            shutil.rmtree(PROJECT_ROOT)
        if not (PROJECT_ROOT / ".git").is_dir():
            run_visible(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    REPOSITORY,
                    str(PROJECT_ROOT),
                ]
            )
        dirty = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if dirty:
            shutil.rmtree(PROJECT_ROOT)
            run_visible(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    REPOSITORY,
                    str(PROJECT_ROOT),
                ]
            )
        run_visible(["git", "-C", str(PROJECT_ROOT), "fetch", "origin", BRANCH])
        run_visible(["git", "-C", str(PROJECT_ROOT), "fetch", "origin", EXPECTED_PROJECT_COMMIT])
        run_visible(
            ["git", "-C", str(PROJECT_ROOT), "checkout", "--detach", EXPECTED_PROJECT_COMMIT]
        )
        checked_commit = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if checked_commit != EXPECTED_PROJECT_COMMIT:
            raise RuntimeError("Checkout, notebook'un sabit uygulama commit'iyle eşleşmiyor")
    print("EdgeGuard uygulama kimliği:", checked_commit)
except BaseException as error:
    persist_failure("immutable-source-checkout", error)
    raise
""",
        ),
        _cell(
            "code",
            """
try:
    if LOCAL_TEST_MODE:
        print("LOCAL_TEST_MODE: private_inputs envanteri atlandı (gerçek Drive verisi yok).")
    else:
        PRIVATE_INPUTS_ROOT = DRIVE_ROOT / "EdgeGuard/private_inputs"
        INVENTORY_OUTPUT_PARENT = DRIVE_ROOT / "EdgeGuard/reports/private_inputs_inventory"
        INVENTORY_ZIP = CONTENT_ROOT / "EdgeGuard_Data_Inventory.zip"
        if PRIVATE_INPUTS_ROOT.is_dir():
            # This step runs before the locked training runtime is provisioned (that can
            # take minutes and pulls in torch/mmseg, which this step does not need), so it
            # cannot rely on that venv. It only needs a few small pure-Python packages on
            # top of the host system Python3 that already runs this notebook's own driver
            # code (Drive mount, files.download) — install them here if missing, and set
            # PYTHONPATH so the un-installed `edgeguard` package (source checkout only,
            # never editable-installed by this notebook) can be imported directly.
            run_visible(
                [
                    "/usr/bin/python3",
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "numpy>=1.24,<3",
                    "Pillow>=10,<13",
                    "pydantic>=2,<3",
                    "PyYAML>=6,<7",
                ]
            )
            inventory_environment = os.environ.copy()
            inventory_environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
            run_visible(
                [
                    "/usr/bin/python3",
                    str(PROJECT_ROOT / "scripts/inventory_private_inputs.py"),
                    "--private-inputs-root",
                    str(PRIVATE_INPUTS_ROOT),
                    "--output-parent",
                    str(INVENTORY_OUTPUT_PARENT),
                    "--zip-out",
                    str(INVENTORY_ZIP),
                ],
                cwd=PROJECT_ROOT,
                env=inventory_environment,
            )
            if INVENTORY_ZIP.is_file():
                from google.colab import files

                files.download(str(INVENTORY_ZIP))
                print("private_inputs envanteri indirildi:", INVENTORY_ZIP)
        else:
            print("private_inputs klasörü bulunamadı, envanter atlandı:", PRIVATE_INPUTS_ROOT)
except BaseException as error:
    print("private_inputs envanteri başarısız oldu (ana kampanya etkilenmedi):", repr(error))
""",
        ),
        _cell(
            "markdown",
            """
## Aşama aşama çalıştırma

Aşağıdaki her hücre kampanyanın **bir fazını** çalıştırır ve bittiğinde o faza ait
logları/kanıtları bir zip olarak indirir. Hücreleri sırayla çalıştırın.

Bir faz daha önce tamamlandıysa tekrar eğitilmez: faz tamamlanma kaydı Drive'daki state
store'a yazılıyor ve doğrulanmış fazlar atlanıyor. Oturum koparsa yeni bir çalışma
zamanında kaldığınız hücreden devam edebilirsiniz; öncesi otomatik olarak atlanır.

İnen zip'leri projede `docs/colab-logs/` altına koyun — sonuç tabloları bunlardan üretiliyor.
""",
        ),
        _cell(
            "code",
            """
MASTER_ENVIRONMENT = os.environ.copy()
for _key in (
    "CONDA_PREFIX",
    "PIP_PREFIX",
    "PIP_REQUIRE_VIRTUALENV",
    "PIP_TARGET",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "VIRTUAL_ENV",
):
    MASTER_ENVIRONMENT.pop(_key, None)
for _key in tuple(MASTER_ENVIRONMENT):
    if _key.startswith("UV_"):
        MASTER_ENVIRONMENT.pop(_key, None)
MASTER_ENVIRONMENT["MPLBACKEND"] = "Agg"
MASTER_ENVIRONMENT["PYTHONNOUSERSITE"] = "1"

MASTER_RESULT = None
BUNDLE_ROOT = CONTENT_ROOT / "edgeguard-phase-bundles"
# Mirrors run_colab_master.py's own work_root; kept here so the log bundler can reach the
# per-phase run directories without importing that script.
PHASE_WORK_ROOT = CONTENT_ROOT / "edgeguard-work-v3"


def eg_phase(target):
    \"\"\"Run one campaign phase, including every prerequisite it still needs.

    The pipeline resolves `target` to its full prerequisite closure and skips whatever the
    Drive state store already records as verified, so re-running a finished cell is cheap
    and a fresh runtime resumes instead of retraining.
    \"\"\"
    global MASTER_RESULT
    if LOCAL_TEST_MODE:
        # Local mode only proves the notebook's own cells are executable. It touches no
        # GPU, no Drive and no dataset, so it must never leave behind anything that could
        # be mistaken for a measured result.
        MASTER_RESULT = {
            "record_type": "edgeguard_notebook_local_contract",
            "status": "passed",
            "campaign_id": CAMPAIGN_ID,
            "project_commit": EXPECTED_PROJECT_COMMIT,
            "phase": target,
            "scientific_status": "not_run",
        }
        print(f"LOCAL_TEST_MODE: '{target}' fazı atlandı (gerçek GPU/veri yok).")
        return MASTER_RESULT
    try:
        run_visible(
            [
                "/usr/bin/python3",
                str(PROJECT_ROOT / "scripts/run_colab_master.py"),
                "--project-root",
                str(PROJECT_ROOT),
                "--project-commit",
                EXPECTED_PROJECT_COMMIT,
                "--drive-root",
                str(DRIVE_ROOT),
                "--content-root",
                str(CONTENT_ROOT),
                "--target",
                target,
                *EXECUTION_MODE_ARGS,
                "--result",
                str(RESULT_PATH),
            ],
            cwd=PROJECT_ROOT,
            env=MASTER_ENVIRONMENT,
        )
    except BaseException as error:
        persist_failure(f"phase-{target}", error)
        # Logs matter most when a phase fails, so package them before propagating.
        eg_bundle(f"{target}-FAILED")
        raise
    MASTER_RESULT = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    print(f"\\n=== '{target}' fazı tamamlandı ===")
    return MASTER_RESULT


def eg_status():
    \"\"\"Print which phases the Drive state store already records as verified.

    After a dropped session this is the first thing to run: it says which cell to resume
    from, instead of re-running everything to find out.
    \"\"\"
    order = [
        "preflight", "restore", "stage-data", "canary", "smoke", "pilot",
        "extension-smoke", "screening", "hpo", "final", "selection", "ablation",
        "accept", "validation-data", "evaluate", "export", "report", "package",
    ]
    if LOCAL_TEST_MODE:
        print("LOCAL_TEST_MODE: durum sorgusu atlandı.")
        return None
    # Mirrors ColabPipeline.state_root. This lives on the local work root, so in a fresh
    # runtime it is empty until the campaign state tarball has been restored from Drive --
    # which the first phase cell does as part of its own prerequisites.
    state_root = PHASE_WORK_ROOT / "pipeline-v3"
    if not state_root.is_dir():
        print(
            "Yerel durum kaydı yok. Taze bir oturumdasınız: ilk faz hücresini "
            "çalıştırın; Drive'daki kampanya durumu geri yüklendikten sonra bitmiş "
            "fazlar otomatik atlanacak. Sonra bu hücreyi tekrar çalıştırabilirsiniz."
        )
        return []
    done = set()
    for phase in order:
        marker = state_root / phase / "completion.json"
        if marker.is_file():
            done.add(phase)
    print("Kampanya durumu (Drive kaydına göre):\\n")
    for phase in order:
        mark = "OK  " if phase in done else "-   "
        print(f"  {mark}{phase}")
    remaining = [phase for phase in order if phase not in done]
    if remaining:
        print(f"\\nSıradaki faz: {remaining[0]}  ->  o fazın hücresinden devam edin.")
    else:
        print("\\nTüm fazlar tamamlanmış görünüyor.")
    return sorted(done)


def eg_bundle(label):
    \"\"\"Zip this phase's logs/evidence and hand them to the browser as a download.

    Never raises: a failed archive must not discard a phase's real training result.
    \"\"\"
    if LOCAL_TEST_MODE:
        print(f"LOCAL_TEST_MODE: '{label}' log paketi atlandı.")
        return None
    try:
        BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
        staging = BUNDLE_ROOT / label
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        sources = [
            CONTENT_ROOT / "edgeguard-evidence",
            CONTENT_ROOT / "edgeguard-logs",
            CONTENT_ROOT / "edgeguard-master-child.log",
            CONTENT_ROOT / "edgeguard-master-stage.json",
            RESULT_PATH,
            PHASE_WORK_ROOT / "runs" / label,
            PHASE_WORK_ROOT / "reports",
        ]
        # Text evidence only. Model weights live in Drive recovery already, and copying
        # them here produced a 335 MiB archive that Colab's browser download could not
        # finish -- which cost the logs entirely, since a bundle that never arrives is
        # worth nothing.
        keep_suffixes = {".log", ".json", ".md", ".csv", ".py", ".txt", ".yaml", ".yml"}
        skipped_bytes = 0

        def _ignore(directory, names):
            ignored = []
            for name in names:
                candidate = Path(directory) / name
                if candidate.is_dir():
                    continue
                if candidate.suffix.lower() not in keep_suffixes:
                    nonlocal skipped_bytes
                    try:
                        skipped_bytes += candidate.stat().st_size
                    except OSError:
                        pass
                    ignored.append(name)
            return ignored

        copied = 0
        for source in sources:
            if not source.exists():
                continue
            target = staging / source.name
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True, ignore=_ignore)
            else:
                shutil.copy2(source, target)
            copied += 1
        # ZIP's DOS-era date field cannot represent a timestamp before 1980-01-01, and
        # `shutil.copy2` above preserves each source file's original mtime. A file
        # restored from a Drive tar/zip with no timestamp metadata lands at the Unix
        # epoch (1970), which is exactly this case: on 2026-08-14 a real failed run's
        # bundle call raised "ZIP does not support timestamps before 1980" and produced
        # no archive at all -- silently, since this function never re-raises -- at
        # exactly the moment a failure diagnostic was most needed. Clamp forward rather
        # than deleting timestamp information a human might still find useful.
        dos_epoch = datetime(1980, 1, 1).timestamp()
        for path in staging.rglob("*"):
            if path.is_file() and path.stat().st_mtime < dos_epoch:
                os.utime(path, (dos_epoch, dos_epoch))
        archive = shutil.make_archive(str(BUNDLE_ROOT / f"edgeguard-{label}-logs"), "zip", staging)
        size_mib = Path(archive).stat().st_size / 1024 ** 2
        print(
            f"{label}: {copied} kaynak paketlendi, {size_mib:.1f} MiB "
            f"(ağırlık dosyaları hariç: {skipped_bytes / 1024 ** 2:.0f} MiB atlandı) -> {archive}"
        )
        from google.colab import files

        files.download(archive)
        return archive
    except BaseException as error:
        print(f"'{label}' log paketi oluşturulamadı (faz sonucu etkilenmedi):", repr(error))
        return None
""",
        ),
        _cell(
            "markdown",
            """
### Durum — nerede kaldım?

Oturum koptuysa önce bunu çalıştırın: hangi fazların bittiğini ve hangi hücreden devam edeceğinizi söyler.
""",
        ),
        _cell(
            "code",
            """
eg_status()
""",
        ),
    ]


def _phase_cells(target: str, title: str) -> list[dict[str, Any]]:
    return [
        _cell("markdown", f"\n### `{target}` — {title}\n"),
        _cell("code", f'\neg_phase("{target}")\neg_bundle("{target}")\n'),
    ]


def _summary_cells() -> list[dict[str, Any]]:
    return [
        _cell(
            "code",
            """
if MASTER_RESULT and MASTER_RESULT.get("status") == "completed":
    deliveries = MASTER_RESULT["drive_deliveries"]
    print("\\nTAMAMLANDI · önerilen model:", MASTER_RESULT["recommended_model"])
    for name, path in deliveries.items():
        print(f"{name}: {path}")
    jetson_path = deliveries.get("EdgeGuard_Jetson_Release.zip")
    if AUTO_DOWNLOAD_JETSON_RELEASE and jetson_path and not LOCAL_TEST_MODE:
        from google.colab import files

        files.download(jetson_path)
elif LOCAL_TEST_MODE:
    print("Yerel notebook sözleşmesi geçti; gerçek bilimsel/GPU sonucu üretilmedi.")
else:
    print("Kampanya henüz tamamlanmadı; yukarıdaki hücreleri sırayla çalıştırın.")
""",
        ),
    ]


def _write_notebook(path: Path, cells: list[dict[str, Any]]) -> Path:
    payload = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": path.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def build_master_notebook(*, branch: str, project_commit: str) -> Path:
    """Write one deterministic notebook pinned to an immutable application commit."""
    cells = _prelude_cells(branch=branch, project_commit=project_commit)
    for target, title in PHASE_SECTIONS:
        cells.extend(_phase_cells(target, title))
    cells.extend(_summary_cells())
    return _write_notebook(NOTEBOOK, cells)


def build_phase_notebooks(*, branch: str, project_commit: str) -> list[Path]:
    """Write one notebook per campaign phase, each independently runnable.

    Colab gives every notebook its own VM, so each of these re-stages the ~8 GB local
    dataset copy before its phase runs. That cost buys a clean slate per phase: a phase
    cannot inherit a half-written directory or a wedged runtime from the phase before it,
    and each produces exactly one log bundle to hand back.
    """
    written: list[Path] = []
    for index, (target, title) in enumerate(PHASE_SECTIONS, start=1):
        cells = _prelude_cells(branch=branch, project_commit=project_commit)
        cells[0] = _cell(
            "markdown",
            f"""
# EdgeGuard · {index:02d} — `{target}`

Bu notebook **yalnızca `{target}` fazını** çalıştırır: {title}

Colab'da **L4 GPU** + **Yüksek RAM** seçin, sonra **Çalışma zamanı → Tümünü çalıştır**.
Bu fazın gerektirdiği önceki aşamalar Drive'daki doğrulanmış kayıtlardan atlanır; eksik
olan varsa burada tamamlanır. Faz bitince `edgeguard-{target}-logs.zip` iner — onu
paylaşın.

Oturum koparsa aynı notebook'ta tekrar **Tümünü çalıştır** deyin; doğrulanmış iş atlanır.
""",
        )
        cells.extend(_phase_cells(target, title))
        cells.append(
            _cell(
                "markdown",
                f"""
### Bitti

`edgeguard-{target}-logs.zip` indi mi? İçindekiler `docs/colab-logs/` altına konulacak
kanıttır. Sonraki faz için bir sonraki numaralı notebook'u açın.
""",
            )
        )
        written.append(
            _write_notebook(PHASE_NOTEBOOK_ROOT / f"EdgeGuard_{index:02d}_{target}.ipynb", cells)
        )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", default="stabilize/colab-v2")
    parser.add_argument("--project-commit")
    args = parser.parse_args()
    commit = (
        args.project_commit
        or subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    print(build_master_notebook(branch=args.branch, project_commit=commit))
    for path in build_phase_notebooks(branch=args.branch, project_commit=commit):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
