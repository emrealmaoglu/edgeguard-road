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


def build_master_notebook(*, branch: str, project_commit: str) -> Path:
    """Write one deterministic notebook pinned to an immutable application commit."""
    if re.fullmatch(r"[0-9a-f]{40}", project_commit) is None:
        raise ValueError("project_commit must be a full lowercase Git SHA")
    cells = [
        _cell(
            "markdown",
            """
# EdgeGuard · Tek Notebook Üretim Pipeline'ı

Colab'da **L4 GPU** ve **Yüksek RAM** seçin, ardından yalnızca **Çalışma zamanı → Tümünü çalıştır** deyin. Notebook; Cityscapes + IDD20K verisini Drive'daki doğrulanmış paketlerden yerel diske alır, beş modeli canary/smoke/pilot/screening/HPO/final aşamalarından geçirir, seçim ve ablation'ları tamamlar, resmî kaynak değerlendirmesini kabul sonrasında açar ve Jetson/tez/Streamlit paketlerini Drive'a yazar.

Oturum kapanırsa yeni L4 + Yüksek RAM oturumunda aynı notebook için yeniden **Tümünü çalıştır** deyin. Hash-doğrulanmış aşamalar atlanır; eksik eğitim Drive checkpoint'inden devam eder. TensorRT engine Colab'da üretilmez; gerçek Jetson üzerinde oluşturulur. Jetson ölçümleri gelene kadar `not_run` kalır.
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
        MASTER_RESULT = {
            "record_type": "edgeguard_notebook_local_contract",
            "status": "passed",
            "campaign_id": CAMPAIGN_ID,
            "project_commit": EXPECTED_PROJECT_COMMIT,
            "scientific_status": "not_run",
        }
    else:
        environment = os.environ.copy()
        for key in (
            "CONDA_PREFIX",
            "PIP_PREFIX",
            "PIP_REQUIRE_VIRTUALENV",
            "PIP_TARGET",
            "PYTHONHOME",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
            "VIRTUAL_ENV",
        ):
            environment.pop(key, None)
        for key in tuple(environment):
            if key.startswith("UV_"):
                environment.pop(key, None)
        environment["MPLBACKEND"] = "Agg"
        environment["PYTHONNOUSERSITE"] = "1"
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
                *EXECUTION_MODE_ARGS,
                "--result",
                str(RESULT_PATH),
            ],
            cwd=PROJECT_ROOT,
            env=environment,
        )
        MASTER_RESULT = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
except BaseException as error:
    persist_failure("production-pipeline", error)
    raise
""",
        ),
        _cell(
            "code",
            """
print(json.dumps(MASTER_RESULT, ensure_ascii=False, indent=2))
if MASTER_RESULT.get("status") == "completed":
    deliveries = MASTER_RESULT["drive_deliveries"]
    print("\\nTAMAMLANDI · önerilen model:", MASTER_RESULT["recommended_model"])
    for name, path in deliveries.items():
        print(f"{name}: {path}")
    jetson_path = deliveries.get("EdgeGuard_Jetson_Release.zip")
    if AUTO_DOWNLOAD_JETSON_RELEASE and jetson_path and not LOCAL_TEST_MODE:
        from google.colab import files

        files.download(jetson_path)
else:
    print("Yerel notebook sözleşmesi geçti; gerçek bilimsel/GPU sonucu üretilmedi.")
""",
        ),
    ]
    payload = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": NOTEBOOK.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return NOTEBOOK


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
