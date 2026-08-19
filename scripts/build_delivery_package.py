"""Assemble the CD/flash delivery tree the course specification asks for.

The specification fixes the folder names and the three document filenames, so the tree is
built rather than curated by hand -- a hand-copied delivery drifts from the repository the
moment either changes, and the drift is invisible until an evaluator opens it.

Two things are deliberately *not* copied, and both are license constraints recorded in
`LICENSES.md` rather than preferences:

* **Trained model weights.** The reference repository is MIT, but that license is not
  asserted to cover the published checkpoints, so they stay out. Each measurement record
  carries the weight's sha256, which lets a reader verify which weights produced a number
  without the weights being redistributed.
* **Dataset archives, images and labels.** None of the four datasets permit
  redistribution.

The thesis PDF is also absent by construction: it can only come out of Word after the
author fills the personal fields and signs both declarations, so the script writes a note
naming what is missing instead of pretending the tree is complete.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "docs" / "teslim"

# (markdown source, destination inside the tree, PDF title)
DOCUMENTS = [
    (
        SOURCES / "kurulum_talimatlari.md",
        "Dokumanlar/Kurulum_Talimatlari.pdf",
        "EdgeGuard-Road — Kurulum Talimatları",
    ),
    (
        SOURCES / "kullanim_kilavuzu.md",
        "Dokumanlar/Kullanim_Kilavuzu.pdf",
        "EdgeGuard-Road — Kullanım Kılavuzu",
    ),
    (
        SOURCES / "sunumlar_hakkinda.md",
        "Sunumlar/00_SUNUMLAR_HAKKINDA.pdf",
        "EdgeGuard-Road — Sunumlar hakkında açıklama",
    ),
]

COPIES = [
    (SOURCES / "lisans_bilgileri.txt", "Dokumanlar/Lisans_Bilgileri.txt"),
    (SOURCES / "oku_beni.md", "OKU_BENI.md"),
    (SOURCES / "eksik_tez_pdf_notu.txt", "_EKSIK_BURAYA_TEZ_PDF_KONULACAK.txt"),
]

# Anything matching these must never reach the tree; asserted after assembly rather than
# trusted, because the source of a leak would be an upstream change, not this script.
FORBIDDEN = ("*.pt", "*.pth", "*.onnx", "*.engine", "*.trt", "*.ckpt", "*.safetensors")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "TESLIM_CD")
    parser.add_argument(
        "--presentation",
        type=Path,
        required=True,
        help="the term-opening presentation PDF; it lives outside the repository",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / ".local" / "presentation" / "results",
        help="measurement records the panel reads",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / ".local" / "presentation" / "video" / "edgeguard_demo.mp4",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    out: Path = args.output

    missing = [p for p in (args.presentation, args.results, args.video) if not p.exists()]
    if missing:
        for path in missing:
            print(f"eksik girdi: {path}", file=sys.stderr)
        return 1

    if out.exists():
        shutil.rmtree(out)
    for folder in ("Sunumlar", "Kaynak_Kod", "Uygulama/Demo_Videosu", "Dokumanlar"):
        (out / folder).mkdir(parents=True)

    # Source code: only what Git tracks, so ignored scratch, caches and local runs cannot
    # ride along into the delivery.
    code = out / "Kaynak_Kod" / "edgeguard-road"
    code.mkdir()
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    subprocess.run(["tar", "-x", "-C", str(code)], input=archive.stdout, check=True)
    # The handoff zip duplicates the folder beside it; one copy is enough.
    (code / "EdgeGuard_TEZ_PAKETI.zip").unlink(missing_ok=True)

    shutil.copy2(args.presentation, out / "Sunumlar" / "01_Donem_Baslangic_Sunumu.pdf")
    shutil.copy2(
        args.video, out / "Uygulama" / "Demo_Videosu" / "EdgeGuard_Sistem_Calisma_Videosu.mp4"
    )
    shutil.copytree(args.results, out / "Uygulama" / "Panel_Sonuc_Paketi")

    for source, target in COPIES:
        shutil.copy2(source, out / target)

    from render_delivery_docs import build  # noqa: PLC0415 -- optional reportlab dependency

    for source, target, title in DOCUMENTS:
        build(source.read_text(encoding="utf-8"), out / target, title)

    leaked = [p for pattern in FORBIDDEN for p in out.rglob(pattern)]
    if leaked:
        for path in leaked:
            print(f"LİSANS İHLALİ RİSKİ, pakette olmamalı: {path}", file=sys.stderr)
        return 1

    files = sum(1 for p in out.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"paket   : {out}")
    print(f"dosya   : {files}")
    print(f"boyut   : {size / 1_048_576:.1f} MB")
    print("eksik   : tezin PDF'i — _EKSIK_BURAYA_TEZ_PDF_KONULACAK.txt dosyasına bakınız")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
