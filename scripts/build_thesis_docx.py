"""Place the written chapters into the thesis template without disturbing its styles.

Generating a `.docx` from scratch would lose what makes the template worth using: its
automatic chapter numbering, its table of contents, its caption styles and its front
matter. So nothing is generated. The template is unpacked, the *body* paragraphs between
the introduction and the references are replaced with paragraphs that reference the
template's own style identifiers, and the file is packed again.

What is deliberately left alone:

* **Front matter** -- cover, evaluation form, both declarations, acknowledgements, table
  of contents. These carry field codes and signature blocks; touching them risks the parts
  of the document that are hardest to repair and easiest to check by eye.
* **Figures.** Captions must be numbered by chapter through the template's field codes and
  the images have to be positioned by hand anyway, so the script writes a marked
  placeholder paragraph and leaves placement to the author.
* **ÖZGEÇMİŞ.** Personal information the author fills in.

Tables are the one construct not translated. A Markdown table becomes a marked placeholder
naming the caption, because a Word table needs its own grid, widths and shading to match
the template, and a table that renders differently from its neighbours is exactly the
"visible disorder" the report is graded against.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import zipfile
from html import escape
from pathlib import Path

# Style identifiers read out of the template itself, not guessed.
STYLE_CHAPTER = "Balk1"
STYLE_SECTION = "Balk2"
STYLE_BODY = "PARAGRAFMETN"
STYLE_CAPTION = "ekilYazs"

PARAGRAPH_PATTERN = re.compile(r"<w:p[ >].*?</w:p>", re.S)
STYLE_PATTERN = re.compile(r'<w:pStyle w:val="([^"]+)"')
TEXT_PATTERN = re.compile(r"<w:t[^>]*>(.*?)</w:t>", re.S)


def paragraph(style: str, text: str) -> str:
    """One styled paragraph. `xml:space` is preserved so leading spaces survive."""
    return (
        f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
        f'<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'
    )


def _plain(text: str) -> str:
    """Strip Markdown emphasis and links; Word carries emphasis through runs, and the
    author applies it while reviewing rather than the script guessing at it."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]*)\*", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return text.strip()


def convert(markdown: str) -> list[str]:
    """Turn one chapter into styled paragraphs, marking what a human must place."""
    out: list[str] = []
    pending_table: str | None = None
    in_table = False

    for raw in markdown.splitlines():
        line = raw.rstrip()

        if line.startswith("|"):
            # A table caption always precedes its table, so it is already emitted; the
            # rows become one marker rather than a mangled Word table.
            if not in_table:
                in_table = True
                label = pending_table or "Çizelge"
                out.append(paragraph(STYLE_BODY, f"[BURAYA TABLO: {label}]"))
            continue
        in_table = False

        if not line:
            continue
        if line.startswith("> "):
            # Block quotes in the source are editorial asides to the author.
            continue
        if line.startswith("**[Şekil"):
            name = re.search(r"\[Şekil ([^\]]+)\]", line)
            out.append(paragraph(STYLE_BODY, f"[BURAYA ŞEKİL: {name.group(1) if name else ''}]"))
            continue
        if line.startswith("Şekil ") or line.startswith("Çizelge "):
            if line.startswith("Çizelge "):
                pending_table = _plain(line)
            out.append(paragraph(STYLE_CAPTION, _plain(line)))
            continue
        if line.startswith("# "):
            out.append(paragraph(STYLE_CHAPTER, _plain(line[2:]).upper()))
            continue
        if line.startswith("## "):
            out.append(paragraph(STYLE_SECTION, _plain(line[3:])))
            continue
        if line.startswith("### "):
            out.append(paragraph(STYLE_SECTION, _plain(line[4:])))
            continue
        if line.startswith(("- ", "* ", "1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ")):
            out.append(paragraph(STYLE_BODY, _plain(re.sub(r"^([-*]|\d+\.)\s+", "", line))))
            continue
        if set(line) <= {"-", " "} or line.startswith("---"):
            continue
        out.append(paragraph(STYLE_BODY, _plain(line)))
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--chapters", type=Path, default=Path("docs/rapor"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="also render a PDF via LibreOffice, so the result can be inspected page by "
        "page rather than trusted",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    work = Path(tempfile.mkdtemp(prefix="thesis-docx-"))
    try:
        with zipfile.ZipFile(args.template) as archive:
            archive.extractall(work)
        document = work / "word" / "document.xml"
        xml = document.read_text(encoding="utf-8")

        # Spans, not just strings: the template interleaves tables between paragraphs,
        # so the body is not a contiguous run of <w:p> elements and can only be cut by
        # character offset. Cutting by offset also removes the placeholder tables
        # sitting inside the replaced range, which is what should happen.
        matches = list(PARAGRAPH_PATTERN.finditer(xml))
        paragraphs = [m.group(0) for m in matches]
        text_of = ["".join(TEXT_PATTERN.findall(p)).strip() for p in paragraphs]

        def index_of(label: str) -> int:
            for position, (body, para) in enumerate(zip(text_of, paragraphs, strict=True)):
                style = STYLE_PATTERN.search(para)
                if body == label and style and style.group(1).startswith("Balk1"):
                    return position
            raise ValueError(f"template landmark not found: {label}")

        start, end = index_of("GİRİŞ"), index_of("KAYNAKLAR")

        replacement: list[str] = []
        for chapter in sorted(args.chapters.glob("0[1-6]_*.md")):
            replacement.extend(convert(chapter.read_text(encoding="utf-8")))

        # Splice by offset so everything outside the range -- field codes, section
        # properties, front matter, the reference list -- passes through untouched.
        cut_from = matches[start].start()
        cut_to = matches[end].start()
        rebuilt = xml[:cut_from] + "".join(replacement) + xml[cut_to:]
        document.write_text(rebuilt, encoding="utf-8")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            args.output.unlink()
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(work.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(work))

        print(f"şablon paragrafı   : {len(paragraphs)}")
        print(f"değiştirilen gövde : #{start}–#{end} ({end - start} paragraf)")
        print(f"yazılan paragraf   : {len(replacement)}")
        print(f"çıktı              : {args.output}")

        if args.pdf:
            for candidate in (
                "/Applications/LibreOffice.app/Contents/MacOS/soffice",
                "soffice",
                "libreoffice",
            ):
                try:
                    subprocess.run(
                        [
                            candidate,
                            "--headless",
                            "--convert-to",
                            "pdf",
                            "--outdir",
                            str(args.output.parent),
                            str(args.output),
                        ],
                        check=True,
                        capture_output=True,
                        timeout=300,
                    )
                    print(f"kontrol PDF'i      : {args.output.with_suffix('.pdf')}")
                    break
                except (OSError, subprocess.SubprocessError):
                    continue
            else:
                print("LibreOffice bulunamadı; PDF kontrolü elle yapılmalı.")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
