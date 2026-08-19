"""Render one of the delivery documents to PDF.

The delivery specification asks for `.pdf`, and the machine that assembles this package
has no office suite, so the PDF is built directly. The subset of Markdown handled here is
exactly the subset the three documents use -- headings, paragraphs, bullet and numbered
lists, fenced code, and pipe tables -- because a general Markdown engine would be a much
larger dependency for no benefit.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# reportlab is needed only to produce the delivery PDFs, so it is an optional extra
# rather than a project dependency. Importing it lazily lets this file load -- and say
# what to install -- in an environment that does not have it.
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_MISSING = ""
except ModuleNotFoundError as exc:  # pragma: no cover -- environment-dependent
    REPORTLAB_MISSING = (
        f"{exc.name} kurulu değil. Teslim belgelerini üretmek için:\n"
        '    python -m pip install -e ".[delivery]"'
    )

# DejaVu carries the Turkish letters that the built-in Type 1 fonts lack (ğ, ş, ı, İ).
FONT_DIRS = [
    Path("/System/Library/Fonts/Supplemental"),
    Path("/Library/Fonts"),
    Path("/usr/share/fonts/truetype/dejavu"),
]


def _register_fonts() -> tuple[str, str, str]:
    """Regular and bold must be different files, or emphasis renders invisibly."""
    candidates = [
        ("Arial.ttf", "Arial Bold.ttf", "Courier New.ttf"),
        ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSansMono.ttf"),
        ("Verdana.ttf", "Verdana Bold.ttf", "Courier New.ttf"),
    ]
    for base in FONT_DIRS:
        for regular, bold, mono in candidates:
            if (base / regular).exists() and (base / bold).exists():
                pdfmetrics.registerFont(TTFont("Body", str(base / regular)))
                pdfmetrics.registerFont(TTFont("BodyBold", str(base / bold)))
                if (base / mono).exists():
                    pdfmetrics.registerFont(TTFont("Mono", str(base / mono)))
                    return "Body", "BodyBold", "Mono"
                return "Body", "BodyBold", "Courier"
    return "Helvetica", "Helvetica-Bold", "Courier"


BODY = BOLD = MONO = ""  # filled by build(), once reportlab is known present


def inline(text: str) -> str:
    """Markdown emphasis to reportlab markup, escaping XML first."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", rf'<font name="{BOLD}">\1</font>', text)
    text = re.sub(r"`(.+?)`", rf'<font name="{MONO}" size="9">\1</font>', text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", text)
    return text


def build(md: str, out: Path, title: str) -> None:
    global BODY, BOLD, MONO
    if REPORTLAB_MISSING:
        raise SystemExit(REPORTLAB_MISSING)
    if not BODY:
        BODY, BOLD, MONO = _register_fonts()
    sheet = getSampleStyleSheet()
    body = ParagraphStyle(
        "body",
        parent=sheet["Normal"],
        fontName=BODY,
        fontSize=10.5,
        leading=15.5,
        spaceAfter=7,
        alignment=TA_LEFT,
    )
    h1 = ParagraphStyle(
        "h1",
        parent=body,
        fontName=BOLD,
        fontSize=17,
        leading=22,
        spaceBefore=16,
        spaceAfter=10,
        textColor=colors.HexColor("#12305a"),
    )
    h2 = ParagraphStyle(
        "h2",
        parent=body,
        fontName=BOLD,
        fontSize=13,
        leading=18,
        spaceBefore=14,
        spaceAfter=7,
        textColor=colors.HexColor("#1d4d8f"),
    )
    h3 = ParagraphStyle(
        "h3",
        parent=body,
        fontName=BOLD,
        fontSize=11.5,
        leading=16,
        spaceBefore=11,
        spaceAfter=5,
    )
    code = ParagraphStyle(
        "code",
        parent=body,
        fontName=MONO,
        fontSize=8.7,
        leading=12,
        leftIndent=10,
        backColor=colors.HexColor("#f2f4f7"),
        borderPadding=6,
        spaceBefore=5,
        spaceAfter=9,
    )
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=16, bulletIndent=5, spaceAfter=4)

    flow: list = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if line.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            text = "<br/>".join(
                b.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace(" ", "&nbsp;")
                for b in buf
            )
            flow.append(Paragraph(text, code))
            continue

        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= {"-", ":", " "} and c for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                width = max(len(r) for r in rows)
                cell = ParagraphStyle("cell", parent=body, fontSize=9, leading=12, spaceAfter=0)
                head = ParagraphStyle("head", parent=cell, fontName=BOLD)
                data = [
                    [
                        Paragraph(inline(c), head if n == 0 else cell)
                        for c in (r + [""] * (width - len(r)))
                    ]
                    for n, r in enumerate(rows)
                ]
                avail = A4[0] - 4.2 * cm
                table = Table(data, colWidths=[avail / width] * width, repeatRows=1)
                table.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa5b1")),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6ecf4")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                flow.append(Spacer(1, 4))
                flow.append(table)
                flow.append(Spacer(1, 10))
            continue

        if not line.strip():
            i += 1
            continue
        if line.startswith("---"):
            i += 1
            continue
        if line.startswith("# "):
            flow.append(Paragraph(inline(line[2:]), h1))
        elif line.startswith("## "):
            flow.append(Paragraph(inline(line[3:]), h2))
        elif line.startswith("### "):
            flow.append(Paragraph(inline(line[4:]), h3))
        elif re.match(r"^\s*([-*]|\d+\.)\s+", line):
            marker = re.match(r"^\s*(\d+)\.\s+", line)
            buf = [re.sub(r"^\s*([-*]|\d+\.)\s+", "", line).strip()]
            i += 1
            # An item continues onto indented lines; those belong to the same bullet.
            while i < len(lines):
                nxt = lines[i].rstrip()
                if (
                    not nxt.strip()
                    or nxt.startswith(("#", "|", "```", "---"))
                    or re.match(r"^\s*([-*]|\d+\.)\s+", nxt)
                    or not nxt.startswith(" ")
                ):
                    break
                buf.append(nxt.strip())
                i += 1
            flow.append(
                Paragraph(
                    inline(" ".join(buf)),
                    bullet,
                    bulletText=f"{marker.group(1)}." if marker else "\u2022",
                )
            )
            continue
        else:
            # Wrapped source lines are one paragraph; joining them first is what lets
            # emphasis that straddles a line break match at all.
            buf = [line.strip()]
            i += 1
            while i < len(lines):
                nxt = lines[i].rstrip()
                if (
                    not nxt.strip()
                    or nxt.startswith(("#", "|", "```", "---"))
                    or re.match(r"^\s*([-*]|\d+\.)\s+", nxt)
                ):
                    break
                buf.append(nxt.strip())
                i += 1
            flow.append(Paragraph(inline(" ".join(buf)), body))
            continue
        i += 1

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(BODY, 8)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawString(2.1 * cm, 1.3 * cm, title)
        canvas.drawRightString(A4[0] - 2.1 * cm, 1.3 * cm, f"{doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=2.1 * cm,
        rightMargin=2.1 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title=title,
        author="EdgeGuard-Road",
    ).build(flow, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    src, dst, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    build(src.read_text(encoding="utf-8"), dst, name)
    print(f"{dst}  ({dst.stat().st_size // 1024} KB)")
