"""Check that every four-decimal number in the thesis documents exists in a record.

Four separate inconsistencies were found by hand in one session: a section leading with a
retracted correlation while the section above it reported the retraction, a limits table
claiming "measured" under a heading saying "not measured", a caption reading 1.52 beside
its own table reading 1.53, and a PRISMA flow whose stages did not subtract to their own
total. Each was found by someone happening to read two places at once. That does not
scale, and a defence is exactly the situation where someone reads two places at once.

So the class gets a check rather than more care. A number written to four decimals in
these documents is a measurement -- nobody writes 0.9614 by accident -- so every such
number must appear in a measurement record. Ones that do not are either stale, mistyped,
or derived; the first two are defects and the third needs a stated derivation, so all
three are worth surfacing.

What this deliberately does **not** do is decide which record a number came from. Matching
prose to fields would need to understand the prose. Presence in the corpus is a weaker
check that costs nothing to maintain and still catches every stale figure.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# Four or more decimals: precise enough that it is a measurement rather than a round
# number someone chose. Turkish decimal comma and English point are both accepted since
# the documents are Turkish and the records are JSON.
REPORTED = re.compile(r"(?<![\w.,])(\d+)[.,](\d{4,})(?![\w])")
# arXiv identifiers have the shape of a four-decimal number and are not measurements.
IDENTIFIER = re.compile(r"(arxiv|arXiv|doi|DOI)[:\s]*\S*$")

# Differences between two records are legitimate but are not themselves in any record,
# and accepting every pairwise difference would weaken the check until it caught
# nothing. Each one is listed instead, with the derivation it must have -- so a derived
# figure costs one line here and cannot be added by accident.
DERIVED: dict[float, str] = {
    0.0085: (
        "SegFormer-B0 minus DDRNet-23-slim dataset mIoU, 0.6934 - 0.6850; also the mean "
        "of the per-class differences, which is the same quantity since mIoU is the "
        "class mean"
    ),
    0.008491: (
        "the same difference before rounding, quoted in THESIS_EVIDENCE to show that "
        "0.0084 -- what subtracting the rounded figures gives -- is the wrong value"
    ),
}


def _walk(value: Any, into: set[float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        into.add(round(float(value), 4))
    elif isinstance(value, dict):
        for item in value.values():
            _walk(item, into)
    elif isinstance(value, list):
        for item in value:
            _walk(item, into)


def known_values(records: Path) -> set[float]:
    """Every numeric value in every record, plus percentages and ratios derived from them.

    Documents legitimately state a measurement as a percentage (0.5007 -> 50.1%) or as a
    ratio between two records (0.808 / 0.630 -> 1.28x). Those are the same measurement in
    another unit, so they belong in the accepted set rather than in the report.
    """
    values: set[float] = set()
    for path in sorted(records.rglob("*.json")):
        if path.name.startswith("._"):
            continue
        try:
            _walk(json.loads(path.read_text(encoding="utf-8")), values)
        except (OSError, ValueError):
            continue
    scaled = {round(value * 100, 4) for value in values}
    ratios = {
        round(left / right, 4)
        for left in values
        for right in values
        if right and abs(right) > 1e-9 and 0.01 < abs(left / right) < 1000
    }
    return values | scaled | ratios


def reported_numbers(document: Path) -> list[tuple[int, str, float]]:
    found: list[tuple[int, str, float]] = []
    for number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), start=1):
        for match in REPORTED.finditer(line):
            if IDENTIFIER.search(line[: match.start()]):
                continue
            found.append((number, match.group(0), float(f"{match.group(1)}.{match.group(2)}")))
    return found


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("reports/measurements"))
    parser.add_argument(
        "--document",
        type=Path,
        action="append",
        default=None,
        help="repeatable; defaults to the documents that carry measured claims",
    )
    parser.add_argument("--tolerance", type=float, default=5e-4)
    return parser


def main() -> int:
    args = _parser().parse_args()
    documents = args.document or [
        Path("docs/THESIS_EVIDENCE.md"),
        Path("docs/THESIS_SECTIONS.md"),
        Path("docs/LITERATURE_PLAN.md"),
    ]
    known = known_values(args.records) | set(DERIVED)
    if not known:
        raise ValueError(f"no records under {args.records}")

    unmatched: list[tuple[Path, int, str]] = []
    checked = 0
    for document in documents:
        if not document.is_file():
            continue
        for line, text, value in reported_numbers(document):
            checked += 1
            if not any(abs(value - candidate) <= args.tolerance for candidate in known):
                unmatched.append((document, line, text))

    print(
        f"{checked} rapor edilen sayı · {len(known):,} bilinen değer "
        f"({len(DERIVED)} türetilmiş, gerekçesi yazılı)"
    )
    if not unmatched:
        print("eşleşmeyen yok")
        return 0
    print(f"\n{len(unmatched)} sayı hiçbir kayıtta bulunamadı:\n")
    for document, line, text in unmatched:
        print(f"  {document}:{line}  {text}")
    print(
        "\nHer biri ya bayat, ya yanlış yazılmış, ya da türetilmiş bir değerdir. "
        "Üçüncüsüyse türetimi metinde yazılmalıdır."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
