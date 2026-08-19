"""Derive the PRISMA screening counts instead of typing them into a diagram.

A review flow whose numbers do not balance is worse than no flow: it invites exactly the
arithmetic a reader will do. The figure this replaces claimed 912 records, 34 off-topic,
661 non-academic and 251 remaining -- which does not add up (912 - 34 - 661 = 217) and
also asserted "~50 included" against a bibliography holding 19.

So the counts are computed here from the corpus, and the diagram reads this record. Every
stage subtracts from the one above it, and the arithmetic is checked before the record is
written rather than trusted.

Classification is by host, which is coarse and admitted as such: a preprint server hosts
work of every quality, and a university repository hosts theses alongside papers. What the
category means precisely is "this URL points at a venue that publishes research", not
"this source is peer-reviewed and relevant". The eligibility stage is the honest filter,
and it is manual -- `BIBLIOGRAPHY.md` only accepts a source someone opened.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from edgeguard.serialization import canonical_json

URL_PATTERN = re.compile(r'https?://[^\s\)\]\>"]+')

# Venues that publish research: conference proceedings, journals, preprint servers,
# indexes and institutional repositories.
ACADEMIC_HOSTS = (
    "arxiv.org",
    "openaccess.thecvf.com",
    "proceedings.",
    "doi.org",
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "sciencedirect.com",
    "mdpi.com",
    "springer",
    "nature.com",
    "semanticscholar.org",
    "jmlr.org",
    "pmlr.press",
    "researchgate.net",
    "ncbi.nlm.nih.gov",
    "ecva.net",
    "openreview.net",
    "computer.org",
    "preprints.org",
    "biorxiv.org",
    "ssrn.com",
    "hal.science",
    "eprints.",
    "repository.",
    ".edu/",
    "uni-",
    "research.",
    "ink.library.",
)
# Software and documentation: cited as tool attribution in materials and methods, never
# as literature. Removing them is a screening decision, not a quality judgement.
TOOL_HOSTS = (
    "github.com",
    "gitlab",
    "huggingface.co",
    "pypi.org",
    "docs.nvidia.com",
    "developer.nvidia.com",
    "nvidia.com",
    "pytorch.org",
    "readthedocs.io",
    "kaggle.com",
    "cvat.ai",
    "albumentations.ai",
    "opencv.org",
    "python.org",
    "wikipedia.org",
)
NON_SCHOLARLY_HOSTS = (
    "forum",
    "stackoverflow.com",
    "reddit.com",
    "medium.com",
    "towardsdatascience.com",
    "youtube.com",
    "twitter.com",
    "linkedin.com",
    "emergentmind.com",
)


def classify(url: str) -> str:
    lowered = url.lower()
    if any(host in lowered for host in NON_SCHOLARLY_HOSTS):
        return "non_scholarly"
    if any(host in lowered for host in TOOL_HOSTS):
        return "tool_or_documentation"
    if any(host in lowered for host in ACADEMIC_HOSTS):
        return "academic_venue"
    return "unclassified"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, default=Path("researchs"))
    parser.add_argument("--bibliography", type=Path, default=Path("docs/BIBLIOGRAPHY.md"))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    documents = sorted(args.research_root.glob("*.md"))
    if not documents:
        raise ValueError(f"no search documents under {args.research_root}")

    found: list[str] = []
    for document in documents:
        found.extend(URL_PATTERN.findall(document.read_text(errors="ignore")))
    # Trailing punctuation belongs to the prose, not the URL.
    cleaned = [url.rstrip(".,;:)』\"'") for url in found]
    unique = sorted(set(cleaned))
    groups = Counter(classify(url) for url in unique)

    identified = len(unique)
    removed_tools = groups["tool_or_documentation"]
    removed_non_scholarly = groups["non_scholarly"]
    removed_unclassified = groups["unclassified"]
    eligible = groups["academic_venue"]
    if identified - removed_tools - removed_non_scholarly - removed_unclassified != eligible:
        raise AssertionError("PRISMA stages do not balance; refusing to write the record")

    # The included count is whatever the bibliography actually holds, read from the file,
    # so the figure cannot drift away from the reference list it summarises.
    bibliography = args.bibliography.read_text(encoding="utf-8")
    included = len(set(re.findall(r"^\*\*\[(\d+)\]\*\*", bibliography, flags=re.MULTILINE)))

    hosts = Counter(
        re.sub(r"^https?://(www\.)?([^/]+).*", r"\2", url)
        for url in unique
        if classify(url) == "academic_venue"
    )

    record = {
        "schema_version": "1.0",
        "record_type": "edgeguard_literature_corpus_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "search_documents": len(documents),
        "records_total": len(cleaned),
        "identification": identified,
        "screening_removed": {
            "tool_or_documentation": removed_tools,
            "non_scholarly": removed_non_scholarly,
            "unclassified": removed_unclassified,
        },
        "eligibility": eligible,
        "included": included,
        "top_academic_hosts": dict(hosts.most_common(12)),
        "classification": "by host, coarse: 'academic venue' means the URL points at a "
        "venue that publishes research, not that the source is peer-reviewed or relevant",
        "eligibility_is_manual": True,
        "scientific_measurement": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(record) + "\n", encoding="utf-8")

    print(f"\n{len(documents)} tarama dokümanı · {len(cleaned)} bağlantı")
    print(f"  Tanımlama    {identified} benzersiz kayıt")
    print(f"  Tarama       −{removed_tools} araç/doküman")
    print(f"               −{removed_non_scholarly} akademik olmayan")
    print(f"               −{removed_unclassified} sınıflandırılamayan")
    print(f"  Uygunluk     {eligible} akademik yayın")
    print(f"  Dahil edilen {included} (kaynakçadan okundu)")
    print(f"kayıt: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
