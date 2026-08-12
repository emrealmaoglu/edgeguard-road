"""Combine real measured per-model training results with real measured per-class
training-data frequency, to support model/class-scope decisions with evidence instead
of guesswork.

Every number here is either parsed directly out of a real Colab training log (mmengine's
own "per class results" table and `Iter(val)`/`Iter(train)` lines) or read from an
existing real per-class frequency artifact this project already produces
(`write_train_fit_statistics`'s `class_weights.json` for Cityscapes,
`audit_training_dataset`'s `summary.json` for IDD20K). Nothing here samples, estimates,
or invents a number for a class/run that was not actually observed; a missing input is
reported as unavailable, never guessed or defaulted to zero.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from edgeguard.rescue.dataset import CITYSCAPES_CLASSES
from edgeguard.serialization import canonical_json

DEFAULT_NEAR_ABSENT_PIXEL_RATIO_THRESHOLD = 0.005

_SUMMARY_LINE_RE = re.compile(
    r'^\{.*"record_type":"semantic_training_summary".*\}\s*$', re.MULTILINE
)
_PER_CLASS_HEADER_RE = re.compile(r"per class results:")
_TABLE_ROW_RE = re.compile(
    r"^\|\s*([A-Za-z][A-Za-z ]*?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*$", re.MULTILINE
)
_ITER_VAL_RE = re.compile(
    r"Iter\(val\)\s*\[\s*\d+/\d+\]\s+aAcc:\s*([\d.]+)\s+mIoU:\s*([\d.]+)\s+mAcc:\s*([\d.]+)"
)
_ITER_TRAIN_TIME_RE = re.compile(r"Iter\(train\)\s*\[\s*\d+/\d+\][^\n]*?\btime:\s*([\d.]+)")


@dataclass
class TrainingRunResult:
    """One real, measured training run's results, parsed from a real Colab log."""

    model: str
    stage: str
    per_class_iou: dict[str, float]
    per_class_acc: dict[str, float]
    aacc: float
    miou: float
    macc: float
    elapsed_seconds: float | None
    iter_seconds: float | None
    source: str
    parse_warnings: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "stage": self.stage,
            "per_class_iou_measured": dict(sorted(self.per_class_iou.items())),
            "per_class_acc_measured": dict(sorted(self.per_class_acc.items())),
            "aAcc_measured": self.aacc,
            "mIoU_measured": self.miou,
            "mAcc_measured": self.macc,
            "elapsed_seconds_measured": self.elapsed_seconds,
            "iter_seconds_measured": self.iter_seconds,
            "source": self.source,
            "parse_warnings": self.parse_warnings,
        }


def parse_training_log(log_text: str, *, source: str = "<text>") -> list[TrainingRunResult]:
    """Extract every real training run recorded in one real Colab log capture.

    Each run in a Colab log ends with a `semantic_training_summary` JSON line; this
    walks the log in order, treating the text between consecutive summary lines as one
    run's segment, and extracts that run's per-class table, aggregate `Iter(val)` line,
    and a representative per-iteration wall time from its last `Iter(train)` line.
    """
    results: list[TrainingRunResult] = []
    chunk_start = 0
    for summary_match in _SUMMARY_LINE_RE.finditer(log_text):
        chunk = log_text[chunk_start : summary_match.start()]
        chunk_start = summary_match.end()
        try:
            summary = json.loads(summary_match.group(0))
        except json.JSONDecodeError:
            continue
        warnings: list[str] = []

        header_positions = [m.start() for m in _PER_CLASS_HEADER_RE.finditer(chunk)]
        per_class_iou: dict[str, float] = {}
        per_class_acc: dict[str, float] = {}
        if header_positions:
            table_region = chunk[header_positions[-1] :]
            for row in _TABLE_ROW_RE.finditer(table_region):
                name = row.group(1).strip()
                per_class_iou[name] = float(row.group(2))
                per_class_acc[name] = float(row.group(3))
            if not per_class_iou:
                warnings.append("found 'per class results:' marker but no table rows parsed")
        else:
            warnings.append("no 'per class results:' table found for this run")

        iter_val_matches = list(_ITER_VAL_RE.finditer(chunk))
        if iter_val_matches:
            last_val = iter_val_matches[-1]
            aacc, miou, macc = (
                float(last_val.group(1)),
                float(last_val.group(2)),
                float(last_val.group(3)),
            )
        else:
            aacc = miou = macc = float("nan")
            warnings.append("no aggregate Iter(val) aAcc/mIoU/mAcc line found")

        iter_train_matches = list(_ITER_TRAIN_TIME_RE.finditer(chunk))
        iter_seconds = float(iter_train_matches[-1].group(1)) if iter_train_matches else None
        if iter_seconds is None:
            warnings.append("no Iter(train) timing found; per-iteration speed unavailable")

        elapsed = summary.get("elapsed_seconds")
        results.append(
            TrainingRunResult(
                model=str(summary.get("model", "unknown")),
                stage=str(summary.get("stage", "unknown")),
                per_class_iou=per_class_iou,
                per_class_acc=per_class_acc,
                aacc=aacc,
                miou=miou,
                macc=macc,
                elapsed_seconds=float(elapsed) if isinstance(elapsed, int | float) else None,
                iter_seconds=iter_seconds,
                source=source,
                parse_warnings=warnings,
            )
        )
    return results


def load_real_class_frequency(
    class_weights_path: Path, idd20k_summary_path: Path | None = None
) -> dict[str, Any]:
    """Load the real, already-measured per-class pixel frequency for the actual
    train_fit mix. Never estimates or samples; a missing/malformed source is reported
    honestly in `sources`, and simply contributes zero to the combined count."""
    sources: dict[str, str] = {}
    combined_pixels: dict[str, int] = dict.fromkeys(CITYSCAPES_CLASSES, 0)

    if class_weights_path.is_file():
        payload = json.loads(class_weights_path.read_text(encoding="utf-8"))
        counts = payload.get("pixel_counts")
        if isinstance(counts, list) and len(counts) == len(CITYSCAPES_CLASSES):
            for name, value in zip(CITYSCAPES_CLASSES, counts, strict=True):
                combined_pixels[name] += int(value)
            sources["cityscapes"] = str(class_weights_path)
        else:
            sources["cityscapes"] = "not_available: malformed class_weights.json"
    else:
        sources["cityscapes"] = "not_available: file not found"

    if idd20k_summary_path is not None and idd20k_summary_path.is_file():
        payload = json.loads(idd20k_summary_path.read_text(encoding="utf-8"))
        counts = payload.get("class_pixel_counts")
        if isinstance(counts, list) and len(counts) == len(CITYSCAPES_CLASSES):
            for name, value in zip(CITYSCAPES_CLASSES, counts, strict=True):
                combined_pixels[name] += int(value)
            sources["idd20k"] = str(idd20k_summary_path)
        else:
            sources["idd20k"] = "not_available: malformed summary.json"
    elif idd20k_summary_path is not None:
        sources["idd20k"] = "not_available: file not found"
    else:
        sources["idd20k"] = "not_available: no path provided"

    return {
        "sources": sources,
        "pixel_counts_measured": combined_pixels,
        "total_pixels_measured": sum(combined_pixels.values()),
    }


def flag_near_absent_classes(
    frequency: dict[str, Any],
    *,
    pixel_ratio_threshold: float = DEFAULT_NEAR_ABSENT_PIXEL_RATIO_THRESHOLD,
) -> list[str]:
    """Flag classes below a fixed, never per-model-tuned real pixel-share threshold.

    The threshold is a single global constant applied identically to every class and
    every model -- never adjusted to make a particular model's projected mIoU look
    better, which would defeat the entire point of grounding this in real evidence.
    """
    total = frequency["total_pixels_measured"]
    if not total:
        return []
    return [
        name
        for name, count in frequency["pixel_counts_measured"].items()
        if (count / total) < pixel_ratio_threshold
    ]


def project_reduced_class_miou(
    result: TrainingRunResult, excluded_classes: list[str]
) -> float | None:
    """Recompute mIoU averaging only over already-measured per-class IoU values for
    classes not in `excluded_classes`. Never invents a number for a class that was not
    parsed from the real log; returns None if nothing is left to average."""
    included = {
        name: iou for name, iou in result.per_class_iou.items() if name not in excluded_classes
    }
    if not included:
        return None
    return sum(included.values()) / len(included)


def build_analysis_report(
    log_texts: dict[str, str],
    *,
    class_weights_path: Path,
    idd20k_summary_path: Path | None = None,
    pixel_ratio_threshold: float = DEFAULT_NEAR_ABSENT_PIXEL_RATIO_THRESHOLD,
) -> dict[str, Any]:
    """Orchestrate: parse every real log, load real frequency, flag near-absent
    classes, and project a reduced-class mIoU using only already-measured numbers."""
    all_results: list[TrainingRunResult] = []
    for source, text in log_texts.items():
        all_results.extend(parse_training_log(text, source=source))

    frequency = load_real_class_frequency(class_weights_path, idd20k_summary_path)
    excluded = sorted(
        flag_near_absent_classes(frequency, pixel_ratio_threshold=pixel_ratio_threshold)
    )

    runs = []
    for result in all_results:
        reduced = project_reduced_class_miou(result, excluded)
        runs.append(
            {
                **result.to_jsonable(),
                "mIoU_supported_classes_only_projected": reduced,
                "excluded_classes_for_projection": excluded,
            }
        )

    return {
        "schema_version": "1.0",
        "record_type": "training_results_class_analysis",
        "pixel_ratio_threshold": pixel_ratio_threshold,
        "class_frequency": frequency,
        "near_absent_classes": excluded,
        "run_count": len(runs),
        "runs": runs,
        "sampling": (
            "exhaustive: every real per-class IoU/frequency number used verbatim, "
            "none sampled, estimated, or fabricated"
        ),
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Training results — class-scope analysis",
        "",
        f"- Runs parsed: {report['run_count']}",
        f"- Near-absent-class pixel-ratio threshold: {report['pixel_ratio_threshold']:.3%}",
        "- Near-absent classes (real, measured): "
        f"{', '.join(report['near_absent_classes']) or 'none'}",
        f"- Cityscapes frequency source: {report['class_frequency']['sources'].get('cityscapes')}",
        f"- IDD20K frequency source: {report['class_frequency']['sources'].get('idd20k')}",
        "",
        "| Model | Stage | mIoU (19-class, measured) | mIoU (supported-only, projected) | "
        "sec/iter (measured) | Source |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in report["runs"]:
        reduced = run["mIoU_supported_classes_only_projected"]
        reduced_display = f"{reduced:.2f}" if reduced is not None else "n/a"
        iter_seconds = run["iter_seconds_measured"]
        iter_display = f"{iter_seconds:.2f}" if iter_seconds is not None else "n/a"
        lines.append(
            f"| {run['model']} | {run['stage']} | {run['mIoU_measured']:.2f} | "
            f"{reduced_display} | {iter_display} | {run['source']} |"
        )
    lines.append("")
    for run in report["runs"]:
        if run["parse_warnings"]:
            lines.append(
                f"- **{run['model']}/{run['stage']}** parse warnings: "
                f"{'; '.join(run['parse_warnings'])}"
            )
    return "\n".join(lines) + "\n"


def write_analysis_report(report: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "training_analysis.json").write_text(canonical_json(report), encoding="utf-8")
    (output_root / "training_analysis.md").write_text(
        render_markdown_report(report), encoding="utf-8"
    )
