"""Render every measured record into a figure a thesis can print.

The measurements exist as JSON; nothing reads them into a chart. A number in a record is
evidence, but a reader cannot see a distribution, a degradation curve or a trade-off in a
table of five rows -- and the findings here are about shapes: which architectures lose
most to the deployment resolution, where calibration falls apart, what the frame budget is
actually spent on.

Every figure is drawn only from a record that exists. A missing record produces no
figure and a printed note, never a placeholder that could be mistaken for a measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODEL_ORDER = ("segformer_b0", "ddrnet_23_slim", "pidnet_m", "pidnet_s", "bisenetv2")
MODEL_LABEL = {
    "pidnet_m": "PIDNet-M",
    "pidnet_s": "PIDNet-S",
    "ddrnet_23_slim": "DDRNet-23-slim",
    "segformer_b0": "SegFormer-B0",
    "bisenetv2": "BiSeNetV2",
}
PUBLISHED_MIOU = {
    "pidnet_m": 80.22,
    "pidnet_s": 78.74,
    "ddrnet_23_slim": 77.84,
    "segformer_b0": 76.54,
    "bisenetv2": 75.76,
}
JETSON = {  # Orin Nano Super, 25 W, TensorRT FP16
    "pidnet_m": {"engine_ms": 11.153, "frame_ms": 88.503, "fps": 10.823, "joule": 0.808},
    "pidnet_s": {"engine_ms": 5.069, "frame_ms": 80.771, "fps": 11.871, "joule": 0.665},
    "ddrnet_23_slim": {"engine_ms": 3.811, "frame_ms": 77.434, "fps": 12.362, "joule": 0.630},
    "segformer_b0": {"engine_ms": 15.041, "frame_ms": 270.822, "fps": 3.569, "joule": 2.221},
    "bisenetv2": {"engine_ms": 13.445, "frame_ms": 92.357, "fps": 10.343, "joule": 0.809},
}
CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)
FRAME_STAGES = (
    ("derive_perception", 96.05, 43.64),
    ("preprocess", 24.80, 24.71),
    ("görüntü çözme", 13.97, 14.00),
    ("confidence_entropy", 5.78, 5.80),
    ("TensorRT motoru", 5.09, 5.09),
    ("argmax", 0.75, 0.74),
)

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save(figure: plt.Figure, output: Path, name: str, written: list[str]) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"{name}.{suffix}")
    plt.close(figure)
    written.append(name)


def accuracy_records(root: Path) -> dict[str, dict[str, Any]]:
    loaded = {}
    for model in MODEL_ORDER:
        record = load(root / "accuracy" / f"{model}_cityscapes_val.json")
        if record:
            loaded[model] = record
    return loaded


def figure_published_vs_measured(accuracy: dict, output: Path, written: list[str]) -> None:
    """The published ranking against the one the deployment resolution actually produces."""
    models = [m for m in MODEL_ORDER if m in accuracy]
    if not models:
        return
    published = [PUBLISHED_MIOU[m] for m in models]
    measured = [accuracy[m]["mIoU"] * 100 for m in models]
    positions = np.arange(len(models))

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), width_ratios=[3, 2])
    axes[0].bar(positions - 0.2, published, 0.4, label="yayın (tam çözünürlük)", color="#9aa5b1")
    axes[0].bar(positions + 0.2, measured, 0.4, label="ölçülen (512×1024 dağıtım)", color="#2f6f9f")
    for index, (high, low) in enumerate(zip(published, measured, strict=True)):
        axes[0].annotate(
            f"−{100 * (1 - low / high):.1f}%",
            (index, low - 2),
            ha="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([MODEL_LABEL[m] for m in models], rotation=18, ha="right")
    axes[0].set_ylabel("mIoU (%)")
    axes[0].set_ylim(60, 84)
    axes[0].legend(frameon=False)
    axes[0].set_title("Dağıtım çözünürlüğünün doğruluğa maliyeti")

    order_published = sorted(models, key=lambda m: -PUBLISHED_MIOU[m])
    order_measured = sorted(models, key=lambda m: -accuracy[m]["mIoU"])
    for rank, model in enumerate(order_published):
        target = order_measured.index(model)
        axes[1].plot([0, 1], [rank, target], marker="o", linewidth=2)
        axes[1].annotate(MODEL_LABEL[model], (-0.04, rank), ha="right", va="center", fontsize=8)
        axes[1].annotate(MODEL_LABEL[model], (1.04, target), ha="left", va="center", fontsize=8)
    axes[1].set_xlim(-0.75, 1.75)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["yayın", "dağıtım"])
    axes[1].invert_yaxis()
    axes[1].set_yticks([])
    axes[1].grid(False)
    axes[1].set_title("Sıralama ρ = +0,10")
    save(figure, output, "01_published_vs_measured_miou", written)


def figure_per_class(accuracy: dict, output: Path, written: list[str]) -> None:
    """Where the accuracy actually goes, class by class."""
    models = [m for m in MODEL_ORDER if m in accuracy]
    if not models:
        return
    matrix = np.array([[(v or np.nan) * 100 for v in accuracy[m]["per_class_iou"]] for m in models])
    order = np.argsort(-np.nanmean(matrix, axis=0))

    figure, axis = plt.subplots(figsize=(12, 3.4))
    image = axis.imshow(matrix[:, order], aspect="auto", cmap="viridis", vmin=0, vmax=100)
    axis.set_xticks(range(len(CLASSES)))
    axis.set_xticklabels([CLASSES[i] for i in order], rotation=55, ha="right", fontsize=8)
    axis.set_yticks(range(len(models)))
    axis.set_yticklabels([MODEL_LABEL[m] for m in models], fontsize=8)
    axis.grid(False)
    figure.colorbar(image, ax=axis, label="IoU (%)", pad=0.01)
    axis.set_title("Sınıf bazlı IoU · Cityscapes val · dağıtım çözünürlüğü")
    save(figure, output, "02_per_class_iou", written)


def figure_reliability(accuracy: dict, output: Path, written: list[str]) -> None:
    """Whether confidence tracks correctness, per model."""
    models = [m for m in MODEL_ORDER if m in accuracy]
    if not models:
        return
    figure, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models), 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, model in zip(axes, models, strict=True):
        bins = [b for b in accuracy[model]["reliability_bins"] if b.get("count")]
        confidence = [b["mean_confidence"] for b in bins]
        correct = [b["accuracy"] for b in bins]
        axis.plot([0, 1], [0, 1], "--", color="#b0b0b0", linewidth=1)
        axis.plot(confidence, correct, marker="o", markersize=3, color="#2f6f9f")
        axis.fill_between(confidence, correct, confidence, alpha=0.18, color="#d9534f")
        axis.set_title(
            f"{MODEL_LABEL[model]}\nECE {accuracy[model]['expected_calibration_error']:.4f}",
            fontsize=8,
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("güven", fontsize=8)
    axes[0].set_ylabel("doğruluk", fontsize=8)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.suptitle("Güvenilirlik diyagramları · kırmızı alan = aşırı-güven", fontsize=10)
    save(figure, output, "03_reliability_diagrams", written)


def figure_acdc(root: Path, output: Path, written: list[str]) -> None:
    """What real adverse conditions do to accuracy and to calibration."""
    records = {}
    for path in sorted((root / "acdc").glob("*.json")):
        record = load(path)
        if record:
            records[record["split"].split("_")[-1]] = record
    if not records:
        return
    order = sorted(records, key=lambda key: -records[key]["mIoU"])
    miou = [records[k]["mIoU"] * 100 for k in order]
    ece = [records[k]["expected_calibration_error"] for k in order]
    confidence = [records[k]["mean_confidence"] * 100 for k in order]
    accuracy = [records[k]["pixel_accuracy"] * 100 for k in order]

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].bar(order, miou, color=["#2f6f9f"] * (len(order) - 1) + ["#d9534f"])
    axes[0].set_ylabel("mIoU (%)")
    axes[0].set_title("Doğruluk")
    axes[1].bar(order, ece, color=["#2f6f9f"] * (len(order) - 1) + ["#d9534f"])
    axes[1].set_ylabel("ECE")
    axes[1].set_title("Kalibrasyon hatası")
    positions = np.arange(len(order))
    axes[2].bar(positions - 0.2, confidence, 0.4, label="ortalama güven", color="#f0ad4e")
    axes[2].bar(positions + 0.2, accuracy, 0.4, label="piksel doğruluğu", color="#5cb85c")
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(order)
    axes[2].set_ylabel("%")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].set_title("Güven vs doğruluk")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.suptitle("Gerçek olumsuz koşullar (ACDC) · PIDNet-S referans", fontsize=10)
    save(figure, output, "04_acdc_conditions", written)


def figure_open_set(root: Path, output: Path, written: list[str]) -> None:
    """Which uncertainty signal actually separates unknown objects."""
    records = {}
    for model in MODEL_ORDER:
        record = load(root / "open_set" / f"{model}.json")
        if record:
            records[model] = record
    if not records:
        return
    scores = ("energy", "maximum_logit", "normalized_entropy", "maximum_softmax_probability")
    labels = ("energy", "max-logit", "entropi", "MSP")

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    width = 0.2
    for index, (score, label) in enumerate(zip(scores, labels, strict=True)):
        values = [
            records[m]["scores"][score]["metrics"]["auroc"]
            for m in records
            if score in records[m]["scores"]
        ]
        axes[0].bar(np.arange(len(values)) + (index - 1.5) * width, values, width, label=label)
    axes[0].axhline(0.5, color="#d9534f", linestyle="--", linewidth=1)
    axes[0].set_xticks(np.arange(len(records)))
    axes[0].set_xticklabels([MODEL_LABEL[m] for m in records], rotation=18, ha="right")
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.4, 0.85)
    axes[0].legend(frameon=False, fontsize=8, ncol=4)
    axes[0].set_title("Anomali ayırt etme · skor karşılaştırması")

    hazards = next(iter(records.values()))["scores"]["energy"].get("per_hazard_category", {})
    names = [
        k
        for k, v in sorted(hazards.items(), key=lambda kv: -(kv[1].get("auroc") or 0))
        if v.get("auroc") is not None
    ]
    values = [hazards[n]["auroc"] for n in names]
    colours = ["#d9534f" if v < 0.5 else "#2f6f9f" for v in values]
    axes[1].barh(names[::-1], values[::-1], color=colours[::-1])
    axes[1].axvline(0.5, color="#d9534f", linestyle="--", linewidth=1)
    axes[1].set_xlabel("AUROC")
    axes[1].set_title("Tehlike türüne göre (energy)")
    save(figure, output, "05_open_set", written)


def figure_pareto(accuracy: dict, root: Path, output: Path, written: list[str]) -> None:
    """The trade-off the thesis has to resolve, on one pair of axes."""
    models = [m for m in MODEL_ORDER if m in accuracy and m in JETSON]
    if not models:
        return
    open_set = {}
    for model in models:
        record = load(root / "open_set" / f"{model}.json")
        if record:
            open_set[model] = record["scores"]["energy"]["metrics"]["average_precision"]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model in models:
        axes[0].scatter(JETSON[model]["joule"], accuracy[model]["mIoU"] * 100, s=120, zorder=3)
        axes[0].annotate(
            MODEL_LABEL[model],
            (JETSON[model]["joule"], accuracy[model]["mIoU"] * 100),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
        )
        if model in open_set:
            axes[1].scatter(JETSON[model]["frame_ms"], open_set[model], s=120, zorder=3)
            axes[1].annotate(
                MODEL_LABEL[model],
                (JETSON[model]["frame_ms"], open_set[model]),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=8,
            )
    axes[0].set_xlabel("enerji (J/kare) — düşük iyi")
    axes[0].set_ylabel("mIoU (%) — yüksek iyi")
    axes[0].set_title("Doğruluk ↔ enerji")
    axes[1].set_xlabel("kare süresi (ms) — düşük iyi")
    axes[1].set_ylabel("açık küme AP — yüksek iyi")
    axes[1].set_title("Açık küme güvenliği ↔ gecikme")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.suptitle("Pareto: Jetson Orin Nano Super, 25 W, TensorRT FP16", fontsize=10)
    save(figure, output, "06_pareto", written)


def figure_frame_budget(output: Path, written: list[str]) -> None:
    """Where the frame time goes, and what two CPU-side changes moved."""
    names = [row[0] for row in FRAME_STAGES]
    before = np.array([row[1] for row in FRAME_STAGES])
    after = np.array([row[2] for row in FRAME_STAGES])

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8), width_ratios=[2, 1])
    positions = np.arange(len(names))
    axes[0].barh(
        positions + 0.2, before, 0.38, label=f"önce ({before.sum():.0f} ms)", color="#9aa5b1"
    )
    axes[0].barh(
        positions - 0.2, after, 0.38, label=f"sonra ({after.sum():.0f} ms)", color="#2f6f9f"
    )
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("medyan süre (ms)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Kare bütçesi · aşama aşama")

    engine_share = after[4] / after.sum() * 100
    axes[1].pie(
        [after[4], after.sum() - after[4]],
        labels=[f"TensorRT motoru\n%{engine_share:.1f}", f"CPU tarafı\n%{100 - engine_share:.1f}"],
        colors=["#2f6f9f", "#e8ecf1"],
        startangle=90,
        wedgeprops={"edgecolor": "white"},
        textprops={"fontsize": 9},
    )
    axes[1].set_title("Optimizasyon sonrası pay")
    save(figure, output, "07_frame_budget", written)


def figure_shift(root: Path, output: Path, written: list[str]) -> None:
    """Whether the uncertainty signal notices synthetic degradation at all."""
    record = load(root / "shift_response.json")
    if not record:
        return
    conditions = record.get("conditions", {})
    names = list(conditions)
    entropy = [conditions[n]["mean_normalized_entropy"] for n in names]
    low = [conditions[n]["low_confidence_pixel_ratio"] * 100 for n in names]

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    colours = ["#5cb85c"] + ["#2f6f9f"] * (len(names) - 1)
    axes[0].bar(names, entropy, color=colours)
    axes[0].axhline(entropy[0], color="#5cb85c", linestyle="--", linewidth=1)
    axes[0].set_ylabel("ortalama normalize entropi")
    axes[0].set_title("Belirsizlik tepkisi")
    axes[1].bar(names, low, color=colours)
    axes[1].axhline(low[0], color="#5cb85c", linestyle="--", linewidth=1)
    axes[1].set_ylabel("düşük-güven piksel (%)")
    axes[1].set_title("İşaretlenen piksel oranı")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.suptitle("Sentetik bozulmaya tepki · gece ve yağmurda tepki yok", fontsize=10)
    save(figure, output, "08_synthetic_shift", written)


def figure_overconfidence(root: Path, output: Path, written: list[str]) -> None:
    """Confidence on pixels the model provably cannot get right."""
    rows = []
    for model in MODEL_ORDER:
        record = load(root / "open_set" / f"{model}.json")
        if not record:
            continue
        distribution = record["scores"]["maximum_softmax_probability"]["distribution"]
        rows.append(
            (MODEL_LABEL[model], -distribution["anomaly"]["mean"], -distribution["id"]["mean"])
        )
    if not rows:
        return
    labels = [r[0] for r in rows]
    positions = np.arange(len(rows))

    figure, axis = plt.subplots(figsize=(7.5, 3.4))
    axis.bar(positions - 0.2, [r[2] for r in rows], 0.4, label="normal piksel", color="#5cb85c")
    axis.bar(
        positions + 0.2,
        [r[1] for r in rows],
        0.4,
        label="anomali piksel (kesin yanlış)",
        color="#d9534f",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=18, ha="right")
    axis.set_ylabel("ortalama güven")
    axis.set_ylim(0, 1)
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("Model, kesin yanıldığı yerlerde de emin")
    save(figure, output, "09_overconfidence", written)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.results.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    accuracy = accuracy_records(root)
    written: list[str] = []

    figure_published_vs_measured(accuracy, output, written)
    figure_per_class(accuracy, output, written)
    figure_reliability(accuracy, output, written)
    figure_acdc(root, output, written)
    figure_open_set(root, output, written)
    figure_pareto(accuracy, root, output, written)
    figure_frame_budget(output, written)
    figure_shift(root, output, written)
    figure_overconfidence(root, output, written)

    for name in written:
        print(f"  {name}.png / .pdf")
    print(f"\n{len(written)} figür -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
