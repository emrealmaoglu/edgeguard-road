"""Draw the structural diagrams a thesis needs and measurements cannot produce.

The figures rendered from records show what was measured. These show what was built and
how it was measured -- the signal path from pixels to a risk ranking, where each metric
was taken and on which hardware, and where the frame budget goes. A reader who cannot see
the pipeline cannot judge whether the numbers describe it.

Drawn rather than measured, so nothing here carries a scientific claim; the values that do
appear are quoted from records rendered elsewhere and labelled with their source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {"figure.dpi": 140, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 8.5}
)

INK = "#1f2933"
EDGE = "#52606d"
BLUE = "#2f6f9f"
GREEN = "#3f8f5f"
AMBER = "#c98a2b"
RED = "#c0392b"
GREY = "#e8ecf1"


def box(axis, x, y, w, h, text, *, face=GREY, edge=EDGE, fontsize=8.5, weight="normal"):
    axis.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.1,
        )
    )
    axis.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        fontweight=weight,
        linespacing=1.45,
    )
    return (x, y, w, h)


def arrow(axis, start, end, *, colour=EDGE, style="-|>", width=1.2, text="", offset=(0, 0.018)):
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": style,
            "color": colour,
            "linewidth": width,
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )
    if text:
        axis.text(
            (start[0] + end[0]) / 2 + offset[0],
            (start[1] + end[1]) / 2 + offset[1],
            text,
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=EDGE,
        )


def blank(width, height):
    figure, axis = plt.subplots(figsize=(width, height))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    return figure, axis


def save(figure, output: Path, name: str, written: list[str]) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"{name}.{suffix}")
    plt.close(figure)
    written.append(name)


def diagram_architecture(output: Path, written: list[str]) -> None:
    """The signal path from a camera frame to a ranked risk list."""
    figure, axis = blank(12.6, 4.8)
    axis.text(
        0.5,
        0.975,
        "Sistem mimarisi · görüntüden risk sıralamasına",
        ha="center",
        fontsize=11,
        color=INK,
        fontweight="bold",
    )

    box(axis, 0.010, 0.60, 0.105, 0.16, "kamera karesi\n2048×1024", face="#ffffff", fontsize=8)
    box(
        axis,
        0.135,
        0.60,
        0.140,
        0.16,
        "ön işleme\nletterbox +\nImageNet · 512×1024",
        face=GREY,
        fontsize=7.8,
    )
    box(
        axis,
        0.295,
        0.58,
        0.130,
        0.20,
        "TensorRT FP16\nmotoru\n\n5,07 ms",
        face=BLUE,
        fontsize=9,
        weight="bold",
    )
    box(axis, 0.445, 0.60, 0.100, 0.16, "logitler\n19×64×128", face="#ffffff", fontsize=8)

    arrow(axis, (0.115, 0.68), (0.135, 0.68))
    arrow(axis, (0.275, 0.68), (0.295, 0.68))
    arrow(axis, (0.425, 0.68), (0.445, 0.68))

    heads = [
        (0.820, "argmax", "semantik maske\n19 sınıf", GREEN),
        (0.645, "softmax", "güven + entropi", AMBER),
        (0.470, "−logsumexp", "energy\n(anomali skoru)", RED),
    ]
    for y, operation, label, colour in heads:
        arrow(axis, (0.545, 0.68), (0.600, y + 0.055), colour=colour)
        axis.text(
            0.572,
            (0.68 + y + 0.055) / 2 + 0.022,
            operation,
            ha="center",
            fontsize=7,
            color=colour,
            fontweight="bold",
        )
        box(axis, 0.600, y, 0.130, 0.11, label, face="#ffffff", edge=colour, fontsize=8)

    box(
        axis,
        0.742,
        0.470,
        0.126,
        0.30,
        "derive_perception\n\nyol maskesi\nsürülebilir koridor\nbölgeler\ngüvenilmez piksel",
        face=GREY,
    )
    for y in (0.875, 0.700, 0.525):
        arrow(axis, (0.722, y), (0.742, 0.620))

    box(
        axis,
        0.882,
        0.540,
        0.105,
        0.16,
        "bağlamsal risk\n7 özellik füzyonu",
        face="#fdf1d6",
        edge=AMBER,
        weight="bold",
    )
    arrow(axis, (0.868, 0.620), (0.882, 0.620))

    axis.text(
        0.5, 0.36, "Yedi risk özelliği", ha="center", fontsize=9, color=INK, fontweight="bold"
    )
    features = [
        ("anomali skoru", GREEN),
        ("bileşen alanı", GREEN),
        ("görüntü konumu", GREEN),
        ("yol örtüşmesi", GREEN),
        ("koridor yakınlığı", GREEN),
        ("dedektör örtüşmesi", "#b0b0b0"),
        ("zamansal kalıcılık", "#b0b0b0"),
    ]
    for index, (name, colour) in enumerate(features):
        box(
            axis,
            0.040 + index * 0.136,
            0.20,
            0.118,
            0.10,
            name,
            face="#ffffff" if colour == GREEN else "#f4f4f4",
            edge=colour,
            fontsize=7.6,
        )
    axis.text(
        0.040 + 6 * 0.136 + 0.118,
        0.155,
        "gri: ölçülemedi → sıfır ağırlık (sıfır değer değil)",
        ha="right",
        fontsize=7.2,
        color=EDGE,
        style="italic",
    )
    axis.text(
        0.5,
        0.055,
        "Motor kare bütçesinin %5,4'ü; kalan %94,6 CPU tarafındaki bu zincirde harcanıyor.",
        ha="center",
        fontsize=8.4,
        color=RED,
        fontweight="bold",
    )
    save(figure, output, "D1_system_architecture", written)


def diagram_protocol(output: Path, written: list[str]) -> None:
    """Which dataset answers which question, measured where."""
    figure, axis = blank(11.5, 4.4)
    axis.text(
        0.5,
        0.965,
        "Ölçüm protokolü · hangi soru, hangi veri, hangi donanım",
        ha="center",
        fontsize=11,
        color=INK,
        fontweight="bold",
    )

    columns = [
        (
            0.02,
            "Veri",
            [
                ("Cityscapes val\n500 kare", "#ffffff"),
                ("ACDC val\n406 kare · 4 koşul", "#ffffff"),
                ("RoadAnomaly\n60 kare · piksel etiketli", "#ffffff"),
                ("Cityscapes demoVideo\n180 ardışık kare", "#ffffff"),
            ],
        ),
        (
            0.35,
            "Ölçülen",
            [
                ("mIoU · sınıf bazlı IoU\nECE · reliability", "#eef4f9"),
                ("koşul başına mIoU\nkalibrasyon bozulması", "#eef4f9"),
                ("AUROC · AP · FPR95\ntehlike kırılımı · bootstrap", "#eef4f9"),
                ("bağlamsal risk\nnitel gösterim", "#eef4f9"),
            ],
        ),
        (
            0.68,
            "Nerede",
            [
                ("ONNX FP32 · CPU\n(donanımdan bağımsız)", "#f2f7f2"),
                ("ONNX FP32 · CPU", "#f2f7f2"),
                ("ONNX FP32 · CPU", "#f2f7f2"),
                ("ONNX FP32 · CPU", "#f2f7f2"),
            ],
        ),
    ]
    for x, header, rows in columns:
        axis.text(x + 0.145, 0.855, header, ha="center", fontsize=9.2, color=INK, fontweight="bold")
        for index, (text, face) in enumerate(rows):
            box(axis, x, 0.66 - index * 0.155, 0.29, 0.115, text, face=face, fontsize=7.8)
    for index in range(4):
        y = 0.7175 - index * 0.155
        arrow(axis, (0.315, y), (0.348, y))
        arrow(axis, (0.645, y), (0.678, y))

    box(
        axis,
        0.02,
        0.055,
        0.955,
        0.105,
        "Jetson Orin Nano Super · 25 W · TensorRT FP16 · 600 sn sürdürülen yük · tam telemetri\n"
        "gecikme · FPS · güç · joule/kare · bellek · termal   —   5 mimarinin hepsi, aynı kareler",
        face="#fdf1d6",
        edge=AMBER,
        fontsize=8.4,
        weight="bold",
    )
    axis.text(
        0.5,
        0.195,
        "cihaza özgü ölçümler yalnızca hedef donanımda",
        ha="center",
        fontsize=7.6,
        color=EDGE,
        style="italic",
    )
    save(figure, output, "D2_measurement_protocol", written)


def diagram_three_axes(output: Path, written: list[str]) -> None:
    """The evaluation frame: three axes that do not agree."""
    figure, axis = blank(9.5, 4.2)
    axis.text(
        0.5,
        0.955,
        "Değerlendirme çerçevesi · üç eksen, üç farklı kazanan",
        ha="center",
        fontsize=11,
        color=INK,
        fontweight="bold",
    )

    axes_data = [
        (
            0.03,
            "Doğruluk",
            BLUE,
            "dağıtım çözünürlüğünde mIoU\nsınıf bazlı IoU\nnadir sınıf başarımı",
            "kazanan\nSegFormer-B0\n69,34",
        ),
        (
            0.355,
            "Güvenilirlik",
            AMBER,
            "ECE · reliability\naçık küme AUROC/AP\nolumsuz koşul dayanıklılığı",
            "kazanan\nSegFormer-B0\nECE 0,0117 · AP 0,347",
        ),
        (
            0.68,
            "Uç maliyet",
            GREEN,
            "kare süresi · FPS\njoule/kare · bellek\ntermal davranış",
            "kazanan\nDDRNet-23-slim\n0,630 J/kare",
        ),
    ]
    for x, title, colour, body, winner in axes_data:
        box(
            axis,
            x,
            0.60,
            0.29,
            0.26,
            f"{title}\n\n{body}",
            face="#ffffff",
            edge=colour,
            fontsize=8.2,
        )
        box(
            axis,
            x,
            0.40,
            0.29,
            0.14,
            winner,
            face="#ffffff",
            edge=colour,
            fontsize=8.2,
            weight="bold",
        )
        arrow(axis, (x + 0.145, 0.60), (x + 0.145, 0.545), colour=colour)

    box(
        axis,
        0.03,
        0.16,
        0.94,
        0.17,
        "Yayınlanmış mIoU sıralaması dağıtım sıralamasını öngörmüyor (ρ = +0,10, n=5)\n"
        "SegFormer-B0 üç kalite ekseninde de birinci; tek kaybı enerji, o da stride-4 çıktısının\n"
        "CPU tarafına 4× piksel vermesinden — mimariden değil entegrasyondan.",
        face="#fdecea",
        edge=RED,
        fontsize=8.6,
        weight="bold",
    )
    save(figure, output, "D3_three_axis_framework", written)


def diagram_frame_budget(output: Path, written: list[str]) -> None:
    """Where a frame's time is spent, and what two changes moved."""
    figure, axis = blank(11, 3.4)
    axis.text(
        0.5,
        0.94,
        "Kare bütçesi · hızlandırıcı darboğaz değil",
        ha="center",
        fontsize=11,
        color=INK,
        fontweight="bold",
    )

    stages = [
        ("ön işleme", 24.71, GREY),
        ("görüntü çözme", 14.00, GREY),
        ("motor", 5.09, BLUE),
        ("argmax", 0.74, GREY),
        ("confidence/entropy", 5.80, GREY),
        ("derive_perception", 43.64, AMBER),
    ]
    total = sum(value for _, value, _ in stages)
    x = 0.03
    for name, value, colour in stages:
        width = 0.94 * value / total
        box(axis, x, 0.46, width, 0.20, "", face=colour)
        if width > 0.06:
            axis.text(
                x + width / 2,
                0.56,
                f"{name}\n{value:.1f} ms",
                ha="center",
                va="center",
                fontsize=7.8,
                color="white" if colour in (BLUE, AMBER) else INK,
                fontweight="bold" if colour == BLUE else "normal",
            )
        x += width
    axis.text(
        0.5,
        0.375,
        f"toplam {total:.1f} ms → {1000 / total:.1f} FPS",
        ha="center",
        fontsize=8.6,
        color=INK,
    )

    axis.annotate(
        "",
        xy=(0.03 + 0.94 * (24.71 + 14.00) / total, 0.70),
        xytext=(0.03 + 0.94 * (24.71 + 14.00 + 5.09) / total, 0.70),
        arrowprops={"arrowstyle": "<->", "color": BLUE, "linewidth": 1.4},
    )
    axis.text(
        0.03 + 0.94 * (24.71 + 14.00 + 2.5) / total,
        0.745,
        "TensorRT motoru: %5,4",
        ha="center",
        fontsize=8.4,
        color=BLUE,
        fontweight="bold",
    )

    box(
        axis,
        0.03,
        0.08,
        0.455,
        0.22,
        "1 · mesafe dönüşümü ayrılabilir hâle getirildi\n"
        "     BFS kuyruğu → 4 accumulate taraması · 34× · çıktı birebir aynı",
        face="#f2f7f2",
        edge=GREEN,
        fontsize=7.9,
    )
    box(
        axis,
        0.515,
        0.08,
        0.455,
        0.22,
        "2 · bileşen etiketleme vektörleştirildi\n"
        "     karar hedef cihazda verildi: Mac'te BFS 1,4× · Jetson'da 1,59×",
        face="#f2f7f2",
        edge=GREEN,
        fontsize=7.9,
    )
    axis.text(
        0.5,
        0.025,
        "146,4 ms → 94,0 ms (1,88× kare, 1,63× enerji) · modele dokunulmadı",
        ha="center",
        fontsize=8.6,
        color=GREEN,
        fontweight="bold",
    )
    save(figure, output, "D4_frame_budget_flow", written)


def diagram_prisma(output: Path, written: list[str]) -> None:
    """The literature screening, drawn as the review flow it actually was."""
    figure, axis = blank(7.5, 6.2)
    axis.text(
        0.5,
        0.965,
        "Literatür tarama akışı (PRISMA)",
        ha="center",
        fontsize=11,
        color=INK,
        fontweight="bold",
    )

    stages = [
        (0.80, "Tanımlama", "36 yapılandırılmış tarama dokümanı\n**912 benzersiz kayıt**", BLUE),
        (0.60, "Tarama", "konu dışı alanlar elendi  −34\nkod/forum/ürün ayrıldı  −661", EDGE),
        (0.40, "Uygunluk", "**251 akademik yayın**\nbaşlık ve özet düzeyinde tarandı", EDGE),
        (0.20, "Dahil edilen", "**~50 kaynak**\nher biri açılıp doğrulandı", GREEN),
    ]
    for y, title, body, colour in stages:
        box(
            axis,
            0.20,
            y,
            0.60,
            0.145,
            f"{title}\n{body.replace('**', '')}",
            face="#ffffff",
            edge=colour,
            fontsize=8.4,
        )
        if y > 0.20:
            arrow(axis, (0.50, y), (0.50, y - 0.055), colour=colour)

    for y, text in (
        (0.665, "konu dışı: su, tarım, biyomedikal"),
        (0.465, "kod deposu, forum, ürün sayfası"),
        (0.265, "bölümlere göre konu ayrımı"),
    ):
        axis.text(0.83, y, text, ha="left", va="center", fontsize=7, color=EDGE, style="italic")

    axis.text(
        0.5,
        0.075,
        "Kod depoları ve resmî dokümanlar akademik kaynak sayılmaz;\n"
        "materyal ve yöntem bölümünde araç/veri atfı olarak ayrı listelenir.",
        ha="center",
        fontsize=7.6,
        color=EDGE,
    )
    save(figure, output, "D5_prisma_flow", written)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    diagram_architecture(output, written)
    diagram_protocol(output, written)
    diagram_three_axes(output, written)
    diagram_frame_budget(output, written)
    diagram_prisma(output, written)
    for name in written:
        print(f"  {name}.png / .pdf")
    print(f"\n{len(written)} şema -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
