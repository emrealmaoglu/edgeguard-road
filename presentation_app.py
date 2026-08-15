"""EdgeGuard-Road sunum paneli — anlattığı cihazın üzerinde çalışır.

Gösterilen her sonuç önceden ölçülmüş ve diske yazılmış kayıtlardan okunur; panel canlı
çıkarım yapmaz. Sebebi basit: sunum videosu çekilirken bir modelin yavaş yüklenmesi ya da
bir kareye takılması, anlatılan şeyle ilgisi olmayan bir riske girmek olur.

Tek istisna cihazın kendi durumu. Panel bir Jetson Orin Nano Super üzerinde koşuyorsa
kenar çubuğunda anlık sıcaklık, güç ve bellek gösterir -- ama asıl telemetri bu değil:
asıl telemetri, gösterilen sonuçlar *üretilirken* kaydedilen 600 saniyelik zaman
serileridir. Boştaki bir cihazın watt'ı, ölçümün yapıldığı andaki yükle ilgisiz bir sayı
olurdu.

Çalıştırma:
    EDGEGUARD_RESULTS=~/eg-presentation streamlit run presentation_app.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

MODEL_ORDER = ("pidnet_m", "pidnet_s", "ddrnet_23_slim", "segformer_b0", "bisenetv2")
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
CITYSCAPES_CLASSES = (
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

POWER_PATTERN = re.compile(r"VDD_IN\s+(?P<mw>\d+)mW")
RAM_PATTERN = re.compile(r"RAM\s+(?P<used>\d+)/(?P<total>\d+)MB")
TEMPERATURE_PATTERN = re.compile(r"(?P<zone>[a-zA-Z0-9_]+)@(?P<celsius>-?\d+(?:\.\d+)?)C")


def results_root() -> Path:
    return Path(os.environ.get("EDGEGUARD_RESULTS", "results")).expanduser().resolve()


def load_json(path: Path) -> dict[str, Any] | None:
    """Read one record, returning None rather than raising for any unreadable file.

    `UnicodeDecodeError` is a `ValueError`, not an `OSError`, so a stray binary file next
    to the records used to take the whole panel down with a traceback -- during a
    recording, that is the difference between a missing table and no presentation.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_group(directory: Path) -> dict[str, dict[str, Any]]:
    if not directory.is_dir():
        return {}
    loaded = {}
    for path in sorted(directory.glob("*.json")):
        # macOS archives carry AppleDouble sidecars named `._<original>`, which match a
        # `*.json` glob while containing binary resource-fork data.
        if path.name.startswith("._"):
            continue
        record = load_json(path)
        if record is not None:
            loaded[path.stem] = record
    return loaded


def tegrastats_series(path: Path, *, stride: int = 5) -> dict[str, list[float]]:
    """Turn a tegrastats log into per-sample series.

    `benchmark.parse_tegrastats` reduces the same log to means and peaks, which is what a
    record needs but hides the shape: whether the device warmed up and settled, or kept
    climbing towards a throttle. The chart wants the shape.
    """
    series: dict[str, list[float]] = {"güç (W)": [], "RAM (GB)": [], "GPU sıcaklık (°C)": []}
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return {}
    for index, line in enumerate(lines):
        if index % stride:
            continue
        power = POWER_PATTERN.search(line)
        ram = RAM_PATTERN.search(line)
        temperatures = {
            m.group("zone"): float(m.group("celsius")) for m in TEMPERATURE_PATTERN.finditer(line)
        }
        if power:
            series["güç (W)"].append(int(power.group("mw")) / 1000.0)
        if ram:
            series["RAM (GB)"].append(int(ram.group("used")) / 1024.0)
        gpu = temperatures.get("gpu") or temperatures.get("GPU") or temperatures.get("tj")
        if gpu is not None:
            series["GPU sıcaklık (°C)"].append(gpu)
    shortest = min((len(v) for v in series.values() if v), default=0)
    return {key: value[:shortest] for key, value in series.items() if value} if shortest else {}


def live_vitals() -> dict[str, Any]:
    """Sample the device's current state, degrading to whatever is readable.

    `tegrastats` is the richest source but is not always runnable without privileges, so
    sysfs is the fallback and 'unavailable' is an honest third outcome -- a panel that
    invents a temperature is worse than one that admits it cannot read it.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as handle:
            log = Path(handle.name)
        process = subprocess.Popen(
            ["tegrastats", "--interval", "500", "--logfile", str(log)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.2)
        process.terminate()
        process.wait(timeout=3)
        lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
        log.unlink(missing_ok=True)
        if lines:
            line = lines[-1]
            power = POWER_PATTERN.search(line)
            ram = RAM_PATTERN.search(line)
            temperatures = {
                m.group("zone"): float(m.group("celsius"))
                for m in TEMPERATURE_PATTERN.finditer(line)
            }
            return {
                "source": "tegrastats",
                "power_w": int(power.group("mw")) / 1000.0 if power else None,
                "ram_used_mib": int(ram.group("used")) if ram else None,
                "ram_total_mib": int(ram.group("total")) if ram else None,
                "gpu_c": temperatures.get("gpu") or temperatures.get("tj"),
                "cpu_c": temperatures.get("cpu"),
            }
    except (OSError, subprocess.SubprocessError, ValueError):
        pass

    vitals: dict[str, Any] = {"source": "sysfs"}
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        total = int(re.search(r"MemTotal:\s+(\d+) kB", meminfo).group(1)) // 1024
        available = int(re.search(r"MemAvailable:\s+(\d+) kB", meminfo).group(1)) // 1024
        vitals["ram_used_mib"] = total - available
        vitals["ram_total_mib"] = total
    except (OSError, AttributeError, ValueError):
        vitals["ram_used_mib"] = None
    try:
        readings = []
        for zone in sorted(Path("/sys/devices/virtual/thermal").glob("thermal_zone*")):
            value = int((zone / "temp").read_text(encoding="utf-8").strip())
            readings.append(value / 1000.0)
        vitals["gpu_c"] = max(readings) if readings else None
    except (OSError, ValueError):
        vitals["gpu_c"] = None
    vitals["power_w"] = None
    if vitals.get("ram_used_mib") is None and vitals.get("gpu_c") is None:
        return {"source": "unavailable"}
    return vitals


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """Render a table without pandas, so the panel stays light on the device."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _number(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}{suffix}"


# A recorded presentation is watched, not read: the panel is styled once here so every
# page inherits the same rhythm instead of each table and caption drifting apart.
THEME = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1500px; }
  h1 { font-size: 2.0rem !important; line-height: 1.18; letter-spacing: -0.015em; }
  h2, h3 { letter-spacing: -0.01em; }
  h4 { font-size: 1.05rem !important; margin-top: 1.1rem; opacity: 0.92; }
  /* Tables carry the evidence, so they get the legibility budget. */
  table { font-variant-numeric: tabular-nums; font-size: 0.92rem; }
  thead tr th { font-weight: 600 !important; opacity: 0.75; }
  tbody tr td { padding-top: 0.34rem !important; padding-bottom: 0.34rem !important; }
  tbody tr:hover { background: rgba(127,127,127,0.07); }
  [data-testid="stMetricValue"] { font-size: 1.5rem; }
  [data-testid="stCaptionContainer"] { opacity: 0.7; }
  section[data-testid="stSidebar"] { border-right: 1px solid rgba(127,127,127,0.18); }
  img { border-radius: 6px; }
</style>
"""


def show_figure(st: Any, root: Path, name: str, caption: str = "") -> bool:
    """Display a rendered figure if it was produced; stay silent if it was not.

    A page that prints "figure missing" in a recorded presentation is noise; a page that
    substitutes a placeholder would be worse. Absent evidence simply does not appear.
    """
    path = root / "thesis_figures" / f"{name}.png"
    if not path.is_file():
        return False
    st.image(str(path), caption=caption or None, use_container_width=True)
    return True


def show_image(st: Any, path: Path, caption: str = "") -> bool:
    if not path.is_file():
        return False
    st.image(str(path), caption=caption or None, use_container_width=True)
    return True


def main() -> None:
    try:
        st = __import__("streamlit")
    except ModuleNotFoundError as error:
        raise RuntimeError("pip install streamlit") from error

    st.set_page_config(page_title="EdgeGuard-Road", layout="wide", page_icon="🛣️")
    st.markdown(THEME, unsafe_allow_html=True)
    root = results_root()

    jetson = load_group(root / "jetson")
    accuracy = load_group(root / "accuracy")
    open_set = load_group(root / "open_set")
    acdc = load_group(root / "acdc")
    risk = load_group(root / "risk")
    drivable = load_group(root / "drivable")
    shift = load_json(root / "shift_response.json")
    profiles = load_group(root / "profile")

    with st.sidebar:
        st.markdown("### Bu panel nerede çalışıyor")
        vitals = live_vitals()
        if vitals.get("source") == "unavailable":
            st.info("Cihaz telemetrisi okunamıyor (Jetson dışında çalışıyor olabilir).")
        else:
            columns = st.columns(2)
            columns[0].metric("Güç", _number(vitals.get("power_w"), 2, " W"))
            columns[1].metric("GPU", _number(vitals.get("gpu_c"), 1, " °C"))
            used, total = vitals.get("ram_used_mib"), vitals.get("ram_total_mib")
            if used and total:
                st.metric("RAM", f"{used / 1024:.2f} / {total / 1024:.1f} GB")
            st.caption(f"kaynak: {vitals['source']} · anlık, boştaki cihaz")
        st.divider()
        page = st.radio(
            "Bölüm",
            [
                "1 · Problem ve yöntem",
                "2 · Model karşılaştırması",
                "3 · Açık küme yol tehlikesi",
                "4 · Belirsizlik ve kalibrasyon",
                "5 · Bağlamsal risk",
                "6 · Uç cihaz: koşu telemetrisi",
                "7 · Sınırlar",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"kayıtlar: `{root}`")
        st.caption(
            "Gösterilen her sayı diskteki bir ölçüm kaydından okunur. Panel canlı çıkarım yapmaz."
        )

    if page.startswith("1"):
        render_problem(st, root)
    elif page.startswith("2"):
        render_comparison(st, accuracy, open_set, jetson, root)
    elif page.startswith("3"):
        render_open_set(st, open_set, root)
    elif page.startswith("4"):
        render_uncertainty(st, accuracy, open_set, acdc, shift, root)
    elif page.startswith("5"):
        render_risk(st, risk, drivable, root)
    elif page.startswith("6"):
        render_edge(st, jetson, profiles, root)
    else:
        render_limits(st)


def render_problem(st: Any, root: Path) -> None:
    st.title(
        "Kaynak Kısıtlı Uç Cihazlarda Belirsizlik Farkındalıklı Açık Küme "
        "Yol Tehlikesi Algılama ve Bağlamsal Risk Analizi"
    )
    st.markdown(
        """
Bir uç cihaz, gördüğünü söylemekle yetinmemeli. **Ne kadar emin olduğunu**, **daha önce
hiç görmediği bir şeyle karşılaşıp karşılaşmadığını** ve **hangi bölgenin operasyonel
olarak önemli olduğunu** da söylemeli — hem de 25 W bütçe içinde, gerçek zamanlı.

Bu çalışma o soruyu dört eksende ölçüyor.
"""
    )
    columns = st.columns(4)
    for column, (title, body) in zip(
        columns,
        [
            ("Doğruluk", "5 gerçek zamanlı mimari, Cityscapes val, dağıtım çözünürlüğünde"),
            ("Belirsizlik", "MSP · entropi · max-logit · energy, kalibrasyonuyla birlikte"),
            ("Açık küme", "RoadAnomaly: 60 gerçek yol tehlikesi, piksel etiketli"),
            ("Uç maliyet", "Jetson Orin Nano Super 25 W, TensorRT FP16, 600 sn sürdürülen yük"),
        ],
        strict=False,
    ):
        column.markdown(f"**{title}**")
        column.caption(body)
    st.divider()
    show_figure(st, root, "D1_system_architecture", "Görüntüden risk sıralamasına sinyal yolu")
    show_figure(
        st, root, "D2_measurement_protocol", "Hangi veri hangi soruyu, nerede ölçülerek yanıtlıyor"
    )
    show_figure(st, root, "D3_three_axis_framework", "Üç eksen, üç farklı kazanan")
    st.divider()
    st.markdown(
        """
#### Yöntem

Modeller Cityscapes'te eğitilmiş resmî referans checkpoint'lerdir (mmsegmentation model
zoo, Apache-2.0); ayrıca Cityscapes + IDD20K karışımıyla kendi sınırlı bütçeli eğitimimiz
yapılmıştır. İkisi ayrı tutulur, karıştırılmaz.

Bütün dağıtım ölçümleri tek bir donanımda, tek bir güç modunda ve aynı karelerle
alınmıştır. Panelde gösterilen her sayı diskteki bir ölçüm kaydından okunur; hiçbiri
canlı üretilmez.
"""
    )


def render_comparison(st: Any, accuracy: dict, open_set: dict, jetson: dict, root: Path) -> None:
    st.title("Dört eksende model karşılaştırması")
    st.caption(
        "Doğruluk, kalibrasyon ve açık küme donanımdan bağımsızdır; maliyet Jetson Orin "
        "Nano Super'de 25 W'ta, 600 saniye sürdürülen yük altında ölçülmüştür."
    )
    columns = st.columns(4)
    columns[0].metric("Doğrulukta birinci", "SegFormer-B0", "69,34 mIoU")
    columns[1].metric("Açık kümede birinci", "SegFormer-B0", "AP 0,347")
    columns[2].metric("Enerjide birinci", "DDRNet-23-slim", "0,630 J/kare")
    columns[3].metric("Gecede en dayanıklı", "SegFormer-B0", "%29,7 korunan")
    st.caption(
        "Dört eksenin üçünü aynı model kazanıyor; enerjiyi kazanan model gecede "
        "işlevsiz kalıyor (7,12 mIoU)."
    )
    rows = []
    for model in MODEL_ORDER:
        measured = accuracy.get(f"{model}_cityscapes_val", {})
        ood = (open_set.get(model) or {}).get("scores", {}).get("energy", {}).get("metrics", {})
        bench = jetson.get(f"{model}_reference_benchmark", {})
        rows.append(
            [
                MODEL_LABEL[model],
                f"{PUBLISHED_MIOU[model]:.2f}",
                _number(measured.get("mIoU")),
                _number(ood.get("auroc")),
                _number(ood.get("average_precision")),
                _number((bench.get("pure_engine_latency_ms") or {}).get("median"), 2, " ms"),
                _number((bench.get("end_to_end_latency_ms") or {}).get("median"), 1, " ms"),
                _number(bench.get("joule_per_frame")),
            ]
        )
    st.markdown(
        _markdown_table(
            [
                "Model",
                "Yayın mIoU",
                "Ölçülen mIoU",
                "OOD AUROC",
                "OOD AP",
                "Motor",
                "Kare",
                "J/kare",
            ],
            rows,
        )
    )
    st.divider()
    show_figure(
        st,
        root,
        "01_published_vs_measured_miou",
        "Yayınlanmış sıralama dağıtım sıralamasını öngörmüyor (ρ = +0,10)",
    )
    show_figure(st, root, "02_per_class_iou", "Sınıf bazlı IoU · dağıtım çözünürlüğü")
    show_figure(st, root, "06_pareto", "Doğruluk ↔ enerji ve açık küme ↔ gecikme")
    st.divider()
    left, right = st.columns(2)
    left.markdown(
        """
#### Bulgu 1 · Yayın sıralaması dağıtımı öngörmüyor

**Spearman ρ = +0,10** (n=5). Dağıtım çözünürlüğünde SegFormer-B0 birinci; yayınlanmış
sıralamada sonuncuydu. Her mimari çözünürlük düşüşünden farklı etkileniyor (−%9,4 ile
−%14,7 arası).

**Sonuç:** uç cihaz için model seçimi yayınlanmış mIoU'ya bakarak yapılamaz.
"""
    )
    right.markdown(
        """
#### Bulgu 2 · Sistem maliyetini çıktı stride'ı belirliyor

SegFormer-B0'ın motoru DDRNet'ten **+11,2 ms** yavaş, ama karesi **+193,4 ms** — 17 katı.
Sebep: stride-4 çıktısı (128×256) CPU tarafına 4× piksel veriyor.

**Ve bu maliyet giderilemiyor:** logitleri stride-8'e indirmek post-processing'i 3,98×
hızlandırıyor ama **2,03 mIoU'ya** mal oluyor. Yüksek çözünürlüklü logit gerçek doğruluk
taşıyor.
"""
    )
    st.info(
        "**Bulgu 3 · Hız kazananı, dayanıklılık kaybedeni.** DDRNet-23-slim en hızlı "
        "(77,43 ms) ve en verimli (0,630 J/kare) model; gecede doğruluğunun yalnızca "
        "%10,4'ünü koruyor (7,12 mIoU) ve kalibrasyonu 14 kat bozuluyor. Aynı testte "
        "SegFormer-B0 %29,7 koruyor."
    )


def render_open_set(st: Any, open_set: dict, root: Path) -> None:
    st.title("Açık küme yol tehlikesi algılama")
    st.caption(
        "RoadAnomaly (Lis ve ark., EPFL CVLab): 60 kare, piksel etiketli gerçek tehlikeler. "
        "Hiçbir model bu veriyle eğitilmedi veya ince ayarlanmadı."
    )
    chosen = st.selectbox("Model", MODEL_ORDER, format_func=lambda key: MODEL_LABEL[key], index=1)
    record = open_set.get(chosen)
    if record is None:
        st.warning("Bu model için açık küme kaydı yok.")
        return
    scores = record.get("scores", {})
    st.markdown("#### Belirsizlik skorlarının anomali ayırt etme gücü")
    st.markdown(
        _markdown_table(
            ["Skor", "AUROC", "AP", "FPR95"],
            [
                [
                    name,
                    _number(m.get("auroc"), 4),
                    _number(m.get("average_precision"), 4),
                    _number(m.get("fpr_at_95_tpr"), 4),
                ]
                for name, m in sorted(
                    ((k, v.get("metrics", {})) for k, v in scores.items()),
                    key=lambda kv: -(kv[1].get("auroc") or 0),
                )
            ],
        )
    )
    st.caption(
        "energy > max-logit > entropi > MSP sıralaması OOD literatürünün bildirdiğiyle "
        "aynı — uygulamayı bu sayılardan bağımsız doğruluyor."
    )
    show_figure(st, root, "05_open_set", "Skor karşılaştırması ve tehlike türü kırılımı")
    energy = scores.get("energy", {})
    per_hazard = energy.get("per_hazard_category", {})
    if per_hazard:
        st.markdown("#### Tehlike türüne göre (energy skoru)")
        st.markdown(
            _markdown_table(
                ["Tehlike", "AUROC", "AP", "Anomali piksel"],
                [
                    [
                        name,
                        _number(v.get("auroc"), 4),
                        _number(v.get("average_precision"), 4),
                        f"{v.get('anomaly_pixel_count', 0):,}",
                    ]
                    for name, v in sorted(
                        per_hazard.items(), key=lambda kv: -(kv[1].get("auroc") or 0)
                    )
                    if v.get("auroc") is not None
                ],
            )
        )
        st.error(
            "**Güvenlik bulgusu:** model kayıp yükte (lost cargo) rastgeleden kötü — "
            "AUROC 0,48. Yola düşmüş yükü fark edemiyor."
        )


def render_uncertainty(
    st: Any, accuracy: dict, open_set: dict, acdc: dict, shift: dict | None, root: Path
) -> None:
    st.title("Belirsizlik ve kalibrasyon")
    st.markdown("#### Model, kesin yanıldığı yerlerde ne kadar emin?")
    st.caption(
        "Anomali pikselleri 19 sınıfın hiçbiri değildir; model ne tahmin ederse etsin "
        "hatalıdır. Oradaki güveni, kalibrasyonun doğrudan testidir."
    )
    rows = []
    for model in MODEL_ORDER:
        distribution = (
            (open_set.get(model) or {})
            .get("scores", {})
            .get("maximum_softmax_probability", {})
            .get("distribution", {})
        )
        anomaly = distribution.get("anomaly", {}).get("mean")
        identity = distribution.get("id", {}).get("mean")
        if anomaly is None or identity is None:
            continue
        rows.append(
            [
                MODEL_LABEL[model],
                _number(-anomaly, 4),
                _number(-identity, 4),
                f"{(-anomaly) / (-identity):.3f}×",
            ]
        )
    if rows:
        st.markdown(_markdown_table(["Model", "Anomali px güven", "Normal px güven", "Oran"], rows))
        st.warning(
            "Model, **kesinlikle yanıldığı** piksellerde bile %73–84 güven veriyor; güveni "
            "yalnızca %4–10 düşüyor. MSP'nin neden zayıf bir OOD skoru olduğunu bu açıklıyor."
        )
    calibration_rows = []
    for model in MODEL_ORDER:
        record = accuracy.get(f"{model}_cityscapes_val")
        if not record:
            continue
        calibration_rows.append(
            [
                MODEL_LABEL[model],
                _number(record.get("expected_calibration_error"), 4),
                _number(record.get("mean_confidence"), 4),
                _number(record.get("mean_accuracy"), 4),
                _number(record.get("confidence_minus_accuracy"), 4),
            ]
        )
    if calibration_rows:
        st.markdown("#### Cityscapes val üzerinde kalibrasyon")
        st.markdown(
            _markdown_table(
                ["Model", "ECE", "Ort. güven", "Ort. doğruluk", "Aşırı-güven"], calibration_rows
            )
        )
    if acdc:
        st.markdown("#### Gerçek olumsuz koşullar (ACDC)")
        st.markdown(
            _markdown_table(
                ["Koşul", "mIoU", "ECE", "Ort. güven", "Aşırı-güven"],
                [
                    [
                        key.split("_")[-1],
                        _number(v.get("mIoU")),
                        _number(v.get("expected_calibration_error"), 4),
                        _number(v.get("mean_confidence"), 4),
                        _number(v.get("confidence_minus_accuracy"), 4),
                    ]
                    for key, v in sorted(acdc.items())
                ],
            )
        )
    if shift:
        st.markdown("#### Sentetik bozulmaya belirsizlik tepkisi")
        conditions = shift.get("conditions", {})
        st.markdown(
            _markdown_table(
                ["Koşul", "Ort. entropi", "Temize oran", "Düşük-güven piksel"],
                [
                    [
                        name,
                        _number(v.get("mean_normalized_entropy"), 4),
                        _number(v.get("entropy_ratio_to_clean"), 2, "×"),
                        _number((v.get("low_confidence_pixel_ratio") or 0) * 100, 2, "%"),
                    ]
                    for name, v in conditions.items()
                ],
            )
        )
        st.caption(
            "Sis ve karda düşük-güven piksel oranı neredeyse ikiye katlanıyor; gece ve "
            "yağmurda tepki yok. Bunlar algoritmik bozulmalardır, gerçek fotoğraf değil."
        )
    show_figure(st, root, "03_reliability_diagrams", "Güvenilirlik diyagramları")
    show_figure(st, root, "04_acdc_conditions", "Gerçek olumsuz koşullar")
    show_figure(st, root, "09_overconfidence", "Kesin yanlış piksellerde güven")
    show_figure(st, root, "08_synthetic_shift", "Sentetik bozulmaya tepki")
    show_image(
        st,
        root / "qualitative" / "conditions_pidnet_s.png",
        "Temiz · sis · gece — segmentasyon ve entropi",
    )
    for name, caption in (
        ("confidence.png", "Güven haritası"),
        ("entropy.png", "Entropi haritası"),
        ("unreliable_mask.png", "Güvenilmez piksel maskesi"),
    ):
        show_image(st, root / "figures" / "pidnet_s" / name, caption)


def render_risk(st: Any, risk: dict, drivable: dict, root: Path) -> None:
    st.title("Bağlamsal risk analizi")
    st.caption(
        "Yedi ağırlıklı özelliğin açıklanabilir füzyonu. Ölçülemeyen özellikler sıfır "
        "değerle değil **sıfır ağırlıkla** dışlanır — sıfır değer, ölçülmemiş bir sinyali "
        "'risk yok' gibi gösterir ve bütün skorları aşağı çeker."
    )
    if not risk:
        st.warning("Risk kaydı bulunamadı.")
        return
    frame = st.selectbox("Kare", sorted(risk))
    record = risk[frame]
    excluded = record.get("zero_weighted_features") or []
    if excluded:
        st.info("Sıfır ağırlıklı (ölçülemedi): " + ", ".join(excluded))
    regions = record.get("regions", [])[:10]
    st.markdown(
        _markdown_table(
            ["#", "Sınıf", "Risk", "Seviye", "Baskın etken", "Koridora uzaklık"],
            [
                [
                    str(index),
                    row.get("class_name", "—"),
                    _number(row.get("risk", {}).get("total_risk_score")),
                    row.get("risk", {}).get("risk_category", "—"),
                    (row.get("risk", {}).get("explanation") or ["—"])[0],
                    _number(row.get("corridor_distance_pixels"), 0, " px"),
                ]
                for index, row in enumerate(regions, start=1)
            ],
        )
    )
    st.caption(
        "Bu bir operasyonel dikkat sıralamasıdır, fiziksel risk olasılığı değildir — "
        "kayıt `calibrated_physical_risk_probability: false` der."
    )
    render_drivable(st, drivable)
    show_image(st, root / "qualitative" / "models_same_frame.png", "Aynı karede beş mimari")
    for name, caption in (
        ("overlay.png", "Segmentasyon"),
        ("drivable_corridor.png", "Sürülebilir koridor"),
        ("regions_overlay.png", "Bölgeler"),
        ("attention_map.png", "Operasyonel dikkat"),
    ):
        show_image(st, root / "figures" / "pidnet_s" / name, caption)


def render_drivable(st: Any, drivable: dict) -> None:
    """Put a number next to the corridor picture the claim has rested on until now."""
    if not drivable:
        return
    rows = [drivable[name] for name in MODEL_ORDER if name in drivable]
    rows.extend(record for name, record in sorted(drivable.items()) if name not in MODEL_ORDER)
    if not rows:
        return
    stride = int(rows[0].get("output_stride", 8))
    frames = rows[0].get("frames")

    st.markdown("#### Sürülebilir alan — Cityscapes val, ölçülmüş")
    st.caption(
        f"{frames} kare, karşılaştırma çerçevesinin geri kalanıyla aynı ONNX grafiklerinden. "
        "**Yanlış-sürülebilir**, aracın gireceği ama yol olmayan piksellerin oranıdır — "
        "güvenlik açısından anlamlı olan sayı budur."
    )
    st.markdown(
        _markdown_table(
            [
                "Mimari",
                "Yol IoU",
                "Koridor IoU",
                "Sınır F1 (1 px)",
                f"Sınır F1 ({stride} px)",
                "Yanlış-sürülebilir",
            ],
            [
                [
                    MODEL_LABEL.get(record.get("model", ""), record.get("model", "—")),
                    _number(record.get("road_mask", {}).get("road_iou"), 4),
                    _number(record.get("ego_corridor", {}).get("road_iou"), 4),
                    _number(record.get("road_mask", {}).get("road_boundary_f1_tolerance_1px"), 4),
                    _number(
                        record.get("road_mask", {}).get(f"road_boundary_f1_tolerance_{stride}px"),
                        4,
                    ),
                    _number(record.get("road_mask", {}).get("false_drivable_rate"), 4),
                ]
                for record in rows
            ],
        )
    )
    st.caption(
        f"İki tolerans bilerek yan yana: 1 piksel, stride-{stride} logit'ten büyütülmüş bir "
        f"maskeden çözünürlüğünün izin vermediği bir kesinlik ister; {stride} piksel, "
        "mimarinin gerçekten sorumlu tutulabileceği soruyu sorar. Aradaki fark modelin "
        "başarısızlığı değil, kaba tahmin edip büyütmenin bedelidir."
    )


def render_edge(st: Any, jetson: dict, profiles: dict, root: Path) -> None:
    st.title("Uç cihaz: koşu telemetrisi")
    st.caption(
        "Aşağıdaki eğriler, gösterilen sonuçlar üretilirken kaydedildi — 600 saniye "
        "sürdürülen yük, saniyede bir örnek. Boştaki cihazın değeri değil, ölçümün "
        "yapıldığı andaki gerçek yük."
    )
    logs = (
        sorted((root / "telemetry").glob("*_tegrastats.log"))
        if (root / "telemetry").is_dir()
        else []
    )
    if logs:
        names = [path.name.replace("_reference_tegrastats.log", "") for path in logs]
        chosen = st.selectbox("Koşu", names)
        path = next(p for p, n in zip(logs, names, strict=True) if n == chosen)
        series = tegrastats_series(path)
        if series:
            columns = st.columns(3)
            for column, (label, values) in zip(columns, series.items(), strict=False):
                column.markdown(f"**{label}**")
                column.line_chart({label: values}, height=200)
            st.caption(
                f"{path.name} · {len(next(iter(series.values())))} örnek · "
                "sıcaklık eğrisinin düzleşmesi termal kısma olmadığını gösterir"
            )
        else:
            st.warning("Telemetri okunamadı.")
    else:
        st.info("Telemetri logları bulunamadı (`telemetry/*_tegrastats.log`).")

    st.divider()
    rows = []
    for model in MODEL_ORDER:
        record = jetson.get(f"{model}_reference_benchmark")
        if not record:
            continue
        telemetry = record.get("telemetry", {})
        rows.append(
            [
                MODEL_LABEL[model],
                _number((record.get("pure_engine_latency_ms") or {}).get("median"), 2, " ms"),
                _number((record.get("end_to_end_latency_ms") or {}).get("median"), 1, " ms"),
                _number(record.get("sustained_fps"), 2),
                _number(telemetry.get("mean_input_power_w"), 2, " W"),
                _number(record.get("joule_per_frame"), 3, " J"),
                f"{telemetry.get('peak_ram_used_mib', '—')} MiB",
                _number(
                    (telemetry.get("temperature_c", {}).get("gpu") or {}).get("peak"), 1, " °C"
                ),
            ]
        )
    if rows:
        st.markdown(
            _markdown_table(
                ["Model", "Motor", "Kare", "FPS", "Güç", "J/kare", "Tepe RAM", "Tepe GPU"], rows
            )
        )
    st.divider()
    st.markdown("#### Kare bütçesi nereye gidiyor")
    st.markdown(
        _markdown_table(
            ["Aşama", "İlk ölçüm", "Optimizasyon sonrası", "Pay"],
            [
                ["derive_perception", "96,05 ms", "43,64 ms", "46,4%"],
                ["preprocess", "24,80 ms", "24,71 ms", "26,3%"],
                ["görüntü çözme", "13,97 ms", "14,00 ms", "14,9%"],
                ["confidence_entropy", "5,78 ms", "5,80 ms", "6,2%"],
                ["**motor**", "**5,09 ms**", "**5,09 ms**", "**5,4%**"],
                ["argmax", "0,75 ms", "0,74 ms", "0,8%"],
                ["**toplam**", "**146,44 ms**", "**93,98 ms**", ""],
            ],
        )
    )
    show_figure(st, root, "D4_frame_budget_flow", "Kare bütçesi akışı")
    show_figure(st, root, "07_frame_budget", "Kare bütçesi ve optimizasyon etkisi")
    video = root / "video" / "edgeguard_demo.mp4"
    if video.is_file():
        st.markdown("#### Hareketli sahnede segmentasyon ve belirsizlik")
        st.video(str(video))
        st.caption("Cityscapes demoVideo · 180 kare · PIDNet-S referans · önceden üretildi")
    st.success(
        "TensorRT motoru kare bütçesinin yalnızca **%5,4'ü**. İki hedefli CPU tarafı "
        "optimizasyonuyla kare **1,88× hızlandı**, enerji **1,63× iyileşti**, modele hiç "
        "dokunulmadı ve çıktılar birebir aynı kaldı."
    )
    st.caption(
        "Etiketleme optimizasyonu kararı hedef cihazda verildi: geliştirme Mac'inde BFS "
        "1,4× kazanıyordu, Jetson'da vektörleştirme 1,59× kazandı. Uç cihaz optimizasyon "
        "kararları geliştirme makinesinde alınamıyor."
    )


def render_limits(st: Any) -> None:
    st.title("Sınırlar ve ölçülmeyenler")
    st.markdown(
        """
Hiçbiri "ölçülmüş" gibi sunulmuyor.

| Eksik | Neden |
|---|---|
| Gerçek zaman kapısı | **geçilmedi** — en iyi 12,36 FPS; darboğaz CPU tarafı, model değil |
| SegFormer stride düzeltmesi | önerildi, ölçülmedi |
| Sürülebilir alan sayısal metriği | GT yol maskeleri gerekiyor |
| FP16 dağıtım sadakati | ölçüm scripti hazır, koşulmayı bekliyor |
| Mühürlü final test verisi | **kasıtlı** — yalnızca insan tetikler |

#### Yöntemsel sınırlar

- **n = 5 mimari.** ρ = −0,90 güçlü bir eğilim, kesin kanıt değil.
- Referans checkpoint'ler farklı reçetelerle eğitilmiştir; kontrollü ablasyon değil,
  yayınlanmış model karşılaştırmasıdır.
- Kendi eğitilen modeller 2.500 adımda kalmıştır — yayınların %0,7'si kadar örnek.
  Sınırlı bütçe altında çok-domainli genelleme deneyi olarak konumlandırılmıştır.
- Sentetik hava bozulmaları algoritmiktir; gerçek ACDC ölçümleri ayrıca verilmiştir.
"""
    )


if __name__ == "__main__":
    main()
