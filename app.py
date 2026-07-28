"""Streamlit image demo for the verified semantic-first backends."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from edgeguard.rescue.dataset import CITYSCAPES_CLASSES
from edgeguard.rescue.inference import discover_demo_models, predict_mmseg, predict_onnx
from edgeguard.rescue.mmseg_runtime import CITYSCAPES_PALETTE
from edgeguard.rescue.visualization import (
    calibrate_inference_result,
    colorize_mask,
    overlay_mask,
)
from edgeguard.serialization import sha256_file


def main() -> None:
    """Render the offline semantic inference application."""
    try:
        st = __import__("streamlit")
    except ModuleNotFoundError as error:
        raise RuntimeError("install the rescue dependencies: pip install -e '.[rescue]'") from error

    st.set_page_config(page_title="EdgeGuard-Road", layout="wide")
    st.title("EdgeGuard-Road · Semantic-First Demo")
    st.caption(
        "Bu arayüz bilimsel değerlendirme yerine tek görüntü inference ve görsel inceleme içindir."
    )
    default_root = Path(os.environ.get("EDGEGUARD_RUN_ROOT", "runs"))
    run_root = Path(st.sidebar.text_input("Run root", str(default_root))).expanduser()
    records = discover_demo_models(run_root)
    if not records:
        st.warning(
            "Doğrulanmış model bulunamadı. Bir ONNX dosyasını run root altına koyun veya "
            "resolved.py ile checkpoint üretin."
        )
        return
    labels = [record["label"] for record in records]
    selected = records[labels.index(st.sidebar.selectbox("Model", labels))]
    st.sidebar.caption(f"Eğitim domainleri: {selected.get('datasets', 'artifact metadata yok')}")
    device = st.sidebar.selectbox("Device", ("cpu", "cuda"), index=0)
    opacity = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.55, 0.05)
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    calibration_file = st.sidebar.file_uploader(
        "Global temperature artifact (optional)", type=("json",)
    )
    uploaded = st.file_uploader("Yol sahnesi görüntüsü yükleyin", type=("png", "jpg", "jpeg"))
    if uploaded is None:
        return
    image = Image.open(uploaded).convert("RGB")
    if st.button("Inference", type="primary"):
        with st.spinner("Model çalışıyor..."):
            if selected["backend"] == "onnx":
                result = predict_onnx(image, Path(selected["model"]))
            else:
                active_device = device
                if device == "cuda":
                    try:
                        torch = __import__("torch")
                        cuda_available = bool(torch.cuda.is_available())
                    except ModuleNotFoundError:
                        cuda_available = False
                    if not cuda_available:
                        st.warning("CUDA kullanılamıyor; inference CPU üzerinde çalıştırıldı.")
                        active_device = "cpu"
                result = predict_mmseg(
                    image,
                    Path(selected["config"]),
                    Path(selected["model"]),
                    device=active_device,
                )
        calibration_status = "raw"
        if calibration_file is not None:
            calibration = json.loads(calibration_file.getvalue().decode("utf-8"))
            if calibration.get("record_type") != "multi_domain_global_temperature":
                raise ValueError("only a multi-domain global temperature artifact is accepted")
            if selected["backend"] == "onnx":
                validation_path = Path(selected["model"]).with_suffix(".validation.json")
                if not validation_path.is_file():
                    raise ValueError("calibrated ONNX display requires its validation record")
                validation = json.loads(validation_path.read_text(encoding="utf-8"))
                checkpoint_sha256 = validation.get("checkpoint_sha256")
            else:
                checkpoint_sha256 = sha256_file(Path(selected["model"]))
            if checkpoint_sha256 != calibration.get("checkpoint_sha256"):
                raise ValueError("temperature artifact does not match the selected checkpoint")
            result = calibrate_inference_result(result, float(calibration["final_temperature"]))
            calibration_status = "global temperature applied"
        columns = st.columns(3)
        columns[0].image(image, caption="Input", use_container_width=True)
        columns[1].image(
            overlay_mask(image, result.mask, opacity),
            caption="Segmentation overlay",
            use_container_width=True,
        )
        columns[2].image(
            Image.fromarray(np.clip(result.entropy * 255, 0, 255).astype(np.uint8)),
            caption="Normalized entropy",
            use_container_width=True,
        )
        st.metric("Inference latency", f"{result.latency_ms:.2f} ms")
        st.caption(f"Backend: {result.backend}; UI süresi latency ölçümüne dahil değildir.")
        st.caption(f"Kalibrasyon: {calibration_status}.")
        low_confidence = float(np.mean(result.confidence < confidence_threshold))
        st.metric("Low-confidence pixel ratio", f"{low_confidence:.2%}")
        counts = np.bincount(result.mask.reshape(-1), minlength=19)
        distribution = [
            {
                "class": name,
                "pixel_ratio": float(counts[index] / result.mask.size),
                "color": f"rgb({','.join(str(value) for value in CITYSCAPES_PALETTE[index])})",
            }
            for index, name in enumerate(CITYSCAPES_CLASSES)
            if counts[index] > 0
        ]
        st.dataframe(distribution, use_container_width=True, hide_index=True)
        st.image(colorize_mask(result.mask), caption="Palette mask", use_container_width=False)


if __name__ == "__main__":
    main()
