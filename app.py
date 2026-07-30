"""Streamlit image demo for the verified semantic-first backends."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from edgeguard.rescue.dataset import CITYSCAPES_CLASSES
from edgeguard.rescue.inference import discover_demo_models, predict_mmseg, predict_onnx
from edgeguard.rescue.mmseg_runtime import CITYSCAPES_PALETTE
from edgeguard.rescue.perception import attention_contract, derive_perception
from edgeguard.rescue.shift import (
    apply_shift_reference,
    frame_uncertainty_summary,
    uncertainty_maps,
)
from edgeguard.rescue.visualization import (
    calibrate_inference_result,
    colorize_mask,
    overlay_mask,
    resize_scalar,
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
    entropy_threshold = st.sidebar.slider("Entropy threshold", 0.0, 1.0, 0.5, 0.05)
    minimum_region_area = st.sidebar.number_input(
        "Minimum region area", min_value=1, max_value=10000, value=32
    )
    calibration_file = st.sidebar.file_uploader(
        "Global temperature artifact (optional)", type=("json",)
    )
    shift_file = st.sidebar.file_uploader("Source shift reference (optional)", type=("json",))
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
        perception = derive_perception(
            result.mask,
            result.confidence,
            result.entropy,
            confidence_threshold=confidence_threshold,
            entropy_threshold=entropy_threshold,
            minimum_region_area=int(minimum_region_area),
        )
        region_overlay = image.copy()
        drawer = ImageDraw.Draw(region_overlay)
        colors = {"low": "green", "medium": "orange", "high": "red"}
        for region in perception.regions:
            x1, y1, x2, y2 = region.bbox_xyxy
            drawer.rectangle(
                (x1, y1, max(x1, x2 - 1), max(y1, y2 - 1)),
                outline=colors[region.attention_level],
                width=2,
            )
        corridor_overlay = image.copy()
        cyan = Image.new("RGB", image.size, color=(0, 220, 220))
        corridor_mask = Image.fromarray(perception.drivable_corridor.astype(np.uint8) * 150)
        corridor_overlay = Image.composite(cyan, corridor_overlay, corridor_mask)
        columns = st.columns(4)
        columns[0].image(image, caption="Input", use_container_width=True)
        columns[1].image(
            overlay_mask(image, result.mask, opacity),
            caption="Segmentation overlay",
            use_container_width=True,
        )
        columns[2].image(
            corridor_overlay,
            caption="Ego-reachable drivable corridor",
            use_container_width=True,
        )
        columns[3].image(
            region_overlay,
            caption="Semantic regions · operational attention",
            use_container_width=True,
        )
        st.metric("Inference latency", f"{result.latency_ms:.2f} ms")
        st.caption(f"Backend: {result.backend}; UI süresi latency ölçümüne dahil değildir.")
        st.caption(f"Kalibrasyon: {calibration_status}.")
        low_confidence = float(np.mean(result.confidence < confidence_threshold))
        st.metric("Low-confidence pixel ratio", f"{low_confidence:.2%}")
        energy = None
        if result.logits is not None:
            energy = uncertainty_maps(result.logits)["energy"]
            if energy.shape != result.mask.shape:
                energy = resize_scalar(energy, image.size)
        frame_summary = frame_uncertainty_summary(
            result.confidence,
            result.entropy,
            energy=energy,
            confidence_threshold=confidence_threshold,
        )
        if shift_file is not None:
            reference = json.loads(shift_file.getvalue().decode("utf-8"))
            if selected["backend"] == "onnx":
                validation_path = Path(selected["model"]).with_suffix(".validation.json")
                if not validation_path.is_file():
                    raise ValueError("shift-aware ONNX display requires its validation record")
                validation = json.loads(validation_path.read_text(encoding="utf-8"))
                checkpoint_sha256 = validation.get("checkpoint_sha256")
            else:
                checkpoint_sha256 = sha256_file(Path(selected["model"]))
            if checkpoint_sha256 != reference.get("checkpoint_sha256"):
                raise ValueError("shift reference does not match the selected model")
            shift = apply_shift_reference(frame_summary, reference)
            if shift["domain_shift_alert"]:
                st.error(
                    f"Domain-shift warning: {shift['alert_votes']}/"
                    f"{shift['available_votes']} source thresholds exceeded"
                )
            else:
                st.success("No source-calibrated domain-shift alert for this frame")
        else:
            st.info("Domain-shift alert unavailable: source calibration artifact not loaded")
        uncertainty_columns = st.columns(3)
        uncertainty_columns[0].image(
            Image.fromarray(np.clip(result.confidence * 255, 0, 255).astype(np.uint8)),
            caption="Maximum-softmax confidence",
            use_container_width=True,
        )
        uncertainty_columns[1].image(
            Image.fromarray(np.clip(result.entropy * 255, 0, 255).astype(np.uint8)),
            caption="Normalized entropy",
            use_container_width=True,
        )
        uncertainty_columns[2].image(
            Image.fromarray(np.clip(perception.attention_map * 255, 0, 255).astype(np.uint8)),
            caption="Operational attention map",
            use_container_width=True,
        )
        st.caption(
            "Operational attention is a deterministic review aid, not collision probability or "
            "a safety-certified decision."
        )
        region_rows = [region.to_dict() for region in perception.regions]
        if region_rows:
            st.dataframe(region_rows, use_container_width=True, hide_index=True)
        with st.expander("Attention contract"):
            st.json(attention_contract())
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
