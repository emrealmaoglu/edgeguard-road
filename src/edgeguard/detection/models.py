"""Bounded real YOLO11n and RT-DETR-R18 local codepath validation."""

from __future__ import annotations

import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from edgeguard.data.transforms import LetterboxReceipt, invert_letterbox_boxes
from edgeguard.detection.contracts import Detection, box_mask_overlap
from edgeguard.serialization import sha256_file, sha256_payload


def _checkpoint_round_trip(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    path: Path,
    identity: str,
    builder: Any,
    torch: Any,
) -> dict[str, Any]:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "identity": identity,
        },
        path,
    )
    restored = torch.load(path, map_location="cpu", weights_only=True)
    resumed = builder()
    resumed.load_state_dict(restored["model"], strict=True)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
    resumed_optimizer.load_state_dict(restored["optimizer"])
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, 1, gamma=0.9)
    resumed_scheduler.load_state_dict(restored["scheduler"])
    return {
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_size_bytes": path.stat().st_size,
        "resume_identity": restored["identity"],
        "exact_resume": restored["identity"] == identity,
    }


def _build_yolo(torch: Any) -> Any:
    from ultralytics.cfg import get_cfg  # type: ignore[import-untyped]
    from ultralytics.nn.tasks import DetectionModel  # type: ignore[import-untyped]

    model = DetectionModel("yolo11n.yaml", ch=3, nc=10, verbose=False)
    model.args = get_cfg()
    return model.to(torch.device("cpu"))


def _build_rtdetr(torch: Any) -> Any:
    from transformers import (
        RTDetrConfig,
        RTDetrForObjectDetection,
        RTDetrResNetConfig,
    )

    backbone = RTDetrResNetConfig(
        depths=[2, 2, 2, 2],
        hidden_sizes=[64, 128, 256, 512],
        layer_type="basic",
        out_features=["stage2", "stage3", "stage4"],
    )
    config = RTDetrConfig(
        backbone_config=backbone,
        num_labels=10,
        encoder_layers=1,
        decoder_layers=1,
        encoder_in_channels=[128, 256, 512],
        decoder_in_channels=[256, 256, 256],
        encoder_hidden_dim=256,
        d_model=256,
        num_queries=50,
    )
    return RTDetrForObjectDetection(config).to(torch.device("cpu"))


def _common_predictions(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    *,
    source_size_hw: tuple[int, int],
) -> list[dict[str, Any]]:
    receipt = LetterboxReceipt(source_size_hw, (128, 128), 1.0, (0.0, 0.0))
    restored = invert_letterbox_boxes(np.asarray(boxes, dtype=np.float32), receipt)
    road_mask = np.zeros(source_size_hw, dtype=np.bool_)
    road_mask[source_size_hw[0] // 2 :, :] = True
    output: list[dict[str, Any]] = []
    for box, score, class_id in zip(restored, scores, classes, strict=True):
        box_tuple = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        detection = Detection(
            box_xyxy=box_tuple,
            class_id=int(class_id),
            confidence=float(score),
        )
        output.append(
            {
                **detection.to_dict(),
                "class_name": f"class_{int(class_id)}",
                "image_id": "synthetic-bdd-frame-0001",
                "road_overlap": box_mask_overlap(detection, road_mask),
            }
        )
    return output


def _run_yolo(output_root: Path, torch: Any, optimizer_steps: int) -> dict[str, Any]:
    from ultralytics.utils.ops import non_max_suppression  # type: ignore[import-untyped]

    torch.manual_seed(20260727)
    model = _build_yolo(torch).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    losses: list[float] = []
    loss_components: list[list[float]] = []
    started = time.perf_counter()
    for _step in range(optimizer_steps):
        batch = {
            "img": torch.rand(1, 3, 128, 128),
            "cls": torch.tensor([[2.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
            "batch_idx": torch.tensor([0.0]),
        }
        components, detached = model(batch)
        loss = components.sum()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("YOLO11n mini loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        loss_components.append([float(value) for value in detached.detach()])
    model.eval()
    with torch.no_grad():
        raw = model(torch.rand(1, 3, 128, 128))[0]
    decoded = non_max_suppression(raw, conf_thres=0.001, max_det=5)[0]
    if decoded.numel():
        boxes = decoded[:, :4].numpy()
        scores = decoded[:, 4].numpy()
        classes = decoded[:, 5].numpy()
    else:
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
        classes = np.empty((0,), dtype=np.int64)
    identity = sha256_payload({"model": "yolo11n", "version": "8.3.175", "steps": optimizer_steps})
    checkpoint = _checkpoint_round_trip(
        model,
        optimizer,
        scheduler,
        output_root / "yolo11n.pt",
        identity,
        lambda: _build_yolo(torch),
        torch,
    )
    return {
        "model_family": "yolo11n",
        "implementation": "ultralytics",
        "distribution_version": importlib.metadata.version("ultralytics"),
        "source_license": "AGPL-3.0",
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "losses": losses,
        "loss_components_box_cls_dfl": loss_components,
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": time.perf_counter() - started,
        "raw_prediction_shape": list(raw.shape),
        "decoded_predictions": _common_predictions(
            boxes, scores, classes, source_size_hw=(128, 128)
        ),
        **checkpoint,
        "scientific_evidence": False,
    }


def _run_rtdetr(output_root: Path, torch: Any, optimizer_steps: int) -> dict[str, Any]:
    torch.manual_seed(20260728)
    model = _build_rtdetr(torch).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    losses: list[float] = []
    loss_components: list[dict[str, float]] = []
    started = time.perf_counter()
    pixels = torch.rand(1, 3, 128, 128)
    labels = [{"class_labels": torch.tensor([2]), "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]])}]
    for _step in range(optimizer_steps):
        output = model(pixel_values=pixels, labels=labels)
        loss = output.loss
        if loss is None or not bool(torch.isfinite(loss)):
            raise FloatingPointError("RT-DETR-R18 mini loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        loss_components.append(
            {name: float(value.detach()) for name, value in output.loss_dict.items()}
        )
    model.eval()
    with torch.no_grad():
        prediction = model(pixel_values=pixels)
    probabilities = prediction.logits.sigmoid()[0]
    scores, classes = probabilities.max(dim=-1)
    top_scores, indices = torch.topk(scores, k=min(5, int(scores.numel())))
    selected_classes = classes[indices]
    cxcywh = prediction.pred_boxes[0, indices]
    scale = torch.tensor([128.0, 128.0, 128.0, 128.0])
    values = cxcywh * scale
    boxes = torch.stack(
        (
            values[:, 0] - values[:, 2] / 2,
            values[:, 1] - values[:, 3] / 2,
            values[:, 0] + values[:, 2] / 2,
            values[:, 1] + values[:, 3] / 2,
        ),
        dim=1,
    ).clamp(0, 128)
    identity = sha256_payload(
        {"model": "rt_detr_r18", "version": "4.53.2", "steps": optimizer_steps}
    )
    checkpoint = _checkpoint_round_trip(
        model,
        optimizer,
        scheduler,
        output_root / "rt_detr_r18.pt",
        identity,
        lambda: _build_rtdetr(torch),
        torch,
    )
    return {
        "model_family": "rt_detr_r18",
        "implementation": "transformers",
        "distribution_version": importlib.metadata.version("transformers"),
        "source_license": "Apache-2.0",
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "losses": losses,
        "loss_components": loss_components,
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": time.perf_counter() - started,
        "raw_logits_shape": list(prediction.logits.shape),
        "raw_boxes_shape": list(prediction.pred_boxes.shape),
        "decoded_predictions": _common_predictions(
            boxes.numpy(), top_scores.numpy(), selected_classes.numpy(), source_size_hw=(128, 128)
        ),
        **checkpoint,
        "scientific_evidence": False,
    }


def run_detector_mini_training(output_root: Path, *, optimizer_steps: int = 2) -> dict[str, Any]:
    """Run two actual weight-free detector mini-training and resume paths."""
    if not 2 <= optimizer_steps <= 5:
        raise ValueError("detector mini training requires 2..5 optimizer steps")
    torch = __import__("torch")
    output_root.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for name, operation in (("yolo11n", _run_yolo), ("rt_detr_r18", _run_rtdetr)):
        try:
            results.append(operation(output_root, torch, optimizer_steps))
        except Exception as error:
            failures.append(
                {
                    "model_family": name,
                    "classification": type(error).__name__,
                    "error": str(error)[:2000],
                }
            )
    report = {
        "schema_version": "1.0",
        "record_type": "two_real_detector_mini_training",
        "results": results,
        "failures": failures,
        "all_models_succeeded": len(results) == 2,
        "fallback_activated": False,
        "ranking_permitted": False,
        "scientific_evidence": False,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return report


def probe_detector_onnx(output_root: Path) -> dict[str, Any]:
    """Export both actual random-weight detectors where the local stack supports it."""
    torch = __import__("torch")
    onnx = __import__("onnx")
    ort = __import__("onnxruntime")
    output_root.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    class YoloWrapper(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model

        def forward(self, inputs: Any) -> Any:
            return self.model(inputs)[0]

    class RTDetrWrapper(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self, model: Any) -> None:
            super().__init__()
            self.model = model

        def forward(self, inputs: Any) -> tuple[Any, Any]:
            output = self.model(pixel_values=inputs)
            return output.logits, output.pred_boxes

    builders: tuple[tuple[str, Any, list[str]], ...] = (
        ("yolo11n", lambda: YoloWrapper(_build_yolo(torch).eval()).eval(), ["predictions"]),
        (
            "rt_detr_r18",
            lambda: RTDetrWrapper(_build_rtdetr(torch).eval()).eval(),
            ["class_logits", "boxes_cxcywh"],
        ),
    )
    for index, (name, builder, output_names) in enumerate(builders):
        try:
            torch.manual_seed(20260729 + index)
            model = builder()
            inputs = torch.rand(1, 3, 128, 128)
            with torch.no_grad():
                expected_output = model(inputs)
            expected = (
                [item.detach().numpy() for item in expected_output]
                if isinstance(expected_output, tuple)
                else [expected_output.detach().numpy()]
            )
            path = output_root / f"{name}.onnx"
            started = time.perf_counter()
            torch.onnx.export(
                model,
                inputs,
                path,
                input_names=["model_input"],
                output_names=output_names,
                opset_version=17,
                dynamo=False,
            )
            graph = onnx.load(path)
            onnx.checker.check_model(graph)
            actual = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
                output_names, {"model_input": inputs.numpy()}
            )
            differences = [
                float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
                for left, right in zip(expected, actual, strict=True)
            ]
            results.append(
                {
                    "model_family": name,
                    "output_names": output_names,
                    "output_shapes": [list(item.shape) for item in actual],
                    "max_absolute_differences": differences,
                    "allclose_atol_1e_4_rtol_1e_4": all(
                        np.allclose(left, right, atol=1e-4, rtol=1e-4)
                        for left, right in zip(expected, actual, strict=True)
                    ),
                    "operator_inventory": sorted({node.op_type for node in graph.graph.node}),
                    "onnx_byte_size": path.stat().st_size,
                    "onnx_sha256": sha256_file(path),
                    "export_seconds": time.perf_counter() - started,
                    "scientific_evidence": False,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "model_family": name,
                    "classification": type(error).__name__,
                    "error": str(error)[:2000],
                }
            )
    report = {
        "schema_version": "1.0",
        "record_type": "actual_detector_onnx_probe",
        "results": results,
        "failures": failures,
        "scientific_evidence": False,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return report
