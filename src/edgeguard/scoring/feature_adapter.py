"""Real semantic feature adapters and a minimal exportable anomaly head."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from edgeguard.serialization import sha256_file, sha256_payload


def train_feature_anomaly_head(
    feature: Any,
    target: Any,
    valid_mask: Any,
    output_dir: Path,
    *,
    model_family: str,
    feature_identity: str,
    optimizer_steps: int = 2,
) -> dict[str, Any]:
    """Optimize a 1x1 feature adapter/head on one deterministic valid-mask target."""
    torch = __import__("torch")
    if not torch.is_tensor(feature) or feature.ndim != 4 or not torch.isfinite(feature).all():
        raise ValueError("anomaly feature input must be one finite NCHW Torch tensor")
    if not torch.is_tensor(target) or not torch.is_tensor(valid_mask):
        raise ValueError("anomaly target and valid mask must be Torch tensors")
    if target.ndim != 3 or valid_mask.shape != target.shape or valid_mask.dtype != torch.bool:
        raise ValueError("anomaly target/valid-mask contract mismatch")
    if not 2 <= optimizer_steps <= 5:
        raise ValueError("feature anomaly mini training requires 2..5 steps")
    torch.manual_seed(20260727)
    adapter = torch.nn.Sequential(
        torch.nn.Conv2d(int(feature.shape[1]), 16, kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 1, kernel_size=1),
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    losses: list[float] = []
    for _step in range(optimizer_steps):
        logits = adapter(feature.detach())
        aligned = torch.nn.functional.interpolate(
            logits, size=target.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0]
        point_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            aligned, target.float(), reduction="none"
        )
        if not bool(valid_mask.any()):
            raise ValueError("anomaly head requires a non-empty valid mask")
        loss = point_loss[valid_mask].mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("anomaly feature-head loss is non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        score = torch.sigmoid(
            torch.nn.functional.interpolate(
                adapter(feature), size=target.shape[-2:], mode="bilinear", align_corners=False
            )[:, 0]
        )
    if not bool(torch.isfinite(score).all()):
        raise ValueError("anomaly feature-head score is non-finite")
    output_dir.mkdir(parents=True, exist_ok=False)
    identity = sha256_payload(
        {
            "model_family": model_family,
            "feature_identity": feature_identity,
            "input_channels": int(feature.shape[1]),
            "steps": optimizer_steps,
        }
    )
    checkpoint = output_dir / "anomaly_head.pt"
    torch.save(
        {
            "model": adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "identity": identity,
        },
        checkpoint,
    )
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    result = {
        "model_family": model_family,
        "feature_identity": feature_identity,
        "feature_shape": list(feature.shape),
        "feature_scale_hw": list(feature.shape[-2:]),
        "adapter": "conv1x1_relu_conv1x1",
        "loss": "bce_with_logits_valid_mask",
        "losses": losses,
        "score_shape": list(score.shape),
        "score_direction": "higher_means_more_anomalous",
        "checkpoint_sha256": sha256_file(checkpoint),
        "resume_identity": restored["identity"],
        "scientific_evidence": False,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return result
