"""Static safety contract for the manually triggered Linux CPU framework gate."""

from pathlib import Path


def test_linux_cpu_probe_is_manual_bounded_and_mmcv_lite_only() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / ".github/workflows/semantic-framework-cpu-probe.yml").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in source
    assert "push:" in source
    assert "branches:" in source
    assert "feat/first-vertical-slice" in source
    for restricted_path in (
        ".github/workflows/semantic-framework-cpu-probe.yml",
        "scripts/dev/**",
        "scripts/train/**",
        "src/edgeguard/**",
        "configs/training/segmentation/**",
        "tests/**",
    ):
        assert restricted_path in source
    assert "pull_request:" not in source
    assert "timeout-minutes: 35" in source
    assert 'python-version: "3.12"' in source
    assert "mmengine==0.10.7 mmcv-lite==2.1.0" in source
    assert "mmcv==2.1.0" not in source
    assert "c685fe6767c4cadf6b051983ca6208f1b9d1ccb8" in source
    assert "validate_local_readiness.py" in source
    assert "--profile linux-cpu" in source
    assert "--mmseg-checkout /tmp/mmsegmentation" in source
    assert "actions/upload-artifact@v4" in source
    assert "Cityscapes" not in source
    assert "CUDA" not in source
