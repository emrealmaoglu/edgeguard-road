# ruff: noqa: E501 -- embeds a real, verbatim Colab log excerpt as a test fixture
from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgeguard.rescue.training_log_analysis import (
    build_analysis_report,
    flag_near_absent_classes,
    load_real_class_frequency,
    parse_training_log,
    project_reduced_class_miou,
    render_markdown_report,
    write_analysis_report,
)

# Real, verbatim excerpt from a real Colab L4 run (segformer_b0, pilot stage,
# application commit f2f2110), pasted by the project owner during this session. Not a
# synthetic/hand-built fixture -- this is genuine observed mmengine output, chosen
# specifically so the parser is tested against the exact real format it must handle.
REAL_SEGFORMER_B0_PILOT_LOG = """\
08/12 11:40:22 - mmengine - INFO - Iter(train) [  50/2000]  lr: 4.9841e-05  eta: 0:11:24  time: 0.2934  data_time: 0.0044  memory: 3725  grad_norm: 3.3291  loss: 2.5882  decode.loss_ce: 2.5882  decode.acc_seg: 46.7585
08/12 11:45:02 - mmengine - INFO - Iter(train) [1000/2000]  lr: 3.3511e-05  eta: 0:04:57  time: 0.2921  data_time: 0.0046  memory: 3444  grad_norm: 10.6306  loss: 1.2301  decode.loss_ce: 1.2301  decode.acc_seg: 55.8981
08/12 11:49:58 - mmengine - INFO - Iter(train) [2000/2000]  lr: 1.0000e-06  eta: 0:00:00  time: 0.2905  data_time: 0.0046  memory: 3444  grad_norm: 11.4647  loss: 1.0727  decode.loss_ce: 1.0727  decode.acc_seg: 64.5374
08/12 11:49:58 - mmengine - INFO - Saving checkpoint at 2000 iterations
08/12 11:50:03 - mmengine - INFO - Iter(val) [  50/2560]    eta: 0:01:49  time: 0.0280  data_time: 0.0029  memory: 579
08/12 11:51:12 - mmengine - INFO - Iter(val) [2550/2560]    eta: 0:00:00  time: 0.0246  data_time: 0.0024  memory: 449
08/12 11:51:12 - mmengine - INFO - per class results:
08/12 11:51:12 - mmengine - INFO -
+---------------+-------+-------+
|     Class     |  IoU  |  Acc  |
+---------------+-------+-------+
|      road     | 76.84 | 87.05 |
|    sidewalk   |  0.0  |  0.0  |
|    building   | 33.82 | 42.58 |
|      wall     |  0.0  |  0.0  |
|     fence     |  0.0  |  0.0  |
|      pole     |  0.0  |  0.0  |
| traffic light |  0.0  |  0.0  |
|  traffic sign |  0.0  |  0.0  |
|   vegetation  | 40.25 | 89.04 |
|    terrain    |  0.0  |  0.0  |
|      sky      | 70.01 | 78.86 |
|     person    |  0.0  |  0.0  |
|     rider     |  0.0  |  0.0  |
|      car      | 24.47 | 36.98 |
|     truck     |  0.0  |  0.0  |
|      bus      |  0.0  |  0.0  |
|     train     |  0.0  |  0.0  |
|   motorcycle  |  0.0  |  0.0  |
|    bicycle    |  0.0  |  0.0  |
+---------------+-------+-------+
08/12 11:51:12 - mmengine - INFO - Iter(val) [2560/2560]    aAcc: 66.4100  mIoU: 12.9100  mAcc: 17.6100  data_time: 0.0029  time: 0.0276
08/12 11:51:14 - mmengine - INFO - The best checkpoint with 12.9100 mIoU at 2000 iter is saved to best_mIoU_iter_2000.pth.
{"checkpoints":[{"bytes":45101896,"filename":"iter_1500.pth","sha256":"f7b5d1c7b687eab43e8e66740b33687f519c60056516f1ab4ddf2c1ea91b59ad"},{"bytes":15351662,"filename":"best_mIoU_iter_2000.pth","sha256":"9db2714e89436574f57129777ed2e05e33017b86ce0c31f07c42920dd7ff6c97"},{"bytes":45256328,"filename":"iter_2000.pth","sha256":"4e8902a96caf80c5a8942e569b3d78e4df591310f93bd1be832aaa953aa69523"}],"dataset_manifest_sha256s":["367a0d231fd503a60bf915161c368f9c8803a3817e07f7017819bbe9c3eed770","4b6804118b2f0ea23ac899f29fdb7373a1b000b01af803c720ace5f5e07b9e2c"],"datasets":["cityscapes","idd20k"],"device_batch":4,"domain_sampling":"uniform","effective_batch":4,"elapsed_seconds":681.286115728999,"environment":{"cudnn_benchmark":true,"device":"cuda","gpu":"NVIDIA L4","mmengine":"0.10.7","mmseg":"1.2.2","peak_gpu_memory_bytes":464816640,"python":"3.11.13","tf32":true,"torch":"2.1.1+cu121"},"gradient_accumulation":1,"initialization":"random","last_checkpoint":"iter_2000.pth","loss":"ce","model":"segformer_b0","ontology_sha256s":["58ae54e634b2deeccd51d576aeaf255ddf5427ee135e21054a76d129c5a3c686"],"precision":"bf16","record_type":"semantic_training_summary","resolved_config_sha256":"519e66878b4baad7bfb07991dc35c9153795a34aa247f69e4dbfd3649102790e","restored_from_drive":false,"schema_version":"1.0","scientific_evidence":true,"seed":20260728,"stage":"pilot","workers":6}
"""


def test_parse_training_log_extracts_real_per_class_table() -> None:
    results = parse_training_log(REAL_SEGFORMER_B0_PILOT_LOG, source="real_excerpt")
    assert len(results) == 1
    result = results[0]
    assert result.model == "segformer_b0"
    assert result.stage == "pilot"
    assert result.per_class_iou["road"] == 76.84
    assert result.per_class_iou["sidewalk"] == 0.0
    assert result.per_class_iou["building"] == 33.82
    assert len(result.per_class_iou) == 19
    assert result.miou == 12.91
    assert result.aacc == 66.41
    assert result.macc == 17.61
    assert result.elapsed_seconds == pytest.approx(681.286115728999)
    assert result.iter_seconds == pytest.approx(0.2905)
    assert result.parse_warnings == []


def test_parse_training_log_handles_multiple_runs_in_one_capture() -> None:
    doubled = REAL_SEGFORMER_B0_PILOT_LOG + REAL_SEGFORMER_B0_PILOT_LOG.replace(
        '"stage":"pilot"', '"stage":"screening"'
    )
    results = parse_training_log(doubled, source="doubled")
    assert [result.stage for result in results] == ["pilot", "screening"]
    assert results[1].model == "segformer_b0"
    assert results[1].per_class_iou["road"] == 76.84


def test_parse_training_log_flags_missing_per_class_table_instead_of_silently_dropping() -> None:
    stripped = REAL_SEGFORMER_B0_PILOT_LOG.split("per class results:")[0] + (
        '\n{"elapsed_seconds":1.0,"model":"x","record_type":"semantic_training_summary",'
        '"schema_version":"1.0","stage":"y"}\n'
    )
    results = parse_training_log(stripped, source="stripped")
    assert len(results) == 1
    assert results[0].per_class_iou == {}
    assert any("per class results" in warning for warning in results[0].parse_warnings)


def test_load_real_class_frequency_from_real_schema(tmp_path: Path) -> None:
    # Mirrors write_train_fit_statistics's exact real output schema
    # (src/edgeguard/rescue/dataset.py:729-738), 19 values in CITYSCAPES_CLASSES order.
    counts = [100, 0, 50] + [0] * 16
    class_weights_path = tmp_path / "class_weights.json"
    class_weights_path.write_text(
        json.dumps({"pixel_counts": counts, "source_role": "train_fit"}), encoding="utf-8"
    )
    frequency = load_real_class_frequency(class_weights_path)
    assert frequency["pixel_counts_measured"]["road"] == 100
    assert frequency["pixel_counts_measured"]["building"] == 50
    assert frequency["total_pixels_measured"] == 150
    assert frequency["sources"]["cityscapes"] == str(class_weights_path)
    assert frequency["sources"]["idd20k"] == "not_available: no path provided"


def test_load_real_class_frequency_merges_idd20k_summary(tmp_path: Path) -> None:
    counts = [100, 0, 50] + [0] * 16
    class_weights_path = tmp_path / "class_weights.json"
    class_weights_path.write_text(json.dumps({"pixel_counts": counts}), encoding="utf-8")
    idd20k_counts = [0, 0, 25] + [0] * 16
    idd20k_path = tmp_path / "summary.json"
    idd20k_path.write_text(
        json.dumps({"class_pixel_counts": idd20k_counts, "dataset_id": "idd20k"}),
        encoding="utf-8",
    )
    frequency = load_real_class_frequency(class_weights_path, idd20k_path)
    assert frequency["pixel_counts_measured"]["building"] == 75
    assert frequency["sources"]["idd20k"] == str(idd20k_path)


def test_load_real_class_frequency_reports_missing_source_honestly(tmp_path: Path) -> None:
    frequency = load_real_class_frequency(tmp_path / "nope.json")
    assert frequency["sources"]["cityscapes"] == "not_available: file not found"
    assert frequency["total_pixels_measured"] == 0


def test_flag_near_absent_classes_uses_a_fixed_global_threshold() -> None:
    frequency = {
        "total_pixels_measured": 1000,
        "pixel_counts_measured": {"road": 990, "train": 5, "bicycle": 5},
    }
    flagged = flag_near_absent_classes(frequency, pixel_ratio_threshold=0.01)
    assert set(flagged) == {"train", "bicycle"}


def test_project_reduced_class_miou_averages_only_real_measured_values() -> None:
    results = parse_training_log(REAL_SEGFORMER_B0_PILOT_LOG, source="real_excerpt")
    result = results[0]
    # Hand-computed: excluding every 0.0 class leaves road/building/vegetation/sky/car
    hand_computed = (76.84 + 33.82 + 40.25 + 70.01 + 24.47) / 5
    excluded = [name for name, iou in result.per_class_iou.items() if iou == 0.0]
    projected = project_reduced_class_miou(result, excluded)
    assert projected == pytest.approx(hand_computed)
    # The original 19-class mIoU stays untouched alongside the projection.
    assert result.miou == 12.91


def test_project_reduced_class_miou_returns_none_when_nothing_is_left() -> None:
    results = parse_training_log(REAL_SEGFORMER_B0_PILOT_LOG, source="real_excerpt")
    result = results[0]
    assert project_reduced_class_miou(result, list(result.per_class_iou)) is None


def test_build_analysis_report_and_write_analysis_report(tmp_path: Path) -> None:
    counts = [1000] * 19
    class_weights_path = tmp_path / "class_weights.json"
    class_weights_path.write_text(json.dumps({"pixel_counts": counts}), encoding="utf-8")

    report = build_analysis_report(
        {"real_excerpt": REAL_SEGFORMER_B0_PILOT_LOG},
        class_weights_path=class_weights_path,
    )
    assert report["record_type"] == "training_results_class_analysis"
    assert "scientific_status" not in report
    assert report["run_count"] == 1
    assert report["runs"][0]["model"] == "segformer_b0"
    assert report["sampling"].startswith("exhaustive")

    output_root = tmp_path / "out"
    write_analysis_report(report, output_root)
    assert (output_root / "training_analysis.json").is_file()
    markdown = (output_root / "training_analysis.md").read_text(encoding="utf-8")
    assert "segformer_b0" in markdown
    assert render_markdown_report(report) == markdown
