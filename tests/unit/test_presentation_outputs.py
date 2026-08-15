"""Real end-to-end exercise of the presentation-output driver.

The driver exists because the campaign chain kept failing on real Colab hardware, so the
one thing it must never do is repeat that pattern: a single missing input or one crashing
child command must not cost every other artefact. These tests drive the actual
`main()` with real subprocesses against a synthetic work root, so the orchestration,
discovery and fail-soft behaviour are proven here rather than in a Colab session.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.build_presentation_outputs import (  # noqa: E402
    _bundled_library_path,
    discover_screening_models,
    discover_training_manifests,
    main,
    resolve_runtime_interpreter,
    sample_display_frames,
)

# Child scripts the driver shells out to. Each stub honours the same output-directory
# contract the real script does, so the driver's reuse/skip logic is exercised for real.
_STUB_SUCCESS = """
import sys
from pathlib import Path

argv = sys.argv[1:]
for flag in ("--output-dir", "--output-root"):
    if flag in argv:
        target = Path(argv[argv.index(flag) + 1])
        target.mkdir(parents=True, exist_ok=False)
        (target / "summary.json").write_text("{}", encoding="utf-8")
print("stub ok")
"""

_STUB_FAILURE = """
import sys

print("stub deliberately failed", file=sys.stderr)
raise SystemExit(3)
"""

# `train.py` differs from the artefact scripts: its `--output-root` is the shared `runs/`
# directory, which already exists. It creates `<root>/<stage>/<model>/<run-name>` itself.
_STUB_TRAIN = """
import json
import sys
from pathlib import Path

argv = sys.argv[1:]


def value(flag):
    return argv[argv.index(flag) + 1]


root = Path(value("--output-root"))
run_dir = root / value("--stage") / value("--model") / value("--run-name")
run_dir.mkdir(parents=True, exist_ok=False)
(run_dir / "summary.json").write_text("{}", encoding="utf-8")
(root / "argv.json").write_text(json.dumps(argv), encoding="utf-8")
print("stub trained")
"""


def _write_stub_scripts(project_root: Path, *, failing: tuple[str, ...] = ()) -> None:
    scripts = project_root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for name in (
        "predict.py",
        "evaluate.py",
        "audit_dataset.py",
        "analyze_training_results.py",
    ):
        body = _STUB_FAILURE if name in failing else _STUB_SUCCESS
        (scripts / name).write_text(body, encoding="utf-8")


def _manifest(dataset_id: str, dataset_root: Path, sample_ids: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "record_type": "edgeguard_dataset_manifest",
        "dataset_id": dataset_id,
        "split_state": "frozen",
        "dataset_root": str(dataset_root),
        "prepared_root": str(dataset_root),
        "roles": {
            "train_fit": [
                {"sample_id": f"{dataset_id}-fit", "group_id": "g0", "image": "images/fit.png"}
            ],
            "train_calibration": [
                {
                    "sample_id": sample_id,
                    "group_id": f"g-{sample_id}",
                    "image": f"images/{sample_id}.png",
                }
                for sample_id in sample_ids
            ],
        },
    }


def _build_work_root(
    root: Path, *, models: tuple[str, ...] = ("pidnet_s", "ddrnet_23_slim")
) -> Path:
    work_root = root / "work"
    for model in models:
        run_dir = work_root / "runs/screening" / model / "ce"
        run_dir.mkdir(parents=True)
        (run_dir / "resolved.py").write_text("# resolved config\n", encoding="utf-8")
        (run_dir / "iter_2500.pth").write_bytes(b"weights")
        (run_dir / "best_mIoU_iter_2500.pth").write_bytes(b"best weights")
        (run_dir / "training.log").write_text("Iter(train) [50/2500] loss: 1.0\n", encoding="utf-8")

    dataset_root = root / "data/cityscapes"
    (dataset_root / "images").mkdir(parents=True)
    sample_ids = ["cs-a", "cs-b", "cs-c", "cs-d"]
    for sample_id in sample_ids:
        (dataset_root / "images" / f"{sample_id}.png").write_bytes(b"png")

    manifest_root = work_root / "manifests/training"
    manifest_root.mkdir(parents=True)
    (manifest_root / "cityscapes.frozen.json").write_text(
        json.dumps(_manifest("cityscapes", dataset_root, sample_ids)), encoding="utf-8"
    )

    statistics = work_root / "multidomain-statistics"
    statistics.mkdir(parents=True)
    (statistics / "class_weights.json").write_text("{}", encoding="utf-8")

    reports = work_root / "reports/screening"
    reports.mkdir(parents=True)
    (reports / "candidate_table.json").write_text('{"candidates": []}', encoding="utf-8")
    return work_root


def _run_driver(
    tmp_path: Path,
    project_root: Path,
    work_root: Path,
    output_root: Path,
    *,
    extra: Sequence[str] = (),
    record: str = "presentation_outputs.json",
) -> dict:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir(exist_ok=True)
    argv = [
        "build_presentation_outputs.py",
        "--project-root",
        str(project_root),
        "--work-root",
        str(work_root),
        "--evidence-root",
        str(evidence_root),
        "--output-root",
        str(output_root),
        "--device",
        "cpu",
        "--frames-per-domain",
        "2",
        *extra,
    ]
    original = sys.argv
    sys.argv = argv
    try:
        exit_code = main()
    finally:
        sys.argv = original
    payload = json.loads((output_root / record).read_text(encoding="utf-8"))
    payload["_exit_code"] = exit_code
    return payload


def test_driver_produces_every_artefact_from_a_complete_work_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _write_stub_scripts(project_root)
    work_root = _build_work_root(tmp_path)
    output_root = tmp_path / "out"

    payload = _run_driver(tmp_path, project_root, work_root, output_root)

    assert payload["_exit_code"] == 0
    assert payload["step_counts"].get("failed") is None
    assert payload["scientific_status"] == "measured"
    # Nothing here is an accepted release and no sealed data is involved; the record has
    # to say so explicitly, because these artefacts end up in a thesis.
    assert payload["accepted_release"] is False
    assert payload["sealed_test_data_opened"] is False

    # 2 models x 1 domain x 2 frames of figures, 2 calibration runs, plus the three
    # single-shot steps.
    steps = {row["step"]: row["status"] for row in payload["steps"]}
    assert steps["figure:pidnet_s:cityscapes:00"] == "produced"
    assert steps["figure:ddrnet_23_slim:cityscapes:01"] == "produced"
    assert steps["calibration:pidnet_s:cityscapes"] == "produced"
    assert steps["dataset_figures"] == "produced"
    assert steps["training_analysis"] == "produced"
    assert steps["screening_reports"] == "produced"


def test_one_crashing_child_never_costs_the_other_artefacts(tmp_path: Path) -> None:
    """The campaign's recurring failure mode was a single stop discarding everything else.

    `predict.py` failing must still leave the calibration records, dataset figures,
    training analysis and screening reports on disk.
    """
    project_root = tmp_path / "project"
    _write_stub_scripts(project_root, failing=("predict.py",))
    work_root = _build_work_root(tmp_path)
    output_root = tmp_path / "out"

    payload = _run_driver(tmp_path, project_root, work_root, output_root)

    steps = {row["step"]: row["status"] for row in payload["steps"]}
    assert steps["figure:pidnet_s:cityscapes:00"] == "failed"
    assert steps["calibration:pidnet_s:cityscapes"] == "produced"
    assert steps["dataset_figures"] == "produced"
    assert steps["training_analysis"] == "produced"
    assert steps["screening_reports"] == "produced"
    # Partial output is still worth handing back, so the run does not report total failure.
    assert payload["_exit_code"] == 0
    failure = next(row for row in payload["steps"] if row["step"].startswith("figure:"))
    assert "3" in failure["reason"]


def test_missing_checkpoints_are_recorded_as_skipped_not_measured(tmp_path: Path) -> None:
    """A work root with no screening run must never yield `scientific_status: measured`."""
    project_root = tmp_path / "project"
    _write_stub_scripts(project_root)
    work_root = _build_work_root(tmp_path, models=())
    output_root = tmp_path / "out"

    payload = _run_driver(tmp_path, project_root, work_root, output_root)

    steps = {row["step"]: row["status"] for row in payload["steps"]}
    assert steps["figures"] == "skipped"
    assert steps["calibration"] == "skipped"
    assert payload["screening_models"] == []
    # The dataset figures and screening reports need no checkpoint, so they still run.
    assert steps["dataset_figures"] == "produced"


def test_reruns_reuse_existing_evidence_instead_of_overwriting_it(tmp_path: Path) -> None:
    """`predict.py` and `evaluate_model` create their output dir with `exist_ok=False`.

    A rerun must therefore treat an existing directory as the artefact already being
    present, never delete a real measurement to make room for a fresh one.
    """
    project_root = tmp_path / "project"
    _write_stub_scripts(project_root)
    work_root = _build_work_root(tmp_path)
    output_root = tmp_path / "out"

    first = _run_driver(tmp_path, project_root, work_root, output_root)
    assert first["step_counts"]["produced"] > 0

    marker = output_root / "figures/pidnet_s/cityscapes/00/summary.json"
    marker.write_text('{"kept": true}', encoding="utf-8")

    second = _run_driver(tmp_path, project_root, work_root, output_root)
    steps = {row["step"]: row["status"] for row in second["steps"]}
    assert steps["figure:pidnet_s:cityscapes:00"] == "reused"
    assert steps["calibration:pidnet_s:cityscapes"] == "reused"
    assert json.loads(marker.read_text(encoding="utf-8")) == {"kept": True}


def test_training_mode_never_overwrites_the_artefact_record(tmp_path: Path) -> None:
    """The artefact record is the only account of what the presentation actually has.

    The longer training run happens in the same output root and can be started hours
    later, so it writes its own record rather than replacing that account.
    """
    project_root = tmp_path / "project"
    _write_stub_scripts(project_root)
    (project_root / "scripts/train.py").write_text(_STUB_TRAIN, encoding="utf-8")
    (project_root / "configs/pretrained").mkdir(parents=True)
    (project_root / "configs/pretrained/pidnet_s.json").write_text("{}", encoding="utf-8")
    work_root = _build_work_root(tmp_path)
    output_root = tmp_path / "out"

    artefacts = _run_driver(tmp_path, project_root, work_root, output_root)
    assert artefacts["mode"] == "artefacts"

    training = _run_driver(
        tmp_path,
        project_root,
        work_root,
        output_root,
        extra=["--train-steps", "10000"],
        record="training_run.json",
    )
    assert training["mode"] == "train"
    assert training["train_steps"] == 10000
    steps = {row["step"]: row["status"] for row in training["steps"]}
    assert steps == {"train_longer": "produced"}

    # The artefact record still says exactly what it said before the training run.
    unchanged = json.loads((output_root / "presentation_outputs.json").read_text(encoding="utf-8"))
    assert unchanged["step_counts"] == artefacts["step_counts"]
    assert unchanged["mode"] == "artefacts"


def test_the_longer_run_never_writes_into_the_drive_recovery_store(tmp_path: Path) -> None:
    """A bonus run must not be able to damage the finished screening evidence.

    `train.py` only touches the Drive recovery store when handed `--recovery-root`, so
    the command must never carry that flag.
    """
    project_root = tmp_path / "project"
    _write_stub_scripts(project_root)
    # The stub records the argv it was handed so the assembled command can be inspected.
    (project_root / "scripts/train.py").write_text(_STUB_TRAIN, encoding="utf-8")
    (project_root / "configs/pretrained").mkdir(parents=True)
    (project_root / "configs/pretrained/pidnet_s.json").write_text("{}", encoding="utf-8")
    work_root = _build_work_root(tmp_path)

    _run_driver(
        tmp_path,
        project_root,
        work_root,
        tmp_path / "out",
        extra=["--train-steps", "4000"],
        record="training_run.json",
    )

    argv = json.loads((work_root / "runs/argv.json").read_text(encoding="utf-8"))
    assert "--recovery-root" not in argv
    assert argv[argv.index("--max-steps") + 1] == "4000"
    assert argv[argv.index("--stage") + 1] == "final"
    assert argv[argv.index("--initialization") + 1] == "pretrained"
    assert argv.count("--data-manifest") == 1


def test_a_missing_pretrained_manifest_is_skipped_rather_than_trained_from_scratch(
    tmp_path: Path,
) -> None:
    """Silently falling back to random initialization would make the run incomparable."""
    project_root = tmp_path / "project"
    _write_stub_scripts(project_root)
    (project_root / "scripts/train.py").write_text(_STUB_TRAIN, encoding="utf-8")
    work_root = _build_work_root(tmp_path)

    training = _run_driver(
        tmp_path,
        project_root,
        work_root,
        tmp_path / "out",
        extra=["--train-steps", "10000"],
        record="training_run.json",
    )

    row = next(row for row in training["steps"] if row["step"] == "train_longer")
    assert row["status"] == "skipped"
    assert "pretrained" in row["reason"]
    assert training["scientific_status"] == "not_run"


def test_bundled_cuda_libraries_precede_the_host_toolkit(tmp_path: Path) -> None:
    """Colab prepends its own CUDA toolkit, which the pinned torch wheel may not match.

    `run_colab_master.py` reorders `LD_LIBRARY_PATH` before launching training; anything
    else invoking that interpreter has to do the same or a checkpoint that trained fine
    can fail to load.
    """
    runtime_python = tmp_path / "runtime/bin/python"
    torch_lib = tmp_path / "runtime/lib/python3.11/site-packages/torch/lib"
    nvidia_lib = tmp_path / "runtime/lib/python3.11/site-packages/nvidia/cublas/lib"
    torch_lib.mkdir(parents=True)
    nvidia_lib.mkdir(parents=True)

    resolved = _bundled_library_path(runtime_python, {"LD_LIBRARY_PATH": "/usr/local/cuda/lib64"})

    entries = resolved.split(":")
    assert entries.index(str(torch_lib)) < entries.index("/usr/local/cuda/lib64")
    assert str(nvidia_lib) in entries
    # The hosted driver path supplies libcuda and must survive the reordering.
    assert "/usr/local/cuda/lib64" in entries


def test_best_checkpoint_is_preferred_over_the_last_iteration(tmp_path: Path) -> None:
    work_root = _build_work_root(tmp_path)
    run_dir = work_root / "runs/screening/pidnet_s/ce"
    (run_dir / "iter_4000.pth").write_bytes(b"later but not best")

    discovered = discover_screening_models(work_root, [])
    entry = next(row for row in discovered if row["model"] == "pidnet_s")
    assert Path(entry["checkpoint"]).name == "best_mIoU_iter_2500.pth"


def test_a_run_directory_without_a_resolved_config_is_not_a_candidate(tmp_path: Path) -> None:
    """A `.pth` alone cannot be evaluated -- `predict.py` refuses without the config."""
    work_root = _build_work_root(tmp_path)
    (work_root / "runs/screening/pidnet_s/ce/resolved.py").unlink()

    discovered = discover_screening_models(work_root, [])
    assert [row["model"] for row in discovered] == ["ddrnet_23_slim"]


def test_display_frames_come_from_the_held_out_calibration_role(tmp_path: Path) -> None:
    """Qualitative figures must not be drawn from the data the models were fitted on."""
    work_root = _build_work_root(tmp_path)
    manifest = work_root / "manifests/training/cityscapes.frozen.json"

    dataset_id, frames = sample_display_frames(manifest, 3)

    assert dataset_id == "cityscapes"
    assert [path.stem for path in frames] == ["cs-a", "cs-b", "cs-c"]
    assert all("fit" not in path.stem for path in frames)


def test_frames_absent_from_disk_are_not_reported_as_available(tmp_path: Path) -> None:
    work_root = _build_work_root(tmp_path)
    manifest = work_root / "manifests/training/cityscapes.frozen.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    (Path(payload["dataset_root"]) / "images/cs-a.png").unlink()

    _, frames = sample_display_frames(manifest, 2)
    assert [path.stem for path in frames] == ["cs-b", "cs-c"]


def test_runtime_interpreter_comes_from_the_receipt_when_it_names_a_real_file(
    tmp_path: Path,
) -> None:
    """The host notebook Python cannot import torch; the pinned venv owns that stack."""
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    interpreter = tmp_path / "runtime/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    (evidence_root / "runtime_receipt.json").write_text(
        json.dumps({"interpreter": str(interpreter)}), encoding="utf-8"
    )

    resolved, source = resolve_runtime_interpreter(evidence_root)
    assert resolved == interpreter
    assert source == "runtime_receipt"


@pytest.mark.parametrize(
    "payload",
    [
        '{"interpreter": "/nonexistent/python"}',
        "not json at all",
        "{}",
    ],
)
def test_an_unusable_receipt_falls_back_instead_of_crashing(tmp_path: Path, payload: str) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    (evidence_root / "runtime_receipt.json").write_text(payload, encoding="utf-8")

    resolved, source = resolve_runtime_interpreter(evidence_root)
    assert resolved == Path(sys.executable)
    assert source == "current_interpreter"


def test_only_frozen_manifests_are_discovered(tmp_path: Path) -> None:
    work_root = _build_work_root(tmp_path)
    candidate = work_root / "manifests/training/idd20k.candidate.json"
    candidate.write_text("{}", encoding="utf-8")

    discovered = discover_training_manifests(work_root)
    assert [path.name for path in discovered] == ["cityscapes.frozen.json"]
