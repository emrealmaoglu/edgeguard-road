"""Produce every presentation/thesis artefact the screening checkpoints can already support.

The campaign chain (hpo -> final -> accept -> package) is not a prerequisite for any of
these outputs, and this driver deliberately does not touch it. It reads the screening run
directories that already exist, then drives the ordinary single-purpose scripts --
`predict.py`, `evaluate.py`, `audit_dataset.py`, `analyze_training_results.py` -- none of
which sit behind the accepted-release gate.

Every step is fail-soft and individually recorded. A step that could not run is written
down as `skipped` or `failed` with its reason; it is never reported as a measurement. That
is the whole point: one missing input must not cost the other twelve artefacts, and a
missing artefact must never look like a produced one.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Screening trains one loss variant per model; the directory layout is
# <work-root>/runs/screening/<model>/<loss>/.
SCREENING_STAGE = "screening"
# The calibration role is the 5% slice held out from fitting, so drawing display frames
# from it keeps the qualitative figures off the data the models were fitted on.
FIGURE_ROLE = "train_calibration"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class StepLog:
    """Collect one honest row per attempted artefact."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, name: str, status: str, **extra: Any) -> dict[str, Any]:
        if status not in {"produced", "reused", "skipped", "failed"}:
            raise ValueError(f"unknown step status: {status}")
        row = {"step": name, "status": status, "recorded_at": _timestamp(), **extra}
        self.rows.append(row)
        marker = {
            "produced": "OK      ",
            "reused": "VAR     ",
            "skipped": "ATLANDI ",
            "failed": "HATA    ",
        }[status]
        detail = extra.get("reason") or extra.get("output_dir") or ""
        print(f"  {marker}{name}  {detail}", flush=True)
        return row

    def counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for row in self.rows:
            tally[row["status"]] = tally.get(row["status"], 0) + 1
        return tally


def resolve_runtime_interpreter(evidence_root: Path) -> tuple[Path, str]:
    """Return the interpreter that owns the pinned torch/mmseg stack.

    `run_colab_master.py` provisions a locked venv and writes its path into the runtime
    receipt. The host Python that runs the notebook cannot import torch, so anything
    touching a checkpoint has to go through that interpreter.
    """
    receipt = evidence_root / "runtime_receipt.json"
    if receipt.is_file():
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            interpreter = Path(str(payload["interpreter"]))
            if interpreter.is_file():
                return interpreter, "runtime_receipt"
        except (OSError, KeyError, json.JSONDecodeError):
            pass
    return Path(sys.executable), "current_interpreter"


def resolve_mmseg_root(evidence_root: Path, override: Path | None) -> Path | None:
    if override is not None:
        return override
    receipt = evidence_root / "runtime_receipt.json"
    if receipt.is_file():
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            candidate = payload.get("mmseg_root")
            if candidate:
                return Path(str(candidate))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _bundled_library_path(runtime_python: Path, environment: dict[str, str]) -> str:
    """Put the pinned wheel's own CUDA libraries ahead of Colab's toolkit.

    Colab prepends its current CUDA toolkit to `LD_LIBRARY_PATH`, and those libraries are
    not necessarily ABI-compatible with the deliberately pinned PyTorch wheel that owns
    this runtime. `run_colab_master.py::_runtime_environment` does exactly this reordering
    before it launches training, so anything else invoking that interpreter has to do it
    too -- otherwise a checkpoint that trained fine can fail to even load here. The hosted
    driver paths are kept, since `libcuda` comes from the host rather than the wheel.
    """
    runtime_root = runtime_python.parent.parent
    bundled: list[str] = []
    for site_packages in sorted(runtime_root.glob("lib/python*/site-packages")):
        candidates = [site_packages / "torch/lib"]
        candidates.extend(sorted(site_packages.glob("nvidia/*/lib")))
        bundled.extend(str(path) for path in candidates if path.is_dir())
    hosted = [value for value in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep) if value]
    ordered = list(dict.fromkeys((*bundled, *hosted)))
    return os.pathsep.join(ordered)


def _checkpoint_sort_key(path: Path) -> tuple[int, int]:
    """Rank checkpoints: the best-mIoU snapshot first, then the latest iteration."""
    iteration = 0
    for chunk in path.stem.split("_"):
        if chunk.isdigit():
            iteration = int(chunk)
    return (1 if path.name.startswith("best_") else 0, iteration)


def discover_screening_models(work_root: Path, requested: Sequence[str]) -> list[dict[str, Any]]:
    """Find every screening run that carries both a checkpoint and its resolved config."""
    stage_root = work_root / "runs" / SCREENING_STAGE
    if not stage_root.is_dir():
        return []
    discovered: list[dict[str, Any]] = []
    for model_dir in sorted(stage_root.iterdir()):
        if not model_dir.is_dir():
            continue
        if requested and model_dir.name not in requested:
            continue
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            resolved = run_dir / "resolved.py"
            checkpoints = sorted(run_dir.glob("*.pth"), key=_checkpoint_sort_key, reverse=True)
            if not resolved.is_file() or not checkpoints:
                continue
            onnx = sorted(run_dir.glob("*.onnx"))
            discovered.append(
                {
                    "model": model_dir.name,
                    "variant": run_dir.name,
                    "run_dir": str(run_dir),
                    "resolved_config": str(resolved),
                    "checkpoint": str(checkpoints[0]),
                    "onnx": str(onnx[0]) if onnx else None,
                }
            )
            break
    return discovered


def discover_training_manifests(work_root: Path) -> list[Path]:
    manifest_root = work_root / "manifests" / "training"
    if not manifest_root.is_dir():
        return []
    return sorted(manifest_root.glob("*.frozen.json"))


def sample_display_frames(manifest_path: Path, count: int) -> tuple[str, list[Path]]:
    """Pick deterministic held-out frames from one frozen manifest.

    Sorted by sample id so a rerun on the same manifest shows the same images -- a figure
    that silently changes between runs is not evidence of anything.
    """
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_id = str(payload["dataset_id"])
    dataset_root = Path(str(payload["dataset_root"]))
    roles = payload.get("roles", {})
    records: list[dict[str, Any]] = roles.get(FIGURE_ROLE) or next(iter(roles.values()), [])
    frames: list[Path] = []
    for record in sorted(records, key=lambda row: str(row["sample_id"])):
        candidate = dataset_root / str(record["image"])
        if candidate.is_file():
            frames.append(candidate)
        if len(frames) >= count:
            break
    return dataset_id, frames


def run_command(
    command: Sequence[str], *, project_root: Path, environment: dict[str, str], log: Path
) -> tuple[int, str]:
    """Run one child command, streaming to the console and appending to a log."""
    rendered = " ".join(str(value) for value in command)
    print(f"    $ {rendered}", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    tail: list[str] = []
    with log.open("a", encoding="utf-8") as sink:
        sink.write(f"\nCOMMAND: {rendered}\n")
        process = subprocess.Popen(
            list(command),
            cwd=str(project_root),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sink.write(line)
            tail.append(line)
            if len(tail) > 60:
                tail.pop(0)
        return_code = process.wait()
        sink.write(f"RETURN_CODE: {return_code}\n")
    return return_code, "".join(tail)


class PresentationBuilder:
    """Drive the ungated scripts and keep one truthful record of what ran."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.project_root = args.project_root.resolve()
        self.work_root = args.work_root.resolve()
        self.evidence_root = args.evidence_root.resolve()
        self.output_root = args.output_root.resolve()
        self.device = args.device
        self.frames_per_domain = args.frames_per_domain
        self.train_steps = args.train_steps
        self.train_model = args.train_model
        self.log_path = self.output_root / "presentation-build.log"
        self.steps = StepLog()
        self.interpreter, self.interpreter_source = resolve_runtime_interpreter(self.evidence_root)
        self.mmseg_root = resolve_mmseg_root(self.evidence_root, args.mmseg_root)
        self.environment = self._environment()

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.project_root / "src")
        environment["PYTHONNOUSERSITE"] = "1"
        environment["MPLBACKEND"] = "Agg"
        if self.mmseg_root is not None:
            environment["MMSEG_ROOT"] = str(self.mmseg_root)
        library_path = _bundled_library_path(self.interpreter, environment)
        if library_path:
            environment["LD_LIBRARY_PATH"] = library_path
        return environment

    def _run(self, name: str, command: Sequence[str], *, output_dir: Path) -> dict[str, Any]:
        """Run one artefact-producing command, refusing to overwrite existing evidence.

        `predict.py` and `evaluate_model` both create their output directory with
        `exist_ok=False`, so an existing directory means the artefact is already there.
        Reuse it rather than deleting a real measurement to make room for a rerun.
        """
        if output_dir.exists():
            return self.steps.record(
                name, "reused", output_dir=str(output_dir), reason="çıktı klasörü zaten var"
            )
        try:
            return_code, tail = run_command(
                command,
                project_root=self.project_root,
                environment=self.environment,
                log=self.log_path,
            )
        except OSError as error:
            return self.steps.record(name, "failed", reason=repr(error))
        if return_code != 0:
            return self.steps.record(
                name,
                "failed",
                reason=f"komut {return_code} koduyla durdu",
                output_dir=str(output_dir),
                tail=tail[-4000:],
            )
        return self.steps.record(name, "produced", output_dir=str(output_dir))

    def build_figures(self, models: list[dict[str, Any]], manifests: list[Path]) -> None:
        """Segmentation, confidence, entropy, drivable corridor and attention per frame."""
        print("\n[1/5] Görsel çıktılar (predict.py)", flush=True)
        if not manifests:
            self.steps.record("figures", "skipped", reason="dondurulmuş veri manifesti bulunamadı")
            return
        for manifest in manifests:
            try:
                dataset_id, frames = sample_display_frames(manifest, self.frames_per_domain)
            except (OSError, KeyError, json.JSONDecodeError) as error:
                self.steps.record(
                    f"figures:{manifest.stem}", "failed", reason=f"manifest okunamadı: {error!r}"
                )
                continue
            if not frames:
                self.steps.record(
                    f"figures:{dataset_id}",
                    "skipped",
                    reason="manifestteki görüntü yolları diskte bulunamadı",
                )
                continue
            for entry in models:
                for index, frame in enumerate(frames):
                    output_dir = (
                        self.output_root / "figures" / entry["model"] / dataset_id / f"{index:02d}"
                    )
                    self._run(
                        f"figure:{entry['model']}:{dataset_id}:{index:02d}",
                        [
                            str(self.interpreter),
                            str(self.project_root / "scripts/predict.py"),
                            "--image",
                            str(frame),
                            "--model",
                            entry["checkpoint"],
                            "--resolved-config",
                            entry["resolved_config"],
                            "--device",
                            self.device,
                            "--emit-regions",
                            "--emit-risk",
                            "--output-dir",
                            str(output_dir),
                        ],
                        output_dir=output_dir,
                    )

    def build_calibration(self, models: list[dict[str, Any]], manifests: list[Path]) -> None:
        """Temperature scaling plus the reliability/ECE record, per model per domain.

        Temperature fitting is only permitted on a source-domain `train_calibration` split
        (`mmseg_runtime.evaluate_model`), which is exactly the role every frozen training
        manifest already carries -- so this is the sanctioned path, not a way around it.
        Only the first domain fits a temperature; the rest are measured under it so the
        per-domain entropy comparison stays on one scale.
        """
        print("\n[2/5] Kalibrasyon, ECE ve kare-başına belirsizlik (evaluate.py)", flush=True)
        if not manifests:
            self.steps.record(
                "calibration", "skipped", reason="dondurulmuş veri manifesti bulunamadı"
            )
            return
        for entry in models:
            fitted_temperature: Path | None = None
            for manifest in manifests:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                dataset_id = str(payload["dataset_id"])
                output_dir = self.output_root / "evaluation" / entry["model"] / dataset_id
                command = [
                    str(self.interpreter),
                    str(self.project_root / "scripts/evaluate.py"),
                    "run",
                    "--resolved-config",
                    entry["resolved_config"],
                    "--checkpoint",
                    entry["checkpoint"],
                    "--dataset",
                    dataset_id,
                    "--dataset-manifest",
                    str(manifest),
                    "--role",
                    FIGURE_ROLE,
                    "--output-dir",
                    str(output_dir),
                ]
                if fitted_temperature is None:
                    command.append("--fit-temperature")
                else:
                    command.extend(("--temperature-file", str(fitted_temperature)))
                row = self._run(
                    f"calibration:{entry['model']}:{dataset_id}", command, output_dir=output_dir
                )
                if row["status"] in {"produced", "reused"} and fitted_temperature is None:
                    candidate = output_dir / "temperature.json"
                    if candidate.is_file():
                        fitted_temperature = candidate

    def build_dataset_figures(self, manifests: list[Path]) -> None:
        print("\n[3/5] Veri seti figürleri (audit_dataset.py)", flush=True)
        if not manifests:
            self.steps.record(
                "dataset_figures", "skipped", reason="dondurulmuş veri manifesti bulunamadı"
            )
            return
        output_dir = self.output_root / "dataset-statistics"
        command = [
            str(self.interpreter),
            str(self.project_root / "scripts/audit_dataset.py"),
            "--output-root",
            str(output_dir),
        ]
        for manifest in manifests:
            command.extend(("--data-manifest", str(manifest)))
        self._run("dataset_figures", command, output_dir=output_dir)

    def build_training_analysis(self) -> None:
        """Training curves and the per-class table, straight from the real Colab logs."""
        print("\n[4/5] Eğitim eğrileri ve sınıf tablosu (analyze_training_results.py)", flush=True)
        class_weights = self.work_root / "multidomain-statistics" / "class_weights.json"
        if not class_weights.is_file():
            self.steps.record(
                "training_analysis", "skipped", reason=f"class_weights.json yok: {class_weights}"
            )
            return
        logs = sorted((self.work_root / "runs" / SCREENING_STAGE).rglob("*.log"))
        if not logs:
            self.steps.record(
                "training_analysis", "skipped", reason="screening eğitim logu bulunamadı"
            )
            return
        output_dir = self.output_root / "training-analysis"
        command = [
            str(self.interpreter),
            str(self.project_root / "scripts/analyze_training_results.py"),
            "--class-weights",
            str(class_weights),
            "--output-root",
            str(output_dir),
        ]
        for log in logs:
            command.extend(("--log-file", str(log)))
        self._run("training_analysis", command, output_dir=output_dir)

    def train_longer(self, steps: int, model: str, manifests: list[Path]) -> None:
        """Train one model for longer, straight through `train.py`, no orchestrator.

        This deliberately omits `--recovery-root`: the run must not write into the Drive
        recovery store the finished screening state lives in. A longer run is a bonus, and
        it must not be able to damage the evidence the presentation already depends on.

        `--max-steps` also shortens the LR schedule horizon, so this is a self-consistent
        run of its own length -- not a continuation of the 2.500-step screening run.
        """
        print(f"\n[+] Daha uzun eğitim: {model}, {steps} adım (orkestratör yok)", flush=True)
        if not manifests:
            self.steps.record(
                "train_longer", "skipped", reason="dondurulmuş veri manifesti bulunamadı"
            )
            return
        pretrained = self.project_root / "configs/pretrained" / f"{model}.json"
        if not pretrained.is_file():
            self.steps.record(
                "train_longer", "skipped", reason=f"pretrained manifesti yok: {pretrained}"
            )
            return
        run_name = f"long{steps}"
        output_dir = self.work_root / "runs" / "final" / model / run_name
        command = [
            str(self.interpreter),
            str(self.project_root / "scripts/train.py"),
            "--stage",
            "final",
            "--model",
            model,
            "--output-root",
            str(self.work_root / "runs"),
            "--initialization",
            "pretrained",
            "--pretrained-manifest",
            str(pretrained),
            "--loss",
            "ce",
            "--run-name",
            run_name,
            "--max-steps",
            str(steps),
        ]
        if self.mmseg_root is not None:
            command.extend(("--mmseg-root", str(self.mmseg_root)))
        for manifest in manifests:
            command.extend(("--data-manifest", str(manifest)))
        self._run("train_longer", command, output_dir=output_dir)

    def collect_screening_reports(self) -> None:
        """Copy the already-measured screening tables next to the new artefacts."""
        print("\n[5/5] Mevcut screening raporları", flush=True)
        source = self.work_root / "reports"
        if not source.is_dir():
            self.steps.record("screening_reports", "skipped", reason=f"rapor klasörü yok: {source}")
            return
        destination = self.output_root / "screening-reports"
        if destination.exists():
            self.steps.record("screening_reports", "reused", output_dir=str(destination))
            return
        try:
            shutil.copytree(source, destination)
        except OSError as error:
            self.steps.record("screening_reports", "failed", reason=repr(error))
            return
        self.steps.record("screening_reports", "produced", output_dir=str(destination))

    def run(self) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        models = discover_screening_models(self.work_root, [])
        manifests = discover_training_manifests(self.work_root)
        print("EdgeGuard sunum çıktıları", flush=True)
        print(f"  yorumlayıcı : {self.interpreter} ({self.interpreter_source})", flush=True)
        print(f"  mmseg kökü  : {self.mmseg_root}", flush=True)
        print(f"  çalışma kökü: {self.work_root}", flush=True)
        print(f"  çıktı kökü  : {self.output_root}", flush=True)
        print(f"  modeller    : {[entry['model'] for entry in models] or 'BULUNAMADI'}", flush=True)
        print(f"  manifestler : {[path.name for path in manifests] or 'BULUNAMADI'}", flush=True)

        if self.train_steps:
            # Training only. It runs for hours and a dropped Colab session cuts it off, so
            # it never shares a run with the artefacts it could otherwise take down.
            self.train_longer(self.train_steps, self.train_model, manifests)
        else:
            if models:
                self.build_figures(models, manifests)
                self.build_calibration(models, manifests)
            else:
                missing = f"{self.work_root / 'runs' / SCREENING_STAGE} altında checkpoint yok"
                self.steps.record("figures", "skipped", reason=missing)
                self.steps.record("calibration", "skipped", reason=missing)
            self.build_dataset_figures(manifests)
            self.build_training_analysis()
            self.collect_screening_reports()

        counts = self.steps.counts()
        summary = {
            "schema_version": "1.0",
            "record_type": "edgeguard_presentation_outputs",
            "generated_at": _timestamp(),
            "work_root": str(self.work_root),
            "output_root": str(self.output_root),
            "interpreter": str(self.interpreter),
            "interpreter_source": self.interpreter_source,
            "mmseg_root": str(self.mmseg_root) if self.mmseg_root else None,
            "device": self.device,
            "frames_per_domain": self.frames_per_domain,
            "mode": "train" if self.train_steps else "artefacts",
            "train_steps": self.train_steps,
            "screening_models": models,
            "training_manifests": [str(path) for path in manifests],
            "steps": self.steps.rows,
            "step_counts": counts,
            # These artefacts are rendered from real screening checkpoints, but nothing
            # here is an accepted release and no sealed test data is touched.
            "scientific_status": "measured" if counts.get("produced") else "not_run",
            "accepted_release": False,
            "sealed_test_data_opened": False,
        }
        # Training writes its own record: it must not overwrite the artefact record from
        # an earlier run, which is the only account of what the presentation actually has.
        record = "training_run.json" if self.train_steps else "presentation_outputs.json"
        # Deliberately stdlib-only: this driver has to stay runnable by whichever
        # interpreter a Colab cell happens to have, before any project package is
        # importable. The record is a build log, not hash-bound scientific evidence.
        (self.output_root / record).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("\n=== Özet ===", flush=True)
        for status in ("produced", "reused", "skipped", "failed"):
            if counts.get(status):
                print(f"  {status}: {counts[status]}", flush=True)
        print(f"  kayıt: {self.output_root / 'presentation_outputs.json'}", flush=True)
        return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mmseg-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames-per-domain", type=int, default=3)
    parser.add_argument(
        "--train-steps",
        type=int,
        help=(
            "train one model for this many steps through scripts/train.py instead of "
            "rendering artefacts. Writes no Drive recovery state, so a longer run cannot "
            "damage the finished screening evidence the presentation depends on."
        ),
    )
    parser.add_argument("--train-model", default="pidnet_s")
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = PresentationBuilder(args).run()
    # A partial set of artefacts is still worth having, so only a total failure to produce
    # anything is worth a non-zero exit; the record already names every missing piece.
    counts = summary["step_counts"]
    return 0 if counts.get("produced") or counts.get("reused") else 1


if __name__ == "__main__":
    raise SystemExit(main())
