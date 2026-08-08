# Project State

Updated 2026-08-08 on `stabilize/colab-v2`.

## Current delivery

The Colab v3 application commit is
`b22fd123e46478c1d7d368b8fbf50a28dbe28fdd`. The only generated notebook is
`notebooks/EdgeGuard_Master_Colab.ipynb`; it pins and verifies that exact commit. The
campaign ID is `semantic-cs-idd-v3`.

The old two delivery notebooks and twelve numbered notebooks are deleted from the current
tree but recoverable through Git history. No old Drive campaign, prepared dataset, audit,
or artifact is deleted.

## Data state

A read-only Drive review confirmed:

- the 8.26 GB verified Cityscapes prepared tar includes train and validation image/label
  roots and remains scientifically eligible;
- the official IDD20K index contains 33 verified shards and 16,063 train+val samples;
- official Cityscapes and IDD source archives remain available in `private_inputs/`;
- v2 audit candidates are reusable by exact identity.

The owner policy freezes only Cityscapes candidate
`74801b9c174778c7c13f5edbed6fdbe9d548139780d6906c6e940fee5281d8db`
(2,975 valid) and IDD20K candidate
`ba76d17b94dfaed93036ba2b3c46675c0b34fde1766f6879576ec862e0ac1762`
(14,018 valid, nine quarantined). Any identity/count drift stops before training.

## Pipeline state

One versioned orchestrator owns:

```text
preflight → restore → stage-data → canary → smoke → pilot → extension-smoke →
screening → hpo → final → selection → ablation → accept → validation-data →
evaluate → export → report → package
```

The hermetic stack is uv 0.8.8, CPython 3.11.13, NumPy 1.26.4, PyTorch
2.1.1/cu121, MMEngine 0.10.7, mmcv-lite 2.1.0, OpenCV headless 4.10.0.84, and
MMSegmentation commit `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8`. The host uv and
host Python stack are not training inputs. GUI OpenCV and dependency re-resolution are
rejected.

The hosted Colab interpreter now imports only Python standard-library modules. A
standard-library bootstrap installs the entire hash-locked environment before any
EdgeGuard, NumPy, Pydantic, Torch, MMCV, or MMSegmentation import. Restore, data staging,
canary, training, evaluation, export, and reporting then run only through the verified
Python 3.11 interpreter. The v3 work root is isolated at `/content/edgeguard-work-v3`.
Ephemeral evidence from another application commit is preserved under an incompatible
suffix and never resumed; a Drive state from another commit is preserved and skipped.

The first real master run at application commit `2495354d…` stopped before useful child
diagnostics were retained. The corrected notebook streams and records the complete child
output, current stage, bootstrap failure, and a bounded log tail in its Drive failure ZIP,
so a future external failure cannot collapse into an unactionable exit-code-only traceback.

The next real run at application commit `1da25ef…` proved that Colab system pip installs
prefix scripts under `local/bin` rather than the previously assumed `bin`. Both bootstrap
layers now discover, execute, and version-check the exact private uv binary across both
POSIX prefix layouts, and the verified discovered directory—not a reconstructed path—is
prepended to the hermetic runtime PATH.

The following L4 run at application commit `55e13db…` completed the full 92-package
hash sync, the two-wheel OpenMMLab install, and both editable installs. It then failed in
the old combined import/CUDA probe, which discarded its child stderr and redundantly ran
the complete dependency sync a second time. The exact lock imports successfully in a
clean Linux x86 GitHub runner. The corrected runtime now puts wheel-owned Torch/NVIDIA
libraries ahead of Colab toolkit libraries while retaining the host driver paths,
isolates every dependency import and CUDA initialization with preserved stderr, and
validates the standard-library bootstrap receipt instead of uninstalling/reinstalling the
environment. Setuptools is held at 80.9.0 so MMEngine's `pkg_resources` runtime path
remains available. A failed, receipt-less canary evidence root is preserved under a
timestamped quarantine name before retry, preventing old failure state from contaminating
the next Run-all attempt.

The next L4 run at application commit `005eb03…` proved that the exact locked environment
and all editable installs completed, then exposed one remaining hosted-notebook leak:
Colab exported `MPLBACKEND=module://matplotlib_inline.backend_inline`, while the locked
headless runtime intentionally does not install `matplotlib-inline`. Application commit
`3f3ef8f…` now removes hosted Python/virtualenv/pip/uv routing state, forces Matplotlib
`Agg`, isolates plotting and framework caches, records a non-secret environment-contract
hash, and renders a real headless PNG before the model canary. The same firewall reaches
bootstrap, canary, training, evaluation, export and report children. The combined
OOM → reduced device batch → intentional interruption path now adds `--resume` and verifies
the interruption checkpoint instead of reopening a non-empty run directory.

All five models pass the runtime canary contract. Core smoke intentionally interrupts at
step 25 of 50 and must resume from the same optimizer/checkpoint identity. Checkpoints are
published every 500 optimizer steps or ten minutes. A single OOM retry may change device
batch only when accumulation keeps effective batch four.

All five models receive 40,000-step final training. Only train-select evidence selects the
recommended model. Official validation opens after policy acceptance and cannot change the
choice. The selected model receives weighted-CE and 256×512 ablations; deployment remains
512×1024.

A cross-file review at application commit `8ed15194…` (Claude Code, real-stack review
against the pinned MMSeg checkout) found that no test in the repository ever ran the real
`Runner.train()`/`model.loss()` path against the five pinned architectures, and that this
let two real defects reach every stage without detection: PIDNet-S's shared training
pipeline was missing the upstream `GenerateEdge` step, so `PIDHead`'s boundary loss raised
`AttributeError: 'SegDataSample' object has no attribute 'gt_edge_map'` on the first
optimizer step of any stage; and the CE-versus-weighted-CE override only matched
`CrossEntropyLoss` nodes, so it silently changed nothing for DDRNet-23-Slim and almost
nothing for PIDNet-S, whose dominant loss terms are `OhemCrossEntropy`. Both are fixed and
were confirmed by actually building each real model from its resolved config and running
one real forward/backward step against the pinned MMSeg checkout
(`tests/unit/test_mmseg_real_training_step.py`, now wired into
`semantic-framework-cpu-probe.yml`). The `stage-data` phase previously only checked that
manifest JSON files existed; it now verifies every referenced image/mask file is present on
local disk (`verify_manifest_data_is_staged`), and Drive archive/shard copies now fail
closed on a stalled read via a bounded stall-timeout guard instead of risking an indefinite
hang.

The same review's follow-up commit `1387322…` fixed a lower-severity bug found in the same
pass — `resize_train_ids()` validated a resized mask against the source label-ID space
(0-33) instead of the train-ID space (0-18/255) — and reconciled several stale docs
(`SYSTEM_ARCHITECTURE.md`'s status boundary, `DATA_CATALOG.md`'s pre-ADR-0008/0009
acquisition status, the eval-config resolution mismatch). The IDD20K native-label-loss
finding from the same review was deliberately left unresolved as a recorded limitation
(`docs/TASKS.md`) rather than acted on, since fixing it means re-processing already-staged
Drive shards.

Commit `42be8d6…` raised `workers` from 2 to 6 in
`configs/rescue/semantic_first.yaml` so a Colab High-RAM instance's vCPUs are used more
fully during data loading; `effective_batch` and every frozen HPO/step budget in
`PROJECT_CHARTER.md` are unchanged. Training precision was confirmed already optimal
(`train_model`'s `precision="auto"` selects bf16 on CUDA without any code change). The
same commit adds `scripts/jetson/run_video_demo.py` (DEMO-02): reads a video frame by
frame, runs it through the static-shape semantic engine (ONNX Runtime locally, or a
target-device TensorRT engine via `scripts/jetson/benchmark.py`'s `TensorRTTorchRunner`),
overlays the semantic mask and derived perception regions, and writes an annotated output
video plus a JSON summary using the same non-fabrication contract as `scripts/predict.py`.
Only the ONNX/CPU path is tested here; real TensorRT execution remains a human-gated
on-device action per `scripts/jetson/AGENTS.md`.

The first real L4 run at application commit `42be8d6…` proved the bootstrap, hermetic
92-package sync, and OpenMMLab install all complete cleanly on real hardware, then failed
`five-model-runtime-canary` on PIDNet-S with "AMP/FP16 stack-probe gradient is missing or
non-finite." `scripts/train/train_semantic.py::_probe_model` hardcoded `torch.float16` for
its mixed-precision canary regardless of what the real training path would ever select;
`train_model`'s own `precision="auto"` policy prefers bf16 whenever the device supports it
(L4 does) specifically because fp16's narrow exponent range can overflow in wide
multi-scale modules like PIDNet's SPP even when bf16 would not. Application commit
`b22fd12…` extracts that decision into
`edgeguard.rescue.mmseg_runtime.resolve_auto_precision()` and makes the probe call it, so
the probe validates the precision that will actually run instead of a stricter one that
never will. This fix is reasoned from the real failure log and the codebase's own stated
precision policy; no CUDA device was available to verify it empirically in this
environment, so it remains unconfirmed until the next real L4 run. The notebook is
repinned to commit `b22fd12…`; the hostile-context remote Linux workflow and a real L4
canary have not yet been re-run against it.

## Deliveries

The package stage produces `EdgeGuard_Jetson_Release.zip`,
`EdgeGuard_Thesis_Bundle.zip`, `EdgeGuard_Streamlit_Demo.zip`, and
`release_index.json`, each hash verified. The Jetson archive contains five ONNX graphs,
checkpoints/configs and golden vectors, but never a TensorRT engine. Jetson telemetry stays
`not_run` until a target-device benchmark is supplied.

## Verification boundary

Local Ruff, mypy, pytest, deterministic notebook generation and claim-safe local cell
execution validate engineering contracts only. No local test creates a scientific metric.
The current delivery passes 495 tests with seventeen environment-gated skips without the
pinned MMSeg stack present; with the pinned stack available (`EDGEGUARD_MMSEG_CHECKOUT`
pointed at the exact commit `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8` checkout) it passes
510 tests with two skips, including the new real per-architecture `model.loss()`
regression tests, the `resize_train_ids` train-ID-space regression test, the
`run_video_demo.py` ONNX/CPU fixture end-to-end test, and the `resolve_auto_precision`
policy tests. The master notebook was generated twice byte-identically at SHA-256
`1b7349e03a65dab05ee1a9a3ace840e65ac540be0c50457f01f2691ba4975559`.
Remote Linux workflow `31129018003` completed successfully at an earlier application commit
(`3f3ef8f…`) with the exact Colab failure context injected
(`MPLBACKEND=module://matplotlib_inline.backend_inline`, host uv and virtualenv state); it
has not yet been re-run at the current commit `b22fd12…`, and claim-safe local cell
execution has not been re-verified at this commit either — both remain pending before the
next real Colab attempt. The AMP-probe precision fix in this commit has also not yet been
confirmed by an actual L4 run.
The notebook is not eligible for a Colab-ready tag until two independent clean L4
five-model FP32/AMP canaries and a real interruption/resume smoke have passed. No training
result, accepted scientific release, TensorRT engine, Jetson measurement, merge, or tag is
claimed by this state file.
