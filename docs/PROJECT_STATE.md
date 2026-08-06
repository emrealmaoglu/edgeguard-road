# Project State

Updated 2026-08-07 on `stabilize/colab-v2`.

## Current delivery

The Colab v3 application commit is
`005eb035e515902cfd4cdd7c8a4426ae3fa52437`. The only generated notebook is
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

All five models pass the runtime canary contract. Core smoke intentionally interrupts at
step 25 of 50 and must resume from the same optimizer/checkpoint identity. Checkpoints are
published every 500 optimizer steps or ten minutes. A single OOM retry may change device
batch only when accumulation keeps effective batch four.

All five models receive 40,000-step final training. Only train-select evidence selects the
recommended model. Official validation opens after policy acceptance and cannot change the
choice. The selected model receives weighted-CE and 256×512 ablations; deployment remains
512×1024.

## Deliveries

The package stage produces `EdgeGuard_Jetson_Release.zip`,
`EdgeGuard_Thesis_Bundle.zip`, `EdgeGuard_Streamlit_Demo.zip`, and
`release_index.json`, each hash verified. The Jetson archive contains five ONNX graphs,
checkpoints/configs and golden vectors, but never a TensorRT engine. Jetson telemetry stays
`not_run` until a target-device benchmark is supplied.

## Verification boundary

Local Ruff, mypy, pytest, deterministic notebook generation and claim-safe local cell
execution validate engineering contracts only. No local test creates a scientific metric.
The notebook is not eligible for a Colab-ready tag until two independent clean L4
five-model FP32/AMP canaries and a real interruption/resume smoke have passed. No training
result, accepted scientific release, TensorRT engine, Jetson measurement, merge, or tag is
claimed by this state file.
