# Project State

Updated 2026-08-06 on `stabilize/colab-v2`.

## Current delivery

The Colab v3 application commit is
`2495354d02e45cc4f8748e94cfcf1862ae48a295`. The only generated notebook is
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
