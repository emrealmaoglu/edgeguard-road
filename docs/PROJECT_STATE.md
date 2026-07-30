# Project State

Updated 2026-07-30 on branch `rescue/semantic-first`.

The published delivery notebooks are pinned to implementation commit
`f25fc4faa6f6a55aae226aec9c269d8c5cf3102c`; a later branch update cannot silently alter
the Colab runtime code or scientific protocol.

Colab enforces that pin after checkout. The external-action-free local notebook harness
intentionally runs the current checkout without cloning, so it records a pin mismatch
instead of rejecting a legitimate later documentation/CI commit. GitHub CI installs the
`rescue` extras as well as development tools, keeping thesis-figure tests representative.

The first real preflight exposed a mounted-Drive archive hashing interruption. Inventory
now records per-file hash read errors without weakening the later integrity gate; archive
copies use bounded retries and are verified locally before extraction. All post-bootstrap
subprocesses stream redacted command logs into failure packages, eliminating the generic
exit-code-only diagnostic gap.

## Implemented and locally verified

- Five-model, two-source scientific training/evaluation/HPO contracts are active for
  Cityscapes + IDD20K. The available Kaggle BDD mirror is audit/smoke-only.
- Safe dataset-specific preparation now consumes untouched Cityscapes, BDD100K, and
  IDD20K archives. It verifies published identities where available, rejects unsafe
  members/collisions/unknown labels, supports IDD Part II JPG, renders pinned AutoNUE
  source masks, and never overwrites native annotations.
- BDD `kaggle_mirror` is automatically marked scientifically ineligible. Only the two
  official BDD 10K Semantic packages can support final scientific training.
- The Colab preflight now prepares one dataset at a time in `/content`, streams it into
  a hash-bound Drive bundle, and removes the temporary tree. Training staging retains
  the 175 GiB dataset/25 GiB runtime reservation.
- Prediction and Streamlit now expose road, ego-reachable corridor, confidence,
  normalized entropy, unreliable pixels, semantic regions, deterministic attention
  contributions, and optional source-frozen frame shift alerts.
- Perception evaluation includes road IoU, boundary F1, false-drivable rate,
  fragmentation, semantic-component coverage, merge and fragmentation. These are not
  detection metrics.
- Frame-shift evaluation reports source-vs-external AUROC/AP and alert rates without
  external threshold tuning.
- Target-only tools build a static TensorRT FP16 engine and run sustained Jetson
  benchmarks. They never change power mode and refuse evidence overwrite. No target
  action was executed during local implementation.
- Delivery notebooks are regenerated, output-free, and syntactically validated.
- The prior Colab failure mode is removed: the selected hosted/fallback environment is
  resolved from its compatibility receipt instead of hard-coding fallback Python and
  MMSeg paths. Both paths install the complete Colab project extras.
- Drive now has explicit archive/quarantine/bundle/manifest/campaign/download/source
  roots. Snapshots exclude staged datasets; a bounded review ZIP makes reports and thesis
  figures downloadable without copying checkpoints or licensed data.
- Frozen multi-domain statistics generate measured-only CSV plus 300-DPI PNG/PDF class
  distribution, imbalance/weights, split-size and source-example figures with hashes.
- Both delivery notebooks executed all 16 code cells locally in safe contract mode. This
  verifies integration and control flow, not Colab CUDA or scientific execution.
- A read-only connected-Drive audit corrected the storage assumption. Existing
  `private_inputs`, prepared Cityscapes v1, its 6.99 GB verified bundle, real manifests,
  historical compatibility evidence and `EG-REAL-001` are preserved. New staging reuses
  the exact pinned Cityscapes bundle and adds BDD/IDD roots without migration.
- A second read-only Drive audit confirmed IDD Part I/II and the Kaggle BDD ZIP now sit
  beside Cityscapes in `private_inputs/`. Notebook discovery accepts that exact layout;
  no upload, move, extraction, or reorganization is required.
- Preparation now has a conservative archive-size multiplier plus 25 GiB disk reserve,
  bounded same-session retry cleanup, idempotent bundle reuse, and post-build inventory
  refresh. Runtime installation adds dependency checking and imports ONNX Runtime,
  Optuna, Streamlit and reporting packages before accepting the five-model probe.
- Both notebooks now persist clone/bootstrap failures and all later unhandled errors as
  redacted, append-only JSON/ZIP evidence under `EdgeGuard/failures/`. Reports include
  stage/commit/platform/disk context and bounded hashed logs, never datasets/checkpoints
  or environment variables. The latest ZIP can be downloaded from the final cell.

## Available external archives

The connected Drive contains official Cityscapes packages, official IDD20K Part I/II,
and a Kaggle BDD mirror under `private_inputs/`. Local archive directories now contain
only metadata files because the user removed the large bytes to recover disk space. The
BDD mirror is retained for preparation/catalog/audit but cannot enter HPO or main claims.
Dataset bytes and machine-local paths are not tracked in Git.

## Evidence boundary

The historical pretrained PIDNet-S Cityscapes-val reference remains separately
documented. No project-trained multi-domain checkpoint, current CUDA model comparison,
HPO result, official source validation, ACDC/sealed-external result, ONNX finalist,
TensorRT engine, or Jetson performance measurement exists yet. Fixture tests and dry
runs are engineering evidence only.

## Immediate external execution order

1. Run `EdgeGuard_Data_Preflight_Colab.ipynb` from the existing `private_inputs/`
   uploads with `RUN_ARCHIVE_PREPARATION=True`; retain hashes and bundle receipts.
2. Stage/audit Cityscapes + IDD20K and review/freeze their group-safe manifests and rare
   classes. Review BDD separately as provisional, non-scientific evidence.
3. If official BDD packages are later obtained, add them as a new source ablation rather
   than replacing the already recorded mirror evidence.
4. Run one-batch, five 50-step smokes, pilots, screenings, and early ONNX checks.
5. Freeze top two, then HPO/ablations/finals/calibration; only afterward open official
   validation, ACDC, and sealed external evaluation.
6. Build/benchmark TensorRT FP16 manually on Jetson, then decide whether the detection
   phase gate passes.

Implementation, local tests, Colab measurements, external results, Jetson measurements,
and human acceptance remain distinct states.
