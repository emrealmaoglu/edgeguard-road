# Project State

Updated 2026-07-31 on branch `rescue/semantic-first`.

The published delivery notebooks are pinned to implementation commit
`5cc578cb9f15aa7a560108840f3055ae2f4e4733`; a later branch update cannot silently alter
the Colab runtime code or scientific protocol.

The Colab-resilience revision replaces the previous snapshot model with content-addressed,
current/previous-generation recovery. Training publishes full optimizer/scheduler/AMP/RNG
state every 500 optimizer steps or ten minutes; HPO persists per rung with an atomic SQLite
backup. IDD preparation publishes verified 500-sample shards, audit publishes 250-sample
catalog chunks, archive hashes use stable receipts, and stage completion is accepted only
through output/input hash receipts. The active source contract is Cityscapes + IDD20K.

Colab enforces that pin after checkout. The external-action-free local notebook harness
intentionally runs the current checkout without cloning, so it records a pin mismatch
instead of rejecting a legitimate later documentation/CI commit. GitHub CI installs the
`rescue` extras as well as development tools, keeping thesis-figure tests representative.

The first real preflight initially exposed only an exit code. After command-tail logging
was added, the rerun established the exact root cause: the child process started in
`/content` and resolved a relative `configs/...` default outside the checkout. All notebook
subprocesses now run from the exact repository root, and active CLI config defaults are
anchored to their script checkout. Inventory also records mounted-Drive hash read errors
without weakening mandatory post-copy verification.

The first official IDD20K preparation then remained silent for more than 12 hours after
archive copy. This is not accepted as normal execution evidence. Inspection found a
pathological gzip TAR access pattern (`getmembers` followed by name-sorted random access),
two full archive reads for MD5/SHA-256, serial full-resolution dual-mask rendering, no
heartbeat, and no reliable child-process cancellation. The preparation path now streams
TAR members once in physical order, computes both hashes in one read, applies a vectorized
ontology LUT with bounded mask workers, emits live progress, preserves verified ephemeral
archive copies for retry, hashes bundles while writing, and terminates subprocess groups
on notebook interruption. These corrections are locally verified but require a new exact
Colab pin before the real IDD run is retried.

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
- Long IDD preparation and bundle/staging operations emit phase, file/byte progress and
  periodic free-disk liveness. A failed retry does not recopy already verified `/content`
  archives, while invalid or partial cache entries remain fail-closed.

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

1. Run `EdgeGuard_Data_Preflight_Colab.ipynb` from the existing `private_inputs/` uploads
   with Run all; retain digest, Cityscapes bundle and IDD shard receipts.
2. Stage/audit Cityscapes + IDD20K and review/freeze their group-safe manifests and rare
   classes. Review BDD separately as provisional, non-scientific evidence.
3. If official BDD packages are later obtained, add them as a new source ablation rather
   than replacing the already recorded mirror evidence.
4. Advance `CAMPAIGN_TARGET` through smoke, pilot, screening, HPO and final. Reset recovery
   is automatic; do not delete campaign recovery objects.
5. Only after final completes, set `source_eval` plus `ALLOW_FINAL_DATA=True`; prepare/open
   ACDC separately and never use its result to change model or preprocessing.
6. Build/benchmark TensorRT FP16 manually on Jetson, then decide whether the detection
   phase gate passes.

Implementation, local tests, Colab measurements, external results, Jetson measurements,
and human acceptance remain distinct states.
