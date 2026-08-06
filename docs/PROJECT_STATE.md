# Project State

Updated 2026-08-06 on branch `stabilize/colab-v2`.

The current Colab v2 application is frozen at commit
`bcfff06bae511da2646fc034291bb5cd33e28405`. Both generated delivery notebooks pin and
verify that exact commit; their double-generation hashes are recorded in the handoff.
The approved application and notebook-delivery commits were pushed to
`origin/stabilize/colab-v2`. The first remote CI run exposed one test-only Streamlit
relative-path incompatibility after all static gates passed; the dashboard test and local
closure probe now use a repository-root absolute path. Merge, tag, release, Drive mutation,
and real training remain outside the approval. Replacement CI run `31107068048` passed the
complete Python 3.10/3.11 matrix. The new runtime still cannot be presented as real Colab
evidence until G1/G3 run in clean Colab sessions.

The first real `audit` target produced a hash-valid 31-file review package. Cityscapes
passed 2975/2975; IDD accounted for all 14,027 official train records, retained 14,018 and
accepted nine known no-usable-class exclusions under the 0.1% quarantine policy. Three
cross-role dHash candidates were visually reviewed as distinct highway scenes. Review then
found that the IDD candidate's `source_manifest_sha256s` was empty: its reported zero
cross-source overlap was therefore not provenance-bound. The new application passes the
Cityscapes candidate to the IDD audit and includes its hash in the completion identity.
Restored legacy IDD audit output is invalidated, while the persistent audit catalog remains
reusable; extraction and full pixel re-audit are not required. The first rerun showed that
the expected-input check rejected the old receipt but the generic quarantine helper still
accepted it without checking those inputs, leaving `idd20k_audit` in place. The helper and
notebook now use the same expected-input identity, so the legacy directory is preserved
under a timestamped quarantine name and the replacement audit gets a clean output root.

The previous hosted/fallback cascade is replaced by one hermetic runtime: uv 0.8.8,
managed CPython 3.11.13, NumPy 1.26.4, PyTorch 2.1.1/cu121, MMEngine 0.10.7,
mmcv-lite 2.1.0, headless OpenCV 4.10.0.84, and MMSegmentation commit
`c685fe6767c4cadf6b051983ca6208f1b9d1ccb8`. Normal dependencies use a generated
manylinux_2_28 hash lock. mmcv-lite uses a separate hash lock with `--no-deps` so its
GUI-OpenCV metadata cannot replace the headless runtime. A failure never selects another
matrix automatically.

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

- The v2 campaign ID is `semantic-cs-idd-v2`. SegFormer-B0, Fast-SCNN, and PIDNet-S must
  pass environment/AMP canary, smoke, recovery, and pilot before DDRNet-23-Slim and
  BiSeNetV2 enter extension smoke and five-model screening. The Kaggle BDD mirror remains
  outside scientific selection.
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
- The notebook generator is output-free and deterministic. Both tracked notebooks now pin
  the approved immutable application commit, compile, execute in the external-action-free
  local harness, and regenerate byte-identically.
- The known Colab bootstrap failures are removed from the implementation: uv no longer
  receives `--upgrade-strategy only-if-needed`, and no later MMCV command can resolve
  NumPy 1.26 to NumPy 2.x. Runtime reuse requires matching project, lock, framework,
  CUDA, FP16 and three-model canary identities.
- The generated training notebook delegates smoke through final training to one Python
  orchestrator. It verifies prerequisites and completed hashes, resumes only matching
  identities, permits one OOM batch/accumulation adjustment at effective batch four, and
  emits uniform run/environment/metrics/resource/artifact/failure records.
- MMEngine/MMCV are isolated in a two-wheel no-deps hash lock; the main lock rejects GUI
  OpenCV and requires headless OpenCV plus direct `rich`. Evaluate/export/report require a
  human-accepted release; manifest freeze and release promotion both require explicit
  candidate-hash-bound human review receipts.
- Streamlit defaults to accepted, hash-verified demo bundles and keeps one selected model
  resource active. Accepted releases can be collected into thesis source/LaTeX/provenance
  bundles with explicit `not_run` coverage instead of fabricated figures.
- Colab can build a device-neutral Jetson ZIP containing static ONNX, preprocessing and
  ontology identities, golden input/output and equivalence evidence. It cannot contain a
  TensorRT engine or Jetson performance claim.
- Drive now has explicit archive/quarantine/bundle/manifest/campaign/download/source
  roots. Snapshots exclude staged datasets; a bounded review ZIP makes reports and thesis
  figures downloadable without copying checkpoints or licensed data.
- Frozen multi-domain statistics generate measured-only CSV plus 300-DPI PNG/PDF class
  distribution, imbalance/weights, split-size and source-example figures with hashes.
- Both P0 delivery notebooks are regenerated from source against the approved application
  commit. The full test suite includes their external-action-free local execution and exact
  pin assertions; real Drive/CUDA behavior remains an external Colab gate.
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
4. After an approved clean commit is pinned, prove the hash lock in two independent clean
   Colab GPU sessions. Advance `CAMPAIGN_TARGET` through smoke, pilot, screening, HPO and
   final. The notebook invokes one v2 orchestrator command; do not delete recovery objects.
5. Only after final completes, review the generated release candidate and promote it with
   a candidate-hash-bound human receipt. Then set `evaluate` plus `ALLOW_FINAL_DATA=True`;
   prepare/open ACDC separately and never use its result to change model or preprocessing.
6. Build/benchmark TensorRT FP16 manually on Jetson, then decide whether the detection
   phase gate passes.

Implementation, local tests, Colab measurements, external results, Jetson measurements,
and human acceptance remain distinct states.
