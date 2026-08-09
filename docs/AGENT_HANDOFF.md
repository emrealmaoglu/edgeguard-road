# Agent Handoff

- **Branch:** `stabilize/colab-v2`
- **Application commit pinned by notebook:**
  `3262af8` (see `git log` for the full SHA)
- **Campaign:** `semantic-cs-idd-v3`
- **Notebook:** `notebooks/EdgeGuard_Master_Colab.ipynb`
- **Classification:** locally verified engineering delivery; real Colab GPU/training and
  Jetson evidence remain external. Remote CI and claim-safe notebook execution have not yet
  been re-run at this commit (see Local gates). A real L4 run at commit `c4008d9…`
  confirmed the five-model canary and data-staging phases both pass cleanly, but crashed on
  the very first `smoke`/`segformer_b0` attempt with `ValueError("Drive recovery checkpoint
  belongs to a different immutable run")` before any training step ran — a fifth real bug,
  fixed by this commit (see below). Not yet re-confirmed on real L4 hardware (see Local
  gates).
- **Note on "claim-safe local cell execution":** this check (see
  `scripts/dev/run_campaign_notebook_harness.py`) only proves the generated notebook's
  cells import and execute their own syntax correctly under
  `EDGEGUARD_NOTEBOOK_LOCAL_TEST=1` — the actual training call is stubbed behind a
  hardcoded `{"scientific_status": "not_run"}` dict and never touches `Runner.train()`,
  `EdgeGuardRecoveryHook`, or `val_dataloader`. Do not read "claim-safe local cell execution
  passed" as "the pipeline was exercised" — the new
  `tests/integration/test_colab_pipeline_cpu_rehearsal.py` (see below) is what actually
  does that.

## What changed

- Replaced fourteen entry notebooks with one generated Run-all notebook.
- Added the `all` production orchestrator from preflight through final delivery packages.
- Reuses exact v2 Cityscapes/IDD audit candidates and verified Drive data without rescans.
- Requires all five canary, screening, and final models; HPO remains top-two.
- Added forced smoke interruption/resume evidence and general immutable checkpoint restore.
- Added owner-preauthorized exact-source/release policy while keeping official validation
  after model selection.
- Added five-model ONNX/golden-vector Jetson packaging, accepted Streamlit packaging, and
  measured thesis tables/figures/galleries.
- Added L4/High-RAM/disk gates and atomic Drive publication of final ZIPs.
- Replaced the hosted-Python project import with a standard-library-only runtime bootstrap.
- Runs restore and data preparation only after the verified Python 3.11 environment exists.
- Isolates v3 work state and preserves/skips incompatible commit-bound local/Drive state.
- Streams the real child error and packages stage, bootstrap and log-tail diagnostics.
- Resolves both `bin/uv` and Colab system-pip `local/bin/uv` private-prefix layouts and
  carries the verified executable path into the runtime installer.
- Prioritizes the pinned wheel's CUDA libraries over Colab's mutable toolkit paths while
  preserving host driver discovery.
- Reuses the hash-verified bootstrap receipt for canary execution instead of performing a
  second destructive package sync.
- Reports the exact failed module or CUDA initialization stderr and retains MMEngine's
  required `pkg_resources` path through the Setuptools 80.9.0 lock.
- Preserves failed canary evidence under a timestamped quarantine root before retry, so
  repeated Run-all attempts start with clean runtime evidence without deleting history.
- Firewalls Colab's hosted Matplotlib/Python/virtualenv/pip/uv state before bootstrap and
  every locked-runtime child; forces `Agg` and isolated cache roots.
- Runs a real headless PNG probe before the five-model canary and records the non-secret
  environment-contract identity in runtime evidence.
- Resumes correctly when the one permitted OOM batch reduction is followed by the forced
  smoke interruption; the resumed command and checkpoint identity are both verified.
- Marks every acceptance-mode phase `not_run`, preventing fixture execution from becoming
  measured or accepted scientific evidence.
- (This commit) Fixed two real training-blocking defects found by a Claude Code cross-file
  review: PIDNet-S's shared pipeline was missing `GenerateEdge`, crashing `PIDHead`'s
  boundary loss on the first optimizer step of any stage; the CE-vs-weighted-CE override
  only matched `CrossEntropyLoss`, silently no-oping for DDRNet-23-Slim and PIDNet-S, whose
  dominant losses are `OhemCrossEntropy`. Both were reproduced and fixed against the real
  pinned MMSeg checkout, not just inspected.
- (This commit) Added `tests/unit/test_mmseg_real_training_step.py`, the first test in the
  repository to run a real `model.loss()` forward/backward step per architecture against
  the real resolved MMSeg config; wired into `semantic-framework-cpu-probe.yml`.
- (This commit) The `stage-data` phase now verifies every manifest-referenced image/mask
  file exists on local disk instead of only checking the manifest JSON exists
  (`verify_manifest_data_is_staged`); Drive archive/shard copies now fail closed on a
  stalled read via a bounded stall-timeout guard instead of risking an indefinite hang.
- (Follow-up commit `1387322…`) Fixed `resize_train_ids()` validating against the wrong
  ID space (source label IDs instead of train IDs); reconciled `SYSTEM_ARCHITECTURE.md`,
  `DATA_CATALOG.md`, and the eval-config resolution-mismatch docs found stale by the same
  review; recorded the IDD20K native-label-loss-on-shard-packaging limitation in
  `docs/TASKS.md` rather than acting on it (owner decision: needs a separate, costly
  Drive shard re-processing approval).
- (Commit `42be8d6…`) Raised `configs/rescue/semantic_first.yaml`'s `workers` from 2 to 6
  to use a Colab High-RAM instance's vCPUs more fully during data loading;
  `effective_batch` and every frozen HPO/step budget are unchanged. Confirmed training
  precision already defaults to bf16/AMP on CUDA (`train_model`'s `precision="auto"`), so
  no code change was needed there. Added `scripts/jetson/run_video_demo.py` (DEMO-02): a
  video-frame perception-overlay demo for ONNX Runtime (local/CPU, tested) and target-
  device TensorRT (human-gated per `scripts/jetson/AGENTS.md`, untested here).
- (Commit `b22fd12…`) A real L4 run at `42be8d6…` failed on PIDNet-S with "AMP/FP16
  stack-probe gradient is missing or non-finite" — `train_semantic.py::_probe_model`
  hardcoded `torch.float16` for its mixed-precision canary instead of following the same
  `precision="auto"` → bf16-on-capable-hardware policy `train_model` already uses, so it
  tested a precision (fp16) the real training run would never actually select on an L4.
  Extracted `edgeguard.rescue.mmseg_runtime.resolve_auto_precision()` and made the probe
  call it. This fix was later **confirmed on real L4 hardware**: the next run passed the
  five-model AMP stack-probe outright (`fp16_finite_model_count: 5`, all five architectures
  including PIDNet-S, GPU `NVIDIA L4`).
- (Commit `ff26422…`) That same real L4 run then failed building `val_dataloader` for
  every model/stage (first hit: `smoke`/`segformer_b0`) with
  `TypeError: Pad.__init__() got an unexpected keyword argument 'seg_pad_val'`.
  `_evaluation_pipeline()` in `mmseg_runtime.py` passed `pad_val` and `seg_pad_val` as two
  separate constructor kwargs to the `Pad` transform; the pinned mmcv-lite `Pad`
  (`mmcv/transforms/processing.py`) only accepts a single `pad_val`, either a number or a
  `dict(img=..., seg=...)` — there is no `seg_pad_val` argument at all. Fixed by combining
  both into `pad_val={"img": 0, "seg": config.ignore_index}`; `_inference_pipeline()`'s
  plain `pad_val=0` was likewise made explicit as `{"img": 0}` for consistency (behavior
  unchanged — mmcv's `Pad` already treated a bare int as image-only padding). This fix was
  later **confirmed on real L4 hardware**: the next run's `smoke`/`segformer_b0` training
  actually started and ran to the intentional interruption at optimizer step 25.
- (Commit `e3f3159…`) That same real L4 run's resume subprocess then failed with
  `FileNotFoundError: recovery_25.pth can not be found.`. `EdgeGuardRecoveryHook`
  (`mmseg_components.py`) wrote only the bare checkpoint filename into
  `<work_dir>/last_checkpoint`; this codebase's own reader (`latest_checkpoint()` in
  `colab_recovery.py`) resolves a relative marker against `work_dir`, but MMEngine's own
  built-in auto-resume (`Runner.load_or_resume()` → `find_latest_checkpoint()`) returns the
  raw marker content verbatim and resolves it relative to the process's cwd — the project
  root for every child process `colab_pipeline.py` spawns, not the run's `work_dir`. The
  identical bug existed in `train_model`'s Drive cross-session recovery path
  (`mmseg_runtime.py`), reachable whenever a new session restores a checkpoint published by
  a dead prior session — the exact scenario this recovery system exists to survive. Fixed
  both write sites to write the absolute path, matching MMEngine's own `CheckpointHook`
  convention; confirmed pathlib join with an absolute right-hand side leaves every existing
  bare-filename reader in this codebase unaffected. Added
  `tests/unit/test_mmseg_recovery_checkpoint_marker.py`, which reproduces the exact failure
  via MMEngine's real `find_latest_checkpoint()` against the marker the hook writes, and
  confirmed it fails on the pre-fix code and passes on the fix. This fix was later
  **confirmed on real L4 hardware**: the resume subprocess found and loaded
  `recovery_25.pth` and continued training.
- (Commit `d7a4430…`) Building a real local CPU rehearsal harness (see next entry) — the
  first thing in this repository to ever exercise a real end-of-stage validation pass —
  surfaced a fourth real bug: `_evaluation_pipeline()`/`_inference_pipeline()`'s `Pad` step
  passed `config.crop_size` (this codebase's own `(h, w)` convention) directly as `Pad`'s
  `size`. The pinned mmcv-lite `Pad` documents `size` as `(w, h)` and internally reverses it
  before calling `mmcv.impad(shape=...)`, which itself expects `(h, w)` — so the pad target
  was silently transposed. Invisible for a square crop or when the swap happens to survive;
  real Cityscapes' non-square 512×1024 crop would not have survived it, but no real Colab
  run had ever reached validation to find out. Fixed by reversing `crop_size` the same way
  the `Resize` step right above it already does. Added a regression test
  (`test_evaluation_and_inference_pad_produce_crop_size_shaped_output`) building the real
  pipeline through `Compose()` with a non-square crop and asserting the packed input
  tensor's shape matches `crop_size` exactly; confirmed it fails on the pre-fix code and
  passes on the fix — not yet re-confirmed on real L4 hardware.
- (Commit `c4008d9…`) Added a real local CPU rehearsal harness
  (`tests/support/tiny_pipeline_fixture.py`,
  `tests/integration/test_colab_pipeline_cpu_rehearsal.py`) that drives the real
  subprocess-spawning `ColabPipeline` (not a mock) through a real smoke-stage
  interrupt-then-resume cycle for all 5 models on CPU with tiny synthetic fixture data —
  the exact class of gap that let all four bugs above reach a real Colab GPU before being
  caught. Confirmed locally: passes end to end for all 3 core models, including a real
  computed mIoU at the final validation step. Wired into
  `semantic-framework-cpu-probe.yml` as a mandatory CI step.
- (Commit `3262af8…`) A real L4 run at `c4008d9…` (five-model canary and data staging both
  passed) crashed on the first `smoke`/`segformer_b0` attempt — no training step ever
  started — with `ValueError("Drive recovery checkpoint belongs to a different immutable
  run")` (`mmseg_runtime.py:890`). Root cause: on a fresh Colab session, local `/content` is
  empty (no local `run_identity.json`), so `ColabPipeline._run_training_phase`'s decision to
  append `--resume` rests entirely on `_recovery_pointer_exists()`, which only checks
  whether a Drive pointer *file* exists for the artifact_id — it has zero identity
  awareness. Every commit changes `project_commit`, which is one of ~20 fields in
  `train_model()`'s `identity` dict, so it invalidates `identity_sha256` for every
  previously-published Drive checkpoint campaign-wide, including ones from unrelated
  stages/models. `train_model()` then correctly detected the mismatch but incorrectly
  treated it as fatal, crashing the whole 5-model campaign instead of just ignoring an
  opportunistic resume that turned out to be stale. Fixed by distinguishing this
  *speculative* auto-resume (no local `run_identity.json` ever existed) from an *explicit*
  resume of a known local run: a stale Drive checkpoint under the speculative path is now
  skipped gracefully (a `stale_recovery_skipped.json` record is written for evidence) and
  training proceeds fresh; the explicit local-resume identity check
  (`existing != identity`, further down in `train_model()`) is untouched — that one is a
  real safety property against silently mixing checkpoints across incompatible protocol
  versions, and should stay a hard failure. Added `colab_recovery.peek_recovery_metadata()`
  to read a pointer's receipt metadata cheaply (no full checkpoint byte-copy) before
  deciding whether to restore. Added a CPU rehearsal regression test
  (`test_smoke_target_skips_stale_cross_commit_drive_recovery_checkpoint`) that publishes a
  Drive recovery pointer with a fabricated mismatched identity before running the pipeline;
  confirmed it fails on the pre-fix code with the exact real-Colab error, and passes on the
  fix. Not yet re-confirmed on real L4 hardware. **Left as an explicit open question, not
  implemented:** whether `project_commit` should remain part of the strict identity-compare
  value at all, versus being recorded as provenance-only metadata (so only
  scientifically-relevant fields — protocol/dataset/hyperparameter hashes — invalidate a
  Drive checkpoint, not any unrelated commit). This is a reproducibility/scientific-
  integrity policy call reserved for the human project owner, not an engineering decision.

## Local gates

- Ruff and format checks pass for the full repository.
- Mypy passes for all 116 configured source modules.
- Full pytest passes: 485 passed, 17 environment-gated skipped without the pinned MMSeg
  stack; 19 of 19 in the mmseg-gated suite (`tests/unit/test_mmseg_real_training_step.py` +
  `tests/integration/test_colab_pipeline_cpu_rehearsal.py`) pass with
  `EDGEGUARD_MMSEG_CHECKOUT` pointed at the pinned `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8`
  checkout, including the real per-architecture `model.loss()` tests, the `last_checkpoint`
  marker regression test, the `Pad` orientation regression test, and the new stale-Drive-
  recovery regression test. None of this exercises real CUDA/AMP behavior — that only
  happens on a real L4.
- All three `tests/integration/test_colab_pipeline_cpu_rehearsal.py` tests (fast-tier 3
  core models, full-tier 5 models via extension-smoke, and the new stale-recovery test)
  pass locally end to end against the real pinned stack (confirmed manually, ~24 minutes
  combined; not yet run in CI at this commit). The new stale-recovery test was confirmed to
  fail on the pre-fix code with the exact real-Colab error
  (`ValueError("Drive recovery checkpoint belongs to a different immutable run")`) via a
  temporary `git stash` of only the two fix files, and to pass with the fix restored.
- Master notebook generation is byte-identical across two runs.
- The notebook SHA-256 after pinning is
  `ad026d4ec1209b40c2b8a8a5499cb19111403f26adc272792cce17dd1ffa4944`.
- **Pending at this commit:** claim-safe local cell execution has not been re-verified,
  remote Linux workflow `semantic-framework-cpu-probe.yml` has not been re-run (including
  the rehearsal step), and this stale-Drive-recovery fix has not been confirmed on real CUDA
  hardware — the AMP-probe, `Pad`/`seg_pad_val`, `last_checkpoint`, and `Pad`-orientation
  fixes all have real-hardware confirmation so far. The prior application commit
  (`3f3ef8f…`) passed remote run `31129018003` with Colab's exact hostile inline backend and
  host uv/virtualenv state injected; that evidence does not carry over to this commit and
  should be re-established before a real Colab attempt.

## Next external action

Push this commit, then open the master notebook from the pushed branch in a fresh Colab L4
+ High-RAM runtime and use Run all (Colab Pro/Pro+ background execution is recommended so
the session survives closing the browser tab). The five-model AMP canary, the
`val_dataloader` build, and the `last_checkpoint` resume are all already confirmed passing
on real hardware; watch specifically whether `smoke` now reaches its end-of-stage
validation pass cleanly for every model (previously untested on real hardware — this
commit's `Pad`-orientation fix targets exactly that step) and whether all five models'
smoke stages complete end to end. If the session ends, repeat Run all in a new compliant
runtime — this is a Colab platform limit, not something the notebook can automate away. Do
not change the notebook or select stages manually.

Do not create `colab-v0.1.0-rc1` until two independent clean L4 sessions pass the exact
lock/five-model FP32/AMP canary and the real 50-step interruption/resume proof. After the
campaign completes, build TensorRT only on the target Jetson and attach actual 25W
telemetry. Do not merge, tag, open sealed datasets, upgrade JetPack, or change Jetson power
mode without a separate explicit decision.
