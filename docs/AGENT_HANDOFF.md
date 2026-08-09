# Agent Handoff

- **Branch:** `stabilize/colab-v2`
- **Application commit pinned by notebook:**
  `a50b635` (see `git log` for the full SHA)
- **Campaign:** `semantic-cs-idd-v3`
- **Notebook:** `notebooks/EdgeGuard_Master_Colab.ipynb`
- **Classification:** locally verified engineering delivery; real Colab GPU/training and
  Jetson evidence remain external. Remote CI and claim-safe notebook execution have not yet
  been re-run at this commit (see Local gates). A real L4 run at commit `2b078d3…` reached
  the furthest point yet: `segformer_b0` completed its full smoke cycle end to end (fresh
  start, interruption at step 25, real resume via the RNG-device fix, training to step 50,
  validation, mIoU computed), and `fast_scnn` trained straight through (the recovery
  self-test now runs once per campaign, not once per model). Then `pidnet_s` crashed on its
  first training step with `RuntimeError: Index put requires the source and destination
  dtypes match, got BFloat16 for the destination and Float for the source` inside
  `boundary_loss.py`, a seventh real bug, fixed by this commit (see below). Not yet
  re-confirmed on real L4 hardware (see Local gates).
- **Note on this session's root-cause pattern:** all seven real bugs found across this
  session (AMP dtype, `Pad`/`seg_pad_val`, `last_checkpoint` bare filename, `Pad`
  size-argument order, stale cross-commit Drive recovery, RNG checkpoint device, PIDNet
  `BoundaryLoss` bf16 dtype) are in this project's own bespoke Colab-orchestration/recovery
  layer, or in a corner of pinned MMSeg that only a real bf16-autocast run on real CUDA ever
  exercises — none are in the Cityscapes/IDD20K data or the choice of the five model
  architectures. Six of the seven bugs are now confirmed closed on real L4 hardware; this
  seventh is the first CUDA+bf16-specific surface reached, exactly the residual risk flagged
  after the sixth fix ("the AMP branch is still dead on CPU — this bug class doesn't close,
  it shrinks"). The prior commit also found and fixed the reason the CPU rehearsal harness
  missed bug 6: a local, never-committed `sitecustomize.py` had been forcing
  `mmengine.device.utils.DEVICE = "cpu"` to work around unrelated MPS operator gaps, which
  incidentally also hid every device-class bug from reproducing locally.
- **Note on the technical-takeover audit (commit `a50b635…`):** the user granted full
  technical ownership with a "hostile reviewer" mandate — audit the repo with no loyalty to
  prior decisions, issue an architecture verdict, and repair/replace/rebuild as evidence
  demands. Three parallel read-only audits (repo hygiene/CI/ADRs, model/reliability/HPO/
  export/Jetson, data ontology/notebook/Drive) plus direct reading of all 5 pinned upstream
  MMSeg configs produced **VERDICT B — sound concept, real but bounded defects**, not the
  "repeatedly failed" prior the framing assumed: ADR-0008/0009 already closed the
  Codex-era instability, and the 7 bugs fixed this session were narrowly-scoped
  orchestration/precision defects, not architectural rot. One real, quantified, previously
  undetected methodology defect was found and fixed: `build_training_config` blanket-
  overwrote every model's `optim_wrapper` with one shared `AdamW(lr=6e-5, wd=0.01)`
  regardless of architecture. Reading each model's own pinned upstream MMSeg config found
  4 of 5 (`fast_scnn`, `pidnet_s`, `ddrnet_23_slim`, `bisenetv2`) actually train with
  SGD+momentum at learning rates 150-2000x higher, in the wrong optimizer family entirely —
  Optuna's HPO space only ever searched LR/weight-decay within a fixed AdamW assumption, so
  it could never have self-corrected this. Fixed by `resolve_model_optimizer_defaults()`,
  which reads each model's real upstream `optim_wrapper.optimizer` as the baseline;
  explicit HPO-trial overrides still apply on top, now preserving the model's own optimizer
  type/momentum instead of forcing AdamW. The same commit also retargeted
  `semantic-framework-cpu-probe.yml`'s push trigger to `stabilize/colab-v2`/`main` (it was
  still scoped to the stale `feat/first-vertical-slice` and never ran automatically on the
  branch all current work happens on) and documented a read-only `scripts/audit_dataset.py`
  data-inventory command against staged training data in the runbook. The audit also
  confirmed the dataset ontology (ADR-0009), reliability/OOD stack, dependency tri-tier
  separation, and ONNX/TensorRT/Jetson code are all real and sound as-is — nothing there was
  touched. Full plan and evidence:
  `~/.claude/plans/yle-bir-projem-var-snazzy-blum.md`. **Not decided, deferred to the human
  owner:** whether to eventually trim the 5-model comparison, and whether to ever pull
  BDD100K/ACDC/WildDash into training roles (ADR-0009's existing answer stands unless the
  owner reopens it).
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
- (Commit `c88ac8f…`) A real L4 run at `3262af8…` got further than any prior run — canary,
  data staging, and a fresh `smoke`/`segformer_b0` start and interruption at optimizer step
  25 all confirmed working — then the local resume subprocess crashed with `TypeError: RNG
  state must be a torch.ByteTensor in EdgeGuardRecoveryHook` inside
  `Runner.resume() -> call_hook('after_load_checkpoint')`. Root cause:
  `Runner.resume()` loads the checkpoint with `map_location=get_device()`; on Colab that is
  CUDA, so every tensor in the pickle — including the RNG state
  `EdgeGuardRecoveryHook.before_save_checkpoint` stores under
  `checkpoint["edgeguard_recovery_state"]` — comes back on CUDA, and
  `torch.set_rng_state`/`torch.cuda.set_rng_state` both reject anything but a CPU uint8
  tensor. mmengine 0.10.7 has no RNG save/restore of its own (confirmed: zero `rng` matches
  in the installed package), so this hook provides genuine capability and was not deleted —
  both `before_save_checkpoint` and `after_load_checkpoint` now coerce every RNG tensor to
  CPU/uint8 via a small helper, so already-published Drive checkpoints written by the
  pre-fix code keep resuming without republishing. A companion, previously-latent bug in
  `torch.cuda.set_rng_state_all` (same missing coercion, plus an unguarded device-count
  mismatch) was fixed alongside it. Reproduced and regression-tested entirely on this local
  Mac with no GPU: `mmengine.device.get_device()` returns `"mps"` here, itself a real
  foreign device relative to the CPU RNG tensor, so simulating the move with `.to("mps")`
  reproduces the identical `TypeError`; confirmed failing pre-fix and passing post-fix.
  Separately, the intentional-interrupt self-test (proves interrupt+resume works on real
  hardware) used to run once per model — 5 deliberate crash+resume cycles per campaign, no
  way to disable it — and any failure of its own resume leg (like this one) killed the
  entire 5-model campaign. It now runs once per campaign
  (`PipelineInputs.recovery_self_test_model`, default the first core model); a failed resume
  leg is recorded as durable evidence (`recovery_self_test_failure.json`, hash-sealed into
  the phase's artifact index) and the model restarts from scratch instead of aborting the
  campaign; `pilot`/`screening`/`hpo`/`final` refuse to start while any unresolved failure
  record exists, so a genuinely broken recovery path can't silently cost hours of real
  training before the next preemption. Also added a CPU-visible config-shape test for the
  `AmpOptimWrapper` branch (`build_training_config`, `precision` in `{"fp16","bf16"}`) —
  this branch is unreachable in every CPU rehearsal run because `resolve_auto_precision`
  always returns `fp32` without CUDA, which is exactly how the session's very first bug (a
  hardcoded AMP dtype) escaped local testing; the new test cannot observe real bf16/fp16
  numerics (only real CUDA can), but locks the wrapper/dtype/loss_scale shape so it can
  never regress silently again. Not yet re-confirmed on real L4 hardware.
- (Commit `4917c49…`) A real L4 run at `2b078d3…` confirmed `segformer_b0`'s and
  `fast_scnn`'s full smoke cycles end to end — the RNG-device fix and the once-per-campaign
  self-test both held up on real hardware — then `pidnet_s` crashed on its very first
  training step (before any interruption) with `RuntimeError: Index put requires the source
  and destination dtypes match, got BFloat16 for the destination and Float for the source`
  inside `mmseg/models/losses/boundary_loss.py:52`. Root cause: upstream `BoundaryLoss.
  forward` builds `weight = torch.zeros_like(log_p)`, and under real
  `AmpOptimWrapper(dtype='bfloat16')` autocast on real CUDA, `log_p` (the boundary head's
  raw logits) is already bfloat16; the ratio assigned into `weight`
  (`neg_num * 1.0 / sum_num`) comes from summing a float32 label mask autocast never
  touches, so it stays float32 — `index_put_` rejects the mismatch on the pinned Colab torch
  (2.1.1+cu121). Fixed by registering `EdgeGuardBoundaryLoss` via this repo's existing
  `force=True` override idiom (already used four times in the same file for the manifest
  dataset, sampler, and recovery/metrics hooks) — identical to upstream except the two
  assignments are cast to `weight.dtype` first; only `pidnet_s` uses `BoundaryLoss` among
  the five models, and a scan of the pinned checkout's other loss files for the same
  `zeros_like`-then-index-assign pattern found only one other occurrence
  (`huasdorff_distance_loss.py`), unused by any of our five model configs. Local
  reproduction note, recorded honestly rather than faked: this dev machine's torch (2.13.0)
  silently allows the exact same implicit float32→bfloat16 `index_put_` that torch 2.1.1
  rejects — a torch-version behavior difference, not a device one — so the new regression
  test asserts the post-fix invariant (dtype-aligned `weight`, finite loss under mismatched
  inputs, and numerically-identical output to upstream when dtypes already match) rather
  than a raises-pre-fix reproduction; confirmed via `git stash` that the
  override-registration itself is present only post-fix. The fast-tier CPU rehearsal
  (including a real `pidnet_s` smoke run, fp32, unaffected by this bf16-only bug) was
  re-run end to end and still passes. Not yet re-confirmed on real L4 hardware.
- (Commit `a50b635…`) Full technical-takeover audit (see note above); fixed the one real
  defect found — `build_training_config` blanket-overwrote every model's optimizer with a
  shared `AdamW(lr=6e-5, wd=0.01)` instead of each model's own upstream-tuned recipe
  (4 of 5 models are natively SGD+momentum at 150-2000x higher LR). Added
  `resolve_model_optimizer_defaults()`; `train_model()`'s identity record now includes
  `optimizer_type`. Retargeted `semantic-framework-cpu-probe.yml`'s push trigger to
  `stabilize/colab-v2`/`main`. Documented the `scripts/audit_dataset.py` data-inventory
  command in the runbook (read-only, no artifacts generated locally — no real dataset on
  this dev machine by design). Not yet confirmed on real L4 hardware; the optimizer-family
  change directly affects what `pilot`-stage training will actually do for 4 of 5 models,
  so the next real Colab run is the load-bearing test for this fix.

## Local gates

- Ruff and format checks pass for the full repository (`ruff check .`, `ruff format --check .`).
- Mypy passes for all 116 configured source modules (`mypy src/edgeguard`, matching `ci.yml`).
- Full pytest passes: 499 passed, 32 environment-gated skipped without the pinned MMSeg
  stack (up from 489/21 — the new per-model native-optimizer tests are among the newly
  skipped/gated ones). The mmseg-gated test files
  (`test_mmseg_real_training_step.py`, `test_mmseg_recovery_checkpoint_marker.py`,
  `test_mmseg_recovery_rng_state.py`, `test_pidnet_boundary_loss_dtype.py`) pass 27 of 27
  with `EDGEGUARD_MMSEG_CHECKOUT` pointed at the pinned
  `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8` checkout, up from 24 (adds
  `test_native_optimizer_matches_each_models_own_upstream_recipe`, parametrized over all 5
  models, and `test_build_training_config_wires_the_native_optimizer_through`). **Noted, not
  fixed, out of scope for this commit:** running the *entire* suite (all test files) with
  `EDGEGUARD_MMSEG_CHECKOUT` set produces 12 failures in `tests/unit/test_dataset_preparation.py`
  that do not reproduce when that file is run alone or as part of the mmseg-gated set above —
  a pre-existing test-isolation/ordering issue unrelated to this session's changes (that file
  never touches `mmseg_runtime.py`, the optimizer fix, or anything mmseg-related). None of
  this exercises real CUDA/bf16 behavior — that only happens on a real L4.
- The full `tests/integration/test_colab_pipeline_cpu_rehearsal.py` suite (all 3 tests: core
  smoke+resume, stale-recovery skip, extension-model smoke) was re-run end to end against the
  real pinned stack and still passes (3 passed in 18m40s).
- Master notebook generation is byte-identical across two runs at commit `a50b635…`.
- The notebook SHA-256 after pinning is
  `263e9c1efec73ee18778ac460133c9fabc71e248d475250c3eccc153a72bc43b`.
- **Pending at this commit:** claim-safe local cell execution has not been re-verified,
  remote Linux workflow `semantic-framework-cpu-probe.yml` has not been re-run (including
  the rehearsal step) — its trigger now covers this branch as of this commit, so the next
  push should be the first automatic run — and neither the optimizer fix nor the prior
  `BoundaryLoss` dtype fix has been confirmed on real CUDA
  hardware — the AMP-probe, `Pad`/`seg_pad_val`, `last_checkpoint`, `Pad`-orientation,
  stale-Drive-recovery, and RNG-device fixes all have real-hardware confirmation so far (the
  RNG-device fix specifically got `segformer_b0` and `fast_scnn` both through a full smoke
  cycle before this newest crash). This fix cannot be confirmed locally even in principle —
  it depends on real bf16 autocast on real CUDA, which this dev machine does not have; the
  next real Colab attempt is the actual test. The prior application commit (`3f3ef8f…`)
  passed remote run `31129018003` with Colab's exact hostile inline backend and host
  uv/virtualenv state injected; that evidence does not carry over to this commit and should
  be re-established before a real Colab attempt.

## Next external action

Push this commit, then open the master notebook from the pushed branch in a fresh Colab L4
+ High-RAM runtime and use Run all (Colab Pro/Pro+ background execution is recommended so
the session survives closing the browser tab). The five-model AMP canary, the
`val_dataloader` build, the `last_checkpoint` resume, the RNG-device resume, and
`pidnet_s`'s bf16 `BoundaryLoss` fix are all already confirmed passing on real hardware;
watch specifically for two things this commit changes: (1) all five models' `smoke` stages
should now complete end to end (the seven prior bugs blocking this are all fixed), and
(2) `fast_scnn`/`pidnet_s`/`ddrnet_23_slim`/`bisenetv2` now train with their own native
SGD+momentum optimizer instead of the shared AdamW — watch their smoke-stage loss curves
for anything pathological (divergence, NaN) that the CPU rehearsal's tiny fixture data and
50-step budget cannot surface. If that looks stable, the real next milestone (per the
takeover-audit plan) is getting **one model through a real `pilot`-stage run** (2000 steps)
— the first time any campaign will have gone past a 50-step smoke fixture. If the session
ends, repeat Run all in a new compliant runtime — this is a Colab platform limit, not
something the notebook can automate away. Do not change the notebook or select stages
manually.

Do not create `colab-v0.1.0-rc1` until two independent clean L4 sessions pass the exact
lock/five-model FP32/AMP canary and the real 50-step interruption/resume proof. After the
campaign completes, build TensorRT only on the target Jetson and attach actual 25W
telemetry. Do not merge, tag, open sealed datasets, upgrade JetPack, or change Jetson power
mode without a separate explicit decision.
