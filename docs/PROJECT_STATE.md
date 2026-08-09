# Project State

Updated 2026-08-09 on `stabilize/colab-v2`.

## Current delivery

The Colab v3 application commit is
`a50b635` (see `git log` for the full SHA). The only generated notebook is
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
never will.

The next real L4 run at application commit `b22fd12…` confirmed that fix empirically: all
five models passed the AMP stack-probe outright (`fp16_finite_model_count: 5`, GPU
`NVIDIA L4`), and full Cityscapes+IDD20K data staging also completed. The run then entered
`production-pipeline` and failed building `val_dataloader` for the first stage
(`smoke`/`segformer_b0`) with `TypeError: Pad.__init__() got an unexpected keyword argument
'seg_pad_val'`. `_evaluation_pipeline()` in `mmseg_runtime.py` passed `pad_val` and
`seg_pad_val` as two separate keyword arguments to the `Pad` transform; the pinned
mmcv-lite `Pad` (`mmcv/transforms/processing.py`) has no `seg_pad_val` parameter at all —
it accepts only `pad_val`, either a plain number or a `dict(img=..., seg=...)`. Application
commit `ff26422…` fixes this by passing `pad_val={"img": 0, "seg": config.ignore_index}` as
a single argument; `_inference_pipeline()`'s bare `pad_val=0` was made the equivalent
explicit `{"img": 0}` for consistency (mmcv's `Pad` already treated a bare int as
image-only padding, so behavior is unchanged there). This fix was reproduced and confirmed
against the real pinned MMSeg checkout by building both the evaluation and inference
pipelines through `Compose()` with the mmseg registry scope active — the same mechanism
real training uses.

The next real L4 run at application commit `ff26422…` confirmed that fix too: all five
models passed the canary again, data staging reused the frozen candidates, and
`smoke`/`segformer_b0` training actually started, ran to the intentional interruption at
optimizer step 25 (by design — the recovery hook saves a checkpoint and raises on purpose
to prove interruption/resume), and the interrupted process exited cleanly. The resume
subprocess then failed with `FileNotFoundError: recovery_25.pth can not be found.`.
Root cause: `EdgeGuardRecoveryHook.after_train_iter` (`mmseg_components.py`) wrote only the
bare checkpoint filename into `<work_dir>/last_checkpoint`. This codebase's own reader
(`latest_checkpoint()` in `colab_recovery.py`) resolves a relative marker against
`work_dir` defensively, but MMEngine's own built-in auto-resume
(`Runner.load_or_resume()` → `find_latest_checkpoint()`) does not — it returns the raw
marker content verbatim and resolves it relative to the process's current working
directory, which is the project root for every child process `colab_pipeline.py` spawns,
not the run's `work_dir`. The identical bare-filename bug also existed in `train_model`'s
Drive cross-session recovery path (`mmseg_runtime.py`), reachable whenever a new Colab
session restores a checkpoint published by a dead prior session and resumes training on
it — the exact scenario this recovery system exists to survive. Application commit
`e3f3159…` fixes both write sites to write the absolute path instead, matching MMEngine's
own `CheckpointHook` convention; a new regression test reproduces the exact failure via
MMEngine's real `find_latest_checkpoint()` against the marker the hook writes and confirms
it fails on the old code and passes on the fix.

The next real L4 run at application commit `e3f3159…` confirmed that fix too: the resume
subprocess found and loaded `recovery_25.pth` and continued training. At this point every
bug found so far had been discovered one at a time on a real Colab GPU, each costing a full
Colab round-trip — three real training-pipeline bugs (never a Drive/staging bug) that a
CPU-only local run could have caught, because nothing in this repository had ever driven
the real orchestrator end to end. The existing "claim-safe local cell execution" check
(`scripts/dev/run_campaign_notebook_harness.py`) only proves the generated notebook's cells
import and execute correctly; the actual training call is stubbed behind a hardcoded
`{"scientific_status": "not_run"}` dict under `EDGEGUARD_NOTEBOOK_LOCAL_TEST=1` and never
touches `Runner.train()`, `EdgeGuardRecoveryHook`, or `val_dataloader`.

Application commit `d7a4430…` builds a real local CPU rehearsal harness
(`tests/support/tiny_pipeline_fixture.py`,
`tests/integration/test_colab_pipeline_cpu_rehearsal.py`) that drives the real
subprocess-spawning `ColabPipeline` — not a mock — through a real smoke-stage
interrupt-then-resume cycle for all 5 models on CPU with tiny synthetic fixture data,
asserting on the real `interruption_resume` records in `metrics.json`. Building it
immediately surfaced a **fourth** real bug, because it is the first thing in this
repository to ever exercise a real end-of-stage validation pass:
`_evaluation_pipeline()`/`_inference_pipeline()`'s `Pad` step passed `config.crop_size`
(this codebase's own `(h, w)` convention, used unchanged everywhere else, e.g.
`RandomCrop`) directly as `Pad`'s `size` argument. The pinned mmcv-lite `Pad` documents
`size` as `(w, h)` and internally reverses it before calling `mmcv.impad(shape=...)`, which
itself expects `(h, w)` — so the pad target was silently transposed. Invisible for a square
crop or when the swap happens to survive; real Cityscapes' non-square 512×1024 crop would
not have survived it, but no real Colab run had ever reached validation to find out (every
prior run crashed or was interrupted before completing a smoke stage). Fixed by reversing
`crop_size` the same way the `Resize` step right above it already does. Application commit
`c4008d9…` adds the rehearsal harness itself and wires it into
`semantic-framework-cpu-probe.yml` as a mandatory CI step, confirmed locally to pass end to
end for all 3 core models — including a real computed mIoU at the final validation
step — closing the exact gap that let all four bugs above reach a real Colab GPU before
being caught. The notebook is repinned to commit `c4008d9…`; the hostile-context remote
Linux workflow and a real L4 run exercising the fixed validation step have not yet been
re-run against it.

A real L4 run at `c4008d9…` confirmed the five-model canary and data staging both pass
cleanly, then crashed on the very first `smoke`/`segformer_b0` attempt — no training step
ever ran — with `ValueError("Drive recovery checkpoint belongs to a different immutable
run")`. This is a **fifth** real bug: on a fresh Colab session with no local
`run_identity.json`, `ColabPipeline._run_training_phase` decides to append `--resume` purely
because a Drive recovery pointer *file* exists for the artifact_id
(`_recovery_pointer_exists` has no identity awareness at all); `train_model()`'s `identity`
dict includes `project_commit` among its ~20 fields, so every commit — even one unrelated to
the model/stage in question — invalidates every previously-published Drive checkpoint's
`identity_sha256` campaign-wide; `train_model()` then correctly detected the resulting
mismatch but incorrectly treated it as fatal, crashing the whole 5-model campaign rather
than simply abandoning an opportunistic resume that turned out to be stale. Fixed by
distinguishing this *speculative* auto-resume (no local run ever existed) from an *explicit*
resume of a known local run: the speculative path now degrades gracefully to a fresh run
(recording a `stale_recovery_skipped.json` evidence file) instead of raising; the explicit
local-resume identity check is untouched, since that one is a genuine safety property.
Application commit `3262af8…` adds this fix, a new `peek_recovery_metadata()` helper in
`colab_recovery.py`, and a CPU rehearsal regression test that reproduces the exact crash
(confirmed to fail on the pre-fix code with the identical error message, and pass on the
fix). **Left open, not implemented:** whether `project_commit` should remain part of the
strict identity-compare value at all, or be recorded as provenance-only metadata so only
scientifically-relevant fields (protocol/dataset/hyperparameter hashes) invalidate a Drive
checkpoint — a reproducibility-policy call reserved for the human project owner. Not yet
confirmed on real L4 hardware.

A real L4 run at `3262af8…` got further than any prior run: canary, data staging, and a
fresh `smoke`/`segformer_b0` start plus the intentional interruption at optimizer step 25
all confirmed working, then the local resume subprocess crashed with `TypeError: RNG state
must be a torch.ByteTensor in EdgeGuardRecoveryHook`. This is a **sixth** real bug:
`Runner.resume()` loads checkpoints with `map_location=get_device()`, which on Colab is
CUDA, so every tensor in the pickle — including the RNG state
`EdgeGuardRecoveryHook.before_save_checkpoint` stores — comes back on CUDA, and
`torch.set_rng_state`/`torch.cuda.set_rng_state` both reject anything but a CPU uint8
tensor. mmengine 0.10.7 has no RNG save/restore of its own, so this hook is genuine
capability, not deletable duplication; both the save and load paths now coerce every RNG
tensor to CPU/uint8, so checkpoints already published to Drive under the buggy format keep
resuming. Reproduced and regression-tested with no GPU at all:
`mmengine.device.get_device()` returns `"mps"` on this dev machine, itself a real foreign
device relative to a CPU RNG tensor, so `.to("mps")` reproduces the identical `TypeError`
without CUDA — confirmed failing pre-fix, passing post-fix. This also surfaced why the CPU
rehearsal harness missed it: a local, never-committed `sitecustomize.py` had been forcing
`mmengine.device.utils.DEVICE = "cpu"` to route around unrelated MPS operator gaps, which
incidentally hid every device-class bug too; the new RNG test avoids this entirely by
checking `torch.backends.mps.is_available()` directly rather than depending on mmengine's
device detection.

Separately, application commit `c88ac8f…` contains a deliberate architecture change: the
intentional-interrupt self-test (proves interrupt+resume on real hardware) used to run once
per model — 5 deliberate crash+resume cycles per campaign, no way to disable it — and three
of this session's six bugs (bare-filename marker, stale Drive recovery, this RNG bug) all
surfaced through it, each one killing the whole 5-model campaign when it hit. It now runs
once per campaign (`PipelineInputs.recovery_self_test_model`, default the first core
model); a failed resume leg is recorded as durable, hash-sealed evidence
(`recovery_self_test_failure.json`) and the model restarts from scratch instead of aborting
the campaign, while `pilot`/`screening`/`hpo`/`final` refuse to start with any unresolved
failure record present — cheap phases can absorb a self-test failure, but the phases that
cost real hours cannot proceed on an unproven recovery path. Also added a CPU-visible
config-shape test for the `AmpOptimWrapper` branch that is unreachable in every CPU
rehearsal run (`resolve_auto_precision` always returns `fp32` without CUDA) — exactly how
the session's very first bug (a hardcoded AMP dtype) escaped local testing. Not yet
confirmed on real L4 hardware.

A real L4 run at `2b078d3…` reached the furthest point yet: `segformer_b0` completed its
full smoke cycle end to end (fresh start, interruption at step 25, real resume via the
RNG-device fix, training to step 50, validation, mIoU computed), and `fast_scnn` trained
straight through under the now-once-per-campaign self-test. Then `pidnet_s` crashed on its
very first training step with `RuntimeError: Index put requires the source and destination
dtypes match, got BFloat16 for the destination and Float for the source` inside
`boundary_loss.py:52`. This is a **seventh** real bug: upstream `BoundaryLoss.forward`
builds `weight = torch.zeros_like(log_p)`, and under real `AmpOptimWrapper(dtype='bfloat16')`
autocast on real CUDA, `log_p` (the boundary head's raw logits) is already bfloat16; the
ratio assigned into `weight` comes from summing a float32 label mask autocast never
touches, so it stays float32 — `index_put_` rejects the mismatch on the pinned Colab torch
(2.1.1+cu121). Fixed by registering `EdgeGuardBoundaryLoss` (application commit `4917c49…`)
via this repo's existing `force=True` override idiom — identical to upstream except the two
assignments are cast to `weight.dtype` first. Only `pidnet_s` uses `BoundaryLoss` among the
five models; a scan of the pinned checkout's other loss files for the same pattern found
only one other occurrence, used by none of our five model configs. This fix cannot be
reproduced as a "raises pre-fix" test on this dev machine even in principle: its torch
(2.13.0) silently permits the same implicit downcast that torch 2.1.1 (pinned for
Colab/CI) rejects — a torch-version difference, not a device one. The new test instead
asserts the fix's actual guarantee (dtype-aligned `weight`, finite loss under mismatched
inputs, and bit-identical output to upstream when dtypes already match), and `git stash`
confirms the override registration itself is present only post-fix. The fast-tier CPU
rehearsal (a real `pidnet_s` smoke run, fp32, unaffected by this bf16-only bug) was re-run
and still passes. Not yet confirmed on real L4 hardware — that is the only test that can
actually exercise the bf16 code path this fixes.

A full technical-takeover audit (application commit `a50b635…`) was performed at the human
owner's explicit request: read the repository with no loyalty to prior architectural
decisions and issue an architecture verdict. Three parallel read-only audits (repo hygiene/
CI/ADRs; model registry/reliability/HPO/ONNX/TensorRT/Jetson; data ontology/notebook/Drive
architecture), plus direct reading of all 5 pinned upstream MMSeg configs, produced
**VERDICT B — sound concept, real but bounded defects**. This is not the "repeatedly failed"
prior the audit's own framing assumed: ADR-0008 (2026-07-28) already demoted the pre-existing
detection/temporal campaign to non-blocking legacy and pivoted to the semantic-first rescue
architecture; ADR-0009 (2026-07-28, amended 2026-07-30) already built the explicit
multi-domain dataset ontology and role system this kind of audit would normally have to
demand from scratch. The seven bugs fixed earlier this session were real but narrowly-scoped
orchestration/precision defects, each fixed in isolation with a regression test — evidence of
a sound design surviving real-world contact, not architectural rot. The audit found no
component meeting the bar for REBUILD or REPLACE; the dataset ontology, reliability/OOD
stack, dependency tri-tier separation, single generated notebook, ONNX export, and data
inventory tooling (`scripts/audit_dataset.py`) were all confirmed real, tested, and already
matching what the audit's own mandate asked for — left untouched.

The audit found one real, quantified, previously undetected methodology defect:
`build_training_config` in `mmseg_runtime.py` unconditionally overwrote every model's
`optim_wrapper` with one shared `AdamW(lr=6e-5, wd=0.01)`, regardless of architecture.
Reading each of the five models' own pinned upstream MMSeg configs directly (not just
trusting a sub-agent's summary) found that only `segformer_b0` actually matches this recipe;
`fast_scnn` (SGD, lr=0.12), `pidnet_s` (SGD, lr=0.01), `ddrnet_23_slim` (SGD, lr=0.01), and
`bisenetv2` (SGD, lr=0.05) all natively train with SGD+momentum at learning rates 150-2000x
higher than what was actually being applied, in the wrong optimizer family entirely. Optuna's
HPO search space (`hpo_runtime.py`) only ever searched learning-rate/weight-decay *within* a
fixed AdamW assumption, so it could never have self-corrected this — a real, previously
unnoticed risk to the upcoming `pilot`/`screening` model comparison, since it would have made
every non-SegFormer model look artificially weak for reasons unrelated to architecture
quality. Application commit `a50b635…` fixes this with a new
`resolve_model_optimizer_defaults()`, which reads each model's real upstream
`optim_wrapper.optimizer` as the training baseline; explicit overrides (what every HPO trial
already supplies) still apply on top, now preserving the model's own optimizer type and
momentum instead of forcing AdamW. `train_model()`'s identity record now includes
`optimizer_type`. The frozen HPO learning-rate/weight-decay *search range*
(`[2e-5, 3e-4]`, tuned for AdamW-scale) was deliberately left untouched — that is now an open
question for the SGD-native models once real HPO execution begins, flagged rather than
silently resolved, since dataset/HPO scope decisions are the human owner's per this project's
governance. The same commit retargets `semantic-framework-cpu-probe.yml`'s push trigger from
the stale `feat/first-vertical-slice` to `stabilize/colab-v2`/`main` — this CI job runs the
exact CPU rehearsal that caught most of this session's real bugs, and had never run
automatically on the branch where all current work happens — and documents (in
`docs/canonical-colab-runbook.md`) the exact `scripts/audit_dataset.py` invocation for
inspecting staged Cityscapes/IDD20K training data before a full campaign; no inventory
artifacts are claimed locally, since this dev machine has no real dataset by design.

Full audit findings, evidence, and the explicit KEEP/REPAIR/REBUILD classification are
recorded in the approved plan; **left open for the human owner, not decided by this audit:**
whether to eventually trim the 5-model comparison to fewer models (defer until real
`pilot`-stage signal exists — the current smoke-stage numbers are 50-step noise, not signal),
and whether to ever pull BDD100K/ACDC/WildDash into training roles (ADR-0009's existing
answer — Cityscapes+IDD20K as the frozen scientific core, the others correctly scoped
narrower — is recommended as final).

## Deliveries

The package stage produces `EdgeGuard_Jetson_Release.zip`,
`EdgeGuard_Thesis_Bundle.zip`, `EdgeGuard_Streamlit_Demo.zip`, and
`release_index.json`, each hash verified. The Jetson archive contains five ONNX graphs,
checkpoints/configs and golden vectors, but never a TensorRT engine. Jetson telemetry stays
`not_run` until a target-device benchmark is supplied.

## Verification boundary

Local Ruff, mypy, pytest, deterministic notebook generation, claim-safe local cell
execution, and the new real `ColabPipeline` CPU rehearsal (see below) validate engineering
contracts only. No local test creates a scientific metric. "Claim-safe local cell
execution" specifically proves only that the generated notebook's cells import and execute
their own syntax correctly — the real training call is stubbed behind a hardcoded
`{"scientific_status": "not_run"}` dict; it does not exercise `Runner.train()`,
`EdgeGuardRecoveryHook`, or `val_dataloader`. The real orchestrator rehearsal
(`tests/integration/test_colab_pipeline_cpu_rehearsal.py`) does exercise all three, on CPU,
against tiny synthetic fixture data, and is what actually caught the fourth (`Pad`
orientation) bug above before any Colab GPU time was spent on it.
The current delivery passes 499 tests with thirty-two environment-gated skips without the
pinned MMSeg stack present (up from 489/21 — several new tests, including the per-model
native-optimizer tests, are gated on the checkout); with the pinned stack available
(`EDGEGUARD_MMSEG_CHECKOUT` pointed at the exact commit
`c685fe6767c4cadf6b051983ca6208f1b9d1ccb8` checkout) the mmseg-gated test files pass 27 of
27 (up from 24 — adds `test_native_optimizer_matches_each_models_own_upstream_recipe`,
parametrized over all 5 models, and `test_build_training_config_wires_the_native_optimizer_through`),
including the real per-architecture `model.loss()` regression tests, the `last_checkpoint`
marker regression test, the `Pad` orientation regression test, the stale-Drive-recovery
regression test, the RNG checkpoint-device regression test, and the PIDNet `BoundaryLoss`
dtype regression test. **Noted, not fixed, out of scope for this commit:** running the
entire test suite (every file) with `EDGEGUARD_MMSEG_CHECKOUT` set produces 12 failures in
`tests/unit/test_dataset_preparation.py` that do not reproduce when that file runs alone or
alongside only the mmseg-gated files above — a pre-existing test-isolation/ordering issue,
since that file has no connection to `mmseg_runtime.py` or this session's changes. The full
`tests/integration/test_colab_pipeline_cpu_rehearsal.py` suite (all 3 tests) was re-run end
to end and still passes (18m40s). The `BoundaryLoss` test still cannot be a "raises pre-fix"
reproduction even in principle on this machine — its torch (2.13.0) silently permits the
exact implicit downcast that the pinned Colab/CI torch (2.1.1) rejects, a torch-version
difference, not a device one — so it asserts the fix's actual guarantee (dtype-aligned
output, bit-identical to upstream when dtypes already match) instead, and `git stash`
confirms the override registration is present only post-fix. The new optimizer-defaults
regression tests use the same `git stash` technique: stashing only the fix produces a clean
`ImportError` on `resolve_model_optimizer_defaults` at test collection, confirmed to fail
pre-fix and pass post-fix. The master notebook was generated twice byte-identically at
SHA-256 `263e9c1efec73ee18778ac460133c9fabc71e248d475250c3eccc153a72bc43b`, pinned to
application commit `a50b635…`.
Remote Linux workflow `31129018003` completed successfully at an earlier application commit
(`3f3ef8f…`) with the exact Colab failure context injected
(`MPLBACKEND=module://matplotlib_inline.backend_inline`, host uv and virtualenv state); it
has not yet been re-run at the current commit (including the rehearsal CI step), and
claim-safe local cell execution has not been re-verified at this commit either — both remain
pending before the next real Colab attempt. As of application commit `a50b635…`, this
workflow's push trigger now covers `stabilize/colab-v2`/`main` (previously scoped only to
the stale `feat/first-vertical-slice`, where it never ran on real work), so the next push to
this branch should be its first automatic run. The AMP-probe precision fix, the
`Pad`/`seg_pad_val` fix, the `last_checkpoint` absolute-path fix, the stale-Drive-recovery
fix, and the RNG checkpoint-device fix have all since been confirmed by real L4 runs (the
RNG-device fix specifically got `segformer_b0` and `fast_scnn` both through a full smoke
cycle before the next, newly-fixed bug was reached); neither the `BoundaryLoss` dtype fix nor
the per-model native-optimizer fix (`a50b635…`) has yet been confirmed by an actual L4 run.
The `BoundaryLoss` fix cannot be confirmed any other way, since it depends on real bf16
autocast on real CUDA; the optimizer fix *could* in principle be judged by smoke-stage loss
behavior on CPU, but the CPU rehearsal's 50-step budget and tiny synthetic fixture are too
short/small to say anything meaningful about optimizer-family correctness — real L4
`smoke`/`pilot`-stage loss curves for the four newly-SGD models are the actual test.
The notebook is not eligible for a Colab-ready tag until two independent clean L4
five-model FP32/AMP canaries and a real interruption/resume smoke have passed. No training
result, accepted scientific release, TensorRT engine, Jetson measurement, merge, or tag is
claimed by this state file.
