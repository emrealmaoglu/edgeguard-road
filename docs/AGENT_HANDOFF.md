# Agent Handoff

- **Branch:** `stabilize/colab-v2`
- **Application commit pinned by notebook:**
  `a5519f4` (see `git log` for the full SHA)
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
- **Note on the eighth real Colab bug (commit `2b1ebff…`):** a real L4 run at `a50b635…`
  got all five models through a full `screening` stage (6000 steps each) for the first
  time — real mIoU: segformer_b0 16.48%, fast_scnn 20.53%, pidnet_s 26.54%,
  ddrnet_23_slim 25.21%, bisenetv2 17.41% — then crashed in the screening-evidence
  evaluation step with `NameError: name 'inf' is not defined`. Root cause:
  `build_training_config()`'s `clip_grad.max_norm` was `float("inf")`
  (`error_if_nonfinite=True` is the real safety net; `max_norm` was never meant to
  actually clip). `train_model()` dumps the resolved config to `resolved.py` via
  `mmengine.Config.dump()` before every training call — including no-op
  resume-and-skip re-entries — and `evaluate.py` (used by the screening/final evidence
  phases) reloads it via `mmengine.Config.fromfile()`, which `eval()`s the dumped Python
  source; mmengine's dumper serializes `float("inf")` as the bare token `inf`, not a
  valid Python literal without `float(...)`/`math.inf` in scope. This is structurally
  unreachable by any test exercising only the in-memory `cfg` — every prior stage did
  exactly that — so it survived undetected through 7 prior bug fixes and ~20 hours of
  real L4 compute until the pipeline finally reached a real dump-then-reload path. Fixed
  by replacing `float("inf")` with a large finite sentinel (`1e9`), which round-trips
  cleanly and stays effectively unbounded for any real gradient norm (this session's
  real runs, including a visibly diverging `bisenetv2` screening run, topped out around
  `grad_norm ~560`). Reproduced with zero GPU dependency (pure Python dump/reload) and
  regression-tested; confirmed via `git stash` to fail pre-fix with the identical error
  and pass post-fix. **Noted, not fixed, confirmed pre-existing and unrelated via the
  same `git stash` check:** the full `tests/integration/test_colab_pipeline_cpu_rehearsal.py`
  suite currently fails on this Mac with `RuntimeError: view size is not compatible with
  input tensor's size and stride` during `backward()` — the known MPS operator-gap class
  of issue already documented in `canonical-colab-runbook.md`, identical on both
  pre-fix and post-fix code. Needs separate investigation, does not block this fix.
- **Note on the private_inputs raw-archive inventory (commit `7b604c9…`):** the user asked
  for a full automatic inventory of every raw archive under Drive's
  `EdgeGuard/private_inputs/` (sizes, image counts, resolutions, formats, and real
  per-class pixel/image frequencies where masks exist), runnable inside the master
  notebook and downloadable at the start of every session — for the thesis report and to
  surface future optimization targets, including a possible class-imbalance-driven
  ontology reduction. The user explicitly rejected a bounded-sample design ("basite
  kaçma") and required every file to get the same depth of inspection regardless of name,
  with real *measured* per-class histograms rather than declared ontology counts, since
  this data may inform an actual future class-reduction decision. Implemented as
  `src/edgeguard/rescue/archive_inventory.py`: dispatch is purely on file format
  (zip/tar/standalone) and entry extension/PIL mode, never on dataset name; every image
  entry is fully decoded (not header-sampled); every label-like entry (mode `L`/`P`/`I`/
  `1`) gets a real `np.unique`-based per-class pixel/image histogram; declared metadata
  from `colab_data_access_v1.yaml` is attached afterward as annotation only, matched by
  exact filename, and an undeclared file is reported as having no known role rather than
  guessed. Deliberately **not** a new `ColabPipeline` phase and **not** gated behind the
  accepted-release-gated `package` phase (`build_colab_release_packages` requires a fully
  accepted 5-model release) — that gate would make the report unavailable for weeks given
  how often real sessions die mid-campaign. Instead it is a new, independent, early
  notebook cell (`scripts/inventory_private_inputs.py`, `build_colab_notebooks.py`),
  wrapped in try/except-and-continue so it can never block or fail the real campaign, run
  immediately after checkout and before the main campaign subprocess. The report
  directory is content-addressed from `private_inputs/`'s file listing
  (names+sizes+mtimes), so an unchanged folder reuses the prior session's report instead
  of re-scanning every file again. `record_type: "raw_archive_inventory"` carries no
  `scientific_status` field (engineering/audit artifact, not a training result); measured
  histograms are labeled `_measured` and kept separate from `known_role_declared`
  throughout. Also extracted `src/edgeguard/rescue/stall_guard.py` (Drive/FUSE
  read-stall guard, previously inlined in `colab_data.py`) and
  `write_verified_zip()` in `serialization.py` (previously `colab_release.py`'s private
  `_zip_members()`) as shared infrastructure used by both the new tool and the existing
  code. Purely additive: does not touch `ColabPipeline`/`PHASES`, does not change any
  dataset role/scope decision, does not affect the in-progress real-L4 campaign state
  above.
- **Note on the private_inputs inventory cell's real-Colab bootstrap bug (commits
  `1a859aa…`, `7ffef5c…`):** the user ran the newly-pushed `7b604c9` on real L4 and the
  new inventory cell immediately crashed with `ModuleNotFoundError: No module named
  'edgeguard'` (caught by the cell's own try/except, so it did not block the campaign,
  exactly as designed — but the report was never produced). Root cause: the cell invoked
  `scripts/inventory_private_inputs.py` via bare `/usr/bin/python3`, the same pattern
  used for `run_colab_master.py` — but `run_colab_master.py` deliberately never imports
  `edgeguard` at its own top level (it only sets `PYTHONPATH` for the children it spawns
  later, inside its own `_runtime_environment()`), while the new script imports
  `edgeguard.rescue.archive_inventory` immediately, and nothing had ever put `src/` on
  that bare interpreter's `sys.path`. Fixed by giving the inventory subprocess its own
  scoped environment dict (`inventory_environment`, deliberately not the shared
  `environment` dict used for the main `run_colab_master.py` call) with `PYTHONPATH` set
  to `src/`, and by `pip install`-ing the small set of pure-Python packages
  (`numpy`/`Pillow`/`pydantic`/`PyYAML`, at `pyproject.toml`'s pinned ranges) the
  import chain needs, since this step deliberately runs before the locked training
  runtime is provisioned and cannot assume Colab's bare system Python already has them —
  a materially lower-risk category of host-touching than the CUDA/torch locked-runtime
  isolation this project hardened earlier (small pure-Python packages into the
  notebook's own already-mutable per-session system Python, not the version-pinned ML
  stack). Verified directly: built a throwaway venv with only the four light packages
  installed and no `edgeguard-road` install at all, ran the CLI against it with only
  `PYTHONPATH` set, confirmed a correct report. `tests/integration/test_notebook.py`'s
  existing `'environment["PYTHONPATH"]' not in source` guard (added 2026-08-06 to keep
  the *shared* `environment` dict feeding `run_colab_master.py` PYTHONPATH-free) used a
  bare substring check that also matched `inventory_environment["PYTHONPATH"]` —
  tightened to a word-boundary regex so it still guards what it always meant to guard.
  Also fixed a stale hardcoded `code_cell_count == 4` in
  `tests/unit/test_delivery_notebooks.py`, missed in the prior commit because the full
  suite was run before that commit's notebook regeneration, not after — a process gap
  for this session, not a design defect; the full suite is now confirmed green *after*
  the notebook regeneration for this commit.
- **Note on the ninth real Colab-path bug — `cityscapes bundle identity mismatch`
  (commit `24dd782…`):** the private_inputs inventory cell's bootstrap fix worked on the
  next real L4 run (application commit `7ffef5c…`) — `EdgeGuard_Data_Inventory.zip`
  downloaded successfully — but the main campaign then crashed in the `data` stage with
  `ValueError: cityscapes bundle identity mismatch` from `_canonical_bundle_receipt`
  (`colab_data.py:902-905`). Root cause: that check compared the receipt's
  `plan_sha256` — a SHA-256 of the *entire* `colab_data_access_v1.yaml` at
  bundle-creation time — against a fresh hash of the *entire current file*. Any edit
  anywhere in that YAML, even to a completely unrelated dataset, invalidates every other
  dataset's already-built, still-correct Drive bundle. That's exactly what happened:
  commit `bd3ea56` (WildDash2/RailSem19 role assignment) edited only the `wilddash2`
  section — `cityscapes`'s own config entry was untouched — but the whole-file hash
  changed anyway, so the already-published ~8.26 GB `cityscapes` bundle on Drive got
  rejected on this run. `create_dataset_bundle`'s own bundle-reuse path never checked
  `plan_sha256` at all (confirmed by reading `colab_data.py:646-654` directly), showing
  this field was never meant to be load-bearing across the whole file — an accidental
  side effect of hashing too much, not a deliberate integrity design; there is also no
  test coverage of `plan_sha256` anywhere in `tests/unit/test_colab_data.py`. Fixed by
  comparing the receipt's `required_paths` field directly (already stored verbatim,
  unchanged, in every previously-published receipt) instead of the whole-plan hash —
  backward-compatible with the bundle already on Drive, so no expensive rebuild is
  needed, and still fails closed if a dataset's own `required_paths` genuinely changes.
  Also scoped the now-informational (no longer gating) `plan_sha256` field written on
  bundle creation to just that dataset's own plan subsection, so it can't set the same
  trap again if something reads it later. Two new regression tests reproduce the exact
  real-world scenario (unrelated dataset edited, staging must still succeed) and confirm
  the check still rejects a genuine change to the dataset's own `required_paths`. This
  was diagnosed with an Explore agent plus direct reads of `colab_data.py`, then
  implemented under `plan mode` (system-enforced) with explicit user approval via
  `ExitPlanMode` before any code changed.
- **Note on live progress output for the private_inputs inventory (commit `f2f2110…`):**
  the user asked for the inventory cell to show, while it runs, which file it's
  currently scanning, which have finished, and an ETA — they had waited on a real Colab
  run with zero output and couldn't tell whether it was working or stuck.
  `build_inventory_report()` (`archive_inventory.py`) now prints, per file, `"[i/N]
  taranıyor: name (size)"` before and `"[i/N] tamamlandı: name — type, image count,
  corrupt count, elapsed, genel tahmini kalan süre"` after; `_inspect_zip()`/
  `_inspect_tar()` print an intra-archive `"... name: completed/total giriş tarandı
  (%X) — tahmini kalan: Y"` line at the first entry, the last entry, and at least every
  2 seconds while scanning, so one very large archive (e.g. IDD20K's 32127-entry
  shards) doesn't look stuck either. The overall ETA is a bytes-remaining /
  bytes-per-second-observed-so-far estimate — an honest proxy for total work, since
  archive sizes in `private_inputs/` vary by orders of magnitude. All prints use
  `flush=True` so they stream through `run_visible()`'s live subprocess output exactly
  as produced, matching the periodic-progress-line pattern already used elsewhere in
  this project (`create_dataset_bundle`'s per-1000-files prints). The CLI's cache-hit
  path also now prints why nothing is happening instead of staying silent. Progress
  lines share stdout with the final `canonical_json` result line (same pattern
  `run_colab_master.py`'s own stages already use), so `test_archive_inventory.py`'s CLI
  test helper was updated to parse only the last stdout line; added a new regression
  test locking in the progress-output shape via `capsys`. Purely an observability
  change — no scan logic, statistics, or report schema changed.
- **Note on the 2026-08-12 scientific-decision delegation and real-evidence training-log
  analysis (commits `122046f…`, `38df2ae…`):** with a hard deadline (presentation video
  in 4 days, report in 8, end of a 6-week development window), the owner explicitly
  lifted the "scientific conclusions" and "HPO scope and thresholds" boundaries in
  `CLAUDE.md` for the rest of this campaign — see that file's 2026-08-12 entry for the
  exact scope and the two boundaries that still do not move (sealed test data, the
  non-fabrication contract). The user separately asked to aggressively narrow model/class
  scope and see "başarım... biraz yüksek" results; that specific framing was declined —
  no result is inflated or cherry-picked — but a real, principled path to a favorable,
  *defensible* headline number does exist: several classes show 0.0 IoU across every
  model measured so far (train, motorcycle, bicycle, wall, fence, pole, traffic light,
  traffic sign, person, rider, truck, bus), and if real training-data frequency confirms
  they're statistically near-absent, excluding them from a *separately labeled* metric is
  legitimate methodology, not fabrication. Built `src/edgeguard/rescue/training_log_analysis.py`
  + `scripts/analyze_training_results.py` to do this with real evidence only: an Explore
  agent confirmed the training loop itself (`mmseg_runtime.py::train_model`) never
  persists per-class IoU (only the late, sealed-test-gated `evaluate_model` path does,
  via `evaluation.json`) — so the tool parses mmengine's real "per class results" table
  straight out of saved Colab log text (the user's own pasted session output is a valid,
  real source), cross-references it with the real per-class pixel frequency this project
  already computes (`write_train_fit_statistics`'s `class_weights.json` for Cityscapes,
  `audit_training_dataset`'s `summary.json` for IDD20K), and projects a "supported
  classes only" mIoU using a single fixed, never-per-model-tuned pixel-ratio threshold —
  always reported side by side with, never in place of, the real 19-class mIoU.
  Deliberately not a second `.ipynb` (would violate
  `test_master_notebook_is_the_only_active_notebook_and_is_output_free`); ships as a
  script usable locally or pasted into an untracked, ad-hoc Colab cell. Tests use the
  user's own real pasted log excerpt as the primary fixture, not synthetic data.
  **Model-scope decision made under the new delegation** (real measured numbers, this
  session): remaining HPO/final compute is prioritized for `segformer_b0` (highest
  measured mIoU, ~18x faster per-iteration than the slowest model) and `pidnet_s`
  (second-best measured mIoU, the only model with any real per-class signal outside the
  classes every other model also scored zero on); `fast_scnn`/`bisenetv2`'s measured
  per-iteration cost would make a full `final` run (40000 steps) infeasible within the
  remaining timeline, and `ddrnet_23_slim` has no real screening result yet and a
  low measured smoke-stage mIoU.
- **2026-08-13 model-scope decision superseding the above, plus a `final`-stage
  restructure (commit `5135f69…`):** the user asked live, mid-real-Colab-session,
  whether to interrupt a running `bisenetv2` screening (~6.5–7s/iter, would take
  ~10 hours to finish). Confirmed via two Explore agents that HPO's
  `select_hpo_models()` (`hpo_runtime.py`) needs at least 2 valid candidates, not all
  5 — so interrupting `bisenetv2` (it already had checkpoints at iter 500/1000/1500)
  does not block the pipeline; directed the user to stop it. By then all five models
  had real screening throughput/mIoU on record for the first time this session (up
  from the pilot/partial numbers the 2026-08-12 decision above was based on):

  | Model | Screening mIoU (6000 iter, measured) | ~sec/iter (measured) |
  | --- | --- | --- |
  | pidnet_s | 27.19 | ~1.0 |
  | ddrnet_23_slim | 24.50 | ~0.87 |
  | fast_scnn | 20.10 (elapsed ~31046s ≈ 8.6h for screening alone) | ~5.3 |
  | segformer_b0 | 16.50 | ~0.29 |
  | bisenetv2 | not completed (smoke-only 6.57 from an earlier stage) | ~6.5–7 |

  This reverses the 2026-08-12 pick of `segformer_b0`+`pidnet_s`: with full screening
  evidence, `pidnet_s` and `ddrnet_23_slim` are the real mIoU leaders, not
  `segformer_b0` (still fastest, but now lowest mIoU of the four completed runs).
  Separately, and more consequentially: `colab_pipeline.py`'s `final` phase was found
  to hard-require **all five** models regardless of HPO's top-2 pick
  (`PipelineInputs.validated()` rejected any `final_models != ALL_MODELS`, a
  deliberate, documented design per `docs/SEMANTIC_FIRST_RUNBOOK.md`/`DECISIONS.md`,
  not unexamined default cruft) — at real measured throughput, 40,000 final steps on
  all five would cost ≈155 GPU-hours, incompatible with a 4-day deadline (fast_scnn
  ≈59h, bisenetv2 ≈72h alone). Under the same 2026-08-12 delegation, restricted
  `final_models` to `segformer_b0`, `pidnet_s`, `ddrnet_23_slim` (≈24 GPU-hours) —
  chosen because their measured per-iteration cost keeps a full 40000-step final run
  feasible; `segformer_b0` is kept despite its lower mIoU because it is cheap (~3.2h)
  and broadens the final comparison at negligible cost. Implemented by relaxing
  `validated()`'s exact-five check to "non-empty, duplicate-free, frozen-order subset
  of `ALL_MODELS`" and switching four call sites (`_accepted_release`,
  `_write_run_contracts`, `_selection_evidence`, the `select_recommended_model` call)
  from a hardcoded `ALL_MODELS` to `self.inputs.final_models`; `_write_release_candidate`
  already used `self.inputs.final_models` and needed no change. Added a repeatable
  `--final-model` flag to `scripts/colab_pipeline.py`; `scripts/run_colab_master.py`
  (the one-button notebook runner) now reads `final_models` straight out of the
  committed owner-authorization policy JSON
  (`configs/campaign/semantic_cs_idd_v3_authorization.json`, which was narrowed to the
  three models with an inline `final_models_scope_decision` provenance note) instead
  of hardcoding it a second time, keeping the policy file the single source of truth.
  `fast_scnn` and `bisenetv2` keep their real screening evidence in the report; they
  are excluded from further compute, never silently dropped. **Not yet confirmed on
  real L4 hardware** — the next real Colab run (after the user stops `bisenetv2` and
  resumes) is the actual end-to-end test of this change.
- **2026-08-13 follow-up (commit `6b30275…`): the user had to force-stop the entire
  Colab runtime**, not just interrupt the `bisenetv2` cell — there is currently no
  running Colab session. Also discovered live: the private_inputs archive-inventory
  cell (2026-08-12 feature) appears to "not run" on a fresh session because it is
  content-addressed-cached (`inventory_identity(private_inputs_root)` — see
  `scripts/inventory_private_inputs.py:54-69`) and `private_inputs/` hasn't changed
  since the last real scan, so it correctly and intentionally reuses the prior report
  and skips rescanning (prints a Turkish "değişmemiş, tarama atlandı" line and the
  cached report) rather than being broken. Realized that a fresh Colab session's
  automatic `Runtime → Run all` would, with only the `5135f69…` fix, still try to
  **resume and finish `bisenetv2`'s screening run** (from its last checkpoint at
  iter ~1500/6000, ~8 more hours at ~6.5-7s/iter) before advancing to HPO — wasted
  time, since `bisenetv2` is already excluded from `final_models` regardless of its
  screening outcome. Added a symmetric `screening_models` field to `PipelineInputs`
  (same non-empty/duplicate-free/frozen-order validation as `final_models`), wired
  through `_run_training_phase`, `_write_run_contracts`, and `_screening_evidence`;
  added a matching `--screening-model` CLI flag; `run_colab_master.py` now also reads
  an optional `screening_models` field from the policy JSON (absent-safe — omitting
  it keeps all-five behavior, so this is backward compatible with any campaign that
  doesn't set it). The policy JSON's `screening_models` is now `segformer_b0`,
  `fast_scnn`, `pidnet_s`, `ddrnet_23_slim` (the four that actually reached the real
  6000-step ceiling); `bisenetv2` keeps its real partial evidence (smoke-stage mIoU
  6.57, interrupted screening checkpoint) in the record and is simply not resumed.
  Application commit `6b30275`; Ruff/format/mypy passed (119 modules, unchanged);
  full local suite 539 passed/32 skipped (up from 536/32 — 3 new
  `test_colab_pipeline.py` cases covering `screening_models`). Notebook regenerated
  twice byte-identically at SHA-256
  `e277eb9ba5653bf407b0ab7e09985c384473d9128a5a168c576078a4176a5791`. **Full restart
  plan for the next real Colab session is in `docs/AI_USAGE_LOG.md`'s 2026-08-13
  entry for this commit.**
- **2026-08-13 second follow-up (commit `90b6bea…`): the `6b30275` fix above was
  itself invalidated by the exact problem it was trying to prevent.** When the user
  actually resumed on the new commit, the real, previously-completed 6000-step
  screening runs for `segformer_b0`/`fast_scnn`/`pidnet_s`/`ddrnet_23_slim` (from the
  `f2f2110` session) were about to be retrained from iteration 0 — `run_colab_master.py`'s
  restore step skips the Drive campaign-state tarball on any `project_commit` mismatch,
  and separately `mmseg_runtime.py`'s per-run `identity` dict bakes in `project_commit`
  directly, so `EdgeGuardRecoveryHook`'s resume check correctly (if expensively) treated
  every real Drive checkpoint as belonging to "a different immutable run" once the commit
  changed — even though `6b30275`'s only change was orchestration-level
  (`screening_models`/`final_models` CLI flags), nothing about how any individual model is
  actually trained. Confirmed by inspection: every other field in the `identity` dict
  (protocol hash, dataset manifest hashes, optimizer config, step counts) is unaffected by
  that change. Extracted the inline identity-dict construction out of `train_model` into a
  standalone, GPU-free `compute_run_identity()` (pure refactor — `train_model` now just
  calls it, one place this logic lives). Added `scripts/migrate_recovery_identity.py`: for
  each of the four models, recomputes the identity under both the old and new commit,
  verifies the recomputed old-commit identity matches what was actually recorded on Drive
  at publish time (proof nothing besides `project_commit` changed) and that
  `project_commit` is the only differing field between old/new, and only then republishes
  the *same* checkpoint bytes under the new commit's identity via the existing
  `publish_recovery_file`/`restore_recovery_file` API (fail-closed — refuses outright,
  never retrains or fabricates, on any mismatch). Application commit `90b6bea`;
  Ruff/format/mypy passed (119 modules, unchanged); full local suite 542 passed/32 skipped
  (up from 539/32 — 3 new `test_compute_run_identity.py` cases). Notebook regenerated
  twice byte-identically at SHA-256
  `24aed1497ad4759663b4498b0e940ac38458a513c56115fd35af05492c6e6ffe`. **Not yet run for
  real on Colab** — the user still needs to invoke the migration script (exact command in
  `docs/AI_USAGE_LOG.md`'s entry for this commit) before resuming the production pipeline.
- **2026-08-13 third follow-up (commit `a5519f4…`): made the migration fully automatic,
  no manual command.** The user's Colab runtime disconnected before they could act on
  the manual instructions above — there is no reliable way to hand-time "watch for
  screening, interrupt, run this command" across an unpredictable disconnect/reconnect
  cycle. Made `migrate_recovery_identity()` (now in `mmseg_runtime.py`, next to
  `compute_run_identity`) self-contained: given just `recovery_root` and the current
  `project_commit`, it reads the OLD `project_commit` directly off the existing Drive
  receipt (new `peek_recovery_receipt` in `colab_recovery.py`) instead of requiring the
  caller to know or pass it. `scripts/run_colab_master.py` — which only imports the
  standard library (`test_host_entrypoints_import_only_the_standard_library` enforces
  this) — now runs `scripts/migrate_recovery_identity.py` as its own subprocess stage
  (`"recovery-identity-migration"`), right after data staging (once the frozen
  manifests exist) and before the production pipeline starts, for every model in
  `screening_models`. Unattended, on every session, and safe by construction: it only
  ever republishes a checkpoint whose training-equivalence it just verified, and does
  nothing (no error, no retrain trigger) if there's nothing to migrate.
  `scripts/migrate_recovery_identity.py --model` is now optional (defaults to all five
  models), matching how the master runner calls it when a policy doesn't set
  `screening_models`. Application commit `a5519f4`; Ruff/format/mypy passed (119
  modules, unchanged); full local suite 547 passed/32 skipped (up from 542/32 — 5 new
  `test_migrate_recovery_identity.py` cases: no-existing-pointer, already-current,
  verified-dry-run, execute-republishes-same-bytes, and refuses-on-tampered-identity).
  `test_colab_master_bootstrap.py`'s full 17-test suite, including the stdlib-only
  import check, stayed green. Notebook regenerated twice byte-identically at SHA-256
  `e8597fdd473c4a74f5c4ce671c03f3687f83ff12d5e590c7b5e88dea5e9627b1`. **No manual step
  needed anymore** — the user can just open a fresh Colab session and Runtime → Run
  all; not yet confirmed on real Colab hardware.
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
- (Commit `2b1ebff…`) Eighth real Colab bug (see note above): `float("inf")` in
  `clip_grad.max_norm` doesn't survive `mmengine.Config.dump()`/`fromfile()`'s
  eval()-based `.py` round-trip, crashing `evaluate.py` the first time the pipeline
  reached the screening-evidence step — after all five models completed a full real
  6000-step `screening` run. Replaced with a large finite sentinel (`1e9`). Reproduced
  and regression-tested with zero GPU dependency.

## Local gates

- **As of commit `a5519f4…` (automatic recovery-identity migration, no manual step):**
  Ruff and format checks pass for the full repository. Mypy passes for all 119
  configured `src/edgeguard` modules (unchanged count — `migrate_recovery_identity` and
  `peek_recovery_receipt` are new functions inside existing modules). Full pytest
  passes: 547 passed, 32 environment-gated skipped without the pinned MMSeg stack (up
  from 542/32 — 5 new `test_migrate_recovery_identity.py` cases, all running locally
  without the pinned stack for the same reason `test_compute_run_identity.py` does).
  `tests/unit/test_colab_master_bootstrap.py`'s full 17 tests pass, including
  `test_host_entrypoints_import_only_the_standard_library` — confirms
  `run_colab_master.py` still shells out to the migration script rather than importing
  `edgeguard.rescue` directly. Master notebook generation is byte-identical across two
  runs at commit `a5519f4…`, SHA-256
  `e8597fdd473c4a74f5c4ce671c03f3687f83ff12d5e590c7b5e88dea5e9627b1`. **Not yet run for
  real on Colab** — the next real session is the load-bearing confirmation.
- **As of commit `90b6bea…` (recovery-identity migration tool):** Ruff and format
  checks pass for the full repository. Mypy passes for all 119 configured
  `src/edgeguard` modules (unchanged count — `compute_run_identity` is a new function
  inside the existing `mmseg_runtime.py` module, not a new module).
  `scripts/migrate_recovery_identity.py` is new and mypy-clean. Full pytest passes: 542
  passed, 32 environment-gated skipped without the pinned MMSeg stack (up from 539/32 —
  3 new `test_compute_run_identity.py` cases; these run locally, unlike most
  `mmseg_runtime.py`-touching tests, because `compute_run_identity` only needs
  `mmengine.Config.fromfile` on a fake upstream-config fixture, not the full pinned
  torch/mmseg stack). Master notebook generation is byte-identical across two runs at
  commit `90b6bea…`, SHA-256
  `24aed1497ad4759663b4498b0e940ac38458a513c56115fd35af05492c6e6ffe` (no cell text
  changed, only `EXPECTED_PROJECT_COMMIT`). `tests/integration/test_notebook.py` (2/2)
  passes. **Not yet run for real on Colab** — this refactors `train_model`'s identity
  path, so a real Colab resume (after running the migration script) is the load-bearing
  confirmation, not yet obtained.
- **As of commit `6b30275…` (screening-stage model-scope restriction, letting a
  fresh Colab session skip resuming `bisenetv2`'s abandoned screening run):** Ruff
  and format checks pass for the full repository. Mypy passes for all 119 configured
  `src/edgeguard` modules (unchanged count). Full pytest passes: 539 passed, 32
  environment-gated skipped without the pinned MMSeg stack (up from 536/32 — 3 new
  `test_colab_pipeline.py` cases for `screening_models`, mirroring the existing
  `final_models` coverage). Same as the prior entry, this changes `ColabPipeline`'s
  `screening` orchestration directly, so the environment-gated CPU rehearsal suite
  staying skipped locally is a real gap — the next real Colab run is load-bearing.
  Master notebook generation is byte-identical across two runs at commit `6b30275…`,
  SHA-256 `e277eb9ba5653bf407b0ab7e09985c384473d9128a5a168c576078a4176a5791`
  (superseding the `5135f69…`/`ba0f377…` pin — no cell text changed, only
  `EXPECTED_PROJECT_COMMIT`). `tests/integration/test_notebook.py` (2/2) passes.
- **As of commit `5135f69…` (final-stage model-scope restriction):** Ruff and format
  checks pass for the full repository. Mypy passes for all 119 configured `src/edgeguard`
  modules (unchanged count — no new source module, existing ones edited). Full pytest
  passes: 536 passed, 32 environment-gated skipped without the pinned MMSeg stack (up
  from 533/32 — net +3: replaced one over-strict `test_colab_pipeline.py` case with four
  narrower ones covering the relaxed `final_models` validation). The CPU rehearsal suite
  (`tests/integration/test_colab_pipeline_cpu_rehearsal.py`) is environment-gated and
  stayed skipped locally, same as every prior local run in this session — this change
  touches `ColabPipeline`'s `final`/`selection`/`accept` orchestration directly, so real
  L4 confirmation on the next Colab run is the load-bearing test, not just documentation.
  Master notebook generation is byte-identical across two runs at commit `5135f69…`,
  SHA-256 `ba0f377549e1eb54354f1df6b0f98b0b916faf191b9ec1b9ce74da9da485d208` (superseding
  the `38df2ae…`/`7237aee…` pin — no cell text changed, only `EXPECTED_PROJECT_COMMIT`).
  `tests/integration/test_notebook.py` (2/2) passes.
- **As of commit `38df2ae…` (real-evidence training-log analysis tool):** Ruff and
  format checks pass for the full repository. Mypy passes for all 119 configured source
  modules (up from 118 — adds `training_log_analysis.py`). Full pytest passes: 533
  passed, 32 environment-gated skipped without the pinned MMSeg stack (up from 523/32 —
  10 new cases in `test_training_log_analysis.py`, using the user's own real pasted log
  excerpt as the primary fixture). This commit touches no training/mmseg-runtime code
  path, so the mmseg-gated test files and the full CPU rehearsal suite were not re-run
  this round. Master notebook generation is byte-identical across two runs at commit
  `38df2ae…`, SHA-256
  `7237aee68f3cd53ea346abe2874c173f235e3e775774eee02671648f42a70290` (superseding the
  `f2f2110…`/`97198ac…` pin — no cell text changed, only `EXPECTED_PROJECT_COMMIT`).
  `tests/integration/test_notebook.py` (3/3) and the local claim-safe execution harness
  (all 5 cells) both pass. Also manually smoke-tested the new CLI end to end against the
  user's real pasted log text and a synthetic `class_weights.json`, confirming the
  output shape and that the projected/measured mIoU are correctly kept distinct.
- **As of commit `f2f2110…` (live progress output for the private_inputs inventory):**
  Ruff and format checks pass for the full repository. Mypy passes for all 118
  configured source modules. Full pytest passes: 523 passed, 32 environment-gated
  skipped without the pinned MMSeg stack (up from 522/32 — 1 new case in
  `test_archive_inventory.py` locking in the progress-output shape via `capsys`). This
  commit touches no training/mmseg-runtime code path, so the mmseg-gated test files and
  the full CPU rehearsal suite were not re-run this round. Master notebook generation is
  byte-identical across two runs at commit `f2f2110…`, SHA-256
  `97198ac21eddf1ed1ffca4376c6ae4cebe0212a2d0fab95ffe4816cfceb5c427` (superseding the
  `24dd782…`/`1110e0ef…` pin — no cell text changed, only `EXPECTED_PROJECT_COMMIT`).
  `tests/integration/test_notebook.py` (3/3) and the local claim-safe execution harness
  (all 5 cells) both pass.
- **As of commit `24dd782…` (ninth real Colab bug — `cityscapes bundle identity
  mismatch`):** Ruff and format checks pass for the full repository. Mypy passes for
  all 118 configured source modules. Full pytest passes: 522 passed, 32
  environment-gated skipped without the pinned MMSeg stack (up from 520/32 — 2 new cases
  in `test_colab_data.py`: one reproduces the exact real-world scenario of an unrelated
  dataset's config being edited and confirms staging still succeeds, one confirms the
  check still rejects a genuine change to the affected dataset's own `required_paths`).
  This commit touches no training/mmseg-runtime code path, so the mmseg-gated test files
  and the full CPU rehearsal suite were not re-run this round. Master notebook
  generation is byte-identical across two runs at commit `24dd782…`, SHA-256
  `1110e0ef65fa675cdf247fb967a860931c848ae62ea3a2e72e505cd4b617ba77` (superseding the
  `7ffef5c…`/`2e89a7ba…` pin — no cell text changed, only `EXPECTED_PROJECT_COMMIT`).
  `tests/integration/test_notebook.py` (3/3, including `test_delivery_notebooks.py`) and
  the local claim-safe execution harness (all 5 cells) both pass.
- **As of commit `7ffef5c…` (private_inputs archive inventory + real-Colab bootstrap
  fix):** Ruff and format checks pass for the full repository (`ruff check .`,
  `ruff format --check .`). Mypy passes for all 118 configured source modules
  (`mypy src/edgeguard`, matching `ci.yml`; up from 116 — adds `archive_inventory.py` and
  `stall_guard.py`). Full pytest passes: 520 passed, 32 environment-gated skipped without
  the pinned MMSeg stack (up from 499/32 — 20 new cases in `test_archive_inventory.py`,
  covering exhaustive zip/tar scanning, measured per-class histogram correctness against
  a hand-built mask, corrupt-entry detection, exact-filename-only role matching that
  refuses to guess undeclared files, CLI cache-hit/cache-miss identity behavior, and
  hash-verified zip output; `test_delivery_notebooks.py`'s cell-count assertion and
  `test_notebook.py`'s PYTHONPATH guard were also corrected at this commit). This commit
  range does not touch any training/mmseg-runtime code path, so the mmseg-gated test
  files and the full CPU rehearsal suite were not re-run this round — nothing in this
  diff can affect their outcome. Master notebook generation is byte-identical across two
  runs at commit `7ffef5c…`, SHA-256
  `2e89a7ba32ed9b5f5c451650231aaca0bd67a6a5de2b4a790a8434f43a2a73d7` (superseding the
  `7b604c9…`/`b06b373a…` pin, which shipped with the PYTHONPATH bootstrap bug described
  above). `tests/integration/test_notebook.py` passes (2/2), and the local claim-safe
  execution harness (`scripts/dev/run_delivery_notebooks_local.py`) passes all 5 code
  cells (up from 4 — the new inventory cell correctly no-ops under `LOCAL_TEST_MODE`) —
  note this harness only proves syntax/import correctness under `LOCAL_TEST_MODE`, which
  is exactly the class of bug (a bare-interpreter import failure) that slipped through it
  before being caught on real Colab; the direct throwaway-venv verification described
  above is what actually confirms the fix.
- **Prior state (commit `2b1ebff…`, the eighth-bug fix):** Ruff/format/mypy passed for 116
  modules; full pytest 499 passed/32 skipped; mmseg-gated files 27/27 with
  `EDGEGUARD_MMSEG_CHECKOUT` at `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8`; the full CPU
  rehearsal suite failed on this Mac with `RuntimeError: view size is not compatible with
  input tensor's size and stride` during `backward()` — the known MPS operator-gap class
  of issue documented in `canonical-colab-runbook.md`, confirmed via `git stash` to be
  identical on pre-fix and post-fix code (unrelated to the `clip_grad` fix; a separate,
  pre-existing local-environment regression that still needs its own investigation — not
  touched by the archive-inventory commit either). Notebook SHA-256 at that commit was
  `25c7393e4ac216700bba35a9b846ba89bc97d5b32700cca09a0dfad6003334e1`.
- **Noted, not fixed, out of scope, carried over from commit `a50b635…`:** running the
  *entire* suite with `EDGEGUARD_MMSEG_CHECKOUT` set produces 12 failures in
  `tests/unit/test_dataset_preparation.py` that do not reproduce when that file is run
  alone or as part of the mmseg-gated set — a pre-existing test-isolation/ordering issue
  unrelated to any change in this session (that file never touches `mmseg_runtime.py` or
  anything mmseg-related).
- **Pending at this commit:** claim-safe local cell execution above only proves the
  generated notebook's cells import/execute their own syntax correctly under
  `EDGEGUARD_NOTEBOOK_LOCAL_TEST=1` (see the note on this above) — the new inventory cell
  itself has not been exercised against a real Drive-mounted `private_inputs/` folder or
  real archive files; only the module-level logic is unit-tested (against small synthetic
  zip/tar fixtures) and the CLI subprocess path is tested end to end locally. Remote
  Linux workflow `semantic-framework-cpu-probe.yml` has not been re-run at this commit,
  and neither the optimizer fix nor the `BoundaryLoss` dtype fix has been confirmed on
  real CUDA hardware yet — the next real Colab attempt is the actual test for those; this
  commit is purely additive engineering and does not change what that attempt needs to
  prove.

## Next external action

Push this commit, then open the master notebook from the pushed branch in a fresh Colab L4
+ High-RAM runtime and use Run all (Colab Pro/Pro+ background execution is recommended so
the session survives closing the browser tab). This picks up mid-campaign: the previous
real L4 run at `a50b635…` already got all five models through smoke, pilot, and a full
6000-step screening run — real screening mIoU: segformer_b0 16.48%, fast_scnn 20.53%,
pidnet_s 26.54%, ddrnet_23_slim 25.21%, bisenetv2 17.41% — before crashing on the
`clip_grad` dump/reload bug this commit fixes. Watch specifically that the
screening-evidence evaluation step (the exact thing that crashed) now completes cleanly
for all five models. `train_model()` dumps a fresh `resolved.py` on every entry —
including no-op resume-and-skip re-entries — so no manual Drive cleanup is needed; the
next run will regenerate a correctly-serializable config automatically. If that holds,
the real next milestone is HPO for the top-two screening models, then `final` (40000
steps) for all five. If the session ends, repeat Run all in a new compliant runtime —
this is a Colab platform limit, not something the notebook can automate away. Do not
change the notebook or select stages manually.

The most recent real attempt (application commit `7ffef5c…`) never reached that
screening-evidence step this time: it crashed earlier, in the `data` stage, with
`cityscapes bundle identity mismatch` (see the ninth-bug note above) — an unrelated
config-file edit (`bd3ea56`, the WildDash2/RailSem19 role commit) had invalidated the
already-staged `cityscapes` bundle's identity check, purely as a side effect of hashing
the whole plan file instead of just the relevant dataset's section. This is now fixed at
`24dd782…`. Watch specifically that the `data` stage completes cleanly for both
`cityscapes` and `idd20k` (staging from the existing Drive bundles, no rebuild needed),
then that the campaign proceeds to resume mid-screening as before.

This run will also, for the first time with the bootstrap fix applied, exercise the
private_inputs archive-inventory cell for real against the actual Drive-mounted
`EdgeGuard/private_inputs/` folder (18 files as of this commit) — it runs early, before
the campaign subprocess, and is wrapped to never block or fail the campaign if it errors.
The first real attempt at commit `7b604c9…` failed cleanly with `ModuleNotFoundError: No
module named 'edgeguard'` (caught, campaign unaffected) — fixed at `7ffef5c…` (see the
note above) by scoping `PYTHONPATH` to this step's own subprocess call and installing its
few light pure-Python dependencies first. Confirm it prints a completed report and that
`EdgeGuard_Data_Inventory.zip` downloads; if it still fails, the printed exception plus
`Drive/EdgeGuard/reports/private_inputs_inventory/` (if partially written) has the
diagnostic — it does not need to succeed for the real campaign to proceed.

**2026-08-13 real attempt at commit `a5519f4` — data staged clean, migration refused (see
`docs/AI_USAGE_LOG.md`'s last entry for the full diagnosis).** `stdlib-hermetic-bootstrap`,
`five-model-runtime-canary`, `restore` (correctly skipped a differently-committed
campaign-state tarball), and `data` (Cityscapes + all 33 IDD20K shards) all completed
cleanly. The new automatic `recovery-identity-migration` stage then ran for the first time
on real hardware and reported `status: verification_failed` for all four
`screening_models` (segformer_b0, fast_scnn, pidnet_s, ddrnet_23_slim) against
`old_project_commit: f2f2110…` — it correctly refused to republish rather than guess, so
nothing was fabricated, but these four models' real 6000-step screening checkpoints
cannot resume and will retrain from scratch when the pipeline next reaches `screening`
(~11-12 GPU-hours estimated from this session's real pilot-stage per-iteration pace,
dominated by `fast_scnn` at ~5.2s/iter). Root cause was investigated (protocol yaml,
`colab_pipeline.py`'s `_train_command`, `train.py`'s CLI defaults, `resolve_auto_precision`,
and the mmsegmentation lock pin were all statically ruled out) but not confirmed —
remaining suspect is the dataset-manifest content, unverifiable without live Colab access.
Decision: accept the retrain cost rather than interrupt the live, deadline-critical run to
test hypotheses. **The screening mIoU evidence from the earlier `a50b635…` run above is
still real and still valid** (segformer_b0 16.48%, fast_scnn 20.53%, pidnet_s 26.54%,
ddrnet_23_slim 25.21%, bisenetv2 17.41%) — only the checkpoint *bytes* can't resume, the
measured numbers were never lost. Session then disconnected mid-`pilot` (fast_scnn, ~iter
1700/2000); the next session needs Run all from a fresh runtime. **HPO, `final` (40000
steps), `selection`, `ablation`, `accept`, `evaluate`, `export`, `thesis`/`report`, and
`package` have never yet been exercised on real Colab hardware at all** — everything
proven so far stops at `screening`. The user is considering switching from L4 to A100 for
the next attempt; confirmed via code audit that this needs zero code changes
(`_resource_gate` already accepts A100 by name, `device_batch`/`effective_batch` in
`configs/rescue/semantic_first.yaml` are static and GPU-type-independent, and the pinned
`torch-2.1.1+cu121` wheel is not architecture-specific) — this is a pure Colab-UI runtime
selection, not a code or notebook change.

Do not create `colab-v0.1.0-rc1` until two independent clean L4 sessions pass the exact
lock/five-model FP32/AMP canary and the real 50-step interruption/resume proof. After the
campaign completes, build TensorRT only on the target Jetson and attach actual 25W
telemetry. Do not merge, tag, open sealed datasets, upgrade JetPack, or change Jetson power
mode without a separate explicit decision.
