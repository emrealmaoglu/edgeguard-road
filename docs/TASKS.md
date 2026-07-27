# Executable Task System

## Status

- `EG-OOD-001 — Current Fishyscapes/OOD foundation closure`: **completed and
  remotely verified** at Commit D
  `345d9fd1dcff0a7aa9c54c6f3929c2c751c24c7c`.
- `EG-THESIS-001 — Scope, title and claim migration`: **implemented, locally
  validated, committed, pushed, and remotely verified** at
  `ee4460bda9b518a4e784cd43ad23d043ad15cd7b`.
- `EG-DATA-001 — Storage, access and ontology gate`: **completed and remotely
  verified** at `0114d4e4778c1d6e53b6359e0a11f71eb15d2fb4`.
- `EG-DATA-002 — Cityscapes Fine train preparation`: **implemented and locally
  tested with synthetic ZIP fixtures; real private-Drive preparation and human split
  selection pending**.
- No dataset acquisition, framework installation, training, Jetson mutation, or
  sealed-data access is in progress.

Each task below is a commit-sized human-gated unit. Compute and storage values are
planning ranges until replaced by measurements.

## Immediate critical path

### EG-DATA-001 — Storage, access and ontology gate

- **Objective:** Freeze the private Drive naming contract, acquisition ledger,
  semantic/detection/OOD ontology, dataset roles, and legal/human gates without
  downloading data.
- **Dependencies:** Accepted `EG-THESIS-001` documentation commit.
- **Human inputs:** Drive root convention; dataset priority; terms/license decisions;
  canonical ontology approval.
- **Local work:** Inspect existing manifests, define path-free archive/catalog record,
  map native labels to the proposed ontology, and estimate task-specific capacity.
- **Colab work:** None beyond a future mount/path probe after approval.
- **Jetson work:** Read-only storage inventory only under a separately authorized
  device task; none here.
- **Expected outputs:** Proposed runtime-root storage map, access-decision matrix,
  versioned machine-readable ontology, human-gate checklist, and updated data catalog.
- **Tests:** Schema/canonical-hash tests, duplicate/path/root rejection, documentation
  link and secret scans.
- **Acceptance:** Local validators/tests pass; then the human freezes ontology,
  roles, storage convention, and first acquisition. No private absolute root enters
  Git and no Drive hierarchy is claimed created.
- **Stop conditions:** Unclear terms, conflicting ontology, unknown Drive convention,
  or request to acquire data before approval.
- **Fallback:** Keep affected dataset blocked and progress only independent approved
  catalog entries.
- **Compute estimate:** Local CPU under 2 hours; 0 GPU hours.
- **Storage estimate:** Git under 1 MiB; no dataset storage created.
- **Commit boundary:** Catalog/schema/documentation/tests only; no downloads.
- **Next task:** `EG-DATA-002` after its Cityscapes access and human-review gates.

### EG-DATA-002 — Cityscapes Fine train preparation

- **Objective:** Verify approved Fine train archives, inventory city/sequence/class
  distributions, and propose three deterministic group-atomic splits. Human
  selection/freeze remains a separate gate.
- **Dependencies:** `EG-DATA-001` and approved Cityscapes Fine train access.
- **Human inputs:** Official archives, hashes/source/terms, split-candidate selection.
- **Local work:** Implement project-specific verification, manifest and distribution
  reports using tiny fixtures; do not expand the full dataset on the Mac.
- **Colab work:** Copy archives to ephemeral storage, verify/extract once, compute
  distributions, and write candidate manifests back to Drive.
- **Jetson work:** None.
- **Expected outputs:** Archive records, class/group inventory, unapproved candidate
  reports, root-free manifests, mapping evidence, and a small evidence package.
- **Tests:** Hash, unsafe archive member, duplicate/missing pair, geometry, mapping,
  group leakage, determinism, root-independence.
- **Acceptance:** Human selects a candidate; no group overlap; distribution and rare
  class diagnostics are complete; official val is absent from trial roles.
- **Stop conditions:** Hash/terms mismatch, insufficient ephemeral disk, mapping
  ambiguity, or group leakage.
- **Fallback:** Produce inventory/candidate report only and leave split gate open.
- **Compute estimate:** 2–6 CPU hours, 0 GPU hours.
- **Storage estimate:** Drive reserve up to 25 GiB; Colab active footprint measured;
  local fixtures only.
- **Commit boundary:** Preparation code, tests, sanitized manifests/summaries; no data.
- **Next task:** `EG-SEG-001` after human split freeze.

### EG-COMPUTE-001 — Accelerator and I/O throughput probe

- **Objective:** Replace assumed T4/L4/A100/H100 allocation with measured behavior for
  one lightweight semantic model, one heavier semantic model, and one detector.
- **Dependencies:** `EG-SEG-002`, a smoke-ready representative detector from
  `EG-DET-001`, and approved representative shards/framework pins.
- **Human inputs:** Available Colab runtimes and maximum probe budget.
- **Local work:** Build probe config/report contract and fixture tests.
- **Colab work:** Fixed-resolution/sample/dataloader FP32 and stable FP16/BF16 probes;
  record throughput, memory, data wait, preprocessing, and sync time.
- **Jetson work:** None.
- **Expected outputs:** Per-accelerator probe reports and revised queue estimates.
- **Tests:** Config identity, finite loss/gradient, effective-batch equality, resume and
  path-free serialization.
- **Acceptance:** Measurements support preferred/fallback assignments and update all
  affected runtime estimates.
- **Stop conditions:** Unfrozen data/model config, I/O-dominated invalid comparison,
  non-finite training, or unavailable representative accelerator.
- **Fallback:** Use the slower validated accelerator and retain wider budget ranges.
- **Compute estimate:** 2–3 GPU hours total.
- **Storage estimate:** Under 5 GiB ephemeral plus small Drive reports/checkpoints.
- **Commit boundary:** Probe implementation/config/report only; measurements external.
- **Next task:** `EG-SEG-003` and calibrated queue-budget updates.

### EG-SEG-001 — Pinned semantic training laboratory

- **Objective:** Select and pin the smallest training framework that supports most of
  the five models while defining checkpoint-to-EdgeGuard handoff contracts.
- **Dependencies:** `EG-DATA-002` split freeze.
- **Human inputs:** Framework/license approval and accepted model source pins.
- **Local work:** Reproducible install guidance, shared config contract, checkpoint
  identity/resume logic, conversion adapter fixtures, and CPU unit tests.
- **Colab work:** One-batch import/forward/backward/checkpoint/resume compatibility
  probes; no broad training.
- **Jetson work:** None.
- **Expected outputs:** Pinned lab decision, environment record, shared training
  contract, and five smoke-ready configs.
- **Tests:** Config validation, label/shape/loss contracts, checkpoint overwrite and
  incompatible-resume refusal, deterministic experiment IDs.
- **Acceptance:** Human approves framework pins and handoff; at least the smoke path
  for all five models is explainable.
- **Stop conditions:** License conflict, more than one uncontrolled framework, or
  checkpoint/export contract cannot be stated.
- **Fallback:** Replace only the unsupported candidate through a separate approved
  source review; do not build a generalized plugin system.
- **Compute estimate:** 1–3 GPU hours compatibility work.
- **Storage estimate:** Under 20 GiB ephemeral; small recovery checkpoints in Drive.
- **Commit boundary:** Training lab/config/tests/docs; no model result claim.
- **Next task:** `EG-SEG-002`.

### EG-SEG-002 — Five-model semantic smoke

- **Objective:** Run bounded smoke experiments for Fast-SCNN, BiSeNetV2, PIDNet-S,
  DDRNet-23-Slim, and SegFormer-B0.
- **Dependencies:** `EG-SEG-001`, frozen fit/select manifests.
- **Human inputs:** Approved initialization/checkpoint terms for each model.
- **Local work:** Config/adapter unit tests and tiny forward where feasible.
- **Colab work:** 1–3 epochs or fixed small subset; verify loss, gradients,
  checkpoint, exact resume, logits, evaluator, and failure recording.
- **Jetson work:** None.
- **Expected outputs:** Five smoke records/checkpoints/curves or explicit failures.
- **Tests:** Native/aligned output contract, finite loss/gradient, ontology, resume,
  overwrite, and path-free completion summary.
- **Acceptance:** Every model has a successful smoke or bounded documented failure;
  no scientific ranking is claimed.
- **Stop conditions:** Data/ontology mismatch, non-finite loss, silent partial load, or
  identity-incompatible resume.
- **Fallback:** One bounded repair; then mark failed and seek human model substitution.
- **Compute estimate:** Included in the 2–3 hour probe/smoke family.
- **Storage estimate:** Under 5 GiB per active smoke; final small records in Drive.
- **Commit boundary:** Smoke code/config/tests and external evidence references.
- **Next task:** `EG-SEG-003`.

### EG-SEG-003 — Common architecture screening

- **Objective:** Compare all smoke-valid models under one short `512×1024` protocol
  and nominate top three.
- **Dependencies:** `EG-SEG-002` and measured `EG-COMPUTE-001` accelerator/I/O
  selection.
- **Human inputs:** Common epoch/sample budget, seed policy, and promotion rubric.
- **Local work:** Validate resolved configs, registry rows, comparison/report code.
- **Colab work:** Execute five short training/evaluation jobs from ephemeral data with
  interruption-safe state.
- **Jetson work:** None; deployment proxies only.
- **Expected outputs:** Five screening checkpoints, curves, metrics, compute ledger,
  failure reports, and top-three proposal.
- **Tests:** Same effective batch/data/augmentation/evaluator, complete run identity,
  deterministic comparison table.
- **Acceptance:** Human confirms fair comparison and top-three candidates; results are
  labeled screening, not final.
- **Stop conditions:** Protocol drift, missing run identity, budget overrun, or fewer
  than three interpretable candidates.
- **Fallback:** Repair invalid runs within the same budget or promote fewer only by an
  explicit human decision.
- **Compute estimate:** 8–15 GPU hours.
- **Storage estimate:** 25–60 GiB Drive for checkpoints/logs; active data shared.
- **Commit boundary:** Screening configs/report references; artifacts external.
- **Next task:** `EG-EXPORT-001`, then `EG-SEG-004`.

### EG-EXPORT-001 — Early export-feasibility gate

- **Objective:** Test native-logit ONNX feasibility for all screened semantic
  candidates before medium training/HPO.
- **Dependencies:** `EG-SEG-003` checkpoints and accepted fixed export fixture.
- **Human inputs:** Interpretation of pass/conditional/fail and promotion impact.
- **Local work:** Export/checker/comparison fixtures where dependencies permit.
- **Colab work:** Export with measured supported path, runtime inference, operator and
  numerical reports; no TensorRT performance claim.
- **Jetson work:** Optional import/build smoke only through separate device approval;
  no sustained benchmark.
- **Expected outputs:** Five export reports or structured failures and deployment
  eligibility flags.
- **Tests:** Output name/shape/dtype/finiteness, native-vs-aligned separation, checker,
  runtime, path-free artifact manifest.
- **Acceptance:** Every candidate has reproducible feasibility evidence; human uses it
  in top-three selection.
- **Stop conditions:** Export requires a large architecture rewrite/custom operator,
  or numerical comparison cannot be interpreted.
- **Fallback:** Preserve failed export and keep model results-only if scientifically
  useful; do not silently switch exporter.
- **Compute estimate:** 2–5 GPU hours.
- **Storage estimate:** Under 20 GiB external exports/reports; ONNX never in Git.
- **Commit boundary:** Export adapter/tests/report schema; binaries external.
- **Next task:** `EG-SEG-004`.

### EG-SEG-004 — Top-three medium-budget training

- **Objective:** Train the promoted top three under equal medium budget and nominate
  top two for HPO.
- **Dependencies:** `EG-EXPORT-001` and human top-three approval.
- **Human inputs:** Medium schedule, selection rubric, initialization decisions.
- **Local work:** Config and artifact validation only.
- **Colab work:** Three interruption-safe training/evaluation jobs with measured
  precision and effective batch.
- **Jetson work:** None.
- **Expected outputs:** Three project checkpoints, curves, select metrics, compute and
  failure ledger, top-two proposal.
- **Tests:** Resume identity, finite training, evaluator parity, hash-verified sync.
- **Acceptance:** Three interpretable runs and human-approved top two.
- **Stop conditions:** Budget overrun, invalid comparison, corrupted recovery state,
  or repeated instability.
- **Fallback:** Re-run only invalid work within budget; preserve negative result.
- **Compute estimate:** 15–24 GPU hours.
- **Storage estimate:** 60–150 GiB Drive including recovery state/logs.
- **Commit boundary:** Medium configs and sanitized summaries only.
- **Next task:** `EG-SEG-005`.

### EG-SEG-005 — Top-two limited HPO

- **Objective:** Optimize only the top two at fixed `512×1024` with pruning and a
  bounded trial budget.
- **Dependencies:** `EG-SEG-004`, human-approved search space/budget/metric.
- **Human inputs:** Trial count, sampler/pruner, promotion rule, seed policy.
- **Local work:** Search-space validation, identity and study-snapshot tests.
- **Colab work:** Resume-safe trials; separate study names; periodic JSON/CSV snapshot;
  no concurrent untested SQLite writers.
- **Jetson work:** None; early export evidence remains a selection input.
- **Expected outputs:** Trial ledger, snapshots, best configurations, failed/pruned
  compute, and final-run proposal.
- **Tests:** Parameter bounds, fixed resolution/effective batch comparison, trial
  identity, resume/overwrite failure.
- **Acceptance:** Human freezes final configs after reviewing top trials and compute.
- **Stop conditions:** Budget exhausted, study corruption, data leakage, or no useful
  improvement over medium baseline.
- **Fallback:** Use the best validated medium config; do not extend HPO automatically.
- **Compute estimate:** 12–22 GPU hours.
- **Storage estimate:** 30–100 GiB Drive; bounded checkpoints, study snapshots.
- **Commit boundary:** HPO config/code/tests and summary; study database external.
- **Next task:** `EG-SEG-006`.

### EG-SEG-006 — Three final semantic runs

- **Objective:** Produce three frozen project-owned final checkpoints, including at
  least one random-initialization training run, then perform one common official-val
  evaluation per checkpoint.
- **Dependencies:** `EG-SEG-005`, human final config/init freeze.
- **Human inputs:** Three final configs, random-init assignment, confirmation gate.
- **Local work:** Preflight identities and post-run artifact verification.
- **Colab work:** Three full interruption-safe runs and common Cityscapes-val
  evaluation from frozen configs/checkpoints after training completes.
- **Jetson work:** None until final export.
- **Expected outputs:** Three checkpoints/curves, confirmation metrics, compute/failure
  ledger, final semantic model decision.
- **Tests:** Exact config/data/init identity, resume, checkpoint hashes, evaluator and
  official-val non-leakage audit.
- **Acceptance:** Human accepts three complete runs, verifies that official val was
  excluded from HPO, `train_select`, and temperature fitting, and selects
  scientific/deployment candidates without mixing training stages.
- **Stop conditions:** Official val influences model selection/calibration, the
  random-init run is missing, recovery evidence is incomplete, or an artifact is
  corrupted.
- **Fallback:** Resume exact run; replace a failed final only by human decision.
- **Compute estimate:** 18–30 GPU hours.
- **Storage estimate:** 100–250 GiB Drive plus verified backup.
- **Commit boundary:** Final configs/summaries/evidence references; checkpoints external.
- **Next task:** `EG-OOD-002`, `EG-EXPORT-002`, and detector/fusion integration.

## Parallel detector track

### EG-DET-001 — Detection ontology, data, and two-model smoke

- **Objective:** Freeze BDD100K mapping and smoke YOLO11n and RT-DETR-R18.
- **Dependencies:** `EG-DATA-001`, approved BDD package and source/framework terms.
- **Human inputs:** Dataset acquisition, class mapping, detector source/license pins.
- **Local work:** Mapping/adapter/config fixtures and metric contracts.
- **Colab work:** Validate archive/shard, then one-batch and bounded smoke/resume runs.
- **Jetson work:** None.
- **Expected outputs:** Frozen detector manifests/ontology and two smoke records.
- **Tests:** Boxes/classes/geometry, group leakage, loss/gradient, resume, export output.
- **Acceptance:** Human accepts ontology and both valid smokes or one documented failure.
- **Stop conditions:** License/mapping ambiguity, corrupt package, or label leakage.
- **Fallback:** Keep package/model blocked; evaluate an approved alternative later.
- **Compute estimate:** 2–5 GPU hours.
- **Storage estimate:** BDD Drive reserve up to 350 GiB; active deterministic shard.
- **Commit boundary:** Adapter/config/tests/docs; no data/checkpoints.
- **Next task:** `EG-DET-002`.

### EG-DET-002 — Two-detector common screening

- **Objective:** Compare two detector families under one data, resolution, metric, and
  compute protocol.
- **Dependencies:** `EG-DET-001`, measured accelerator allocation.
- **Human inputs:** Short budget, seed and promotion rubric.
- **Local work:** Report validation and tiny fixtures.
- **Colab work:** Two short training/evaluation jobs with interruption-safe state.
- **Jetson work:** Early export smoke may be measured after screening.
- **Expected outputs:** Common mAP/precision/recall/cost reports and primary proposal.
- **Tests:** Protocol equality, ontology, metric, resume, artifact hashes.
- **Acceptance:** Human accepts fair comparison and primary detector.
- **Stop conditions:** Protocol drift, incomplete class mapping, or repeated failure.
- **Fallback:** Preserve failure and select only a validated detector by human decision.
- **Compute estimate:** 4–8 GPU hours.
- **Storage estimate:** 30–80 GiB Drive.
- **Commit boundary:** Screening configs/summaries; artifacts external.
- **Next task:** `EG-DET-003`; optional `EG-DET-004` aggressive.

### EG-DET-003 — Primary final detector

- **Objective:** Fine-tune the selected detector to a frozen final checkpoint and
  prepare integration/export evidence.
- **Dependencies:** `EG-DET-002` human promotion.
- **Human inputs:** Final schedule/config and metric acceptance rule.
- **Local work:** Config/export contract validation.
- **Colab work:** Full interruption-safe training and frozen confirmation.
- **Jetson work:** Final export/benchmark only after `EG-EXPORT-002`.
- **Expected outputs:** Project detector checkpoint, curves, metrics, compute ledger.
- **Tests:** Resume, artifact hashes, ontology and output contract.
- **Acceptance:** Human accepts final detector or records a negative result.
- **Stop conditions:** Budget/instability/export blocker without bounded resolution.
- **Fallback:** Use best screening checkpoint or approved alternate detector.
- **Compute estimate:** 4–8 GPU hours within required detector budget.
- **Storage estimate:** 40–100 GiB Drive and verified backup.
- **Commit boundary:** Final detector config/summary; checkpoint external.
- **Next task:** `EG-FUSION-001` and `EG-EXPORT-002`.

### EG-DET-004 — Secondary detector HPO and final run

- **Objective:** Full HPO/final comparison for the second detector.
- **Dependencies:** Core detector evidence safe and aggressive budget approved.
- **Human inputs:** Search/budget/promotion decision.
- **Local work:** HPO validation.
- **Colab work:** Pruned resume-safe trials and one final run.
- **Jetson work:** Only if final backend exports successfully.
- **Expected outputs:** Secondary study, checkpoint, comparison and failures.
- **Tests:** Same as primary plus HPO identity.
- **Acceptance:** Human accepts aggressive comparison.
- **Stop conditions:** Week-5 core risk or budget overrun.
- **Fallback:** Retain screening-only results card.
- **Compute estimate:** Part of additional 90–220 GPU-hour aggressive queue.
- **Storage estimate:** 50–150 GiB Drive.
- **Commit boundary:** Aggressive configs/summaries only.
- **Next task:** Optional broader detector ablations.

## Parallel OOD, context, and deployment track

### EG-OOD-002 — Static development and zero-shot baselines

- **Objective:** Prepare approved Fishyscapes Static and compare four zero-shot scores.
- **Dependencies:** `EG-DATA-001`, selected semantic checkpoint, approved generator.
- **Human inputs:** Generator pin/terms/input approval and development protocol.
- **Local work:** Generator/manifest checks and OOD report fixtures.
- **Colab work:** Generate to external storage, run four scores on Static, no threshold
  or Lost & Found access.
- **Jetson work:** Score-cost microbench only later.
- **Expected outputs:** Static manifest, score AP/FPR95 reports, failures.
- **Tests:** Generator identity, score direction/finiteness/grid, metric edge cases.
- **Acceptance:** Human accepts development comparison and next method candidates.
- **Stop conditions:** Unapproved inputs, generator mismatch, or attempted L&F/SMIYC use.
- **Fallback:** Retain code foundation and block real OOD claims.
- **Compute estimate:** 2–5 GPU hours.
- **Storage estimate:** Up to 50 GiB generated Drive reserve.
- **Commit boundary:** Generator integration/tests/summaries; generated data external.
- **Next task:** `EG-CAL-001` and `EG-OOD-003`.

### EG-CAL-001 — Semantic calibration

- **Objective:** Fit and evaluate semantic calibration without conflating it with OOD
  thresholding.
- **Dependencies:** Final semantic checkpoint and frozen `train_calibration`.
- **Human inputs:** Calibrator candidates and acceptance protocol.
- **Local work:** Calibration metric/tests and artifact contract.
- **Colab work:** Fit calibrator; report ECE/NLL/Brier and OOD before/after on dev.
- **Jetson work:** Measure selected calibrator overhead later.
- **Expected outputs:** Calibrator/config/hash and calibration report.
- **Tests:** Split enforcement, numerical stability, serialization, no holdout access.
- **Acceptance:** Human freezes calibrator or accepts no-calibration negative result.
- **Stop conditions:** Leakage, semantic degradation outside frozen rule, instability.
- **Fallback:** Keep uncalibrated baseline and document result.
- **Compute estimate:** 1–3 GPU hours.
- **Storage estimate:** Under 10 GiB.
- **Commit boundary:** Calibration code/config/tests/summaries.
- **Next task:** `EG-OOD-003`.

### EG-OOD-003 — Trainable anomaly-aware method

- **Objective:** Select a minimal exportable feature tap and loss, train on synthetic
  OOD, and compare against zero-shot baselines on Static.
- **Dependencies:** Semantic winner, `EG-OOD-002`, `EG-CAL-001`, synthetic data gate.
- **Human inputs:** Candidate tap/loss set and promotion rule.
- **Local work:** Adapter/loss/export fixtures and semantic-retention tests.
- **Colab work:** Bounded candidate runs, then selected trainable method; no L&F until
  all decisions freeze.
- **Jetson work:** Early selected-head cost/export check.
- **Expected outputs:** Feature/loss decision, checkpoint, Static comparison, semantic
  retention and export report.
- **Tests:** Tap shape, loss finite/gradient, direction, semantic retention, export.
- **Acceptance:** Human freezes one method or accepts zero-shot fallback.
- **Stop conditions:** Semantic damage, non-exportable complexity, leakage, budget.
- **Fallback:** Best zero-shot method remains core.
- **Compute estimate:** 3–7 GPU hours required; broader ablations aggressive.
- **Storage estimate:** 30–100 GiB Drive.
- **Commit boundary:** Method code/config/tests/summaries; model external.
- **Next task:** Frozen L&F holdout gate, `EG-CONTEXT-001`.

### EG-CONTEXT-001 — Road context and connected components

- **Objective:** Convert anomaly maps to road-aware regions and measure false-alarm
  reduction versus anomaly retention.
- **Dependencies:** Frozen OOD development method.
- **Human inputs:** Candidate parameter ranges and acceptance trade-off.
- **Local work:** Pure array/component implementation and fixtures.
- **Colab work:** Development ablations only.
- **Jetson work:** Selected configuration cost later.
- **Expected outputs:** Region contract, ablation report, frozen parameters.
- **Tests:** Geometry, connectivity, ignore/road overlap, edge cases, determinism.
- **Acceptance:** Human accepts interpretable trade-off.
- **Stop conditions:** Holdout leakage or relevant anomaly suppression outside rule.
- **Fallback:** Disable context filter and retain raw OOD result.
- **Compute estimate:** 1–3 GPU hours evaluation; primarily CPU.
- **Storage estimate:** Under 20 GiB external summaries/visuals.
- **Commit boundary:** Context/component code/tests/config/report.
- **Next task:** `EG-TEMP-001`.

### EG-TEMP-001 — Lightweight temporal persistence

- **Objective:** Stabilize component events over frames with explicit sequence state.
- **Dependencies:** `EG-CONTEXT-001`, approved temporal dataset and sequence split.
- **Human inputs:** Dataset access, event metric and persistence candidates.
- **Local work:** State machine/association implementation and synthetic sequence tests.
- **Colab work:** Development sequence evaluation.
- **Jetson work:** Selected lightweight method cost.
- **Expected outputs:** Temporal method/config, event/flicker ablation, failure cases.
- **Tests:** Reset, gaps, ordering, ID lifecycle, deterministic association.
- **Acceptance:** Human accepts stability/retention/cost trade-off.
- **Stop conditions:** Sequence leakage, hidden state across videos, excessive cost.
- **Fallback:** Stateless components or simpler persistence counter.
- **Compute estimate:** 1–4 GPU hours evaluation; mostly CPU.
- **Storage estimate:** Up to 100 GiB approved temporal data; small Git summaries.
- **Commit boundary:** Temporal code/tests/config/report.
- **Next task:** `EG-FUSION-001`.

### EG-DEPTH-001 — Bounded relative-depth feasibility

- **Objective:** Decide whether relative depth adds useful proximity evidence within
  the Jetson budget without blocking core integration.
- **Dependencies:** Approved model/data terms and stable core outputs.
- **Human inputs:** Candidate model, bounded dataset, cost/utility gate.
- **Local work:** Output contract and tiny fixtures.
- **Colab work:** Small plausibility/export evaluation only.
- **Jetson work:** Bounded latency/memory/power probe.
- **Expected outputs:** Go/defer report and optional profile contract.
- **Tests:** Finiteness, monotonic/relative checks, resize/alignment, missing-depth path.
- **Acceptance:** Human enables depth only if utility and device cost pass.
- **Stop conditions:** Core schedule impact, metric-distance overclaim, export/OOM cost.
- **Fallback:** `depth_status=disabled`; core fusion continues.
- **Compute estimate:** Aggressive queue, 2–8 GPU hours plus Jetson device-hours.
- **Storage estimate:** 10–50 GiB external.
- **Commit boundary:** Spike code/config/report only.
- **Next task:** Optional input to `EG-FUSION-001`.

### EG-FUSION-001 — Explainable operational-risk fusion

- **Objective:** Combine known-object, semantic, OOD, context, temporal, and optional
  depth evidence into deterministic explainable labels.
- **Dependencies:** `EG-DET-003`, `EG-OOD-003`, `EG-TEMP-001`; depth optional.
- **Human inputs:** Rule candidates, claim wording, acceptance ablations.
- **Local work:** Typed fusion contract, rules, failure-safe states, fixtures.
- **Colab work:** Development-stream ablations.
- **Jetson work:** Integrated selected-profile measurement later.
- **Expected outputs:** Per-event contribution log, rule/config hash, ablation report.
- **Tests:** Missing components, contradictory signals, determinism, sequence reset.
- **Acceptance:** Human accepts explainability and operational-label boundary.
- **Stop conditions:** Opaque learned fusion, probability overclaim, hidden fallback.
- **Fallback:** Simpler rule set or component-only display.
- **Compute estimate:** 1–3 GPU hours evaluation; primarily CPU.
- **Storage estimate:** Under 20 GiB external evidence.
- **Commit boundary:** Fusion code/tests/config/report.
- **Next task:** `EG-EXPORT-002`.

### EG-EXPORT-002 — Final ONNX/TensorRT numerical gate

- **Objective:** Export frozen selected components and validate PyTorch/ONNX/TensorRT
  output equivalence before sustained benchmarking.
- **Dependencies:** Final semantic/detector/OOD/fusion selections.
- **Human inputs:** Numerical gates after pilot evidence and backend promotion decision.
- **Local work:** Comparison/report tests and artifact verification.
- **Colab work:** Final ONNX export and CUDA runtime validation.
- **Jetson work:** Build TensorRT engines on target; run equivalence fixtures.
- **Expected outputs:** ONNX/engine manifests, operator and numerical reports, failures.
- **Tests:** Shapes/dtypes/finiteness, argmax/score/task differences, hashes, provenance.
- **Acceptance:** Human approves deployed backend per component.
- **Stop conditions:** Unsupported operator, unacceptable numerical drift, stale engine,
  or unclean source.
- **Fallback:** Alternate promoted model/profile; failed model remains results-only.
- **Compute estimate:** 2–8 GPU hours plus Jetson build/device-hours.
- **Storage estimate:** 20–100 GiB Drive exports/metadata; binaries outside Git.
- **Commit boundary:** Export code/tests/sanitized reports; binaries external.
- **Next task:** `EG-JETSON-001`.

### EG-JETSON-001 — Sustained selected-pipeline benchmark

- **Objective:** Measure frozen deployed profiles on Jetson Orin Nano Super.
- **Dependencies:** `EG-EXPORT-002` and explicit device authorization/inventory.
- **Human inputs:** Power mode, benchmark duration, profile and acceptance gates.
- **Local work:** Analyze returned telemetry and verify evidence archive.
- **Colab work:** None.
- **Jetson work:** Warm-up, synchronized stage/end-to-end timing, memory, power/energy
  when reliable, temperature, throttling, sustained prerecorded-stream run.
- **Expected outputs:** Profile benchmark, telemetry, environment and engine manifests.
- **Tests:** Timestamp/synchronization, reset, telemetry parsing, artifact hashes.
- **Acceptance:** Human accepts one selected core pipeline; broader A/B/C is aggressive.
- **Stop conditions:** OOM, thermal instability, stale engine, unsafe device mutation,
  or unreliable timing.
- **Fallback:** Approved smaller profile/model; no silent resolution/metric change.
- **Compute estimate:** 0 GPU hours; separately reported Jetson device-hours.
- **Storage estimate:** Under 50 GiB external benchmark/video/log evidence.
- **Commit boundary:** Deployment scripts/tests/sanitized summaries; engines external.
- **Next task:** `EG-DEMO-001`.

### EG-DEMO-001 — Streamlit prerecorded-stream dashboard

- **Objective:** Demonstrate the selected deployed pipeline and browse all model
  results without contaminating benchmark evidence.
- **Dependencies:** `EG-JETSON-001`, approved demo videos, results registry records.
- **Human inputs:** Presentation flow, live-backend whitelist, fallback assets.
- **Local work:** Frontend, structured IPC, results-only cards, unit/UI smoke tests.
- **Colab work:** None.
- **Jetson work:** Run verified backend; Streamlit may run locally or on approved host.
- **Expected outputs:** Dashboard, prerecorded demo, fallback recording/screenshots.
- **Tests:** Backend identity, live-switch whitelist, missing telemetry, UI/backend
  timing separation, offline fallback.
- **Acceptance:** Human accepts demonstration and disaster chain.
- **Stop conditions:** Unverified model offered live, UI FPS relabeled as inference,
  secret/path leakage.
- **Fallback:** Results-only cards and prevalidated recording/static screenshots.
- **Compute estimate:** 0–1 GPU hours; Jetson device time separate.
- **Storage estimate:** Up to 50 GiB external demo media; no video in Git.
- **Commit boundary:** UI/backend/tests/docs; media external.
- **Next task:** `EG-THESIS-002`.

### EG-THESIS-002 — Evidence reconciliation and thesis closure

- **Objective:** Reconcile every claim with accepted experiments, negative results,
  figures, and reproducibility artifacts.
- **Dependencies:** Week-5 core evidence and explicit holdout/sealed decisions.
- **Human inputs:** Claim interpretation, title approval status, thesis/presentation
  format, sealed-test unlock.
- **Local work:** Claim matrix updates, figures, references, archive verification.
- **Colab work:** Only approved missing report generation; no new exploratory tuning.
- **Jetson work:** Only explicitly approved missing benchmark evidence.
- **Expected outputs:** Thesis evidence index, figures/tables, limitations, final archive.
- **Tests:** Hash/link/claim checks, no invented values, secret/path/binary scans.
- **Acceptance:** Human confirms every claim is supported or explicitly withdrawn.
- **Stop conditions:** Missing provenance, leakage, unresolved title/data permission.
- **Fallback:** Narrow unsupported claims and document limitations.
- **Compute estimate:** Small targeted remainder only; no unbounded queue.
- **Storage estimate:** Presentation/thesis/evidence under Drive hierarchy with backup.
- **Commit boundary:** Documentation and sanitized evidence only.
- **Next task:** Human submission/release decisions.
