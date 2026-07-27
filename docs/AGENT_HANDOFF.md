# Agent Handoff

- **From:** Codex
- **To:** Human project owner
- **Repository root:** `~/Projects/edgeguard-road`
- **Branch:** `feat/first-vertical-slice`
- **Base commit:** `345d9fd1dcff0a7aa9c54c6f3929c2c751c24c7c`
  (Commit D, pushed and remotely verified)
- **Task:** `EG-THESIS-001 — Scope, title and claim migration`
- **State:** Documentation migration implemented, locally validated, and accepted
  by the human owner for one coherent commit; remote CI verification pending
- **Excluded:** No dataset acquisition, framework installation, training, HPO,
  Fishyscapes inference, SMIYC access, model/checkpoint creation, ONNX/TensorRT
  export, Jetson modification, CI expansion, commit, or push

## Identity and scope migration

The recommended title is:

> EdgeGuard-Road: Multi-Model Open-Set Road Safety Perception with Object
> Detection, Semantic Segmentation, Uncertainty Calibration and Temporal Risk
> Fusion on Resource-Constrained Edge Devices

The recommended Turkish title is:

> Kaynak Kısıtlı Uç Cihazlarda Çok Modelli Açık Küme Yol Güvenliği Algısı:
> Nesne Tespiti, Semantik Bölütleme, Belirsizlik Kalibrasyonu ve Zamansal Risk
> Füzyonu

Both remain proposals pending human and university approval. The prior title is
retained as the fallback:

> EdgeGuard-Road: Uncertainty-Calibrated Open-Set Road Hazard Segmentation with
> Contextual and Temporal Risk Analysis on Resource-Constrained Edge Devices

The documentation now separates:

- **Core:** Five-model semantic screening, three final project-owned semantic
  checkpoints, limited top-two HPO, two-detector comparison with one full final
  detector, zero-shot OOD, calibration, one trainable anomaly method, context,
  lightweight temporal processing, selected Jetson pipeline, and Streamlit.
- **Aggressive:** Secondary detector HPO, coarse-to-fine training, depth
  integration, extra seeds, complete A/B/C deployment profiles, and broader
  ablations.
- **Stretch:** Mapillary, INT8, metric distance, learned risk fusion, and advanced
  tracking.

## Preserved evidence and boundaries

- `EG-OOD-001` is complete and remotely verified; Python 3.10 and 3.11 GitHub
  Actions jobs passed at Commit D.
- The full Cityscapes run remains 500/500 with zero failures: mIoU
  `0.7875813077220126`, pixel accuracy `0.9619008903101843`, and mean class
  accuracy `0.8618737663500519`.
- The run used clean Commit C
  `aa8803e8060af8cd704f81fb7c6903d0d48e2a6e`; external ZIP SHA-256 is
  `756abf1a983b8eed11b22f0c10b3cabf093d6e614a4bec2a6d223c41202132b7`.
- Four uncertainty score implementations and the manual-only Fishyscapes
  adapter/AP/FPR95 foundation are preserved.
- No real Fishyscapes inference, model training, detector experiment,
  calibration, temporal experiment, export validation, or Jetson benchmark has
  occurred.
- Official Cityscapes val is reserved for future frozen final confirmation.
  Fishyscapes Static is the future OOD development/HPO source; complete Lost &
  Found validation is a one-time frozen holdout; SMIYC remains sealed final.

## New documentation structure

- `docs/MASTER_PLAN_V2.md`: Scope bands, Week 3–6 schedule, model/detector
  portfolios, compute queues, storage policy, task graph, and risks.
- `docs/SYSTEM_ARCHITECTURE.md`: Planned training and runtime component flow,
  output contracts, deployment handoff, and benchmark/UI separation.
- `docs/DATA_CATALOG.md`: Dataset roles, ontology, access gates, Drive policy,
  manifests, and provenance.
- `docs/EXPERIMENT_MATRIX.md`: Stable experiment families with evidence-aware
  statuses.
- `docs/THESIS_CLAIM_MATRIX.md`: Claim-to-experiment evidence requirements and
  current status.

The charter, README, work packages, experiment protocol, decisions, tasks, state,
and historical research documents were migrated without upgrading planned work
to implemented status.

## Task graph and resources

The critical semantic path is:

`EG-DATA-001 → EG-DATA-002 → EG-SEG-001 → EG-SEG-002 →`
`EG-COMPUTE-001 → EG-SEG-003 → EG-EXPORT-001 → EG-SEG-004 → EG-SEG-005 →`
`EG-SEG-006`

Detector, OOD/calibration, context/temporal/fusion, deployment, and thesis tracks
branch only after their dependencies and human gates are satisfied. Depth is an
aggressive, non-blocking branch.

Private Google Drive has approximately 5 TB available for canonical archives,
datasets, checkpoints, experiments, exports, and evidence. The Mac has only
approximately 18 GiB free. Active Colab work must use ephemeral local storage and
sync verified results back to Drive; accelerator assignments and the provisional
70–130 GPU-hour required queue must be recalibrated from measured throughput.

## Immediate next action and human gates

After accepting this diff, the one next task is `EG-DATA-001 — Storage, access
and ontology gate`. It must inspect existing private Drive naming before proposing
or creating a final structure. It may not acquire data until the relevant human
terms and role decisions are approved.

Still unresolved:

- proposed title and university approval process;
- compatible private Drive root/naming convention;
- Cityscapes/BDD100K/Fishyscapes/SOS/optional Mapillary access decisions;
- canonical semantic and detector ontology;
- distribution-audited Cityscapes train split;
- training framework and model source pins;
- HPO/compute promotion budgets;
- calibration and threshold protocol;
- one-time Lost & Found holdout and sealed SMIYC opening gates;
- Jetson access, deployment profiles, and artifact promotion.

## Validation and publication

- Local quality gate: Ruff check passed; Ruff format check passed for 98 files;
  mypy passed for 27 source files; pytest passed 174 with 2 expected opt-in skips;
  `git diff --check` passed
- Repository safety: all 3 notebooks parsed and 3 focused notebook integration
  tests passed; changed-file path, secret, >1 MiB, binary, and forbidden-artifact
  scans were clean
- Human review: the complete 17-file documentation block is accepted for commit
- Publication gate: one documentation-only commit and push to
  `origin/feat/first-vertical-slice` are authorized; resulting SHA and CI status
  must be taken from Git/GitHub after execution

## Changed-file inventory

- Identity: `PROJECT_CHARTER.md`, `README.md`
- Planning: `docs/MASTER_PLAN_V2.md`, `docs/WORK_PACKAGES.md`,
  `docs/EXPERIMENT_PROTOCOL.md`, `docs/DECISIONS.md`
- Architecture/data/experiments/claims: `docs/SYSTEM_ARCHITECTURE.md`,
  `docs/DATA_CATALOG.md`, `docs/EXPERIMENT_MATRIX.md`,
  `docs/THESIS_CLAIM_MATRIX.md`
- Research role migration: `docs/research/FISHYSCAPES_DEVELOPMENT_PLAN.md`,
  `docs/research/WP03_WP05_SOURCE_REVIEW.md`,
  `docs/research/CITYSCAPES_FULL_VAL_EVIDENCE.md`
- Agent memory and execution: `docs/TASKS.md`, `docs/PROJECT_STATE.md`,
  `docs/AGENT_HANDOFF.md`, `docs/AI_USAGE_LOG.md`
