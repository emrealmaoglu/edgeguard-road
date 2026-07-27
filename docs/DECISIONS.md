# Decision Index

| ADR | Decision | Status |
| --- | --- | --- |
| ADR-0001 | Use a `src` package layout | Accepted |
| ADR-0002 | Use lean project tooling | Accepted |
| ADR-0003 | Keep Colab execution-only | Accepted |
| ADR-0004 | Keep Jetson deployment-only | Accepted |
| ADR-0005 | Seal final-test data | Accepted |
| ADR-0006 | Keep runtime artifacts outside Git | Accepted |
| ADR-0007 | Preserve AI/human approval boundaries | Accepted |

New architectural decisions receive the next sequential ADR number. Scientific
results and transient implementation notes do not become ADRs.

## Approved planning constraints pending implementation

- The expanded multi-model direction and proposed titles are recorded in
  `PROJECT_CHARTER.md`; university title approval remains open.
- The authoritative staged program, scope bands, compute queues, storage policy, and
  task graph live in `MASTER_PLAN_V2.md`.
- Dataset roles and ontology live in `DATA_CATALOG.md`; experiment status lives in
  `EXPERIMENT_MATRIX.md`; evidence-to-claim gates live in
  `THESIS_CLAIM_MATRIX.md`.
- These documents do not themselves approve data access, model promotion, thresholds,
  sealed evaluation, or deployment claims, and do not require a new ADR until an
  implementation architecture is actually selected.

## EG-DATA-001 locally implemented contracts pending human freeze

- `EDGEGUARD_EXTERNAL_ROOT` is the proposed runtime anchor for the `EdgeGuard/`
  project root. Existing `private_inputs/` is a legacy/current child; its migration
  or reuse is deferred to EG-DATA-002 or the first relevant acquisition task. The
  relative hierarchy was not created.
- `configs/dataset/ontology_v1.yaml` is the proposed project-specific ontology source
  for `semantic_cityscapes19`, `known_detection10`, `ood_binary`, and
  `risk_operational`.
- BDD100K's reviewed ten source names map explicitly into `known_detection10`;
  unlisted runtime classes are rejected. Numeric source IDs remain unset until the
  real package specification is inspected.
- Mapillary mapping remains deferred. Official Cityscapes val is
  `official_val_common_eval`: common evaluation for frozen final models, but not
  routine HPO, `train_select`, temperature-fitting data, sealed, or previously
  unseen. Lost & Found remains a full frozen holdout, and SMIYC remains sealed.
- These contracts are locally implemented proposals. The ontology remains
  provisional; human approval is still needed for the Drive root, ontology freeze,
  every acquisition, and the Cityscapes split.
