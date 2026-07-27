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

## EG-DATA-001 remotely verified contracts

- `EDGEGUARD_EXTERNAL_ROOT` is the human-approved runtime anchor for the `EdgeGuard/`
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
- EG-DATA-001 is remotely verified at
  `0114d4e4778c1d6e53b6359e0a11f71eb15d2fb4`. The ontology remains provisional;
  validation does not freeze it or a Cityscapes split.

## EG-DATA-002 local preparation architecture

- The existing `scripts/prepare_cityscapes.py` is extended with a project-specific
  `--split train` path; its existing val path remains available. No generic dataset
  or extraction framework is introduced.
- The train path verifies the two pinned archive hashes, validates every ZIP member,
  extracts only Fine `train` RGB and native `labelIds` into Colab-local staging,
  generates deterministic `0..18/255` masks from the shared LUT, analyzes masks
  streaming, and promotes only a validated incoming tree.
- Every generated mask is hashed. Original RGB/label integrity is anchored to the
  immutable source-archive hashes and root-free sample paths; measured mask-hashing
  time is reported rather than assumed.
- `CSF-SPLIT-A/B/C` are deterministic bounded candidates near `85/10/5`, `80/15/5`,
  and `90/5/5`. The heuristic records sample deviation, class-presence coverage,
  pixel-distribution divergence, and rare-class absence. Its recommendation is not
  an approval or freeze.
- The local runtime cannot access the approved private Drive mount. No real archive,
  dataset, class frequency, group count, or split result exists yet.

## EG-SEG-001 framework proposal pending Colab evidence

- The narrow proposal is MMSegmentation `v1.2.2` at exact source commit
  `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8`, with MMEngine `0.10.7`, MMCV `2.1.0`,
  and OpenMIM `0.3.9`. The upstream-declared compatible ranges are preserved.
- The repository does not add Torch, CUDA, MMSegmentation, MMEngine, MMCV, or OpenMIM
  to core dependencies. Colab's installed Torch/CUDA identity is inspected first;
  OpenMIM selects a compatible runtime build without a hard-coded CUDA wheel URL.
- This is an implementation proposal, not an accepted framework decision. The pin
  remains `proposal_pending_colab_probe` until all five models pass construction,
  synthetic forward/backward, native-logit validation, and checkpoint resume on the
  exact reviewed project commit.
- Configs are independent YAML documents. No inheritance, registry platform,
  database tracker, or general framework abstraction is introduced.
- Fast-SCNN and BiSeNetV2 declare random project-training baselines. PIDNet-S,
  DDRNet-23-Slim, and SegFormer-B0 pretrained sources remain explicit unresolved
  human inputs. The historical PIDNet Cityscapes checkpoint is not a project-training
  initialization.
- EG-DATA-002 outputs remain immutable. A separate human acceptance record binds one
  candidate and its dataset/split hashes before any real training can start.
