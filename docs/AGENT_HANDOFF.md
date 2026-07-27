# Agent Handoff

- **From:** Codex
- **To:** Human project owner
- **Repository root:** `.`
- **Branch:** `feat/first-vertical-slice`
- **Base commit:** `ee4460bda9b518a4e784cd43ad23d043ad15cd7b`
  (EG-THESIS-001, pushed and remotely verified)
- **Task:** `EG-DATA-001 — Storage, access and ontology gate`
- **State:** Locally implemented and tested; human review pending
- **Excluded:** No dataset download/extraction/inspection, Drive folder creation,
  framework installation, Colab training, SMIYC access, Jetson action, stage,
  commit, or push

## Storage contract

The proposal uses `EDGEGUARD_EXTERNAL_ROOT` as the runtime-only anchor for the
`EdgeGuard/` project root; no absolute private path is committed. Existing
`private_inputs/` remains a legacy/current child, not the canonical project root.
Relative siblings cover immutable archives, shared extracted datasets, manifests,
generated OOD data, segmentation/detection/OOD checkpoints, HPO, experiments, ONNX
exports, Jetson evidence, presentation/thesis assets, and verified backups.

No part of that hierarchy was created. After approval, Colab stages only the active
subset/shard in `/content/edgeguard-work/<run_id>/`, trains from ephemeral storage,
and atomically syncs hash-verified recovery/final outputs to Drive. The approximately
18 GiB-free Mac is excluded from heavy-data relay/storage duties; approximately 5 TB
private Drive remains canonical capacity.

Migration, reuse, or retention of existing `private_inputs/` files is decided during
EG-DATA-002 or the first relevant acquisition task. No file was moved or copied.

Lifecycle rules cover immutable archive identity, shared extraction, periodic
identity-protected `last` and metric-selected `best` checkpoints, final transfer hash
verification, an independently verified backup for irreplaceable project-owned
checkpoints/registries, and bounded failed-HPO retention without deleting audit or
negative-result evidence.

## Dataset access result

The access matrix now records role, official/manual source, human account/terms gate,
expected packages, planning storage, relative destination, allowed/prohibited uses,
redistribution boundary, manifest, next task, and current status for:

- Cityscapes Fine and Coarse/trainextra;
- BDD100K detection;
- Fishyscapes Static and full Lost & Found;
- project synthetic OOD sources;
- SOS or an approved temporal alternative;
- optional Mapillary Vistas;
- demo-only prerecorded videos;
- SMIYC RoadAnomaly21 and RoadObstacle21.

No dataset is acquired. Full Lost & Found remains a one-time frozen holdout. Both
SMIYC datasets have no destination, loader, config, manifest, inspection, or debug
path before a human sealed-final event.

## Ontology result

`configs/dataset/ontology_v1.yaml` is the single compact provisional source for:

- `semantic_cityscapes19`: exact train IDs `0..18`, ignore `255`;
- `known_detection10`: person, rider, car, truck, bus, train, motorcycle, bicycle,
  traffic_light, traffic_sign at IDs `0..9`;
- `ood_binary`: `0=id`, `1=anomaly`, ignore `255`;
- `risk_operational`: `0=low`, `1=medium`, `2=high`.

BDD100K mappings explicitly implement pedestrian→person, motor→motorcycle,
bike→bicycle, traffic light/sign normalization, and the seven identity mappings.
Source IDs remain `null` rather than invented because the reviewed annotation
contract is name-keyed; the real package must confirm this. Unknown source classes
are rejected, ignored/unsupported lists are explicitly empty, and Mapillary is
deferred until its real label specification is reviewed.

The narrow validator rejects duplicate project IDs/names, unknown actions, duplicate
source mappings, incomplete fixed namespaces, mismatched targets, duplicate YAML
keys, and absolute/user-relative paths. It does not create a registry or plugin
system.

Validation does not freeze this ontology. Its machine-readable status remains
`provisional` until human acceptance.

## Cityscapes split-analysis handoff

EG-DATA-002 must inventory authorized Fine train data by city/sequence, frame and
pixel class counts, rare-class coverage, and full-distribution deviation. It then
proposes two or three deterministic group-atomic candidates for `train_fit`,
`train_select`, and `train_calibration`. No percentage is pre-frozen; the human
selects one candidate only after the report. Official val is
`official_val_common_eval`: it is excluded from routine HPO, `train_select`, and
temperature fitting and used for common final-model evaluation. It is not sealed or
previously unseen because all 500 images were already evaluated and selected results
and visualizations were inspected. The existing PIDNet-S measurement remains a
historical baseline; SMIYC is the actual sealed-final boundary. Groups cannot cross
roles, and Cityscapes test labels are never used.

## Human gates

- Approve the `EdgeGuard/` root convention and runtime variable; decide later how
  existing `private_inputs/` files are reused or migrated.
- Approve Cityscapes Fine train access/terms and exact archive identities.
- Approve BDD100K access/terms, package version, and split basis.
- Approve Fishyscapes sources and separate Lost & Found annotation/image terms.
- Select legal synthetic object-mask sources.
- Select SOS or another temporal dataset.
- Decide whether YOLO AGPL obligations are acceptable.
- Resolve the proposed title/university process.
- Treat accelerator availability as unknown until measured.
- Measure Jetson storage in a separately authorized task.
- Accept or revise provisional `edgeguard-ontology-v1`, then authorize the first
  acquisition task.

## Next task and publication

The exact next implementation task is `EG-DATA-002 — Cityscapes Fine train
preparation`, gated by the applicable human approvals. `EG-DET-001` remains blocked
on BDD100K and YOLO decisions. Neither task was started.

- Final validation: Ruff check passed; Ruff format check passed for 100 files; mypy
  passed for 28 source files; pytest passed 183 with 2 expected opt-in skips;
  ontology tests passed 9/9; `git diff --check` passed
- Repository safety: all 3 notebooks parsed and 3 focused notebook tests passed;
  changed-file path, secret, >1 MiB, binary, and forbidden-artifact scans were clean
- Ontology canonical payload SHA-256:
  `24fd17b54a4aa461e004eaf8c5feebe7b3115c0906559ff24f3bd7f2e1510a10`
- Current changes: unstaged
- Publication: no commit or push

## Changed-file inventory

- Runtime/config: `.env.example`, `configs/dataset/README.md`,
  `configs/dataset/ontology_v1.yaml`
- Validation code/tests: `src/edgeguard/data/ontology.py`,
  `tests/unit/test_ontology.py`
- Contracts/planning: `docs/DATA_CATALOG.md`, `docs/MASTER_PLAN_V2.md`,
  `docs/EXPERIMENT_PROTOCOL.md`, `docs/DECISIONS.md`, `docs/TASKS.md`
- Agent memory/audit: `docs/PROJECT_STATE.md`, `docs/AGENT_HANDOFF.md`,
  `docs/AI_USAGE_LOG.md`
