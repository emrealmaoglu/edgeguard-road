# Data Catalog, Storage Contract, Access Gates, and Ontology

## Gate status and existing conventions

This document is a planning and validation contract; it does not authorize or perform
acquisition. At EG-DATA-001 start, the repository used runtime CLI paths and the
empty environment-variable names `EDGEGUARD_PIDNET_CHECKPOINT`,
`EDGEGUARD_PIDNET_CHECKOUT`, and `EDGEGUARD_CITYSCAPES_ROOT`. Dataset manifests were
root-free, runtime artifacts were ignored, and no generic storage abstraction
existed.

EG-DATA-001 adds `EDGEGUARD_EXTERNAL_ROOT` as the single future private-storage
anchor while preserving the narrower variables for existing tools. It must point at
runtime to the human-approved `EdgeGuard/` project root. The existing
`private_inputs/` directory is a legacy/current private input location below that
root; it is not the canonical root for archives, experiments, checkpoints, exports,
or evidence. These are Drive-relative names, not committed absolute paths, and no
directory was created or moved.

Two legacy role values remain intentionally identifiable:

- the completed Cityscapes baseline records its historical
  `id_validation/semantic_development` contract;
- the unexecuted Fishyscapes adapter still emits the old `ood_development` Lost &
  Found role.

They do not override the forward protocol. Official Cityscapes val is now
`official_val_common_eval`: it supports common final-model evaluation but is neither
sealed nor previously unseen. Lost & Found requires a bounded role-label correction
before its one-time holdout run. Historical artifact records are not rewritten.

## Proposed Drive-relative storage contract

All paths below are relative to `EDGEGUARD_EXTERNAL_ROOT`, proposed to resolve to the
private `EdgeGuard/` project root after human approval:

```text
private_inputs/
archives/
  cityscapes/
  bdd100k/
  fishyscapes/
  temporal/
  mapillary/
  demo/
datasets/
  cityscapes/
  bdd100k/
  fishyscapes/
  temporal/
  mapillary/
  demo/
manifests/
  datasets/
  generated/
generated/
  ood/
checkpoints/
  segmentation/
  detection/
  ood/
hpo/
experiments/
  segmentation/
  detection/
  ood/
  calibration/
  deployment/
exports/
  onnx/
evidence/
  jetson/
presentation/
thesis/
backups/
  checkpoints/
  experiment_registries/
```

This hierarchy is proposed only. EG-DATA-001 does not create it in Drive or locally.
TensorRT engines remain device-built and are not canonical cross-device binaries;
only their metadata and Jetson evidence belong under the external root.
Approximately 5 TB of private Drive capacity is available, but capacity alone never
authorizes acquisition.

Migration, reuse, or retention of files already under `private_inputs/` is decided
case by case during EG-DATA-002 or the first relevant acquisition task. No automatic
move, copy, deduplication, or deletion is implied.

## Storage lifecycle

1. **Acquire:** A human-approved task records archive filename, exact byte size,
   SHA-256, official source URL/reference, access date, terms/license reference, and
   approver before the archive becomes canonical.
2. **Freeze archives:** Canonical downloaded archives are immutable. A changed byte
   creates a new identity/version; it never overwrites the recorded object.
3. **Extract once:** A verified archive is extracted into a versioned shared dataset
   location. Experiments reference one deterministic manifest and do not duplicate an
   extracted dataset per run.
4. **Stage actively:** Colab copies or extracts only the required dataset, subset, or
   deterministic shard into `/content/edgeguard-work/<run_id>/`. Mounted Drive is not
   the per-sample training filesystem when measured I/O is inadequate.
5. **Recover:** Expensive jobs write active files locally and periodically sync an
   identity-protected `last` checkpoint plus structured logs to Drive at a bounded,
   configured interval. `best` is updated only on the frozen selection metric.
6. **Finalize:** Resolved config, manifests, metrics, environment, logs, `best`, and
   required final checkpoints sync atomically. Every transferred final artifact is
   re-hashed and compared with its source before completion is recorded.
7. **Back up:** Every irreplaceable project-owned promoted checkpoint and experiment
   registry has at least one independently hash-verified backup. Backup location and
   verification date remain external metadata.
8. **Retain/clean:** Failed HPO trials retain identity, resolved config, failure,
   resource use, final metric, and the minimum diagnostic/recovery checkpoint needed
   for review. Redundant periodic checkpoints, caches, and verbose logs may be removed
   only after bounded retention and human-approved study reconciliation; promoted,
   best, negative-result, and audit records are preserved.
9. **Keep the Mac lean:** The approximately 18 GiB-free local M1 stores code, fixtures,
   small sanitized summaries, and bounded probes only. It is not a relay or primary
   store for heavy datasets, archives, checkpoints, or exports when Drive-to-Colab
   transfer is available.

Git contains code, path-free configs/manifests, small summaries, and documentation.
It never contains datasets, large checkpoints, ONNX/TensorRT binaries, videos, raw
logits, HPO databases, or large logs.

## Dataset access-decision matrix

Storage values are planning envelopes, not measured package sizes. Exact values
replace them only after authorized acquisition.

| Dataset | Role | Acquisition source and human gate | Expected package(s) | Reserve | Canonical Drive-relative destination | Allowed use | Prohibited use | Redistribution boundary | Required manifest | Next task / status |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| Cityscapes Fine | Train: `train_fit`, `train_select`, `train_calibration`; official val: `official_val_common_eval` | Official Cityscapes portal; human account, terms, research-use approval, and archive hash review | `leftImg8bit_trainvaltest.zip`, `gtFine_trainvaltest.zip`; exact downloaded identities required | Up to 25 GiB | `archives/cityscapes/fine/`, `datasets/cityscapes/fine/`, `manifests/datasets/cityscapes/fine/` | Approved train roles; common evaluation of final frozen models on official val | Official val as routine HPO, `train_select`, temperature-fitting/calibration data, sealed holdout, or previously unseen evidence; Cityscapes test labels; cross-role groups | No archive/image/label redistribution | Archive records; native-label inventory; group split manifests; provisional ontology version; official-val manifest | `EG-DATA-002`; historical 500-image PIDNet-S baseline exists, train not acquired in this task |
| Cityscapes Coarse/trainextra | `id_train_extra`, aggressive only | Official Cityscapes portal; separate human terms, role, and experiment approval | Expected official `leftImg8bit_trainextra.zip` and `gtCoarse.zip`; identities must be confirmed | Up to 100 GiB | `archives/cityscapes/coarse/`, `datasets/cityscapes/coarse/`, `manifests/datasets/cityscapes/coarse/` | Approved coarse-to-fine ablation | Core-path dependency or use before Fine baseline freeze | No archive/image/label redistribution | Archive, label-version, mapping, split, and leakage manifests | Later aggressive acquisition task; `not_acquired` |
| BDD100K detection images and labels | `det_train`, `det_select`, frozen detector confirmation | Official BDD100K source; human account/terms, package/version, ontology, and split approval | Official detection image and label packages; exact release filenames pending real source review | Up to 350 GiB | `archives/bdd100k/detection/`, `datasets/bdd100k/detection/`, `manifests/datasets/bdd100k/detection/` | Two-detector common training, selection, and frozen confirmation | Silent class dropping, unfrozen splits, semantic/OOD substitution | No dataset redistribution; only sanitized summaries/mappings | Archive/package records; sequence/group split; class counts; mapping version; geometry/duplicate audit | `EG-DET-001`, blocked on access/terms and YOLO license decision; `not_acquired` |
| Fishyscapes Static | `ood_development` | Official human-reviewed generator pin plus legally available Cityscapes inputs | Generated dataset, not an assumed downloadable ZIP; generator source/config/input identities required | Up to 50 GiB | `generated/ood/fishyscapes_static/`, `manifests/generated/fishyscapes_static/` | Zero-shot development, normalization, HPO, calibration impact, trainable-method selection | Final/holdout claim; redistribution; use as semantic train labels | Generated images remain private; source terms remain attached | Input Cityscapes manifest, generator commit/config hash, output/shard hashes, ontology/mapping version | `EG-OOD-002`; `not_generated` |
| Fishyscapes Lost & Found full validation | `ood_frozen_holdout` | Official Fishyscapes-linked annotations plus underlying Lost & Found provider; separate human terms/provenance approvals | Public validation annotations and matching underlying images; exact package identities pending manual acquisition | Up to 10 GiB | `archives/fishyscapes/lost_and_found/`, `datasets/fishyscapes/lost_and_found/`, `manifests/datasets/fishyscapes/lost_and_found/` | One complete holdout run after model/config/calibration/threshold/evaluator freeze | Routine subsets, HPO, debugging, threshold selection, repeated opening | Annotation terms do not automatically cover images; no image/derived-visual redistribution | Separate source/license/hash records, 100-pair root-free manifest, holdout-open record | Later human holdout gate; adapter fixtures only, no real inference |
| Project synthetic OOD sources | `ood_train` | Human-approved object-mask/source collections and approved generator | Source masks/assets plus generated deterministic shards; exact sources pending | Up to 300 GiB | `archives/fishyscapes/synthetic_sources/` only if applicable; `generated/ood/project_v1/`, `manifests/generated/project_v1/` | Trainable anomaly/OE training only under frozen generator protocol | Holdout/sealed samples as sources; unlicensed assets; unmanifested generation | Each source's terms govern redistribution; generated data stays private by default | Source asset hashes/terms, generator commit/config hash, seed, shard/sample manifests | `EG-OOD-003` prerequisites; source decision pending, `not_generated` |
| SOS or selected temporal dataset | `temporal_unseen` | Official selected provider; human chooses dataset and accepts sequence terms | Exact SOS/alternative package and version pending selection | Up to 100 GiB | `archives/temporal/<approved_dataset>/`, `datasets/temporal/<approved_dataset>/`, `manifests/datasets/temporal/<approved_dataset>/` | Sequence-level temporal development and event evaluation | Frame-level leakage; undeclared substitution; semantic/OOD final reuse | Provider terms control videos/labels; no Git redistribution | Archive/source record, sequence manifest, group roles, codec/frame/label metadata | `EG-TEMP-001`; dataset choice pending, `not_acquired` |
| Optional Mapillary Vistas | `optional_unseen`, stretch | Official Mapillary source; human version/terms/role approval | Exact packages unknown until real label specification review | Up to 500 GiB | `archives/mapillary/<approved_version>/`, `datasets/mapillary/<approved_version>/`, `manifests/datasets/mapillary/<approved_version>/` | Separately approved broad-domain/pretraining ablation | Invented label mapping; Core-path dependency; access before role approval | Provider terms; no dataset redistribution | Source/version, native ontology, human-reviewed mapping, split and archive manifests | Future stretch task; mapping deliberately deferred, `not_acquired` |
| Demo-only prerecorded videos | `demo_only` | Approved public source or user-owned video; per-asset human rights review | Exact media files selected individually | Up to 50 GiB | `archives/demo/`, `datasets/demo/`, `manifests/datasets/demo/` | Streamlit qualitative demonstration and prerecorded plumbing | Scientific test by default; training; redistribution without permission | Per-video terms; derived recordings require separate review | URL/source, access date, SHA-256, size, codec, FPS, frame count, usage scope | `EG-DEMO-001`; no video selected |
| SMIYC RoadAnomaly21 | `sealed_final` | Human-triggered official sealed evaluation only | None defined before the sealed gate | No working reserve | **No destination defined before sealed-final authorization** | One frozen final evaluation after explicit human unlock | Download, path, loader, config, manifest, inspection, debugging, tuning | No redistribution; provider terms apply after authorized access | Defined only inside the future sealed event | Human sealed-final task; `blocked` by design and unaccessed |
| SMIYC RoadObstacle21 | `sealed_final` | Human-triggered official sealed evaluation only | None defined before the sealed gate | No working reserve | **No destination defined before sealed-final authorization** | One frozen final evaluation after explicit human unlock | Download, path, loader, config, manifest, inspection, debugging, tuning | No redistribution; provider terms apply after authorized access | Defined only inside the future sealed event | Human sealed-final task; `blocked` by design and unaccessed |

## Canonical ontology source

The machine-readable provisional source of truth is
`configs/dataset/ontology_v1.yaml`, version `edgeguard-ontology-v1`, with
`ontology_status=provisional`. It is validated by `edgeguard.data.ontology`, remains
independent of private paths, and is not frozen until human acceptance.

### `semantic_cityscapes19`

The existing Cityscapes train IDs are preserved without renumbering:

```text
0 road, 1 sidewalk, 2 building, 3 wall, 4 fence, 5 pole,
6 traffic_light, 7 traffic_sign, 8 vegetation, 9 terrain, 10 sky,
11 person, 12 rider, 13 car, 14 truck, 15 bus, 16 train,
17 motorcycle, 18 bicycle; ignore=255
```

### `known_detection10`

```text
0 person, 1 rider, 2 car, 3 truck, 4 bus, 5 train,
6 motorcycle, 7 bicycle, 8 traffic_light, 9 traffic_sign
```

The explicit BDD100K name mappings are:

```text
pedestrian→person, rider→rider, car→car, truck→truck, bus→bus,
train→train, motor→motorcycle, bike→bicycle,
traffic light→traffic_light, traffic sign→traffic_sign
```

The reviewed annotation contract is name-keyed, so `source_id` remains `null` rather
than inventing numeric IDs. The real package audit must confirm that assumption.
Unlisted runtime classes are rejected. The ignored and unsupported lists are
explicitly empty for this ten-name contract; no class may be silently dropped.
Mapillary mapping remains a future human-reviewed task after its exact label
specification and version are known.

### `ood_binary` and `risk_operational`

- `ood_binary`: `0=id`, `1=anomaly`, `ignore=255`.
- `risk_operational`: `0=low`, `1=medium`, `2=high`.

“Unknown” is evidence, not a new semantic class. Risk labels are explainable
operational categories, not physical collision probabilities. Namespace IDs are
local to each namespace and must not be joined by numeric equality.

The validator rejects duplicate project IDs/names, unknown mapping actions,
duplicate source mappings, incomplete fixed namespaces, incorrect target ID/name
pairs, and absolute/user-relative paths.

## Cityscapes split-analysis protocol for EG-DATA-002

No percentage is frozen in EG-DATA-001. After authorized Fine-train acquisition,
EG-DATA-002 must:

1. verify archive identity and build a native, root-free inventory;
2. parse city, sequence, and frame identifiers and treat each sequence group as
   atomic;
3. count frames and pixel-level `semantic_cityscapes19` frequencies per group and
   globally, preserving ignore pixels;
4. identify rare-class coverage and groups that uniquely carry rare classes;
5. generate two or three deterministic candidate allocations for `train_fit`,
   `train_select`, and `train_calibration` without assuming `80/10/10` or another
   fixed percentage;
6. report each candidate's groups/samples/pixels, per-class coverage, rare-class
   coverage, and deviation from the full training distribution;
7. hard-fail any group overlap or unmapped/invalid label condition;
8. let the human select and freeze exactly one candidate plus its algorithm/config
   and manifest hashes.

Official Cityscapes val is `official_val_common_eval`. It is not routine HPO,
`train_select`, or temperature-fitting/calibration data. It is used for common
evaluation of final frozen models, but it is not a sealed or previously unseen
holdout because all 500 images were already evaluated and selected results and
visualizations were inspected. The existing PIDNet-S result remains a historical
measured baseline. SMIYC remains the actual sealed-final boundary. Cityscapes test
labels are never used. Until an internal split candidate is human-frozen, no semantic
training task may begin.

## Human decision checklist

- [ ] Approve `EdgeGuard/` as the project root addressed at runtime by
  `EDGEGUARD_EXTERNAL_ROOT`, with existing `private_inputs/` retained as a
  legacy/current child pending task-specific migration/reuse decisions.
- [ ] Confirm Cityscapes Fine train access/terms and exact archive identities.
- [ ] Confirm BDD100K access/terms, package version, and group-split basis.
- [ ] Confirm Fishyscapes Static generator/source acquisition and separate Lost &
  Found annotation/image terms.
- [ ] Select and approve legal synthetic object-mask sources.
- [ ] Select SOS or another temporal dataset and approve its sequence protocol.
- [ ] Decide whether the proposed YOLO path's AGPL obligations are acceptable before
  `EG-DET-001` chooses a framework/source.
- [ ] Approve or reject the proposed thesis title through the university process.
- [ ] Treat T4/L4/A100/H100 as availability assumptions until Colab throughput is
  measured.
- [ ] Measure and approve Jetson storage inventory in its separately authorized task.

Ontology validation is complete locally, but `edgeguard-ontology-v1` remains
provisional until human acceptance. Dataset acquisition and split freeze remain
blocked until their applicable human decisions close.
