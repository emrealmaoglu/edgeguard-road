# Data Catalog, Roles, and Ontology

## Catalog rules

No entry in this document authorizes acquisition. Every dataset requires a scientific
role, access/terms decision, storage estimate, acquisition task, manifest, and linked
experiment. Planning reserves below are capacity envelopes, not measured archive
sizes. Actual filename, byte size, SHA-256, source, access date, and terms reference
replace estimates at acquisition.

## Dataset catalog

| Dataset | Frozen/project role | Intended use | Planning storage reserve | Current access and human gate |
| --- | --- | --- | ---: | --- |
| Cityscapes Fine train | `id_train` split into future `train_fit`, `train_select`, `train_calibration` | Five-model semantic training, HPO, calibration | Up to 25 GiB archive+working reserve | Terms previously accepted for val work; train acquisition and split freeze still require human approval |
| Cityscapes Fine official val | `id_final_confirmation` | One frozen confirmation per final semantic checkpoint | Existing external val footprint; no duplicate | 500-pair manifest exists with SHA-256 `7e91ab791d1814aa355b9ff3a765697fed9d56897e9aff6aa74463501b84f852`; future tuning use prohibited |
| Cityscapes Coarse / trainextra | `id_train_extra` | Aggressive coarse-to-fine experiment only | Up to 100 GiB reserve | Not acquired; separate terms, class mapping, cost, and experiment approval required |
| BDD100K detection packages | `det_train` / `det_select` / frozen confirmation candidate | Two detector families under one known-object ontology | Up to 350 GiB reserve | Not acquired; package/version/terms and group split require human approval |
| Fishyscapes Static | `ood_development` | Zero-shot OOD development, normalization, HPO, calibration impact, trainable-method selection | Up to 50 GiB generated reserve | Not generated; official generator commit/version and legal Cityscapes inputs require approval |
| Fishyscapes Lost & Found full validation | `ood_frozen_holdout` | One-time frozen OOD holdout after all relevant decisions freeze | Up to 10 GiB reserve | Manual-only adapter exists; annotation and underlying image provenance/terms remain separate human gates; no real inference yet |
| Project synthetic outlier exposure | `ood_train` | Trainable anomaly/OE method | Up to 300 GiB, shardable | Not generated; source-object terms, generator config, ontology, and manifest schema require approval |
| SOS or approved temporal dataset | `temporal_unseen` | Temporal persistence and event-level generalization | Up to 100 GiB reserve | SOS role evidence exists; actual acquisition and sequence protocol not approved |
| SMIYC RoadObstacle21 | `sealed_final` | Human-triggered final open-set evaluation | No working reserve before gate | Must not be opened, downloaded, configured, manifested, or debugged during development |
| SMIYC RoadAnomaly21 | `sealed_final` | Human-triggered final open-set evaluation | No working reserve before gate | Same sealed boundary as RoadObstacle21 |
| Mapillary Vistas | `optional_unseen` / stretch | Broader semantic-domain or pretraining ablation | Up to 500 GiB reserve | Not reviewed or acquired; stretch only |
| Approved public prerecorded road videos | `demo_only` | Streamlit demonstration and qualitative plumbing | Up to 50 GiB reserve | Each video needs source, terms, hash, frame/codec metadata, and human approval; never scientific test by default |

The previous Cityscapes campaign remains a measured historical baseline. Its former
`semantic_development` use does not authorize future trial tuning on official val.

## Canonical ontology

### Semantic ontology

The primary semantic contract is Cityscapes 19 train IDs:

```text
0 road, 1 sidewalk, 2 building, 3 wall, 4 fence, 5 pole,
6 traffic light, 7 traffic sign, 8 vegetation, 9 terrain, 10 sky,
11 person, 12 rider, 13 car, 14 truck, 15 bus, 16 train,
17 motorcycle, 18 bicycle; ignore=255
```

Native dataset labels are preserved; mappings into this ontology are explicit,
versioned, and tested. A missing or unmapped label is not background.

### Known-object detection ontology

Core known-road-object classes are:

```text
person, rider, car, truck, bus, train, motorcycle, bicycle,
traffic_light, traffic_sign
```

BDD100K-to-canonical mapping is frozen only after category/count inspection.
Infrastructure classes may support context but are not automatically treated as
hazard objects. Detector confidence is not OOD confidence.

### OOD and risk ontology

- Pixel OOD labels: `0=id`, `1=anomaly`, `255=ignore`.
- “Unknown” is an evidence state, not a new semantic class unless a later trainable
  contract explicitly defines it.
- Operational risk labels `low`, `medium`, and `high` carry component evidence and
  are not physical collision probabilities.
- Relative proximity is `near`/`medium`/`far` or continuous relative depth; metric
  distance is stretch and requires calibrated ground truth.

## Split policy

### Cityscapes Fine train

`EG-DATA-002` first inventories city/sequence groups, frame counts, and pixel class
distributions. It then produces two or three deterministic candidate allocations for
`train_fit`, `train_select`, and `train_calibration` with:

- no group/sequence overlap;
- class and rare-class coverage diagnostics;
- group/sample/pixel counts;
- divergence from the full training distribution;
- stable algorithm/config hash and root-free manifests.

No fixed `80/10/10` split is assumed. A human freezes one candidate before semantic
training begins.

### OOD and temporal data

- Synthetic project data is the only OOD training source in the current plan.
- Fishyscapes Static is reusable development/HPO data.
- Lost & Found is not split for routine tuning. Its full manifest is opened once only
  after commit, configs, models, calibration/threshold policy, and evaluator freeze.
- SMIYC remains unavailable until the human sealed-final event.
- Temporal datasets split by complete sequence, never frame.

## Private storage hierarchy

The compatible logical Google Drive root is private and runtime-supplied:

```text
EdgeGuard/
  datasets/
    archives/
    extracted/
    manifests/
    generated/
  checkpoints/{segmentation,detection,ood,depth}/
  experiments/{segmentation,detection,ood,calibration,deployment}/
  hpo/
  exports/{onnx,tensorrt_metadata}/
  evidence/
  presentation/
  thesis/
  backups/
```

- Approximately 5 TB private Drive capacity is available, but acquisition remains
  demand-driven.
- The local M1 has approximately 18 GiB free and must not hold expanded datasets or
  act as a transfer relay.
- Colab copies/extracts the active subset or deterministic shard into
  `/content/edgeguard-work/<run_id>` and trains from ephemeral storage.
- Canonical archives are immutable. Extracted data is shared, not duplicated by
  experiment. Training checkpoints, optimizer state, logs, studies, exports, and raw
  outputs remain outside Git.
- At least one verified backup protects project-owned checkpoints and experiment
  registries that cannot be reproduced cheaply.

## Manifest and provenance requirements

Every archive record contains:

- dataset/version/role, filename, byte size, SHA-256, official source, access date,
  terms/license reference, acquisition approver, and immutable storage key.

Every extracted/generated manifest contains:

- archive/input identities; generator commit and config hash when applicable;
- root-free relative paths; deterministic ordering; sample/group IDs; label presence;
  geometry; per-file or approved shard hashes; counts; duplicates/missing/corrupt
  summaries; ontology/mapping version; split role; manifest SHA-256.

Every experiment references manifest hashes rather than private absolute roots. Git
may store sanitized manifests and small summaries, but never datasets, large
checkpoints, ONNX/TensorRT binaries, videos, logits, or large logs.
