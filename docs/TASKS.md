# Active Task System

This file supersedes the former detection/OOD/temporal campaign graph. Status values
are `done`, `ready`, `external`, `blocked-by-gate`, and `future`.

## Data foundation

| ID | Deliverable | Status | Exit evidence |
| --- | --- | --- | --- |
| DATA-01 | Safe archive preparation for Cityscapes, BDD100K, IDD20K | done | Hash checks, traversal rejection, native/canonical separation, tests |
| DATA-02 | Drive one-file bundles under 175 GiB staging policy | ready | Bundle and receipt hashes from preflight notebook |
| DATA-03 | Full audit, duplicate/leakage checks, frozen source manifests | external | Real Drive/Colab audits and human-reviewed freeze |
| DATA-04 | Rare-five classes and train-fit-only CE weights | external | Hash-bound statistics from all frozen fit roles |

## Model campaign

| ID | Deliverable | Status | Exit evidence |
| --- | --- | --- | --- |
| SEG-01 | Three-model hermetic canary, 50-step smoke and forced resume | external | Two clean Colab lock hashes; finite FP32/FP16 probes; reloadable checkpoints |
| SEG-02 | Three 2,000-step pilots, extension smoke, five 6,000-step screenings | external | Core receipts, domain-macro/select tables and failure records |
| SEG-03 | Early ONNX compatibility and top-two selection | external | Numerical export reports and frozen selection record |
| SEG-04 | Top-two bounded HPO | external | 12 unique/resumable trials per model; no final/external access |
| SEG-05 | Scientific source-composition ablation | external | Cityscapes versus Cityscapes+IDD under equal budget; BDD mirror reported only as provisional audit evidence |
| SEG-06 | CE versus weighted CE | external | Overall and rare-class comparison |
| SEG-07 | Three-finalist scientific and edge final training | external | Hash-bound final checkpoints; isolated weighted-CE ablation |

## Reliability and perception

| ID | Deliverable | Status | Exit evidence |
| --- | --- | --- | --- |
| REL-01 | Global temperature fitting and before/after metrics | ready | Source-calibration artifact and ECE/NLL/Brier tables |
| REL-02 | Source-frozen frame shift alert | done | Four scores, 95th-quantile reference, AUROC/AP/reporting contracts |
| PER-01 | Road and ego-reachable corridor | done | Road IoU, boundary F1, false-drivable and fragmentation code/tests |
| PER-02 | Semantic component regions | done | Region masks/boxes/centroids/confidence/entropy and coverage metrics |
| PER-03 | Explainable operational-attention layer | done | Frozen formula/thresholds, contribution output, deterministic tests |

## Final evaluation and deployment

| ID | Deliverable | Status | Exit evidence |
| --- | --- | --- | --- |
| EVAL-01 | Official source validations and ACDC | external | Frozen-checkpoint domain/class/reliability tables |
| EVAL-02 | MUSES or one WildDash submission | external | Sealed release record and immutable result; KITTI fallback if needed |
| DEP-01 | Static ONNX FP32 validation | ready | Shape/class/finiteness/allclose and ORT latency report |
| DEP-02 | Jetson TensorRT FP16 build | external | Engine/build manifest on target; no overwrite |
| DEP-03 | 25W and MAXN SUPER sustained benchmark | external | 200 warm-up; 5,000 frames or 10 min; telemetry and acceptance result |
| DEMO-01 | Accepted-artifact Streamlit perception/reliability demo | ready | Offline accepted bundle, cache, CPU/missing-artifact and Jetson pages |
| DOC-01 | Fresh-runtime Colab and accepted thesis evidence | external | Application commit and immutable notebook repin complete; remote CI plus real Colab G1/G3 and an accepted run remain required |

## Conditional detection

`DET-01` is `blocked-by-gate`. Only RTMDet-Tiny may be opened after every charter gate
passes. Until then all detector, temporal, anomaly-head, tracking, and INT8 material is
experimental/legacy and cannot appear as active thesis progress.

## Known limitations

- **IDD20K shard packaging drops native labels.** `_publish_idd_shards`
  (`src/edgeguard/data/preparation.py`) renders both the native 40-class
  `_gtFine_labelids.png` mask and the canonical Cityscapes19
  `_gtFine_labelTrainIds.png` mask during staging, but only tars the
  canonical mask into each of the 33 already-staged shards
  (`"prepared_payload": "images_and_cityscapes19_canonical_masks_only"`).
  This means the native IDD label can no longer be audited or re-mapped from
  a staged shard after the fact — a partial exception to
  `src/edgeguard/data/AGENTS.md`'s "preserve every dataset's native labels"
  rule. The Cityscapes19 mapping itself is verified correct
  (`configs/dataset/semantic_ontology_v2.yaml`), so this does not affect
  training correctness; it only limits future native-label auditability.
  Re-processing already-staged shards to include native labels is a real
  IDD-source re-processing job, not a code change, and needs an explicit
  decision before being scheduled (found 2026-08-08, Claude Code review;
  deliberately left unresolved).
