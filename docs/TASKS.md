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
| SEG-01 | Five-model one-batch and 50-step smoke | external | Five real CUDA logs and reloadable checkpoints |
| SEG-02 | Five 2,000-step pilots and valid 6,000-step screenings | external | Domain-macro/select tables plus failure records |
| SEG-03 | Early ONNX compatibility and top-two selection | external | Numerical export reports and frozen selection record |
| SEG-04 | Top-two bounded HPO | external | 12 unique/resumable trials per model; no final/external access |
| SEG-05 | Scientific source-composition ablation | external | Cityscapes versus Cityscapes+IDD under equal budget; BDD mirror reported only as provisional audit evidence |
| SEG-06 | CE versus weighted CE | external | Overall and rare-class comparison |
| SEG-07 | Scientific and edge final training | external | Hash-bound final checkpoints; optional extra seeds |

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
| DEMO-01 | Streamlit perception/reliability demo | done | CPU fallback, maps, regions, attention and latency |
| DOC-01 | Fresh-runtime Colab and thesis evidence | ready | Output-free notebooks and measured tables after external runs |

## Conditional detection

`DET-01` is `blocked-by-gate`. Only RTMDet-Tiny may be opened after every charter gate
passes. Until then all detector, temporal, anomaly-head, tracking, and INT8 material is
experimental/legacy and cannot appear as active thesis progress.
