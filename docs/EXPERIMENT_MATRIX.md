# Experiment Matrix

## Status vocabulary

`planned` → `implemented` → `locally_tested` → `colab_measured` or
`jetson_measured` → `human_accepted`. `blocked` and `failed` preserve explicit
negative state. A later state is never inferred from code or a completed notebook cell.

## Completed foundation and measured baseline

| Experiment ID | Model/task | Data | Resolution/init | Metrics/output | Dependency | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `EGX-BASE-PIDNET-SPIKE-001` | PIDNet-S strict checkpoint and forward | Approved upstream sample | `512×1024`, restricted pretrained checkpoint | Native/aligned logits, repeat diagnostics | Approved checkpoint and pinned upstream | `human_accepted` as plumbing evidence only |
| `EGX-BASE-PIDNET-CITYSCAPES-001` | PIDNet-S single-scale semantic evaluation | Cityscapes official val, 500 images | `1024×2048`, restricted pretrained checkpoint | mIoU `0.7875813077220126`, pixel accuracy `0.9619008903101843`, mean class accuracy `0.8618737663500519`, 500/500 success | Commit C and manifest `7e91ab…f852` | `human_accepted` within documented claim boundary |
| `EGX-OOD-METRIC-FOUNDATION-001` | Pixel AP/FPR95 implementation | Synthetic unit fixtures | N/A | Perfect/reversed/ties/imbalance/void/undefined cases | EG-OOD-001 | `locally_tested` |
| `EGX-OOD-LAF-ADAPTER-001` | Manual Lost & Found pairing/manifest contract | Synthetic fixtures only | Native geometry | Root-free manifest and mask contract | EG-OOD-001 | `locally_tested`; no real data |
| `EGX-DATA-CS-FINE-PREP-001` | Fine-train archive, mapping, analysis, and split-candidate preparation | Synthetic ZIP fixtures only | Native label geometry; no model | Root-free manifests, class/group summaries, three unapproved candidates | EG-DATA-002 | `locally_tested`; real Drive run pending |

The Cityscapes timing is end-to-end evaluation-pipeline timing, not pure inference or
Jetson FPS. Four uncertainty-score summaries were finite ID-only evidence; no real OOD,
calibration, training, detector, or deployment claim exists.

## Planned semantic sequence

| Experiment family | Models/data | Initialization and resolution | Budget/promotion | Metrics and expected artifacts | Dependency | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `EGX-SEG-SMOKE-{FASTSCNN,BISEV2,PIDNETS,DDRNET23S,SEGFORMERB0}` | Five models, tiny deterministic `train_fit` subset | Declared init; `512×1024` | 1–3 epochs or bounded samples; no claim | Loss/gradient, checkpoint/resume, logits, tiny semantic report | EG-SEG-001, frozen split | `planned` |
| `EGX-SEG-SCREEN-{FASTSCNN,BISEV2,PIDNETS,DDRNET23S,SEGFORMERB0}` | Same five, common train/select data | Same policy; `512×1024` | Common short budget; promote top three | mIoU/class IoU, stability, throughput, memory, curves | All smokes and `EG-COMPUTE-001` | `planned` |
| `EGX-EXPORT-SCREEN-{FASTSCNN,BISEV2,PIDNETS,DDRNET23S,SEGFORMERB0}` | Screened checkpoints, fixed fixture | Native-logit static export | Pass/conditional/fail evidence; informs promotion | ONNX/operator/runtime/numerical report | Screening | `planned` |
| `EGX-SEG-MEDIUM-{01..03}` | Promoted top three | Approved init; `512×1024` | Equal medium budget; promote top two | Project checkpoint, curves, select metrics, compute record | Early export gate | `planned` |
| `EGX-SEG-HPO-{A,B}-T*` | Top two on fit/select | Fixed `512×1024` | Limited pruned trials; no official val | Study snapshot, trial configs, compute/failure ledger | Medium ranking | `planned` |
| `EGX-SEG-FINAL-{01..03}` | Three human-approved final configs | At least one random initialization | Full project-owned run | Final checkpoints, curves, config/data/model hashes | HPO/final config freeze | `planned` |
| `EGX-SEG-FINAL-CONFIRM-{01..03}` | Three frozen final checkpoints | Optional `1024×2048` evaluation | One common official-val evaluation each; not sealed/unseen | Common semantic report and evidence package | Final runs | `planned` |
| `EGX-SEG-RES-ABL-{512,768}` | Selected semantic model | `512×1024` vs `768×1536` | Controlled aggressive ablation | Accuracy/cost/memory comparison | Model freeze | `planned` |

## Planned detector, OOD, fusion, and deployment families

| Experiment family | Task/data | Metrics/output | Dependency | Band | Status |
| --- | --- | --- | --- | --- | --- |
| `EGX-DET-SMOKE-{YOLO11N,RTDETRR18}` | Two detectors on deterministic BDD fixture/subset | Loss/gradient/checkpoint/export plumbing | Ontology and split freeze | Core | `planned` |
| `EGX-DET-SCREEN-{YOLO11N,RTDETRR18}` | Common BDD training/selection protocol | mAP50, mAP50-95, precision, recall, small-object recall, cost | Detector smokes | Core | `planned` |
| `EGX-DET-FINAL-PRIMARY` | Stronger selected detector | Final project checkpoint and frozen metrics | Detector screening | Core | `planned` |
| `EGX-DET-HPO-SECONDARY-*` | Second detector full HPO/final | Study and final comparison | Core primary complete | Aggressive | `planned` |
| `EGX-OOD-ZS-{MSP,ENTROPY,MAXLOGIT,ENERGY}` | Selected semantic logits; Fishyscapes Static dev | AP, FPR95, score distributions; no threshold claim | Semantic candidate outputs and Static manifest | Core | `planned` |
| `EGX-CAL-SEMANTIC-*` | `train_calibration` plus frozen semantic model | ECE/NLL/Brier and OOD before/after | Final semantic model | Core | `planned` |
| `EGX-OOD-SYNTH-*` | Project synthetic outlier exposure | Generator/manifest identity and dev AP/FPR95 | Synthetic data gate | Core | `planned` |
| `EGX-OOD-TRAINABLE-*` | Minimal feature/loss candidates | Static AP/FPR95, semantic retention, export/cost | Winning semantic model | Core | `planned` |
| `EGX-OOD-LAF-HOLDOUT-001` | Full Lost & Found one-time holdout | Frozen AP/FPR95 and failure report | All relevant decisions frozen, human unlock | Core | `planned` |
| `EGX-CONTEXT-ABL-*` | Road/near-road and components | OOD false-positive/retention and region metrics | OOD dev pipeline | Core | `planned` |
| `EGX-TEMP-LITE-*` | Approved sequence data | Event precision/recall, persistence, flicker, cost | Context/components | Core | `planned` |
| `EGX-DEPTH-SPIKE-*` | Relative-depth candidate | Plausibility, export, latency/memory/power | Approved data/model terms | Aggressive | `planned` |
| `EGX-FUSION-RULE-*` | Detector+semantic+OOD+context+temporal | Per-signal ablation and operational-risk explanation | Component freeze | Core | `planned` |
| `EGX-EXPORT-FINAL-*` | Frozen semantic/detector/OOD pipeline | PyTorch/ONNX/TensorRT numerical evidence | Final models | Core | `planned` |
| `EGX-JETSON-PROFILE-{A,B,C}` | Sustained prerecorded-stream benchmark | Stage/end-to-end latency, FPS, memory, power, thermal | Verified TensorRT backends | A/B core-selected; full A/B/C aggressive | `planned` |
| `EGX-DEMO-PRERECORDED-001` | Streamlit dashboard | Verified live backend plus results-only cards | Selected Jetson pipeline | Core | `planned` |
| `EGX-SMIYC-SEALED-*` | Human-triggered sealed final evaluation | Frozen final report | Thesis freeze and explicit human unlock | Core evidence gate | `blocked` by design |

## Queue record required for every real job

Before execution, each concrete experiment expands its family ID and records:

- priority and queue `Q1`–`Q6`;
- exact dataset shard/manifest;
- preferred and acceptable fallback accelerator;
- estimated runtime range updated after the first real run;
- unique checkpoint/resume path and experiment fingerprint;
- dependencies, promotion condition, stop condition, and expected artifacts;
- precision, effective batch, samples, optimizer steps, accelerator-hours,
  interruption overhead, and failed-run compute.

Concurrent jobs may not share a checkpoint directory or an untested mutable HPO
database. Jobs using an unfrozen ontology, split, evaluation protocol, or experiment
identity do not enter a GPU queue.
