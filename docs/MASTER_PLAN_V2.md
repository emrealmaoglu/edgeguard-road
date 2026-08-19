# EdgeGuard-Road Master Plan V2

> **Experimental/legacy plan.** The human-approved rescue scope is now the
> semantic-first path in `docs/SEMANTIC_FIRST_RUNBOOK.md`. Detection, temporal,
> anomaly-head, broad HPO, depth, and fusion tasks below are not delivery-critical.

## Executive decision

The expanded project is technically coherent and thesis-level if it remains staged,
uses shared contracts, and preserves strict data/claim boundaries. It is feasible from
Week 3 only with parallel local/documentation and accelerator queues, interruption-safe
Colab jobs, early export checks, and promotion gates that stop low-value runs.

The largest combined risk is schedule loss from data acquisition, I/O, framework
compatibility, or export failure discovered too late. The mitigation is to measure the
pipeline before allocating the full queue, keep Cityscapes val out of tuning and OOD holdouts frozen,
and separate scientific merit from deployment eligibility.

The existing PIDNet-S/Cityscapes result is the first measured baseline, not evidence
that the expanded system, training program, OOD claims, or Jetson target is complete.

## Scope bands

| Band | Included work |
| --- | --- |
| Core | Five semantic model screening; three final project-owned semantic checkpoints; limited top-two HPO; two detector comparison with at least one full final detector; zero-shot OOD; calibration; one trainable anomaly method; road context; connected components; lightweight temporal persistence; selected Jetson pipeline; Streamlit prerecorded-stream dashboard |
| Aggressive | Second detector full HPO; coarse-to-fine semantic training; depth integration; extra seeds; complete A/B/C deployment profiles; broader OOD, context, temporal, and fusion ablations |
| Stretch | Mapillary Vistas; INT8; metric distance; learned risk fusion; advanced tracking |

Depth is never a dependency of the core segmentation+detection+OOD path. Failed
training or export remains reportable evidence; it does not silently disappear.

## Model portfolios

### Semantic segmentation

| Candidate | Initialization plan | Role and deployment question |
| --- | --- | --- |
| Fast-SCNN | At least one random-init final candidate if promoted | Lightweight lower-bound and edge throughput |
| BiSeNetV2 | Pretrained when terms/provenance permit, then fine-tuned | Dual-path real-time candidate |
| PIDNet-S | Existing pretrained baseline plus project training if approved | Continuity with measured baseline and boundary-aware design |
| DDRNet-23-Slim | Pretrained when approved, then fine-tuned | High-resolution dual-resolution comparison |
| SegFormer-B0 | Pretrained when approved, then fine-tuned | Transformer-style edge-oriented comparison |

Every model must produce a project-owned checkpoint and training curve to qualify as
a final trained model. At least one of the three final runs uses random initialization.
Substitution requires official source/license, framework, export, and Jetson evidence
plus human approval.

### Known-object detection

Primary families are YOLO11n and RT-DETR-R18, subject to license, framework, and
export review. Both receive a common smoke and screening protocol; at least the
stronger detector receives a full final run. Full HPO of the second detector is
aggressive. Detection reports boxes/classes/confidence and is never conflated with
semantic segmentation or OOD.

## Training and selection sequence

```text
three-model canary/smoke/pilot
→ two-model extension smoke
→ five-model short screening
→ early export-feasibility gate
→ top-three medium-budget training
→ top-two limited HPO at fixed 512×1024
→ three final project-owned training runs
→ common Cityscapes-val evaluation of frozen final models
```

- Smoke verifies data, labels, loss, gradients, checkpoints, resume, and evaluator;
  it supports no scientific claim.
- Screening shares split, augmentation, effective batch, resolution, seed policy,
  metric code, and budget.
- Top-three promotion combines semantic quality, stability, measured cost/memory,
  and export evidence. At least two export-feasible candidates are preferred when
  available; a scientifically useful failed-export model may remain results-only.
- Initial HPO searches learning rate, weight decay, optimizer/scheduler, batch and
  accumulation, bounded augmentation, class weighting/OHEM, auxiliary weight, and
  model-relevant regularization. Resolution is excluded.
- Resolution is a separate controlled ablation at `512×1024` and `768×1536`; full
  `1024×2048` is optional evaluation, not a guaranteed training setting.
- Common evaluation uses frozen configs/checkpoints. Official Cityscapes val is not
  consulted during trials, used as `train_select`, or used for temperature fitting;
  it is neither sealed nor previously unseen.

## OOD and calibration program

1. Compare MSP, predictive entropy, MaxLogit, and Energy on the same aligned raw
   logits and score direction.
2. Train project anomaly methods only on deterministic synthetic outlier-exposure
   data with generator/config manifests.
3. Use Fishyscapes Static for development, HPO, normalization, and ablations.
4. Fit semantic calibration only on `train_calibration`; record before/after semantic
   calibration and OOD effects separately.
5. After the semantic winner is known, compare minimal exportable feature taps and at
   least BCEWithLogits versus BCE+Dice candidates. Freeze the selected adapter/loss
   before holdout use.
6. Open full Fishyscapes Lost & Found once as a frozen holdout after code, configs,
   checkpoint, threshold protocol, and artifact identities are locked.
7. Access SMIYC only through a later human-triggered sealed-final gate.

AP and FPR95 are OOD evaluation metrics; deployment thresholds are separate human
decisions. No score is described as anomaly probability without calibration evidence.

## Context, temporal, depth, and explainable fusion

- Road context uses semantic road/sidewalk and configurable proximity bands without
  turning missing labels into background.
- Connected components convert anomaly maps into regions with area, location,
  overlap, score, and known-object relationships.
- Lightweight temporal persistence links or filters regions across prerecorded
  frames with explicit state reset, missing-frame, and sequence-boundary behavior.
- A bounded relative-depth spike measures accuracy plausibility, exportability, and
  Jetson cost. It joins profiles only after human acceptance; metric-distance claims
  are stretch and need separate ground truth.
- Risk fusion exposes contributing signals and deterministic rules. `low`, `medium`,
  and `high` are operational categories, not collision probabilities.

## Export, Jetson, and Streamlit

- `EG-EXPORT-001` follows screening and tests each candidate's native-logit export,
  operator support, checker/runtime execution, shape/dtype, and measured numerical
  differences. It prevents late discovery of deployment blockers.
- After model freeze, final ONNX and TensorRT validation compares frozen PyTorch,
  ONNX, and Jetson TensorRT outputs under human-approved numerical gates.
- Jetson benchmarking separates warm-up, semantic, detection, OOD/postprocess,
  fusion, and end-to-end prerecorded-stream timing. It records FPS, latency
  distributions, memory, power, energy when reliable, temperature, throttling, and
  sustained behavior.
- Streamlit runs as a frontend to a measured backend. Only verified deployed
  backends are live-switchable. Other trained models appear as results-only cards,
  including failed-export evidence. UI/encoding time is separate from backend timing.

## Compute queues and accelerator policy

Before assigning the full queue, `EG-COMPUTE-001` probes one lightweight semantic
model, one heavier semantic model, and one detector at fixed input/sample/dataloader
settings. It records accelerator, CUDA capability, VRAM, batch, accumulation,
images/s, seconds/iteration, epoch estimate, data-wait fraction, peak allocated and
reserved memory, precision, preprocessing location, data source, and checkpoint sync
time.

| Queue | Work | Preferred allocation after measurement |
| --- | --- | --- |
| Q1 | Semantic smoke and screening | T4/L4; local for tiny probes |
| Q2 | Semantic medium training and HPO | L4/A100; H100 only with measured benefit |
| Q3 | Detection | L4 screening; A100 for selected full training |
| Q4 | OOD and calibration | T4/L4 development; A100 for synthetic/trainable runs |
| Q5 | Evaluation and export | T4/L4/Jetson according to backend |
| Q6 | Final confirmation | Best measured accelerator; frozen configs only |

Independent jobs may run concurrently only with separate checkpoint paths and safe
study storage. No job depends on a guaranteed A100/H100. High-end accelerators are
not used for archive extraction, unit tests, tiny inference, or I/O-bound work.

### Compute budgets

| Required Week-5 family | Provisional GPU hours |
| --- | ---: |
| Throughput probe and smoke | 2–3 |
| Five-model short screening | 8–15 |
| Top-three medium training | 15–24 |
| Top-two limited HPO | 12–22 |
| Three final semantic runs | 18–30 |
| Two-detector comparison and primary final | 8–16 |
| Zero-shot OOD, calibration, trainable baseline | 5–12 |
| Export, evaluation, integrated preparation | 2–8 |
| **Total** | **70–130** |

The aggressive queue adds a provisional `90–220` GPU hours, for a combined planning
band of `160–350`. First real runs replace estimates with measured accelerator-hours,
samples, optimizer steps, effective batch, wall time, interruptions, and failed-run
compute. Jetson device-hours are reported separately. Energy is reported only when a
reliable measurement exists.

## Storage and interruption policy

Private Google Drive has approximately 5 TB available; the local M1 Mac has only
approximately 18 GiB free and must not hold expanded datasets. The proposed project
root is `EdgeGuard/`, supplied at runtime through `EDGEGUARD_EXTERNAL_ROOT`. The
existing `private_inputs/` directory remains a legacy/current child, not the canonical
root. The relative child contract is:

```text
private_inputs/
archives/
datasets/
manifests/
generated/ood/
checkpoints/segmentation/
checkpoints/detection/
checkpoints/ood/
hpo/
experiments/segmentation/
experiments/detection/
experiments/ood/
experiments/calibration/
experiments/deployment/
exports/onnx/
evidence/jetson/
presentation/
thesis/
backups/checkpoints/
backups/experiment_registries/
```

No directory in this proposal is created by EG-DATA-001. Directories are created only
by an approved acquisition/experiment task after the root convention is accepted.
Absolute private roots never enter Git. Canonical archives are immutable and record
filename, byte size, SHA-256, source, access date, and terms. Extracted data is shared
rather than duplicated per run. Active data is copied or extracted into
`/content/edgeguard-work/<run_id>`; results and recovery state sync atomically back to
Drive with verified hashes. Oversized active data uses deterministic shards only
after fit/I/O measurement. The Mac is not a large-data relay. Irreplaceable
checkpoints and experiment registries retain at least one verified backup.

Migration or reuse of files already under `private_inputs/` is decided during
EG-DATA-002 or the first relevant acquisition task; this plan performs no automatic
move or copy.

Every expensive job supports new, exact resume, overwrite refusal, identity checks,
`last`/`best` and bounded recovery checkpoints, and path-free completion summaries.
Optuna, if justified, uses distinct study IDs and no untested concurrent SQLite
writers.

## Week 3–6 schedule

| Window | Critical work | Parallel work and gate |
| --- | --- | --- |
| Week 3 | `EG-THESIS-001`; `EG-DATA-001/002`; pinned semantic lab; five smokes and short screening | Acquisition decisions, accelerator probe, detector ontology/data planning; intermediate presentation uses only measured PIDNet baseline plus planned scope |
| Week 4 | Early export gate; top-three medium training; top-two limited HPO | Detector smoke/screening; synthetic OOD and Fishyscapes Static preparation; calibration implementation |
| Week 5 | Three final semantic runs; primary final detector; zero-shot/trainable OOD; calibration; context/temporal; final export; selected Jetson profile; Streamlit | Required 70–130 GPU-hour queue prioritized for final presentation; aggressive work only when critical evidence is safe |
| Week 6 | Evidence reconciliation, approved holdout/sealed evaluation, figures, thesis writing, reproducibility archive | Aggressive negative results and remaining ablations; no new claim without completed evidence |

Runtime availability and first-run measurements may reorder independent jobs but do
not change dataset roles or promotion order.

## Task graph

```text
EG-THESIS-001 → EG-DATA-001 → EG-DATA-002 repository implementation
                                      ├─→ EG-DATA-002 real preparation/split acceptance ─┐
                                      └─→ EG-SEG-001 repository laboratory/stack probe ──┤
                                                                                          ↓
                                                                                   EG-SEG-002
                                                           ↓
EG-DATA-001 → EG-DET-001 ───────────────────────→ EG-COMPUTE-001
                                                           ↓
                                                    EG-SEG-003 → EG-EXPORT-001
                                                           → EG-SEG-004
                                                           → EG-SEG-005
                                                           → EG-SEG-006
                                                              ├─ EG-OOD-002 → EG-CAL-001 → EG-OOD-003
                                                              └─ EG-EXPORT-002 → EG-JETSON-001

EG-DET-001 → EG-DET-002 → EG-DET-003
                          └─ EG-DET-004 [aggressive]

EG-OOD-003 + EG-DET-003 → EG-CONTEXT-001 → EG-TEMP-001 → EG-FUSION-001
EG-DEPTH-001 [aggressive] ───────────────────────────────┘ optional only
EG-FUSION-001 + EG-JETSON-001 → EG-DEMO-001 → EG-THESIS-002
```

## Risk register

| Risk | Early signal | Mitigation and stop rule |
| --- | --- | --- |
| Data/license delay | Missing human terms/source record | Do not download; run only approved fixtures or independent work |
| Local disk exhaustion | Less than active-task headroom | Keep canonical data in Drive and active shards in Colab; stop local extraction |
| Drive/Colab I/O bottleneck | High data-wait fraction | Copy/extract to `/content`; measure before streaming/cache design |
| Runtime interruption | Lost optimizer/trial state | Identity-protected resume and bounded atomic sync |
| Framework incompatibility | Smoke import/backward failure | One bounded repair; record failure; evaluate approved substitute |
| HPO overrun | Weak pruning or long trials | Enforce queue budget and promotion checkpoints |
| Export failure | Unsupported operator or unstable output | Preserve failure; limit live deployment; continue results-only science if justified |
| Jetson OOM/thermal instability | Build failure, throttling, memory pressure | Smaller batch/profile or alternate promoted model; no silent metric-changing fallback |
| OOD leakage | Holdout used during tuning | Invalidate affected results and refreeze before human-controlled access |
| Scope expansion | Core evidence slips behind aggressive work | Pause aggressive/stretch queues until Week-5 required evidence is safe |

## Immediate next actions

1. Complete and verify the real EG-DATA-002 Colab preparation, then accept or reject
   one measured group-atomic split.
2. Review and commit the dependency-light EG-SEG-001 repository laboratory, then run
   its exact-commit synthetic CUDA stack probe in Colab.
3. Do not begin EG-SEG-002 until both paths succeed and the framework/source gates are
   human-approved. Repository implementation may proceed in parallel; real training
   may not.
