# Thesis Claim Matrix

This matrix prevents planned capabilities from becoming thesis claims before their
evidence exists. Human acceptance conditions are frozen before the relevant holdout
or sealed evaluation; numerical thresholds are not invented here.

| Claim | Required experiment | Dataset | Metric/evidence | Required artifact | Acceptance condition | Current status |
| --- | --- | --- | --- | --- | --- | --- |
| The official PIDNet-S checkpoint is loaded strictly and produces finite raw semantic logits | `EGX-BASE-PIDNET-SPIKE-001` | Approved fixed-checkout sample | Exact keys/shapes, `strict=True`, native/aligned tensor checks, repeat evidence | Checkpoint/load report and run metadata | Reviewed checkpoint identity and reproducible forward | **Supported within plumbing boundary** |
| The project has a reproducible single-scale PIDNet-S Cityscapes-val baseline | `EGX-BASE-PIDNET-CITYSCAPES-001` | Cityscapes official val, 500 images | mIoU `0.7875813077220126`, pixel accuracy `0.9619008903101843`, mean class accuracy `0.8618737663500519`, 500/500 success | External evidence ZIP SHA-256 `756abf1a…132b7` and evidence record | Hash/provenance/pixel consistency verified | **Supported; not official-paper reproduction** |
| Five semantic architectures were compared fairly | `EGX-SEG-SMOKE-*`, `EGX-SEG-SCREEN-*` | Frozen Cityscapes train fit/select split | Shared semantic metrics, compute and stability | Five configs/checkpoints/curves and comparison table | Common protocol and human promotion review | `planned` |
| Three final project-owned semantic models were trained, including one from scratch | `EGX-SEG-FINAL-{01..03}` | Cityscapes train roles; official-val common evaluation | Training curves and common semantic metrics | Three checkpoint bundles with init provenance | All runs complete; one `random_init`; official val excluded from HPO/select/temperature fitting; human accepts comparison | `planned` |
| The system detects known road objects with two detector families | `EGX-DET-SCREEN-*`, `EGX-DET-FINAL-PRIMARY` | Frozen BDD100K roles | mAP50, mAP50-95, precision, recall, small-object recall | Detector checkpoints, ontology, reports | Two-family comparison and one accepted final detector | `planned` |
| Four zero-shot OOD scores are implemented consistently | Existing score code plus `EGX-OOD-ZS-*` | Fishyscapes Static development | AP, FPR95, finite direction checks | Score configs and development report | Same logits/grid/evaluator and reviewed protocol | Implementation foundation exists; real OOD result `planned` |
| The pipeline demonstrates open-set road-hazard discrimination | `EGX-OOD-ZS-*`, `EGX-OOD-LAF-HOLDOUT-001`, later sealed run | Static dev, full L&F holdout, SMIYC sealed final | AP, FPR95 and failure analysis | Frozen model/config/manifests and holdout/sealed reports | Human-approved dev/holdout/sealed protocol completed without leakage | `planned` |
| Semantic confidence is uncertainty-calibrated | `EGX-CAL-SEMANTIC-*` | `train_calibration`; frozen confirmation | ECE, NLL, Brier, reliability diagrams | Calibrator/config/report | Improvement and non-degradation rule frozen and accepted by human | `planned` |
| A trainable anomaly-aware method improves over zero-shot baselines | `EGX-OOD-SYNTH-*`, `EGX-OOD-TRAINABLE-*` | Synthetic OOD train, Static dev, L&F holdout | AP/FPR95, semantic retention, export and cost | Synthetic manifest, model checkpoint, ablation report | Candidate adapter/loss frozen; human accepts improvement/retention evidence | `planned` |
| Road context reduces irrelevant false alarms without hiding hazards | `EGX-CONTEXT-ABL-*` | OOD development data | False-positive reduction and anomaly retention | Parameterized ablation report | Human-frozen trade-off accepted | `planned` |
| Temporal persistence improves event stability | `EGX-TEMP-LITE-*` | SOS or approved sequence-level temporal data | Event precision/recall, persistence, flicker, latency | Sequence manifests and temporal ablation | Improvement and reset correctness accepted | `planned` |
| Relative depth adds useful proximity evidence within edge cost | `EGX-DEPTH-SPIKE-*` | Approved development inputs | Relative ordering/plausibility, latency, memory, power | Feasibility and export report | Human accepts utility and Jetson budget; no metric-distance overclaim | `planned` aggressive |
| Explainable fusion produces traceable operational-risk labels | `EGX-FUSION-RULE-*` | Approved prerecorded development streams | Rule/component ablations and failure cases | Per-event signal contribution log | Deterministic output and human-reviewed claim boundary | `planned` |
| The selected pipeline runs in real time on Jetson | `EGX-EXPORT-FINAL-*`, `EGX-JETSON-PROFILE-*` | Frozen prerecorded benchmark set | Sustained synchronized latency/FPS, memory, power, thermal | Engine manifest, equivalence and benchmark reports | Human defines and accepts the real-time/device gates before benchmark | `planned`; no Jetson benchmark exists |
| Streamlit demonstrates the deployed system without contaminating benchmarks | `EGX-DEMO-PRERECORDED-001` | Approved demo-only video | Functional demo and separate backend/UI timing | Demo config, backend whitelist, fallback recording | Only verified backends live-switch; disaster fallback passes | `planned` |

## Prohibited claim upgrades

- ID-only score summaries are not OOD performance.
- A pretrained baseline is not project-owned model training.
- Colab CUDA timing is not Jetson latency or FPS.
- Relative depth is not metric distance.
- An operational-risk category is not collision probability.
- A completed notebook cell is not a verified artifact.
- Fishyscapes adapter/unit tests are not real Fishyscapes inference.
- No SMIYC statement is allowed before the recorded human sealed-test event.
