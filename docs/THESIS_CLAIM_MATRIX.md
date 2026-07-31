# Thesis Claim Matrix

| Claim | Required evidence | Current state |
| --- | --- | --- |
| Two scientific source datasets are merged without ontology or split leakage | Full audits, frozen manifests, unknown-label and cross-domain duplicate reports; BDD mirror provenance reported separately | Implemented; real audit pending |
| Five lightweight models are compared fairly | Common random-init protocol, five pilots, valid screenings, failures retained | Campaign implemented; measurements pending |
| Multi-domain training improves unseen-domain generalization | CS versus CS+IDD ablation plus frozen ACDC and sealed external results | Pending |
| Weighted CE changes rare-class performance | Fit-only weights, CE/weighted CE common-budget comparison | Implemented; measurements pending |
| Calibration improves reliability | Equal-source temperature, ECE/NLL/Brier before/after without semantic changes | Implemented; measurements pending |
| Source uncertainty detects domain shift | Source-only thresholds, source/external AUROC/AP and alert rates | Implemented; measurements pending |
| Road/corridor output is useful and bounded | Road IoU, boundary F1, false-drivable and fragmentation | Implemented; measurements pending |
| Regions localize semantic objects | Component coverage/merge/fragmentation; explicit non-instance boundary | Implemented; measurements pending |
| Attention output is explainable | Frozen formula, per-term contributions, deterministic examples/ablation | Implemented; not physical-risk probability |
| Selected semantic system is real time on Jetson | Valid ONNX/TRT, 25W sustained latency/FPS/power/thermal benchmark | Pending real device run |
| RTMDet-Tiny adds value | Every conditional phase-two gate plus separate mAP/edge evidence | Blocked by gate |

Pretrained references, project-trained models, source validation, public domain shift,
sealed external tests, heuristics, and hardware measurements must occupy separate tables.
Colab timing is never a Jetson claim; uncertainty is never called anomaly ground truth;
semantic components are never called instance detections.
