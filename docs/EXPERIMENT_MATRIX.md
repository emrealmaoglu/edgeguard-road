# Experiment Matrix

The executable task authority is `docs/TASKS.md`; the claim authority is
`docs/THESIS_CLAIM_MATRIX.md`. This compact matrix fixes the scientific comparison.

| Stage | Candidates/data | Budget | Promotion evidence |
| --- | --- | --- | --- |
| Integrity | 2 scientific adapters + 1 provisional BDD adapter | one-batch each | Audit, provenance, split, label, duplicate and reload gates |
| Core canary/smoke | SegFormer-B0, Fast-SCNN, PIDNet-S; 2 sources | FP32/FP16 probe + 50 steps/model | Finite AMP gradients, forced resume, checkpoint reload |
| Core pilot | same 3 models, 2 scientific sources | 2,000 steps/model | Domain/select and rare-class metrics |
| Extension smoke | DDRNet-23-Slim, BiSeNetV2 | probe + 50 steps/model | Same canary/recovery contract |
| Screening | all 5 models | 6,000 steps/model | Domain-macro mIoU and early ONNX result |
| HPO | frozen top two | 12 × 6,000 steps/model | LR, WD, scheduler, warm-up only |
| Data ablation | finalists | CS; CS+IDD | Common budget, source-select only; BDD mirror excluded |
| Loss ablation | final candidate | CE; weighted CE | Overall and rare-class mIoU |
| Final | 3 explicitly frozen finalists | 40,000 steps/model | Convergence, checkpoint and provenance |
| Reliability | frozen finals | source calibration | ECE, NLL, Brier, shift thresholds |
| Final evaluation | frozen finals | source val, ACDC, sealed external | Domain/class/rare/reliability tables |
| Deployment | frozen finalists | ONNX then Jetson TRT FP16 | Numerical match, sustained device evidence |

No external dataset, loss choice, resolution, augmentation, initialization, or sampler
is an HPO variable. Failed integrations and exports remain structured negative results.
