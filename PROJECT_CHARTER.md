# EdgeGuard-Road Project Charter

## Purpose

EdgeGuard-Road is an undergraduate research prototype that compares lightweight
pixel-level road-scene models and deploys the best accuracy–latency–reliability
trade-off on NVIDIA Jetson Orin Nano Super.

The core research question is whether controlled multi-domain training with
Cityscapes and IDD20K improves accuracy, rare-class performance,
calibration, and robustness on unseen road domains without losing edge real-time
feasibility.

## Required phase-one outputs

For each image the system produces Cityscapes19 semantics, the road mask and
ego-reachable drivable corridor, pixel confidence and normalized entropy, a
source-calibrated frame shift alert, semantic connected-component regions, an
explainable operational-attention map, and measured backend latency.

Connected components are not instance detections: touching objects of one class may
merge. The attention value is not collision probability or learned physical risk.
EdgeGuard is not a safety-certified ADAS or a vehicle controller.

## Scientific protocol

- Scientific source domains: Cityscapes Fine and official IDD20K.
- Provisional engineering domain: the available Kaggle BDD100K semantic mirror; it is
  audited separately and excluded from HPO/model selection until official packages exist.
- Ontology: Cityscapes19 with `ignore=255`; uncertain IDD concepts are ignored.
- Roles: group-atomic fit/select/calibration; official validation is final-only.
- External evaluation: ACDC after checkpoint freeze; sealed MUSES/WildDash 2 after
  model/protocol freeze; KITTI Semantic is the declared access fallback.
- Models: SegFormer-B0, PIDNet-S, DDRNet-23-Slim, BiSeNetV2, and Fast-SCNN.
- Fairness: random initialization in the primary table; fixed `512×1024`, AdamW,
  effective batch four, augmentation family, seed, and optimizer-step budget.
- Selection: domain-macro source-select mIoU, rare-class mIoU tie-break, followed by
  measured ONNX/TensorRT deployment evidence.
- HPO: top two only, 12 TPE trials/model, 6,000 steps/trial, fixed search space.
- Reliability: equal-source global temperature scaling; ECE, NLL, Brier, confidence,
  entropy, maximum logit, and energy. Shift thresholds are source-calibration 95th
  quantiles frozen before external data.
- Every measurement is bound to configuration, code state, seed, dataset/ontology
  hashes, checkpoint hash, environment, and append-only run evidence.

Synthetic fixtures prove only engineering behavior. Pretrained references, project
training, source validation, public domain shift, sealed external, and Jetson results
are always reported separately. Missing measurements are never estimated.

## Edge acceptance

Deployment is PyTorch checkpoint → static ONNX FP32 → ONNX Runtime numerical check →
Jetson TensorRT FP16 → sustained device benchmark. INT8 is outside phase one.

The primary device claim uses 25W mode and requires end-to-end median at most 50 ms,
p95 at most 66.7 ms, sustained throughput at least 20 FPS, complete telemetry, no OOM,
and no sustained throttling. MAXN SUPER is a performance/thermal comparison only.
The benchmark records pure-engine and end-to-end latency, memory, input power,
joule/frame, temperatures, software versions, power mode, and engine hash.

## Conditional phase two

RTMDet-Tiny detection may start only after phase-one scientific/external results are
complete, the semantic TensorRT FP16 system is measured on Jetson, its 25W p95 is at
most 40 ms, at least 2 GiB safe memory remains, at least 20 GPU-hours and two weeks
remain, and official BDD100K detection labels/provenance exist. No second detector
family is allowed. Failing this gate is a valid completion: phase one remains the thesis.

## Authority and integrity

The human owner controls licenses/accounts, sealed submissions, scientific
interpretation, Git publication, and privileged Jetson operations. Automation may
prepare commands and evidence contracts but cannot accept licenses, reveal secrets,
open sealed tests early, invent metrics, change Jetson power state, or publish results.
