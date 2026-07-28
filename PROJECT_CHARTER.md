# EdgeGuard-Road Project Charter

## Formal identity

EdgeGuard-Road is a computer-engineering undergraduate thesis prototype for
leakage-safe semantic road-scene segmentation, confidence calibration, adverse-domain
evaluation, and deployment trade-off measurement on resource-constrained edge devices.

The proposed thesis title, pending human and university approval, is:

> **EdgeGuard-Road: Lightweight Road-Scene Semantic Segmentation with Confidence
> Calibration and Adverse-Condition Evaluation on Resource-Constrained Edge Devices**

The proposed Turkish title is:

> **Kaynak Kısıtlı Uç Cihazlarda Hafif Yol Sahnesi Semantik Bölütleme:
> Güven Kalibrasyonu ve Olumsuz Koşul Değerlendirmesi**

The fallback title remains:

> **EdgeGuard-Road: Uncertainty-Calibrated Open-Set Road Hazard Segmentation with
> Contextual and Temporal Risk Analysis on Resource-Constrained Edge Devices**

Nothing in this repository asserts that the university-approved title has changed.

## Purpose and research questions

The project studies:

1. How SegFormer-B0, Fast-SCNN, PIDNet-S, DDRNet-23-Slim, and BiSeNetV2 compare under
   one leakage-safe resolution, training, metric, initialization, and provenance protocol.
2. Whether controlled Cityscapes + BDD100K + IDD20K training improves source-domain
   macro mIoU and genuinely unseen-domain robustness over Cityscapes-only training.
3. Whether train-fit-only median-frequency weighting improves rare-class IoU over
   standard CrossEntropy without unacceptable overall degradation.
4. How equal-domain scalar temperature calibration changes ECE, NLL, Brier,
   confidence, and entropy.
5. How frozen models degrade on ACDC and sealed WildDash 2/MUSES external domains.
6. What accuracy, model-size, ONNX latency, and optional Jetson trade-offs appear for
   the scientific and edge finalists.

## Scope

- **Core:** Cityscapes/BDD100K/IDD20K audit and source-role manifests; five random-init
  pilots/screening; top-two bounded HPO; dataset-composition and loss ablations;
  equal-domain calibration; ACDC plus sealed external evaluation; ONNX validation;
  Streamlit, Colab, and thesis evidence tables.
- **Conditional:** sustained Jetson Orin Nano Super benchmark when the approved device
  is available.
- **Experimental/legacy:** detection, temporal fusion, trainable
  anomaly heads, relative depth, INT8, learned fusion, and advanced tracking.

The five-architecture comparison starts from random initialization. The measured
external pretrained PIDNet checkpoint remains a separately labeled reference.

## Claim and safety boundary

The prototype is not a safety-certified ADAS product, braking or steering
controller, collision-probability estimator, or authorization for operation on
public roads. `low`, `medium`, and `high` are explainable operational-risk labels,
not calibrated physical-risk probabilities. Depth is relative unless a separately
validated metric-distance protocol exists.

## Human authority and research integrity

The human project owner approves title changes, dataset access and roles, ontology,
split freeze, HPO budget, model promotion, thresholds, holdout and sealed-test
opening, artifact promotion, scientific interpretation, Git publication, and
privileged Jetson operations.

- No measurement or scientific result may be invented.
- Train, select, calibration, development, holdout, and sealed-final roles remain
  disjoint; video data is split by group/sequence.
- Cityscapes official validation supports common evaluation of frozen final models;
  it is excluded from routine HPO, `train_select`, and temperature fitting but is
  not sealed or previously unseen.
- Project synthetic anomalies train OOD methods; Fishyscapes Static supports
  development/HPO; full Lost & Found is a one-time frozen holdout; SMIYC remains
  sealed final.
- Failed runs and failed exports remain evidence.
- Planned, implemented, locally tested, Colab measured, Jetson measured, and human
  accepted states are never conflated.

## Execution path

```text
Local contracts, adapters, tests, and tiny probes
→ reviewed Git commit
→ Drive-backed, interruption-safe Colab execution from ephemeral storage
→ hash-verified external artifacts
→ early export feasibility
→ frozen model and final ONNX/TensorRT validation
→ sustained Jetson benchmark
→ Streamlit prerecorded-stream demonstration
→ human-gated holdout/sealed evaluation and thesis evidence package
```
