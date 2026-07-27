# EdgeGuard-Road Project Charter

## Formal identity

EdgeGuard-Road is a computer-engineering undergraduate thesis prototype for
multi-model, open-set road-safety perception on resource-constrained edge devices.
It combines known-object detection, semantic segmentation, uncertainty and OOD
analysis, contextual and temporal processing, optional relative-depth evidence, and
explainable operational-risk fusion over prerecorded road video.

The proposed thesis title, pending human and university approval, is:

> **EdgeGuard-Road: Multi-Model Open-Set Road Safety Perception with Object
> Detection, Semantic Segmentation, Uncertainty Calibration and Temporal Risk
> Fusion on Resource-Constrained Edge Devices**

The proposed Turkish title is:

> **Kaynak Kısıtlı Uç Cihazlarda Çok Modelli Açık Küme Yol Güvenliği Algısı:
> Nesne Tespiti, Semantik Bölütleme, Belirsizlik Kalibrasyonu ve Zamansal Risk
> Füzyonu**

The fallback title remains:

> **EdgeGuard-Road: Uncertainty-Calibrated Open-Set Road Hazard Segmentation with
> Contextual and Temporal Risk Analysis on Resource-Constrained Edge Devices**

Nothing in this repository asserts that the university-approved title has changed.

## Purpose and research questions

The project studies:

1. How five edge-oriented semantic architectures compare under one leakage-safe
   data, resolution, training, metric, and provenance protocol.
2. How two known-object detector families complement semantic and open-set evidence.
3. How MSP, MaxLogit, predictive entropy, Energy, calibration, synthetic outlier
   exposure, and one trainable anomaly method compare under explicit dataset roles.
4. How road context, connected components, lightweight temporal persistence, and an
   optional relative-depth signal affect explainable risk decisions.
5. What accuracy, OOD quality, latency, memory, power, energy, and thermal trade-offs
   appear after ONNX/TensorRT deployment on Jetson Orin Nano Super.

## Scope

- **Core:** five-model semantic screening, three final semantic checkpoints, limited
  top-two HPO, two detector comparison with at least one final detector, zero-shot
  OOD, calibration, one trainable anomaly method, context, lightweight temporal,
  selected Jetson pipeline, and Streamlit prerecorded-stream dashboard.
- **Aggressive:** second detector full HPO, coarse-to-fine semantic training, depth
  integration, extra seeds, complete deployment profiles, and broader ablations.
- **Stretch:** Mapillary, INT8, metric distance, learned risk fusion, and advanced
  tracking.

At least one final semantic checkpoint must be trained from random initialization.
Pretrained initialization is permitted for other project-owned training runs.

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
