# EdgeGuard-Road Project Charter

## Purpose

EdgeGuard-Road is a computer-engineering undergraduate thesis prototype. It studies
whether inexpensive anomaly signals derived from the raw logits of a real-time,
closed-set semantic-segmentation model can mark previously unseen road obstacles.
The intended pipeline combines those signals with soft drivable-area context,
connected components, and lightweight temporal persistence, then evaluates the
result offline on an NVIDIA Jetson Orin Nano Super.

The prototype is not a product, safety-certified ADAS, braking controller, collision
probability estimator, or authorization for real-world autonomous operation.

## Research questions

1. How can multiple segmentation candidates be compared fairly under the same data,
   preprocessing, resolution, output, and evaluator contracts?
2. How do MSP, MaxLogit, Entropy, and Energy compare when all are derived from the
   same raw semantic logits and evaluated under one OOD protocol?
3. How much do soft context, component filtering, and temporal persistence reduce
   false alarms without hiding relevant anomaly events?
4. How well does the selected method generalize to unseen datasets, and what
   accuracy, latency, energy, memory, and thermal trade-offs appear on Jetson?

## Core scope

- Reproducible dataset lineage and leakage-safe split roles.
- A fair primary-model and comparison-model protocol.
- Raw-logit anomaly scoring with semantic calibration kept separate from OOD
  normalization and threshold selection.
- Context, component, and temporal post-processing with explicit ablations.
- Task-level PyTorch/ONNX/TensorRT equivalence checks.
- Offline Jetson benchmarking and a clearly separated demonstration mode.
- Traceable runs, artifacts, decisions, limitations, and negative results.

## Out of scope

- Online vehicle control, braking, steering, or deployment on public roads.
- Safety certification or claims of physical risk probability.
- Agent-selected scientific conclusions, final thresholds, or sealed-test access.
- Unbounded model search or HPO.
- Training on Jetson or using Colab as an unreviewed source-code workspace.

## Human authority

The human project owner approves dataset roles, split boundaries, model selection,
HPO space and budget, thresholds, sealed-test opening, artifact promotion, scientific
interpretation, branch integration, release, and all privileged Jetson operations.

## Research integrity

- No metric or performance value may be invented.
- Train, calibration, development, and final-test roles must remain disjoint.
- Sequence-level leakage is prohibited.
- Test data may not influence training, HPO, normalization, or threshold selection.
- Negative results and failed runs remain part of the evidence trail.
- One primary and one comparison model are preferred; additional models must not
  delay the core result.

## Success definition

Success means a reproducible and auditable research prototype whose claims are tied
to run IDs, source state, configs, manifests, raw results, and environment records.
Numerical scientific success thresholds are intentionally pending human decisions.

## Execution path

```text
Local implementation and tests
→ reviewed Git commit/tag
→ Colab execution
→ hash-addressed artifacts
→ local validation and promotion
→ device-built TensorRT engine
→ Jetson equivalence, benchmark, and offline demo
→ sealed final evaluation and thesis evidence package
```
