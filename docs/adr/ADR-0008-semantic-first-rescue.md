# ADR-0008: Make semantic-first the delivery-critical path

- **Status:** Accepted by project owner
- **Date:** 2026-07-28

## Context

The expanded feature branch proved many synthetic and random-weight engineering
paths, but real Cityscapes project training had not started. Detection, temporal,
HPO, anomaly-head, and evidence-orchestration scope increased delivery risk without
producing the thesis-critical semantic comparison.

## Decision

The active path is Cityscapes audit and `CSF-SPLIT-D`, three random-init semantic
models, one loss ablation, temperature calibration, ACDC domain shift, static ONNX,
Streamlit, Colab, optional Jetson measurement, and evidence-only reporting. It uses
stock MMSeg Runner/IoUMetric and six public command surfaces.

Expanded campaign, detection, temporal, learned anomaly, depth, HPO, INT8, and fusion
code remains available as experimental/legacy evidence but cannot block delivery or
contribute synthetic fixture values to scientific tables.

## Consequences

- A complete thesis deliverable no longer depends on detector data or Jetson access.
- Missing ACDC or Jetson access narrows claims rather than blocking the core.
- Any later scope expansion requires the semantic-first gates to be complete first.
- Existing expanded-scope documents remain historical and carry a legacy notice.
