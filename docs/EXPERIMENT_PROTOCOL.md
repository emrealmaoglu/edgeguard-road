# Experiment Protocol Invariants

This document records invariants before any real experiment is run.

## Split roles

- Training, semantic calibration, OOD development, and final test are disjoint roles.
- Video/sequence data are split at sequence level, never frame level across roles.
- Missing labels are unknown/missing, never silently converted to background.
- Final-test manifests are sealed and unavailable to automated tuning or debugging.

## Prohibited leakage

- No HPO, normalization, threshold selection, early stopping, or implementation
  debugging on final-test data.
- Development-protocol results using OOD samples are not described as zero-shot.
- Opening sealed test is a human-triggered, recorded event after commit, configs,
  model artifacts, and manifests are frozen.

## Calibration and anomaly processing

- Semantic confidence calibration and OOD score normalization are distinct stages.
- All anomaly scores use the convention “higher means more anomalous.”
- Threshold protocols and budgets are fixed by the human owner before evaluation.

## Provenance

- Every real run has a unique run ID.
- Deterministic config and experiment fingerprints are separate from volatile run
  metadata.
- Every promotable run records a clean Git commit, config hash, dataset-manifest hash,
  model-artifact hash, environment, and raw outputs.
- `unborn` and `dirty` source states are development evidence, not promotion evidence.
- Negative and failed results remain in the audit trail.

## Benchmark versus demo

- Benchmark mode disables GUI and recording, fixes inputs, warms up, and records
  stage timing and telemetry.
- Demo mode may render overlays and HUDs but its numbers are not benchmark evidence.
