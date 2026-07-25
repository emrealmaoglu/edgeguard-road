# ADR-0006: Keep runtime artifacts outside Git

- **Status:** Accepted
- **Date:** 2026-07-25

## Decision

Datasets, checkpoints, logits, caches, ONNX files, TensorRT engines, generated media,
and actual run outputs remain outside Git. Git stores schemas, scripts, manifests,
checksums, and small examples.

## Consequences

Artifact promotion relies on hashes and external storage rather than repository
history.
