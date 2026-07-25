# ADR-0004: Keep Jetson deployment-only

- **Status:** Accepted
- **Date:** 2026-07-25

## Decision

Jetson is used for measured inventory, TensorRT build, equivalence, benchmarking,
and offline demonstration—not training or uncontrolled agent work.

## Consequences

Privileged operations remain manual and engines are built on the target device after
environment inventory.
