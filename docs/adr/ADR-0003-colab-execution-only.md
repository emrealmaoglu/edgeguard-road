# ADR-0003: Keep Colab execution-only

- **Status:** Accepted
- **Date:** 2026-07-25

## Decision

Colab notebooks contain checkout, install, doctor, smoke, and test commands only.
All reusable logic lives in the package or reviewed scripts.

## Consequences

Temporary notebook edits are not a source of truth and never flow directly to the
main branch.
