# ADR-0005: Seal final-test data

- **Status:** Accepted
- **Date:** 2026-07-25

## Decision

Final-test data and manifests are inaccessible to automated development and tuning.
The human owner opens the test only after source, config, model, and manifest hashes
are frozen.

## Consequences

Any post-test change creates a separately labeled exploratory run; the original
result is retained.
