# ADR-0001: Use a src layout

- **Status:** Accepted
- **Date:** 2026-07-25

## Decision

Install the `edgeguard` package from `src/edgeguard` rather than importing directly
from the repository root.

## Consequences

Tests exercise the installed package and catch packaging mistakes. Editable install
is required for normal development.
