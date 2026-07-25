# ADR-0002: Use lean tooling

- **Status:** Accepted
- **Date:** 2026-07-25

## Decision

Use setuptools, Pydantic, PyYAML, NumPy, pytest, Ruff, and mypy. Do not introduce
Hydra, DVC, MLflow, Docker, Ray, or a custom orchestration framework in WP-01.

## Consequences

The foundation remains CPU-only and easy to inspect. New infrastructure requires a
demonstrated need and a separate decision.
