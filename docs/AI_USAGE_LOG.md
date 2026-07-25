# AI Usage Log

Record material Codex and Claude work here. Append entries; do not rewrite prior
accepted history. Missing evidence is written as “not run” or “pending,” never
inferred.

| Date (UTC) | Tool / agent | Task | Branch | Changed files or areas | Tests | Human review | Acceptance | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-25 | Codex | WP-00/WP-01 repository bootstrap | `chore/repo-foundation` | Initial repository foundation | Pending | Pending | In progress | No commit or push authorized |
| 2026-07-25 | Codex | WP-00/WP-01 implementation and local verification | `chore/repo-foundation` | Governance, ADRs, package, configs, contracts, CLI, tests, CI, notebook | Ruff lint/format, mypy, 28 pytest tests, doctor, smoke, notebook JSON passed | Pending | Ready for review | Repo remains unborn; nothing staged, committed, or pushed |
| 2026-07-25 | Codex | Pre-commit Python CI matrix correction | `chore/repo-foundation` | `.github/workflows/ci.yml`, project state, handoff, AI usage log | Local Python 3.11.9: Ruff lint/format, mypy, and 28 pytest tests passed | Pending | Ready for review | CI now targets Python 3.10 and 3.11; remote jobs not yet run; nothing staged, committed, or pushed |
| 2026-07-25 | Human project owner | Independent WP-00/WP-01 verification | `chore/repo-foundation` | Full repository foundation | Python 3.11.9 `.venv`; Ruff check passed; Ruff format check passed; mypy passed; pytest 28/28 passed; doctor passed; normal smoke passed; deterministic smoke byte-level comparison passed; secret scan clean | Independently performed by human | G0 and G1-LOCAL approved; G1-CI pending | Scientific decisions and gate approval are human responsibilities |
| 2026-07-25 | Codex | Pre-commit documentation cleanup and staging preparation | `chore/repo-foundation` | Project state, handoff, AI usage log | Python 3.11.9: Ruff check/format, mypy, pytest 28/28, and requested cache ignore checks passed | Pending staged-diff review | Ready for review | No code, architecture, or dependency change; no commit or push |
