# EdgeGuard-Road agent rules

These rules are binding for all automated coding sessions in this repository.
Deeper `AGENTS.md` files may add stricter rules for their own directories.

## Read first

At the start of every task, read:

1. `PROJECT_CHARTER.md`
2. `docs/PROJECT_STATE.md`
3. `docs/TASKS.md`

Read the applicable deeper `AGENTS.md` before modifying its directory.

## Authority and research integrity

- Scientific decisions, HPO scope, thresholds, dataset roles, artifact promotion,
  final interpretation, and sealed-test access belong to the human project owner.
- Never invent accuracy, latency, FPS, power, energy, thermal, or other measurements.
- Never use final-test data for training, calibration, normalization, development,
  HPO, threshold selection, or debugging.
- Preserve negative results and provenance. Label synthetic outputs clearly.

## Data, artifacts, and secrets

- Do not add datasets, checkpoints, logits, caches, ONNX files, TensorRT engines, or
  generated videos to Git.
- Do not read, print, create, or modify secrets, `.env`, SSH keys, API tokens, or
  credentials. Only `.env.example` may document variable names without values.
- Do not connect to Jetson, run `sudo`, or change JetPack, CUDA, TensorRT, storage,
  networking, or power modes.

## Code and verification

- New code must have type hints and concise docstrings where behavior is not obvious.
- Keep dependencies lean and platform-neutral unless an approved work package says
  otherwise.
- Add or update relevant tests whenever behavior changes.
- Run the relevant subset during development and the full Definition of Done before
  handoff.
- Keep changes small, reviewable, and scoped to the active work package.
- Do not stage files merely to create a diff.

## Git and handoff

- Commits, pushes, merges, tags, releases, and artifact promotions require explicit
  human approval.
- At task end update `docs/PROJECT_STATE.md`, `docs/AGENT_HANDOFF.md`, and the
  append-only `docs/AI_USAGE_LOG.md` when the task is material.
- Report `git status --short` and a file inventory; untracked files may not appear in
  `git diff --stat`.
