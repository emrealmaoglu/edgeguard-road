# Agent Handoff

- **Milestone:** Colab v2 hermetic runtime and resumable semantic campaign implementation.
- **Branch:** `stabilize/colab-v2`, based on `origin/rescue/semantic-first`.
- **Classification:** locally tested engineering implementation plus one real Colab data
  audit; GPU canary/training, accepted-release, and Jetson measurements are not run.
- **Publication state:** the current application commit is
  `5f665cdbe0caad011ff66ad7b210ab8121d9fad1`; both regenerated notebooks pin that exact
  commit. Its notebook-delivery commit and replacement remote CI are pending. No merge,
  tag, release, training, or artifact promotion occurred.

## Root causes closed in code

- uv is exactly `0.8.8`; a differing hosted uv is ignored and the pinned binary is installed
  under the EdgeGuard cache prefix. The removed `--upgrade-strategy only-if-needed` option is gone.
- A single managed Python 3.11.13 environment replaces hosted/fallback resolution.
- NumPy 1.26.4 and PyTorch 2.1.1/cu121 are installed by one hash-locked `uv pip sync`.
- MMEngine 0.10.7 and mmcv-lite 2.1.0 are removed from the main dependency solution and
  installed together from a two-wheel hash lock with `--no-deps`. The main lock directly
  supplies `rich`; GUI OpenCV is rejected and headless OpenCV 4.10.0.84 is mandatory.
- MMSegmentation is frozen at v1.2.2 commit `c685fe6767c4cadf6b051983ca6208f1b9d1ccb8`
  and installed without dependency resolution. Failure writes `failure.json`; no other
  matrix is silently selected.
- The notebook generator imports `canonical_json`, fixing the PR #1 Ruff failure at source.

## Delivered contracts

- `semantic-cs-idd-v2` 3→5 gate: core canary/smoke/pilot, extension smoke, five-model
  screening, top-two HPO, three explicit finalists, and one CE/weighted-CE ablation.
- CUDA canary checks FP32 forward/backward, finite AMP/FP16 output/loss/gradients, and exact
  checkpoint reload for SegFormer-B0, Fast-SCNN and PIDNet-S.
- Python orchestrator owns prerequisite closure, resume identity, one OOM retry with
  effective batch four, screening evaluation/ONNX/reporting, and uniform artifacts.
- Public orchestrator targets are exactly `smoke`, `pilot`, `screening`, `hpo`, `final`,
  `evaluate`, `export`, and `report`. The final target writes a measured release candidate;
  later targets require a separately human-approved, hash-bound accepted release.
- Training/validation manifests cannot be frozen merely by selecting a later target.
  Candidate SHA-256, campaign, project commit, reviewer, and decision must match an explicit
  human review receipt. Cityscapes official val now has a separate final-only manifest.
- Every secondary scientific-domain audit is bound to the SHA-256 identities of earlier
  source candidates. Candidate manifests are accepted for overlap checking only after
  their own audit passes; old completion receipts without these source hashes are rejected.
- Stale audit output is quarantined using the same expected-input identity that rejected
  its completion receipt, preventing a restored legacy directory from blocking recreation.
- Accepted final checkpoints are restored from immutable Drive generations before later
  phases, then rechecked against the accepted release hash.
- Streamlit accepted-bundle mode, cached tables/models, single active model resource, CPU
  fallback, calibration/failure review and Jetson evidence page.
- Accepted-only thesis collector with source files, LaTeX, provenance, hash-bound claim
  index and honest `not_run` coverage.
- Jetson handoff ZIP with static ONNX and golden vectors; TensorRT stays target-built.

## Local verification

- Mypy: 115 source files passed.
- Pytest: 460 passed and 2 environment-gated skips after both tracked notebooks were
  regenerated against the immutable application commit.
- Both tracked notebooks compile and regenerate byte-identically. Their SHA-256 values are
  `db3cfe28e525dd43ace13d902eeef44081bf243aabf6388869a7c27504d315f5` (preflight) and
  `dd56a9ffd39f9175193d7f2207e05cf62686c4aa5bf2635789c4643147487d86` (training).
- Runtime lock SHA-256 values are
  `ecf59a924106d68ca49cc8e4a6c52e1206fc22fa91fb7e41a0f18069fa25865d` (main) and
  `bf96b00b6753d4eacd3a62e18eda6c96e0f26fcd5679d7926f66cc930fd4e924` (OpenMMLab).
- Installer, pipeline, thesis and Jetson package CLIs load successfully; source/script
  compilation and `git diff --check` pass.
- The first branch CI run passed installation, Ruff, formatting, and mypy on Python 3.10
  and 3.11, then exposed one Streamlit test-only relative-path failure on both versions.
  The test and the matching local-closure probe now derive the dashboard entry point from
  the repository root instead of depending on Streamlit's version-specific caller path.
- Replacement CI run `31107068048` passed every install, Ruff, format, mypy, and pytest
  step on both Python 3.10 and 3.11.
- The real audit package hash and all 31 indexed files verified. Cityscapes was 2975/2975;
  IDD accounted for 14,027 records with 14,018 valid and nine accepted all-ignore-source
  exclusions. Three cross-role dHash candidates were visually and numerically confirmed as
  distinct highway scenes. The first provenance rerun exposed and now closes a restored
  legacy-output quarantine identity mismatch before any data scan or training began.

## External acceptance sequence

1. Publish the notebook-delivery commit and require remote G0 CI to pass.
2. Run two independent clean Colab GPU canaries and compare lock hashes (G1).
3. Reuse the verified Cityscapes bundle and IDD receipts, then human-freeze v2 manifests
   only through review receipts.
4. Run smoke with a deliberate interruption/resume, then pilot, extension smoke, screening,
   HPO, and final as separate targets.
5. Review `accepted_release.candidate.json`, create a candidate-hash-bound human receipt,
   and promote it with `scripts/accept_colab_release.py` before evaluate/export/report.
6. Transfer the Jetson bundle, inspect JetPack/L4T/TensorRT versions, build FP16 on-device,
   and run the existing 25W sustained benchmark. Do not auto-upgrade or change power mode.

No external dataset, checkpoint, ONNX, TensorRT engine, Drive artifact, model metric, merge,
tag, or release is implied by this handoff.
