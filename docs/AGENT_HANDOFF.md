# Agent Handoff

- **Milestone:** Colab v2 hermetic runtime and resumable semantic campaign implementation.
- **Branch:** `stabilize/colab-v2`, based on `origin/rescue/semantic-first`.
- **Classification:** locally tested engineering implementation; real Colab, Drive training,
  scientific, accepted-release, and Jetson measurements are not run.
- **Publication state:** the approved application commit is
  `24b59a282e0bd7239ec3bd9edc179288dcca7460`; both generated notebooks pin that exact
  commit. Delivery commit `1cdce46cd9b1bef5735733b0783fadb18139e28e` was pushed to the
  branch. This handoff also includes a narrow CI portability correction for Streamlit's
  file-path resolution. No merge, tag, release, Drive write, or artifact promotion occurred.

## Root causes closed in code

- uv is exactly `0.8.8`; the removed `--upgrade-strategy only-if-needed` option is gone.
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
  `baab1bbeb40923f1ea4e0a1dd80f4f9a2beb620f16e0b845efe45f0312c45369` (preflight) and
  `6bb5c05b49117ada827fda15ab939fa15f08321976801def8e84d5c761594359` (training).
- Runtime lock SHA-256 values are
  `ecf59a924106d68ca49cc8e4a6c52e1206fc22fa91fb7e41a0f18069fa25865d` (main) and
  `bf96b00b6753d4eacd3a62e18eda6c96e0f26fcd5679d7926f66cc930fd4e924` (OpenMMLab).
- Installer, pipeline, thesis and Jetson package CLIs load successfully; source/script
  compilation and `git diff --check` pass.
- The first branch CI run passed installation, Ruff, formatting, and mypy on Python 3.10
  and 3.11, then exposed one Streamlit test-only relative-path failure on both versions.
  The test and the matching local-closure probe now derive the dashboard entry point from
  the repository root instead of depending on Streamlit's version-specific caller path.

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
