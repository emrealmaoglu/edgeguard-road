# Agent Handoff

- **From:** Codex
- **To:** Human project owner
- **Repository root:** `~/Projects/edgeguard-road`
- **Branch:** `feat/first-vertical-slice`
- **Base commit:** `05f72b4bdd8c3a69bba62a9023f61ceb2504c355` (Commit B)
- **Scope:** Commit C candidate only: semantic metrics, four aligned-logit scores,
  minimal Cityscapes evaluator, mandatory local 1/5/10-image campaign support,
  compact artifact packaging, and thin full-resolution Colab preparation
- **Excluded scope:** No resume platform, optional 20-image campaign,
  boundary/interior analysis, correlation matrix, Fishyscapes, SMIYC, ONNX,
  TensorRT, Jetson, vendoring, dependency expansion, threshold selection, or push
- **Approved upstream:** `https://github.com/XuJiacong/PIDNet.git` at
  `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`
- **Checkpoint pin:** `PIDNet_S_Cityscapes_val.pt`, SHA-256
  `b51aa935bdb64a0779d776f38267fd49f7cce59413910abbbf0a74934b3d7c01`;
  license remains `OPEN QUESTION`, and the checkpoint remains outside Git
- **Cityscapes evidence:** External val-only dataset remains 500 image/label pairs
  across 3 cities with deterministic manifest SHA-256
  `7e91ab791d1814aa355b9ff3a765697fed9d56897e9aff6aa74463501b84f852`
- **Implementation summary:** Added streaming 19-class semantic metrics with explicit
  ignore/absent-class behavior; numerically stable MSP, predictive entropy,
  MaxLogit, and Energy scores; reusable PIDNet inference session; deterministic
  `city_round_robin_v1` selection; a project-specific minimal evaluator with
  path-free metadata, small visuals, structured failures, and non-empty-output
  refusal; local/full-resolution configs; and a safe deterministic ZIP packager
- **Output contract:** `native_logits` remains the direct softmax-before model output.
  `aligned_logits` remains a bilinear derivative. Semantic mask and all four score
  summaries explicitly record `aligned_logits`; no score is called an anomaly
  probability.
- **Mandatory failure coverage:** Wrong checkpoint hash, missing dataset root,
  missing image-label pair, image-label geometry mismatch, and non-empty output
  collision are tested.
- **Notebook result:** `02_pidnet_cityscapes_val_eval.ipynb` is execution-only,
  path/secret-safe, and locally JSON/compile validated. It checks out a human-entered
  reviewed Commit C SHA, verifies the checkpoint and Cityscapes archives, performs
  a clean CUDA single-image spike, runs a full-resolution one-image preflight, then
  invokes the full 500-image campaign and packages its artifacts. It does not contain
  or claim a real Colab result.
- **Local 1-image result:** CPU, 1/1 successful, evaluation time about 0.777 s,
  development-grid mIoU about 0.6632, and pixel accuracy about 0.9637
- **Local 5-image result:** MPS, 5/5 successful, evaluation time about 3.945 s,
  development-grid mIoU about 0.6121, and pixel accuracy about 0.9514
- **Local 10-image result:** MPS, 10/10 successful, evaluation time about 5.710 s,
  development-grid mIoU about 0.6246, and pixel accuracy about 0.9565
- **Metric boundary:** These are deterministic small-subset, resized-input-grid,
  local development measurements. They are not a final benchmark, OOD result,
  threshold result, anomaly probability, or reproduction of the official PIDNet
  paper protocol.
- **Score result:** MSP, predictive entropy, MaxLogit, and Energy summaries were
  finite for every local campaign. They are ID-only plumbing/development evidence.
- **Package evidence:** External final 10-image ZIP contains 15 files, is 1,132,620
  bytes, and has SHA-256
  `40c0033c13de1a6641f3cf73fca317d695a2a79cc51d7832b3789362ede1d408`
- **Validation result:** Ruff check passed; Ruff format check passed for 87 files;
  mypy passed for 25 source files; pytest passed 155 with 2 expected opt-in skips;
  notebook JSON and six code cells compiled; path/secret and large-file scans were
  clean; `git diff --check` passed
- **Sealed boundary:** No SMIYC files were opened, downloaded, configured,
  manifested, or used for debugging
- **Repository artifact boundary:** Dataset, checkpoint, upstream checkout, logits,
  generated visuals, evaluator outputs, and ZIP packages remain external/ignored.
  No large or binary runtime artifact was added to Git.
- **Publication:** Commits A, B, and C exist locally. The branch is three commits
  ahead of origin; nothing was pushed, merged, tagged,
  released, or promoted.
- **Human decision:** The project owner approved the Commit C diff and local commit.
- **Blocker:** Push is not authorized. The Colab campaign additionally requires the
  exact pushed commit to be approved.
- **Next action:** Human reviews the reported Commit C SHA and authorizes or declines
  push.

## Changed file inventory

- Config and contracts: `configs/cityscapes_eval_local.yaml`,
  `configs/cityscapes_eval_colab.yaml`, `src/edgeguard/config.py`, and
  `src/edgeguard/contracts.py`.
- Model/data/evaluation/scoring: `src/edgeguard/models/pidnet_spike.py`,
  `src/edgeguard/data/cityscapes.py`, `src/edgeguard/evaluation/__init__.py`,
  `src/edgeguard/evaluation/semantic.py`,
  `src/edgeguard/evaluation/cityscapes_runner.py`,
  `src/edgeguard/scoring/__init__.py`, and
  `src/edgeguard/scoring/uncertainty.py`.
- Execution surfaces: `scripts/run_cityscapes_eval.py`,
  `scripts/package_eval_artifacts.py`, and
  `notebooks/colab/02_pidnet_cityscapes_val_eval.ipynb`.
- Tests: `tests/integration/test_notebook.py`,
  `tests/unit/test_cityscapes.py`, `tests/unit/test_cityscapes_runner.py`,
  `tests/unit/test_config.py`, `tests/unit/test_contracts.py`,
  `tests/unit/test_package_eval_artifacts.py`,
  `tests/unit/test_semantic_metrics.py`, and
  `tests/unit/test_uncertainty.py`.
- State/evidence: `docs/PROJECT_STATE.md`, `docs/TASKS.md`,
  `docs/AGENT_HANDOFF.md`, and `docs/AI_USAGE_LOG.md`.
