# Agent Handoff

- **From:** Codex
- **To:** Human project owner
- **Repository root:** `~/Projects/edgeguard-road`
- **Branch:** `feat/first-vertical-slice`
- **Base commit:** `aa8803e8060af8cd704f81fb7c6903d0d48e2a6e` (Commit C, pushed)
- **Scope:** Independent full-validation Cityscapes evidence verification; narrow
  selection/visual provenance correction; manual-only Fishyscapes Lost & Found
  validation foundation; NumPy pixel AP and FPR95 metrics
- **Excluded scope:** No Colab rerun, 500-image rerun, automatic dataset download,
  real Fishyscapes evaluation, FS Static generation, SMIYC access, threshold,
  calibration, training, fine-tuning, context, morphology, temporal processing,
  generic registry/framework, new dependency, commit, or push

## Cityscapes evidence result

- External ZIP: `~/edgeguard-data/evidence/cityscapes/full-val/2026-07-26/edgeguard-cityscapes-eval.zip`
- ZIP identity: SHA-256
  `756abf1a983b8eed11b22f0c10b3cabf093d6e614a4bec2a6d223c41202132b7`,
  19,966,292 bytes, 43 entries, all CRC checks passed
- Internal identity: all 42 files in `artifact_manifest.json` matched; canonical
  file-map SHA-256
  `a10b66ae0108c838c0f9c76970af47108d2e9629d2602229fc6bca6591c83d7d`
- Dataset manifest SHA-256:
  `7e91ab791d1814aa355b9ff3a765697fed9d56897e9aff6aa74463501b84f852`
- Selection manifest SHA-256:
  `93efa0d5a4ab9c91c0f8ccf3865252646a22f4d41605032d003c6a524091d2a9`
- Provenance: exact Commit C, `git_state=clean`, `git_dirty=false`
- Completion: 500 selected, 500 successful, 0 failures
- Shapes: input `[1,3,1024,2048]`, native logits `[1,19,128,256]`, aligned
  logits `[1,19,1024,2048]`
- Semantic measurements: mIoU `0.7875813077220126`, pixel accuracy
  `0.9619008903101843`, mean class accuracy `0.8618737663500519`
- Pixel consistency: confusion-matrix total 917,018,489 evaluated pixels;
  131,557,511 ignored; sum 1,048,576,000 equals `500 × 1024 × 2048`
- Runtime: 827.00028256 seconds total and mean 1.6539497549533844 seconds/sample
  are end-to-end evaluation-pipeline timing, not pure inference latency or Jetson FPS
- Memory: peak PyTorch CUDA allocated memory 217,726,976 bytes
- Scores: MSP, predictive entropy, MaxLogit, and Energy summaries were finite and
  each covered 1,048,576,000 pixels
- Content checks: no absolute user/Drive path or secret-like value was found in
  textual members
- Claim boundary: This is the EdgeGuard-Road single-scale PIDNet-S evaluation, not
  guaranteed reproduction of the official PIDNet paper protocol. No OOD, threshold,
  calibration, or anomaly-probability claim is made.

The ZIP, PNGs, dataset, checkpoint, logits, and other generated outputs remain
external and were not copied into Git.

## Selection provenance correction

The completed `--all` run evaluated the same 500 sorted samples recorded in its
manifest, but labeled the strategy `city_round_robin_v1`; its first-five visual
rule also yielded only Frankfurt samples. This is a non-blocking provenance and
visual-diversity defect, not a metric defect.

The follow-up changes:

- record `all_sorted_v1` for `--all`;
- retain `city_round_robin_v1` for subset-size selection;
- record `subset_manifest_preserved_v1` for explicit manifests;
- select visuals independently with the existing deterministic city round-robin;
- preserve evaluation order, metrics, and score summaries.

Focused tests passed. One local CPU image smoke also passed 1/1 with zero failures;
it was a Git-dirty development smoke and is not promoted evidence. The 500-image
campaign was not rerun.

## Fishyscapes development foundation

- Lost & Found validation role: `ood_development`
- FS Static validation role: `ood_development`, generation-preparation only
- Source mode: manual only; no downloader exists
- Mask contract: ID `0`, anomaly `1`, ignored/void `255`
- Score direction: higher means more anomalous
- Adapter safeguards: deterministic annotation ordering, exact train/test pairing,
  RGB decoding, geometry checks, native mask validation, relative paths, per-file
  SHA-256, and root-free deterministic manifest
- Metrics: non-interpolated pixel Average Precision and FPR at the first threshold
  achieving at least 95% TPR; no operating threshold is selected or stored
- Undefined cases: no anomaly pixels gives `null` AP/FPR95; no negative ID pixels
  gives `null` FPR95
- Test coverage: perfect/reversed rankings, ties, extreme imbalance, ignored pixels,
  missing positive/negative classes, invalid labels/shapes, and NaN/Inf rejection
- Sealed boundary: no SMIYC file, path, loader, config, manifest, or debug access

## Required human data steps

1. Review and accept the original Lost & Found image terms.
2. Manually obtain the 100-image public Fishyscapes annotation archive from the
   official Fishyscapes-linked Zenodo record and underlying images from their
   official provider.
3. Record source URL, filename, access date, byte size, SHA-256, and separate terms
   for annotations and images; do not infer that one license covers both.
4. Extract outside Git into the layout documented in
   `docs/research/FISHYSCAPES_DEVELOPMENT_PLAN.md` and approve the resulting
   deterministic manifest before a real run.
5. For FS Static, separately approve a pinned official generator source/version and
   legally available Cityscapes/generator inputs. Generate outside Git and do not
   redistribute generated images.

## Validation and publication

- Targeted selection/Fishyscapes/OOD tests: passed
- One-image CPU smoke: passed, 1/1, zero failures
- Original remote failure: GitHub Actions run `30190797659`, job
  `Quality (Python 3.10)`, reported `src/edgeguard/scoring/uncertainty.py:43:
  error: Argument 1 to "validate_anomaly_map" has incompatible type
  "floating[_32Bit] | ndarray[tuple[int, ...], dtype[Any]]"; expected
  "ndarray[tuple[int, ...], dtype[Any]]" [arg-type]`.
- Python 3.10 compatibility: an isolated Python 3.10.11 environment used the CI
  dependency combination NumPy 2.2.6 and mypy 1.20.2. Editable dev install, Ruff
  check, Ruff format check, mypy, and pytest all passed; pytest reported 174 passed
  with 2 expected opt-in skips.
- Local Python 3.11.9 quality gate: Ruff check passed; Ruff format check passed for
  93 files; mypy passed for 27 source files; pytest passed 174 with 2 expected
  opt-in skips; `git diff --check` passed.
- Remote CI: still pending; the correction is local and unstaged until human review,
  commit approval, and separate push authorization.
- Artifact/path/secret/large-file scans: clean
- Current changes: unstaged
- Publication: nothing new committed, pushed, merged, tagged, released, or promoted
- **Next action:** Human reviews the unstaged coherent diff and approves or rejects a
  local commit.

## Changed file inventory

- Selection: `src/edgeguard/evaluation/cityscapes_runner.py` and
  `tests/unit/test_cityscapes_runner.py`
- Fishyscapes adapter: `src/edgeguard/data/fishyscapes.py` and
  `tests/unit/test_fishyscapes.py`
- OOD metrics and score typing: `src/edgeguard/evaluation/ood.py`,
  `src/edgeguard/scoring/uncertainty.py`, and `tests/unit/test_ood_metrics.py`
- Evidence/role documentation:
  `docs/research/CITYSCAPES_FULL_VAL_EVIDENCE.md`,
  `docs/research/FISHYSCAPES_DEVELOPMENT_PLAN.md`, and
  `docs/research/WP03_WP05_SOURCE_REVIEW.md`
- State/audit: `docs/PROJECT_STATE.md`, `docs/TASKS.md`,
  `docs/AGENT_HANDOFF.md`, and `docs/AI_USAGE_LOG.md`
