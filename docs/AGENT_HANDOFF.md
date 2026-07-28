# Agent Handoff

- **Milestone:** Colab data access, 200 GiB staging, and delivery notebooks implemented.
- **Branch:** `rescue/semantic-first`; source catalog/probe and CI hardening are pushed
  through `be66669` before this final state record.
- **Classification:** implementation plus real public-source engineering evidence; no
  scientific model result.

## Latest completion

- Added a strict 5 TiB Drive / 200 GiB Colab access contract and human-readable runbook.
- Split delivery into data-preflight and training notebooks; both are output-free and
  their code cells compile.
- Added archive SHA-256/MD5 inventory, native prepared-root gates, deterministic tar
  receipts, safe staged extraction, 175 GiB data ceiling, and 25 GiB reserve.
- Added exact-commit single-file work snapshots so interrupted sessions can resume
  without training directly against mounted Drive.
- Corrected MUSES to its official direct package directory and KITTI to registered
  access; removed inaccessible Hugging Face road probes and third-party mirrors.
- Full local gate: Ruff 259 files, mypy 99 source files, pytest 374 passed/10 skipped,
  notebook compile, and diff check passed.

## Earlier source-catalog completion

- Replaced the stale mixed-task dataset catalog with a nine-dataset semantic portfolio.
- Recorded official counts/splits, native label contracts, access/license state,
  canonical merge policy, and protected stages.
- Added a certificate-verified, byte-bounded public sample downloader that refuses
  registered/sealed datasets and writes hash-bound external receipts.
- Downloaded and inspected one official A2D2 RGB/semantic pair plus license, README and
  class list under ignored cache; every file matched its pinned size and SHA-256.
- Added the complete official 55-color A2D2 mapping proposal with unknown-color rejection
  and explicit ignore reasons; verified the real sample has no undeclared colors.
- Added deterministic size-power domain quotas for a future alpha ablation while keeping
  uniform domains as the primary and only active sampler.
- Documented merge, leakage, class-imbalance and HPO boundaries, plus the real probe.
- Removed non-road plumbing probes and inaccessible road candidates from the active
  catalog; the official A2D2 sample evidence remains documented separately.

## Verification

- Nine of nine official landing pages reachable through certificate-verified HEAD probes.
- A2D2 public bundle: five files, 3,892,400 bytes total, exact per-file SHA-256 verified;
  image/mask 1208x1920; 55 declared colors, 21 present, zero unknown.
- Focused catalog/multi-domain tests: `19 passed`.
- All-tree Ruff format/lint: 253 files passed.
- Mypy: 98 source files passed.
- Full pytest: `369 passed, 10 skipped`.
- Catalog regeneration, idempotent real sample probe, nine-source HTTPS probe and
  `git diff --check` passed.
- GitHub CI run `30332310919`: Python 3.10 and 3.11 jobs passed with Node 24-based
  official action majors. The earlier optional-`mmengine` core-CI failure is fixed.

## External execution order

1. Run `EdgeGuard_Data_Preflight_Colab.ipynb`; acquire only approved official
   Cityscapes, BDD100K and IDD20K packages and bind hashes.
2. Audit, review and explicitly freeze all three source manifests.
3. Run the five-model smoke/pilot/screening path and validate ONNX feasibility.
4. Run top-two HPO, dataset-composition and CE/weighted-CE ablations.
5. Freeze finalists, fit source-only calibration, then open official source val and ACDC.
6. Author the sealed release and perform one WildDash 2 or MUSES external evaluation.
7. Consider A2D2 only after the primary thesis table is complete and its full mapping,
   sequence grouping, duplicates and usable-pixel coverage are reviewed.

No licensed corpus, GPU training, model promotion, ACDC/sealed inference, submission,
ONNX artifact or Jetson action occurred.
