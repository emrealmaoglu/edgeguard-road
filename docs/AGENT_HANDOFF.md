# Agent Handoff

- **From:** Codex
- **To:** Human project owner
- **Repository root:** `.`
- **Branch:** `feat/first-vertical-slice`
- **Base commit:** `9d269c35f0adc08be193ec3ee50b7c505c485fae`
- **Task:** `EG-SEG-001 — Pinned semantic training laboratory`
- **State:** Repository implementation and diversity-aware split rebuild prepared;
  local validation results recorded at task end; real split rebuild, Colab stack
  probe, and human framework acceptance pending
- **Excluded:** No real dataset/Drive access, split rebuild execution, framework installation,
  GPU execution, scientific model training, official-val model selection,
  Fishyscapes/SMIYC access, Jetson action, stage, commit, or push

## Architecture

The implementation keeps the core package lightweight. Pydantic models validate:

- an exact MMSeg source/runtime proposal;
- one common Cityscapes policy and five independent model specs;
- deterministic config/fingerprint identities;
- explicit real-versus-synthetic dataset identities;
- cryptographically verified `policy_selected` split handoff;
- exact checkpoint resume metadata;
- root-relative JSONL registry records; and
- direct NCHW, 19-class native-logit probes.

There is no YAML inheritance, generalized plugin platform, database tracker, Docker
image, MLflow, DVC, or Ray dependency. Torch, CUDA, MMSegmentation, MMEngine, MMCV,
and OpenMIM are imported or installed only inside the real Colab execution path.

## Proposed framework identity

```text
MMSegmentation v1.2.2
source commit c685fe6767c4cadf6b051983ca6208f1b9d1ccb8
MMEngine 0.10.7
MMCV 2.1.0
OpenMIM 0.3.9
Torch/CUDA resolved from the actual Colab runtime
```

The proposal uses the upstream-declared MMEngine/MMCV compatibility ranges and does
not hard-code a CUDA wheel URL. Its status remains
`proposal_pending_colab_probe`. The local task did not install or import the stack.

## Five models and initialization

| Model | Stack probe | Proposed project training |
| --- | --- | --- |
| Fast-SCNN | Random, no download | Random required baseline |
| BiSeNetV2 | Random, no download | Random baseline; pretrained alternative needs a new decision |
| PIDNet-S | Random, no download | Approved ImageNet source unresolved; historical Cityscapes checkpoint prohibited as a new project checkpoint |
| DDRNet-23-Slim | Random, no download | Exact approved ImageNet source unresolved; random smoke possible |
| SegFormer-B0 | Random, no download | Exact approved ImageNet source unresolved |

Every config records architecture/backbone/head descriptions, class/ignore IDs,
normalization, baseline crop, output contract, export risk, and memory notes. No
pretrained filename, hash, revision, license, or access date is fabricated.

## Dataset and resume boundary

The fast split-only command reads the existing root-free dataset manifest and group
summary; it does not read images or masks. It generates D (`80/15/5`) and E
(`85/10/5`), keeps every group larger than 50 samples in `train_fit`, applies the
city/group/class/rare-class constraints, and selects the lowest-objective passing
candidate. The selected record binds the policy version/config hash, candidate hash,
dataset-manifest hash, and ontology hash with status `policy_selected`.

The training join verifies those hashes, rejects altered identities,
missing/duplicate samples, and any city+sequence group crossing roles. It does not
require a separate human-selected flag. `train_fit` and `train_select` are the only
runner-facing roles.
Official val remains `official_val_common_eval`, not routine selection, calibration,
sealed, or unseen data.

Checkpoint metadata binds config/fingerprint, framework, dataset/split,
initialization, model, Git, precision, and seed. Resume rejects any mismatch. The
framework checkpoint is expected to contain model, optimizer, scheduler, AMP scaler,
epoch/step, and best/last state. The JSONL registry refuses experiment-ID collisions
and stores only external-root-relative artifact paths.

## Colab probe

`notebooks/colab/04_semantic_training_stack_probe.ipynb`:

1. requires the exact reviewed 40-character project commit;
2. verifies a clean detached checkout and CUDA runtime;
3. installs the isolated proposal without a hard-coded CUDA wheel;
4. runs config/contract tests;
5. constructs all five official MMSeg models with random no-download initialization;
6. performs synthetic forward/backward and validates direct 19-class raw logits;
7. verifies one strict checkpoint/resume round trip;
8. packages a small path-free evidence ZIP; and
9. stops before Drive mount or real Cityscapes training.

Probe timing/memory is non-scientific compatibility evidence only.

## Local validation

- Ruff check: passed
- Ruff format check: 120 files passed
- mypy: 37 source files passed
- pytest: 227 passed, 2 expected opt-in skips
- Cityscapes-split/training-focused tests: 39 passed
- Synthetic scale check: 1,890 groups rebuilt in 15.298 seconds
- `git diff --check`: passed
- Framework installation: not run
- CUDA/GPU stack probe: not run
- Real semantic training: not run

## Next execution gates

1. Rebuild the real D/E split from existing external manifests without touching data:

   ```text
   python scripts/rebuild_cityscapes_splits.py \
     --dataset-manifest "$EDGEGUARD_EXTERNAL_ROOT/manifests/cityscapes/fine/v1/dataset_manifest.json" \
     --group-summary "$EDGEGUARD_EXTERNAL_ROOT/manifests/cityscapes/fine/v1/group_summary.json" \
     --output-directory "$EDGEGUARD_EXTERNAL_ROOT/manifests/cityscapes/fine/v1/split-policy-v1"
   ```

2. Run the stack notebook at the exact remotely verified commit and inspect all five model
   records, resolved versions, failures, raw-logit shapes, checkpoint resume, and ZIP
   hash.
3. Approve the framework and every real pretrained source before EG-SEG-002.

`EG-SEG-002` remains blocked. No result in this handoff claims real model training,
semantic accuracy, accelerator throughput, or deployment performance.
