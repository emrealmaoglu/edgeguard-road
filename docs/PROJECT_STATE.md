# Project State

- **Repository/branch:** `rescue/semantic-first`; multi-domain rescue baseline `cbef065`,
  dataset-source hardening `0df7801`, optional-integration CI fix `2828630`, and Node 24
  action update `be66669` are committed and pushed.
- **Active work package:** `EG-MULTIDOMAIN-001`, source acquisition and catalog gate.
- **Research scope:** domain generalization for Cityscapes19 semantic segmentation with
  Cityscapes, BDD100K, and IDD20K source data; five lightweight MMSeg models; bounded
  HPO; class imbalance; reliability; ACDC; sealed WildDash 2/MUSES; ONNX/demo.
- **Implemented source-data path:** strict versioned ontology; BDD100K/IDD20K native
  adapters; official-count, corrupt, geometry, label, exact-hash and perceptual-hash
  audit; group-atomic 80/15/5 candidates; explicit human freeze; cross-domain duplicate
  evidence; domain-uniform distributed sampler; pooled train-fit rare classes and
  bounded mean-one median-frequency weights.
- **Source catalog:** semantic catalog v2 covers Cityscapes, BDD100K, IDD20K, ACDC,
  WildDash 2, MUSES, KITTI Semantics, Mapillary Vistas and A2D2 with official counts,
  native labels, access/license state, allowed roles and merge policy. A bounded HTTPS
  probe reached all nine official landing pages on 2026-07-28.
- **Public sample evidence:** an official A2D2 front-center image/mask pair, license,
  README and 55-color class list were downloaded into ignored `data/cache/` with exact
  byte/SHA-256 verification. Geometry was 1208x1920; the mask used 21 declared colors,
  no unknown color; the phase-two exact-only mapping retained 96.7306% of this sample's
  pixels. This is engineering evidence only and is not generalized to the full corpus.
- **Phase-two A2D2 path:** complete 55-color fail-closed RGB mapping proposal; 31 colors
  map to Cityscapes19 and 24 become ignore with explicit reasons. It is not a frozen
  ontology decision and cannot enter training before full audit/split/duplicate review.
- **Merge decision:** native trees remain separate; generated canonical masks are
  separate; primary sampling is domain-uniform. Size-power sampling alpha `{0,0.5,1}`
  is implemented/documented only as a post-baseline data ablation and is excluded from
  HPO.
- **Hugging Face probe:** anonymous Dataset Viewer access was verified for the 20-row
  `nateraw/ade20k-tiny` image/label fixture as plumbing evidence only. Road-specific
  candidates returning 401 are treated unavailable, not inferred public.
- **Local quality:** all-tree Ruff format/lint passed for 253 files, mypy passed for
  98 source files, and the full suite passed with `369 passed, 10 skipped`. The catalog
  command, real A2D2 sample probe, nine-source HTTPS probe and `git diff --check` passed.
- **Remote quality:** GitHub CI run `30332310919` passed both Python 3.10 and 3.11 using
  official `actions/checkout@v7` and `actions/setup-python@v7`. The preceding CI failure
  was traced to one optional MMSeg test assuming `mmengine` in the core environment;
  it now skips only when that optional integration is absent and still runs locally.
- **Validation separation:** BDD100K/IDD20K official val remains final-only; ACDC is
  domain-shift-only; WildDash 2/MUSES remain sealed and have no sample download path.
- **Implemented experiment/deployment path:** five-model step protocol, measured-only
  top-two Optuna HPO, equal-domain calibration, sealed release, static ONNX validation,
  aspect-preserving inference, Streamlit demo, Colab notebook, evidence tables and
  append-only run ledgers remain unchanged from `cbef065`.
- **Scientific evidence:** no real multi-domain training, HPO, calibration, ACDC,
  sealed external, ONNX model or GPU result exists. Local fixtures and the A2D2 public
  sample are engineering evidence only.
- **Local data/compute:** only the bounded public A2D2 probe is present under ignored
  cache. No licensed Cityscapes/BDD100K/IDD20K/ACDC/WildDash/MUSES corpus is available;
  this Apple M1 host has no CUDA training path.
- **Next action:** acquire licensed Cityscapes/BDD100K/IDD20K packages outside Git,
  record archive receipts, execute the three real audits, review/freeze manifests, then
  run five 50-step CUDA smokes. A2D2 stays phase two until the primary comparison exists.

Implemented, locally verified, externally executed, scientifically measured and human
accepted remain separate states.
