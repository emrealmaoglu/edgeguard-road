# Notebook index

The semantic-first delivery has two ordered notebooks:

1. `EdgeGuard_Data_Preflight_Colab.ipynb` initializes the Drive layout, prints only
   official acquisition routes, discovers the current `private_inputs/` uploads,
   inventories/hashes archives, validates prepared roots, and creates one SHA-256-bound
   tar per source dataset.
2. `EdgeGuard_Road_Colab.ipynb` enforces the 175 GiB staging plus 25 GiB reserve policy,
   stages bundles to `/content`, then owns audit → training → evaluation → ONNX →
   immutable checkpoint recovery, HPO rung resume, and review ZIP creation. It defaults to
   `CAMPAIGN_TARGET="audit"`; later targets run only missing prerequisites. Set
   `DOWNLOAD_REVIEW_PACKAGE=True` only when a stage has finished and you want the small
   reports/figures package downloaded to your computer.

Both notebooks support the repository-only `EDGEGUARD_NOTEBOOK_LOCAL_TEST=1` contract
mode. `python scripts/dev/run_delivery_notebooks_local.py` executes every code cell while
disabling Drive, network, environment installation, real data, GPU training and external
evaluation. This catches notebook syntax/order/name-integration regressions; it is not a
substitute for the real Colab CUDA and dataset gates.

Current scientific source manifests are Cityscapes + IDD20K. The uploaded Kaggle BDD
archive is optional engineering evidence, but cannot enter HPO or the main scientific table
unless replaced by the two official BDD semantic packages.

Both notebooks install an append-only, secret-redacting Drive failure reporter. After an
error, set `DOWNLOAD_LATEST_FAILURE_REPORT=True` and run the final cell to download the
small failure ZIP. See `docs/COLAB_FAILURE_REPORTING.md` before retrying.

Do not train directly from the mounted Drive tree. Do not open ACDC, WildDash, MUSES,
or KITTI before their declared final-only gate.

The experimental campaign notebooks are listed in
`docs/canonical-colab-runbook.md` and use the `00/10/20/30/40` sequence.

Older numbered notebooks remain in `notebooks/colab/` only as historical implementation
evidence. Their first cell says `DEPRECATED — NON-CANONICAL`; do not execute them or use
them as campaign entry points.
