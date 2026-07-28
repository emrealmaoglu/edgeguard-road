# Notebook index

The semantic-first delivery has two ordered notebooks:

1. `EdgeGuard_Data_Preflight_Colab.ipynb` initializes the Drive layout, prints only
   official acquisition routes, inventories/hashes archives, validates prepared roots,
   and creates one SHA-256-bound tar per source dataset.
2. `EdgeGuard_Road_Colab.ipynb` enforces the 175 GiB staging plus 25 GiB reserve policy,
   stages bundles to `/content`, then owns audit → training → evaluation → ONNX →
   single-file Drive snapshots. It defaults to audit-only.

Do not train directly from the mounted Drive tree. Do not open ACDC, WildDash, MUSES,
or KITTI before their declared final-only gate.

The experimental campaign notebooks are listed in
`docs/canonical-colab-runbook.md` and use the `00/10/20/30/40` sequence.

Older numbered notebooks remain in `notebooks/colab/` only as historical implementation
evidence. Their first cell says `DEPRECATED — NON-CANONICAL`; do not execute them or use
them as campaign entry points.
