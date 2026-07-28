# Notebook index

`EdgeGuard_Road_Colab.ipynb` is the only semantic-first delivery notebook. It owns
the audit → training → evaluation → ONNX → Drive-sync path and defaults to audit-only
until the dataset gate passes.

The experimental campaign notebooks are listed in
`docs/canonical-colab-runbook.md` and use the `00/10/20/30/40` sequence.

Older numbered notebooks remain in `notebooks/colab/` only as historical implementation
evidence. Their first cell says `DEPRECATED — NON-CANONICAL`; do not execute them or use
them as campaign entry points.
