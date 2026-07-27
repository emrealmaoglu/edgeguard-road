# Canonical Colab runbook

Only the following notebooks are active campaign entry points, in this order:

1. `notebooks/colab/00_campaign_control.ipynb`
2. `notebooks/colab/10_semantic_campaign.ipynb`
3. `notebooks/colab/20_ood_calibration_risk.ipynb`
4. `notebooks/colab/30_detection_temporal_fusion.ipynb`
5. `notebooks/colab/40_export_and_reporting.ipynb`

All other `.ipynb` files in `notebooks/colab/` are historical, prominently marked
`DEPRECATED — NON-CANONICAL`, and must not be used to start or resume a campaign.

## Preflight

Before opening Colab, run the project-owned preflight command against evidence generated
for the exact clean commit:

```text
python scripts/dev/check_precolab_readiness.py \
  --expected-commit <40-character-reviewed-commit> \
  --closure-summary <external-closure-root>/campaign_summary.json \
  --equivalence-report <external-equivalence-root>/report.json \
  --deployment-validation <external-precolab-root>/deployment-validation.json
```

The command must report `status=passed`. It verifies the exact clean commit, canonical
notebook set, local closure lineage, PIDNet-S/RT-DETR ONNX classifications, deployment
fixture inference, available disk, and that every remaining gate is external data,
CUDA, Jetson, or a human scientific decision.

## Boundaries

- Start from the reviewed detached commit; do not run from a dirty checkout.
- Use runtime-supplied private paths and secrets only. Do not write them into Git.
- Do not open Lost & Found or SMIYC during development.
- Random-weight local artifacts validate engineering paths only; they do not select models.
- Colab measurements do not assign Jetson deployment profiles. TensorRT and sustained
  device measurements remain separate Jetson gates.
