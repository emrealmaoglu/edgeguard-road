# Semantic training laboratory

This directory contains independent, path-free YAML documents. It does not implement
YAML inheritance. The runner validates one framework file, one common policy file,
and one model file, then records the resolved canonical hashes.

`framework_mmseg.yaml` pins the proposed MMSegmentation source and narrow OpenMMLab
package versions. Torch and CUDA remain runtime-resolved in Colab and are not project
core dependencies. The framework status stays `proposal_pending_colab_probe` until
the five-model GPU compatibility notebook succeeds and a human accepts the evidence.

`common_cityscapes.yaml` is a stack-probe/smoke-ready policy, not a frozen scientific
training protocol. The synthetic fixture identity is permitted only for
`EGX-SEG-STACK-*`; it cannot replace the real dataset and human-selected split hashes.
No model config contains a private path or invented pretrained checkpoint identity.

The model files reference official configs relative to the exact external MMSeg
checkout. Their stack probe always disables downloads and uses random initialization.
Future pretrained project runs remain blocked until every source field is resolved and
human-approved.
