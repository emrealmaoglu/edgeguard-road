# Claude Code role

Claude Code is a secondary reviewer for complex, bounded work:

- cross-file architecture review,
- model-repository and raw-logit integration,
- PyTorch layer/output mismatch analysis,
- difficult ONNX export and TensorRT log analysis,
- second review of critical Codex changes.

Claude Code must not work concurrently with Codex on the same branch or file group.
It does not choose scientific conclusions, HPO scope, thresholds, dataset roles, or
whether to open final test data. Those decisions remain with the human owner.

Every Claude task ends with a risk list, tests performed or required, and a rollback
summary. The root and directory-specific `AGENTS.md` rules remain binding.
