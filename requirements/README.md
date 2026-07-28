# Requirements wrappers

`pyproject.toml` is the single dependency source. Run these wrappers from the
repository root; they intentionally do not duplicate package lists.

- `base.txt` installs the editable core package.
- `dev.txt` adds lint, type-check, and test tools.
- `colab.txt` installs the platform-neutral demo, reporting, ONNX, and validation
  dependencies used by the semantic-first notebook. PyTorch/MMCV/MMSeg remain
  runtime-resolved by the pinned installer because CUDA wheels are platform-specific.
- `pip install -e '.[rescue]'` installs the same non-CUDA dependencies for a local
  CPU demo/export environment.
- `jetson.txt` selects the currently empty Jetson extra.

Do not add CUDA, TensorRT, JetPack-specific PyTorch, or other platform packages until
WP-02 records the actual environment and the human owner approves compatibility.
