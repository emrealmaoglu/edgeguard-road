# Requirements wrappers

`pyproject.toml` is the single dependency source. Run these wrappers from the
repository root; they intentionally do not duplicate package lists.

- `base.txt` installs the editable core package.
- `dev.txt` adds lint, type-check, and test tools.
- `colab.txt` selects the currently empty Colab extra.
- `jetson.txt` selects the currently empty Jetson extra.

Do not add CUDA, TensorRT, JetPack-specific PyTorch, or other platform packages until
WP-02 records the actual environment and the human owner approves compatibility.
