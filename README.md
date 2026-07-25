# EdgeGuard-Road

EdgeGuard-Road is an offline academic research prototype for studying road-obstacle
anomaly signals derived from semantic-segmentation logits. It is designed for a
Local → GitHub → Colab → Artifact → Jetson workflow with explicit human approval
gates.

It is **not** a safety-certified ADAS product, a braking controller, or a physical
risk estimator.

## WP-01 quick start

Python 3.10 or newer is required; Python 3.11 is used by CI.

```bash
python -m pip install -e '.[dev]'
python -m edgeguard doctor
python -m edgeguard doctor --json
python -m edgeguard smoke --config configs/smoke.yaml
python -m edgeguard smoke --config configs/smoke.yaml --deterministic
```

Run the local quality gate from the repository root:

```bash
ruff check .
ruff format --check .
mypy src/edgeguard
pytest -q
```

WP-01 uses synthetic CPU-only data. It does not download datasets or models and
does not claim real accuracy, latency, energy, or safety performance.

## Repository policy

- Scientific decisions and access to sealed test data remain with the human project
  owner.
- Datasets, checkpoints, logits, ONNX files, TensorRT engines, generated videos, and
  runtime artifacts stay outside Git.
- Colab notebooks are thin execution wrappers; implementation belongs in the package.
- TensorRT engines are eventually built on the target Jetson, after environment
  inventory and explicit human approval.
- No public software license has been granted yet. See `LICENSES.md`.

Start with `PROJECT_CHARTER.md`, `AGENTS.md`, and `docs/PROJECT_STATE.md`.
