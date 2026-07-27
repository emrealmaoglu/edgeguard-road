# EdgeGuard-Road

EdgeGuard-Road is an offline academic research prototype for multi-model open-set
road-safety perception over prerecorded video. The approved expanded direction
combines known-object detection, semantic segmentation, uncertainty/OOD analysis,
calibration, contextual and temporal risk reasoning, optional relative depth,
Jetson deployment, and a Streamlit dashboard.

The expanded thesis title is proposed and still requires human/university approval;
see `PROJECT_CHARTER.md`. The project is not a safety-certified ADAS product, vehicle
controller, or physical-risk estimator.

## Current evidence boundary

- Strict PIDNet-S checkpoint validation and CPU/MPS/T4 forwards exist.
- A clean 500-image Cityscapes validation campaign completed with 500 successes and
  zero failures. Its measured mIoU is `0.7875813077220126`; see
  `docs/research/CITYSCAPES_FULL_VAL_EVIDENCE.md` for the full claim boundary.
- MSP, predictive entropy, MaxLogit, and Energy score plumbing is implemented.
- Fishyscapes manual-only adapter and AP/FPR95 metric foundations are implemented and
  tested, but no real Fishyscapes inference has occurred.
- No project model training, detector experiment, calibration experiment, Jetson
  benchmark, or sealed SMIYC evaluation has started.

## Local quick start

Python 3.10 or newer is required; CI validates Python 3.10 and 3.11.

```bash
python -m pip install -e '.[dev]'
python -m edgeguard doctor --json
python -m edgeguard smoke --config configs/smoke.yaml --deterministic
ruff check .
ruff format --check .
mypy src/edgeguard
pytest -q
```

Training frameworks are deliberately absent from core dependencies. Datasets,
checkpoints, optimizer state, logs, ONNX models, TensorRT engines, videos, logits,
and generated media remain outside Git. Colab notebooks stay thin; Jetson is for
deployment and benchmarking, not training.

Start with `PROJECT_CHARTER.md`, `AGENTS.md`, `docs/PROJECT_STATE.md`, and
`docs/MASTER_PLAN_V2.md`.
