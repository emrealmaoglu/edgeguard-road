#!/usr/bin/env bash
# Build and benchmark every reference engine on this Jetson, unattended.
#
# Each model costs roughly 12 minutes to build and 10 to soak, so the whole set runs for
# about an hour and a half. The point of doing it in one pass is the comparison: the
# accuracy and open-set axes are already measured off-device, and the third axis only
# becomes a Pareto front when every architecture is timed on the same hardware, at the
# same power mode, against the same frames.
#
# sudo is needed twice per model (tegrastats up, tegrastats down), and its credential
# cache would expire during a build, so a keeper refreshes it in the background rather
# than leaving the run to stall on a password prompt at minute 40.
#
# Usage:  sudo -v && bash scripts/jetson/run_all_models.sh ~/onnx ~/benchframes ~/results

set -euo pipefail

ONNX_DIR="${1:?usage: run_all_models.sh <onnx-dir> <image-root> <output-dir>}"
IMAGE_ROOT="${2:?usage: run_all_models.sh <onnx-dir> <image-root> <output-dir>}"
OUTPUT_DIR="${3:?usage: run_all_models.sh <onnx-dir> <image-root> <output-dir>}"

if ! sudo -n true 2>/dev/null; then
  echo "Run 'sudo -v' first so this can start and stop tegrastats without prompting." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Keep the sudo credential alive for the whole run; stop it whatever happens.
while true; do sudo -n true; sleep 50; done &
SUDO_KEEPER=$!
cleanup() {
  kill "$SUDO_KEEPER" 2>/dev/null || true
  sudo -n pkill tegrastats 2>/dev/null || true
}
trap cleanup EXIT

echo "Power mode:"
nvpmodel -q

for onnx in "$ONNX_DIR"/*.onnx; do
  model="$(basename "$onnx" .onnx)"
  engine="$OUTPUT_DIR/$model.plan"
  manifest="$OUTPUT_DIR/${model}_engine.json"
  telemetry="$OUTPUT_DIR/${model}_tegrastats.log"
  benchmark="$OUTPUT_DIR/${model}_benchmark.json"
  profile="$OUTPUT_DIR/${model}_stage_profile.json"

  echo
  echo "=== $model ==="

  if [ -f "$engine" ] && [ -f "$manifest" ]; then
    echo "engine already built, reusing"
  else
    echo "building engine (about 12 minutes)"
    python scripts/jetson/build_tensorrt.py \
      --onnx "$onnx" --engine "$engine" --manifest "$manifest" --execute > "$OUTPUT_DIR/${model}_build.json"
    echo "built: $(python -c "import json,sys; print(json.load(open(sys.argv[1]))['engine_sha256'])" "$OUTPUT_DIR/${model}_build.json")"
  fi

  # Telemetry has to cover the soak and nothing else: an idle tail drags the mean power
  # down and makes joule-per-frame look better than it is.
  rm -f "$telemetry"
  sudo -n tegrastats --interval 1000 --logfile "$telemetry" > /dev/null 2>&1 &
  sleep 3
  if [ ! -s "$telemetry" ]; then
    echo "tegrastats wrote nothing for $model, skipping its benchmark" >&2
    sudo -n pkill tegrastats 2>/dev/null || true
    continue
  fi

  echo "benchmarking (10 minutes)"
  python scripts/jetson/benchmark.py \
    --engine "$engine" --engine-manifest "$manifest" \
    --image-root "$IMAGE_ROOT" --telemetry-log "$telemetry" \
    --output "$benchmark" --power-profile 25W \
    --warmup 200 --minimum-iterations 30000 > /dev/null
  sudo -n pkill tegrastats 2>/dev/null || true

  python -m scripts.jetson.profile_pipeline \
    --engine "$engine" --image-root "$IMAGE_ROOT" --frames 60 --output "$profile" > /dev/null

  python - "$benchmark" <<'PYTHON'
import json
import sys

record = json.load(open(sys.argv[1]))
print(
    f"  engine {record['pure_engine_latency_ms']['median']:6.2f} ms | "
    f"frame {record['end_to_end_latency_ms']['median']:7.2f} ms | "
    f"{record['sustained_fps']:5.2f} FPS | "
    f"{record['telemetry']['mean_input_power_w']:5.2f} W | "
    f"{record['joule_per_frame']:.3f} J/frame"
)
PYTHON
done

echo
echo "All records are under $OUTPUT_DIR"
