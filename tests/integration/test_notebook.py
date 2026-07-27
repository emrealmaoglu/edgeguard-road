"""Static checks for the thin Colab execution wrapper."""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "notebooks/colab/00_environment_and_smoke.ipynb"
PIDNET_NOTEBOOK = REPO_ROOT / "notebooks/colab/01_pidnet_single_image_spike.ipynb"
CITYSCAPES_NOTEBOOK = REPO_ROOT / "notebooks/colab/02_pidnet_cityscapes_val_eval.ipynb"
CITYSCAPES_TRAIN_NOTEBOOK = REPO_ROOT / "notebooks/colab/03_prepare_cityscapes_fine_train.ipynb"
PIDNET_RUNNER = REPO_ROOT / "scripts/run_pidnet_spike.py"


def test_notebook_is_valid_thin_wrapper_without_secret_markers() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert payload["nbformat"] == 4
    assert payload["nbformat_minor"] >= 5
    source = "\n".join(line for cell in payload["cells"] for line in cell.get("source", [])).lower()
    assert "git clone" in source
    assert "pip install -e ." in source
    assert "python -m edgeguard doctor" in source
    assert "python -m edgeguard smoke" in source
    assert "pytest -q" in source
    for prohibited in ("api_key", "password=", "token=", "credential="):
        assert prohibited not in source


def test_pidnet_notebook_is_valid_execution_only_wrapper() -> None:
    payload = json.loads(PIDNET_NOTEBOOK.read_text(encoding="utf-8"))

    assert payload["nbformat"] == 4
    assert payload["nbformat_minor"] >= 5
    assert all(not cell.get("outputs") for cell in payload["cells"])
    source = "\n".join(line for cell in payload["cells"] for line in cell.get("source", []))
    assert "4c158cf24ce432f0a8cb43364fae38d93cee0dc3" in source
    assert "PIDNet_S_Cityscapes_val.pt" in source
    assert "REPLACE_WITH_DOWNLOAD_DATE_YYYY_MM_DD" in source
    assert "REPLACE_WITH_CHECKOUT_ACCESS_DATE_YYYY_MM_DD" in source
    assert "frankfurt_000000_002196_leftImg8bit.png" not in source
    assert "files.upload()" in source
    assert "spike_config.checkpoint.filename" in source
    assert "spike_config.checkpoint.sha256" in source
    assert "actual_checkpoint_sha256 != spike_config.checkpoint.sha256" in source
    assert "--sample-access-date" in source
    assert "--expected-checkpoint-sha256" not in source
    assert "--image-root" not in source
    assert "scripts/run_pidnet_spike.py" in source
    assert (
        'EDGEGUARD_REPOSITORY_URL = "https://github.com/emrealmaoglu/edgeguard-road.git"' in source
    )
    assert 'EDGEGUARD_BRANCH = "feat/first-vertical-slice"' in source
    assert "edgeguard_commit != EDGEGUARD_EXPECTED_COMMIT" in source
    assert 'os.environ["PYTHONDONTWRITEBYTECODE"] = "1"' in source
    assert "sys.dont_write_bytecode = True" in source
    assert '[sys.executable, "-m", "pip"' in source
    assert '[sys.executable, "-m", "edgeguard"' in source
    assert '[sys.executable, "-m", "pytest"' in source
    assert "sys.path.insert(0, str(SOURCE_ROOT))" in source
    assert "Path(edgeguard.__file__).resolve()" in source
    assert "env=SUBPROCESS_ENV" in source
    assert "capture_output=True" in source
    assert "completed.stdout" in source
    assert "completed.stderr" in source
    assert '"python",' not in source
    assert "rmtree(" not in source
    assert ".unlink(" not in source
    lowered = source.lower()
    for prohibited in ("api_key", "password=", "token=", "credential="):
        assert prohibited not in lowered

    runner_source = PIDNET_RUNNER.read_text(encoding="utf-8")
    assert '"command": ["python", "scripts/run_pidnet_spike.py"]' in runner_source
    assert "list(sys.argv)" not in runner_source


def test_cityscapes_eval_notebook_is_thin_and_claim_safe() -> None:
    payload = json.loads(CITYSCAPES_NOTEBOOK.read_text(encoding="utf-8"))

    assert payload["nbformat"] == 4
    assert payload["nbformat_minor"] >= 5
    assert all(not cell.get("outputs") for cell in payload["cells"])
    source = "\n".join(line for cell in payload["cells"] for line in cell.get("source", []))
    assert "EdgeGuard-Road single-scale PIDNet-S Cityscapes-val evaluation" in source
    assert "does not claim reproduction of the official PIDNet paper protocol" in source
    assert "https://github.com/emrealmaoglu/edgeguard-road.git" in source
    assert "Reviewed Commit C SHA" in source
    assert "configs/cityscapes_eval_colab.yaml" in source
    assert "scripts/verify_pidnet_checkpoint.py" in source
    assert "scripts/prepare_cityscapes.py" in source
    assert "scripts/run_cityscapes_eval.py" in source
    assert "scripts/package_eval_artifacts.py" in source
    assert re.search(r'"--subset-size"\s*,\s*"1"', source)
    assert '"--all"' in source
    assert re.search(r'"--device"\s*,\s*"cuda"', source)
    assert "drive.mount" in source
    assert "files.download" in source
    assert "/Users/emrealmaoglu" not in source
    assert "weights_only=False" not in source
    lowered = source.lower()
    for prohibited in ("api_key", "password=", "token=", "credential="):
        assert prohibited not in lowered


def test_cityscapes_train_notebook_is_thin_and_stops_before_training() -> None:
    payload = json.loads(CITYSCAPES_TRAIN_NOTEBOOK.read_text(encoding="utf-8"))

    assert payload["nbformat"] == 4
    assert payload["nbformat_minor"] >= 5
    assert all(not cell.get("outputs") for cell in payload["cells"])
    source = "\n".join(line for cell in payload["cells"] for line in cell.get("source", []))
    assert "Reviewed EG-DATA-002 commit SHA" in source
    assert "https://github.com/emrealmaoglu/edgeguard-road.git" in source
    assert 'drive.mount("/content/drive")' in source
    assert 'Path("/content/drive/MyDrive/EdgeGuard")' in source
    assert "private_inputs/leftImg8bit_trainvaltest.zip" in source
    assert "private_inputs/gtFine_trainvaltest.zip" in source
    assert "datasets/cityscapes/fine/v1" in source
    assert "manifests/cityscapes/fine/v1" in source
    assert "scripts/prepare_cityscapes.py" in source
    assert '"--split"' in source and '"train"' in source
    assert "--verify-only" in source
    assert "recommended_pending_human_approval" in source
    assert "files.download" in source
    assert "MMSegmentation" not in source
    assert "run_cityscapes_eval.py" not in source
    assert "scripts/train.py" not in source
    assert "SMIYC" in source
    assert "/Users/" not in source
    lowered = source.lower()
    for prohibited in ("api_key", "password=", "token=", "credential="):
        assert prohibited not in lowered
