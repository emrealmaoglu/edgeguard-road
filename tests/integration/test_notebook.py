"""Static checks for the thin Colab execution wrapper."""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "notebooks/colab/00_environment_and_smoke.ipynb"
PIDNET_NOTEBOOK = REPO_ROOT / "notebooks/colab/01_pidnet_single_image_spike.ipynb"
CITYSCAPES_NOTEBOOK = REPO_ROOT / "notebooks/colab/02_pidnet_cityscapes_val_eval.ipynb"
CITYSCAPES_TRAIN_NOTEBOOK = REPO_ROOT / "notebooks/colab/03_prepare_cityscapes_fine_train.ipynb"
SEMANTIC_STACK_NOTEBOOK = REPO_ROOT / "notebooks/colab/04_colab_semantic_compatibility_probe.ipynb"
SEMANTIC_SMOKE_NOTEBOOK = REPO_ROOT / "notebooks/colab/05_semantic_five_model_smoke.ipynb"
ACQUISITION_NOTEBOOK = REPO_ROOT / "notebooks/colab/06_acquire_edgeguard_datasets.ipynb"
PIDNET_RUNNER = REPO_ROOT / "scripts/run_pidnet_spike.py"
CANONICAL_NOTEBOOKS = {
    "00_campaign_control.ipynb",
    "10_semantic_campaign.ipynb",
    "20_ood_calibration_risk.ipynb",
    "30_detection_temporal_fusion.ipynb",
    "40_export_and_reporting.ipynb",
}


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


def test_only_canonical_campaign_notebooks_are_active() -> None:
    active: set[str] = set()
    deprecated: set[str] = set()
    for path in (REPO_ROOT / "notebooks/colab").glob("*.ipynb"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        first_markdown = next(cell for cell in payload["cells"] if cell["cell_type"] == "markdown")
        source = "".join(first_markdown["source"])
        if "DEPRECATED — NON-CANONICAL" in source:
            deprecated.add(path.name)
        else:
            active.add(path.name)
    assert active == CANONICAL_NOTEBOOKS
    assert len(deprecated) == 7


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


def test_semantic_stack_notebook_is_thin_exact_commit_gated_and_stops_before_data() -> None:
    payload = json.loads(SEMANTIC_STACK_NOTEBOOK.read_text(encoding="utf-8"))

    assert payload["nbformat"] == 4
    assert payload["nbformat_minor"] >= 5
    assert all(not cell.get("outputs") for cell in payload["cells"])
    source = "\n".join(line for cell in payload["cells"] for line in cell.get("source", []))
    assert "REPLACE_WITH_REVIEWED_LOCAL_FIRST_COMMIT_SHA" in source
    assert 're.fullmatch(r"[0-9a-f]{40}"' in source
    assert 'checkout", "--detach", EDGEGUARD_EXPECTED_COMMIT' in source
    assert 'status", "--porcelain=v1' in source
    assert "install_semantic_stack.py" in source
    assert "--runtime-current-root" in source
    assert "--runtime-py311-root" in source
    assert "--cache-root" in source
    assert "bootstrap_failure.json" in source
    assert "five_model_probe" in source
    assert "checkpoint_resume_verified" in source
    assert "drive.mount" not in source
    assert "private_inputs" not in source
    assert "stage_cityscapes_training.py" not in source
    assert "RoadAnomaly21" not in source
    assert "RoadObstacle21" not in source
    assert "/Users/" not in source
    lowered = source.lower()
    for prohibited in ("api_key", "password=", "token=", "credential="):
        assert prohibited not in lowered


def test_semantic_smoke_notebook_is_thin_identity_gated_and_stops_before_screening() -> None:
    payload = json.loads(SEMANTIC_SMOKE_NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(line for cell in payload["cells"] for line in cell.get("source", []))

    assert payload["nbformat"] == 4
    assert all(not cell.get("outputs") for cell in payload["cells"])
    assert "REPLACE_WITH_REVIEWED_LOCAL_FIRST_COMMIT_SHA" in source
    assert "install_semantic_stack.py" not in source
    assert "compatibility_receipt.json" in source
    assert "checkpoint_resume_verified" in source
    assert source.index("compatibility_receipt.json") < source.index("drive.mount")
    assert "inventory_edgeguard_storage.py" in source
    assert "--require-cityscapes-reusable" in source
    assert "rebuild_cityscapes_splits.py" in source
    assert "stage_cityscapes_training.py" in source
    assert "run_semantic_smoke.py" in source
    assert "split-policy-v1" in source
    assert "ready_for_common_screening" in source
    assert "--dry-run" in source
    assert "--create-bundle" not in source
    assert "Do not start common screening" in source
    assert "/Users/" not in source
    for prohibited in ("api_key", "password=", "token=", "credential="):
        assert prohibited not in source.lower()


def test_dataset_acquisition_notebook_is_thin_and_runtime_secret_only() -> None:
    payload = json.loads(ACQUISITION_NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(line for cell in payload["cells"] for line in cell.get("source", []))

    assert payload["nbformat"] == 4
    assert all(not cell.get("outputs") for cell in payload["cells"])
    assert "REPLACE_WITH_REVIEWED_EG_SEG_002_COMMIT_SHA" in source
    assert "acquire_edgeguard_datasets.py" in source
    assert "--list" in source
    assert "required artifacts or pinned generator identity" in source
    assert "artifact completion and dataset readiness remain separate" in source
    assert '"--dataset-id"' not in source
    assert "RoadAnomaly21" not in source
    assert "RoadObstacle21" not in source
    assert "Lost & Found" in source
    assert "/Users/" not in source
    for prohibited in ("api_key", "password=", "token=", "credential="):
        assert prohibited not in source.lower()
