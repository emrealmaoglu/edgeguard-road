"""Unit tests for fixed-source PIDNet spike safeguards."""

import hashlib
import subprocess
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from edgeguard.models.pidnet_spike import (
    NormalizedStateDict,
    PIDNetSpikeError,
    _load_state_dict_strict,
    difference_summary,
    preprocess_pidnet_rgb,
    verify_checkpoint_file,
    verify_upstream_checkout,
)
from edgeguard.models.pidnet_spike import (
    normalize_state_dict_keys as _normalize_state_dict_keys,
)
from edgeguard.models.pidnet_spike import (
    validate_state_dict_shapes as _validate_state_dict_shapes,
)

UPSTREAM_URL = "https://github.com/XuJiacong/PIDNet.git"


def _tensor(shape: tuple[int, ...] = (1,)) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def _is_numpy_tensor(value: object) -> bool:
    return isinstance(value, np.ndarray)


def _normalize(state_dict: Mapping[str, object], model_keys: set[str]) -> NormalizedStateDict:
    return _normalize_state_dict_keys(state_dict, model_keys, is_tensor=_is_numpy_tensor)


def _validate_shapes(
    state_dict: Mapping[str, object], model_state_dict: Mapping[str, object]
) -> dict[str, list[int]]:
    return _validate_state_dict_shapes(state_dict, model_state_dict, is_tensor=_is_numpy_tensor)


class TensorLikeNonTensor:
    shape = (19,)
    dtype = np.dtype(np.float32)


def _inference_state(key_count: int = 453) -> dict[str, np.ndarray]:
    return {f"layer_{index:03d}.weight": _tensor() for index in range(key_count)}


def _official_training_state(
    inference_state: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    state = {f"model.{key}": value for key, value in inference_state.items()}
    state.update({f"model.seghead_p.parameter_{index:02d}": _tensor() for index in range(13)})
    state.update({f"model.seghead_d.parameter_{index:02d}": _tensor() for index in range(13)})
    state["sem_loss.criterion.weight"] = _tensor((19,))
    state["sb_loss.criterion.weight"] = _tensor((19,))
    return state


def test_preprocess_pidnet_rgb_preserves_rgb_channel_order() -> None:
    raw = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)

    model_input = preprocess_pidnet_rgb(
        raw,
        height=1,
        width=2,
        pixel_scale=1.0 / 255.0,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    assert model_input.shape == (1, 3, 1, 2)
    np.testing.assert_allclose(model_input[0, :, 0, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(model_input[0, :, 0, 1], [0.0, 0.0, 1.0])


def test_checkpoint_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "PIDNet_S_Cityscapes_val.pt"
    checkpoint.write_bytes(b"checkpoint fixture")

    with pytest.raises(PIDNetSpikeError, match="SHA-256 mismatch"):
        verify_checkpoint_file(
            checkpoint,
            expected_filename=checkpoint.name,
            expected_sha256="0" * 64,
        )


def test_checkpoint_hash_and_filename_are_recorded(tmp_path: Path) -> None:
    checkpoint = tmp_path / "PIDNet_S_Cityscapes_val.pt"
    content = b"checkpoint fixture"
    checkpoint.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest()

    report = verify_checkpoint_file(
        checkpoint,
        expected_filename=checkpoint.name,
        expected_sha256=expected_hash,
    )

    assert report == {
        "filename": checkpoint.name,
        "sha256": expected_hash,
        "size_bytes": len(content),
    }


def test_state_dict_allows_only_exact_or_uniform_known_prefix() -> None:
    model_keys = {"stem.weight", "head.bias"}
    exact = _normalize({"stem.weight": _tensor(), "head.bias": _tensor()}, model_keys)
    prefixed = _normalize(
        {"model.stem.weight": _tensor(), "model.head.bias": _tensor()},
        model_keys,
    )

    assert set(exact.state_dict) == model_keys
    assert exact.transformation_policy == "exact_inference_state_dict"
    assert set(prefixed.state_dict) == model_keys
    assert prefixed.transformation_policy == "removed_uniform_prefix:model."


def test_state_dict_mismatch_is_rejected_without_partial_filtering() -> None:
    with pytest.raises(PIDNetSpikeError, match="refusing partial or unreviewed"):
        _normalize({"model.stem.weight": 1}, {"stem.weight", "head.bias"})


def test_official_training_layout_accepts_exact_481_key_evidence() -> None:
    inference_state = _inference_state()
    training_state = _official_training_state(inference_state)

    normalized = _normalize(training_state, set(inference_state))

    assert len(training_state) == 481
    assert normalized.transformation_policy == "reviewed_official_training_checkpoint"
    assert normalized.raw_checkpoint_key_count == 481
    assert normalized.loaded_inference_key_count == 453
    assert normalized.excluded_auxiliary_key_count == 26


def test_official_training_layout_preserves_all_453_inference_keys() -> None:
    inference_state = _inference_state()

    normalized = _normalize(_official_training_state(inference_state), set(inference_state))

    assert normalized.state_dict.keys() == inference_state.keys()
    assert all(normalized.state_dict[key] is inference_state[key] for key in inference_state)


def test_official_training_layout_ignores_only_two_reviewed_loss_roots() -> None:
    inference_state = _inference_state(2)

    normalized = _normalize(_official_training_state(inference_state), set(inference_state))

    assert normalized.ignored_training_root_keys == (
        "sb_loss.criterion.weight",
        "sem_loss.criterion.weight",
    )


def test_official_training_layout_reports_auxiliary_groups() -> None:
    inference_state = _inference_state(2)

    normalized = _normalize(_official_training_state(inference_state), set(inference_state))

    assert normalized.excluded_auxiliary_group_counts == {
        "seghead_d.": 13,
        "seghead_p.": 13,
    }
    assert normalized.excluded_auxiliary_prefixes == ("seghead_d.", "seghead_p.")


def test_strict_load_contract_is_preserved() -> None:
    class IncompatibleKeys:
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []

    class RecordingModel:
        strict: bool | None = None
        received_keys: set[str] = set()

        def load_state_dict(
            self, state_dict: dict[str, object], *, strict: bool
        ) -> IncompatibleKeys:
            self.strict = strict
            self.received_keys = set(state_dict)
            return IncompatibleKeys()

    model = RecordingModel()
    _load_state_dict_strict(model, {"layer.weight": _tensor()})

    assert model.strict is True
    assert model.received_keys == {"layer.weight"}


def test_official_training_layout_rejects_missing_required_inference_key() -> None:
    inference_state = _inference_state(3)
    training_state = _official_training_state(inference_state)
    del training_state["model.layer_001.weight"]

    with pytest.raises(PIDNetSpikeError, match="inference keys do not match"):
        _normalize(training_state, set(inference_state))


def test_official_training_layout_rejects_unexpected_extra_inference_key() -> None:
    inference_state = _inference_state(3)
    training_state = _official_training_state(inference_state)
    training_state["model.unreviewed.weight"] = _tensor()

    with pytest.raises(PIDNetSpikeError, match="inference keys do not match"):
        _normalize(training_state, set(inference_state))


def test_official_training_layout_rejects_missing_auxiliary_key() -> None:
    inference_state = _inference_state(3)
    training_state = _official_training_state(inference_state)
    del training_state["model.seghead_p.parameter_00"]

    with pytest.raises(PIDNetSpikeError, match="seghead_p.*expected 13"):
        _normalize(training_state, set(inference_state))


def test_official_training_layout_rejects_unexpected_auxiliary_prefix() -> None:
    inference_state = _inference_state(3)
    training_state = _official_training_state(inference_state)
    training_state["model.seghead_x.parameter_00"] = _tensor()

    with pytest.raises(PIDNetSpikeError, match="inference keys do not match"):
        _normalize(training_state, set(inference_state))


def test_official_training_layout_rejects_missing_loss_root() -> None:
    inference_state = _inference_state(3)
    training_state = _official_training_state(inference_state)
    del training_state["sem_loss.criterion.weight"]

    with pytest.raises(PIDNetSpikeError, match="roots_missing"):
        _normalize(training_state, set(inference_state))


def test_official_training_layout_rejects_unknown_root_key() -> None:
    inference_state = _inference_state(3)
    training_state = _official_training_state(inference_state)
    training_state["optimizer.state"] = _tensor()

    with pytest.raises(PIDNetSpikeError, match="roots_unexpected"):
        _normalize(training_state, set(inference_state))


def test_official_training_layout_rejects_reviewed_loss_shape_mismatch() -> None:
    inference_state = _inference_state(3)
    training_state = _official_training_state(inference_state)
    training_state["sem_loss.criterion.weight"] = _tensor((18,))

    with pytest.raises(PIDNetSpikeError, match="training-only root shape mismatch"):
        _normalize(training_state, set(inference_state))


def test_official_training_layout_rejects_inference_shape_mismatch() -> None:
    inference_state = _inference_state(3)
    training_state = _official_training_state(inference_state)
    training_state["model.layer_001.weight"] = _tensor((2,))
    normalized = _normalize(training_state, set(inference_state))

    with pytest.raises(PIDNetSpikeError, match="parameter shape mismatch"):
        _validate_shapes(normalized.state_dict, inference_state)


@pytest.mark.parametrize(
    "malformed_key",
    ["sem_loss.criterion.weight", "model.seghead_d.parameter_00"],
)
def test_official_training_layout_rejects_malformed_non_tensor_entry(
    malformed_key: str,
) -> None:
    inference_state = _inference_state(3)
    training_state: dict[str, object] = _official_training_state(inference_state)
    training_state[malformed_key] = [0.0] * 19

    with pytest.raises(PIDNetSpikeError, match="is not a tensor"):
        _normalize(training_state, set(inference_state))


def test_official_training_layout_rejects_tensor_like_non_tensor_loss_root() -> None:
    inference_state = _inference_state(3)
    training_state: dict[str, object] = _official_training_state(inference_state)
    training_state["sem_loss.criterion.weight"] = TensorLikeNonTensor()

    with pytest.raises(PIDNetSpikeError, match="is not a tensor"):
        _normalize(training_state, set(inference_state))


def test_official_training_layout_rejects_tensor_like_non_tensor_auxiliary() -> None:
    inference_state = _inference_state(3)
    training_state: dict[str, object] = _official_training_state(inference_state)
    training_state["model.seghead_p.parameter_00"] = TensorLikeNonTensor()

    with pytest.raises(PIDNetSpikeError, match="is not a tensor"):
        _normalize(training_state, set(inference_state))


def test_official_training_layout_rejects_tensor_like_non_tensor_inference() -> None:
    inference_state = _inference_state(3)
    training_state: dict[str, object] = _official_training_state(inference_state)
    training_state["model.layer_001.weight"] = TensorLikeNonTensor()

    with pytest.raises(PIDNetSpikeError, match="is not a tensor"):
        _normalize(training_state, set(inference_state))


def test_state_dict_shape_mismatch_is_rejected() -> None:
    model_state = {"stem.weight": np.zeros((2, 3), dtype=np.float32)}

    with pytest.raises(PIDNetSpikeError, match="parameter shape mismatch"):
        _validate_shapes({"stem.weight": np.zeros((3, 2), dtype=np.float32)}, model_state)

    manifest = _validate_shapes({"stem.weight": np.zeros((2, 3), dtype=np.float32)}, model_state)
    assert manifest == {"stem.weight": [2, 3]}


def test_upstream_checkout_requires_exact_clean_origin_and_commit(tmp_path: Path) -> None:
    checkout = tmp_path / "pidnet"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    (checkout / "source.txt").write_text("fixed source\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "source.txt"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "fixture"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "add", "origin", UPSTREAM_URL], check=True
    )
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    report = verify_upstream_checkout(
        checkout,
        expected_repository_url=UPSTREAM_URL,
        expected_commit=commit,
    )

    assert report["actual_commit"] == commit
    assert report["git_clean"] is True
    with pytest.raises(PIDNetSpikeError, match="commit mismatch"):
        verify_upstream_checkout(
            checkout,
            expected_repository_url=UPSTREAM_URL,
            expected_commit="0" * 40,
        )


def test_difference_summary_reports_exact_and_changed_arrays() -> None:
    first = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    exact = difference_summary(first, first.copy())
    changed = difference_summary(first, np.array([0.0, 2.0, 2.0], dtype=np.float32))

    assert exact["byte_equal"] is True
    assert exact["max_absolute_difference"] == 0.0
    assert changed["byte_equal"] is False
    assert changed["max_absolute_difference"] == 1.0
