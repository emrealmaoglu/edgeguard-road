"""Drive every panel page against a fake Streamlit, with and without records.

The panel is what gets recorded, so a crash in it is not a failed test run -- it is a
failed presentation, with the device already set up and the camera rolling. It also has no
import-time entry point to exercise: `main()` needs a real `streamlit`. What it does have
is `render_*` functions that take `st` as a parameter, which is exactly enough to drive
them from here.

Both halves matter. With records, a page must produce output. With none, it must still
finish without raising -- the whole design of `load_group`/`show_image` is that absent
evidence simply does not appear, and that promise is only worth something if it holds on
every page.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "presentation_app.py"


def _load_panel() -> Any:
    spec = importlib.util.spec_from_file_location("presentation_app", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


panel = _load_panel()


class FakeStreamlit:
    """Record what a page emitted, and accept the whole surface the pages use."""

    def __init__(self, calls: list[tuple[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, Any]] = [] if calls is None else calls

    def _record(self, name: str):
        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args if len(args) != 1 else args[0]))
            return None

        return call

    def __getattr__(self, name: str) -> Any:
        if name in {"columns"}:
            # Columns record into the same list as the page: content placed in a column is
            # still content on the page, and a fake that dropped it would let a metric
            # disappear without failing anything.
            return lambda spec, **kwargs: [FakeStreamlit(self.calls) for _ in range(spec)]
        if name in {"selectbox", "radio"}:
            return lambda label, options, **kwargs: list(options)[0]
        if name in {"line_chart", "bar_chart", "area_chart"}:
            return self._record(name)
        return self._record(name)

    @property
    def emitted(self) -> list[str]:
        return [name for name, _ in self.calls]


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


@pytest.fixture
def populated_root(tmp_path: Path) -> Path:
    """A results tree with one record per group, shaped like the real ones."""
    _write(
        tmp_path / "accuracy" / "pidnet_s.json",
        '{"model": "pidnet_s", "mIoU": 0.6634, "ece": 0.0323, "frames": 120,'
        ' "per_class_iou": {"road": 0.97}}',
    )
    _write(
        tmp_path / "open_set" / "pidnet_s.json",
        '{"model": "pidnet_s", "auroc": 0.667, "average_precision": 0.347, "fpr_at_95_tpr": 0.8}',
    )
    _write(
        tmp_path / "jetson" / "pidnet_s.json",
        '{"model": "pidnet_s", "median_latency_ms": 58.39, "sustained_fps": 17.1,'
        ' "joule_per_frame": 1.02, "mean_power_w": 17.4}',
    )
    _write(
        tmp_path / "acdc" / "pidnet_s_night.json",
        '{"model": "pidnet_s", "condition": "night", "mIoU": 0.20, "retention": 0.297}',
    )
    _write(
        tmp_path / "risk" / "frame_000.json",
        '{"regions": [{"class_name": "car", "corridor_distance_pixels": 12.0,'
        ' "risk": {"total_risk_score": 0.42, "risk_category": "medium",'
        ' "explanation": ["anomaly_score"]}}], "zero_weighted_features": ["detector_overlap"]}',
    )
    _write(
        tmp_path / "drivable" / "pidnet_s.json",
        '{"model": "pidnet_s", "frames": 200, "output_stride": 8,'
        ' "road_mask": {"road_iou": 0.9614, "road_boundary_f1_tolerance_1px": 0.1604,'
        ' "road_boundary_f1_tolerance_8px": 0.5651, "false_drivable_rate": 0.0076},'
        ' "ego_corridor": {"road_iou": 0.9557, "road_boundary_f1_tolerance_1px": 0.1530,'
        ' "road_boundary_f1_tolerance_8px": 0.5388, "false_drivable_rate": 0.0058}}',
    )
    _write(
        tmp_path / "temporal" / "pidnet_s.json",
        '{"model": "pidnet_s", "sequence": "stuttgart_00", "frames": 150,'
        ' "region_observations": 5208, "tracks_with_a_fair_chance": 1382,'
        ' "transient_track_fraction": 0.5007, "median_track_lifetime_frames": 1.0,'
        ' "top_region_change_fraction": 0.2}',
    )
    _write(tmp_path / "shift_response.json", '{"model": "pidnet_s", "ratio": 1.0}')
    return tmp_path


def _pages(root: Path, groups: dict[str, dict]) -> list[tuple[str, Any]]:
    return [
        ("problem", lambda st: panel.render_problem(st, root)),
        (
            "comparison",
            lambda st: panel.render_comparison(
                st, groups["accuracy"], groups["open_set"], groups["jetson"], root
            ),
        ),
        ("open_set", lambda st: panel.render_open_set(st, groups["open_set"], root)),
        (
            "uncertainty",
            lambda st: panel.render_uncertainty(
                st, groups["accuracy"], groups["open_set"], groups["acdc"], groups["shift"], root
            ),
        ),
        (
            "risk",
            lambda st: panel.render_risk(
                st, groups["risk"], groups["drivable"], groups["temporal"], root
            ),
        ),
        ("edge", lambda st: panel.render_edge(st, groups["jetson"], {}, root)),
        ("limits", lambda st: panel.render_limits(st)),
    ]


def _groups(root: Path) -> dict[str, dict]:
    return {
        name: panel.load_group(root / name)
        for name in ("accuracy", "open_set", "jetson", "acdc", "risk", "drivable", "temporal")
    } | {"shift": panel.load_json(root / "shift_response.json")}


@pytest.mark.parametrize(
    "page",
    [
        name
        for name, _ in _pages(
            Path("/"),
            dict.fromkeys(
                (
                    "accuracy",
                    "open_set",
                    "jetson",
                    "acdc",
                    "risk",
                    "drivable",
                    "temporal",
                    "shift",
                ),
                {},
            ),
        )
    ],
)
def test_every_page_renders_with_records(page: str, populated_root: Path) -> None:
    groups = _groups(populated_root)
    render = dict(_pages(populated_root, groups))[page]
    st = FakeStreamlit()

    render(st)

    assert st.emitted, f"{page} produced nothing"


@pytest.mark.parametrize(
    "page",
    [
        name
        for name, _ in _pages(
            Path("/"),
            dict.fromkeys(
                (
                    "accuracy",
                    "open_set",
                    "jetson",
                    "acdc",
                    "risk",
                    "drivable",
                    "temporal",
                    "shift",
                ),
                {},
            ),
        )
    ],
)
def test_every_page_survives_missing_records(page: str, tmp_path: Path) -> None:
    """Absent evidence must not appear -- and must not raise, which is the harder half."""
    groups = _groups(tmp_path)
    render = dict(_pages(tmp_path, groups))[page]

    render(FakeStreamlit())


def test_the_drivable_table_reports_both_boundary_tolerances(populated_root: Path) -> None:
    """One tolerance alone misleads: 1 px asks a stride-8 mask for a precision its
    resolution forbids. The page has to show both or it argues the model failed.
    """
    st = FakeStreamlit()

    panel.render_drivable(st, panel.load_group(populated_root / "drivable"))

    tables = [value for name, value in st.calls if name == "markdown" and isinstance(value, str)]
    assert any("0.1604" in table and "0.5651" in table for table in tables)
    assert any("0.0076" in table for table in tables)


def test_a_binary_sidecar_next_to_the_records_does_not_take_the_panel_down(
    populated_root: Path,
) -> None:
    """macOS archives carry `._name.json` AppleDouble files holding resource-fork bytes,
    and they match the same glob the records do.
    """
    (populated_root / "drivable" / "._pidnet_s.json").write_bytes(b"\x00\x05\x16\x07\xff")

    loaded = panel.load_group(populated_root / "drivable")

    assert set(loaded) == {"pidnet_s"}


def test_the_risk_page_answers_its_own_zero_weight_notice(populated_root: Path) -> None:
    """The page tells the reader `temporal_persistence` was excluded for want of a second
    frame. That invites the follow-up -- what is being given up? -- and the sequence
    measurement is the answer, so it has to appear on the same page as the question.
    """
    st = FakeStreamlit()

    panel.render_temporal(st, panel.load_group(populated_root / "temporal"))

    text = " ".join(str(value) for _, value in st.calls)
    assert "50.1" in text
    assert "20.0" in text
    # The limit travels with the finding: transient is not the same as false.
    assert "etiketsiz" in text
