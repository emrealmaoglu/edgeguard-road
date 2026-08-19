"""Exercise the bundle builder against a real results tree.

The bundle is how measurements reach the device, and its failure mode is silence: the
panel renders a smaller page rather than complaining, so a result left behind disappears
without anyone noticing until the recording. These tests pin the two things that prevent
that -- the manifest names what was absent, and `--require` turns a named absence into an
error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.build_jetson_bundle import main  # noqa: E402


def _results(root: Path) -> Path:
    for group, name in (("accuracy", "pidnet_s.json"), ("drivable", "pidnet_s.json")):
        target = root / group / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"model": "pidnet_s"}', encoding="utf-8")
    (root / "shift_response.json").write_text('{"ratio": 1.0}', encoding="utf-8")
    return root


def _run(argv: list[str]) -> None:
    original = sys.argv
    sys.argv = ["build_jetson_bundle.py", *argv]
    try:
        main()
    finally:
        sys.argv = original


def test_the_bundle_carries_the_records_and_names_what_it_did_not_find(tmp_path: Path) -> None:
    results = _results(tmp_path / "results")
    output = tmp_path / "bundle"

    _run(["--results", str(results), "--output", str(output)])

    manifest = json.loads((output / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert (output / "drivable" / "pidnet_s.json").is_file()
    assert (output / "shift_response.json").is_file()
    assert "drivable" in manifest["present"]
    # The groups that were never produced have to be visible somewhere that gets read.
    assert "jetson" in manifest["absent"]
    assert manifest["required_and_missing"] == []


def test_a_required_result_that_is_missing_stops_the_build(tmp_path: Path) -> None:
    results = _results(tmp_path / "results")

    with pytest.raises(ValueError, match="jetson"):
        _run(
            [
                "--results",
                str(results),
                "--output",
                str(tmp_path / "bundle"),
                "--require",
                "jetson",
            ]
        )


def test_a_required_result_that_is_present_does_not(tmp_path: Path) -> None:
    results = _results(tmp_path / "results")

    _run(
        [
            "--results",
            str(results),
            "--output",
            str(tmp_path / "bundle"),
            "--require",
            "drivable",
            "--require",
            "accuracy",
        ]
    )


def test_appledouble_sidecars_are_left_behind(tmp_path: Path) -> None:
    """The panel already learned this once: `._name.json` matches the records' glob but
    holds resource-fork bytes, and macOS creates them whenever the tree crosses an archive.
    """
    results = _results(tmp_path / "results")
    (results / "drivable" / "._pidnet_s.json").write_bytes(b"\x00\x05\x16\x07")
    (results / "drivable" / ".DS_Store").write_bytes(b"\x00\x00\x00\x01")
    output = tmp_path / "bundle"

    _run(["--results", str(results), "--output", str(output)])

    assert [path.name for path in sorted((output / "drivable").iterdir())] == ["pidnet_s.json"]


def test_rebuilding_does_not_keep_a_result_that_was_removed(tmp_path: Path) -> None:
    """A stale record surviving a rebuild is the same silent failure from the other side:
    the panel would show a number no longer backed by the results tree.
    """
    results = _results(tmp_path / "results")
    output = tmp_path / "bundle"
    _run(["--results", str(results), "--output", str(output)])
    assert (output / "drivable" / "pidnet_s.json").is_file()

    (results / "drivable" / "pidnet_s.json").unlink()
    _run(["--results", str(results), "--output", str(output)])

    assert not (output / "drivable" / "pidnet_s.json").exists()


def test_the_archive_is_written_when_asked(tmp_path: Path) -> None:
    results = _results(tmp_path / "results")
    archive = tmp_path / "eg_presentation_bundle.tgz"

    _run(
        [
            "--results",
            str(results),
            "--output",
            str(tmp_path / "bundle"),
            "--archive",
            str(archive),
        ]
    )

    assert archive.is_file() and archive.stat().st_size > 0


def test_a_device_run_directory_is_filed_into_the_groups_the_panel_reads(tmp_path: Path) -> None:
    """The device writes one flat directory; the panel reads three groups. Sorting that by
    hand under deadline is how a telemetry log gets left behind.
    """
    results = _results(tmp_path / "results")
    run = tmp_path / "device-run"
    run.mkdir()
    for name in (
        "pidnet_s_reference_benchmark.json",
        "ddrnet_23_slim_reference_benchmark.json",
        "pidnet_s_reference_stage_profile.json",
        "pidnet_s_reference_tegrastats.log",
        "pidnet_s_reference.plan",
        "pidnet_s_reference_build.json",
    ):
        (run / name).write_text("{}", encoding="utf-8")
    output = tmp_path / "bundle"

    _run(
        [
            "--results",
            str(results),
            "--output",
            str(output),
            "--jetson-run",
            str(run),
            "--require",
            "jetson",
            "--require",
            "telemetry",
        ]
    )

    assert (output / "jetson" / "pidnet_s_reference_benchmark.json").is_file()
    assert (output / "jetson" / "ddrnet_23_slim_reference_benchmark.json").is_file()
    assert (output / "profile" / "pidnet_s_reference_stage_profile.json").is_file()
    assert (output / "telemetry" / "pidnet_s_reference_tegrastats.log").is_file()
    # The engine and its build record are device build artefacts, not results; they are
    # large and the panel never reads them.
    assert not (output / "jetson" / "pidnet_s_reference.plan").exists()
    assert not any(path.name.endswith("_build.json") for path in (output / "jetson").iterdir())
