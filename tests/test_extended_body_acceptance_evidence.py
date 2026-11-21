"""A successful process exit must not hide missing, skipped or substituted proof."""
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools/testing/extended_body_acceptance.py"
spec = importlib.util.spec_from_file_location("extended_body_acceptance", SOURCE)
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


def test_verifier_selftest_exercises_false_green_rejection():
    report = acceptance.selftest()
    assert set(report["rejected"]) == {
        "missing_result", "empty_result", "omitted_guard", "skipped_guard", "failed_guard",
        "missing_baseline", "tampered_baseline",
        "clean_source_commit_changed",
    }
    assert report["positive_controls_passed"] == 3


def test_zero_exit_style_summary_does_not_replace_executed_cases(tmp_path):
    report = tmp_path / "summary.xml"
    report.write_text('<testsuites tests="4" failures="0" errors="0" skipped="0"/>', encoding="utf-8")
    with pytest.raises(acceptance.AcceptanceError, match="no executed"):
        acceptance.validate_junit(report, {"test.guard::required"})


def test_baseline_file_set_cannot_hide_an_extra_import_module(tmp_path):
    root = tmp_path / "baseline"
    source = root / "lingshu/world/physics.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"original implementation\n")
    expected = {"lingshu/world/physics.py": acceptance.blob_sha(source.read_bytes())}
    assert acceptance.verify_baseline_export(root, expected)["files"] == 1
    (root / "math.py").write_text("substitute = True\n", encoding="utf-8")
    with pytest.raises(acceptance.AcceptanceError, match="file set differs"):
        acceptance.verify_baseline_export(root, expected)


def test_failure_of_an_extra_case_invalidates_the_whole_result(tmp_path):
    report = tmp_path / "summary.xml"
    report.write_text('<testsuite><testcase classname="test.guard" name="required"/>'
                      '<testcase classname="test.guard" name="other"><failure/></testcase>'
                      '</testsuite>', encoding="utf-8")
    with pytest.raises(acceptance.AcceptanceError, match="other"):
        acceptance.validate_junit(report, {"test.guard::required"})
