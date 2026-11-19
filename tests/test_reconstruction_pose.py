"""Independent pose benchmark parsing, gauge, coverage, provenance and units."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
rotation_type = pytest.importorskip("scipy.spatial.transform").Rotation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("independent_reconstruction_pose", ROOT / "bench/reconstruction_pose.py")
pose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pose)


def _pose(center, rotation):
    return np.column_stack((rotation, -rotation @ np.asarray(center)))


def _par(path, records):
    intrinsic = np.array([[500., 2., 320.], [0., 510., 240.], [0., 0., 1.]])
    lines = [str(len(records))]
    for name, extrinsic in records.items():
        numbers = [*intrinsic.ravel(), *extrinsic[:, :3].ravel(), *extrinsic[:, 3]]
        lines.append(name + " " + " ".join(format(v, ".17g") for v in numbers))
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def sample_run(tmp_path, monkeypatch):
    run = tmp_path / "saved-run"
    (run / "00_input").mkdir(parents=True)
    model = run / "01_sfm/sparse/0"
    model.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (model / name).write_bytes(name.encode())
    names = [f"{i:06d}.jpg" for i in range(4)]
    official = [f"view-{i}.png" for i in range(4)]
    manifest = [{"name": name, "source": "C:\\data\\official\\" + source}
                for name, source in zip(names, official)]
    (run / "00_input/input_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "report.json").write_text(json.dumps({"stages": {"sfm": {"model": "01_sfm/sparse/0"}}}), encoding="utf-8")
    centers = np.array([[-1., .2, .3], [.3, 1., -.5], [1., -.4, .8], [2., .7, 1.5]])
    rotations = rotation_type.from_rotvec([[.2, -.4, .1], [-.1, .3, .7], [.8, .2, -.3], [-.4, -.2, .6]]).as_matrix()
    expected = [_pose(c, r) for c, r in zip(centers, rotations)]
    gauge = rotation_type.from_euler("xyz", [.4, -.7, 1.1]).as_matrix()
    scale, offset = 2.7, np.array([.3, -1.2, 2.1])
    native = {name: _pose(gauge.T @ (c-offset)/scale, r @ gauge)
              for name, c, r in zip(names, centers, rotations)}
    calibration = _par(tmp_path / "camera_par.txt", dict(zip(official, expected)))
    monkeypatch.setattr(pose, "_read_native_poses", lambda path: native)
    return run, calibration, native, manifest, gauge, scale, offset


def test_full_nonaxis_sim3_uses_existing_math_and_dataset_units(sample_run):
    run, calibration, _, _, gauge, scale, offset = sample_run
    result = pose.evaluate_pose_run(run, calibration)
    similarity = result["evaluation_similarity"]
    assert similarity["method"] == "all_camera_centers_umeyama"
    assert similarity["scale"] == pytest.approx(scale)
    np.testing.assert_allclose(similarity["rotation"], gauge, atol=1e-12)
    np.testing.assert_allclose(similarity["translation"], offset, atol=1e-12)
    assert result["pose"]["rotation_degrees"]["max"] < 1e-5
    assert result["pose"]["center_translation_dataset_units"]["max"] < 1e-12
    assert result["conventions"]["length_units"] == similarity["coordinate_units"] == "dataset"
    assert not result["conventions"]["absolute_metric_scale_observable"]
    assert "translation_m_after_sim3" not in json.dumps(result)
    assert result["coverage"]["common_calibrated_registered_images"] == 4
    assert result["coverage"]["evaluated_all_calibrations"]
    assert len(result["relative_pairs"]) == result["gauge_invariant_relative_rotation_degrees"]["samples"] == 6
    assert result["gauge_invariant_relative_rotation_degrees"]["max"] < 1e-5
    assert result["fingerprints"]["source_sha256"]["reconstruction_truth.py"] == pose._sha256(ROOT / "bench/reconstruction_truth.py")


def test_relative_rotations_do_not_hide_a_common_center_orientation_inconsistency(sample_run):
    run, calibration, native, _, _, _, _ = sample_run
    common = rotation_type.from_rotvec([.01, -.02, .03]).as_matrix()
    for name, value in native.items():
        native[name] = _pose(pose.truth.camera_center(value), value[:, :3] @ common)
    result = pose.evaluate_pose_run(run, calibration)
    assert result["pose"]["rotation_degrees"]["mean"] > 2
    assert result["gauge_invariant_relative_rotation_degrees"]["max"] < 1e-5


def test_original_native_model_precedes_rectified_report(sample_run):
    run, _, _, _, _, _, _ = sample_run
    rectified = run / "02_undistorted/sparse"
    rectified.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin"):
        (rectified / name).write_bytes(b"rectified")
    chosen, selection = pose.select_native_model(run, {"stages": {"sfm": {"model": str(rectified)}}})
    assert chosen == run / "01_sfm/sparse/0"
    assert selection == "original_01_sfm_discovered"


def test_ambiguous_original_models_are_not_silently_selected(sample_run):
    run, _, _, _, _, _, _ = sample_run
    second = run / "01_sfm/sparse/1"
    second.mkdir()
    for name in ("cameras.bin", "images.bin"):
        (second / name).write_bytes(b"other component")
    with pytest.raises(ValueError, match="ambiguous"):
        pose.select_native_model(run, {})


@pytest.mark.parametrize("report", [{"stages": []}, {"stages": {"sfm": []}}])
def test_invalid_report_structure_is_a_controlled_error(sample_run, report):
    run, _, _, _, _, _, _ = sample_run
    with pytest.raises(ValueError, match="must be an object"):
        pose.select_native_model(run, report)


def test_invalid_components_structure_is_a_controlled_error(sample_run):
    run, _, _, _, _, _, _ = sample_run
    (run / "01_sfm/sparse/components.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="components.json must be an object"):
        pose.select_native_model(run, {})


def test_native_name_mismatch_rejected_instead_of_dropping_it(sample_run):
    run, calibration, native, _, _, _, _ = sample_run
    native["unrelated.jpg"] = native.pop("000003.jpg")
    with pytest.raises(ValueError, match="do not match input manifest"):
        pose.evaluate_pose_run(run, calibration)


def test_missing_calibration_is_coverage_and_less_than_three_is_an_error(sample_run):
    run, calibration, native, _, _, _, _ = sample_run
    del native["000003.jpg"]
    result = pose.evaluate_pose_run(run, calibration)
    assert result["coverage"]["common_calibrated_registered_images"] == 3
    assert result["coverage"]["unregistered_input_native_names"] == ["000003.jpg"]
    assert result["coverage"]["calibrations_without_registered_pose"] == ["view-3.png"]
    assert not result["coverage"]["evaluated_all_calibrations"]
    del native["000002.jpg"]
    with pytest.raises(ValueError, match="at least three common"):
        pose.evaluate_pose_run(run, calibration)


@pytest.mark.parametrize("field", ["name", "source"])
def test_duplicate_manifest_mapping_rejected(sample_run, field):
    run, calibration, _, manifest, _, _, _ = sample_run
    manifest[1][field] = manifest[0][field]
    (run / "00_input/input_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate manifest"):
        pose.evaluate_pose_run(run, calibration)


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "duplicate calibration"), ("count", "declared count"),
    ("few_numbers", "21 numbers"), ("nonfinite", "non-finite"),
    ("bad_rotation", "proper orthonormal"), ("reflection", "proper orthonormal"),
    ("bad_focal", "invalid K"), ("bad_K_bottom", "invalid K"),
    ("name_path", "basename"),
])
def test_bad_calibration_rejected(sample_run, mutation, match):
    _, calibration, _, _, _, _, _ = sample_run
    lines = calibration.read_text().splitlines()
    values = lines[1].split()
    if mutation == "duplicate":
        lines[2] = lines[1]
    elif mutation == "count":
        lines[0] = "5"
    elif mutation == "few_numbers":
        lines[1] = " ".join(values[:-1])
    elif mutation == "nonfinite":
        values[-1] = "nan"
    elif mutation in ("bad_rotation", "reflection"):
        rotation = np.diag([-1., 1., 1.]) if mutation == "reflection" else np.eye(3)*2
        values[10:19] = [str(v) for v in rotation.ravel()]
    elif mutation == "bad_focal":
        values[1] = "0"
    elif mutation == "bad_K_bottom":
        values[7] = "1"
    elif mutation == "name_path":
        values[0] = "folder/view.png"
    if mutation not in ("duplicate", "count", "few_numbers"):
        lines[1] = " ".join(values)
    calibration.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        pose.parse_calibration(calibration)


def test_collinear_centers_do_not_switch_to_an_orientation_gauge(sample_run):
    run, calibration, native, _, _, _, _ = sample_run
    for i, name in enumerate(native):
        native[name] = _pose([i, 0., 0.], np.eye(3))
    with pytest.raises(ValueError, match="center-only Sim3"):
        pose.evaluate_pose_run(run, calibration)


def test_cli_writes_only_the_new_result_and_protects_inputs(sample_run, tmp_path):
    run, calibration, _, _, _, _, _ = sample_run
    output = tmp_path / "new-result/pose.json"
    before = pose._sha256(calibration)
    assert pose.main(["--run", str(run), "--calibration", str(calibration), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["schema"] == "lingshu.independent-reconstruction-pose/1"
    assert pose.main(["--run", str(run), "--calibration", str(calibration), "--output", str(output)]) == 0
    with pytest.raises(SystemExit) as error:
        pose.main(["--run", str(run), "--calibration", str(calibration), "--output", str(calibration)])
    assert error.value.code == 2
    assert pose._sha256(calibration) == before


def test_cli_refuses_existing_image_without_overwriting_it(sample_run, capsys):
    run, calibration, _, _, _, _, _ = sample_run
    image = run / "00_input/images/000000.jpg"
    image.parent.mkdir()
    content = b"\xff\xd8\xff original input image bytes \xff\xd9"
    image.write_bytes(content)
    with pytest.raises(SystemExit) as error:
        pose.main(["--run", str(run), "--calibration", str(calibration), "--output", str(image)])
    assert error.value.code == 2
    assert "existing output must be this evaluator's JSON receipt" in capsys.readouterr().err
    assert image.read_bytes() == content
