"""Independent benchmark checks: gauge, source pixels, errors and fixed failures."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")


ROOT = Path(__file__).resolve().parents[1]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


truth = _module("independent_reconstruction_truth", ROOT / "bench/reconstruction_truth.py")
generator = _module("independent_truth_generator", ROOT / "tools/reconstruct/generate_truth.py")
asset_check = _module("independent_truth_asset_check", ROOT / "tools/reconstruct/evaluate_truth_assets.py")


def _rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _pose(center, rotation=None):
    rotation = np.eye(3) if rotation is None else rotation
    return np.column_stack((rotation, -rotation @ np.asarray(center)))


@pytest.mark.parametrize("collinear", [False, True])
def test_one_scene_similarity_removes_only_the_global_gauge(collinear):
    centers = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0 if collinear else 1, 0]])
    scale, rotation, offset = 2.4, _rotation_z(.6), np.array([.2, -.7, 1.1])
    source, target = [], []
    for i, center in enumerate(centers):
        r = _rotation_z(.1 * i)
        target.append(_pose(center, r))
        source.append(_pose((center - offset) @ rotation / scale, r @ rotation))
    similarity = truth.fit_similarity(source, target)
    assert similarity["scale"] == pytest.approx(scale)
    np.testing.assert_allclose(similarity["rotation"], rotation, atol=1e-12)
    np.testing.assert_allclose(similarity["translation"], offset, atol=1e-12)
    result = truth.pose_errors(source, target, similarity)
    assert result["rotation_degrees"]["max"] < 1e-5
    assert result["translation_m_after_sim3"]["max"] < 1e-12
    assert similarity["absolute_scale_observable"] is False


def test_collapsed_camera_baseline_does_not_create_a_metric_scale():
    cameras = [_pose([0, 0, 0])] * 3
    with pytest.raises(ValueError, match="baseline is degenerate"):
        truth.fit_similarity(cameras, cameras)


def test_local_pose_errors_remain_after_a_shared_gauge():
    identity = {"scale": 1, "rotation": np.eye(3).tolist(), "translation": [0, 0, 0]}
    result = truth.pose_errors([_pose([.1, 0, 0], _rotation_z(np.radians(2)))],
                               [_pose([0, 0, 0])], identity)
    assert result["rotation_degrees"]["max"] == pytest.approx(2)
    assert result["translation_m_after_sim3"]["max"] == pytest.approx(.1)


def test_full_3d_similarity_preserves_noncommuting_camera_extrinsics():
    rotation_type = pytest.importorskip("scipy.spatial.transform").Rotation
    centers = np.array([[-1, 0, .2], [.2, 1, -.5], [1, -.3, .8], [2, 1, 1.5]])
    scale, offset = 2.7, np.array([.3, -1.2, 2.1])
    gauge = rotation_type.from_euler("xyz", [.4, -.7, 1.1]).as_matrix()
    camera_rotations = rotation_type.from_rotvec(
        [[.2, -.4, .1], [-.1, .3, .7], [.8, .2, -.3], [-.4, -.2, .6]]).as_matrix()
    expected = [_pose(center, rotation) for center, rotation in zip(centers, camera_rotations)]
    predicted = [_pose(gauge.T @ (center - offset) / scale, rotation @ gauge)
                 for center, rotation in zip(centers, camera_rotations)]

    similarity = truth.fit_similarity(predicted, expected)

    np.testing.assert_allclose(similarity["rotation"], gauge, atol=1e-12)
    np.testing.assert_allclose(similarity["translation"], offset, atol=1e-12)
    assert similarity["scale"] == pytest.approx(scale)
    result = truth.pose_errors(predicted, expected, similarity)
    assert result["rotation_degrees"]["max"] < 1e-5
    assert result["translation_m_after_sim3"]["max"] < 1e-12
    raw_points = np.array([[.2, .5, 3.], [-.8, 1.2, 4.5]])
    aligned = truth.transform_points(raw_points, similarity)
    for source, target in zip(predicted, expected):
        source_camera = raw_points @ source[:, :3].T + source[:, 3]
        target_camera = aligned @ target[:, :3].T + target[:, 3]
        np.testing.assert_allclose(target_camera, scale * source_camera, atol=1e-12)


def test_relative_rotation_agreement_keeps_center_orientation_inconsistency_visible():
    rotation_type = pytest.importorskip("scipy.spatial.transform").Rotation
    centers = np.array([[-1, 0, .2], [.2, 1, -.5], [1, -.3, .8], [2, 1, 1.5]])
    common_offset = rotation_type.from_rotvec([np.radians(.8), 0, 0]).as_matrix()
    camera_rotations = rotation_type.from_rotvec(
        [[.2, -.4, .1], [-.1, .3, .7], [.8, .2, -.3], [-.4, -.2, .6]]).as_matrix()
    expected = [_pose(center, rotation) for center, rotation in zip(centers, camera_rotations)]
    predicted = [_pose(center, rotation @ common_offset)
                 for center, rotation in zip(centers, camera_rotations)]
    for i in range(len(centers)):
        for j in range(i):
            source_relative = predicted[i][:, :3] @ predicted[j][:, :3].T
            truth_relative = expected[i][:, :3] @ expected[j][:, :3].T
            np.testing.assert_allclose(source_relative, truth_relative, atol=1e-12)

    similarity = truth.fit_similarity(predicted, expected)
    result = truth.pose_errors(predicted, expected, similarity)

    assert similarity["method"] == "all_camera_centers_umeyama"
    np.testing.assert_allclose(similarity["rotation"], np.eye(3), atol=1e-12)
    assert result["rotation_degrees"]["max"] == pytest.approx(.8, abs=1e-9)
    assert result["translation_m_after_sim3"]["max"] < 1e-12


def test_blender_camera_export_uses_right_down_forward_world_to_camera(monkeypatch):
    rotation_type = pytest.importorskip("scipy.spatial.transform").Rotation
    rotation = rotation_type.from_euler("xyz", [.4, -.7, 1.1]).as_matrix()
    expected = _pose([1.2, -3.4, 2.1], rotation)
    blender_inverse = np.diag([1., -1., -1.]) @ expected
    updates = []
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace(
        context=SimpleNamespace(view_layer=SimpleNamespace(update=lambda: updates.append(True)))))
    camera = SimpleNamespace(
        matrix_world=SimpleNamespace(inverted=lambda: np.vstack((blender_inverse, [0, 0, 0, 1]))),
        data=SimpleNamespace(lens=50, sensor_width=36))

    exported = generator._camera_record(camera, SimpleNamespace(width=640, height=480))

    assert updates == [True]
    np.testing.assert_allclose(exported["cam_from_world"], expected, atol=1e-12)
    focal = 50 / 36 * 640
    np.testing.assert_allclose(exported["intrinsics"], [[focal, 0, 320], [0, focal, 240], [0, 0, 1]])


def test_depth_abs_rel_is_mean_error_against_truth_without_a_per_view_fit():
    result = truth.depth_statistics([2.1, 4.4, np.nan, 0], [2, 4, 8, 1], [1, 1, 1, 1])
    assert result["samples"] == 2
    assert result["mean"] == pytest.approx(.075)


def test_coverage_counts_distinct_truth_pixels_in_the_full_truth_domain():
    mask = np.ones((2, 3), dtype=bool)
    pixels = np.array([[.5, .5], [.6, .6], [1.5, .5], [2.5, 1.5], [99, 99]])
    assert truth.covered_truth_pixels(pixels, [1, 1, 1, 0, 1], mask) == 2
    assert mask.sum() == 6  # Invalid predictions do not shrink this denominator.


def test_depth_acceptance_uses_input_mask_for_confidence_percentile_and_coverage():
    positive, accepted, cutoff = truth.depth_acceptance(
        np.ones((2, 2)), np.array([[10., 20.], [0., 0.]]), 50,
        np.array([[1, 1], [0, 0]], dtype=np.uint8))
    assert cutoff == 15
    assert positive.sum() == 2
    np.testing.assert_array_equal(accepted, [[False, True], [False, False]])
    with pytest.raises(ValueError, match="binary"):
        truth.depth_acceptance(np.ones((2, 2)), np.ones((2, 2)), 25, np.full((2, 2), .5))


def test_rectified_depth_pixels_map_through_original_camera_and_size():
    class OriginalCamera:
        width, height = 100, 80

        @staticmethod
        def img_from_cam(rays):
            assert rays.shape[1] == 3
            return rays[:, :2] / rays[:, 2:3] * [100, 80] + [50, 40]

    pose = {"intrinsics": [[200, 0, 50], [0, 160, 40], [0, 0, 1]], "width": 100, "height": 80}
    camera = {"width": 200, "height": 160}
    # Depth is half the rectified image dimensions, then source is mapped to GT.
    result = truth.source_pixels([[35, 20]], pose, OriginalCamera(), camera, (40, 50))
    np.testing.assert_allclose(result, [[120, 80]])


def test_truth_backprojection_uses_camera_z_and_the_extrinsic_inverse():
    camera = {"intrinsics": [[100, 0, 10], [0, 100, 20], [0, 0, 1]],
              "cam_from_world": _pose([1, 2, 3]).tolist()}
    point = truth.backproject_pixels([[10, 20]], [4], camera)
    np.testing.assert_allclose(point, [[1, 2, 7]])


def test_truth_raycast_parameter_is_forward_z_even_for_off_axis_rotated_rays():
    rotation = _rotation_z(.7)
    camera = {"width": 2, "height": 2,
              "intrinsics": [[1, 0, 1], [0, 1, 1], [0, 0, 1]],
              "cam_from_world": _pose([1, 2, 3], rotation).tolist()}
    rays = asset_check.camera_rays(camera).reshape(-1, 6)
    np.testing.assert_allclose(rays[:, :3], np.tile([1, 2, 3], (4, 1)))
    world_points = rays[:, :3] + 5 * rays[:, 3:]
    camera_points = world_points @ rotation.T - rotation @ np.array([1, 2, 3])
    np.testing.assert_allclose(camera_points[:, 2], 5)
    assert np.all(np.linalg.norm(rays[:, 3:], axis=1) > 1)


def test_sparse_truth_raycast_respects_continuous_subpixel_rays():
    o3d = pytest.importorskip("open3d")
    raycaster = o3d.t.geometry.RaycastingScene(nthreads=1)
    vertices = np.array([[-10, -10, 5], [10, -10, 5], [10, 10, 5], [-10, 10, 5]], dtype=np.float32)
    raycaster.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)))
    camera = {"intrinsics": [[1, 0, 1], [0, 1, 1], [0, 0, 1]],
              "cam_from_world": _pose([0, 0, 0]).tolist()}
    points, depth, valid = truth.raycast_truth_points([[.1, .3], [1.6, .8], [99, 99]], camera, raycaster, 1)
    np.testing.assert_array_equal(valid, [True, True, False])
    np.testing.assert_allclose(depth[:2], 5, atol=1e-6)
    np.testing.assert_allclose(points[:2], [[-4.5, -3.5, 5], [3, -1, 5]], atol=1e-6)


def test_mesh_sampling_and_identical_surface_have_zero_error():
    vertices, triangles = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]])
    points = truth.sample_surface(vertices, triangles, count=100)
    assert np.all(points[:, 0] >= 0) and np.all(points[:, 1] >= 0)
    assert np.all(points[:, :2].sum(axis=1) <= 1 + 1e-12)
    result = truth.surface_statistics(points, points, .01)
    assert result["coverage"] == 1
    assert result["chamfer_mean_euclidean_m"] == 0
    assert truth.mask_iou([[1, 1], [0, 0]], [[1, 0], [0, 1]]) == pytest.approx(1 / 3)


def test_suite_has_twenty_distinct_geometry_material_and_camera_configs():
    scenes = generator.scene_specs()
    assert len(scenes) == len({scene["id"] for scene in scenes}) == len({scene["seed"] for scene in scenes}) == 20
    assert len({scene["family"] for scene in scenes}) == 3
    assert len({scene["objects"] for scene in scenes}) > 1
    assert len({scene["focal_mm"] for scene in scenes}) > 1
    assert len({scene["texture_contrast"] for scene in scenes}) > 1
    assert min(scene["camera_arc_radians"] for scene in scenes) < .15


def test_missing_truth_is_preserved_in_the_fixed_twenty_scene_denominator(tmp_path):
    suite, runs, output = tmp_path / "suite", tmp_path / "runs", tmp_path / "result.json"
    suite.mkdir()
    (suite / "suite.json").write_text(json.dumps({"truth_source": "Blender Cycles CPU",
                                                  "scenes": generator.scene_specs()}), encoding="utf-8")
    code = truth.main(["--suite", str(suite), "--output", str(output), "--run-root", str(runs), "--evaluate-only"])
    result = json.loads(output.read_text(encoding="utf-8"))
    assert code == 1
    assert len(result["scenes"]) == result["summary"]["failed"] == result["fixed_denominator"] == 20
    assert result["summary"]["w3_sim3_aligned_pass"] == 0
