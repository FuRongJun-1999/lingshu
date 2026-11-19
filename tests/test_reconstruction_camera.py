"""Native reconstruction cameras connect to the existing multiview consumer."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import math

import pytest

from lingshu.world.multiview import ViewObs, _triangulate
from lingshu.world.reconstruction_camera import ReconstructionCamera
from lingshu.world.world3d import (
    Camera3D, format_camera_observation, parse_camera_observation,
)


def _pose(*, center=(1.0, -0.4, 0.7), angle=0.23):
    c, s = math.cos(angle), math.sin(angle)
    rotation = ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))
    extrinsic = [list(row) + [-sum(row[i] * center[i] for i in range(3))]
                 for row in rotation]
    return {"width": 640, "height": 480, "source": "capture/photo.png",
            "source_index": 3, "model": "PINHOLE",
            "intrinsics": [[710.0, 17.0, 301.5], [0.0, 570.0, 253.25],
                           [0.0, 0.0, 1.0]], "cam_from_world": extrinsic}


def test_native_projection_matches_matrix_and_unprojects_with_arbitrary_intrinsics():
    pose = _pose()
    camera = ReconstructionCamera.from_pose("000003.jpg", pose)
    point = (2.0, 0.8, 8.0)
    c = [sum(row[i] * point[i] for i in range(3)) + row[3]
         for row in pose["cam_from_world"]]
    expected = (710 * c[0] / c[2] + 17 * c[1] / c[2] + 301.5,
                570 * c[1] / c[2] + 253.25)
    pixel = camera.project(point)
    assert pixel == pytest.approx(expected, abs=1e-12)
    assert camera.camera_depth(point) == pytest.approx(c[2], abs=1e-12)
    assert camera.unproject(*pixel, c[2]) == pytest.approx(point, abs=1e-12)
    assert camera.name == "000003.jpg" and camera.source_index == 3
    assert camera.project(camera.unproject(11, 12, 2)) == pytest.approx((11, 12))


def test_scale_changes_world_geometry_and_depth_together_without_changing_pixels():
    plain = ReconstructionCamera.from_pose("view", _pose())
    scaled = ReconstructionCamera.from_pose("view", _pose(), coordinate_scale=0.125)
    point = (2.0, 0.8, 8.0)
    metric = tuple(x * 0.125 for x in point)
    pixel = plain.project(point)
    assert scaled.project(metric) == pytest.approx(pixel, abs=1e-12)
    assert scaled.camera_depth(metric) == pytest.approx(plain.camera_depth(point) * 0.125)
    assert scaled.unproject(*pixel, scaled.camera_depth(metric)) == pytest.approx(metric)
    virtual = scaled.as_camera3d()
    assert (virtual.cx, virtual.cy, virtual.cz) == pytest.approx((0.125, -0.05, 0.0875))


def test_virtual_pixel_conversion_matches_existing_camera3d_and_is_reversible():
    camera = ReconstructionCamera.from_pose("view", _pose())
    point = (2.0, 0.8, 8.0)
    native = camera.project(point)
    virtual = camera.to_virtual_pixel(*native)
    assert camera.as_camera3d().project(point, 640, 480) == pytest.approx(virtual, abs=1e-12)
    assert camera.to_native_pixel(*virtual) == pytest.approx(native, abs=1e-12)
    assert virtual != native
    # Vertical handedness is explicit: native image Y grows downward.
    upper = camera.to_virtual_pixel(native[0], native[1] + 10)
    assert upper[1] < virtual[1]


def test_actual_multiview_triangulation_recovers_the_native_world_point():
    cameras = [ReconstructionCamera.from_pose(str(i), _pose(center=center, angle=angle))
               for i, (center, angle) in enumerate([((-1.0, 0.3, 0.0), 0.11),
                                                   ((1.5, -0.4, 0.5), -0.18),
                                                   ((0.2, 0.7, -0.1), 0.02)])]
    point = (0.5, 1.1, 6.0)
    views = [ViewObs(camera=c.as_camera3d(), bbox_center=c.to_virtual_pixel(*c.project(point)),
                     screen_w=c.width, screen_h=c.height) for c in cameras]
    assert _triangulate(views) == pytest.approx(point, abs=1e-11)


def test_existing_observation_metadata_roundtrip_preserves_the_virtual_camera_ray():
    camera = ReconstructionCamera.from_pose("view", _pose())
    point = (2.0, 0.8, 8.0)
    u, v = camera.to_virtual_pixel(*camera.project(point))
    packed = format_camera_observation("feature", (u-1, v-1, u+1, v+1),
                                        camera.as_camera3d(), 640, 480,
                                        eid="track:9", tick=0, source="reconstruction")
    parsed = parse_camera_observation(packed["content"])
    restored = Camera3D.from_meta(parsed["camera"], strict=True)
    assert restored.project(point, 640, 480) == pytest.approx((u, v), abs=1e-12)
    assert restored.unproject(u, v, camera.camera_depth(point), 640, 480) == pytest.approx(point)


@pytest.mark.parametrize("key,value", [
    ("width", 0), ("height", True), ("width", 640.0),
    ("source_index", -1), ("source_index", True), ("source", ""),
    ("model", "SIMPLE_RADIAL"), ("model", "OPENCV"),
    ("intrinsics", [[1, 0, 0], [0, 1, 0]]),
    ("intrinsics", [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ("intrinsics", [[1, 0, 0], [0, 0, 0], [0, 0, 1]]),
    ("intrinsics", [[1, 0, 0], [1, 1, 0], [0, 0, 1]]),
    ("intrinsics", [[1, 0, 0], [0, 1, 0], [0, 0, 2]]),
    ("intrinsics", [[1, float("nan"), 0], [0, 1, 0], [0, 0, 1]]),
    ("cam_from_world", [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0]]),
    ("cam_from_world", [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
    ("cam_from_world", [[1, 0, 0, float("inf")], [0, 1, 0, 0], [0, 0, 1, 0]]),
    ("cam_from_world", [[True, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
])
def test_bad_native_camera_data_is_rejected(key, value):
    pose = _pose()
    pose[key] = value
    with pytest.raises(ValueError):
        ReconstructionCamera.from_pose("view", pose)


@pytest.mark.parametrize("scale", [0, -1, True, float("nan"), float("inf"), "1"])
def test_bad_coordinate_scales_are_rejected(scale):
    with pytest.raises(ValueError):
        ReconstructionCamera.from_pose("view", _pose(), coordinate_scale=scale)


def test_missing_pose_fields_and_missing_name_are_not_replaced_by_defaults():
    for key in _pose():
        pose = _pose()
        del pose[key]
        with pytest.raises(ValueError, match="missing"):
            ReconstructionCamera.from_pose("view", pose)
    with pytest.raises(ValueError, match="pose"):
        ReconstructionCamera.from_pose("view", None)
    with pytest.raises(ValueError, match="name"):
        ReconstructionCamera.from_pose("", _pose())


@pytest.mark.parametrize("point", [(0, 1), (0, 1, float("nan")), (True, 1, 2), "abc"])
def test_bad_world_points_are_rejected(point):
    camera = ReconstructionCamera.from_pose("view", _pose())
    with pytest.raises(ValueError):
        camera.project(point)
    with pytest.raises(ValueError):
        camera.camera_depth(point)


@pytest.mark.parametrize("depth", [0, -1, True, float("inf"), float("nan")])
def test_unprojection_requires_positive_finite_camera_z(depth):
    camera = ReconstructionCamera.from_pose("view", _pose())
    with pytest.raises(ValueError):
        camera.unproject(10, 20, depth)


def test_nonpositive_depth_has_no_projection_and_source_mutation_cannot_change_camera():
    pose = _pose(center=(0, 0, 0), angle=0)
    camera = ReconstructionCamera.from_pose("view", pose)
    assert camera.project((0, 0, -1)) is None
    assert camera.project((0, 0, 0)) is None
    original = deepcopy(camera.intrinsics)
    pose["intrinsics"][0][0] = 1
    assert camera.intrinsics == original
    with pytest.raises(FrozenInstanceError):
        camera.width = 12


@pytest.mark.parametrize("method,args", [
    ("unproject", (float("nan"), 0, 1)),
    ("to_virtual_pixel", (0, float("inf"))),
    ("to_native_pixel", (True, 0)),
])
def test_pixel_conversion_rejects_nonfinite_or_boolean_values(method, args):
    camera = ReconstructionCamera.from_pose("view", _pose())
    with pytest.raises(ValueError):
        getattr(camera, method)(*args)
