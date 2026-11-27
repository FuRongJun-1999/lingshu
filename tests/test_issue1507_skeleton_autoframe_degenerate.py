"""Issue #1507 LS-04: automatic framing keeps degenerate skeletons visible."""
from __future__ import annotations

import math

import pytest

from lingshu.world.skeleton3d import Skeleton3D, fit_camera


def _assert_points_project(points, camera, width=400, height=400):
    projected = [camera.project(point, width, height) for point in points]
    assert all(point is not None for point in projected)
    assert all(camera.depth_of(point) > 0.1 for point in points)
    return projected


def test_single_joint_is_centered_in_front_of_near_plane_and_rendered():
    pytest.importorskip("PIL")
    skeleton = Skeleton3D(center=(0.0, 0.0, 5.0)).add_joint("root", (0.0, 0.0, 0.0))
    point = skeleton.world_pos("root")

    camera = skeleton.frame_camera(400, 400)
    projected = _assert_points_project([point], camera)[0]
    assert projected == pytest.approx((200.0, 200.0), abs=0.5)
    image = skeleton.render(400, 400, camera=camera)
    assert image.getpixel((200, 200)) == (255, 180, 120)


def test_coincident_and_near_coincident_points_get_a_stable_camera():
    points = [(2.0, -3.0, 1.5), (2.0, -3.0, 1.5), (2.0, -3.0, 1.50000001)]
    camera = fit_camera(points, 400, 400)
    projected = _assert_points_project(points, camera)
    assert all(point == pytest.approx(projected[0], abs=1e-5) for point in projected)


def test_points_along_camera_view_direction_remain_projectable():
    azimuth = math.radians(180.0)
    elevation = math.radians(8.0)
    direction = (
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
        math.cos(elevation) * math.cos(azimuth),
    )
    target = (1.0, 2.0, 3.0)
    points = [tuple(target[k] + scale * direction[k] for k in range(3))
              for scale in (-0.2, 0.0, 0.3)]

    camera = fit_camera(points, 400, 400)
    projected = _assert_points_project(points, camera)
    assert all(point == pytest.approx(projected[0], abs=1e-5) for point in projected)


def test_non_degenerate_skeleton_stays_inside_framing_margin():
    skeleton = Skeleton3D(center=(0.0, 0.0, 5.0))
    skeleton.add_joint("root", (0.0, 0.0, 0.0))
    skeleton.add_joint("left", (-0.4, 0.5, 0.0), parent="root")
    skeleton.add_joint("right", (0.8, 0.3, 0.1), parent="root")
    points = [skeleton.world_pos(name) for name in skeleton.joints]

    camera = skeleton.frame_camera(640, 360, margin_frac=0.08)
    projected = _assert_points_project(points, camera, width=640, height=360)
    assert all(49 <= x <= 591 and 26 <= y <= 334 for x, y in projected)
