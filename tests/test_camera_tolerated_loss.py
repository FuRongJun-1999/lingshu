"""Calibration tolerance permits an approximation; it does not make fy exact."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingshu.world.camera_bridge import (
    BridgeError, camera3d_from_intrinsics, intrinsics_from_camera3d,
)
from lingshu.world.camera_calib import CameraIntrinsics


def intrinsics(fy):
    return CameraIntrinsics(fx=1000.0, fy=fy, cx=1000.0, cy=600.0,
                            width=2000, height=1200, sensor_w_mm=0.0,
                            sensor_h_mm=0.0, lens_mm=0.0)


@pytest.mark.parametrize('fy', [995.0, 1005.0, 1010.0],
                         ids=['smaller', 'larger', 'boundary'])
def test_tolerated_anisotropy_is_reported(fy):
    k = intrinsics(fy)
    saved = copy.deepcopy(k.to_dict())
    cam, report = camera3d_from_intrinsics(k, eye=(0, 0, 0))
    assert report['representable_exactly'] is False, 'tolerated fy loss must be explicit'
    assert any('各向异性' in s for s in report['lossy'])
    assert cam.focal(k.width) == pytest.approx(k.fx)
    assert report['fx'] == k.fx and report['fy'] == k.fy
    assert k.to_dict() == saved


def test_projection_error_cannot_be_reported_as_exact():
    k = intrinsics(1005.0)
    cam, report = camera3d_from_intrinsics(k.to_dict(), eye=(0, 0, 0))
    point = (0.0, 1.0, 10.0)
    projected = cam.project(point, k.width, k.height)
    calibrated_y = k.cy - k.fy * point[1] / point[2]
    assert projected[1] - calibrated_y == pytest.approx(0.5)
    back = intrinsics_from_camera3d(cam, width=k.width, height=k.height)
    assert back.fy == pytest.approx(1000.0)
    assert report['representable_exactly'] is False, 'pixel residual forbids an exact report'


def test_equal_focals_remain_exact():
    cam, report = camera3d_from_intrinsics(intrinsics(1000.0), eye=(0, 0, 0))
    assert report['representable_exactly'] is True and report['lossy'] == []
    assert cam.project((0, 1, 10), 2000, 1200) == pytest.approx((1000, 500))


def test_strict_tolerance_still_rejects_large_difference():
    with pytest.raises(BridgeError, match='各向异性'):
        camera3d_from_intrinsics(intrinsics(1020.0), eye=(0, 0, 0))


def test_non_strict_still_reports_large_difference():
    _, report = camera3d_from_intrinsics(intrinsics(1020.0), eye=(0, 0, 0), strict=False)
    assert report['representable_exactly'] is False and report['lossy']
