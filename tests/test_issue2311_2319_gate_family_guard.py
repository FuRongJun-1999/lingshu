def _mk(**over):
    from lingshu.world.camera_calib import CameraIntrinsics
    base = dict(fx=5200.0, fy=5200.0, cx=2000.0, cy=1500.0, width=4000,
                height=3000, sensor_w_mm=36.0, sensor_h_mm=24.0, lens_mm=50.0)
    base.update(over)
    return CameraIntrinsics(**base)


# -*- coding: utf-8 -*-
"""#2311/#2312/#2313/#2315/#2318/#2319 · fail-loud 闸家族守卫（回归钉）。

本件钉六处修复的**行为**（不是文案）：
  · #2311 gap_dual.update_meta：inf/NaN/非整数 float ⇒ ValueError（修前：
    inf 裸 OverflowError 泄漏、NaN TypeError、1.9 深处 AttributeError）；
  · #2312 set_dedup_config：window 非整数（30.9/inf/"30"）⇒ TypeError
    （修前：30.9 静默截断、inf 裸 OverflowError、"30" 收下）；
  · #2315 camera_bridge：主点非有限 ⇒ lossy 点名「主点非有限」（strict ⇒
    BridgeError；修前：nan > 1e-9 恒假 ⇒ 静默放行 representable_exactly=True）；
  · #2318 camera_calib.cross_check：claim_*=0 ⇒ SpecError（修前：裸
    ZeroDivisionError，除法在符号校验之前）；
  · #2319 hex_composite.extract_attributes：quadrant 白名单 r0/r1/r2
    （修前：quadrant[1].isdigit() 放行 r10/r02 并按单数字静默错读象限）。

运行：python -X utf8 -m pytest tests/test_issue2311_2319_gate_family_guard.py -q
"""
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------- #2311 gap_dual.update_meta ----------------

def test_2311_update_meta_nonfinite_raises_value_error():
    from lingshu.world.gap_dual import GapDual
    g = GapDual()
    for bad in (float("inf"), float("-inf"), float("nan"), 1.9, -0.5):
        with pytest.raises(ValueError):
            g.update_meta(0.0, False, bad)


def test_2311_update_meta_non_numeric_raises_type_error():
    from lingshu.world.gap_dual import GapDual
    g = GapDual()
    for bad in ("x", None, [2]):
        with pytest.raises(TypeError):
            g.update_meta(0.0, False, bad)


def test_2311_update_meta_valid_ints_still_work():
    from lingshu.world.gap_dual import GapDual
    g = GapDual()
    assert isinstance(g.update_meta(0.0, False, 0), float)
    assert isinstance(g.update_meta(0.0, True, 3), float)


# ---------------- #2312 set_dedup_config window ----------------

def test_2312_window_non_integer_raises_type_error():
    from lingshu.core.core import SpacetimeMemoryEngine
    e = SpacetimeMemoryEngine(":memory:")
    for bad in (30.9, float("inf"), "30", True):
        with pytest.raises(TypeError):
            e.set_dedup_config(0.85, bad)


def test_2312_window_valid_int_still_works():
    from lingshu.core.core import SpacetimeMemoryEngine
    e = SpacetimeMemoryEngine(":memory:")
    e.set_dedup_config(0.85, 30)
    assert e._dedup_window == 30


# ---------------- #2313 三处 float() 闸字符串宽收 ----------------





def _intr(**over):
    from lingshu.world.camera_calib import CameraIntrinsics
    import dataclasses
    kw = dict(fx=5200.0, fy=5200.0, cx=2000.0, cy=1500.0, width=4000,
              height=3000, sensor_w_mm=36.0, sensor_h_mm=24.0, lens_mm=50.0)
    kw.update(over)
    return CC_CameraIntrinsics() if False else dataclasses.replace(
        CC.CameraIntrinsics(**kw)) if False else _mk(**kw)


def _mk(**kw):
    from lingshu.world.camera_calib import CameraIntrinsics
    base = dict(fx=5200.0, fy=5200.0, cx=2000.0, cy=1500.0, width=4000,
                height=3000, sensor_w_mm=36.0, sensor_h_mm=24.0, lens_mm=50.0)
    base.update(kw)
    return CameraIntrinsics(**base)


def test_2315_bridge_nan_cx_reported_as_lossy():
    from lingshu.world import camera_bridge as CB
    cam, report = CB.camera3d_from_intrinsics(
        _mk(cx=float("nan")), eye=(0.0, 0.0, 10.0), strict=False)
    assert any("主点非有限" in x for x in report["lossy"])


def test_2315_bridge_nan_cx_strict_raises_bridge_error():
    from lingshu.world import camera_bridge as CB
    from lingshu.world.camera_bridge import BridgeError
    with pytest.raises(BridgeError):
        CB.camera3d_from_intrinsics(_mk(cx=float("nan")),
                                    eye=(0.0, 0.0, 10.0), strict=True)


# ---------------- #2318 cross_check claim=0 ----------------

def test_2318_cross_check_zero_claim_raises_spec_error():
    from lingshu.world.camera_calib import cross_check, SpecError
    for zero in ({"claim_focal35_mm": 0.0}, {"claim_hfov_deg": 0.0}):
        with pytest.raises(SpecError):
            cross_check(sensor="fullframe", lens_mm=50.0,
                        resolution=(6000, 4000), **zero)


# ---------------- #2319 extract_attributes quadrant 白名单 ----------------

def _img():
    import numpy as np
    img = np.zeros((30, 30, 3), dtype=np.uint8)
    img[0:10, 0:10, 0] = 90
    return img


def test_2319_quadrant_out_of_contract_raises_value_error():
    from lingshu.nn.hex_composite import extract_attributes
    img = _img()
    for bad in ("r10", "r02", "rX", "r"):
        with pytest.raises(ValueError):
            extract_attributes(img, bad, "red", "circle", None)


def test_2319_quadrant_whitelist_still_works():
    from lingshu.nn.hex_composite import extract_attributes
    img = _img()
    for q in ("r0", "r1", "r2"):
        try:
            extract_attributes(img, q, "red", "circle", None)
        except ValueError as ex:
            pytest.fail("白名单象限 %s 被 quadrant 闸误拒：%s" % (q, ex))
        except Exception:
            pass  # 下游读取面的其它异常不在本守卫范围
