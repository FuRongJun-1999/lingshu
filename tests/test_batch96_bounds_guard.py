# -*- coding: utf-8 -*-
"""批九十六守卫 B 组：值域/有限性闸族（#2427 / #2432）回归钉。

覆盖
--------------------------------------------------------------------------
#2427 `SpacetimeConsistency.__init__` 与 `_check_invariants` 的剩余参数面
    G1 `wander_bound_factor=inf` ⇒ ValueError（改前 `_reach_bound=inf` ⇒
       任意偏差判命中，23 m/tick 瞬移被判 self_consistent，漂移检测恒不触发）
    G2 `wander_bound_factor=nan` / `wander_bound_pad=nan/inf` ⇒ ValueError
       （改前 nan 被 `max(0.5, nan)` 吃成 0.5 ⇒ 正常 wander 全判失败、假漂移）
    G3 `ground_level=nan/-inf` ⇒ ValueError（改前 `y < gnd+0.4` 恒假 ⇒
       below_ground 不变量整条变死代码，地下 100 m 的世界判 self_consistent）
    G4 纵深：直接改 `scene.world.ground_level` 后 `_check_invariants` 报
       `bad_ground_level`（不只依赖构造期闸）
    G5 正对照：#2419 的三 rate 闸原样有效；默认参数与合法 0.0 边界不变

#2432 边界参数（`coord_contract` + `autoframe`）
    G6 `apply_move(size=nan/-24)` ⇒ ValueError（改前任意位置被静默压成
       `(0.5, 1.0, 0.5)`）
    G7 `apply_move(margin=nan/inf)` ⇒ ValueError（改前直接产出 NaN/inf 坐标）
    G8 `clamp_xz` 同闸（它是公开 helper）
    G9 `frame_points(screen_w=0)` ⇒ ValueError（改前裸 ZeroDivisionError）
    G10 `frame_points(screen_w=nan)` / `azimuth_deg=nan` / `elevation_deg=nan`
        ⇒ ValueError（改前相机 cx/cy/cz/yaw/pitch 全 NaN）
    G11 正对照：`SIZES=(4.0, 10.0, 1.0, 0.5)` 逐位不变（`test_coord_contract_
        single_source_guard` 的既有契约）；`frame_points(640,480)` 相机全有限

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch96_bounds_guard.py -q
"""
from __future__ import annotations

import math

import pytest

from lingshu.world import autoframe as AF
from lingshu.world import coord_contract as CC
from lingshu.world.spacetime_consistency import SpacetimeConsistency


PTS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]


# ================================================ #2427 SpacetimeConsistency
def test_G1_wander_bound_factor_inf_rejected():
    """G1：`factor=inf` 必须 raise（改前让漂移检测永久失效）。"""
    with pytest.raises(ValueError) as ei:
        SpacetimeConsistency(wander_bound_factor=float("inf"))
    assert "wander_bound_factor" in str(ei.value)


@pytest.mark.parametrize("kw,frag", [
    ({"wander_bound_factor": float("nan")}, "wander_bound_factor"),
    ({"wander_bound_factor": -1.0}, "wander_bound_factor"),
    ({"wander_bound_pad": float("nan")}, "wander_bound_pad"),
    ({"wander_bound_pad": float("inf")}, "wander_bound_pad"),
    ({"wander_bound_factor": True}, "wander_bound_factor"),
])
def test_G2_wander_bound_nan_neg_bool_rejected(kw, frag):
    """G2：nan/inf/负值/bool 全拒（改前 nan 让正常 wander 全判失败）。"""
    with pytest.raises(ValueError) as ei:
        SpacetimeConsistency(**kw)
    assert frag in str(ei.value)


@pytest.mark.parametrize("gnd", [float("nan"), float("-inf"), float("inf")])
def test_G3_ground_level_nonfinite_rejected(gnd):
    """G3：`ground_level` 非有限必须 raise（改前 below_ground 变死代码）。"""
    with pytest.raises(ValueError) as ei:
        SpacetimeConsistency(ground_level=gnd)
    assert "ground_level" in str(ei.value)


def test_G4_check_invariants_rejects_tampered_ground():
    """G4 纵深：绕过构造闸直接改 `scene.world.ground_level` 也要被不变量判据抓到。"""
    stc = SpacetimeConsistency(size=24, ground_level=1)
    stc.add_entity("e1", pos=(10.0, 5.0, 10.0))
    ok, issues = stc._check_invariants()
    assert ok and not any("bad_ground_level" in s for s in issues), issues
    stc.scene.world.ground_level = float("nan")
    ok2, issues2 = stc._check_invariants()
    assert not ok2 and any("bad_ground_level" in s for s in issues2), issues2


def test_G5_rate_gates_and_legal_bounds_unchanged():
    """G5 正对照：#2419 三 rate 闸仍在；默认值与合法 0.0 边界不变。"""
    for kw in ({"hit_threshold": float("nan")},
               {"drift_rate": 1.5},
               {"consistent_rate": -0.1}):
        with pytest.raises(ValueError):
            SpacetimeConsistency(**kw)
    d = SpacetimeConsistency()
    assert (d.wander_bound_factor, d.wander_bound_pad) == (1.5, 0.2)
    z = SpacetimeConsistency(wander_bound_factor=0.0, wander_bound_pad=0.0,
                             ground_level=-5)
    assert (z.wander_bound_factor, z.wander_bound_pad) == (0.0, 0.0)


# ==================================================== #2432 边界参数闸
@pytest.mark.parametrize("size", [float("nan"), float("inf"), -24, 0])
def test_G6_apply_move_bad_size_rejected(size):
    """G6：`size` 非有限/非正必须 raise（改前任意位置压成 (0.5, 1.0, 0.5)）。"""
    with pytest.raises(ValueError) as ei:
        CC.apply_move((10, 1, 10), 0.5, (1, 0, 0), size)
    assert "size" in str(ei.value)


@pytest.mark.parametrize("margin", [float("nan"), float("inf"), -0.5])
def test_G7_apply_move_bad_margin_rejected(margin):
    """G7：`margin` 非有限/负必须 raise（改前直接产出 NaN/inf 坐标）。"""
    with pytest.raises(ValueError) as ei:
        CC.apply_move((10, 1, 10), 0.5, (1, 0, 0), 24, margin)
    assert "margin" in str(ei.value)


def test_G8_clamp_xz_same_gate():
    """G8：`clamp_xz` 是公开 helper，同闸（改前 `size=nan` 静默返回 0.5）。"""
    with pytest.raises(ValueError):
        CC.clamp_xz(10.0, 10.0, float("nan"))
    with pytest.raises(ValueError):
        CC.clamp_xz(10.0, 10.0, 24, float("nan"))
    assert CC.clamp_xz(20.0, 20.0, 24, 0.5) == (20.0, 20.0)


@pytest.mark.parametrize("args", [(PTS, 0, 480), (PTS, 640, 0),
                                  (PTS, -100, 480)])
def test_G9_frame_points_zero_or_negative_screen_rejected(args):
    """G9：`screen_w/h <= 0` 必须 raise（改前裸 ZeroDivisionError）。"""
    with pytest.raises(ValueError) as ei:
        AF.frame_points(*args)
    assert "screen_w" in str(ei.value) or "screen_h" in str(ei.value)


@pytest.mark.parametrize("kw,frag", [
    ({"screen_w": float("nan")}, "screen_w"),
    ({"screen_h": float("inf")}, "screen_h"),
    ({"azimuth_deg": float("nan")}, "azimuth_deg"),
    ({"elevation_deg": float("nan")}, "elevation_deg"),
])
def test_G10_frame_points_nonfinite_rejected(kw, frag):
    """G10：非有限尺寸/角度必须 raise（改前整台相机 NaN 且 auto_view 当最优）。"""
    base = dict(screen_w=640, screen_h=480)
    base.update(kw)
    with pytest.raises(ValueError) as ei:
        AF.frame_points(PTS, base.pop("screen_w"), base.pop("screen_h"), **base)
    assert frag in str(ei.value)


def test_G11_legal_values_unchanged():
    """G11 正对照：既有守卫要求的 SIZES 逐位不变；frame_points 相机全有限。"""
    # 这四个值是 `test_coord_contract_single_source_guard` 的 SIZES，逐位钉死
    # （`clamp_axis` 会先把坐标钳到 margin=0.5，故小 size 全落 (0.5, _, 0.5)）。
    expected = {4.0: (3.5, 1.0, 3.5), 10.0: (9.5, 1.0, 9.5),
                1.0: (0.5, 1.0, 0.5), 0.5: (0.5, 1.0, 0.5)}
    for sz, want in expected.items():
        assert CC.apply_move((10, 1, 10), 0.5, (1, 0, 0), sz) == want, sz
    cam = AF.frame_points(PTS, 640, 480)
    for v in (cam.cx, cam.cy, cam.cz, cam.yaw, cam.pitch):
        assert math.isfinite(v)
    # #731 / #2154 既有闸未被破坏
    with pytest.raises(ValueError):
        AF.frame_points(PTS, 640, 480, fov_deg=-30.0)
    with pytest.raises(ValueError):
        AF.frame_points(PTS, 640, 480, margin_frac=0.6)
