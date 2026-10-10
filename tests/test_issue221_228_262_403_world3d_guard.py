# -*- coding: utf-8 -*-
"""守卫 · #221/#228/#262/#403 · lingshu/world/world3d.py（本组四件合一）
============================================================================
#221 [反投影] add_vprim 的 2D→3D 反投影不使用相机位姿（写死「相机正对 Z 轴」：
     X=(cx_px-sw/2)*Z/f、Y=(sh/2-cy_px)*Z/f，忽略 cx/cy/cz/yaw/pitch），且天空
     物体的地平线钳制 `max(Y, (sh-horizon_px)*Z/f*0.5)` 既与相机位姿无关、又带
     一个无出处的 0.5 系数。
     判据来源：docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md §3.2
     「2D↔3D 对齐（映射问题，非猜测）……语义对齐走投影-关联（已知相机参数→3D
     锚点投影回2D验证）」——反投影须是已知相机参数下的精确逆映射，可回代投影。
     地平线钳制的 0.5 系数：经验标定，追不到出处（全仓无理论/提交依据）；改为
     该参数的几何含义「同深度地平线像素的世界高度」。

#228 [渲染] 画家算法排序与球体透视半径用世界 z 冒充相机距离（Object3D.depth
     返回 center[2]）。相机有 yaw/pitch/平移时，世界 z 与相机前向深度不等，
     排序与半径均错。判据来源：同 §3.2 投影-关联（尺度/遮挡须按相机坐标 z）。

#262 [锚点验证] 验证记录不跟锚点身份走：SemanticAnchor.id 为随机 uuid（每次
     build_anchor_graph 重建即换新 id），且 verify 对不存在的 anchor_id 不校验、
     直接新建记录。判据来源：semantic_anchor_graph.py 模块 docstring「事物是其
     关系的总和」——身份须由内容（类别+位置+来源）决定，方可跨重建稳定。

#403 [渲染] 近平面裁剪全有或全无：_project_corners 任一角点投影失败（z<=0.1）
     即整物不画。判据来源：§3.2 透视投影——近平面应做多边形裁剪（部分可见），
     非整物丢弃。

变异自证（人工改码须分别触发）：
    · #221 反投影改回 `X=(cx_px-sw/2)*Z/f; Y=(sh/2-cy_px)*Z/f` ⇒ test_backproject_* 必红；
    · #221 地平线钳制改回 `(sh-horizon_px)*Z/f*0.5` ⇒ test_sky_object_horizon_* 必红；
    · #228 Object3D.depth 改回 `return self.center[2]` ⇒ test_painter_* / test_sphere_* 必红；
    · #262 SemanticAnchor.id 改回随机 uuid ⇒ test_anchor_id_is_deterministic_* 必红；
    · #262 SemanticAnchorGraph.verify 去掉 anchor_id 存在性校验 ⇒ test_verify_unknown_* 必红；
    · #403 _draw_box 改回 _project_corners 单点失败即整物丢弃 ⇒ test_box_straddling_* 必红。

运行：python -X utf8 -m pytest tests/test_issue221_228_262_403_world3d_guard.py -q --no-header
"""
from __future__ import annotations

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world.world3d import Camera3D, Object3D, World3D  # noqa: E402
from lingshu.world.vprim import VPrim  # noqa: E402

SW, SH = 800, 600


def _posed_cam():
    """非平凡位姿相机（平移 + yaw + pitch 全非零）——缺陷只在位姿非平凡时暴露。"""
    return Camera3D(fov_deg=60.0, cx=1.0, cy=1.2, cz=-3.0, yaw=0.6, pitch=-0.15)


# ===========================================================================
# #221 反投影必须使用相机位姿
# ===========================================================================

def test_unproject_is_exact_inverse_of_project():
    """project/unproject 互为精确逆（「投影-关联」可回代）——#221 的几何基座。"""
    cam = _posed_cam()
    for p in ((0.0, 1.2, 5.0), (2.0, 3.0, 12.0), (-4.0, 0.5, 7.5), (1.5, -1.0, 20.0)):
        px, py = cam.project(p, SW, SH)
        z = cam.depth_of(p)
        back = cam.unproject(px, py, z, SW, SH)
        for a, b in zip(back, p):
            assert abs(a - b) < 1e-6, f"unproject 非 project 之逆：{back} != {p}"


def test_backproject_uses_camera_pose_roundtrip_posed():
    """天空物体的 3D 反投影回代投影必须命中原 bbox 中心（含相机位姿）。

    缺陷形态（旧码写死「相机正对 Z 轴」）：此例实测回代投影 = (653.6, 66.9)，
    远离 bbox 中心 (220,140)。"""
    cam = _posed_cam()
    w = World3D(camera=cam)
    vp = VPrim("moon", (180, 100, 260, 180), 0.8)   # 中心 (220,140)
    obj = w.add_vprim(vp, SW, SH)
    sx, sy = cam.project(obj.center, SW, SH)
    assert abs(sx - 220.0) < 0.5 and abs(sy - 140.0) < 0.5, (
        "#221：反投影未用相机位姿——回代投影 (%r,%r) != bbox 中心 (220,140)" % (sx, sy))


def test_backproject_roundtrip_default_camera():
    """默认相机下亦须精确回代（旧码的天空钳制 0.5 系数会把 y 抬到 166.9）。"""
    cam = Camera3D()
    w = World3D(camera=cam)
    obj = w.add_vprim(VPrim("moon", (180, 100, 260, 180), 0.8), SW, SH)
    sx, sy = cam.project(obj.center, SW, SH)
    assert abs(sx - 220.0) < 0.5 and abs(sy - 140.0) < 0.5, (
        "#221：默认相机回代投影 (%r,%r) != (220,140)" % (sx, sy))


def test_sky_object_horizon_clamp_is_camera_consistent():
    """天空物体被地平线钳制时，其世界高度须正好回代投影到地平线像素。

    缺陷形态：旧式 `(sh-horizon_px)*Z/f*0.5` 与相机位姿无关，此例回代投影 y=166.9
    而非地平线像素 270。"""
    cam = Camera3D()                       # 水平相机：地平线世界高度 = cy
    w = World3D(camera=cam)
    # bbox 中心 (220,340) 在地平线像素 270 之下 ⇒ 触发钳制
    obj = w.add_vprim(VPrim("moon", (180, 300, 260, 380), 0.8), SW, SH)
    horizon_px = SH * 0.45
    sy = cam.project(obj.center, SW, SH)[1]
    assert abs(sy - horizon_px) < 0.5, (
        "#221：天空物体地平线钳制与相机不一致——回代投影 y=%r != 地平线 %r"
        % (sy, horizon_px))


def test_ground_object_still_bottom_on_ground():
    """非回归：地面物体仍贴地（世界 y 底 = 0），且水平方向回代命中 bbox 中心。"""
    cam = Camera3D()                       # 水平相机：贴地改 y 不影响 x
    w = World3D(camera=cam)
    vp = VPrim("person", (300, 250, 340, 520), 0.9)   # 中心 (320,385)
    obj = w.add_vprim(vp, SW, SH)
    assert abs((obj.center[1] - obj.size[1] / 2) - 0.0) < 1e-9, "地面物体未贴地"
    sx = cam.project(obj.center, SW, SH)[0]
    assert abs(sx - 320.0) < 0.5, "#221：地面物体水平反投影未用相机位姿：%r" % sx


# ===========================================================================
# #228 画家算法排序 / 球体透视半径须用相机深度（非世界 z）
# ===========================================================================

def _occlusion_scene():
    """两个世界 z 序与相机深度序相反的 box（yaw=75° 侧视）。

    实测：A 世界 z=4、相机深度 39.67；B 世界 z=6、相机深度 38.26。
    世界 z 序（旧口径）判 A 在前，相机深度序判 B 在前——两者相反。"""
    cam = Camera3D(fov_deg=60.0, cx=0.0, cy=1.2, cz=0.0, yaw=math.radians(75.0))
    a = Object3D("a", (-40.0, 1.0, 4.0), (4.0, 4.0, 4.0), (255, 0, 0), "box")
    b = Object3D("b", (-38.0, 1.0, 6.0), (4.0, 4.0, 4.0), (0, 0, 255), "box")
    return cam, a, b


def test_object_depth_uses_camera_coordinate_not_world_z():
    """Object3D.depth(cam) 必须等于相机坐标 z，且与世界 z 不同（此例）。"""
    cam, a, b = _occlusion_scene()
    assert abs(a.depth(cam) - cam.depth_of(a.center)) < 1e-9
    assert abs(b.depth(cam) - cam.depth_of(b.center)) < 1e-9
    assert abs(a.depth(cam) - a.center[2]) > 1.0, (
        "#228：depth() 仍等于世界 z（相机位姿被忽略）——depth=%r world_z=%r"
        % (a.depth(cam), a.center[2]))
    # 序相反：世界 z 判 A 远、相机深度判 B 远
    assert a.center[2] < b.center[2] and a.depth(cam) > b.depth(cam)


def test_painter_algorithm_orders_by_camera_depth():
    """渲染时近（相机深度小）者后画、遮远者——按相机深度而非世界 z。

    缺陷形态（旧码按世界 z）：世界 z=6 的 B 被先画、世界 z=4 的 A 后画，
    A 中心像素显示 A 的红；修后 A 中心像素显示更近的 B 的蓝。"""
    cam, a, b = _occlusion_scene()
    w = World3D(camera=cam)
    w.objects = [a, b]
    img = w.render(SW, SH, camera=cam)
    px = img.load()
    ax, ay = cam.project(a.center, SW, SH)
    r, g, bl = px[int(ax), int(ay)]
    assert bl > r, (
        "#228：画家算法用世界 z 冒充相机距离——A 中心像素 %r 是 A 的红而非更近的 B 的蓝"
        % ((r, g, bl),))


def test_sphere_radius_uses_camera_depth_not_world_z():
    """球体屏幕半径 = f*直径/2/相机深度（非世界 z）。

    缺陷形态（旧码用世界 z=4）：此球半径 = 86.6px（直径 173），几乎铺满屏；
    修后相机深度 39.67 ⇒ 半径 ≈ 8.7px（直径 ≈ 17）。"""
    cam, a, _ = _occlusion_scene()
    w = World3D(camera=cam)
    w.objects = [Object3D("ball", a.center, (2.0, 2.0, 2.0), (0, 255, 0), "sphere")]
    img = w.render(SW, SH, camera=cam)
    px = img.load()
    xs = [x for x in range(SW) for y in range(SH)
          if px[x, y][1] > 150 and px[x, y][0] < 80 and px[x, y][2] < 80]
    assert xs, "球体未渲染"
    width = max(xs) - min(xs) + 1
    f = cam.focal(SW)
    want = f * 2.0 / cam.depth_of(a.center)      # 直径（px），按相机深度
    wrong = f * 2.0 / a.center[2]                # 直径（px），按世界 z（缺陷口径）
    assert abs(width - want) < 6, (
        "#228：球体半径未按相机深度——实测直径 %r，相机深度口径 %r，世界 z 口径 %r"
        % (width, want, wrong))
    assert width < wrong / 3, "#228：球体半径疑似仍按世界 z（%r vs %r）" % (width, wrong)


def test_scene_text_orders_by_camera_depth():
    """scene_text 的远→近顺序亦按相机深度（旧码按世界 z 会给出相反顺序）。"""
    cam, a, b = _occlusion_scene()
    w = World3D(camera=cam)
    w.objects = [a, b]
    text = w.scene_text()
    assert text.index("a@3D") < text.index("b@3D"), (
        "#228：scene_text 未按相机深度排序——%r" % text)


# ===========================================================================
# #262 锚点身份确定性 + 验证记录跟身份走
# ===========================================================================

def test_anchor_id_is_deterministic_from_content():
    """同内容锚点 ⇒ 同 id；内容变 ⇒ id 变（身份由内容导出，非随机）。"""
    from lingshu.world.semantic_anchor_graph import SemanticAnchor
    a1 = SemanticAnchor("chair", (1.0, 0.5, 2.0), provenance="world3d")
    a2 = SemanticAnchor("chair", (1.0, 0.5, 2.0), provenance="world3d")
    assert a1.id == a2.id, "#262：同内容锚点 id 不同（随机 id）——%r != %r" % (a1.id, a2.id)
    a3 = SemanticAnchor("chair", (1.0, 0.5, 2.5), provenance="world3d")
    assert a3.id != a1.id, "#262：内容不同却同 id"


def test_build_anchor_graph_ids_stable_across_rebuild():
    """World3D 重建锚点图 ⇒ 同一物体的锚点 id 不变（旧码随机 uuid 每建必变）。"""
    w = World3D()
    w.objects = [Object3D("chair", (0.0, 0.5, 4.0), (0.5, 1.0, 0.5), (150, 120, 90), "box"),
                 Object3D("table", (2.0, 0.4, 5.0), (1.2, 0.8, 0.8), (140, 100, 60), "box")]
    ids1 = [a["id"] for a in w.build_anchor_graph()["anchors"]]
    ids2 = [a["id"] for a in w.build_anchor_graph()["anchors"]]
    assert ids1 == ids2 and ids1, "#262：重建后锚点 id 变化——%r vs %r" % (ids1, ids2)


def test_verification_records_survive_rebuild():
    """验证记录跟锚点身份走：重建锚点图后，该锚点的通道证据仍在。"""
    w = World3D()
    w.objects = [Object3D("chair", (0.0, 0.5, 4.0), (0.5, 1.0, 0.5), (150, 120, 90), "box")]
    aid = w.build_anchor_graph()["anchors"][0]["id"]
    w.verify_anchor(aid, "tactile", 1.0)
    w.verify_anchor(aid, "audio", 1.0)
    before = w.verify_anchor(aid, "tactile", 1.0)
    w.build_anchor_graph()                       # 重建（同物体）
    after = w.verify_anchor(aid, "tactile", 1.0)
    assert after.get("channel_evidence"), (
        "#262：重建后验证记录失联——%r" % after)
    assert after["channel_evidence"].get("audio") == 1.0, (
        "#262：重建后验证记录丢失——%r" % after["channel_evidence"])
    assert after["verified_rounds"] >= before["verified_rounds"], (
        "#262：重建后稳定轮数倒退——%r < %r" % (after["verified_rounds"], before["verified_rounds"]))


def test_verify_unknown_anchor_does_not_create_record():
    """图里不存在的 anchor_id ⇒ unknown/error，且不得凭空建验证记录（phantom 锚点）。"""
    from lingshu.world.semantic_anchor_graph import SemanticAnchorGraph
    g = SemanticAnchorGraph()
    g.add("chair", (0.0, 0.5, 0.0))
    r = g.verify("anchor_deadbeef", "tactile", 1.0)
    assert r["confirmation"] == "unknown", "#262：phantom 锚点被接受——%r" % r
    verifier = getattr(g, "_verifier", None)
    assert verifier is None or "anchor_deadbeef" not in verifier._anchors, (
        "#262：为 phantom 锚点建了验证记录")
    rc = g.verify_conflict("anchor_deadbeef", "visual", "a", "b")
    assert rc["confirmation"] == "unknown" and rc["conflict_detected"] is False, rc


# ===========================================================================
# #403 近平面裁剪（部分可见，非全有或全无）
# ===========================================================================

def test_clip_near_keeps_partial_polygon():
    """多边形一半在近平面后 ⇒ 裁剪后保留可见部分（非返回空）。"""
    quad = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 0.0, -1.0), (0.0, 0.0, -1.0)]
    clipped = World3D._clip_near(quad, 0.1)
    assert len(clipped) >= 3, "#403：部分可见多边形被整块丢弃——%r" % (clipped,)
    assert all(p[2] >= 0.1 - 1e-9 for p in clipped), "#403：裁剪后仍有近平面后的点"


def test_clip_near_drops_fully_behind_polygon():
    """完全在近平面后 ⇒ 裁剪为空（该部分确不可见）。"""
    quad = [(0.0, 0.0, -1.0), (1.0, 0.0, -1.0), (1.0, 0.0, -2.0), (0.0, 0.0, -2.0)]
    assert World3D._clip_near(quad, 0.1) == []


def test_project_face_straddling_near_plane_nonempty():
    """跨近平面的面 ⇒ 投影多边形非空；完全在后的面 ⇒ 空。"""
    cam = Camera3D(fov_deg=60.0, cx=0.0, cy=1.2, cz=0.0)
    w = World3D(camera=cam)
    # 面跨近平面：近侧两点 z=0.05（< 0.1），远侧两点 z=1.0
    straddle = [(0.3, 0.7, 0.05), (1.3, 0.7, 0.05), (1.3, 1.7, 1.0), (0.3, 1.7, 1.0)]
    poly = w._project_face(straddle, cam, SW, SH)
    assert len(poly) >= 3, "#403：跨近平面面被整块丢弃——%r" % (poly,)
    behind = [(0.3, 0.7, -1.0), (1.3, 0.7, -1.0), (1.3, 1.7, -1.0), (0.3, 1.7, -1.0)]
    assert w._project_face(behind, cam, SW, SH) == []


def test_box_straddling_near_plane_still_renders():
    """box 前角点落在近平面后 ⇒ 仍渲染可见部分（旧码整物不画）。"""
    cam = Camera3D(fov_deg=60.0, cx=0.0, cy=1.2, cz=0.0)
    w = World3D(camera=cam)
    # 前脸 4 角点相机 z=0.0（< 0.1 近平面后）；后脸 z=1.0 可见
    w.objects = [Object3D("near", (0.8, 1.2, 0.5), (1.0, 1.0, 1.0), (255, 0, 0), "box")]
    img = w.render(SW, SH, camera=cam)
    px = img.load()
    red = sum(1 for x in range(SW) for y in range(SH)
              if px[x, y][0] > 150 and px[x, y][1] < 80)
    assert red > 0, "#403：跨近平面的 box 整物未画（全有或全无）——实测红像素 %d" % red


# ===========================================================================
# 自身可收集 + 直跑入口
# ===========================================================================

def test_guard_is_collectable():
    """本件必须能被 pytest 收集（防自身落入 test_gate_coverage 的「同族盲区」）。"""
    assert os.path.basename(__file__).startswith("test_")
    assert Camera3D is not None and World3D is not None
    assert Object3D is not None


def _main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print("FAIL %-58s %s" % (t.__name__, e))
        else:
            passed += 1
            print("ok   %s" % t.__name__)
    print("\n%d/%d passed" % (passed, len(tests)))
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
