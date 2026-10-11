# -*- coding: utf-8 -*-
"""守卫 · #135/#220/#231 · lingshu/world 三件（本组三件合一）
============================================================================
#135 [world/multiview] 多视角融合按 category 单桶：同类两个物体（不同位置）
     的观测被塞进同一个桶 `rec["views"].append(obs)`，一起送去三角化；且
     `merge_dist`（空间近邻阈值）自 __init__ 赋值后全文再无引用——「同类别近距
     才合并」的意图从未生效。缺陷形态：同类两物被融成一个居中错位的 3D 点。
     判据来源：world3d.py:348「时间序列收敛：同类 + 距离近 → 更新位置」的合并
     语义（近距才合并）；docs/theory/世界模型与语义时空图_完整理论整理与实现
     路线.md §3.2「三视图/双目视差——已知相机参数下的映射，非猜测」：多视角
     交汇须按空间一致性归属，而非仅按类别标签。

#220 [world/anchored_verification] `cognitive_baseline` 声明「跨时间稳定 +
     预测长期命中率」共同决定认知基底层是否建立，但 `stable_rounds < 3` 分支
     只往 notes 追加一句提示、**不翻 `baseline_ok`**——于是稳定轮数不足时
     仍返回 `baseline_ok=True`，与函数契约（"世界长期不骗人"的基底判据）矛盾。
     判据来源：anchored_verification.py:102-106 docstring 与 anchor_verify.py:38
     `STABLE_ROUNDS = 3`（跨时间稳定轮数门槛）——稳定不足即基底未建立。

#231 [world/shapes] 部件相对坐标的零点不是物体中心：模块头 :29 声明
     `pos(相对物体中心)`，但 world3d._draw_part 用 `part_center = 物体中心 +
     pos` 直接叠加——于是部件必须自报「相对物体**底部/地面**」的 y 才能落地，
     与 :29 的「相对物体中心」口径冲突（person 躯干 y=0.4、头 y=0.95，整体
     质心并不在 0；table 腿 y=-0.2 直插地面下）。缺陷形态：部件坐标口径与
     装配口径不一致，渲染出的形状相对物体中心整体偏移。
     判据来源：shapes.py:29 注释口径 + world3d.py:404 `pos(相对物体中心)`。

运行：python -X utf8 -m pytest tests/test_issue135_220_231_world_misc_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.world.world3d import Camera3D, World3D  # noqa: E402
from lingshu.world.multiview import MultiViewFusion, ViewObs  # noqa: E402
from lingshu.world.anchored_verification import AnchoredVerification  # noqa: E402

SW, SH = 800, 600


# ===========================================================================
# #135 多视角融合：同类多物各自成桶 + merge_dist 生效
# ===========================================================================

def _views_of(cams, points):
    """给定相机与真值点，返回 [(cam, px, py, point)] 的观测序列。"""
    out = []
    for cam in cams:
        for p in points:
            px, py = cam.project(p, SW, SH)
            out.append((cam, px, py, p))
    return out


def test_same_category_two_objects_stay_separate():
    """#135：同类两物（相距 > merge_dist）须各自三角化到真值，不得并成一桶。

    缺陷形态：单桶实现把两物共 4 条射线一起最小二乘 → 只产出 1 个错位中心。
    """
    cams = [Camera3D(), Camera3D(cx=3.0, cz=-2.0, yaw=0.5)]
    truth = [(0.0, 1.2, 5.0), (2.5, 1.2, 5.0)]      # 同类两物，相距 2.5m
    mv = MultiViewFusion(merge_dist=1.0)
    for cam, px, py, _ in _views_of(cams, truth):
        mv.add_observation("person", ViewObs(cam, (px, py), SW, SH))

    fused = mv.fused_objects()
    assert len(fused) == 2, (
        "#135：同类两物被并成一桶——fused=%r" % [f["center"] for f in fused])
    centers = sorted(f["center"] for f in fused)
    for got, want in zip(centers, sorted(truth)):
        assert max(abs(a - b) for a, b in zip(got, want)) < 0.2, (
            "#135：桶中心 %r 偏离真值 %r" % (got, want))
    # 两桶都拿到 2 视角（每个物体各被两相机看到一次）
    assert all(f["views"] == 2 for f in fused), (
        "#135：桶视角数异常——%r" % [f["views"] for f in fused])


def test_same_object_from_two_cameras_still_merges():
    """#135 非回归：同一物体的两视角仍须合成一桶（不得因修复而过度切分）。"""
    cams = [Camera3D(), Camera3D(cx=3.0, cz=-2.0, yaw=0.5)]
    truth = [(1.0, 1.2, 6.0)]
    mv = MultiViewFusion(merge_dist=1.0)
    last = None
    for cam, px, py, _ in _views_of(cams, truth):
        last = mv.add_observation("person", ViewObs(cam, (px, py), SW, SH))
    fused = mv.fused_objects()
    assert len(fused) == 1, "#135：同一物体被切成多桶——%r" % fused
    assert fused[0]["views"] == 2
    assert last["triangulated"] is True
    assert max(abs(a - b) for a, b in zip(fused[0]["center"], truth[0])) < 0.2


def test_merge_dist_gates_new_bucket_on_established_center():
    """#135：merge_dist 必须真正参与归属判定（旧码赋值后全文无引用）。

    已建立 A 的中心后：射线穿过 A 附近 → 归入 A 桶（不另起桶）；明显偏离 A
    （> merge_dist）的观测 → 另起新桶，该桶经第二视角三角化后独立成物。
    """
    cams = [Camera3D(), Camera3D(cx=3.0, cz=-2.0, yaw=0.5)]
    a = (0.0, 1.2, 5.0)
    mv = MultiViewFusion(merge_dist=0.5)
    for cam, px, py, _ in _views_of(cams, [a]):
        mv.add_observation("person", ViewObs(cam, (px, py), SW, SH))
    assert len(mv.fused_objects()) == 1

    cam3 = Camera3D(cx=-2.0, cz=1.0, yaw=-0.3)
    # 近：射线几乎穿过 A → 仍归 A 桶（不得另起桶）
    px, py = cam3.project(a, SW, SH)
    near = mv.add_observation("person", ViewObs(cam3, (px, py), SW, SH))
    assert len(mv.fused_objects()) == 1, (
        "#135：近距观测被误判为新物体——%r" % mv.fused_objects())
    assert near["triangulated"] is True
    assert max(abs(x - y) for x, y in zip(near["center_3d"], a)) < 0.2

    # 远：B 明显偏离 A（射线到 A 中心 > merge_dist）→ 另起新桶，经两视角成物
    b = (6.0, 1.2, 5.0)
    cams_b = [Camera3D(cx=-2.0, cz=1.0, yaw=-0.3),
              Camera3D(cx=1.0, cz=0.0, yaw=0.1)]
    for cam in cams_b:
        px, py = cam.project(b, SW, SH)
        mv.add_observation("person", ViewObs(cam, (px, py), SW, SH))
    fused = mv.fused_objects()
    assert len(fused) == 2, (
        "#135：远距观测被并入 A 桶——merge_dist 未生效：%r" % fused)
    far = [f for f in fused if max(abs(x - y) for x, y in zip(f["center"], a)) > 0.5]
    assert len(far) == 1, "#135：B 桶中心 %r" % fused
    assert max(abs(x - y) for x, y in zip(far[0]["center"], b)) < 0.2, (
        "#135：B 桶中心 %r 偏离真值 %r" % (far[0]["center"], b))


def test_world3d_add_view_keeps_two_same_category_objects():
    """#135 端到端：World3D.add_view 对同类两物须产出两个世界物体。"""
    w = World3D()
    cams = [Camera3D(), Camera3D(cx=3.0, cz=-2.0, yaw=0.5)]
    truth = [(0.0, 1.2, 5.0), (2.5, 1.2, 5.0)]
    for cam in cams:
        for p in truth:
            px, py = cam.project(p, SW, SH)
            w.add_view("person", (px - 20, py - 20, px + 20, py + 20), SW, SH, camera=cam)
    centers = sorted(o.center for o in w.objects)
    assert len(centers) == 2, "#135：add_view 只产出 %d 个物体" % len(centers)
    for got, want in zip(centers, sorted(truth)):
        assert max(abs(a - b) for a, b in zip(got, want)) < 0.2, (got, want)


# ===========================================================================
# #220 认知基底层：稳定轮数不足必须翻 baseline_ok
# ===========================================================================

def test_baseline_ok_false_when_stable_rounds_insufficient():
    """#220：stable_rounds < 3 时 baseline_ok 必须为 False（旧码只写 note 不翻旗标）。

    缺陷形态：`cognitive_baseline(hit_rate=0.9, stable_rounds=0)` 旧码返回
    baseline_ok=True（稳定轮 0 却宣称基底层已建立）。
    """
    av = AnchoredVerification()
    r = av.cognitive_baseline(hit_rate=0.9, stable_rounds=0)
    assert r["baseline_ok"] is False, (
        "#220：稳定轮 0 却 baseline_ok=True——%r" % r)
    assert any("稳定轮数不足" in n for n in r["notes"]), r["notes"]

    r2 = av.cognitive_baseline(hit_rate=0.9, stable_rounds=2)
    assert r2["baseline_ok"] is False, "#220：稳定轮 2 仍应为 False——%r" % r2


def test_baseline_ok_true_when_both_conditions_met():
    """#220 非回归：命中率与稳定轮都达标 → baseline_ok=True（不得过度翻旗标）。"""
    av = AnchoredVerification()
    r = av.cognitive_baseline(hit_rate=0.9, stable_rounds=3)
    assert r["baseline_ok"] is True, "#220：条件均满足却 False——%r" % r
    assert r["notes"] == [], r["notes"]


def test_baseline_ok_false_on_low_hit_rate():
    """#220 非回归：命中率低于 0.4 仍须 False（原判据未被修复削弱）。"""
    av = AnchoredVerification()
    r = av.cognitive_baseline(hit_rate=0.2, stable_rounds=10)
    assert r["baseline_ok"] is False, "#220：低命中率却 True——%r" % r
    assert any("命中率低于基线" in n for n in r["notes"]), r["notes"]


# ===========================================================================
# #231 部件相对坐标：零点必须是物体中心
# ===========================================================================

def _parts_bbox(parts):
    xs, ys, zs = [], [], []
    for p in parts:
        dx, dy, dz = p["pos"]
        w, h, d = p["size"]
        xs += [dx - w / 2, dx + w / 2]
        ys += [dy - h / 2, dy + h / 2]
        zs += [dz - d / 2, dz + d / 2]
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2)


def test_every_shape_parts_centered_on_origin():
    """#231：每个内置形状的部件包围盒中心必须在原点（pos 相对物体中心）。

    缺陷形态（旧码）：person 部件包围盒 y 中心 = 0.405、table = −0.18、
    tree = 1.25 —— pos 的零点在物体底部而非中心，与 :29 契约冲突。
    """
    from lingshu.world.shapes import SHAPE_LIBRARY
    assert SHAPE_LIBRARY, "形状库为空"
    for cat, parts in SHAPE_LIBRARY.items():
        cx, cy, cz = _parts_bbox(parts)
        assert max(abs(cx), abs(cy), abs(cz)) < 1e-9, (
            "#231：%s 部件零点不是物体中心——bbox 中心=(%r,%r,%r)" % (cat, cx, cy, cz))


def test_ground_object_parts_straddle_object_center():
    """#231 端到端：地面物体装配后，部件包围盒中心须落在 obj.center（不再整体抬高）。

    world3d._draw_part 用 `part_center = 物体中心 + pos`；world3d.add_vprim 把
    地面物体中心置于 spec.size[1]/2（底部贴地）。旧码下 person 部件被整体抬高
    0.405m，装配后的部件中心 ≠ obj.center。
    """
    from lingshu.world.shapes import get_shape
    from lingshu.world.vprim import VPrim
    w = World3D()
    obj = w.add_vprim(VPrim("person", (300, 250, 340, 520), 0.9), SW, SH)
    parts = get_shape("person")
    ys = []
    for p in parts:
        wy = obj.center[1] + p["pos"][1]
        ys += [wy - p["size"][1] / 2, wy + p["size"][1] / 2]
    mid_y = (min(ys) + max(ys)) / 2
    assert abs(mid_y - obj.center[1]) < 1e-9, (
        "#231：部件装配后中心 y=%r 偏离物体中心 y=%r" % (mid_y, obj.center[1]))


# ===========================================================================
# 自身可收集
# ===========================================================================

def test_guard_is_collectable():
    assert os.path.basename(__file__).startswith("test_")
    assert MultiViewFusion is not None and World3D is not None
