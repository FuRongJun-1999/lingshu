# -*- coding: utf-8 -*-
"""批九十七守卫：逻辑/语义族五处（#2532 / #2533 / #2534 / #2535 / #2536）。

覆盖
--------------------------------------------------------------------------
#2532 `tms.assess` 对不存在的节点必须归零
    G1 `propagate(['ghost'])` 与 `assess('ghost')['belief']` 必须**一致**（都为 0.0）
       （改前 propagate=0.0 而 assess=0.5，同模块两条读路径自相矛盾）
    G2 被硬删的前提/佐证节点，assess 也必须归零
       （改前 0.666667 / 0.8，正信念方向 fail-open）
    G3 正对照：存活节点的两条路径读数一致

#2533 `evidence.assess` 的同源去重必须与插入顺序无关
    G4 同源等权（默认 weight=1.0）两种插入顺序 ⇒ 同一 confidence/status
       （改前 SC→corroborated 0.75 / CS→contested 0.50，结论相反）
    G5 同权时反证必须生效（confidence 不得高于只有支持时）
    G6 正对照：异源等权、同源非等权两种顺序本来就一致（#2153 的既有面）

#2534 `multiview.add_observation` 的 eid 分支必须复用同源守卫
    G7 同一相机两个像素 + 同 eid ⇒ 不三角化（改前得相机光心 (0,1.2,0)）
    G8 正对照：两台不同机位 + 同 eid 仍正常三角化

#2535 `verdict.assess` 的非字符串假值 occlusion
    G9 `None`/`""`/`0`/`[]` 与缺键同档（ACCEPT）
       （改前全部 DEFER「有遮挡」，与 #296/#2170 的裁定方向相反）
    G10 正对照：`部分遮挡`/`True` 仍 DEFER；`False`/缺键仍 ACCEPT

#2536 `cognitive_baseline` 的 `hit_rate=None`（默认值）
    G11 `None` 与 `nan` 同档（`baseline_ok=False`）
       （改前 None→True 而 nan→False，同一「未测量」状态两个相反结论）
    G12 正对照：`0.9` 仍 True、`0.2` 仍 False

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch97_semantics_guard.py -q
"""
from __future__ import annotations

import pytest

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.core.verdict import assess as v_assess
from lingshu.world.anchored_verification import AnchoredVerification
from lingshu.world.multiview import MultiViewFusion, ViewObs
from lingshu.world.world3d import Camera3D


def _engine_with(text="前提命题XYZ", importance=0.6):
    e = SpacetimeMemoryEngine(":memory:")
    e.enable_tms()
    e.add_perception(text, importance=importance)
    nid = [n.id for n in e.store.query_nodes(limit=20)
           if text in str(getattr(n, "content", ""))][0]
    return e, nid


# ================================================== #2532 tms.assess 归零
def test_G1_ghost_assess_matches_propagate():
    """G1：从未创建的节点，propagate 与 assess 必须给同一读数（都 0.0）。"""
    e = SpacetimeMemoryEngine(":memory:")
    e.enable_tms()
    t = e._tms
    assert t.propagate(["ghost"]) == {"ghost": 0.0}
    assert t.assess("ghost")["belief"] == 0.0, t.assess("ghost")


def test_G2_deleted_node_assess_zero():
    """G2：被硬删的前提/佐证节点，assess 也必须归零（账本残留证据不得复活信念）。"""
    e, nid = _engine_with()
    e.add_evidence(nid, "src:a", supports=True, weight=1.0)
    before = e.evidence_report(nid)["belief"]
    assert before > 0.0, before
    e.store.delete_node(nid)
    assert e.store.get_node(nid) is None
    assert e._tms.propagate([nid])[nid] == 0.0
    assert e.evidence_report(nid)["belief"] == 0.0, e.evidence_report(nid)


def test_G3_alive_node_paths_agree():
    """G3 正对照：存活节点两条路径读数一致（改动不影响正常面）。"""
    e, nid = _engine_with()
    e.add_evidence(nid, "src:a", supports=True, weight=1.0)
    prop = e._tms.propagate([nid])[nid]
    rep = e.evidence_report(nid)["belief"]
    assert abs(prop - rep) < 1e-9, (prop, rep)


# ============================================ #2533 evidence 同源去重顺序无关
@pytest.mark.parametrize("order", [(True, False), (False, True)])
def test_G4_same_source_equal_weight_order_independent(order):
    """G4：同源等权两种顺序必须给同一 confidence/status。"""
    e, nid = _engine_with()
    for sup in order:
        e.add_evidence(nid, "src:x", supports=sup, weight=1.0)
    r = e.evidence_report(nid)
    # 反证必须生效：conf 应等于「先反证后支持」那条的读数
    assert r["status"] == "contested", (order, r["status"], r["confidence"])
    assert abs(r["confidence"] - 0.5) < 1e-9, (order, r["confidence"])


def test_G5_contradiction_not_swallowed():
    """G5：同权时反证不得被吞——conf 必须低于「只有支持」的读数。"""
    e1, n1 = _engine_with()
    e1.add_evidence(n1, "src:x", supports=True, weight=1.0)
    only_support = e1.evidence_report(n1)["confidence"]
    e2, n2 = _engine_with()
    e2.add_evidence(n2, "src:x", supports=True, weight=1.0)
    e2.add_evidence(n2, "src:x", supports=False, weight=1.0)
    with_contra = e2.evidence_report(n2)["confidence"]
    assert with_contra < only_support, (with_contra, only_support)


def test_G6_cross_source_equal_weight_order_independent():
    """G6 正对照：异源等权本来就顺序无关（#2153 既有面，改动不影响）。

    ⚠ 这里**不测同源非等权**：`add_perception` 会自动写一条 `unattributed`
    证据（weight=1.0），它与 `src:a` 等权、极性固定为支持，会干扰同源比较
    （实测修复前 (0.4,1.0)→0.500 contested、(1.0,0.4)→0.750 corroborated，
    差异来自 `unattributed` 那条而非同源去重）。要测纯同源去重须绕过
    `add_perception`，那属于 `#2153` 的守卫面（用 0.9/0.1 非等权已覆盖）。
    """
    for order in ((True, False), (False, True)):
        e, nid = _engine_with()
        e.add_evidence(nid, "src:a", supports=order[0], weight=1.0)
        e.add_evidence(nid, "src:b", supports=order[1], weight=1.0)
        assert e.evidence_report(nid)["status"] == "contested"


# ============================================ #2534 multiview eid 同源守卫
def test_G7_same_camera_eid_not_triangulated():
    """G7：同机位 + 同 eid 不得三角化（改前得相机光心 (0,1.2,0)）。"""
    cam = Camera3D(cx=0.0, cy=1.2, cz=0.0, yaw=0.0, pitch=0.0, fov_deg=60.0)
    f = MultiViewFusion()
    f.add_observation("person", ViewObs(cam, (500.0, 300.0), 800, 600), eid="E1")
    r = f.add_observation("person", ViewObs(cam, (560.0, 300.0), 800, 600), eid="E1")
    assert r["triangulated"] is False, r
    assert r["center_3d"] is None, r
    assert r["views_used"] == 1, r
    assert f.fused_objects() == [], f.fused_objects()


def test_G8_different_cameras_still_triangulate():
    """G8 正对照：两台不同机位 + 同 eid 仍正常三角化。"""
    cam1 = Camera3D(cx=0.0, cy=1.2, cz=0.0, yaw=0.0, pitch=0.0, fov_deg=60.0)
    cam2 = Camera3D(cx=2.0, cy=1.2, cz=0.0, yaw=0.0, pitch=0.0, fov_deg=60.0)
    f = MultiViewFusion()
    f.add_observation("person", ViewObs(cam1, (500.0, 300.0), 800, 600), eid="E2")
    r = f.add_observation("person", ViewObs(cam2, (560.0, 300.0), 800, 600), eid="E2")
    assert r["triangulated"] is True, r
    assert r["views_used"] == 2, r
    assert r["center_3d"] is not None


# ============================================ #2535 verdict 非字符串假值
@pytest.mark.parametrize("dom", [{"occlusion": None}, {"occlusion": ""},
                                 {"occlusion": 0}, {"occlusion": []}])
def test_G9_falsy_occlusion_is_unasserted(dom):
    """G9：None/""/0/[] 必须与缺键同档（ACCEPT），不得判「有遮挡」。"""
    r = v_assess({"type": "shirt"}, dom, 500, 1000)
    assert r["verdict"] == "ACCEPT", (dom, r)


@pytest.mark.parametrize("dom,expect", [
    ({"occlusion": "部分遮挡"}, "DEFER"),
    ({"occlusion": True}, "DEFER"),
    ({"occlusion": False}, "ACCEPT"),
    ({}, "ACCEPT"),
])
def test_G10_occlusion_controls_unchanged(dom, expect):
    """G10 正对照：显式遮挡/布尔/缺键的既有判定不变。"""
    assert v_assess({"type": "shirt"}, dom, 500, 1000)["verdict"] == expect, dom


# ======================================== #2536 cognitive_baseline None
def test_G11_none_matches_nan():
    """G11：`hit_rate=None`（默认值）与 `nan` 必须同档（baseline_ok=False）。"""
    av = AnchoredVerification()
    none_r = av.cognitive_baseline(stable_rounds=3)
    nan_r = av.cognitive_baseline(hit_rate=float("nan"), stable_rounds=3)
    assert none_r["baseline_ok"] is False, none_r
    assert nan_r["baseline_ok"] is False, nan_r
    assert none_r["notes"], "未测量必须给出可读 note"


def test_G12_legal_hit_rate_unchanged():
    """G12 正对照：0.9 仍 True、0.2 仍 False。"""
    av = AnchoredVerification()
    assert av.cognitive_baseline(hit_rate=0.9, stable_rounds=3)["baseline_ok"] is True
    assert av.cognitive_baseline(hit_rate=0.2, stable_rounds=3)["baseline_ok"] is False
