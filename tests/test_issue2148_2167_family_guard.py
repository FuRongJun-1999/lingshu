# -*- coding: utf-8 -*-
"""test_issue2148_2167_family_guard · core/world 族四处修复回归钉
============================================================================
本件钉 PR fix/verify-gates-batch11 四处修复的**行为**（每条都是修前形态必红）：

  · #2148 core.world3d("add"/"add_view")：confidence/tick/timestamp 走与
    world3d 姊妹实现同口径的 `_meta_number` 闸——修前裸 `float()` 让 NaN/±inf
    原样入世界并经 status/graph 导出面漏出（严格 JSON dumps ValueError）。
  · #2150 core._eval_escalation_value_bound：value 侧补类型/有限性闸（与阈值侧
    #747 三闸同口径）——修前非数值对象可用富比较伪造布尔让危机触发器**无条件
    命中**、`str` 让 TypeError 逃出公开 API。
  · #2165 activation.activate：self_condition 的 [0,1] 权重域显式闸——修前只判
    `> 0.0`，w>1 时 act0 = max(w·act_prev, seeds) 成为几何放大算子（1.5^30
    ≈ 127834），并污染库内历史值被后续合法调用继承。
  · #2167 seven_layer_loop._verify + world_model.verify：hit_rate 与
    bound_coverage 分别计数——修前两者赋成同一个 rate（而 hits 判据是区间边界
    dist < p["bound"]、point_hits 是点半径 dist < hit_threshold），两字段永远
    同值、各自失去判别力。

运行（仓根）：python -X utf8 -m pytest tests/test_issue2148_2167_family_guard.py
"""
import json
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# ---------------------------------------------------------------- #2148

def _engine():
    from lingshu.core.core import SpacetimeMemoryEngine
    return SpacetimeMemoryEngine(db_path=":memory:")


def test_2148_nonfinite_confidence_rejected():
    e = _engine()
    for bad in (float("inf"), float("-inf"), float("nan")):
        r = e.world3d("add", {"category": "cup", "bbox": [1, 1, 10, 10],
                             "confidence": bad})
        assert r.get("status") == "error", (
            "非有限 confidence 必须 fail-closed（修前原样入世界）：%r -> %r"
            % (bad, r.get("status")))


def test_2148_non_numeric_confidence_rejected():
    e = _engine()
    for bad in ("x", None if False else "0.9"):
        r = e.world3d("add", {"category": "cup", "bbox": [1, 1, 10, 10],
                             "confidence": bad})
        assert r.get("status") == "error"


def test_2148_status_remains_strict_json():
    e = _engine()
    r = e.world3d("add", {"category": "cup", "bbox": [1, 1, 10, 10],
                          "confidence": 0.9})
    assert r.get("status") == "ok", "正常 confidence 不受影响"
    st = e.world3d("status")
    json.dumps(st, allow_nan=False)  # 不得抛 ValueError


# ---------------------------------------------------------------- #2150

def _with_esc():
    e = _engine()
    e.store.conn.execute(
        "INSERT INTO escalation_points VALUES (?,?,?,?,?,?,?,?)",
        ("esc", "ESC-900", "持续退化信号", "value > 0.9 时上报", "立即上报",
         "high", 1, 0.0))
    e.store.conn.commit()
    return e


def test_2150_spoofed_comparison_cannot_fire():
    e = _with_esc()

    class Exploit:
        def __float__(self):
            return 5e-324

        def __gt__(self, other):
            return True

    assert e.check_escalation("持续退化信号", value=Exploit()) == [], (
        "非数值对象不得靠富比较伪造命中（修前 -> ['ESC-900']）")


def test_2150_str_value_does_not_raise():
    e = _with_esc()
    r = e.check_escalation("持续退化信号", value="0.95")
    assert r == [], "str 值不得让 TypeError 逃出公开 API（修前 TypeError）"


def test_2150_legit_values_unchanged():
    e = _with_esc()
    assert [p.get("code") for p in
            e.check_escalation("持续退化信号", value=0.95)] == ["ESC-900"], (
        "正常越界值仍触发")
    assert e.check_escalation("持续退化信号", value=5e-324) == [], "正常小值不触发"
    assert e.check_escalation("持续退化信号", value=float("nan")) == [], (
        "NaN 保持 fail-closed 不触发")


# ---------------------------------------------------------------- #2165

def test_2165_self_condition_gate():
    import tempfile
    from lingshu.core.activation import ActivationEngine
    d = tempfile.mkdtemp(prefix="r63_2165_")
    e = ActivationEngine(db_path=os.path.join(d, "a.db"))
    for bad in (1.5, -0.1, float("nan"), float("inf"), "0.5", True):
        with pytest.raises(ValueError):
            e.activate("测试节点内容甲乙丙", self_condition=bad)


def test_2165_valid_self_condition_unchanged():
    import tempfile
    from lingshu.core.activation import ActivationEngine
    d = tempfile.mkdtemp(prefix="r63_2165b_")
    e = ActivationEngine(db_path=os.path.join(d, "a.db"))
    r = e.activate("测试节点内容甲乙丙", self_condition=0.5)
    assert isinstance(r, dict) and r.get("status") == "ok", (
        "合法权重路径不受影响：%r" % (r.get("status"),))


# ---------------------------------------------------------------- #2167

def test_2167_verify_fields_are_distinct():
    import inspect
    from lingshu.world.seven_layer_loop import SevenLayerLoop
    from lingshu.world import world_model as WM

    src = inspect.getsource(SevenLayerLoop._verify)
    assert '"bound_coverage": coverage' in src, (
        "seven_layer_loop._verify 的 bound_coverage 必须独立于 hit_rate（修前同值）")
    assert "round(point_hits / total" in src, "hit_rate 应归点命中计数"

    wm_src = inspect.getsource(WM.UnifiedWorldModel.verify)
    assert '"bound_coverage": coverage' in wm_src, (
        "world_model.verify 同站点同修")


def test_2167_point_vs_bound_semantics():
    """行为面：点阈值 < 边界 ⇒ 同一预测下 hit_rate ≤ bound_coverage 且可不同。"""
    from lingshu.world.seven_layer_loop import SevenLayerLoop
    loop = SevenLayerLoop()
    # 构造一个预测：距离落在 [hit_threshold, bound) 之间 ⇒ 覆盖但不点中
    loop.explorer.hit_threshold = 0.5
    eid = loop.add_entity("person", behavior="wander", pos=(2.0, 1.5, 2.0))
    node = loop.world.entities[eid]
    actual = tuple(node.pos)
    pred = {"predicted": (actual[0] + 0.8, actual[1], actual[2]),
            "bound": 1.0, "mode": "t"}
    rep = loop._verify({eid: pred})
    assert rep["total"] == 1
    assert rep["hit_rate"] == 0.0, "0.8 > 0.5 点阈值 ⇒ 点不命中：%r" % (rep["hit_rate"],)
    assert rep["bound_coverage"] == 1.0, (
        "0.8 < 1.0 边界 ⇒ 覆盖命中（修前两字段同值，无法区分）：%r"
        % (rep["bound_coverage"],))
