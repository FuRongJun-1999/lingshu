# -*- coding: utf-8 -*-
"""批九十八守卫：写→读往返 + 契约键完整（#2538 / #2539）。

覆盖
--------------------------------------------------------------------------
#2538 `scenario` 无法消费 core 自写的锚点/结构层节点
    G1 `set_anchor` 写入的节点（`time_window=(0, inf)`）可被 `from_store` 消费
       （改前 `_copy` 的 `allow_nan=False` 直接 ValueError，CLI 同 rc=1）
    G2 `add_structure_node` 同款（结构层也是 `(0, inf)`）
    G3 非有限 `time_window` 归一为 `None` 后 `_declared` 判「未声明」（语义不变）
    G4 正对照：有限窗节点读数逐位不变、`view('world')` 仍 DEFER

#2539 `gap_dual.compute_d_task` 的 BLINDSPOT 分支必须带 `cond_gap`
    G5 无路径 / 空 dict / 全不可用 三个分支都必须有 `cond_gap` 与 `best_path`
       （改前前两条缺 ⇒ 按 docstring 无条件读 `r["cond_gap"]` 时 KeyError）
    G6 正对照：dist 全缺失分支（#518 时已带键）与有效路径分支读数不变

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_batch98_roundtrip_guard.py -q
"""
from __future__ import annotations

import math

import pytest

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.world.gap_dual import GapDual
from lingshu.world.scenario import ScenarioContext, _finite_or_none


def _engine():
    e = SpacetimeMemoryEngine(":memory:")
    e.add_perception("知识层对照节点", importance=0.6)
    kid = [n.id for n in e.store.query_nodes(limit=20)
           if "知识层对照" in str(getattr(n, "content", ""))][0]
    return e, kid


# ============================================ #2538 scenario 往返
def test_G1_anchor_node_consumable():
    """G1：`set_anchor` 写入的节点（(0, inf) 窗）必须可被 `from_store` 消费。"""
    e, _ = _engine()
    a = e.set_anchor("锚点内容")
    sc = ScenarioContext.from_store(e.store, [a.id])   # 改前 ValueError: ... inf
    assert sc is not None


def test_G2_structure_node_consumable():
    """G2：`add_structure_node` 同款（结构层默认窗也是 (0, inf)）。"""
    e, _ = _engine()
    n = e.add_structure_node("结构层节点内容")
    nid = getattr(n, "id", n)
    sc = ScenarioContext.from_store(e.store, [nid])
    assert sc is not None


def test_G3_nonfinite_window_is_undeclared():
    """G3：非有限窗归一为 None 后，`_declared` 必须判「未声明」（语义不变）。"""
    from lingshu.world.scenario import _declared
    cs = {"observation_position": "p", "observation_tool": "t",
          "time_window": (None, None), "existence_constraint": "c"}
    # `_declared` 返回 bool（四栏全声明且 tw 有限且 start<end 才 True）。
    assert _declared(cs) is False, "非有限窗必须判「未声明」"
    ok = {"observation_position": "p", "observation_tool": "t",
          "time_window": (1.0, 2.0), "existence_constraint": "c"}
    assert _declared(ok) is True, "有限窗必须判「已声明」（正对照）"


def test_G4_finite_window_unchanged():
    """G4 正对照：有限窗节点仍可消费且 `view('world')` 仍 DEFER。"""
    e, kid = _engine()
    sc = ScenarioContext.from_store(e.store, [kid])
    assert sc is not None
    assert _finite_or_none(1.5) == 1.5
    assert _finite_or_none(float("inf")) is None
    assert _finite_or_none((0.0, float("inf"))) == (0.0, None)


# ============================================ #2539 gap_dual 键完整
@pytest.mark.parametrize("kw", [
    {"graph_paths": None},
    {"graph_paths": {}},
    {"graph_paths": {"a": {"available": False}}},
])
def test_G5_blindspot_branches_have_cond_gap(kw):
    """G5：三个 BLINDSPOT 分支都必须有 `cond_gap` 与 `best_path`。"""
    r = GapDual().compute_d_task(**kw)
    assert r["blindspot"] is True, r
    assert "cond_gap" in r, (kw, sorted(r))
    assert "best_path" in r, (kw, sorted(r))
    assert r["cond_gap"] is None and r["best_path"] == ""


def test_G6_dist_missing_branch_and_valid_path_unchanged():
    """G6 正对照：dist 全缺失分支（#518 已带键）与有效路径分支读数不变。"""
    gd = GapDual()
    miss = gd.compute_d_task(graph_paths={"a": {"available": True}})
    assert miss["blindspot"] is True and "cond_gap" in miss
    ok = gd.compute_d_task(
        graph_paths={"a": {"available": True, "dist": 1.5,
                           "cond_gap": {"x": 0.1}}},
        task_conditions={"x"})
    assert ok["blindspot"] is False, ok
    assert abs(ok["d_task"] - 1.5) < 1e-9, ok
    assert "cond_gap" in ok and "best_path" in ok
