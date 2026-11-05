# -*- coding: utf-8 -*-
"""test_issue2381_scene_relation_kind_guard —— scene_model 关系 kind 闭集守卫。

缺陷（dsh-bounty #2381）
============================================================================
`lingshu/world/scene_model.py` 的语义关系 kind 零校验：
  · `WorldModel.relation()` docstring 声明闭集（left_of/right_of/near/far/
    in_front/behind），实现只查两端实体存在，kind 原样收边；
  · `apply_relations()` 六分支无 else——域外 kind（拼写错/大小写变体/任意串）
    静默 no-op；
  · `ingest_scene(relations=...)` 写路原样落库（metadata JSON）。
危害：调用方无法区分「约束已生效」与「kind 打错被忽略」，scene_text 还把坏
关系当真关系显示；与 #727 注释「约束静默失判（无告警）」同一危害形态。
同仓惯例：physics.MODEL_KINDS / shapes.PART_KINDS 同形闭集均 fail-loud。

实现侧口径（本守卫的判据）
--------------------------------------------------------------------------
写面 fail-loud（relation() 与 ingest_scene 的 relations= 参数均 raise
ValueError，与「relation requires both entities」的 caller 错误 raise 口径
一致）；读面（load_world_from_memory）存量坏 kind 边跳过不重建（修前写入口
不拦、存量库可能有坏 kind；沿 ingest_scene 对坏行「跳过不入库」既有口径）。

守卫用例
--------------------------------------------------------------------------
G1 缺陷态：拼错 kind（rigth_of）/ 大小写变体（Left_Of）/ 空串 ⇒ relation()
    raise ValueError（改前静默收边）。
G2 正对照：六种合法 kind 全部照旧收边（防"一刀切全拒"假绿）。
G3 缺陷态（写路）：ingest_scene(relations=[坏 kind]) ⇒ raise（改前落库）。
G4 读面存量兜底：存量 metadata 带坏 kind 边 ⇒ load_world_from_memory 不炸、
    坏边跳过、好边照常重建（范围控制：不破坏存量加载）。
G5 约束求解仍活着：合法 kind 数据下 right_of 真的把实体挪到右侧（正对照，
    防"恒不动假绿"）。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue2381_scene_relation_kind_guard.py -q
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lingshu.world.scene_model import (
    RELATION_KINDS,
    WorldModel,
    ingest_scene,
    load_world_from_memory,
)


class _Agent:
    """scene_model 消费面：同时提供 `.store` 与 `.engine`。"""

    def __init__(self, engine):
        self.engine = engine
        self.store = engine.store


def _engine():
    from lingshu.core.core import SpacetimeMemoryEngine
    return SpacetimeMemoryEngine(":memory:")


def _wm_pair():
    m = WorldModel()
    m.add_entity("A", "creature", (0.0, 0.0, 0.0))
    m.add_entity("B", "creature", (5.0, 0.0, 0.0))
    return m


# ---------------------------------------------------------------- G1

def test_2381_bad_kind_raises():
    """拼错 / 大小写变体 / 空串 ⇒ relation() 显式拒绝（改前静默收边）。"""
    for bad in ("rigth_of", "Left_Of", "infront", "", "near "):
        with pytest.raises(ValueError):
            _wm_pair().relation("A", bad, "B")
    # 错误文案可判读：点名坏 kind 与闭集
    try:
        _wm_pair().relation("A", "rigth_of", "B")
    except ValueError as e:
        assert "rigth_of" in str(e) and "right_of" in str(e), str(e)


# ---------------------------------------------------------------- G2

def test_2381_valid_kinds_still_accepted():
    """正对照：六种合法 kind 全部照旧收边。"""
    m = _wm_pair()
    for kind in RELATION_KINDS:
        m.relation("A", kind, "B")
    assert m.entities["A"].relations == [("B", k) for k in RELATION_KINDS], (
        m.entities["A"].relations)


# ---------------------------------------------------------------- G3

def test_2381_ingest_bad_relation_kind_raises():
    """写路：ingest_scene(relations=[坏 kind]) ⇒ raise（改前原样落库）。"""
    with pytest.raises(ValueError):
        ingest_scene(_Agent(_engine()),
                     "a|box|1,2,3|neutral\nb|box|4,5,6|neutral",
                     relations=[("a", "rigth_of", "b")])


# ---------------------------------------------------------------- G4

def _fake_scene_node(name, cat, x, y, z, relations_meta):
    """存量库节点的最小投影（load_world_from_memory 只读这几个面）。"""
    content = ("场景实体 %s（%s）位于 (%s,%s,%s)，状态 neutral" % (name, cat, x, y, z)
               + "\n```lingshu-scene\n"
               + json.dumps({"eid": name, "relations": relations_meta},
                            ensure_ascii=False)
               + "\n```")
    return SimpleNamespace(
        spatial_coordinates={"x": x, "y": y, "z": z},
        content=content,
        tags=["spatial", "cat:" + cat, "ent:" + name],
        state_attributes={})


def test_2381_load_skips_stored_bad_kind():
    """读面存量兜底：存量 metadata 带坏 kind 边 ⇒ 不炸、坏边跳过、好边重建。"""
    class _FakeStore:
        def __init__(self, nodes):
            self._nodes = nodes

        def get_nodes_by_tag(self, tag, limit=200):
            return self._nodes[:limit]

    nodes = [
        # 实体 a 的存量关系：一条好边 + 一条修前写入的坏 kind 边
        _fake_scene_node("a", "box", 0.0, 0.0, 5.0,
                         [("b", "right_of"), ("b", "rigth_of")]),
        _fake_scene_node("b", "box", 5.0, 0.0, 5.0, []),
    ]
    wm = load_world_from_memory(_FakeStore(nodes))
    kinds = [k for rels in (en.relations for en in wm.entities.values())
             for _, k in rels]
    assert "rigth_of" not in kinds, (
        "#2381 存量坏 kind 边被重建（读面兜底失效）：%r" % (kinds,))
    assert "right_of" in kinds, "好边照常重建：%r" % (kinds,)


# ---------------------------------------------------------------- G5

def test_2381_solver_still_moves_entities():
    """正对照：合法 kind 下约束求解真的挪实体（防"恒不动假绿"）。"""
    m = _wm_pair()
    m.relation("A", "right_of", "B")
    m.apply_relations()
    assert m.entities["A"].pos[0] > 5.0, (
        "right_of 约束应把 A 挪到 B 右侧：%r" % (m.entities["A"].pos,))
