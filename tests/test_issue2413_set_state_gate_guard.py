# -*- coding: utf-8 -*-
"""test_issue2413_set_state_gate_guard —— WorldModel.set_state 未知实体名闸守卫。

缺陷（dsh-bounty #2413）
============================================================================
`scene_model.WorldModel.set_state` 对未知实体名**静默吞**：
`if name in self.entities: …` 不成立就什么都不做、照常返回 self——拼错实体名
（尾随空格/别名）时状态不生效、场景按 neutral 渲染、零信号。而同类 mutator
`relation()` 对同型错误（实体不存在）显式 `raise ValueError`（:103-104）——
一图两制，调用方无法建立稳定预期。与 #2391（add_entity behavior 非字符串）、
#2410（add_rejected_path path_type 非字符串）同族：身份参数被静默收下。

实现侧口径（本守卫的判据）
--------------------------------------------------------------------------
对齐 relation() 口径：未知实体 ⇒ `ValueError`（文案点名实体与在场景清单）。
范围：只闸「未知实体名」；`state` 值本身的域不在本票（呈现层
`STATE_TINT.get` 对任意可哈希值有缺省回落，属宽松设计）。
兼容性：全仓 `WorldModel.set_state` 零调用方（grep 核实）；`ingest_scene`
的状态写入走 SQL 直写不经本方法。

守卫用例
--------------------------------------------------------------------------
G1 缺陷态：未知实体名 ⇒ ValueError（改前静默返回 self）。
G2 正对照：已知实体状态真被更新（防"一刀切全拒"假绿）。
G3 一图一制：relation() 与 set_state 对同型错误（未知实体）同判 ValueError。
G4 范围控制：ingest_scene → load_world_from_memory 的状态写入路径不受影响
   （不经 set_state，状态照常落库与重建）。
G5 文案可判读：点名坏实体与在场景清单。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue2413_set_state_gate_guard.py -q
"""
from __future__ import annotations

import pytest

from lingshu.core.core import SpacetimeMemoryEngine
from lingshu.world.scene_model import WorldModel, ingest_scene, load_world_from_memory


def _wm():
    m = WorldModel()
    m.add_entity("A", "creature", (0.0, 0.0, 0.0))
    return m


# ---------------------------------------------------------------- G1

def test_2413_unknown_name_raises():
    m = _wm()
    with pytest.raises(ValueError) as ei:
        m.set_state("B不存在的实体", "happy")
    assert "B不存在的实体" in str(ei.value), str(ei.value)


# ---------------------------------------------------------------- G2

def test_2413_known_entity_state_updated():
    m = _wm()
    r = m.set_state("A", "happy")
    assert r is m, "链式语义保持"
    assert m.entities["A"].state == "happy", m.entities["A"].state


# ---------------------------------------------------------------- G3

def test_2413_same_contract_as_relation():
    m = _wm()
    with pytest.raises(ValueError):
        m.relation("不存在", "near", "A")
    with pytest.raises(ValueError):
        m.set_state("不存在", "happy")


# ---------------------------------------------------------------- G4

def test_2413_ingest_state_path_unaffected():
    class _Agent:
        def __init__(s, e):
            s.engine = e
            s.store = e.store
    e = SpacetimeMemoryEngine(":memory:")
    ids = ingest_scene(_Agent(e), "肥鱼|fatfish|1,2,3|shy")
    assert len(ids) == 1, ids
    wm = load_world_from_memory(e.store)
    assert wm.entities["肥鱼"].state == "shy", (
        "ingest→load 状态写入不经 set_state，须照常生效")


# ---------------------------------------------------------------- G5

def test_2413_message_names_entity_and_roster():
    m = _wm()
    try:
        m.set_state("幽灵", "sad")
    except ValueError as e:
        msg = str(e)
        assert "幽灵" in msg and "A" in msg, msg
