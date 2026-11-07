# -*- coding: utf-8 -*-
"""test_issue2457_last4_facade_gates_guard —— seven_layer_loop/world_model 门面 entity/path 收口守卫。

缺陷（dsh-bounty #2457，#2424 复制粘贴族最后两个门面）
============================================================================
`seven_layer_loop` 与 `world_model` 门面的 `entity`/`path` 动作直接调用底层
（`add_entity`/`add_path`，下游闸抛可读 ValueError），无 try/except 转 error
dict ⇒ 坏输入不可读异常直达模型驱动面。已收口对照：voxel_world/scene_simulator/
world_learner/curiosity_explorer（#2424）/spacetime_consistency（#2423）——
本票补齐最后两个门面后全仓 action 门面错误契约统一。

实现侧口径（本守卫的判据）
--------------------------------------------------------------------------
四站点各包 try/except (ValueError, TypeError) ⇒ error dict（同 #2423/#2424
惯例）；合法输入行为逐位不变。

守卫用例
--------------------------------------------------------------------------
G1 sll entity 坏 pos ⇒ error dict；合法 ⇒ ok。
G2 sll path 坏 points ⇒ error dict。
G3 wm entity 坏 pos ⇒ error dict；合法 ⇒ ok。
G4 wm path 坏 points ⇒ error dict。
G5 文案可读：下游闸中文原因。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue2457_last4_facade_gates_guard.py -q
"""
from __future__ import annotations

import pytest

from lingshu.core.core import SpacetimeMemoryEngine


def _e():
    e = SpacetimeMemoryEngine(":memory:")
    e.world_model("init", {}); e.world_model("create", {})
    e.seven_layer_loop("create", {})
    return e


# ---------------------------------------------------------------- G1

def test_2457_sll_entity_bad_pos():
    e = _e()
    r = e.seven_layer_loop("entity", {"category": "a", "pos": [1, 2]})
    assert r.get("status") == "error" and "三元组" in r.get("error", ""), r
    r = e.seven_layer_loop("entity", {"category": "a"})
    assert r.get("status") == "ok", r


# ---------------------------------------------------------------- G2

def test_2457_sll_path_bad_points():
    e = _e()
    e.seven_layer_loop("entity", {"category": "a"})
    r = e.seven_layer_loop("path", {"path_id": "p", "points": "bad"})
    assert r.get("status") == "error", r


# ---------------------------------------------------------------- G3

def test_2457_wm_entity_bad_pos():
    e = _e()
    r = e.world_model("entity", {"category": "p", "pos": [1, 2]})
    assert r.get("status") == "error" and "三元组" in r.get("error", ""), r
    r = e.world_model("entity", {"category": "p"})
    assert r.get("status") == "ok", r


# ---------------------------------------------------------------- G4

def test_2457_wm_path_bad_points():
    e = _e()
    r = e.world_model("path", {"path_id": "p", "points": [[1, 2]]})
    assert r.get("status") == "error" and "path" in r.get("error", ""), r


# ---------------------------------------------------------------- G5

def test_2457_error_messages_readable():
    e = _e()
    r = e.world_model("entity", {"category": "p", "pos": [1, 2]})
    assert "收到" in r.get("error", ""), r
