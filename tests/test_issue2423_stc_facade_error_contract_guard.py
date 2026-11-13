# -*- coding: utf-8 -*-
"""test_issue2423_stc_facade_error_contract_guard —— spacetime_consistency 门面错误契约守卫。

缺陷（dsh-bounty #2423）
============================================================================
`core.spacetime_consistency` 门面的 `entity`/`path`/`teleport`/`create`/`run`
五动作直接调用 `self._stc.*`（下游闸抛的 ValueError/TypeError）与裸
`int()`/`float()`，无任何转 error dict ⇒ 坏输入的**不可读异常直达模型驱动
面**（实测 `could not convert string to float: 'x'`、`pos 需为 (x,y,z) 有限
坐标三元组…` 均裸穿）。一图多制：
  · 同门面 `teleport` 对未知实体返回 `{"status": "error"}`（二态转换）；
  · 姊妹门面全转：`world3d`（camera/bbox/通道）、`wm_simloop`（#691 Z3）、
    `seven_layer_loop`（configure_action）。

实现侧口径（本守卫的判据）
--------------------------------------------------------------------------
五动作各包 try/except (ValueError, TypeError) ⇒ `{"status": "error",
"error": str(_e)}`（同 seven_layer #691 Z3 惯例）；合法输入行为逐位不变；
teleport 未知实体的二态返回保持。

守卫用例
--------------------------------------------------------------------------
G1 缺陷态：entity 坏 pos / path 坏 points / teleport 坏 pos / create 坏
   trees / run 坏 n ⇒ 门面 error dict（改前裸异常穿透）。
G2 正对照：合法 entity/create 照旧 ok。
G3 一图一制：teleport 未知实体仍走二态 error（既有语义保持）。
G4 文案可判读：error 含下游闸的可读原因（非实现层裸文本）。

运行（lingshu 仓根）：
    python -X utf8 -m pytest tests/test_issue2423_stc_facade_error_contract_guard.py -q
"""
from __future__ import annotations

import pytest

from lingshu.core.core import SpacetimeMemoryEngine


def _engine():
    e = SpacetimeMemoryEngine(":memory:")
    r = e.spacetime_consistency("init", {})
    assert r.get("status") == "ok", r
    r = e.spacetime_consistency("create", {})
    assert r.get("status") == "ok", r
    return e


# ---------------------------------------------------------------- G1

def test_2423_entity_bad_pos_error_dict():
    e = _engine()
    r = e.spacetime_consistency("entity", {"category": "p", "pos": [1.0, 2.0]})
    assert r.get("status") == "error" and "三元组" in r.get("error", ""), (
        "#2423 entity 坏 pos 仍裸穿（改前实测裸 ValueError）：%r" % (r,))


def test_2423_path_bad_points_error_dict():
    e = _engine()
    r = e.spacetime_consistency("path", {"path_id": "p1", "points": [[1, 2]]})
    assert r.get("status") == "error" and "path" in r.get("error", ""), r


def test_2423_teleport_bad_pos_error_dict():
    e = _engine()
    eid = e.spacetime_consistency("entity", {"category": "a"})["entity_id"]
    r = e.spacetime_consistency("teleport", {"entity_id": eid, "pos": ["x", 0, 0]})
    assert r.get("status") == "error", (
        "#2423 teleport 坏 pos 仍裸穿（改前 'could not convert string to float'）：%r" % (r,))


def test_2423_create_bad_trees_error_dict():
    e = _engine()
    r = e.spacetime_consistency("create", {"trees": "x"})
    assert r.get("status") == "error", r


def test_2423_run_bad_n_error_dict():
    e = _engine()
    r = e.spacetime_consistency("run", {"n": "x"})
    assert r.get("status") == "error" and "n" in r.get("error", "").lower() or \
        r.get("status") == "error", r


# ---------------------------------------------------------------- G2

def test_2423_legal_paths_unchanged():
    e = _engine()
    r = e.spacetime_consistency("entity", {"category": "ok1"})
    assert r.get("status") == "ok" and r.get("entity_id"), r
    r = e.spacetime_consistency("create", {"trees": 2})
    assert r.get("status") == "ok", r


# ---------------------------------------------------------------- G3

def test_2423_teleport_unknown_entity_two_state_kept():
    e = _engine()
    eid = e.spacetime_consistency("entity", {"category": "a"})["entity_id"]
    r = e.spacetime_consistency("teleport", {"entity_id": "ghost", "pos": [1, 1, 1]})
    assert r.get("status") == "error" and r.get("queued") is False, (
        "teleport 未知实体的二态语义（非异常路径）须保持：%r" % (r,))


# ---------------------------------------------------------------- G4

def test_2423_error_messages_readable():
    e = _engine()
    r = e.spacetime_consistency("entity", {"category": "p", "pos": [1.0, 2.0]})
    assert "received" not in r.get("error", "") and "三元组" in r.get("error", ""), (
        "error 应为下游闸的可读中文原因：%r" % (r,))
