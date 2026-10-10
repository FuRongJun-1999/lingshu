# -*- coding: utf-8 -*-
"""#139 subgraph_replace 先删后挂、无范围/保护校验 —— 守卫（lingshu/core/core.py）

缺陷（lingshu issue #139；分诊表 `triage_lingshu_core_front.md` 行 139 判定成立）：
  `SpacetimeMemoryEngine.subgraph_replace` 旧实现：
      removed = [old_sub_root_id] + [r["node_id"] for r in old_desc]
      for nid in removed:
          self.store.delete_node(nid)          # ← 删除早于任何校验，布尔返回被丢弃
      new_root_id = _attach(new_subtree, parent_id)   # ← 其后才挂新子树
  ① `new_subtree` 缺 `"id"` ⇒ `_attach` 抛 `KeyError`，旧子树**已删**（删了没挂上）；
  ② 传 `old_sub_root_id == parent_id` 即删父；
  ③ 旧子树含受保护/不可变层节点 ⇒ 该点被拒删、其子孙照删，且 `removed_nodes`
     仍报候选集长度（谎报成功）。
  判据来源（修法）：设计稿 `docs/plans/删除恢复持久化完整性_修复设计_v0.1.md` §2
  （子缺陷 ①②③⑤）＋ 待裁清单 **B-1**（保护判定沿用既有角色闸）/ **S-2=(a)**
  （保护命中 ⇒ 整体拒绝，全有或全无）/ **S-3**（范围校验）。

断言组（抽掉任一修复即红）：
  G1 新子树缺 "id"（深层） ⇒ 抛错，且旧子树**逐节点完好**、节点总数不变。
  G2 `old_sub_root_id == parent_id` ⇒ 拒绝，父节点及其子树完好。
  G3 旧子根不是父的 hierarchical 后代（范围外） ⇒ 拒绝，范围外子树完好。
  G4 旧子树含受保护节点 ⇒ 整体拒绝，受保护点**与其子孙**全部完好，
     且库内节点数不变（不得部分替换）。
  G5 正对照：合法替换照旧生效（旧子树消失、新根挂到 parent）。
  G6 `removed_nodes` 报**实际删除成功数**（与库内旧子树节点数逐项核对）。
  G7 旧子根不存在 / parent 不存在 ⇒ 拒绝（不静默建孤儿）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue139_subgraph_replace_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import (  # noqa: E402
    EdgeType, MemoryLayer, SpacetimeMemoryEngine,
)


def _engine():
    return SpacetimeMemoryEngine(":memory:")


def _mk_subtree(root_id, children=()):
    return {"id": root_id, "content": {"shape": root_id}, "importance": 0.6,
            "spatial_coordinates": {"image2d": [1, 2]},
            "subgraph": {"nodes": [_mk_subtree(c) for c in children], "edges": []}}


def _old_subtree(eng, parent_id, root_id, child_ids=()):
    """在 parent 下建一棵 hierarchical 旧子树（子→父），返回全部节点 id 列表。"""
    root = eng.add_perception("旧子根-" + root_id, skip_dedup=True)
    eng.add_edge(root.id, parent_id, relation_type=EdgeType.HIERARCHICAL)
    ids = [root.id]
    for cid in child_ids:
        ch = eng.add_perception("旧子-" + cid, skip_dedup=True)
        eng.add_edge(ch.id, root.id, relation_type=EdgeType.HIERARCHICAL)
        ids.append(ch.id)
    return root.id, ids


def test_g1_missing_id_in_new_subtree_leaves_old_intact():
    """G1：新子树缺 "id" ⇒ 抛错，旧子树逐节点完好（旧缺陷：旧子树已删）。"""
    e = _engine()
    parent = e.add_perception("父节点", skip_dedup=True)
    root, ids = _old_subtree(e, parent.id, "r", ["c1", "c2"])
    before = e.store.get_stats()

    bad = {"id": "newroot", "content": {},
           "subgraph": {"nodes": [{"content": {}, "id": ""}], "edges": []}}  # 深层缺 id
    raised = None
    try:
        e.subgraph_replace(parent.id, root, bad)
    except Exception as ex:      # noqa: BLE001
        raised = ex
    assert raised is not None, "G1 非法新子树未被拒（旧缺陷：先删后挂）"
    for nid in ids:
        assert e.store.get_node(nid) is not None, "G1 旧子树节点被删：%s" % nid
    assert e.store.get_stats() == before, \
        "G1 被拒后库内发生变化：%r -> %r" % (before, e.store.get_stats())


def test_g2_old_root_equal_parent_rejected():
    """G2：old_sub_root_id == parent_id ⇒ 拒绝（旧缺陷：直接删父）。"""
    e = _engine()
    parent = e.add_perception("父节点", skip_dedup=True)
    before = e.store.get_stats()
    try:
        e.subgraph_replace(parent.id, parent.id, _mk_subtree("newroot"))
        raise AssertionError("G2 传父节点自身未被拒（旧缺陷行为）")
    except ValueError:
        pass
    assert e.store.get_node(parent.id) is not None, "G2 父节点被删"
    assert e.store.get_stats() == before, "G2 被拒后库内变化"


def test_g3_out_of_range_old_root_rejected():
    """G3：旧子根不是 parent 的 hierarchical 后代 ⇒ 拒绝，范围外子树完好。"""
    e = _engine()
    parent = e.add_perception("父节点", skip_dedup=True)
    other = e.add_perception("无关节点", skip_dedup=True)
    # 造一棵挂在 other 下的子树（与 parent 无层级关系）
    orphan_root, orphan_ids = _old_subtree(e, other.id, "o", ["o1"])
    before = e.store.get_stats()
    try:
        e.subgraph_replace(parent.id, orphan_root, _mk_subtree("newroot"))
        raise AssertionError("G3 范围外旧子根未被拒（旧缺陷行为）")
    except ValueError:
        pass
    for nid in orphan_ids:
        assert e.store.get_node(nid) is not None, "G3 范围外节点被删：%s" % nid
    assert e.store.get_stats() == before, "G3 被拒后库内变化"


def test_g4_protected_descendant_causes_whole_reject():
    """G4：旧子树含受保护节点 ⇒ 整体拒绝，其子孙全部完好、节点数不变。"""
    e = _engine()
    parent = e.add_perception("父节点", skip_dedup=True)
    root, ids = _old_subtree(e, parent.id, "r", ["c1", "c2"])
    # 保护中间那个子节点（旧缺陷：该点拒删、其子孙照删，且谎报成功）
    protected = ids[1]
    e.protect_node(protected, "守卫：不可遗忘")
    before = e.store.get_stats()

    try:
        e.subgraph_replace(parent.id, root, _mk_subtree("newroot"))
        raise AssertionError("G4 受保护节点在旧子树中未被拒（旧缺陷行为）")
    except PermissionError:
        pass
    for nid in ids:
        assert e.store.get_node(nid) is not None, "G4 旧子树节点被删：%s" % nid
    assert e.store.get_stats() == before, "G4 部分替换发生了：库内变化"


def test_g5_valid_replace_still_works():
    """G5：正对照——合法替换照旧生效。"""
    e = _engine()
    parent = e.add_perception("父节点", skip_dedup=True)
    root, ids = _old_subtree(e, parent.id, "r", ["c1"])
    ret = e.subgraph_replace(parent.id, root, _mk_subtree("newroot", ["nc1"]))
    for nid in ids:
        assert e.store.get_node(nid) is None, "G5 旧子树节点未移除：%s" % nid
    assert e.store.get_node(ret["new_root_id"]) is not None, "G5 新根未挂入"
    assert ret["new_nodes"] == 2, "G5 新节点计数不符：%r" % (ret["new_nodes"],)
    # 新根以 hierarchical 边指向 parent
    out = e.store.traverse(parent.id, relation_types=["hierarchical"], direction="in",
                           max_depth=2)
    assert ret["new_root_id"] in {r["node_id"] for r in out}, "G5 新根未挂到 parent"


def test_g6_removed_nodes_counts_actual_deletions():
    """G6：removed_nodes 报实际删除成功数（与库内旧子树节点数核对）。"""
    e = _engine()
    parent = e.add_perception("父节点", skip_dedup=True)
    root, ids = _old_subtree(e, parent.id, "r", ["c1", "c2"])
    ret = e.subgraph_replace(parent.id, root, _mk_subtree("newroot"))
    assert ret["removed_nodes"] == len(ids), \
        "G6 removed_nodes 与实际不符：%r vs %r" % (ret["removed_nodes"], len(ids))
    for nid in ids:
        assert e.store.get_node(nid) is None


def test_g7_missing_endpoints_rejected():
    """G7：旧子根/父不存在 ⇒ 拒绝（不静默建孤儿）。"""
    e = _engine()
    parent = e.add_perception("父节点", skip_dedup=True)
    for args in [(parent.id, "no_such_root"), ("no_such_parent", parent.id)]:
        try:
            e.subgraph_replace(args[0], args[1], _mk_subtree("newroot"))
            raise AssertionError("G7 端点缺失未被拒：%r" % (args,))
        except ValueError:
            pass
    assert e.store.get_stats()["knowledge_nodes"] == 1, "G7 被拒后库内变化"
