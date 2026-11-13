# -*- coding: utf-8 -*-
"""test_issue1319_zero_edge_causal_guard · #1319 零边因果路径与缺口漏报守卫。

缺陷（issue #1319，本端在 main `612670d` 逐字复现）：同端点查询
（start==end 且图无自环）时 `_causal` 的 DFS 起点压栈（path=()）即命中
`current == end` ⇒ 追加一条**零边路径**——上层把「什么都没找到」当作
「找到了因果连接」；且 `paths` 非空 ⇒ 视图层不报 `causal_path_not_found`
缺口（漏报）。

修法：`current == end` 追加路径时要求 `path` 非空（零边不构成因果路径；
真自环边走出的环路径 path 非空，不受影响）⇒ 无自环 ⇒ paths 空 ⇒ 视图层
正常报 `causal_path_not_found`。

判据来源：issue #1319 最小复现三行对照表（action→result [8]／action→action
[0]／result→action []+缺口）＋ docs/scenario_usage.md 因果输出约定。

断言组：
  G1 同端点无自环：paths 空 ＋ gaps 含 causal_path_not_found（修前红）。
  G2 正常链路回归：跨端点路径边数 > 0，缺口不含 causal_path_not_found。
  G3 真自环边：start→start 自环走出的环路径 path 非空仍返回（不受修法影响）。
  G4 另一端点对（result→action）：无路径 ⇒ 缺口如实（修前后行为一致）。
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.core.core import ConditionSpace, EdgeType, LayeredStore, STEdge, STNode  # noqa: E402
from lingshu.world.scenario import ScenarioContext  # noqa: E402

_CHAIN = ["action", "stage-1", "result"]


def _ctx(extra_edges=()):
    """最小合成图：action → stage-1 → result 因果链（无自环），可注入额外边。"""
    store = LayeredStore(":memory:")
    cs = ConditionSpace("合成灯塔观察位", "离线验收", (100, 300), "合成灯塔场景")
    for i, nid in enumerate(_CHAIN):
        store.add_node(STNode(nid, f"节点 {nid}", "text", {}, 100 + i * 10, cs,
                              tags=["role:alice", "event"]))
    for i in range(len(_CHAIN) - 1):
        store.add_edge(STEdge(f"path-{i}", _CHAIN[i], _CHAIN[i + 1], EdgeType.CAUSAL,
                              cs, source_evidence="inferred"))
    for eid, src, dst in extra_edges:
        store.add_edge(STEdge(eid, src, dst, EdgeType.CAUSAL, cs,
                              source_evidence="extracted", verified=True))
    return ScenarioContext.from_store(store, _CHAIN, role_id="alice")


def _edges_of(rep):
    return [len(p["edges"]) for p in rep["payload"]["paths"]]


def test_g1_same_endpoint_no_self_loop():
    """G1：同端点无自环 ⇒ 零边路径拒绝 ＋ 缺口如实（修前红：paths=[0] 无缺口）。"""
    rep = _ctx().view("causal", start_id="action", end_id="action")
    edges = _edges_of(rep)
    assert edges == [], "零边路径不得当作因果路径返回：%r" % (edges,)
    assert "causal_path_not_found" in rep["gaps"], (
        "paths 空时缺口须如实报 causal_path_not_found：%r" % (rep["gaps"],))


def test_g2_normal_path_regression():
    """G2：跨端点正常链路回归（action→result 边数 > 0，缺口不含 causal_path_not_found）。"""
    rep = _ctx().view("causal", start_id="action", end_id="result")
    edges = _edges_of(rep)
    assert edges and all(e > 0 for e in edges), (
        "正常链路路径不得为空/零边：%r" % (edges,))
    assert "causal_path_not_found" not in rep["gaps"]


def test_g3_true_self_loop_marks_cyclic():
    """G3：真自环边 ⇒ cyclic 如实标记 ＋ 零边路径仍拒绝 ＋ 缺口如实。

    `_causal` 是防环简单路径搜索（target in seen 只标 cyclic 不压栈）——
    自环环路径在修前也不产出（修前返回的 [0] 是「起点即命中」的零边路径，
    与自环边无关），故本件修法不改变自环边的可见效果。"""
    ctx = _ctx(extra_edges=[("e_self", "action", "action")])
    rep = ctx.view("causal", start_id="action", end_id="action")
    assert rep["payload"]["cyclic"] is True, "自环边应被 DFS 如实标记 cyclic"
    edges = [len(p["edges"]) for p in rep["payload"]["paths"]]
    assert edges == [], "零边路径不得当作因果路径返回：%r" % (edges,)
    assert "causal_path_not_found" in rep["gaps"]


def test_g4_reverse_pair_gap_unchanged():
    """G4：result→action（不可达方向）无路径 ⇒ 缺口如实（修前后行为一致）。"""
    rep = _ctx().view("causal", start_id="result", end_id="action")
    edges = _edges_of(rep)
    assert edges == []
    assert "causal_path_not_found" in rep["gaps"] or any(
        g.startswith("qualification_pending") for g in rep["gaps"]), (
        "不可达方向缺口须如实：%r" % (rep["gaps"],))
