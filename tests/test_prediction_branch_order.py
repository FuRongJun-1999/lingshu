# -*- coding: utf-8 -*-
"""prediction 分支候选截断守卫（对应 issue #561）。

`_generate_routes` 用 `self._branch_candidates(current)[:max_branches]` 截断，
而 `_branch_candidates` 修前按「因果边 DB 返回序 + 语义邻近序」拼接、不排序，
于是 max_branches 的实际含义是「取 DB 返回的前 N 条」而非「取最可信的 N 条」。
"""
import os
import shutil
import tempfile
import time

import pytest

from lingshu.core.core import (ConditionSpace, EdgeType, SpacetimeMemoryEngine,  # noqa: E402
                                STNode)
from lingshu.world.prediction import PredictionEngine  # noqa: E402


def _cs():
    return ConditionSpace("桌", "文本", (0.0, 0.0), "c")


def _mk(engine, nid, content="o"):
    n = STNode(id=nid, content=content + nid, modality="text",
               spatial_coordinates={}, temporal_coordinate=time.time(),
               condition_space=_cs())
    engine.store.add_node(n)
    return n


@pytest.fixture()
def engine():
    """每个用例一个独立临时库，避免同一 engine 上重复建图。"""
    d = tempfile.mkdtemp()
    e = SpacetimeMemoryEngine(os.path.join(d, "g.db"))
    yield e
    shutil.rmtree(d, ignore_errors=True)


def _linear_graph(engine, order):
    """root -> order 中每个 id 一条 causal 边（confidence 0.11），最后一条强边 0.99。"""
    _mk(engine, "root", "root")
    for tgt in order:
        _mk(engine, tgt)
        engine.add_edge("root", tgt, EdgeType.CAUSAL, confidence=0.11)
    _mk(engine, "strong")
    engine.add_edge("root", "strong", EdgeType.CAUSAL, confidence=0.99)


def test_max_branches_keeps_highest_confidence_candidate(engine):
    """核心：max_branches=3 时，0.99 的强因果边必须入选（不被 0.11 的弱边挤出）。"""
    _linear_graph(engine, ["weak1", "weak2", "weak3", "weak4", "weak5"])
    p = PredictionEngine(engine)
    routes = p.predict_routes(start_id="root", horizon=1, max_branches=3)
    paths = [tuple(r["path"]) for r in routes["routes"]]
    assert any("strong" in path for path in paths), paths


def test_candidates_sorted_by_confidence_desc(engine):
    """_branch_candidates 返回必须按可信度降序。"""
    _linear_graph(engine, ["weak1", "weak2", "weak3"])
    p = PredictionEngine(engine)
    cands = p._branch_candidates("root")
    confs = [c[1] for c in cands]
    assert confs == sorted(confs, reverse=True), confs
    assert confs[0] == pytest.approx(0.99)


def test_branch_candidates_are_insertion_order_independent(engine):
    """同一张图、两种插入序 ⇒ 候选 id 序列逐位相同。"""
    _linear_graph(engine, ["a", "b", "c", "d", "e"])
    p = PredictionEngine(engine)
    first = [c[0] for c in p._branch_candidates("root")]

    d2 = tempfile.mkdtemp()
    e2 = SpacetimeMemoryEngine(os.path.join(d2, "g.db"))
    try:
        _linear_graph(e2, ["e", "d", "c", "b", "a"])
        p2 = PredictionEngine(e2)
        second = [c[0] for c in p2._branch_candidates("root")]
        assert first == second, (first, second)
    finally:
        shutil.rmtree(d2, ignore_errors=True)


def test_routes_are_reproducible_across_insertion_order(engine):
    """同一张图、两种插入序 ⇒ predict_routes 路线集合逐位相同。"""
    _linear_graph(engine, ["a", "b", "c", "d", "e"])
    p = PredictionEngine(engine)
    r1 = sorted(tuple(r["path"]) for r in
                p.predict_routes(start_id="root", horizon=1, max_branches=3)["routes"])

    d2 = tempfile.mkdtemp()
    e2 = SpacetimeMemoryEngine(os.path.join(d2, "g.db"))
    try:
        _linear_graph(e2, ["e", "d", "c", "b", "a"])
        p2 = PredictionEngine(e2)
        r2 = sorted(tuple(r["path"]) for r in
                    p2.predict_routes(start_id="root", horizon=1, max_branches=3)["routes"])
        assert r1 == r2, (r1, r2)
    finally:
        shutil.rmtree(d2, ignore_errors=True)


def test_ties_broken_deterministically(engine):
    """同分候选必须按 target_id 定序（切断对 rowid 的依赖）。"""
    _linear_graph(engine, ["zeta", "alpha", "mid"])
    p = PredictionEngine(engine)
    cands = [c for c in p._branch_candidates("root") if c[1] == pytest.approx(0.11)]
    ids = [c[0] for c in cands]
    assert ids == sorted(ids), ids


def test_guard_is_discriminating(engine):
    """判别力锚定：修前的「拼接不排序」在 0.99 晚插时必然把它挤出前 3。"""
    _linear_graph(engine, ["weak1", "weak2", "weak3", "weak4", "weak5"])
    p = PredictionEngine(engine)
    raw = []
    for e in engine.store.get_outgoing_edges("root"):
        if e.relation_type.value in ("causal", "sequential"):
            raw.append((e.target_id, e.confidence))
    # 修前语义：直接取 DB 返回序前 3 条
    naive = [t for t, _ in raw[:3]]
    assert "strong" not in naive, naive          # 修前确实挤掉强边
    # 修后语义：按置信度降序取前 3 条
    fixed = [t for t, _ in sorted(raw, key=lambda x: -x[1])[:3]]
    assert "strong" in fixed, fixed
