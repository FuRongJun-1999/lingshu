# -*- coding: utf-8 -*-
"""Min-Heap Greatest Fixed Point (GFP) 传播算法的等价性、边界、守卫与扩展性测试。

覆盖：
  1. 随机 DAG / 含环超图 / 多前提合取 / 多依据析取 / 闭包外前提 / 已佐证豁免：
     与参考 Gauss-Seidel 实现逐位一致（含返回键序、写回 round(b, 6)、幂等性、
     闭包外前提只读不落库）。
  2. 显式边界：已删/缺失节点（0.0）、重复/共享前提、自支撑环（GFP 保留而非 LFP 归零）、
     弱链环、析取逃生分支、已佐证节点下游传导、#769 共享层写权限闸与事务回滚原子性。
  3. 鉴别守卫：证明若退化为 LFP、破坏已佐证/环不动点、或回退到逐轮扫描 Gauss-Seidel，
     对应断言必红。
  4. 大闭包（500–2000 节点）扩展性健全检查。
"""
from __future__ import annotations

import random
import sqlite3
import time

import pytest

from lingshu.core import tms as TMS
from lingshu.core.core import ConditionSpace, MemoryLayer, Role, SpacetimeMemoryEngine, STNode

pytestmark = pytest.mark.filterwarnings("ignore")

_CS_JSON = ConditionSpace("tms-gfp", "tms-gfp", (0.0, 1e12), "ok").to_json()


def _reference_gauss_seidel(nodes, own, just_cache, ext_belief_fn):
    """参考实现：原 O(|V_closure| * |E_closure|) Gauss-Seidel 最大不动点迭代。"""
    belief = {n: own[n]["confidence"] for n in nodes}
    for _ in range(len(nodes) + 1):
        changed = False
        for n in nodes:
            o = own[n]
            js = just_cache.get(n)
            if not js or o["status"] == "corroborated":
                b = o["confidence"]
            else:
                def bp(p):
                    return belief[p] if p in belief else ext_belief_fn(p)

                jsup = max(min(bp(p) for p in j) for j in js)
                b = min(o["confidence"], jsup)
            if abs(b - belief[n]) > 1e-12:
                belief[n] = b
                changed = True
        if not changed:
            break
    return belief


def _reference_lfp(nodes, own, just_cache, ext_belief_fn):
    """错误对照实现：最小不动点（LFP，有依据的非佐证节点从 0.0 起算，会令自支撑环塌缩）。"""
    belief = {}
    for n in nodes:
        o = own[n]
        js = just_cache.get(n)
        belief[n] = o["confidence"] if (not js or o["status"] == "corroborated") else 0.0
    for _ in range(len(nodes) + 1):
        changed = False
        for n in nodes:
            o = own[n]
            js = just_cache.get(n)
            if not js or o["status"] == "corroborated":
                continue

            def bp(p):
                return belief[p] if p in belief else ext_belief_fn(p)

            jsup = max(min(bp(p) for p in j) for j in js)
            b = min(o["confidence"], jsup)
            if abs(b - belief[n]) > 1e-12:
                belief[n] = b
                changed = True
        if not changed:
            break
    return belief


def _bulk_insert_nodes(engine, node_confs, layer="knowledge"):
    with engine.store._structure_event_transaction() as c:
        c.executemany(
            "INSERT INTO nodes (id, content, modality, spatial_coordinates, "
            "temporal_coordinate, condition_space, importance, confidence, layer, tags) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(nid, nid, "text", "{}", 1.0, _CS_JSON, 0.5, conf, layer, "[]")
             for nid, conf in node_confs],
        )


def _bulk_insert_justifications(engine, entries):
    """entries: list of (jid, conclusion, premises_iterable)."""
    with engine.store._structure_event_transaction() as c:
        c.executemany(
            "INSERT INTO justifications VALUES (?,?,?,?)",
            [(jid, concl, "", float(idx)) for idx, (jid, concl, _) in enumerate(entries)],
        )
        prem_rows = [(jid, p) for jid, _, prems in entries for p in sorted(set(prems))]
        c.executemany("INSERT INTO justification_premises VALUES (?,?)", prem_rows)


def _install_own(tms, own_map):
    """将 ledger.assess 覆写为确定性的 own_map（仅返回指定 confidence/status）。"""
    def assess(nid):
        entry = own_map.get(nid)
        if entry is None:
            return {"node_id": nid, "confidence": 0.5, "status": "unverified", "entries": 0}
        return {
            "node_id": nid,
            "confidence": entry["confidence"],
            "status": entry["status"],
            "entries": entry.get("entries", 1),
        }

    tms.ledger.assess = assess


def _run_random_parity_trial(seed, *, allow_cycles):
    rng = random.Random(seed)
    n_internal = rng.randint(10, 26)
    n_external = rng.randint(2, 6)
    palette = [0.0, 0.1, 0.25, 0.333333, 0.5, 0.666667, 0.75, 0.9, 1.0]

    int_ids = [f"n{i:02d}" for i in range(n_internal)]
    ext_ids = [f"ext{i:02d}" for i in range(n_external)]

    e = SpacetimeMemoryEngine(":memory:")
    try:
        tms = e.enable_tms()
        ext_confs = {nid: rng.choice(palette) for nid in ext_ids}
        int_init = {nid: rng.choice(palette) for nid in int_ids}
        _bulk_insert_nodes(e, list(int_init.items()) + list(ext_confs.items()))

        own_map = {}
        for nid in int_ids + ext_ids:
            conf = rng.choice(palette) if rng.random() < 0.6 else round(rng.uniform(0.0, 1.0), 6)
            status = "corroborated" if rng.random() < 0.18 else "single_source"
            own_map[nid] = {"confidence": conf, "status": status}
        _install_own(tms, own_map)

        just_entries = []
        jid_seq = 0
        for idx, concl in enumerate(int_ids):
            if not allow_cycles and idx == 0:
                continue
            if rng.random() < 0.2:
                continue
            pool = (int_ids[:idx] if not allow_cycles else [x for x in int_ids if x != concl]) + ext_ids
            n_just = rng.randint(1, 3)
            for _ in range(n_just):
                k = rng.randint(1, min(4, len(pool)))
                prems = rng.sample(pool, k)
                just_entries.append((f"j{jid_seq}", concl, prems))
                jid_seq += 1
        _bulk_insert_justifications(e, just_entries)

        # 随机删除 0..2 个节点（可能落在闭包内或闭包外），检验缺失节点归零语义
        deleted = set(rng.sample(int_ids + ext_ids, rng.randint(0, 2)))
        for nid in deleted:
            e.store.delete_node(nid)

        roots = rng.sample(int_ids, rng.randint(1, 3))
        closure = tms._closure(roots)
        closure_set = set(closure)

        # 构造参考输入（与 propagate 入口完全一致）
        ref_own = {}
        for n in closure:
            o = tms.ledger.assess(n)
            if e.store.get_node(n) is None:
                o = {**o, "confidence": 0.0}
            ref_own[n] = o
        ref_just = {n: tms.justifications_of(n) for n in closure}
        ext_before = {
            nid: (None if e.store.get_node(nid) is None else float(e.store.get_node(nid).confidence))
            for nid in ext_ids if nid not in closure_set
        }
        ref = _reference_gauss_seidel(closure, ref_own, ref_just, tms._external_belief)

        got = tms.propagate(roots)

        assert list(got.keys()) == list(ref.keys()), f"seed={seed}: key order mismatch"
        assert got == ref, f"seed={seed}: bit-for-bit parity mismatch\ngot={got}\nref={ref}"

        # 写回 round(b, 6) 逐位一致
        for n, v in ref.items():
            node = e.store.get_node(n)
            if n in deleted:
                assert node is None
            else:
                assert float(node.confidence) == round(v, 6), f"seed={seed}, node={n}"

        # 闭包外前提只读不写回
        for nid, old_val in ext_before.items():
            node = e.store.get_node(nid)
            cur_val = None if node is None else float(node.confidence)
            assert cur_val == old_val, f"seed={seed}: external premise {nid} was modified"

        # 幂等性：再次传播结果逐位相同
        second = tms.propagate(roots)
        assert second == got, f"seed={seed}: propagation not idempotent"
    finally:
        e.close()


# ============================================================ 1. 随机逐位等价
@pytest.mark.parametrize("seed", list(range(100, 125)))
def test_parity_randomized_dags(seed):
    """随机多前提/多依据 DAG（含外部前提、佐证豁免、缺失节点、置信度并列）逐位等价。"""
    _run_random_parity_trial(seed, allow_cycles=False)


@pytest.mark.parametrize("seed", list(range(200, 230)))
def test_parity_randomized_cyclic_hypergraphs(seed):
    """随机含环超图（自支撑环、弱链环、多析取分支、外部前提、佐证豁免）逐位等价。"""
    _run_random_parity_trial(seed, allow_cycles=True)


def test_parity_solver_stress_with_ties_duplicates_and_self_loops():
    """直接对 _solve_beliefs 跑 100 组随机高压测试（重度并列、重复前提、重复依据、自环、重复 nodes）。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        tms = e.enable_tms()
        palette = [0.0, 0.25, 0.5, 0.75, 1.0]
        for seed in range(300, 400):
            rng = random.Random(seed)
            n_nodes = rng.randint(5, 25)
            n_ext = rng.randint(1, 5)
            nodes = [f"v{i}" for i in range(n_nodes)]
            exts = [f"e{i}" for i in range(n_ext)]
            own = {
                n: {
                    "confidence": rng.choice(palette),
                    "status": "corroborated" if rng.random() < 0.2 else "single_source",
                }
                for n in nodes
            }
            ext_vals = {p: rng.choice(palette) for p in exts}
            just_cache = {}
            all_prems = nodes + exts
            for n in nodes:
                if rng.random() < 0.2:
                    just_cache[n] = []
                    continue
                js = []
                for _ in range(rng.randint(1, 4)):
                    k = rng.randint(1, 4)
                    # 允许重复前提与自环，检验 j_capped 与 finalized 守卫
                    js.append([rng.choice(all_prems) for _ in range(k)])
                if rng.random() < 0.25 and js:
                    js.append(list(js[0]))  # 重复依据
                just_cache[n] = js

            # 刻意在 nodes 末尾附加重复项，检验去重稳定性
            nodes_with_dup = nodes + ([nodes[0]] if rng.random() < 0.3 else [])
            tms._just_cache = just_cache
            tms._external_belief = lambda p, ev=ext_vals: ev[p]
            try:
                got = tms._solve_beliefs(nodes_with_dup, own)
            finally:
                tms._just_cache = {}

            ref = _reference_gauss_seidel(nodes, own, just_cache, lambda p, ev=ext_vals: ev[p])
            assert got == ref, f"seed={seed}: solver mismatch\ngot={got}\nref={ref}"
    finally:
        e.close()


# ============================================================ 2. 显式边界测试
def test_edge_self_supporting_cycle_and_weak_link_cycle():
    """自支撑环保持 GFP 高信念；含弱环节的环全环撤回至最弱节点；弱旁支不拉低自支撑环。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        tms = e.enable_tms()
        _bulk_insert_nodes(e, [(nid, 0.9) for nid in ("a", "b", "c", "w", "x", "y")])
        # 1) 自支撑环 a <-> b，另有一条弱析取依据 a <- [w]
        _bulk_insert_justifications(e, [
            ("j1", "a", ["b"]),
            ("j2", "a", ["w"]),
            ("j3", "b", ["a"]),
            # 2) 弱链 3-环 x <- y <- c <- x
            ("j4", "x", ["y"]),
            ("j5", "y", ["c"]),
            ("j6", "c", ["x"]),
        ])
        _install_own(tms, {
            "a": {"confidence": 0.85, "status": "single_source"},
            "b": {"confidence": 0.9, "status": "single_source"},
            "w": {"confidence": 0.15, "status": "single_source"},
            "x": {"confidence": 0.9, "status": "single_source"},
            "y": {"confidence": 0.7, "status": "single_source"},
            "c": {"confidence": 0.2, "status": "single_source"},
        })

        got_self = tms.propagate(["w"])
        assert got_self == {"w": 0.15, "a": 0.85, "b": 0.85}

        got_weak = tms.propagate(["x"])
        assert got_weak == {"x": 0.2, "c": 0.2, "y": 0.2}
    finally:
        e.close()


def test_edge_cycle_disjunctive_escape_and_external_premise_read_only():
    """环内节点凭闭包外替代依据逃生；外部前提只读且仅查询一次。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        tms = e.enable_tms()
        _bulk_insert_nodes(e, [("a", 0.9), ("b", 0.9), ("ext_ok", 0.7), ("ext_unused", 0.1)])
        _bulk_insert_justifications(e, [
            ("j1", "a", ["b"]),
            ("j2", "a", ["ext_ok"]),
            ("j3", "b", ["a"]),
            ("j4", "b", ["ext_ok"]),
        ])
        _install_own(tms, {
            "a": {"confidence": 0.9, "status": "single_source"},
            "b": {"confidence": 0.2, "status": "single_source"},
            "ext_ok": {"confidence": 0.99, "status": "single_source"},  # 闭包外不读 ledger
        })
        calls = []
        orig_ext = tms._external_belief

        def tracked_ext(p):
            calls.append(p)
            return orig_ext(p)

        tms._external_belief = tracked_ext
        got = tms.propagate(["a"])
        assert got == {"a": 0.7, "b": 0.2}
        assert "ext_ok" not in got
        assert calls == ["ext_ok"], f"external premise queried {len(calls)} times: {calls}"
        assert float(e.store.get_node("ext_ok").confidence) == 0.7
    finally:
        e.close()


def test_edge_duplicate_and_shared_premises():
    """同一前提共享于同一结论的多条依据与多个下游结论时，计数与撤回严格准确。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        tms = e.enable_tms()
        _bulk_insert_nodes(e, [(nid, 0.9) for nid in ("p", "q1", "q2", "c1", "c2")])
        _bulk_insert_justifications(e, [
            ("j1", "c1", ["p", "q1"]),
            ("j2", "c1", ["p", "q2"]),
            ("j3", "c2", ["p", "c1"]),
        ])
        _install_own(tms, {
            "p": {"confidence": 0.25, "status": "single_source"},
            "q1": {"confidence": 0.8, "status": "single_source"},
            "q2": {"confidence": 0.9, "status": "single_source"},
            "c1": {"confidence": 0.95, "status": "single_source"},
            "c2": {"confidence": 0.95, "status": "single_source"},
        })
        got = tms.propagate(["p"])
        assert got == {"p": 0.25, "c1": 0.25, "c2": 0.25}
    finally:
        e.close()


def test_edge_corroborated_skips_external_lookup_and_supports_downstream():
    """已佐证节点不受前提影响、不查询其专属外部前提，并向非佐证下游传递自身 own。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        tms = e.enable_tms()
        _bulk_insert_nodes(e, [("ext_bad", 0.05), ("p", 0.1), ("corr", 0.8), ("down", 0.95)])
        _bulk_insert_justifications(e, [
            ("j1", "corr", ["p", "ext_bad"]),
            ("j2", "down", ["corr"]),
        ])
        _install_own(tms, {
            "p": {"confidence": 0.1, "status": "refuted"},
            "corr": {"confidence": 0.8, "status": "corroborated"},
            "down": {"confidence": 0.95, "status": "single_source"},
        })
        ext_calls = []
        orig_ext = tms._external_belief
        tms._external_belief = lambda p: (ext_calls.append(p), orig_ext(p))[1]

        got = tms.propagate(["p"])
        assert got == {"p": 0.1, "corr": 0.8, "down": 0.8}
        assert ext_calls == [], f"corroborated node should not query external premises: {ext_calls}"
    finally:
        e.close()


def test_edge_deleted_external_and_deleted_corroborated_nodes():
    """已删除外部前提与已删除（但账本残留 corroborated）中间节点均按 0.0 传播。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        tms = e.enable_tms()
        _bulk_insert_nodes(e, [("ext_gone", 0.9), ("mid_gone", 0.9), ("c1", 0.9), ("c2", 0.9)])
        _bulk_insert_justifications(e, [
            ("j1", "c1", ["ext_gone"]),
            ("j2", "c2", ["mid_gone"]),
        ])
        _install_own(tms, {
            "mid_gone": {"confidence": 0.85, "status": "corroborated"},
            "c1": {"confidence": 0.9, "status": "single_source"},
            "c2": {"confidence": 0.9, "status": "single_source"},
        })
        assert e.store.delete_node("ext_gone")
        assert e.store.delete_node("mid_gone")

        assert tms.propagate(["c1"]) == {"c1": 0.0}
        assert float(e.store.get_node("c1").confidence) == 0.0
        assert tms.propagate(["mid_gone"]) == {"mid_gone": 0.0, "c2": 0.0}
        assert float(e.store.get_node("c2").confidence) == 0.0
    finally:
        e.close()


def test_edge_round_6_writeback_precision():
    """返回值保持原始浮点，写回 SQLite nodes.confidence 统一使用 round(b, 6)。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        tms = e.enable_tms()
        raw_p = 0.123456789
        raw_c = 0.876543219
        _bulk_insert_nodes(e, [("p", 0.9), ("c", 0.9)])
        _bulk_insert_justifications(e, [("j1", "c", ["p"])])
        _install_own(tms, {
            "p": {"confidence": raw_p, "status": "single_source"},
            "c": {"confidence": raw_c, "status": "single_source"},
        })
        got = tms.propagate(["p"])
        assert got["p"] == raw_p and got["c"] == raw_p
        assert float(e.store.get_node("p").confidence) == round(raw_p, 6)
        assert float(e.store.get_node("c").confidence) == round(raw_p, 6)
    finally:
        e.close()


def test_edge_shared_layer_write_gate_769_and_rollback_atomicity(tmp_path):
    """#769 共享层写权限闸与触发器中断回滚原子性：被拒/失败时零落库且清空 _just_cache。"""
    db = str(tmp_path / "gate_rollback.db")
    p = SpacetimeMemoryEngine(db)
    try:
        cs = ConditionSpace("gate", "gate", (0.0, 1e12), "ok")
        p.store.add_node(STNode(
            id="anchor_node", content="anchor", modality="text",
            spatial_coordinates={}, temporal_coordinate=1.0,
            condition_space=cs, importance=1.0, confidence=1.0,
            layer=MemoryLayer.ANCHOR, tags=[],
        ))
        p.store.add_node(STNode(
            id="local_prem", content="local", modality="text",
            spatial_coordinates={}, temporal_coordinate=1.0,
            condition_space=cs, importance=0.5, confidence=0.8,
            layer=MemoryLayer.KNOWLEDGE, tags=[],
        ))
        p.store.add_node(STNode(
            id="local_mid", content="mid", modality="text",
            spatial_coordinates={}, temporal_coordinate=1.0,
            condition_space=cs, importance=0.5, confidence=0.8,
            layer=MemoryLayer.KNOWLEDGE, tags=[],
        ))
        ptms = p.enable_tms()
        ptms.add_justification("local_mid", ["local_prem"])
        ptms.add_justification("anchor_node", ["local_mid"])
    finally:
        p.close()

    # 1) SUB 沿多跳闭包触达共享层锚点 ⇒ 整体拒、零落库、缓存已清
    sub = SpacetimeMemoryEngine(db, role=Role.SUB)
    try:
        stms = sub.enable_tms()
        _install_own(stms, {
            "local_prem": {"confidence": 0.1, "status": "refuted"},
            "local_mid": {"confidence": 0.8, "status": "single_source"},
            "anchor_node": {"confidence": 0.9, "status": "single_source"},
        })
        before = {nid: float(sub.store.get_node(nid).confidence)
                  for nid in ("local_prem", "local_mid", "anchor_node")}
        with pytest.raises(PermissionError):
            stms.propagate(["local_prem"])
        assert stms._just_cache == {}
        after = {nid: float(sub.store.get_node(nid).confidence)
                 for nid in ("local_prem", "local_mid", "anchor_node")}
        assert after == before, f"#769 partial write occurred: {before} -> {after}"
    finally:
        sub.close()

    # 2) PRIMARY 写回中途触发器中止 ⇒ 事务完整回滚、缓存已清
    p2 = SpacetimeMemoryEngine(db, role=Role.PRIMARY)
    try:
        ptms2 = p2.enable_tms()
        _install_own(ptms2, {
            "local_prem": {"confidence": 0.1, "status": "refuted"},
            "local_mid": {"confidence": 0.8, "status": "single_source"},
            "anchor_node": {"confidence": 0.9, "status": "single_source"},
        })
        before = {nid: float(p2.store.get_node(nid).confidence)
                  for nid in ("local_prem", "local_mid", "anchor_node")}
        p2.store.conn.execute(
            "CREATE TRIGGER abort_anchor BEFORE UPDATE OF confidence ON nodes "
            "WHEN OLD.id='anchor_node' BEGIN SELECT RAISE(ABORT, 'mid-write fail'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="mid-write fail"):
            ptms2.propagate(["local_prem"])
        assert not p2.store.conn.in_transaction
        assert ptms2._just_cache == {}
        after = {nid: float(p2.store.get_node(nid).confidence)
                 for nid in ("local_prem", "local_mid", "anchor_node")}
        assert after == before, f"rollback not atomic: {before} -> {after}"
    finally:
        p2.close()


# ============================================================ 3. 鉴别守卫
def test_guard_discriminator_lfp_fails_gfp_cycle_invariant(monkeypatch):
    """鉴别守卫：自支撑环必须保持 GFP（0.85）；若退化为 LFP（0.0），守卫断言必红。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        tms = e.enable_tms()
        _bulk_insert_nodes(e, [("a", 0.85), ("b", 0.85)])
        _bulk_insert_justifications(e, [("j1", "a", ["b"]), ("j2", "b", ["a"])])
        _install_own(tms, {
            "a": {"confidence": 0.85, "status": "single_source"},
            "b": {"confidence": 0.85, "status": "single_source"},
        })

        def check_gfp_invariant():
            got = tms.propagate(["a"])
            assert got == {"a": 0.85, "b": 0.85}, f"GFP cycle invariant broken: {got}"

        # 正向通过
        check_gfp_invariant()

        # 变异为 LFP ⇒ 守卫必红
        def buggy_lfp(self, nodes, own):
            return _reference_lfp(nodes, own, self._just_cache, self._external_belief)

        monkeypatch.setattr(TMS.TruthMaintenance, "_solve_beliefs", buggy_lfp)
        with pytest.raises(AssertionError, match="GFP cycle invariant broken"):
            check_gfp_invariant()
    finally:
        e.close()


def test_guard_discriminator_corroborated_and_cycle_retraction(monkeypatch):
    """鉴别守卫：破坏已佐证豁免或破坏弱链环撤回，均必触发断言失败。"""
    e = SpacetimeMemoryEngine(":memory:")
    try:
        tms = e.enable_tms()
        _bulk_insert_nodes(e, [("a", 0.9), ("b", 0.9), ("corr", 0.8)])
        _bulk_insert_justifications(e, [
            ("j1", "a", ["b"]),
            ("j2", "b", ["a"]),
            ("j3", "corr", ["a"]),
        ])
        _install_own(tms, {
            "a": {"confidence": 0.9, "status": "single_source"},
            "b": {"confidence": 0.2, "status": "single_source"},
            "corr": {"confidence": 0.8, "status": "corroborated"},
        })

        def check_invariants():
            got = tms.propagate(["a"])
            assert got["a"] == 0.2 and got["b"] == 0.2, f"weak-link cycle broken: {got}"
            assert got["corr"] == 0.8, f"corroborated override broken: {got}"

        check_invariants()

        # 变异 1：忽略 corroborated 豁免
        orig_solve = TMS.TruthMaintenance._solve_beliefs

        def ignore_corroborated(self, nodes, own):
            stripped = {n: {**o, "status": "single_source"} for n, o in own.items()}
            return orig_solve(self, nodes, stripped)

        monkeypatch.setattr(TMS.TruthMaintenance, "_solve_beliefs", ignore_corroborated)
        with pytest.raises(AssertionError, match="corroborated override broken"):
            check_invariants()

        # 变异 2：不传播环上撤回（仅返回 own）
        monkeypatch.setattr(
            TMS.TruthMaintenance,
            "_solve_beliefs",
            lambda self, nodes, own: {n: own[n]["confidence"] for n in nodes},
        )
        with pytest.raises(AssertionError, match="weak-link cycle broken"):
            check_invariants()
    finally:
        e.close()


def test_guard_discriminator_single_pass_vs_gauss_seidel_ext_queries(monkeypatch):
    """鉴别守卫：Min-Heap GFP 对共享外部前提仅查 1 次；回退至 Gauss-Seidel 时查询暴涨必红。"""
    n = 60
    e = SpacetimeMemoryEngine(":memory:")
    try:
        tms = e.enable_tms()
        ids = [f"n{i:03d}" for i in range(n)]
        _bulk_insert_nodes(e, [(nid, 0.8) for nid in ids] + [("ext0", 0.8)])
        # 逆序登记，使 BFS 顺序为 [n0, n_{n-1}, ..., n1]，诱发 Gauss-Seidel O(N) 轮扫描
        entries = []
        for i in range(n - 1, 0, -1):
            prems = [ids[0], "ext0"] + ([ids[i - 1]] if i > 1 else [])
            entries.append((f"j{i}", ids[i], prems))
        _bulk_insert_justifications(e, entries)
        _install_own(tms, {
            nid: {"confidence": 0.1 if nid == ids[1] else 0.8, "status": "single_source"}
            for nid in ids
        })

        orig_ext = tms._external_belief

        def run_and_count():
            calls = [0]

            def counted(p):
                calls[0] += 1
                return orig_ext(p)

            tms._external_belief = counted
            got = tms.propagate([ids[0]])
            assert got[ids[-1]] == 0.1
            return calls[0]

        # Min-Heap GFP：严格 1 次外部查询
        heap_calls = run_and_count()
        assert heap_calls == 1, f"expected 1 external lookup, got {heap_calls}"

        # 回退至参考 Gauss-Seidel：外部查询次数暴涨至 O(N^2) ⇒ 守卫必红
        def reverted_gs(self, nodes, own):
            return _reference_gauss_seidel(nodes, own, self._just_cache, self._external_belief)

        monkeypatch.setattr(TMS.TruthMaintenance, "_solve_beliefs", reverted_gs)
        gs_calls = run_and_count()
        assert gs_calls > n * 10, f"reverted Gauss-Seidel did not exhibit quadratic lookups: {gs_calls}"
    finally:
        e.close()


# ============================================================ 4. 扩展性健全检查
def test_scaling_heap_gfp_large_reverse_closure():
    """大闭包（N=1000 逆序依赖链 + N=2000 纯求解器）应在毫秒级完成并精确收敛。"""
    n = 1000
    e = SpacetimeMemoryEngine(":memory:")
    try:
        tms = e.enable_tms()
        ids = [f"n{i:04d}" for i in range(n)]
        _bulk_insert_nodes(e, [(nid, 0.8) for nid in ids] + [("ext0", 0.8)])
        entries = []
        for i in range(n - 1, 0, -1):
            prems = [ids[0], "ext0"] + ([ids[i - 1]] if i > 1 else [])
            entries.append((f"j{i}", ids[i], prems))
        _bulk_insert_justifications(e, entries)
        _install_own(tms, {
            nid: {"confidence": 0.1 if nid == ids[1] else 0.8, "status": "single_source"}
            for nid in ids
        })

        t0 = time.perf_counter()
        got = tms.propagate([ids[0]])
        elapsed = time.perf_counter() - t0

        assert len(got) == n
        assert got[ids[0]] == 0.8
        assert got[ids[1]] == 0.1
        assert got[ids[-1]] == 0.1
        # Min-Heap GFP 实测 ~50-80ms（原 Gauss-Seidel 约 >30s）；2.0s 阈值留足余量防抖动
        assert elapsed < 2.0, f"N={n} propagate took {elapsed:.3f}s (expected < 2.0s)"

        # 纯求解器 N=2000 压力验证（无 SQLite 开销，实测数毫秒）
        n2 = 2000
        ids2 = [f"m{i:04d}" for i in range(n2)]
        nodes2 = [ids2[0]] + list(reversed(ids2[1:]))
        own2 = {
            nid: {"confidence": 0.1 if nid == ids2[1] else 0.8, "status": "single_source"}
            for nid in ids2
        }
        tms._just_cache = {
            ids2[i]: [[ids2[0], ids2[i - 1]]] if i > 1 else [[ids2[0]]]
            for i in range(1, n2)
        }
        tms._just_cache[ids2[0]] = []
        try:
            t1 = time.perf_counter()
            got2 = tms._solve_beliefs(nodes2, own2)
            solve_elapsed = time.perf_counter() - t1
        finally:
            tms._just_cache = {}
        assert got2[ids2[0]] == 0.8
        assert got2[ids2[-1]] == 0.1
        assert solve_elapsed < 0.5, f"N={n2} _solve_beliefs took {solve_elapsed:.3f}s"
    finally:
        e.close()
