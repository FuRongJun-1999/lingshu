# -*- coding: utf-8 -*-
"""ActivationEngine frontier optimization parity, caching, and discriminator guards.

Verifies:
1. 100% exact numerical & behavioral parity against an independent full-graph
   max-plus reference oracle (including #53 cond_match, #229/#521 high-conf
   unidirectional & low-conf bidirectional opposite inhibition, #280 boundedness,
   #314 first_hop/sources, #505 query, self_condition, multi-hop, audit steps,
   and scipy vs edge-list backend parity).
2. Sound graph cache invalidation across borrowed :memory:/file connections,
   caller transactions + ROLLBACK, cross-connection writes, DELETE and WAL
   journal modes (including silent WAL auto-checkpoint updates), triggers on
   activation_nodes, and module monkeypatches.
3. Discriminator guards verifying that active-support frontier propagation,
   descending-edge early stopping, B_inh inhibitory protection, and graph
   caching are all genuinely exercised (and fail immediately on regression).

Run:
    python -X utf8 -m pytest tests/test_activation_frontier.py -q
"""
from __future__ import annotations

import json
import random
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import lingshu.core.activation as activation
    from lingshu.core.activation import (
        ActivationEngine,
        COND_MATCH_NEUTRAL,
        DEFAULT_DECAY,
        EDGE_BASE_DECAY,
        cond_match,
    )
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    pytest.skip("ActivationEngine requires NumPy", allow_module_level=True)

try:
    from scipy.sparse import csr_matrix as _scipy_csr_matrix
except ImportError:
    _scipy_csr_matrix = None

from lingshu.core.core import ConditionSpace, EdgeType, LayeredStore, STEdge, STNode  # noqa: E402


# --------------------------------------------------------------------------- #
# Independent full-graph max-plus reference oracle (zero caching, zero pruning)
# --------------------------------------------------------------------------- #
def _reference_activate(con, query, workset="default", top_k=12, hops=2,
                        act_floor=0.15, self_condition=0.0,
                        prior_workset=None, carry_map=None):
    """Exact full-graph max-plus reference matching the #53/#229/#280/#314/#521 spec."""
    cur = con.cursor()
    try:
        node_rows = cur.execute("SELECT id, condition_space, content, tags FROM nodes").fetchall()
    except sqlite3.OperationalError:
        node_rows = []
    node_ids = [r[0] for r in node_rows]
    idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    cur_cs = {}
    for nid, cs_json, _content, _tags in node_rows:
        try:
            cur_cs[nid] = json.loads(cs_json) if cs_json else {}
        except (json.JSONDecodeError, TypeError):
            cur_cs[nid] = {}

    try:
        ecols = {r[1] for r in cur.execute("PRAGMA table_info(edges)")}
    except sqlite3.OperationalError:
        ecols = set()
    conf_expr = "confidence" if "confidence" in ecols else "0.3 AS confidence"
    try:
        edges = cur.execute(
            "SELECT source_id, target_id, relation_type, " + conf_expr + " FROM edges"
        ).fetchall()
    except sqlite3.OperationalError:
        edges = []

    exc_edges = []
    inh_edges = []
    unv_edges = []
    for src, dst, rtype, edge_conf in edges:
        si, di = idx.get(src), idx.get(dst)
        if si is None or di is None:
            continue
        d = EDGE_BASE_DECAY.get(rtype, DEFAULT_DECAY)
        try:
            d = d * cond_match(cur_cs.get(src), cur_cs.get(dst))
        except (TypeError, ValueError):
            d = d * COND_MATCH_NEUTRAL
        if d < 0:
            try:
                _c = float(edge_conf) if edge_conf is not None else 0.5
            except (TypeError, ValueError):
                _c = 0.5
            if _c != _c:
                _c = 0.5
            _c = min(1.0, max(0.0, _c))
            if _c < 0.5:
                m = float(np.float32(-d * _c))
                inh_edges.append((si, di, np.float32(m)))
                unv_edges.append((si, di, np.float32(m)))
            else:
                inh_edges.append((si, di, np.float32(-d)))
        else:
            exc_edges.append((si, di, np.float32(d)))

    # Seed by text
    if not (query or "").strip():
        seeds = []
    else:
        toks = {query[i:i + 2] for i in range(len(query) - 1)} | {query}
        scores = []
        for nid, _cs, content, tags_json in node_rows:
            hay = content or ""
            try:
                hay += " " + " ".join(json.loads(tags_json)) if tags_json else ""
            except (json.JSONDecodeError, TypeError):
                pass
            s = sum(1 for t in toks if t in hay) / max(1, len(toks))
            if s > 0:
                scores.append((nid, s))
        scores.sort(key=lambda x: -x[1])
        seeds = scores[:top_k]

    act = np.zeros(n, dtype=np.float32)
    seed_ids = []
    for nid, s in seeds:
        i = idx.get(nid)
        if i is not None:
            act[i] = max(act[i], s)
            seed_ids.append(nid)

    carried_ids = []
    if self_condition > 0.0:
        if carry_map is None:
            prior_name = prior_workset or workset
            carry_map = {
                r[0]: float(r[1])
                for r in cur.execute(
                    "SELECT node_id, activation FROM activation_nodes WHERE workset=?",
                    (prior_name,),
                ).fetchall()
            }
        for nid, a in carry_map.items():
            i = idx.get(nid)
            if i is None:
                continue
            carried_ids.append(nid)
            act[i] = max(act[i], float(a) * self_condition)

    first_hop = {nid: 0 for nid in seed_ids}
    for nid in carried_ids:
        first_hop[nid] = 0

    prop_steps = []
    for hop in range(1, hops + 1):
        prop = np.zeros(n, dtype=np.float32)
        for si, di, w in exc_edges:
            if act[si] > 0:
                v = np.float32(act[si] * w)
                if v > prop[di]:
                    prop[di] = v
        act = np.maximum(act, prop)

        supp = np.zeros(n, dtype=np.float32)
        for si, di, m in inh_edges:
            if act[si] > 0:
                v = np.float32(act[si] * m)
                if v > supp[di]:
                    supp[di] = v
        for si, di, m in unv_edges:
            if act[di] > 0:
                v = np.float32(act[di] * m)
                if v > supp[si]:
                    supp[si] = v
        if supp.any():
            act = np.maximum(act - supp, 0.0)

        above = np.nonzero(act >= act_floor)[0]
        for i in above:
            nid = node_ids[i]
            if nid not in first_hop:
                first_hop[nid] = hop
        prop_steps.append({
            "step": hop,
            "phase": "propagate",
            "activated_count": int(above.size),
            "max_act": round(float(act.max()), 4) if n else 0.0,
        })

    keep = np.nonzero(act >= act_floor)[0]
    members = [(node_ids[i], round(float(act[i]), 4)) for i in keep]
    members.sort(key=lambda x: -x[1])
    sources = {nid: "seed" for nid in seed_ids}
    for nid in carried_ids:
        sources[nid] = "self_condition"
    member_meta = {
        nid: (a, sources.get(nid, "propagate"), first_hop.get(nid, 0))
        for nid, a in members
    }
    return {
        "seeds": len(seed_ids),
        "carried": len(carried_ids),
        "size": len(members),
        "top": members[:10],
        "members": members,
        "member_meta": member_meta,
        "prop_steps": prop_steps,
    }


def _create_raw_graph_db(path_or_mem, nodes, edges, journal_mode=None):
    con = sqlite3.connect(path_or_mem)
    if journal_mode is not None:
        con.execute("PRAGMA journal_mode=%s" % journal_mode)
    con.execute(
        "CREATE TABLE nodes (id TEXT PRIMARY KEY, content TEXT, tags TEXT, condition_space TEXT)"
    )
    con.execute(
        "CREATE TABLE edges (id TEXT PRIMARY KEY, source_id TEXT, target_id TEXT, "
        "relation_type TEXT, condition_space TEXT, confidence REAL)"
    )
    con.executemany("INSERT INTO nodes VALUES (?,?,?,?)", nodes)
    con.executemany("INSERT INTO edges VALUES (?,?,?,?,?,?)", edges)
    con.commit()
    return con


def _build_random_graph(rng, n_nodes, edge_prob, inh_ratio=0.18):
    cs_pool = [
        None,
        json.dumps({"observation_position": "p0", "observation_tool": "t0",
                    "time_window": [100.0, 200.0], "existence_constraint": "c0"}),
        json.dumps({"observation_position": "p0", "observation_tool": "t1",
                    "time_window": [150.0, 250.0], "existence_constraint": "c0"}),
        json.dumps({"observation_position": "p1", "observation_tool": "t1",
                    "time_window": [500000.0, 500100.0], "existence_constraint": "c1"}),
        json.dumps({"observation_position": "p0", "observation_tool": "t0",
                    "time_window": [None, None], "existence_constraint": "c0"}),
    ]
    vocab = ["量子", "超导", "认知", "图谱", "推理", "海马", "记忆", "同步", "衰减", "冲突"]
    nodes = []
    for i in range(n_nodes):
        words = [rng.choice(vocab) for _ in range(rng.randint(1, 3))]
        tags = json.dumps([rng.choice(vocab)]) if rng.random() < 0.3 else "[]"
        nodes.append(("n%d" % i, " ".join(words) + " #%d" % i, tags, rng.choice(cs_pool)))

    exc_types = ["causal", "similar", "counterpart", "hierarchical", "sequential", "spatial"]
    edges = []
    eid = 0
    seen_exc = set()
    seen_inh = set()
    for u in range(n_nodes):
        for v in range(n_nodes):
            if u == v:
                continue
            if rng.random() < edge_prob:
                if rng.random() < inh_ratio:
                    if (u, v) in seen_inh:
                        continue
                    seen_inh.add((u, v))
                    conf = rng.choice([0.0, 0.2, 0.3, 0.49, 0.5, 0.8, 1.0, None])
                    edges.append(("e%d" % eid, "n%d" % u, "n%d" % v, "opposite", None, conf))
                else:
                    if (u, v) in seen_exc:
                        continue
                    seen_exc.add((u, v))
                    edges.append(("e%d" % eid, "n%d" % u, "n%d" % v, rng.choice(exc_types), None, 0.5))
                eid += 1
    return nodes, edges, vocab


def _read_last_audit(audit_path):
    lines = Path(audit_path).read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1])


@pytest.fixture(params=("scipy", "edge-list"))
def backend(request, monkeypatch):
    if request.param == "edge-list":
        monkeypatch.setattr(activation, "csr_matrix", None)
    elif _scipy_csr_matrix is None:
        pytest.skip("SciPy is optional; install it to exercise the sparse backend")
    return request.param


# --------------------------------------------------------------------------- #
# 1. Randomized parity against exact full-graph reference oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", list(range(12)))
def test_randomized_multihop_and_self_condition_exact_parity(backend, tmp_path, seed):
    """Exact bit-for-bit parity across random mixed graphs, multi-hop, and self_condition."""
    rng = random.Random(202600 + seed)
    n_nodes = rng.randint(18, 65)
    edge_prob = rng.uniform(0.04, 0.14)
    inh_ratio = rng.choice([0.0, 0.15, 0.35])
    nodes, edges, vocab = _build_random_graph(rng, n_nodes, edge_prob, inh_ratio=inh_ratio)

    con = _create_raw_graph_db(":memory:", nodes, edges)
    audit = tmp_path / ("rand_%s_%d.jsonl" % (backend, seed))
    try:
        eng = ActivationEngine(conn=con, audit_path=str(audit))
        for round_idx in range(3):
            q = " ".join(rng.sample(vocab, rng.randint(1, 3)))
            hops = rng.choice([0, 1, 2, 3, 4])
            act_floor = rng.choice([0.0, 0.05, 0.15, 0.30, 0.75])
            sc = 0.0 if round_idx == 0 else rng.choice([0.0, 0.35, 0.5, 1.0])

            # Capture exact prior workset BEFORE activate overwrites it
            prior_carry = eng.carry_vector("ws")
            ref = _reference_activate(
                con, q, workset="ws", top_k=10, hops=hops,
                act_floor=act_floor, self_condition=sc, carry_map=prior_carry,
            )
            got = eng.activate(
                q, workset="ws", top_k=10, hops=hops,
                act_floor=act_floor, self_condition=sc,
            )
            last_audit = _read_last_audit(audit)
            got_prop_steps = [s for s in last_audit["steps"] if s["phase"] == "propagate"]

            assert got["seeds"] == ref["seeds"], (seed, round_idx)
            assert got["carried"] == ref["carried"], (seed, round_idx)
            assert got["size"] == ref["size"], (seed, round_idx)
            assert got["top"] == ref["top"], (seed, round_idx)
            assert got_prop_steps == ref["prop_steps"], (seed, round_idx, got_prop_steps, ref["prop_steps"])

            snap = eng.export_workset("ws")
            got_meta = {
                m["node_id"]: (m["activation"], m["source"], m["hop"])
                for m in snap["members"]
            }
            assert got_meta == ref["member_meta"], (seed, round_idx)
            if round_idx > 0:
                assert eng._graph_load_count == 1
                assert eng._last_frontier_stats["cache_hit"] is True
    finally:
        con.close()


def test_scipy_and_edge_list_backends_match_bit_for_bit(monkeypatch, tmp_path):
    """SciPy sparse backend and edge-list fallback agree 100% including low-conf opposite edges."""
    if _scipy_csr_matrix is None:
        pytest.skip("SciPy is optional")
    rng = random.Random(99911)
    nodes, edges, vocab = _build_random_graph(rng, 55, 0.10, inh_ratio=0.30)

    outputs = {}
    for mode in ("scipy", "edge-list"):
        con = _create_raw_graph_db(":memory:", nodes, edges)
        audit = tmp_path / ("parity_%s.jsonl" % mode)
        try:
            if mode == "edge-list":
                monkeypatch.setattr(activation, "csr_matrix", None)
            else:
                monkeypatch.setattr(activation, "csr_matrix", _scipy_csr_matrix)
            eng = ActivationEngine(conn=con, audit_path=str(audit))
            r1 = eng.activate("量子 超导 冲突", workset="w", hops=3, act_floor=0.12)
            r2 = eng.activate("推理 记忆", workset="w", hops=3, act_floor=0.12, self_condition=0.5)
            snap = eng.export_workset("w")
            steps = [json.loads(ln)["steps"] for ln in audit.read_text(encoding="utf-8").splitlines()]
            outputs[mode] = (
                r1["top"], r2["top"], r1["size"], r2["size"],
                sorted((m["node_id"], m["activation"], m["source"], m["hop"]) for m in snap["members"]),
                steps,
            )
        finally:
            con.close()
    assert outputs["scipy"] == outputs["edge-list"]


# --------------------------------------------------------------------------- #
# 2. Explicit #521 reverse di -> si inhibition with SciPy installed
# --------------------------------------------------------------------------- #
def test_low_conf_opposite_reverse_inhibition_di_to_si_with_scipy(backend, tmp_path):
    """#521: Low-conf (<0.5) opposite edge A -> B MUST inhibit A when B is active (di -> si).

    Also catches the pre-existing CSC-transpose bug where main's _max_plus(A.T, act)
    ignored reverse inhibition on SciPy CSC matrices.
    """
    cs = json.dumps({"observation_position": "p", "observation_tool": "t",
                     "time_window": [10.0, 20.0], "existence_constraint": "c"})
    nodes = [
        ("A", "甲节点 锚点", "[]", cs),
        ("B", "乙节点 锚点 强化词", "[]", cs),
    ]
    # Stored direction is A -> B with confidence=0.3 (<0.5 => bidirectional!)
    edges = [
        ("e_opp", "A", "B", "opposite", None, 0.3),
    ]
    con = _create_raw_graph_db(":memory:", nodes, edges)
    try:
        eng = ActivationEngine(conn=con, audit_path=str(tmp_path / "rev521.jsonl"))
        # Query "乙节点 锚点 强化词" matches B at 1.0 and A only via "锚点" (< 1.0)
        ref = _reference_activate(con, "乙节点 锚点 强化词", workset="w", hops=1, act_floor=0.05)
        res = eng.activate("乙节点 锚点 强化词", workset="w", hops=1, act_floor=0.05)
        carried = eng.carry_vector("w")
        assert carried == {nid: val for nid, val in ref["members"]}
        # Specifically: B (1.0) must reverse-inhibit A by 1.0 * 0.3 = 0.3!
        seed_a = dict(eng._seed_by_text("乙节点 锚点 强化词"))["A"]
        assert seed_a > 0.0
        expected_a = round(max(seed_a - 0.3, 0.0), 4)
        if expected_a >= 0.05:
            assert carried["A"] == pytest.approx(expected_a, abs=1e-4)
        else:
            assert "A" not in carried
        assert res["size"] == ref["size"]
    finally:
        con.close()


def test_max_plus_handles_plain_scipy_csc_transpose_directly():
    """Direct unit check: ActivationEngine._max_plus(csr.T, act) computes true transpose."""
    if _scipy_csr_matrix is None:
        pytest.skip("SciPy is optional")
    adj = _scipy_csr_matrix((np.array([0.3], dtype=np.float32), ([0], [1])), shape=(2, 2))
    assert adj.T.format == "csc"
    # Edge 0 -> 1 transposed is 1 -> 0: act=[0, 1] must inhibit node 0 by 0.3!
    out = ActivationEngine._max_plus(adj.T, np.array([0.0, 1.0], dtype=np.float32))
    assert out.tolist() == pytest.approx([0.3, 0.0], abs=1e-6)
    out2 = ActivationEngine._max_plus(adj.T, np.array([1.0, 0.0], dtype=np.float32))
    assert out2.tolist() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_high_conf_opposite_stays_strictly_unidirectional(backend, tmp_path):
    """#229/#521 contract: confidence >= 0.5 opposite edge A -> B does NOT inhibit A from B."""
    cs = json.dumps({"observation_position": "p", "observation_tool": "t",
                     "time_window": [10.0, 20.0], "existence_constraint": "c"})
    nodes = [
        ("A", "共有词 甲", "[]", cs),
        ("B", "共有词 乙 满权", "[]", cs),
    ]
    edges = [("e_opp", "A", "B", "opposite", None, 0.5)]
    con = _create_raw_graph_db(":memory:", nodes, edges)
    try:
        eng = ActivationEngine(conn=con, audit_path=str(tmp_path / "uni.jsonl"))
        ref = _reference_activate(con, "共有词 乙 满权", workset="w", hops=1, act_floor=0.05)
        eng.activate("共有词 乙 满权", workset="w", hops=1, act_floor=0.05)
        seed_a = dict(eng._seed_by_text("共有词 乙 满权"))["A"]
        assert eng.carry_vector("w")["A"] == pytest.approx(round(seed_a, 4), abs=1e-4)
        assert eng.carry_vector("w") == {nid: val for nid, val in ref["members"]}
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# 3. Exact B_inh & max_act adversarial edge cases + discriminator guards
# --------------------------------------------------------------------------- #
def test_subthreshold_upstream_inhibitory_chain_protected_by_b_inh(backend, tmp_path, monkeypatch):
    """Adversarial B_inh test + discriminator guard!

    Graph:
      seed (1.0) --(w=0.125 < act_floor=0.20)--> u1 --(w=0.85)--> u2 -|(opp, 1.0)-> target
      seed (1.0) --(w=0.25  >= act_floor=0.20)--> target

    At hop 1: u1 gets 0.125 < 0.20; at hop 2: u2 gets 0.10625 < 0.20 and inhibits
    target from 0.25 down to 0.14375 < 0.20!
    - Real implementation protects u1, u2 via B_inh(hops - hop) => target excluded.
    - Discriminator check: disabling B_inh (_get_inh_exc_csr -> None) MUST fail
      by falsely keeping target in the workset!
    """
    cs_all = json.dumps({"observation_position": "p", "observation_tool": "t",
                         "time_window": [10.0, 20.0], "existence_constraint": "c"})
    # 1 of 4 dimensions match => cond_match = 0.25 => spatial(0.5)*0.25 = 0.125
    cs_quarter = json.dumps({"observation_position": "p", "observation_tool": "x",
                             "time_window": [999999.0, 999999.5], "existence_constraint": "y"})
    # 2 of 4 dimensions match => cond_match = 0.5 => spatial(0.5)*0.5 = 0.25
    cs_half = json.dumps({"observation_position": "p", "observation_tool": "t",
                          "time_window": [999999.0, 999999.5], "existence_constraint": "y"})
    nodes = [
        ("seed", "唯一触发种子", "[]", cs_all),
        ("u1", "上游节点甲", "[]", cs_quarter),
        ("u2", "抑制源节点乙", "[]", cs_quarter),
        ("target", "被抑制目标丙", "[]", cs_half),
    ]
    # Note: u2 -> target opposite uses cond_match(cs_quarter, cs_half) = 0.25? Let's make u2 match target 100%!
    # Give u1->u2 cond_match=0.5 (causal 0.85*0.5=0.425) or match cs_half:
    # Actually: seed(cs_all) -> u1(cs_quarter): spatial 0.5*0.25 = 0.125.
    # u1(cs_quarter) -> u2(cs_quarter): causal 0.85*1.0 = 0.85 => u2 = 0.10625.
    # target uses cs_quarter too? No, seed->target needs 0.25: use causal(0.85)*0.25 = 0.2125!
    nodes = [
        ("seed", "唯一触发种子", "[]", cs_all),
        ("u1", "上游节点甲", "[]", cs_quarter),
        ("u2", "抑制源节点乙", "[]", cs_quarter),
        ("target", "被抑制目标丙", "[]", cs_quarter),
    ]
    edges = [
        ("e_s_u1", "seed", "u1", "spatial", None, 0.5),   # 0.5 * 0.25 = 0.125 (< 0.20)
        ("e_u1_u2", "u1", "u2", "causal", None, 0.5),     # 0.85 * 1.0 = 0.85  => u2 = 0.10625
        ("e_u2_t", "u2", "target", "opposite", None, 1.0),  # -1.0 * 1.0 => supp = 0.10625
        ("e_s_t", "seed", "target", "causal", None, 0.5),  # 0.85 * 0.25 = 0.2125 (>= 0.20)
    ]
    con = _create_raw_graph_db(":memory:", nodes, edges)
    try:
        eng = ActivationEngine(conn=con, audit_path=str(tmp_path / "binh.jsonl"))
        ref = _reference_activate(con, "唯一触发种子", workset="w", hops=2, act_floor=0.20)
        eng.activate("唯一触发种子", workset="w", hops=2, act_floor=0.20)
        carried = eng.carry_vector("w")
        # target = 0.2125 - 0.10625 = 0.10625 < 0.20 => MUST NOT be in workset!
        assert "target" not in carried
        assert carried == {nid: val for nid, val in ref["members"]}

        # Discriminator guard: if B_inh protection is broken, target falsely survives!
        monkeypatch.setattr(eng, "_get_inh_exc_csr", lambda _rem: None)
        eng.activate("唯一触发种子", workset="w_broken", hops=2, act_floor=0.20)
        broken = eng.carry_vector("w_broken")
        assert "target" in broken, "Discriminator failed: broken B_inh did not change output"
    finally:
        con.close()


def test_subthreshold_upstream_reverse_low_conf_inhibition_protected(backend, tmp_path):
    """B_inh also protects nodes upstream of the target endpoint of a low-conf opposite edge."""
    cs_all = json.dumps({"observation_position": "p", "observation_tool": "t",
                         "time_window": [10.0, 20.0], "existence_constraint": "c"})
    cs_quarter = json.dumps({"observation_position": "p", "observation_tool": "x",
                             "time_window": [999999.0, 999999.5], "existence_constraint": "y"})
    nodes = [
        ("seed", "触发锚点", "[]", cs_all),
        ("u1", "上游中间", "[]", cs_quarter),
        ("u2", "低信反向源", "[]", cs_quarter),
        ("target", "目标被反向压制", "[]", cs_quarter),
    ]
    # Opposite edge stored as target -> u2 with conf=0.4 (<0.5 => u2 inhibits target!)
    # seed->target = 0.85*0.25 = 0.2125; seed->u1 = 0.5*0.25 = 0.125; u1->u2 = 0.85 => 0.10625
    # reverse supp on target = 0.10625 * 0.4 = 0.0425 => target = 0.17 < 0.20!
    edges = [
        ("e1", "seed", "u1", "spatial", None, 0.5),
        ("e2", "u1", "u2", "causal", None, 0.5),
        ("e_rev_opp", "target", "u2", "opposite", None, 0.4),
        ("e_s_t", "seed", "target", "causal", None, 0.5),
    ]
    con = _create_raw_graph_db(":memory:", nodes, edges)
    try:
        eng = ActivationEngine(conn=con, audit_path=str(tmp_path / "rev_binh.jsonl"))
        ref = _reference_activate(con, "触发锚点", workset="w", hops=2, act_floor=0.20)
        eng.activate("触发锚点", workset="w", hops=2, act_floor=0.20)
        assert "target" not in eng.carry_vector("w")
        assert eng.carry_vector("w") == {nid: val for nid, val in ref["members"]}
    finally:
        con.close()


def test_all_seeds_inhibited_preserves_exact_audit_max_act(backend, tmp_path):
    """When all active seeds can be inhibited (in I_dst), pruning turns off so audit max_act stays exact."""
    cs_all = json.dumps({"observation_position": "p", "observation_tool": "t",
                         "time_window": [10.0, 20.0], "existence_constraint": "c"})
    cs_quarter = json.dumps({"observation_position": "p", "observation_tool": "x",
                             "time_window": [999999.0, 999999.5], "existence_constraint": "y"})
    nodes = [
        ("seed", "主种子词", "[]", cs_all),
        ("quiet_nbr", "弱兴奋邻居", "[]", cs_quarter),
    ]
    # Self-opposite edge suppresses seed from 1.0 to 0.0 at hop 1,
    # while seed excites quiet_nbr to 0.5 * 0.25 = 0.125 (< act_floor=0.20).
    # After hop 1: act=[0.0, 0.125], so audit max_act MUST be 0.125 (not 0.0!).
    edges = [
        ("e_self_inh", "seed", "seed", "opposite", None, 1.0),
        ("e_exc", "seed", "quiet_nbr", "spatial", None, 0.5),
    ]
    con = _create_raw_graph_db(":memory:", nodes, edges)
    audit = tmp_path / "max_act.jsonl"
    try:
        eng = ActivationEngine(conn=con, audit_path=str(audit))
        ref = _reference_activate(con, "主种子词", workset="w", hops=2, act_floor=0.20)
        eng.activate("主种子词", workset="w", hops=2, act_floor=0.20)
        got_steps = [s for s in _read_last_audit(audit)["steps"] if s["phase"] == "propagate"]
        assert got_steps == ref["prop_steps"]
        assert got_steps[0]["max_act"] == pytest.approx(0.125, abs=1e-4)
    finally:
        con.close()


def test_pruning_and_active_support_discriminator_guards(backend, tmp_path):
    """Discriminator guard: early stopping & active-support MUST skip provably redundant edges."""
    cs_all = json.dumps({"observation_position": "p", "observation_tool": "t",
                         "time_window": [10.0, 20.0], "existence_constraint": "c"})
    cs_quarter = json.dumps({"observation_position": "p", "observation_tool": "x",
                             "time_window": [999999.0, 999999.5], "existence_constraint": "y"})
    nodes = [("seed", "枢纽种子节点", "[]", cs_all)]
    edges = []
    # 5 strong edges (causal * 1.0 = 0.85 >= 0.30) + 95 weak edges (spatial * 0.25 = 0.125 < 0.30)
    for i in range(5):
        nodes.append(("strong_%d" % i, "强邻居%d" % i, "[]", cs_all))
        edges.append(("es_%d" % i, "seed", "strong_%d" % i, "causal", None, 0.5))
    for i in range(95):
        nodes.append(("weak_%d" % i, "弱邻居%d" % i, "[]", cs_quarter))
        edges.append(("ew_%d" % i, "seed", "weak_%d" % i, "spatial", None, 0.5))
    # Plus 300 disconnected background nodes with 600 edges (must NEVER be touched!)
    for i in range(300):
        nodes.append(("bg_%d" % i, "背景孤立%d" % i, "[]", cs_all))
    for i in range(600):
        u = i % 300
        v = (u + 1 + (i // 300)) % 300
        edges.append(("ebg_%d" % i, "bg_%d" % u, "bg_%d" % v, "causal", None, 0.5))

    con = _create_raw_graph_db(":memory:", nodes, edges)
    try:
        eng = ActivationEngine(conn=con, audit_path=str(tmp_path / "disc.jsonl"))
        ref = _reference_activate(con, "枢纽种子节点", workset="w", hops=1, act_floor=0.30)
        eng.activate("枢纽种子节点", workset="w", hops=1, act_floor=0.30)
        assert eng.carry_vector("w") == {nid: val for nid, val in ref["members"]}
        stats = eng._last_frontier_stats
        # Only the 5 strong edges of seed should be visited; 95 weak edges pruned; 600 bg edges untouched!
        assert stats["edges_visited"] == 5, stats
        assert stats["edges_pruned"] == 95, stats
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# 4. Sound graph caching & invalidation guards (same-conn, cross-conn, WAL/DELETE, rollback)
# --------------------------------------------------------------------------- #
def test_borrowed_memory_cache_hits_and_same_conn_mutations_invalidate(tmp_path):
    """Borrowed :memory: connection caches unchanged graph and invalidates on every DML/DDL."""
    cs = json.dumps({"observation_position": "p", "observation_tool": "t",
                     "time_window": [10.0, 20.0], "existence_constraint": "c"})
    con = _create_raw_graph_db(
        ":memory:",
        [("a", "初始锚点甲", "[]", cs), ("b", "邻居节点乙", "[]", cs)],
        [("e1", "a", "b", "similar", None, 0.5)],
    )
    try:
        eng = ActivationEngine(conn=con, audit_path=str(tmp_path / "mem.jsonl"))
        eng.activate("初始锚点甲", workset="w", hops=1)
        assert eng._graph_load_count == 1
        assert eng._last_frontier_stats["cache_hit"] is False
        assert eng.carry_vector("w") == {"a": 1.0, "b": 0.75}

        # Unchanged graph => MUST be a cache hit (discriminator guard against reverting cache!)
        eng.activate("初始锚点甲", workset="w", hops=1)
        assert eng._graph_load_count == 1
        assert eng._last_frontier_stats["cache_hit"] is True

        # In-place UPDATE edges (same row count & rowid!) => MUST invalidate!
        con.execute("UPDATE edges SET relation_type='causal' WHERE id='e1'")
        con.commit()
        eng.activate("初始锚点甲", workset="w", hops=1)
        assert eng._graph_load_count == 2
        assert eng._last_frontier_stats["cache_hit"] is False
        assert eng.carry_vector("w") == {"a": 1.0, "b": 0.85}

        # In-place UPDATE nodes.content => direct _seed_by_text AND activate MUST observe!
        con.execute("UPDATE nodes SET content='独一无二丙' WHERE id='a'")
        con.commit()
        assert dict(eng._seed_by_text("独一无二丙")) == {"a": 1.0}
        eng.activate("独一无二丙", workset="w", hops=1)
        assert eng._graph_load_count == 3
        assert eng.carry_vector("w") == {"a": 1.0, "b": 0.85}

        # DELETE FROM edges without WHERE clause (SQLite truncate optimization) => MUST invalidate!
        con.execute("DELETE FROM edges")
        con.commit()
        eng.activate("独一无二丙", workset="w", hops=1)
        assert eng._graph_load_count == 4
        assert eng.carry_vector("w") == {"a": 1.0}
    finally:
        con.close()


def test_borrowed_conn_transaction_rollback_never_serves_stale_graph(tmp_path):
    """Caller BEGIN IMMEDIATE -> mutate -> activate() -> ROLLBACK -> activate() stays 100% exact."""
    cs = json.dumps({"observation_position": "p", "observation_tool": "t",
                     "time_window": [10.0, 20.0], "existence_constraint": "c"})
    con = _create_raw_graph_db(
        ":memory:",
        [("a", "锚点节点", "[]", cs), ("b", "目标节点", "[]", cs)],
        [("e1", "a", "b", "similar", None, 0.5)],
    )
    try:
        eng = ActivationEngine(conn=con, audit_path=str(tmp_path / "rb.jsonl"))
        eng.activate("锚点节点", workset="w", hops=1)
        assert eng.carry_vector("w") == {"a": 1.0, "b": 0.75}

        # Delete b inside caller transaction, activate inside transaction, then ROLLBACK
        con.execute("BEGIN IMMEDIATE")
        con.execute("DELETE FROM nodes WHERE id='b'")
        eng.activate("锚点节点", workset="w", hops=1)
        assert eng.carry_vector("w") == {"a": 1.0}
        con.rollback()

        # After ROLLBACK, b is back! Cache MUST NOT serve the rolled-back graph!
        eng.activate("锚点节点", workset="w", hops=1)
        assert eng.carry_vector("w") == {"a": 1.0, "b": 0.75}
    finally:
        con.close()


@pytest.mark.parametrize("journal_mode", ["DELETE", "WAL"])
def test_cross_connection_mutations_invalidate_borrowed_and_file_engines(tmp_path, journal_mode):
    """Cross-connection DML/DDL invalidates both borrowed-conn and file-backed engines."""
    db_path = str(tmp_path / ("cross_%s.db" % journal_mode))
    cs = json.dumps({"observation_position": "p", "observation_tool": "t",
                     "time_window": [10.0, 20.0], "existence_constraint": "c"})
    owner = _create_raw_graph_db(
        db_path,
        [("a", "跨连接锚点", "[]", cs), ("b", "邻居节点", "[]", cs)],
        [("e1", "a", "b", "similar", None, 0.5)],
        journal_mode=journal_mode,
    )
    try:
        eng_borrowed = ActivationEngine(conn=owner, audit_path=str(tmp_path / "b.jsonl"))
        eng_file = ActivationEngine(db_path=db_path, audit_path=str(tmp_path / "f.jsonl"))

        for eng in (eng_borrowed, eng_file):
            eng.activate("跨连接锚点", workset="w", hops=1)
            assert eng._graph_load_count == 1
            assert eng.carry_vector("w") == {"a": 1.0, "b": 0.75}
            # Warm cache hit
            eng.activate("跨连接锚点", workset="w", hops=1)
            assert eng._graph_load_count == 1
            assert eng._last_frontier_stats["cache_hit"] is True

        # Separate external connection mutates edge relation_type in-place!
        ext = sqlite3.connect(db_path)
        try:
            ext.execute("UPDATE edges SET relation_type='causal' WHERE id='e1'")
            ext.commit()
        finally:
            ext.close()

        for eng in (eng_borrowed, eng_file):
            eng.activate("跨连接锚点", workset="w", hops=1)
            assert eng._graph_load_count == 2, journal_mode
            assert eng._last_frontier_stats["cache_hit"] is False
            assert eng.carry_vector("w") == {"a": 1.0, "b": 0.85}
    finally:
        owner.close()


def test_standalone_wal_silent_auto_checkpoint_update_invalidates_cache(tmp_path):
    """Adversarial WAL test: external open -> in-place UPDATE -> close (auto-checkpoint)
    keeps the 100-byte main DB header & file size bit-for-bit identical, yet cache MUST invalidate!
    """
    db_path = str(tmp_path / "wal_checkpoint.db")
    cs = json.dumps({"observation_position": "p", "observation_tool": "t",
                     "time_window": [10.0, 20.0], "existence_constraint": "c"})
    c = _create_raw_graph_db(
        db_path,
        [("a", "静默更新锚点", "[]", cs), ("b", "下游目标", "[]", cs)],
        [("e1", "a", "b", "similar", None, 0.5)],
        journal_mode="WAL",
    )
    c.close()

    eng = ActivationEngine(db_path=db_path, audit_path=str(tmp_path / "wc.jsonl"))
    eng.activate("静默更新锚点", workset="w", hops=1)
    assert eng.carry_vector("w") == {"a": 1.0, "b": 0.75}
    # Standalone warm query still caches via exact table content digest
    eng.activate("静默更新锚点", workset="w", hops=1)
    assert eng._graph_load_count == 1
    assert eng._last_frontier_stats["cache_hit"] is True

    # External connection opens, performs same-length in-place UPDATE, commits, and closes
    # (auto-checkpointing WAL and deleting -wal/-shm with unchanged main DB header!)
    ext = sqlite3.connect(db_path)
    ext.execute("UPDATE edges SET relation_type='spatial' WHERE id='e1'")
    ext.commit()
    ext.close()

    eng.activate("静默更新锚点", workset="w", hops=1)
    assert eng._graph_load_count == 2
    assert eng._last_frontier_stats["cache_hit"] is False
    assert eng.carry_vector("w") == {"a": 1.0, "b": 0.5}


def test_trigger_on_activation_nodes_prevents_stale_self_write_tracking(tmp_path):
    """If an AFTER INSERT trigger on activation_nodes mutates edges, cache MUST NOT go stale."""
    cs = json.dumps({"observation_position": "p", "observation_tool": "t",
                     "time_window": [10.0, 20.0], "existence_constraint": "c"})
    con = _create_raw_graph_db(
        ":memory:",
        [("a", "触发器锚点", "[]", cs), ("b", "目标节点", "[]", cs)],
        [("e1", "a", "b", "similar", None, 0.5)],
    )
    try:
        eng = ActivationEngine(conn=con, audit_path=str(tmp_path / "trig.jsonl"))
        con.execute(
            "CREATE TRIGGER mutate_edge AFTER INSERT ON activation_nodes "
            "BEGIN UPDATE edges SET relation_type='causal' WHERE id='e1'; END"
        )
        con.commit()
        eng.activate("触发器锚点", workset="w", hops=1)
        assert eng.carry_vector("w") == {"a": 1.0, "b": 0.75}
        # Trigger mutated e1 -> causal during _write_workset; next activate MUST see 0.85!
        eng.activate("触发器锚点", workset="w", hops=1)
        assert eng._graph_load_count == 2
        assert eng.carry_vector("w") == {"a": 1.0, "b": 0.85}
    finally:
        con.close()


def test_env_monkeypatch_invalidates_graph_cache(monkeypatch, tmp_path):
    """Monkeypatching csr_matrix or cond_match on a warm engine invalidates cache cleanly."""
    if _scipy_csr_matrix is None:
        pytest.skip("SciPy is optional")
    store = LayeredStore(str(tmp_path / "env.db"))
    cond = ConditionSpace("p", "t", (10.0, 20.0), "c")
    try:
        store.add_node(STNode(id="a", content="环境测试锚点", modality="text",
                              spatial_coordinates={}, temporal_coordinate=10.0, condition_space=cond))
        store.add_node(STNode(id="b", content="目标", modality="text",
                              spatial_coordinates={}, temporal_coordinate=10.0, condition_space=cond))
        store.add_edge(STEdge(id="e1", source_id="a", target_id="b",
                              relation_type=EdgeType.SIMILAR, condition_space=cond))
        eng = ActivationEngine(store=store, audit_path=str(tmp_path / "env.jsonl"))
        eng.activate("环境测试锚点", workset="w", hops=1)
        assert eng._adj is not None
        assert eng.carry_vector("w") == {"a": 1.0, "b": 0.75}

        # Switch to edge-list dynamically => cache MUST invalidate
        monkeypatch.setattr(activation, "csr_matrix", None)
        eng.activate("环境测试锚点", workset="w", hops=1)
        assert eng._adj is None
        assert eng._graph_load_count == 2
        assert eng.carry_vector("w") == {"a": 1.0, "b": 0.75}

        # Override cond_match dynamically => cache MUST invalidate
        monkeypatch.setattr(activation, "cond_match", lambda _a, _b: 0.5)
        eng.activate("环境测试锚点", workset="w", hops=1)
        assert eng._graph_load_count == 3
        assert eng.carry_vector("w") == {"a": 1.0, "b": 0.375}
    finally:
        store.close()
