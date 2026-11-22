# -*- coding: utf-8 -*-
"""稀疏激活路径平行边 max-plus 聚合守卫（防 csr_matrix 重复坐标自动求和）。

根因：``lingshu/core/activation.py`` 的 ``_load_graph()`` 此前把 ``(rows, cols, weights)``
直接交给 ``scipy.sparse.csr_matrix``，SciPy 会对重复 ``(si, di)`` 坐标自动**求和**而非取
``max``，导致：
  1. 平行兴奋边权重叠加越过 1.0（如 ``causal 0.85 + similar 0.75 + counterpart 0.75 = 2.35``），
     使 ``scipy`` 与 ``edge-list`` 分叉并翻转 Top-1 排序；
  2. 环路/自环上的平行兴奋边在多跳传播下跨跳指数放大，突破 #280 有界性不变量；
  3. 抑制矩阵 ``adj_inh``、低置信抑制矩阵 ``adj_inh_unv`` 存在同型重复边累加失真。

修法：构造 ``adj``、``adj_inh``、``adj_inh_unv`` 前统一经 ``_build_max_csr`` 按
``(si, di) -> max(weight)`` 聚合，使稀疏路径与 ``edge-list`` 在平行边拓扑下口径一致。

运行：
  pytest：python -X utf8 -m pytest tests/test_activation_sparse_parallel_max.py -q
  直跑：  python -X utf8 tests/test_activation_sparse_parallel_max.py
  变异：  python -X utf8 tests/test_activation_sparse_parallel_max.py --mutate
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

_REPO = Path(__file__).resolve().parents[1]
_SANDBOX = os.environ.get("LINGSHU_MUTATE_SANDBOX")
if _SANDBOX:
    if _SANDBOX not in sys.path:
        sys.path.insert(0, _SANDBOX)
elif str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    import pytest
except ImportError:
    pytest = None

try:
    from scipy.sparse import issparse as _scipy_issparse
except ImportError:
    _scipy_issparse = None

from lingshu.core import activation as A  # noqa: E402
from lingshu.core.core import (  # noqa: E402
    ConditionSpace,
    EdgeType,
    LayeredStore,
    STEdge,
    STNode,
)

COND = ConditionSpace("fixture", "fixture", (100.0, 101.0), "synthetic")
COND_DISJOINT = ConditionSpace("other_pos", "other_tool", (900000.0, 900001.0), "other_c")


def _require_scipy_for_pytest():
    if pytest is not None:
        if A.np is None:
            pytest.skip("ActivationEngine requires NumPy")
        if _scipy_issparse is None or A.csr_matrix is None:
            pytest.skip("SciPy is optional; install it to exercise the sparse backend")


def _add_node(store: LayeredStore, node_id: str, content: str,
              cond: ConditionSpace = COND):
    store.add_node(STNode(
        id=node_id, content=content, modality="text", spatial_coordinates={},
        temporal_coordinate=100.0, condition_space=cond))


def _add_edge(store: LayeredStore, edge_id: str, src: str, dst: str,
              relation, conf: float = 0.5):
    if isinstance(relation, EdgeType):
        store.add_edge(STEdge(
            id=edge_id, source_id=src, target_id=dst,
            relation_type=relation, condition_space=COND))
        store.conn.execute(
            "UPDATE edges SET confidence=? WHERE id=?", (conf, edge_id))
    else:
        store.add_edge(STEdge(
            id=edge_id, source_id=src, target_id=dst,
            relation_type=EdgeType.SIMILAR, condition_space=COND))
        store.conn.execute(
            "UPDATE edges SET relation_type=?, confidence=? WHERE id=?",
            (relation, conf, edge_id))
    store.conn.commit()


def _run_mode(mode: str, setup_fn, query: str, hops: int = 1,
              rounds: int = 1, self_condition: float = 0.0):
    saved = A.csr_matrix
    if mode == "edge-list":
        A.csr_matrix = None
    try:
        with tempfile.TemporaryDirectory() as td:
            store = LayeredStore(os.path.join(td, "memory.db"))
            audit = os.path.join(td, "audit.jsonl")
            try:
                setup_fn(store)
                eng = A.ActivationEngine(store=store, audit_path=audit)
                last_res = None
                round_maxima = []
                for _ in range(rounds):
                    last_res = eng.activate(
                        query, workset="ws", hops=hops,
                        self_condition=self_condition)
                    cv = eng.carry_vector("ws")
                    round_maxima.append(max(cv.values()) if cv else 0.0)
                audit_records = [
                    json.loads(line)
                    for line in Path(audit).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                return {
                    "carried": eng.carry_vector("ws"),
                    "top": last_res["top"],
                    "round_maxima": round_maxima,
                    "audit": audit_records,
                    "idx": dict(eng._node_index),
                    "adj": eng._adj,
                    "adj_inh": eng._adj_inh,
                    "adj_inh_unv": eng._adj_inh_unv,
                }
            finally:
                store.close()
    finally:
        A.csr_matrix = saved


def _close(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(float(a) - float(b)) <= tol


def _dict_close(a: dict, b: dict, tol: float = 1e-4) -> bool:
    return set(a) == set(b) and all(_close(a[k], b[k], tol) for k in a)


# --------------------------------------------------------------------------- #
# 独立断言函数（pytest 与直跑/变异共用同一套口径）
# --------------------------------------------------------------------------- #
def _check_a01_parallel_excitatory_max():
    """A01: 平行兴奋边（causal 0.85 + similar 0.75 + counterpart 0.75）取 max=0.85 而非 2.35。"""
    def setup_desc(store):
        _add_node(store, "src", "起始源锚")
        _add_node(store, "dst", "目标节点")
        _add_edge(store, "e1", "src", "dst", EdgeType.CAUSAL)
        _add_edge(store, "e2", "src", "dst", EdgeType.SIMILAR)
        _add_edge(store, "e3", "src", "dst", "counterpart")

    def setup_asc(store):
        _add_node(store, "src", "起始源锚")
        _add_node(store, "dst", "目标节点")
        _add_edge(store, "e1", "src", "dst", "spatial")
        _add_edge(store, "e2", "src", "dst", EdgeType.SIMILAR)
        _add_edge(store, "e3", "src", "dst", EdgeType.CAUSAL)

    for tag, fn in (("desc", setup_desc), ("asc", setup_asc)):
        sp = _run_mode("scipy", fn, "起始源锚", hops=1)
        el = _run_mode("edge-list", fn, "起始源锚", hops=1)
        si, di = sp["idx"]["src"], sp["idx"]["dst"]
        if sp["adj"] is None or not _scipy_issparse(sp["adj"]):
            return False, "%s: expected real CSR matrix" % tag
        w = float(sp["adj"][si, di])
        if sp["adj"].nnz != 1 or not _close(w, 0.85):
            return False, "%s: adj[src,dst]=%r nnz=%d (expected 0.85, nnz=1)" % (
                tag, w, sp["adj"].nnz)
        if not _dict_close(sp["carried"], el["carried"]):
            return False, "%s: scipy=%r != edge-list=%r" % (
                tag, sp["carried"], el["carried"])
        if sp["top"] != el["top"]:
            return False, "%s: top mismatch scipy=%r edge-list=%r" % (
                tag, sp["top"], el["top"])
    return True, ""


def _check_a02_no_top1_ranking_flip():
    """A02: 2×sequential(0.60) 不得累加为 1.20 压过单条 causal(0.85) 导致 Top-1 翻转。"""
    def setup(store):
        _add_node(store, "src", "起始源锚")
        _add_node(store, "strong", "强单边靶")
        _add_node(store, "weak_par", "弱双边靶")
        _add_edge(store, "e_s", "src", "strong", EdgeType.CAUSAL)
        _add_edge(store, "e_w1", "src", "weak_par", EdgeType.SEQUENTIAL)
        _add_edge(store, "e_w2", "src", "weak_par", EdgeType.SEQUENTIAL)

    sp = _run_mode("scipy", setup, "起始源锚", hops=1)
    el = _run_mode("edge-list", setup, "起始源锚", hops=1)
    if sp["top"] != el["top"]:
        return False, "ranking flipped: scipy=%r edge-list=%r" % (sp["top"], el["top"])
    if [k for k, _ in sp["top"]] != ["src", "strong", "weak_par"]:
        return False, "unexpected order: %r" % (sp["top"],)
    if not _close(sp["carried"]["weak_par"], 0.6):
        return False, "weak_par=%r expected 0.6" % (sp["carried"]["weak_par"],)
    return True, ""


def _check_a03_cycle_multihop_bounded():
    """A03: 2 节点平行边环路多跳与自条件跨轮严格有界 <= 1.0（保 #280 不变量）。"""
    def setup(store):
        _add_node(store, "a", "循环甲锚")
        _add_node(store, "b", "循环乙靶")
        _add_edge(store, "ab1", "a", "b", EdgeType.SIMILAR)
        _add_edge(store, "ab2", "a", "b", "counterpart")
        _add_edge(store, "ba1", "b", "a", EdgeType.SIMILAR)
        _add_edge(store, "ba2", "b", "a", "counterpart")

    sp = _run_mode("scipy", setup, "循环甲锚", hops=6, rounds=4, self_condition=0.5)
    el = _run_mode("edge-list", setup, "循环甲锚", hops=6, rounds=4, self_condition=0.5)
    if not _dict_close(sp["carried"], el["carried"]):
        return False, "cycle divergence: scipy=%r edge-list=%r" % (
            sp["carried"], el["carried"])
    if max(sp["round_maxima"]) > 1.0 + 1e-4:
        return False, "unbounded maxima=%r" % (sp["round_maxima"],)
    for rec in sp["audit"]:
        for step in rec["steps"]:
            if step["phase"] == "propagate" and step["max_act"] > 1.0 + 1e-4:
                return False, "audit max_act=%r > 1.0" % (step["max_act"],)
    return True, ""


def _check_a04_selfloop_and_direction_isolation():
    """A04: 平行自环取 max 不发散，且反向边 (a->b) 与 (b->a) 互不混淆。"""
    def setup(store):
        _add_node(store, "a", "自环甲锚")
        _add_node(store, "b", "双向乙靶")
        _add_edge(store, "aa1", "a", "a", EdgeType.SIMILAR)
        _add_edge(store, "aa2", "a", "a", "counterpart")
        _add_edge(store, "ab1", "a", "b", EdgeType.CAUSAL)
        _add_edge(store, "ba1", "b", "a", "spatial")

    sp = _run_mode("scipy", setup, "自环甲锚", hops=4)
    el = _run_mode("edge-list", setup, "自环甲锚", hops=4)
    ia, ib = sp["idx"]["a"], sp["idx"]["b"]
    w_aa = float(sp["adj"][ia, ia])
    w_ab = float(sp["adj"][ia, ib])
    w_ba = float(sp["adj"][ib, ia])
    if not (_close(w_aa, 0.75) and _close(w_ab, 0.85) and _close(w_ba, 0.50)):
        return False, "weights=(aa=%r, ab=%r, ba=%r) expected (0.75, 0.85, 0.50)" % (
            w_aa, w_ab, w_ba)
    if not _dict_close(sp["carried"], el["carried"]):
        return False, "scipy=%r != edge-list=%r" % (sp["carried"], el["carried"])
    return True, ""


def _check_b01_parallel_verified_inhibition_max():
    """B01: 平行已复核 opposite 边（2×conf=0.8）在 adj_inh 取 max=1.0 而非 2.0。"""
    def setup(store):
        _add_node(store, "src", "起始源锚")
        _add_node(store, "rival", "起始对手")
        _add_node(store, "tgt", "目标节点")
        _add_edge(store, "e_exc", "src", "tgt", EdgeType.CAUSAL)
        _add_edge(store, "e_i1", "rival", "tgt", EdgeType.OPPOSITE, conf=0.8)
        _add_edge(store, "e_i2", "rival", "tgt", EdgeType.OPPOSITE, conf=0.8)

    sp = _run_mode("scipy", setup, "起始源", hops=1)
    el = _run_mode("edge-list", setup, "起始源", hops=1)
    ir, it = sp["idx"]["rival"], sp["idx"]["tgt"]
    w = float(sp["adj_inh"][ir, it])
    if sp["adj_inh"].nnz != 1 or not _close(w, 1.0):
        return False, "adj_inh[rival,tgt]=%r nnz=%d (expected 1.0, nnz=1)" % (
            w, sp["adj_inh"].nnz)
    if not _dict_close(sp["carried"], el["carried"], tol=1e-3):
        return False, "scipy=%r != edge-list=%r" % (sp["carried"], el["carried"])
    return True, ""


def _check_b02_parallel_mixed_conf_inhibition_max():
    """B02: 同对节点混有高置信(0.8→1.0)与低置信(0.3→0.3) opposite 边时 adj_inh 取 1.0 而非 1.3。"""
    def setup(store):
        _add_node(store, "src", "起始源锚")
        _add_node(store, "rival", "起始对手")
        _add_node(store, "tgt", "目标节点")
        _add_edge(store, "e_exc", "src", "tgt", EdgeType.CAUSAL)
        _add_edge(store, "e_hi", "rival", "tgt", EdgeType.OPPOSITE, conf=0.8)
        _add_edge(store, "e_lo", "rival", "tgt", EdgeType.OPPOSITE, conf=0.3)

    sp = _run_mode("scipy", setup, "起始源", hops=1)
    el = _run_mode("edge-list", setup, "起始源", hops=1)
    ir, it = sp["idx"]["rival"], sp["idx"]["tgt"]
    w_inh = float(sp["adj_inh"][ir, it])
    w_unv = float(sp["adj_inh_unv"][ir, it])
    if not (_close(w_inh, 1.0) and _close(w_unv, 0.3)):
        return False, "adj_inh=%r adj_inh_unv=%r (expected 1.0, 0.3)" % (w_inh, w_unv)
    if not _close(sp["carried"]["tgt"], el["carried"]["tgt"], tol=1e-3):
        return False, "tgt scipy=%r edge-list=%r" % (
            sp["carried"]["tgt"], el["carried"]["tgt"])
    return True, ""


def _check_b03_parallel_unverified_inhibition_max():
    """B03: 平行低置信 opposite 边（conf=0.2 与 0.4）在 adj_inh 与 adj_inh_unv 均取 max=0.4。"""
    def setup(store):
        _add_node(store, "src", "起始源锚")
        _add_node(store, "rival", "起始对手")
        _add_node(store, "tgt", "目标节点")
        _add_edge(store, "e_exc", "src", "tgt", EdgeType.CAUSAL)
        _add_edge(store, "e_u1", "rival", "tgt", EdgeType.OPPOSITE, conf=0.2)
        _add_edge(store, "e_u2", "rival", "tgt", EdgeType.OPPOSITE, conf=0.4)

    sp = _run_mode("scipy", setup, "起始", hops=1)
    el = _run_mode("edge-list", setup, "起始", hops=1)
    ir, it = sp["idx"]["rival"], sp["idx"]["tgt"]
    w_inh = float(sp["adj_inh"][ir, it])
    w_unv = float(sp["adj_inh_unv"][ir, it])
    if sp["adj_inh"].nnz != 1 or not _close(w_inh, 0.4):
        return False, "adj_inh[rival,tgt]=%r nnz=%d (expected 0.4, nnz=1)" % (
            w_inh, sp["adj_inh"].nnz)
    if sp["adj_inh_unv"].nnz != 1 or not _close(w_unv, 0.4):
        return False, "adj_inh_unv[rival,tgt]=%r nnz=%d (expected 0.4, nnz=1)" % (
            w_unv, sp["adj_inh_unv"].nnz)
    if not _close(sp["carried"]["tgt"], el["carried"]["tgt"], tol=1e-3):
        return False, "tgt scipy=%r edge-list=%r (expected 0.45)" % (
            sp["carried"]["tgt"], el["carried"]["tgt"])
    return True, ""


def _check_c01_zero_weight_and_cond_match():
    """C01: 条件空间全不重合（cond_match=0.0）的平行边保持 0 权，不破坏非零边。"""
    def setup(store):
        _add_node(store, "src", "条件源锚", cond=COND)
        _add_node(store, "zero_dst", "零权靶点", cond=COND_DISJOINT)
        _add_node(store, "live_dst", "有效靶点", cond=COND)
        _add_edge(store, "z1", "src", "zero_dst", EdgeType.CAUSAL)
        _add_edge(store, "z2", "src", "zero_dst", EdgeType.SIMILAR)
        _add_edge(store, "l1", "src", "live_dst", EdgeType.SIMILAR)
        _add_edge(store, "l2", "src", "live_dst", EdgeType.CAUSAL)

    sp = _run_mode("scipy", setup, "条件源锚", hops=1)
    el = _run_mode("edge-list", setup, "条件源锚", hops=1)
    isrc, iz, il = sp["idx"]["src"], sp["idx"]["zero_dst"], sp["idx"]["live_dst"]
    if not (_close(float(sp["adj"][isrc, iz]), 0.0)
            and _close(float(sp["adj"][isrc, il]), 0.85)):
        return False, "weights=(%r, %r)" % (
            float(sp["adj"][isrc, iz]), float(sp["adj"][isrc, il]))
    if not _dict_close(sp["carried"], el["carried"]):
        return False, "scipy=%r != edge-list=%r" % (sp["carried"], el["carried"])
    return True, ""


def _check_d01_multigraph_scipy_edgelist_equivalence():
    """D01: 多节点混合平行兴奋+已复核抑制多跳图，scipy 与 edge-list 逐项等价且有界。"""
    def setup(store):
        _add_node(store, "s0", "主查锚点")
        _add_node(store, "s1", "主查旁支")
        _add_node(store, "m1", "中继甲")
        _add_node(store, "m2", "中继乙")
        _add_node(store, "t1", "终靶甲")
        _add_node(store, "t2", "终靶乙")
        # s0 -> m1 三条平行兴奋边（max=0.85）
        _add_edge(store, "e1", "s0", "m1", EdgeType.SEQUENTIAL)
        _add_edge(store, "e2", "s0", "m1", EdgeType.CAUSAL)
        _add_edge(store, "e3", "s0", "m1", EdgeType.SIMILAR)
        # m1 <-> m2 平行双向环（max=0.75）
        _add_edge(store, "e4", "m1", "m2", EdgeType.SIMILAR)
        _add_edge(store, "e5", "m1", "m2", "counterpart")
        _add_edge(store, "e6", "m2", "m1", EdgeType.SIMILAR)
        _add_edge(store, "e7", "m2", "m1", EdgeType.HIERARCHICAL)
        # m2 -> t1 平行兴奋（max=0.70）；m1 -> t2 平行兴奋（max=0.85）
        _add_edge(store, "e8", "m2", "t1", EdgeType.SEQUENTIAL)
        _add_edge(store, "e9", "m2", "t1", EdgeType.HIERARCHICAL)
        _add_edge(store, "e10", "m1", "t2", EdgeType.CAUSAL)
        _add_edge(store, "e11", "m1", "t2", EdgeType.SIMILAR)
        # s1 -> t2 两条平行已复核 opposite（max=1.0）
        _add_edge(store, "i1", "s1", "t2", EdgeType.OPPOSITE, conf=0.9)
        _add_edge(store, "i2", "s1", "t2", EdgeType.OPPOSITE, conf=0.7)

    sp = _run_mode("scipy", setup, "主查锚", hops=3)
    el = _run_mode("edge-list", setup, "主查锚", hops=3)
    if not _dict_close(sp["carried"], el["carried"], tol=1e-3):
        return False, "multigraph mismatch: scipy=%r edge-list=%r" % (
            sp["carried"], el["carried"])
    if sp["top"] != el["top"]:
        return False, "multigraph top mismatch: scipy=%r edge-list=%r" % (
            sp["top"], el["top"])
    if max(sp["carried"].values()) > 1.0 + 1e-4:
        return False, "multigraph unbounded: %r" % (sp["carried"],)
    return True, ""


_SUITE = [
    ("A01", _check_a01_parallel_excitatory_max),
    ("A02", _check_a02_no_top1_ranking_flip),
    ("A03", _check_a03_cycle_multihop_bounded),
    ("A04", _check_a04_selfloop_and_direction_isolation),
    ("B01", _check_b01_parallel_verified_inhibition_max),
    ("B02", _check_b02_parallel_mixed_conf_inhibition_max),
    ("B03", _check_b03_parallel_unverified_inhibition_max),
    ("C01", _check_c01_zero_weight_and_cond_match),
    ("D01", _check_d01_multigraph_scipy_edgelist_equivalence),
]


# --------------------------------------------------------------------------- #
# pytest 入口
# --------------------------------------------------------------------------- #
def test_a01_parallel_excitatory_max():
    _require_scipy_for_pytest()
    ok, detail = _check_a01_parallel_excitatory_max()
    assert ok, detail


def test_a02_no_top1_ranking_flip():
    _require_scipy_for_pytest()
    ok, detail = _check_a02_no_top1_ranking_flip()
    assert ok, detail


def test_a03_cycle_multihop_bounded():
    _require_scipy_for_pytest()
    ok, detail = _check_a03_cycle_multihop_bounded()
    assert ok, detail


def test_a04_selfloop_and_direction_isolation():
    _require_scipy_for_pytest()
    ok, detail = _check_a04_selfloop_and_direction_isolation()
    assert ok, detail


def test_b01_parallel_verified_inhibition_max():
    _require_scipy_for_pytest()
    ok, detail = _check_b01_parallel_verified_inhibition_max()
    assert ok, detail


def test_b02_parallel_mixed_conf_inhibition_max():
    _require_scipy_for_pytest()
    ok, detail = _check_b02_parallel_mixed_conf_inhibition_max()
    assert ok, detail


def test_b03_parallel_unverified_inhibition_max():
    _require_scipy_for_pytest()
    ok, detail = _check_b03_parallel_unverified_inhibition_max()
    assert ok, detail


def test_c01_zero_weight_and_cond_match():
    _require_scipy_for_pytest()
    ok, detail = _check_c01_zero_weight_and_cond_match()
    assert ok, detail


def test_d01_multigraph_scipy_edgelist_equivalence():
    _require_scipy_for_pytest()
    ok, detail = _check_d01_multigraph_scipy_edgelist_equivalence()
    assert ok, detail


# --------------------------------------------------------------------------- #
# 直跑 + 沙箱定点变异自证（不改动工作区源文件）
# --------------------------------------------------------------------------- #
def _run_assertions() -> int:
    if A.np is None or _scipy_issparse is None or A.csr_matrix is None:
        print("SKIP: 需要 numpy 与 scipy 才能运行稀疏路径守卫")
        return 0
    fails = 0
    for cid, fn in _SUITE:
        ok, detail = fn()
        if ok:
            print("  PASS %s" % cid)
        else:
            print("  FAIL %s  %s" % (cid, detail[:240]))
            fails += 1
    print("通过 %d / 失败 %d" % (len(_SUITE) - fails, fails))
    return fails


_MUTANTS = {
    # M1: 退回 adj 裸 csr_matrix（平行兴奋边恢复求和）
    "M1": (
        "lingshu/core/activation.py",
        [(
            "        adj = self._build_max_csr(rows, cols, decays, n)",
            "        adj = (csr_matrix((np.array(decays, dtype=np.float32), "
            "(rows, cols)), shape=(n, n)) if (csr_matrix is not None and rows) else None)",
        )],
        {"A01", "A02", "A03", "A04", "C01", "D01"},
    ),
    # M2: 退回 adj_inh 裸 csr_matrix（平行抑制边恢复求和）
    "M2": (
        "lingshu/core/activation.py",
        [(
            "        adj_inh = self._build_max_csr(inh_rows, inh_cols, inh_mags, n)",
            "        adj_inh = (csr_matrix((np.array(inh_mags, dtype=np.float32), "
            "(inh_rows, inh_cols)), shape=(n, n)) if (csr_matrix is not None and inh_rows) else None)",
        )],
        {"B01", "B02", "B03", "D01"},
    ),
    # M3: 退回 adj_inh_unv 裸 csr_matrix（平行低置信抑制边恢复求和）
    "M3": (
        "lingshu/core/activation.py",
        [(
            "        adj_inh_unv = self._build_max_csr(unv_rows, unv_cols, unv_mags, n)",
            "        adj_inh_unv = (csr_matrix((np.array(unv_mags, dtype=np.float32), "
            "(unv_rows, unv_cols)), shape=(n, n)) if (csr_matrix is not None and unv_rows) else None)",
        )],
        {"B03"},
    ),
    # M4: 把 _build_max_csr 内的 max 聚合改成求和聚合
    "M4": (
        "lingshu/core/activation.py",
        [(
            "            if prev is None or w > prev:\n"
            "                best[key] = w",
            "            best[key] = w if prev is None else prev + w",
        )],
        {"A01", "A02", "A03", "A04", "B01", "B02", "B03", "C01", "D01"},
    ),
}


def _run_mutant_in_sandbox(rel: str, edits):
    with tempfile.TemporaryDirectory() as td:
        sandbox = Path(td)
        shutil.copytree(
            _REPO / "lingshu", sandbox / "lingshu",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        target = sandbox / rel
        norm = target.read_text(encoding="utf-8").replace("\r\n", "\n")
        for old, new in edits:
            cnt = norm.count(old)
            if cnt != 1:
                return None, "锚点命中 %d 次（期望 1）：%r" % (cnt, old[:80])
            norm = norm.replace(old, new, 1)
        target.write_text(norm, encoding="utf-8", newline="\n")
        env = dict(os.environ)
        env["LINGSHU_MUTATE_SANDBOX"] = str(sandbox)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        r = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", __file__, "--assert-only"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(_REPO), env=env, timeout=180)
        out = (r.stdout or "") + (r.stderr or "")
        failed = set(re.findall(r"(?m)^\s*FAIL\s+([A-Z]\d+)", r.stdout or ""))
        return failed, out


def _mutate(argv) -> int:
    want = ""
    for a in argv:
        if a == "--mutate":
            want = want or "ALL"
        elif a.startswith("--mutate="):
            want = a.split("=", 1)[1]
    if not want:
        return 0
    ids = sorted(_MUTANTS) if want == "ALL" else [want]
    red = 0
    for mid in ids:
        if mid not in _MUTANTS:
            print("[ANCHOR-MISS] %s：未登记该变异臂" % mid)
            return 2
        rel, edits, expect = _MUTANTS[mid]
        actual, out = _run_mutant_in_sandbox(rel, edits)
        if actual is None:
            print("[ANCHOR-MISS] %s：%s" % (mid, out))
            return 2
        if actual == expect:
            print("[MUTATE-OK]   %s：指定断言转红 %s" % (mid, sorted(actual)))
        else:
            print("[MUTATE-FAIL] %s：期望红 %s，实际红 %s"
                  % (mid, sorted(expect), sorted(actual)))
            for line in out.splitlines():
                if line.strip().startswith("FAIL"):
                    print("    " + line)
            red += 1
    print("反证结果：%d 条变异，%s"
          % (len(ids), "全部恰好命中" if red == 0 else "%d 条未命中" % red))
    return 1 if red else 0


if __name__ == "__main__":
    _f = _run_assertions()
    if "--assert-only" in sys.argv:
        sys.exit(1 if _f else 0)
    _m = _mutate(sys.argv[1:])
    sys.exit(1 if (_f or _m) else 0)
