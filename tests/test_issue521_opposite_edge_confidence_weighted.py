# -*- coding: utf-8 -*-
"""#521 守卫：OPPOSITE（抑制）边的抑制强度按边 confidence 成比例。

缺陷（修前）：`ActivationEngine._load_graph` 的 edges 腿只
`SELECT source_id, target_id, relation_type, condition_space`——**从不装载**
`confidence`/`verified`，抑制幅值恒 = |EDGE_BASE_DECAY['opposite']|(=1.0) × cond_match。
于是 `LayeredStore.register_conflict` 建的矛盾边（confidence=0.3、verified=False）
与手工置 confidence=1.0 / verified=True 的边读数逐位相同：目标激活一律被压到 0
（未复核的低置信边等价于「已确认否决」，抑制不成比例）。

修后：抑制幅值 = |base| × cond_match × `suppression_weight(edge.confidence)`；
confidence 缺失/NULL（老库无该列）仍为 1.0（既有「无条件抑制」语义不变），
高置信（1.0）边仍把目标压到 0（保留既有语义）。

读数面（source 种子 1.0，SIMILAR 兴奋 0.75，OPPOSITE 抑制 c·cond_match）：
  target = max(0, 0.75·m − c·m) = m·(0.75 − c)，m = cond_match(两端节点条件空间)。

运行：python -m pytest tests/test_issue521_opposite_edge_confidence_weighted.py -q
"""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.core.core import ConditionSpace, EdgeType, LayeredStore, STEdge, STNode

try:
    import lingshu.core.activation as activation
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    pytest.skip("ActivationEngine requires NumPy", allow_module_level=True)

try:
    from scipy.sparse import issparse as _scipy_issparse
except ImportError:
    _scipy_issparse = None

CONDITION = ConditionSpace("fixture", "fixture", (100.0, 101.0), "synthetic")


@pytest.fixture(params=("edge-list", "scipy"))
def backend(request, monkeypatch):
    """两后端都真跑：scipy 缺失时该参数组显式 SKIP（不静默放绿）。"""
    if request.param == "edge-list":
        monkeypatch.setattr(activation, "csr_matrix", None)
    elif _scipy_issparse is None:
        pytest.skip("SciPy is optional; install it to exercise the sparse backend")
    return request.param


def add_node(store, node_id, content):
    store.add_node(STNode(
        id=node_id, content=content, modality="text", spatial_coordinates={},
        temporal_coordinate=100.0, condition_space=CONDITION))


def add_edge(store, edge_id, src, dst, relation, confidence=None, verified=False):
    kwargs = dict(id=edge_id, source_id=src, target_id=dst, relation_type=relation,
                  condition_space=CONDITION)
    if confidence is None:                       # 不传 ⇒ STEdge 声明缺省 0.5
        store.add_edge(STEdge(**kwargs))
    else:
        store.add_edge(STEdge(confidence=confidence, verified=verified, **kwargs))


def build(tmp_path, name, confidence=None, verified=False, with_exc=True):
    """source（种子 1.0）+ target；可选 SIMILAR 兴奋边 + OPPOSITE 抑制边。"""
    store = LayeredStore(str(tmp_path / (name + ".db")))
    engine = activation.ActivationEngine(store.db_path,
                                         str(tmp_path / (name + ".jsonl")))
    add_node(store, "source", "合成源锚")
    add_node(store, "target", "目标节点")
    if with_exc:
        add_edge(store, "e_exc", "source", "target", EdgeType.SIMILAR)
    if confidence is not None:
        add_edge(store, "e_inh", "source", "target", EdgeType.OPPOSITE,
                 confidence, verified)
    return store, engine


def read_target(engine, workset="ws"):
    """跑一轮（hops=1）后读 target 的激活强度；不在工作态 ⇒ 0.0。"""
    engine.activate("合成源", workset=workset, hops=1)
    return float(engine.carry_vector(workset).get("target", 0.0))


# --------------------------------------------------------------------------- #
# ① 成比例：同一图、仅改边 confidence
# --------------------------------------------------------------------------- #
def test_low_confidence_suppression_is_proportional_to_confidence(backend, tmp_path):
    """0.3 的未复核边抑制显著弱于 1.0 的已复核边，且差值成比例（非「都是 0」）。"""
    store, engine = build(tmp_path, "prop", confidence=0.3, verified=False)
    try:
        low = read_target(engine)
        # 抑制幅值 = 1.0 × cond_match(=1.0) × 0.3
        assert engine._inh_meta[2] == pytest.approx([0.3], abs=1e-6), engine._inh_meta
        assert engine._inh_conf == [(0.3, False, 0.3)], engine._inh_conf
        assert low == pytest.approx(0.75 - 0.3, abs=1e-3), (
            "confidence=0.3 的目标激活 %r；成比例期望 0.75−0.3=0.45（修前恒 0.0）" % (low,))
    finally:
        store.close()

    store2, engine2 = build(tmp_path, "prop_hi", confidence=1.0, verified=True)
    try:
        high = read_target(engine2)
        assert engine2._inh_meta[2] == pytest.approx([1.0], abs=1e-6)
        assert engine2._inh_conf == [(1.0, True, 1.0)], engine2._inh_conf
        assert high == pytest.approx(0.0, abs=1e-6), (
            "confidence=1.0/verified=True 的目标激活 %r；期望 0.0" % (high,))
    finally:
        store2.close()

    # 「显著弱于」须有量级门槛（不只断言两者不等，浮点误差也能蒙过）
    assert low - high > 0.4, "低置信边抑制 %r 与高置信边 %r 之差不足量级" % (low, high)


@pytest.mark.parametrize("confidence,expected", [
    (0.25, 0.75 - 0.25), (0.5, 0.75 - 0.5), (0.75, 0.75 - 0.75)])
def test_suppression_scales_monotonically_with_confidence(backend, tmp_path,
                                                          confidence, expected):
    """逐档：target = 0.75 − confidence（修前各档恒 0.0，与 confidence 无关）。"""
    store, engine = build(tmp_path, "mono_%s" % confidence, confidence=confidence,
                          verified=False)
    try:
        assert read_target(engine) == pytest.approx(expected, abs=1e-3)
    finally:
        store.close()


def test_high_confidence_verified_edge_still_vetoes(backend, tmp_path):
    """保留既有语义：已确认（1.0/verified=True）的 OPPOSITE 边仍把目标压到 0。"""
    store, engine = build(tmp_path, "veto", confidence=1.0, verified=True)
    try:
        engine.activate("合成源", workset="ws", hops=1)
        carried = engine.carry_vector("ws")
        assert carried == {"source": 1.0}, (
            "高置信相反边未否决目标：%r" % (carried,))
        assert "target" not in carried
    finally:
        store.close()


def test_verified_flag_does_not_gate_strength(backend, tmp_path):
    """边界声明：强度只按 confidence 成比例，verified 不额外加权（母栏裁定口径）。"""
    store, engine = build(tmp_path, "unver", confidence=1.0, verified=False)
    try:
        assert read_target(engine) == pytest.approx(0.0, abs=1e-6), (
            "confidence=1.0/verified=False 未按 1.0 强度抑制（verified 不应折减强度）")
    finally:
        store.close()


def test_default_confidence_edge_suppresses_at_half_strength(backend, tmp_path):
    """语义变更点（显式成文，供母栏裁定）：STEdge 默认 confidence=0.5 的**未标注**
    OPPOSITE 边，抑制幅值随本笔从 1.0 变为 0.5（target 由 0.0 变为 0.25）。

    这是「按 confidence 成比例」的直接推论——0.5 的中性缺省不再等价于满强度。
    既有 `tests/test_issue229_280_activation_suppression_bounded.py` 的 helper 正是用
    STEdge 默认 confidence=0.5 建 OPPOSITE 边并断言幅值 1.0（target=0.75−1/3），
    故该文件 4 项参数化断言随本笔过期（该文件不在本卡 write_scope，未改）。
    """
    # 既有 test_issue229_280 的 helper 不传 confidence ⇒ 落 STEdge 声明的默认 0.5
    assert STEdge(id="probe", source_id="a", target_id="b",
                  relation_type=EdgeType.OPPOSITE,
                  condition_space=CONDITION).confidence == pytest.approx(0.5)
    store, engine = build(tmp_path, "half", confidence=0.5, verified=False)
    try:
        assert read_target(engine) == pytest.approx(0.75 - 0.5, abs=1e-3)
        assert engine._inh_meta[2] == pytest.approx([0.5], abs=1e-6)
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# ② 真实入口：register_conflict 默认边（confidence=0.3、verified=False）
# --------------------------------------------------------------------------- #
def test_register_conflict_default_edge_reads_proportionally(backend, tmp_path):
    """register_conflict 建的 OPPOSITE(0.3, False)：读数 0.45，不再无条件压到 0。"""
    store = LayeredStore(str(tmp_path / "reg.db"))
    engine = activation.ActivationEngine(store.db_path, str(tmp_path / "reg.jsonl"))
    try:
        add_node(store, "source", "合成源锚")
        add_node(store, "target", "目标节点")
        add_edge(store, "e_exc", "source", "target", EdgeType.SIMILAR)
        edge = store.register_conflict("source", "target")
        assert edge is not None, "register_conflict 未建边（前置不成立）"
        assert edge.relation_type is EdgeType.OPPOSITE
        assert edge.confidence == pytest.approx(0.3)
        assert edge.verified is False
        target = read_target(engine)
        assert target == pytest.approx(0.75 - 0.3, abs=1e-3), (
            "register_conflict 默认边下目标激活 %r；期望 0.45（修前 0.0）" % (target,))
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# ③ 缺列兜底：老库 edges 无 confidence/verified 列 ⇒ 系数 1.0（既有语义）
# --------------------------------------------------------------------------- #
def _legacy_db(path):
    """造一个 edges 只有四列的老库（#185 之前的 schema）。

    nodes 保留 content/tags（`_seed_by_text` 的投票面）+ condition_space
    （`_load_graph` 的条件面），其余列一律不建。
    """
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE nodes (id TEXT PRIMARY KEY, content TEXT, tags TEXT, "
                "condition_space TEXT)")
    con.execute("CREATE TABLE edges (source_id TEXT, target_id TEXT, "
                "relation_type TEXT, condition_space TEXT)")
    cs = CONDITION.to_json()
    for nid, content in (("source", "合成源锚"), ("target", "目标节点")):
        con.execute("INSERT INTO nodes VALUES (?,?,?,?)", (nid, content, "[]", cs))
    con.execute("INSERT INTO edges VALUES (?,?,?,?)",
                ("source", "target", "similar", cs))
    con.execute("INSERT INTO edges VALUES (?,?,?,?)",
                ("source", "target", "opposite", cs))
    con.commit()
    con.close()


def test_missing_confidence_columns_fall_back_to_full_strength(backend, tmp_path):
    """缺列 ⇒ suppression_weight(None)=1.0：老库抑制强度不因本笔凭空削弱。"""
    db = tmp_path / "legacy.db"
    _legacy_db(db)
    engine = activation.ActivationEngine(str(db), str(tmp_path / "legacy.jsonl"))
    assert activation.suppression_weight(None) == 1.0
    assert engine.activate("合成源", workset="ws", hops=1)["seeds"] >= 1
    carried = engine.carry_vector("ws")
    assert carried.get("target", 0.0) == pytest.approx(0.0, abs=1e-6), (
        "缺列时抑制未按 1.0 强度生效：%r" % (carried,))
    assert engine._inh_meta[2] == pytest.approx([1.0], abs=1e-6)
    assert engine._inh_conf == [(None, False, 1.0)], engine._inh_conf


def test_suppression_weight_boundaries():
    """系数取值面：None⇒1.0；非有限/非数值⇒中性 0.5；越界钳到 [0,1]。"""
    assert activation.suppression_weight(None) == 1.0
    assert activation.suppression_weight(0.0) == 0.0
    assert activation.suppression_weight(1.0) == 1.0
    assert activation.suppression_weight(2.5) == 1.0
    assert activation.suppression_weight(-1.0) == 0.0
    assert activation.suppression_weight(float("nan")) == 0.5
    assert activation.suppression_weight("bad") == 0.5
