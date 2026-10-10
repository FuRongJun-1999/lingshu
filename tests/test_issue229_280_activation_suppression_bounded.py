# -*- coding: utf-8 -*-
"""激活引擎两条 P1 缺陷的定点守卫（issue #229 / #280）。

#229 负记忆无抑制通路：`opposite` 边原被当作兴奋边（`DEFAULT_DECAY=0.5`）正传播，
     相反记忆不但不扣减目标激活，反而抬升它。修复后 opposite 走独立抑制通路。
#280 自条件 × Σ 扩散跨轮指数发散：稀疏路径用 `adj.T @ act`（Σ 半环）逐跳累加，
     自条件回注（act₀ ← weight·act_prev）下跨轮指数放大。修复后改 max-plus 口径。

两条缺陷在 scipy 稀疏路径与 edge-list 降级路径上各有一份实现，故本守卫对
两种后端都跑（同 tests/test_activation_graph_refresh.py 的 scipy 可选处置）。

运行：python -m pytest tests/test_issue229_280_activation_suppression_bounded.py -q
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def add_node(store, node_id, content):
    store.add_node(STNode(
        id=node_id, content=content, modality="text", spatial_coordinates={},
        temporal_coordinate=100.0, condition_space=CONDITION))


def add_edge(store, edge_id, src, dst, relation):
    store.add_edge(STEdge(
        id=edge_id, source_id=src, target_id=dst,
        relation_type=relation, condition_space=CONDITION))


@pytest.fixture(params=("edge-list", "scipy"))
def backend(request, monkeypatch):
    """两后端都真跑：scipy 缺失时该参数组显式 SKIP（不静默放绿）。"""
    if request.param == "edge-list":
        monkeypatch.setattr(activation, "csr_matrix", None)
    elif _scipy_issparse is None:
        pytest.skip("SciPy is optional; install it to exercise the sparse backend")
    return request.param


def make_engine(tmp_path, name):
    store = LayeredStore(str(tmp_path / ("%s.db" % name)))
    engine = activation.ActivationEngine(store.db_path, str(tmp_path / ("%s.jsonl" % name)))
    return store, engine


# --------------------------------------------------------------------------- #
# #229 负记忆抑制通路
# --------------------------------------------------------------------------- #
def test_opposite_edge_is_registered_as_negative_decay():
    """判据来源：activation.py 模块 docstring :15「第五篇:…负记忆同价…」。

    缺陷形态下 `EDGE_BASE_DECAY` 无 opposite 项，落 `DEFAULT_DECAY=0.5`（正号=兴奋）。
    """
    assert activation.EDGE_BASE_DECAY.get("opposite", 0.0) < 0, (
        "opposite 边必须是负衰减（抑制），否则相反记忆与相似记忆同价正传播")


def test_opposite_edge_does_not_excite_its_target(backend, tmp_path):
    """#229 核心：相反边不得把目标节点抬进工作态。"""
    store, engine = make_engine(tmp_path, "opp229a")
    try:
        add_node(store, "source", "合成源锚")
        add_node(store, "target", "目标节点")
        add_edge(store, "opp", "source", "target", EdgeType.OPPOSITE)
        result = engine.activate("合成源锚", workset="ws", hops=2)
        carried = engine.carry_vector("ws")
        # 缺陷形态（opposite 走 0.5 兴奋）：carried == {"source":1.0,"target":0.5}
        assert carried == {"source": 1.0}, (
            "相反边把目标激活为 %r；缺陷形态下 target 会以 0.5 出现在工作态" % (carried,))
        assert result["size"] == 1
    finally:
        store.close()


def test_opposite_edge_subtracts_from_excitatory_activation(backend, tmp_path):
    """#229 抑制语义：opposite 边扣减目标激活，而非叠加。

    rival 以 1/3 种子强度经 opposite(幅值 1.0) 抑制 target；
    target 同时被 source 经 similar(0.75) 兴奋。判据：act(target)=0.75−1/3≈0.4167，
    缺陷形态（Σ 且 opposite=+0.5）则得 0.75+1/3·0.5≈0.9167。
    """
    store, engine = make_engine(tmp_path, "opp229b")
    try:
        add_node(store, "source", "合成源锚")
        add_node(store, "rival", "合成对手")
        add_node(store, "target", "目标节点")
        add_edge(store, "e_exc", "source", "target", EdgeType.SIMILAR)
        add_edge(store, "e_inh", "rival", "target", EdgeType.OPPOSITE)
        engine.activate("合成源", workset="ws", hops=1)
        carried = engine.carry_vector("ws")
        # 工作态值经 _write_workset 落表时 round(...,4)，故容差取 1e-3
        assert carried["source"] == pytest.approx(1.0, abs=1e-3)
        assert carried["rival"] == pytest.approx(1.0 / 3.0, abs=1e-3)
        # max-plus 兴奋 = 1.0·0.75；抑制 = (1/3)·1.0
        assert carried["target"] == pytest.approx(0.75 - 1.0 / 3.0, abs=1e-3), (
            "target=%r；期望 0.75−1/3≈0.4167（抑制通路），"
            "缺陷形态为 Σ 叠加 ≈0.9167" % (carried.get("target"),))
    finally:
        store.close()


def test_opposite_suppression_is_clamped_at_zero(backend, tmp_path):
    """#229 边界：抑制不得把目标激活压成负数（工作态值域 [0,∞)）。"""
    store, engine = make_engine(tmp_path, "opp229c")
    try:
        add_node(store, "source", "合成源锚")
        add_node(store, "rival", "合成对手")
        add_node(store, "target", "目标节点")
        add_edge(store, "e_exc", "source", "target", EdgeType.SIMILAR)     # 兴奋 0.75
        add_edge(store, "e_inh", "rival", "target", EdgeType.OPPOSITE)     # 抑制 1.0
        engine.activate("合成源", workset="ws", hops=1)
        carried = engine.carry_vector("ws")
        # 抑制量 (1/3)·1.0 = 1/3 < 0.75 ⇒ 目标保留，且值 = 0.75−1/3
        assert carried["target"] == pytest.approx(0.75 - 1.0 / 3.0, abs=1e-3)
        assert carried["target"] >= 0.0
        # 满激活源单独做完全抑制：抑制量 1.0 抵消目标全部兴奋 ⇒ 目标被清零且不入工作态
        store2, engine2 = make_engine(tmp_path, "opp229d")
        try:
            add_node(store2, "src2", "合成源锚")
            add_node(store2, "tgt2", "目标节点")
            add_edge(store2, "o2", "src2", "tgt2", EdgeType.OPPOSITE)
            engine2.activate("合成源锚", workset="ws2", hops=1)
            assert engine2.carry_vector("ws2") == {"src2": 1.0}
        finally:
            store2.close()
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# #280 Σ 扩散跨轮指数发散
# --------------------------------------------------------------------------- #
def test_propagation_is_max_plus_not_sum_across_sources(backend, tmp_path):
    """#280 判据来源：activation.py 核心公式 :18「max over u∈N(v)」。

    两源汇聚同一目标时，Σ 会相加（0.75+0.25·0.75），max-plus 只取最大（0.75）。
    """
    store, engine = make_engine(tmp_path, "div280a")
    try:
        add_node(store, "a", "合成锚点甲")     # 种子 1.0（四元组全中）
        add_node(store, "b", "锚点乙")         # 种子 0.25（仅中「锚点」）
        add_node(store, "hub", "目标节点")
        add_edge(store, "ea", "a", "hub", EdgeType.SIMILAR)
        add_edge(store, "eb", "b", "hub", EdgeType.SIMILAR)
        engine.activate("合成锚点", workset="ws", hops=1)
        carried = engine.carry_vector("ws")
        assert carried["a"] == pytest.approx(1.0)
        assert carried["b"] == pytest.approx(0.25)
        assert carried["hub"] == pytest.approx(0.75), (
            "hub=%r；max-plus 期望 1.0·0.75=0.75，Σ 半环会得 0.75+0.25·0.75=0.9375"
            % (carried.get("hub"),))
    finally:
        store.close()


def test_self_conditioned_activation_stays_bounded_across_rounds(backend, tmp_path):
    """#280 核心：自条件回注下跨轮不放大。

    星形图（hub 与 4 叶双向 similar）；叶0 恒为查询种子。自条件权重 0.5。
    缺陷形态（Σ）：第 1 轮 hub=1.5（=1.0·0.75+3·(1/3)·0.75）起，逐轮 2.02→2.45→… 指数放大。
    max-plus：act 上界 = max(种子, 0.5·act_prev) ⇒ 恒为 1.0，不随轮数增长。
    """
    store, engine = make_engine(tmp_path, "div280b")
    try:
        add_node(store, "hub", "中心枢纽")
        for i in range(4):
            add_node(store, "leaf%d" % i, "叶锚%d" % i)
            add_edge(store, "in%d" % i, "leaf%d" % i, "hub", EdgeType.SIMILAR)
            add_edge(store, "out%d" % i, "hub", "leaf%d" % i, EdgeType.SIMILAR)
        maxima = []
        for _ in range(6):
            engine.activate("叶锚0", workset="ws", hops=2, self_condition=0.5)
            maxima.append(max(engine.carry_vector("ws").values()))
        assert max(maxima) <= 1.0 + 1e-3, (
            "跨轮峰值 %r 越过 1.0 —— Σ 扩散在自条件回注下放大（#280）" % (maxima,))
        # 严格不增长：缺陷形态下从第 2 轮起严格递增（1.5→2.02→…）
        assert maxima == pytest.approx([maxima[0]] * len(maxima), abs=1e-4), (
            "跨轮峰值 %r 未收敛为常数（缺陷形态逐轮递增）" % (maxima,))
    finally:
        store.close()


def test_audit_max_act_is_bounded_under_self_condition(backend, tmp_path):
    """#280 可观测面：审计里的每跳 max_act 也不得越过种子上界。"""
    store, engine = make_engine(tmp_path, "div280c")
    try:
        add_node(store, "hub", "中心枢纽")
        for i in range(4):
            add_node(store, "leaf%d" % i, "叶锚%d" % i)
            add_edge(store, "in%d" % i, "leaf%d" % i, "hub", EdgeType.SIMILAR)
            add_edge(store, "out%d" % i, "hub", "leaf%d" % i, EdgeType.SIMILAR)
        engine.activate("叶锚0", workset="ws", hops=2, self_condition=0.5)
        engine.activate("叶锚0", workset="ws", hops=2, self_condition=0.5)
        records = [json.loads(line) for line in
                   Path(engine.audit_path).read_text(encoding="utf-8").splitlines()]
        assert len(records) == 2
        for record in records:
            for step in record["steps"]:
                if step["phase"] == "propagate":
                    assert step["max_act"] <= 1.0 + 1e-3, (
                        "第 %s 轮 step%d max_act=%s 越过 1.0（#280）"
                        % (record["query"], step["step"], step["max_act"]))
    finally:
        store.close()
