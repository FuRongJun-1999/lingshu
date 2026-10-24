# -*- coding: utf-8 -*-
"""activation 引擎五条 P2 缺陷的定点守卫（issue #7 / #161 / #293 / #314 / #53）
============================================================================
本件只钉「缺陷不再存在」，不钉「代码能跑」。逐件判据与来源：

#7  `lingshu/core/activation.py` 顶层裸 `import numpy as np` 击穿「轻核承诺」
    （README:「轻核：core 纯标准库，零依赖可跑」；`pyproject.toml:12`
    `dependencies = []`）。修后 numpy 走 `try/except`，缺失时模块仍可导入、
    `activate()` 返回 `{"status": "activation_not_ready", ...}` 显式降级。
    ★ 本件用**子进程 + `sys.modules["numpy"] = None`** 真造出「无 numpy」环境
    （issue #7 的核心论点就是「装齐依赖后跑 import 冒烟」是同义反复式验收），
    故断言在**它自己约定的条件**下成立，而非在有 numpy 的进程里再验一遍。
#161 空查询 / 纯空白查询：`toks` 含 `""`（或 `" "`）而 `"" in hay` 恒真 ⇒ 按表序
    前 top_k 个节点全以 1.0 种入、覆盖 workset。修后 `_seed_by_text` 短路返回空种子。
#293 `ActivationEngine` 接不上默认后端：`:memory:` 每次 `_con()` 都是新空库
    （`no such table: nodes`）；默认 DB 相对 cwd，父目录不存在则构造即崩。
    修后 `store=` / `conn=` 复用调用方连接（`_BorrowedConnection`），
    单给 `:memory:` 显式 `ValueError`，文件路径自动建父目录。
#314 工作态落表审计字段失真：种子 `source` 被兜底记成 `propagate`、`hop` 列恒 0
    ⇒ 快照无法还原扩散路径。修后 `source` 记 `seed`/`propagate`/`self_condition`，
    `hop` 记首次激活跳号（种子/自条件 = 0）。
#53  `cond_match` 零调用（装载期是空循环）⇒ docstring 核心公式
    `decay = base(边类型) × cond_match(两端条件空间)` 未实现。修后在 `_load_graph`
    装载边时按节点对算一次，稀疏与 edge-list 两后端同口径。

★ 跨批覆盖重叠处置（纪律）：本仓 main 上已有 `#229/#280` 守卫
  （`tests/test_issue229_280_activation_suppression_bounded.py`）。#53 的
  「cond_match 接线」与 #280 的「max-plus」都改变扩散读数，若只用
  `carry_vector` 断言，两条守卫可能互相代偿。故本件 #53 的夹具**只让 cond_match
  机制生效**：边类型取 `similar`（base 0.75，与 #229/#280 的 opposite/多源汇聚无关），
  两端条件空间**完全不相交** ⇒ 期望 target 落在工作态**之外**（act = 0）。
  该断言在「cond_match 未接线」形态下必红，且 #229/#280 的修复对它没有任何代偿作用
  （它们不改 `similar` 边的 base，也不改节点条件空间比对）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue7_161_293_314_53_activation_contract.py -q --no-header
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.core.core import (  # noqa: E402
    ConditionSpace, EdgeType, LayeredStore, STEdge, STNode, SpacetimeMemoryEngine,
)

try:
    import lingshu.core.activation as activation
except ModuleNotFoundError as error:   # pragma: no cover - 修好 #7 后不应再走到这里
    if error.name != "numpy":
        raise
    pytest.skip("ActivationEngine requires NumPy", allow_module_level=True)

try:
    from scipy.sparse import issparse as _scipy_issparse
except ImportError:                     # 软探针（吞异常，不被依赖闭包守卫判硬依赖）
    _scipy_issparse = None

REPO = str(Path(__file__).resolve().parents[1])
CONDITION = ConditionSpace("fixture", "fixture", (100.0, 101.0), "synthetic")


def add_node(store, node_id, content, condition=None):
    store.add_node(STNode(
        id=node_id, content=content, modality="text", spatial_coordinates={},
        temporal_coordinate=100.0, condition_space=condition or CONDITION))


def add_edge(store, edge_id, src, dst, relation=EdgeType.SIMILAR, condition=None):
    store.add_edge(STEdge(
        id=edge_id, source_id=src, target_id=dst,
        relation_type=relation, condition_space=condition or CONDITION))


def make_engine(tmp_path, name):
    store = LayeredStore(str(tmp_path / ("%s.db" % name)))
    engine = activation.ActivationEngine(store.db_path, str(tmp_path / ("%s.jsonl" % name)))
    return store, engine


@pytest.fixture(params=("edge-list", "scipy"))
def backend(request, monkeypatch):
    """两后端都真跑：scipy 缺失时该参数组显式 SKIP（不静默放绿）。"""
    if request.param == "edge-list":
        monkeypatch.setattr(activation, "csr_matrix", None)
    elif _scipy_issparse is None:
        pytest.skip("SciPy is optional; install it to exercise the sparse backend")
    return request.param


# --------------------------------------------------------------------------- #
# #7 无保护 import numpy
# --------------------------------------------------------------------------- #
def _run_without_numpy(code: str) -> subprocess.CompletedProcess:
    """在**真·无 numpy** 的子进程里跑 code（`sys.modules["numpy"] = None` 使任何
    `import numpy` 立即 ImportError——issue #7 的原始复现手法）。"""
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", timeout=180)


def test_module_imports_without_numpy():
    """#7：轻核承诺的核心——**无 numpy 也要能 import**。"""
    proc = _run_without_numpy(textwrap.dedent('''
        import sys
        sys.modules["numpy"] = None
        import lingshu.core.activation as a
        print("IMPORTED", a.np is None)
    '''))
    assert proc.returncode == 0, (
        "无 numpy 时 activation 导入失败（裸 import numpy 击穿轻核承诺）：\n%s\n%s"
        % (proc.stdout, proc.stderr))
    assert "IMPORTED True" in proc.stdout, proc.stdout


def test_core_package_imports_without_numpy():
    """#7：core 七个模块在无 numpy 环境下**逐一可导入**（issue #7 的验收脚本）。"""
    proc = _run_without_numpy(textwrap.dedent('''
        import sys
        mods = ("lingshu.core.core", "lingshu.core.verdict",
                "lingshu.core.provenance", "lingshu.core.condition_normalize",
                "lingshu.core.time_core", "lingshu.core.longterm_gate",
                "lingshu.core.activation")
        for mod in mods:
            sys.modules["numpy"] = None
            sys.modules.pop(mod, None)
            try:
                __import__(mod); print("OK", mod)
            except ImportError as e:
                print("BREAK", mod, "->", e)
    '''))
    assert proc.returncode == 0, proc.stderr
    assert "BREAK" not in proc.stdout, proc.stdout
    assert proc.stdout.count("OK ") == 7, proc.stdout


def test_activate_degrades_loudly_without_numpy(monkeypatch, tmp_path):
    """#7：无 numpy 时 `activate()` 显式 not_ready，不抛异常、不静默返回假结果。"""
    store, engine = make_engine(tmp_path, "nonumpy")
    try:
        monkeypatch.setattr(activation, "np", None)
        result = engine.activate("合成查询", workset="ws")
        assert result["status"] == "activation_not_ready", result
        assert result["size"] == 0 and result["seeds"] == 0 and result["top"] == []
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# #161 空查询 / 纯空白查询
# --------------------------------------------------------------------------- #
def _fill_unrelated(store, n=20):
    """灌入 n 个与查询无关的节点（表序在前，旧实现下会被全量满分种入）。"""
    for i in range(n):
        add_node(store, "n%02d" % i, "无关内容%d" % i)
    add_node(store, "cat", "猫在睡觉")


@pytest.mark.parametrize("query", ["", " ", "   ", "\t\n"])
def test_empty_query_seeds_nothing(query, tmp_path):
    """#161：空/纯空白查询无可投票词元 ⇒ seeds=0、无任何满分成员。"""
    store, engine = make_engine(tmp_path, "empty161")
    try:
        _fill_unrelated(store)
        result = engine.activate(query, workset="s")
        assert result["seeds"] == 0, (
            "空查询 repr(%r) 种入 %d 个种子：%r（旧实现下 `\"\" in hay` 恒真）"
            % (query, result["seeds"], result["top"][:3]))
        assert result["size"] == 0, result["top"][:3]
        assert all(score != 1.0 for _, score in result["top"])
        assert engine.carry_vector("s") == {}
    finally:
        store.close()


def test_empty_query_does_not_promote_unrelated_nodes(tmp_path):
    """#161 可观测后果：空查询**不得**把按表序前 top_k 个无关节点写进工作态。"""
    store, engine = make_engine(tmp_path, "empty161b")
    try:
        _fill_unrelated(store)
        engine.activate("猫", workset="s")
        assert engine.carry_vector("s") == {"cat": 1.0}
        engine.activate("", workset="s")
        assert engine.carry_vector("s") == {}, (
            "空查询覆盖了上一轮工作态：%r" % (engine.carry_vector("s"),))
    finally:
        store.close()


def test_normal_query_seed_scores_unchanged(tmp_path):
    """#161 防误杀：非空查询的种子分数与修复前一致（词元集合未被动过）。

    「灵枢认知图上下文」整串与全部二元组都命中同内容节点 ⇒ s = 1.0；
    不相关节点得 0（不入种子）。
    """
    store, engine = make_engine(tmp_path, "empty161c")
    try:
        add_node(store, "a", "灵枢认知图上下文")
        add_node(store, "b", "激活工作态演化")
        assert dict(engine._seed_by_text("灵枢认知图上下文")) == {"a": 1.0}
        assert engine._seed_by_text("") == []
        assert engine._seed_by_text("   ") == []
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# #293 接不上默认后端 / 默认路径
# --------------------------------------------------------------------------- #
def test_activation_reads_engine_memory_store(tmp_path):
    """#293 核心：`store=` 复用引擎连接，`:memory:` 后端的节点对激活引擎可见。"""
    engine_mem = SpacetimeMemoryEngine()          # 默认 db_path=":memory:"
    try:
        node = engine_mem.add_perception("灵枢认知图上下文")
        act = activation.ActivationEngine(
            store=engine_mem.store, audit_path=str(tmp_path / "a.jsonl"))
        result = act.activate("认知图", workset="w")
        assert node.id in [m[0] for m in result["top"]], (
            "激活引擎读不到引擎 :memory: 后端的节点：%r" % (result["top"],))
        # 借用连接不得被引擎的 close() 关掉（调用方仍可用）
        assert engine_mem.store.get_node(node.id) is not None
        assert act.activate("认知图", workset="w")["size"] >= 1
    finally:
        engine_mem.store.close()


def test_memory_path_without_shared_connection_fails_loudly(tmp_path):
    """#293：只给 `:memory:` 而不共享连接 ⇒ 显式 ValueError，不再静默拿空库。"""
    with pytest.raises(ValueError) as excinfo:
        activation.ActivationEngine(":memory:", str(tmp_path / "a.jsonl"))
    assert ":memory:" in str(excinfo.value)


def test_missing_parent_dir_is_created(tmp_path):
    """#293：默认相对路径在干净目录下构造即崩 ⇒ 修后自动建父目录。

    （裸激活库没有 `nodes` 表，故本件只验构造与建目录面，不调 `activate()`。）
    """
    target = tmp_path / "data" / "nested" / "aeis_memory.db"
    assert not target.parent.exists()
    activation.ActivationEngine(str(target), str(tmp_path / "a.jsonl"))
    assert target.exists(), "父目录未创建（旧实现：unable to open database file）"


def test_conn_kwarg_shares_caller_connection(tmp_path):
    """#293 正对照：`conn=` 亦可共享（`store=` 之外的第二入口）。"""
    store = LayeredStore(str(tmp_path / "c.db"))
    try:
        add_node(store, "a", "合成源锚")
        engine = activation.ActivationEngine(
            conn=store.conn, audit_path=str(tmp_path / "a.jsonl"))
        assert engine.activate("合成源锚", workset="w")["seeds"] == 1
        # 引擎用完不得关掉调用方连接
        assert store.get_node("a") is not None
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# #314 source / hop 落表
# --------------------------------------------------------------------------- #
def test_seed_and_hop_recorded(tmp_path):
    """#314：a→b→c（causal），查询命中 a ⇒ ('seed',0)/('propagate',1)/('propagate',2)。"""
    store, engine = make_engine(tmp_path, "prov314")
    try:
        add_node(store, "a", "灵枢认知图上下文")
        add_node(store, "b", "完全无关的天气记录")
        add_node(store, "c", "另一条毫不相干的流水账")
        add_edge(store, "e1", "a", "b", EdgeType.CAUSAL)
        add_edge(store, "e2", "b", "c", EdgeType.CAUSAL)
        engine.activate("认知图", workset="w", hops=2)
        snap = {m["node_id"]: m for m in engine.export_workset("w")["members"]}
        got = {k: (snap[k]["source"], snap[k]["hop"]) for k in ("a", "b", "c")}
        assert got == {"a": ("seed", 0), "b": ("propagate", 1), "c": ("propagate", 2)}, (
            "落表元数据失真：%r（旧实现下三项均为 ('propagate', 0)）" % (got,))
    finally:
        store.close()


def test_self_condition_label_kept(tmp_path):
    """#314：自条件回注节点记 `self_condition`（标签行为不变），hop 为 0。

    第 1 轮 `hops=0` 只把种子 `a` 写进工作态；第 2 轮开自条件 ⇒ `a` 被回注
    ⇒ source 覆盖为 `self_condition`；新扩散到的 `b` 记 `propagate` / hop 1。
    """
    store, engine = make_engine(tmp_path, "prov314b")
    try:
        add_node(store, "a", "合成源锚")
        add_node(store, "b", "关联目标")
        add_edge(store, "e", "a", "b")
        engine.activate("合成源锚", workset="w", hops=0)
        assert engine.carry_vector("w") == {"a": 1.0}
        engine.activate("合成源锚", workset="w", hops=1, self_condition=0.5)
        snap = {m["node_id"]: m for m in engine.export_workset("w")["members"]}
        assert snap["a"]["source"] == "self_condition", snap["a"]
        assert snap["a"]["hop"] == 0
        assert snap["b"]["source"] == "propagate", snap["b"]
        assert snap["b"]["hop"] == 1
    finally:
        store.close()


def test_import_workset_roundtrip_preserves_path(tmp_path):
    """#314：快照往返后仍能区分种子与 2 跳联想（审计面可用）。"""
    store, engine = make_engine(tmp_path, "prov314c")
    try:
        add_node(store, "a", "灵枢认知图上下文")
        add_node(store, "b", "完全无关的天气记录")
        add_node(store, "c", "另一条毫不相干的流水账")
        add_edge(store, "e1", "a", "b", EdgeType.CAUSAL)
        add_edge(store, "e2", "b", "c", EdgeType.CAUSAL)
        engine.activate("认知图", workset="w", hops=2)
        snapshot = engine.export_workset("w")
        engine.import_workset(snapshot, workset="w2")
        got = {m["node_id"]: m["hop"] for m in engine.export_workset("w2")["members"]}
        assert got == {"a": 0, "b": 1, "c": 2}, got
    finally:
        store.close()


# --------------------------------------------------------------------------- #
# #53 cond_match 接线
# --------------------------------------------------------------------------- #
def test_cond_match_actually_gates_propagation(backend, tmp_path):
    """#53 核心：条件空间完全不相交的两端 ⇒ 边衰减为 0 ⇒ 目标**不进**工作态。

    夹具只让 cond_match 机制生效：边类型 `similar`（base 0.75），单源单边、单跳，
    与 #229（opposite）/ #280（多源 Σ）无关。旧实现（cond_match 未接线）下
    target 会以 0.75 出现在工作态。
    """
    store, engine = make_engine(tmp_path, "cond53")
    try:
        cs_a = ConditionSpace("位置A", "工具A", (100.0, 101.0), "约束A")
        cs_b = ConditionSpace("位置B", "工具B", (900000.0, 900100.0), "约束B")
        add_node(store, "A", "合成源锚", condition=cs_a)
        add_node(store, "B", "目标节点", condition=cs_b)
        add_edge(store, "e", "A", "B", EdgeType.SIMILAR, condition=cs_a)
        result = engine.activate("合成源锚", workset="w", hops=1)
        assert result["seeds"] == 1
        assert engine.carry_vector("w") == {"A": 1.0}, (
            "条件不相交的邻居仍被激活：%r（decay 应含 cond_match 因子）"
            % (engine.carry_vector("w"),))
    finally:
        store.close()


def test_cond_match_factor_is_the_only_difference(backend, tmp_path):
    """#53 正对照：两端条件**完全相同** ⇒ cond_mat=1.0，读数回到边 base（0.75）。"""
    store, engine = make_engine(tmp_path, "cond53b")
    try:
        add_node(store, "A", "合成源锚")
        add_node(store, "C", "同条件节点")
        add_edge(store, "e", "A", "C", EdgeType.SIMILAR)
        engine.activate("合成源锚", workset="w", hops=1)
        assert engine.carry_vector("w") == {"A": 1.0, "C": 0.75}
    finally:
        store.close()


def test_cond_match_half_overlap_scales_decay(backend, tmp_path):
    """#53 定量：4 维中 2 维相同（位置/工具同，时间窗远离、存在约束异）
    ⇒ cond_match = 2/4 = 0.5 ⇒ target = 1.0 × 0.75 × 0.5 = 0.375。"""
    store, engine = make_engine(tmp_path, "cond53c")
    try:
        cs_a = ConditionSpace("位置A", "工具A", (100.0, 101.0), "约束X")
        cs_b = ConditionSpace("位置A", "工具A", (900000.0, 900100.0), "约束Y")
        assert activation.cond_match(
            json.loads(cs_a.to_json()), json.loads(cs_b.to_json())) == 0.5
        add_node(store, "A", "合成源锚", condition=cs_a)
        add_node(store, "B", "目标节点", condition=cs_b)
        add_edge(store, "e", "A", "B", EdgeType.SIMILAR, condition=cs_a)
        engine.activate("合成源锚", workset="w", hops=1)
        assert engine.carry_vector("w")["B"] == pytest.approx(0.375, abs=1e-3)
    finally:
        store.close()


def test_cond_match_backends_agree(monkeypatch, tmp_path):
    """#53：cond_match 接线后，稀疏与 edge-list 两后端**同口径**（不因装没装 scipy 变读数）。"""
    if _scipy_issparse is None:
        pytest.skip("SciPy is optional; install it to exercise the sparse backend")
    cs_a = ConditionSpace("位置A", "工具A", (100.0, 101.0), "约束X")
    cs_b = ConditionSpace("位置A", "工具A", (900000.0, 900100.0), "约束Y")
    readings = {}
    for mode in ("scipy", "edge-list"):
        store, engine = make_engine(tmp_path, "cond53d_%s" % mode)
        try:
            add_node(store, "A", "合成源锚", condition=cs_a)
            add_node(store, "B", "目标节点", condition=cs_b)
            add_edge(store, "e", "A", "B", EdgeType.SIMILAR, condition=cs_a)
            if mode == "edge-list":
                monkeypatch.setattr(activation, "csr_matrix", None)
            engine.activate("合成源锚", workset="w", hops=1)
            readings[mode] = engine.carry_vector("w")
        finally:
            store.close()
    assert readings["scipy"] == readings["edge-list"] == {"A": 1.0, "B": 0.375}
