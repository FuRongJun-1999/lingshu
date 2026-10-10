# -*- coding: utf-8 -*-
"""prediction 引擎守卫（本组 #40 / #117 / #395）
============================================================================
本组三条 P1 缺陷都落在 `lingshu/world/prediction.py`：

  #40（语义通道整体静默失效）
      `_similarity` 旧实现：
          try: from spacetime_memory_core import LayeredStore
          except Exception: return 0.0
          ...
          return LayeredStore.char_bigram_jaccard(a.content, b.content)
      私域（AEIS）把 `spacetime_memory_core` 以命名空间别名注册成**同一** LayeredStore；
      本仓形态没有该别名 ⇒ 异常分支 `return 0.0` 恒定命中 ⇒ 通道4（语义邻近）
      在本仓**整体失效**（`semantic_neighbors` 恒空、`_branch_candidates` 永远
      拿不到 structural/preference 候选）。修法：异常分支回落到**仓内**
      `..core.core.LayeredStore`（`char_bigram_jaccard` 与私域实现逐字同式）。

  #117（预测反馈写回不对称 · 命中即把目标节点**全部**因果入边永久标 verified）
      旧实现命中支：
          for e in self.engine.store.get_incoming_edges(predicted_node_id):
              if e.relation_type.value in ("causal","sequential"):
                  self.engine.store.verify_edge(e.id, min(1.0, e.confidence + 0.05))
      遍历的是**目标节点的全部因果入边**，与"本轮哪条边真正参与了预测"无关；
      且无条件 `verified=1`（`verify_edge` 的 SQL 无 verified 判据）⇒ 一次命中
      把**非参与**边也永久置 verified，此后 `decay_cycle` 的 `WHERE e.verified = 0`
      豁免它们（core.py，本批不改 core）。
      修法（判据来源＝设计者裁定 D-40/D-41，docs/plans/待裁清单_v0.1.md §附-E）：
      `_generate_routes` 记录**本轮实际参与**的边 id（edge 级），命中只强化参与边，
      且只强化**尚未 verified** 者（幂等）；无参与记录时 fail-closed 不强化任何边。

  #395（路线生成资源耗尽 + 判错）
      旧 `dfs` 只有 `if depth >= horizon: return`，且分支循环无 `if nid in path`、
      无总量预算 ⇒ ① 环/自环图上产出 `a→b→a→b…` 这类**带重复节点**的伪路线（判错）；
      ② 路线数按 `max_branches ** horizon` 指数增长（资源耗尽）。
      修法：简单路径去重（`nid in path` 跳过）+ 总量预算 `MAX_ROUTES`。

判据来源（不编造）：
  · #117 修法口径＝设计者裁定 D-40（edge 级参与记录）/ D-41（未命中只不增信＋留痕），
    `docs/plans/待裁清单_v0.1.md` §附-E；缺陷成立依据 `.tmp/verify_pkg3.md` 三节。
  · #40 修法＝「本仓自带模块不该走外部注入」（D-52 同族口径）；`char_bigram_jaccard`
    的**同式**由 `lingshu/core/core.py` 的 `LayeredStore.char_bigram_jaccard` 给出。
  · #395 的 `MAX_ROUTES` 具体数值＝**经验标定，追不到理论出处**（仓内理论稿
    docs/theory/世界模型与语义时空图_完整理论整理与实现路线.md 未给路线规模口径）；
    「必须有界」「简单路径」两条属工程 fail-closed，非理论规定。

削掉的判别力（如实声明）：
  · #395 简单路径生效后，环状因果结构不再产出"绕环一圈"的候选路线；大 horizon
    下候选未来被截到 MAX_ROUTES 条（尾部低分路线不再出现）。
  · #40 修复使语义邻近通道在本仓恢复可用 ⇒ 同一图上的候选边比修复前更多，
    "修复前后路线条数逐字相同"**不成立**（已实测：12 节点图 horizon=3 由 39 → 43）。

定点变异自证（抽掉修复 ⇒ 必红，逐条对应）：
  · #40：`_similarity` 回退支改回 `return 0.0` ⇒ G1 红（semantic_neighbors 恒空）。
  · #117：命中支改回"遍历全部因果入边 + 无条件 verify" ⇒ G2 红（非参与边被 verified）。
  · #395①：删 `if nid in path: continue` ⇒ G3 红（环图上出现重复节点路线）。
  · #395②：删 `if len(routes) >= self.MAX_ROUTES: return` ⇒ G4 红（路线数 > MAX_ROUTES）。

运行（仓根）：python -X utf8 tests/test_prediction_engine_guards.py
             / python -X utf8 -m pytest tests/test_prediction_engine_guards.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.core.core import (  # noqa: E402
    ConditionSpace, EdgeType, LayeredStore, STEdge, STNode,
)
from lingshu.world.prediction import PredictionEngine  # noqa: E402

_PASS: list = []

def ok(cond, msg, extra=""):
    """脚本式报告（记录通过项）；失败即抛 AssertionError（pytest 与直跑同判据）。"""
    if cond:
        _PASS.append(msg)
        print("  PASS " + msg)
    else:
        print("  FAIL " + msg + (("  ← " + str(extra)) if extra else ""))
        raise AssertionError(msg + (("  ← " + str(extra)) if extra else ""))

# ---------------------------------------------------------------- 工具

def _cs():
    return ConditionSpace("测试观测位", "测试工具", (1.0, 2.0), "协议实例运行中")

def _add_node(store, nid, content):
    store.add_node(STNode(
        id=nid, content=content, modality="text", spatial_coordinates={},
        temporal_coordinate=1.0, condition_space=_cs()))

def _add_edge(store, eid, src, dst, conf=0.5, rel=EdgeType.CAUSAL):
    store.add_edge(STEdge(
        id=eid, source_id=src, target_id=dst, relation_type=rel,
        condition_space=_cs(), confidence=conf))

class _FakeEngine:
    """最小 engine 门面：预测引擎只用到 store + 两个记账方法。"""

    def __init__(self, store):
        self.store = store
        self.rejected: list = []

    def register_rejected_path(self, **kw):
        self.rejected.append(kw)
        return "rej_1"

    def add_perception(self, **kw):
        return type("N", (), {"id": "cond_1"})()

def _fresh_store():
    d = tempfile.mkdtemp(prefix="pred_guard_")
    return LayeredStore(os.path.join(d, "memory.db"))

def _engine(store):
    return PredictionEngine(_FakeEngine(store))

# ---------------------------------------------------------------- #40

def test_40_semantic_channel():
    """#40：语义邻近通道在本仓可用（不得恒空）。"""
    store = _fresh_store()
    # 两段内容共享大量中文二元组（Jaccard 显著 > 0.05），且**无** semantic_coordinates
    # ⇒ 只能走 char_bigram_jaccard 回退支（即缺陷所在支）。
    _add_node(store, "a", "预测引擎语义通道在本仓整体静默失效")
    _add_node(store, "b", "预测引擎语义通道在本仓整体失效")
    _add_node(store, "z", "完全无关的另一种内容甲乙丙丁戊己")
    pe = _engine(store)

    sim = pe._similarity(store.get_node("a"), store.get_node("b"))
    ok(sim > 0.05,
       "G1a `_similarity` 回退支给出非零相似度（旧实现恒 0.0）", f"sim={sim!r}")

    nbrs = [n.id for n in pe.semantic_neighbors("a", k=5)]
    ok("b" in nbrs,
       "G1b `semantic_neighbors` 在本仓返回语义邻居（旧实现恒空）", f"neighbors={nbrs}")
    ok("z" not in nbrs,
       "G1c 低相似节点不入邻近集（通道可用但不乱收）", f"neighbors={nbrs}")

    # 有语义坐标时仍优先走 SemanticSpaceProvider（组件缺失则回退 Jaccard，不崩）
    na = store.get_node("a")
    nb = store.get_node("b")
    na.semantic_coordinates = {"protocol": {"concept": {"x": 1.0}}}
    nb.semantic_coordinates = {"protocol": {"concept": {"x": 1.0}}}
    sim2 = pe._similarity(na, nb)
    ok(isinstance(sim2, float) and sim2 > 0.05,
       "G1d 带语义坐标时不崩且仍给非零相似度", f"sim2={sim2!r}")

# ---------------------------------------------------------------- #117

def _feedback_graph(store):
    """a→b、c→b、d→b 三条因果入边指向 b（b 是预测目标）。"""
    for nid in ("a", "b", "c", "d"):
        _add_node(store, nid, "节点-" + nid)
    _add_edge(store, "eab", "a", "b", conf=0.5)
    _add_edge(store, "ecb", "c", "b", conf=0.5)
    _add_edge(store, "edb", "d", "b", conf=0.5)

def test_117_participation_scope():
    """#117：命中只强化**本轮参与**的边，不刷全部入边。"""
    store = _fresh_store()
    _feedback_graph(store)
    pe = _engine(store)

    # 本轮只预测 a→b（start=a），故只有 eab 参与。
    res = pe.predict_routes(start_id="a", horizon=2)
    paths = [r["path"] for r in res["routes"]]
    ok(any(p[-1] == "b" for p in paths),
       "G2a 预测路线到达 b（前置：eab 参与本轮）", f"paths={paths}")

    pe.update_prediction_feedback("b", "b", hit=True)
    eab, ecb, edb = (store.get_edge(x) for x in ("eab", "ecb", "edb"))
    ok(eab.verified and abs(eab.confidence - 0.55) < 1e-9,
       "G2b 参与边 eab 被强化（verified=1 · conf=0.55）",
       f"verified={eab.verified} conf={eab.confidence}")
    ok(not ecb.verified and abs(ecb.confidence - 0.5) < 1e-9,
       "G2c **非参与**边 ecb 未被刷 verified（旧实现会标 True）",
       f"verified={ecb.verified} conf={ecb.confidence}")
    ok(not edb.verified and abs(edb.confidence - 0.5) < 1e-9,
       "G2d **非参与**边 edb 未被刷 verified（旧实现会标 True）",
       f"verified={edb.verified} conf={edb.confidence}")

    # 幂等：重复反馈不再抬 conf（旧实现每轮 +0.05 且无条件写）
    pe.update_prediction_feedback("b", "b", hit=True)
    ok(abs(store.get_edge("eab").confidence - 0.55) < 1e-9,
       "G2e 重复命中不重复增信（幂等 · conf 仍 0.55）",
       f"conf={store.get_edge('eab').confidence}")

    # fail-closed：从未预测过该节点（无参与记录）⇒ 不得强化任何入边
    store2 = _fresh_store()
    _feedback_graph(store2)
    pe2 = _engine(store2)
    pe2.update_prediction_feedback("b", "b", hit=True)
    others = [store2.get_edge(x).verified for x in ("eab", "ecb", "edb")]
    ok(not any(others),
       "G2f 无参与记录时 fail-closed：不得强化任何入边（旧实现全刷 verified）",
       f"verified={others}")

    # 未命中侧：对称留痕（被拒路径登记）且不惩罚参与边
    store3 = _fresh_store()
    _feedback_graph(store3)
    pe3 = _engine(store3)
    pe3.predict_routes(start_id="a", horizon=2)
    pe3.update_prediction_feedback("b", "d", hit=False, note="落空")
    ok(pe3.engine.rejected,
       "G2g 未命中登记被拒路径（对称留痕）", f"rejected={pe3.engine.rejected}")
    e = store3.get_edge("eab")
    ok(not e.verified and abs(e.confidence - 0.5) < 1e-9,
       "G2h 未命中不惩罚参与边（只不增信 · conf 仍 0.5）",
       f"verified={e.verified} conf={e.confidence}")

# ---------------------------------------------------------------- #395

def test_395_bounds():
    """#395：简单路径去重 + 路线总量预算。"""
    # ---- G3：环图（a→b→a）不得产出带重复节点的伪路线 ----
    store = _fresh_store()
    for nid in ("a", "b"):
        _add_node(store, nid, "环-" + nid)
    _add_edge(store, "x1", "a", "b", conf=0.9)
    _add_edge(store, "x2", "b", "a", conf=0.9)
    pe = _engine(store)
    res = pe.predict_routes(start_id="a", horizon=8, max_branches=5)
    paths = [r["path"] for r in res["routes"]]
    repeats = [p for p in paths if len(set(p)) < len(p)]
    ok(not repeats,
       "G3 环图上无重复节点路线（旧实现产出 a→b→a→b… 伪路线）",
       f"repeats={repeats}")

    # ---- G4：指数增长被总量预算封顶 ----
    store2 = _fresh_store()
    N = 12
    for i in range(N):
        _add_node(store2, "n%d" % i, "节点-%d" % i)
    k = 0
    for i in range(N):
        for j in range(i + 1, min(i + 4, N)):   # 出度 ≤3
            _add_edge(store2, "e%d" % k, "n%d" % i, "n%d" % j, conf=0.9)
            k += 1
    pe2 = _engine(store2)
    counts = {}
    for h in (3, 6, 12):
        r = pe2.predict_routes(start_id="n0", horizon=h, max_branches=5)
        counts[h] = len(r["routes"])
        ok(len(r["routes"]) <= PredictionEngine.MAX_ROUTES,
           "G4 horizon=%d 路线数 ≤ MAX_ROUTES(%d)（旧实现按 max_branches**horizon 爆炸）"
           % (h, PredictionEngine.MAX_ROUTES),
           f"count={len(r['routes'])}")
    ok(counts[12] <= PredictionEngine.MAX_ROUTES,
       "G4b 大 horizon 不随深度指数增长（封顶后不再增长）", f"counts={counts}")
    ok(counts[12] == PredictionEngine.MAX_ROUTES,
       "G4c 封顶生效：大 horizon 恰好停在 MAX_ROUTES", f"counts={counts}")

def main():
    for fn in (test_40_semantic_channel,
               test_117_participation_scope,
               test_395_bounds):
        print("== %s ==" % fn.__name__)
        fn()
    print("\n== 汇总：PASS %d ==" % len(_PASS))
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print("\n== 红：%s ==" % exc)
        sys.exit(1)
