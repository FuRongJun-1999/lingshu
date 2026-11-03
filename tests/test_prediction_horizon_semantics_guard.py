# -*- coding: utf-8 -*-
"""test_prediction_horizon_semantics_guard · #2331 horizon 深度语义守卫
============================================================================
#159 家族的契约：horizon 必须**真正驱动**预测步数（world_learner / world_model
的 horizon 失效已有守卫钉住）。`PredictionEngine.predict_routes(start, horizon)`
的 horizon 驱动 `_generate_routes` 的路线展开深度（`if depth >= horizon: return`）
——这是第三个 horizon 消费面，此前**无任何守卫**钉它的深度语义。

变异实测（#2331）：`depth >= horizon` → `depth >= horizon + 1`（horizon 驱动
整体偏移一档，#159 同型回归）后，既有 `test_prediction_engine_guards.py`
18 项全绿（A 组语义通道 / B 组参与范围 / C 组 #395 上界——它们都不比较
**不同 horizon 之间**的路线集）。与 #2281（bench_world 的 horizon **探针**
失明，bench 侧）互补不重叠：本件钉的是**单元守卫**对 PredictionEngine 侧
horizon 语义的判别力。

钉法（运行级，夹具 = 链式图 n0→n1→n2→n3 + 旁路 n0→n4，节点内容互不相关
⇒ 语义通道不贡献候选边，读数纯结构）：
  H1 h=1 精确形态：2 条路线、长度全为 2——(n0,n1) / (n0,n4)；
  H2 h=2 / h=3 精确计数：3 / 4 条（实测 2026-11-04，main 53058c7）；
  H3 计数随 horizon 严格单调：n(1) < n(2) < n(3)（horizon 死掉 ⇒ 平 ⇒ 红）；
  H4 最长路线长度严格单调：2 < 3 < 4（深度截断失效 ⇒ 最长不变 ⇒ 红）。
M43（h 偏移一档）⇒ H1 计数 3 ≠ 2 红；`depth > horizon`（反向差一）⇒ H1 红；
horizon 驱动死掉 ⇒ H3/H4 红。

运行（仓根）：python -X utf8 -m pytest tests/test_prediction_horizon_semantics_guard.py
判据来源：`lingshu/world/prediction.py` `_generate_routes` 深度截断；#159 家族
教训（horizon 只回填不驱动）；#2331 变异实测记录。
"""
import os
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.core.core import (  # noqa: E402
    ConditionSpace, EdgeType, LayeredStore, STEdge, STNode,
)
from lingshu.world.prediction import PredictionEngine  # noqa: E402


class _FakeEngine:
    """最小 engine 门面（同 test_prediction_engine_guards.py 的夹具口径）。"""

    def __init__(self, store):
        self.store = store
        self.rejected: list = []

    def register_rejected_path(self, **kw):
        self.rejected.append(kw)
        return "rej_1"

    def add_perception(self, **kw):
        return type("N", (), {"id": "cond_1"})()


_CONTENTS = {
    "n0": "机房巡检记录表填写规范",
    "n1": "库存月度盘点流程说明",
    "n2": "高速公路养护施工预案",
    "n3": "沿海台风季防汛演练",
    "n4": "员工食堂菜单轮换安排",
}
_EDGES = {"e01": ("n0", "n1"), "e12": ("n1", "n2"),
          "e23": ("n2", "n3"), "e04": ("n0", "n4")}


def _cs():
    return ConditionSpace("测试观测位", "测试工具", (1.0, 2.0), "协议实例运行中")


@pytest.fixture()
def chain_pe():
    d = tempfile.mkdtemp(prefix="horizon_guard_")
    store = LayeredStore(os.path.join(d, "memory.db"))
    for nid, txt in _CONTENTS.items():
        store.add_node(STNode(id=nid, content=txt, modality="text",
                              spatial_coordinates={}, temporal_coordinate=1.0,
                              condition_space=_cs()))
    for eid, (s, t) in _EDGES.items():
        store.add_edge(STEdge(id=eid, source_id=s, target_id=t,
                              relation_type=EdgeType.CAUSAL,
                              condition_space=_cs(), confidence=0.5))
    return PredictionEngine(_FakeEngine(store))


def _routes(pe, horizon):
    """predict_routes 的路线集 → 有序 (路径元组) 列表（容两种返回形态）。"""
    r = pe.predict_routes("n0", horizon=horizon)
    routes = r["routes"] if isinstance(r, dict) and "routes" in r else r
    return sorted(tuple(x["path"]) if isinstance(x, dict) and "path" in x
                  else tuple(x) for x in routes)


def test_horizon_h1_exact_shape(chain_pe):
    """H1：h=1 的精确形态——2 条路线、长度全为 2（M43 偏移一档即 3 条 ⇒ 红）。"""
    paths = _routes(chain_pe, 1)
    assert len(paths) == 2, "h=1 应恰好 2 条路线（n0→n1 / n0→n4），得到 %r" % (paths,)
    assert all(len(p) == 2 for p in paths), "h=1 全部长度 2，得到 %r" % (paths,)
    assert set(paths) == {("n0", "n1"), ("n0", "n4")}, (
        "h=1 路线集应为 {n0→n1, n0→n4}，得到 %r" % (paths,))


def test_horizon_exact_counts_grow(chain_pe):
    """H2：h=2 ⇒ 3 条、h=3 ⇒ 4 条（实测 2026-11-04，main 53058c7）。"""
    assert len(_routes(chain_pe, 2)) == 3, (
        "h=2 应 3 条（+n0→n1→n2），得到 %r" % (_routes(chain_pe, 2),))
    assert len(_routes(chain_pe, 3)) == 4, (
        "h=3 应 4 条（+n0→n1→n2→n3），得到 %r" % (_routes(chain_pe, 3),))


def test_horizon_counts_strictly_monotonic(chain_pe):
    """H3：路线计数随 horizon 严格单调（horizon 驱动死掉 ⇒ 平 ⇒ 红）。"""
    counts = [len(_routes(chain_pe, h)) for h in (1, 2, 3)]
    assert counts[0] < counts[1] < counts[2], (
        "路线计数应随 horizon 严格单调增，实测 %r（#159 同型回归即平）" % (counts,))


def test_horizon_max_len_strictly_monotonic(chain_pe):
    """H4：最长路线长度随 horizon 严格单调（深度截断失效 ⇒ 最长不变 ⇒ 红）。"""
    maxlens = [max(len(p) for p in _routes(chain_pe, h)) for h in (1, 2, 3)]
    assert maxlens[0] < maxlens[1] < maxlens[2], (
        "最长路线长度应随 horizon 严格单调增（2<3<4），实测 %r" % (maxlens,))
