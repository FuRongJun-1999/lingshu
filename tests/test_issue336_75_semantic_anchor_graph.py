# -*- coding: utf-8 -*-
"""守卫 · issue#336 + issue#75 · lingshu/world/semantic_anchor_graph.py

#336 [资源耗尽] infer_relations 逐对做 `any(... for e in self.edges)` 全表扫描，
     整体 O(N²·E)；应改为一次性建边索引（O(E)）+ 逐对 O(1) 查询。
#75  「支撑」判据与定义相反：旧判据 `v_gap < 0.2`（中心高度差小）把等高并排物
     判成「支撑」，而真正垂直叠放的桌子/杯子反被判「相邻」。

本文件钉住「缺陷不再存在」：
  - 支撑仅当垂直叠放（下方顶面 ≈ 上方底面）时成立，且 source=下方物体；
  - 等高并排物一律「相邻」，绝不「支撑」；
  - infer_relations 内不再有逐对全表边扫描（按迭代元素计数为线性）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lingshu.world.semantic_anchor_graph import SemanticAnchorGraph  # noqa: E402


class _CountingList(list):
    """记录被迭代（yield）的元素次数——用于暴露逐对全表扫描。"""

    def __init__(self, *args):
        super().__init__(*args)
        self.iterated = 0

    def __iter__(self):
        for item in super().__iter__():
            self.iterated += 1
            yield item


def _edges_between(g, ca, cb):
    """取类别为 ca/cb 的关系边（无向匹配）。"""
    out = []
    for e in g.edges:
        sc = g.get(e.source).category
        tc = g.get(e.target).category
        if {sc, tc} == {ca, cb}:
            out.append((sc, e.relation, tc, e.attrs))
    return out


def test_support_requires_vertical_stacking():
    """#75：真正叠放的桌子/杯子 → 支撑（下撑上），不得被判「相邻」。"""
    g = SemanticAnchorGraph()
    g.add("table", (0.0, 0.4, 0.0), (1.0, 0.8, 1.0))   # 顶面 y=0.8
    g.add("cup", (0.0, 0.9, 0.0), (0.1, 0.2, 0.1))     # 底面 y=0.8
    g.infer_relations()

    edges = _edges_between(g, "table", "cup")
    rels = [e[1] for e in edges]
    assert "支撑" in rels, f"叠放物未被判支撑: {edges}"
    assert "相邻" not in rels, f"叠放物被误判相邻: {edges}"
    support = next(e for e in edges if e[1] == "支撑")
    # 方向：source=下方物体（table 支撑 cup）
    assert support[0] == "table" and support[2] == "cup", f"支撑方向反了: {support}"


def test_same_height_side_by_side_is_adjacent_not_support():
    """#75：等高并排物 → 相邻，绝不得判「支撑」。"""
    g = SemanticAnchorGraph()
    g.add("chair", (1.0, 0.5, 0.0), (0.5, 1.0, 0.5))
    g.add("box", (1.6, 0.5, 0.0), (0.5, 1.0, 0.5))     # 同高、水平间距 0.6
    g.infer_relations()

    edges = _edges_between(g, "chair", "box")
    rels = [e[1] for e in edges]
    assert "支撑" not in rels, f"等高并排物被误判支撑: {edges}"
    assert "相邻" in rels, f"等高并排物未判相邻: {edges}"


def test_infer_relations_has_no_per_pair_edge_scan():
    """#336：infer_relations 内边表迭代元素数必须线性（不得逐对全表扫描）。"""
    g = SemanticAnchorGraph()
    n = 40
    for k in range(n):
        g.add(f"o{k}", (k * 0.02, 0.5, 0.0))   # 两两距离 < 1.5，全部成边
    g.edges = _CountingList(g.edges)

    added = g.infer_relations()

    assert added == n * (n - 1) // 2, f"应两两成边共 780 条，实得 {added}"
    # 线性上界：新实现只对初始边表扫一次（此处初始为 0）；旧实现为 O(N²·E)
    assert g.edges.iterated <= n, (
        f"infer_relations 迭代边元素 {g.edges.iterated} 次，超过线性上界 {n} "
        f"——存在逐对全表边扫描（O(N²·E)）"
    )
