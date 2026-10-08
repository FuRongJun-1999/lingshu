# -*- coding: utf-8 -*-
"""test_issue145_causal_depth_cycles · lingshu issue #145 守卫
============================================================================
缺陷（lingshu issue #145 · 报告人 WASD258-jpg；基线 c7cc588）：
  ① reason_causal（不传 end_id）到 max_depth 直接 return 不存链、路径无 visited
     ⇒ 7 步链在 max_depth=5 下返回 0 条；能走到环的分支全弃；有环有分支时
     在深度内反复绕圈，指数耗时
  ② find_cycles 条件 len(path) >= 2 漏 2 环（A⇄B）与自环；visited 不含起点；
     同一环被每个节点重复报告；无剪枝、每步一次 SQL
  ③ self_check 每次整趟 find_cycles(max_depth=8)，继承上述 ⇒ 45 节点约 29s 量级
修法：reason_causal 到深度预算仍存截断链（truncated=True），路径内 visited、
遇回边/自环存链（cyclic=True）；find_cycles 单次 SQL 建索引＋Tarjan SCC 剪枝＋
环内最小节点起枚举＋环边集去重（覆盖自环与 2 环）；self_check 先三色 DFS
O(V+E) 探测、无环免整趟枚举。

断言组：
  A 组（超深截断）：8 节点 7 步链 max_depth=5 ⇒ 恰 1 条、truncated=True、
     cyclic=False、长度=max_depth、仍是 list（兼容）；max_depth=7 ⇒ 1 条不截断
  B 组（回边带环）：A→B→C→B reason_causal(A) ⇒ ≥1 条、cyclic=True 且尾边
     闭回 B；纯链图照旧产出普通链（两标记皆 False）
  C 组（环检测）：A⇄B + C→C ⇒ find_cycles ≥2（含 2 环与自环）、
     self_check cycles_found ≥2；3 环 A→B→C→A ⇒ 该环恰报 1 次（去重）
  D 组（规模）：9 层×2/3/4/5（18/27/36/45 节点）self_check 均 < 1s
     （旧读数量级：0.02s / 0.43s / 4.51s / 29.3s）
  E 组（语义区分）：无出边、仅非因果出边 ⇒ 空列表（真无因果后果，不得标截断）

运行（lingshu 仓根）：python -X utf8 tests/test_issue145_causal_depth_cycles.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import EdgeType, SpacetimeMemoryEngine  # noqa: E402

_PASS = []
_FAIL = []


def ok(cond, msg, extra=""):
    (_PASS if cond else _FAIL).append(msg)
    print(("  PASS " if cond else "  FAIL ") + msg
          + (("  ← " + str(extra)) if (extra and not cond) else ""))


def mk(e, n):
    """报告人同款节点工厂（独立观察记录，skip_dedup 保独立身份）。"""
    return [e.add_perception(f"事件节点{i}：独立观察记录{i * 7919}",
                             skip_dedup=True).id for i in range(n)]


def edge_ids(chain):
    return frozenset(e.id for e in chain)


def flag(chain, name):
    return getattr(chain, name, None)


# ---------- A 组：超深链 ⇒ 截断链（旧：直接丢弃返回 0 条） ----------

def group_a_truncated():
    e = SpacetimeMemoryEngine(":memory:")
    ids = mk(e, 8)
    for a, b in zip(ids, ids[1:]):
        e.add_edge(a, b, EdgeType.CAUSAL)

    chains5 = e.reason_causal(ids[0], max_depth=5)
    ok(len(chains5) == 1, "A1 7 步链 max_depth=5 ⇒ 恰 1 条（旧：0 条）",
       len(chains5))
    if chains5:
        c = chains5[0]
        ok(isinstance(c, list) and len(c) == 5 and c[-1].target_id == ids[5],
           "A2 截断链仍是 list 且长度=max_depth=5、终点=第 5 跳",
           (type(c).__name__, len(c), c[-1].target_id == ids[5]))
        ok(flag(c, "truncated") is True and flag(c, "cyclic") is False,
           "A3 截断链标 truncated=True / cyclic=False（两标记不混）",
           (flag(c, "truncated"), flag(c, "cyclic")))
    edges_iter = [e_.id for c_ in chains5 for e_ in c_]
    ok(len(edges_iter) == 5, "A4 截断链可照旧迭代/拆解边（兼容用法）",
       len(edges_iter))

    chains7 = e.reason_causal(ids[0], max_depth=7)
    ok(len(chains7) == 1 and flag(chains7[0], "truncated") is False,
       "A5 预算内完整链照旧 1 条且不标截断（原读数=1 保持）",
       [flag(chains7[0], "truncated") if chains7 else None])

    weighted = e.reason_causal(ids[0], max_depth=5, importance_weighted=True)
    ok(len(weighted) == 1 and isinstance(weighted[0], list),
       "A6 importance_weighted 分支对截断链照常排序（兼容）", len(weighted))
    sub = e.reason_causal(ids[0], max_depth=5, include_subgraph=True)
    ok(len(sub) == 1 and isinstance(sub[0]["path"], list),
       "A7 include_subgraph 分支照常装饰（兼容）", len(sub))


# ---------- B 组：回边/环 ⇒ 存链并标 cyclic（旧：整支丢弃） ----------

def group_b_back_edge():
    e = SpacetimeMemoryEngine(":memory:")
    a, b, c = mk(e, 3)
    for s, t in [(a, b), (b, c), (c, b)]:
        e.add_edge(s, t, EdgeType.CAUSAL)
    chains = e.reason_causal(a)
    ok(len(chains) >= 1, "B1 A→B→C→B ⇒ ≥1 条（旧：0 条）", len(chains))
    cyc = [x for x in chains if flag(x, "cyclic") is True]
    ok(len(cyc) >= 1, "B2 带环链标 cyclic=True（可达部分不吞掉）", len(cyc))
    if cyc:
        c0 = cyc[0]
        ok(len(c0) == 3 and c0[-1].target_id == b,
           "B3 带环链含回边（A→B→C→B，尾边闭回 B）",
           (len(c0), c0[-1].target_id == b))
        ok(flag(c0, "truncated") is False,
           "B4 带环链不标截断（回边与截断是两种终点）", flag(c0, "truncated"))

    # 纯链图：照旧产出普通链，两标记皆 False
    e2 = SpacetimeMemoryEngine(":memory:")
    x, y, z = mk(e2, 3)
    for s, t in [(x, y), (y, z)]:
        e2.add_edge(s, t, EdgeType.CAUSAL)
    plain = e2.reason_causal(x)
    ok(len(plain) == 1 and flag(plain[0], "truncated") is False
       and flag(plain[0], "cyclic") is False,
       "B5 无环纯链照旧 1 条、两标记皆 False（兼容）",
       [len(plain), flag(plain[0], "truncated"), flag(plain[0], "cyclic")]
       if plain else [])


# ---------- C 组：环检测覆盖 2 环/自环 + 去重 ----------

def group_c_cycles():
    e = SpacetimeMemoryEngine(":memory:")
    a, b, c = mk(e, 3)
    e_ab = e.add_edge(a, b, EdgeType.CAUSAL)
    e_ba = e.add_edge(b, a, EdgeType.CAUSAL)
    e_cc = e.add_edge(c, c, EdgeType.CAUSAL)
    cycles = e.store.find_cycles()
    ok(len(cycles) >= 2, "C1 A⇄B + C→C find_cycles ≥2 个环（旧：0）", len(cycles))
    sets = [edge_ids(cyc) for cyc in cycles]
    ok(frozenset({e_ab.id, e_ba.id}) in sets,
       "C2 含 2 环 A⇄B（旧条件 len(path)>=2 漏报）", [sorted(s) for s in sets])
    ok(frozenset({e_cc.id}) in sets, "C3 含自环 C→C（旧漏报）")
    rep = e.self_check()
    ok(rep["cycles_found"] >= 2,
       "C4 self_check cycles_found ≥2（旧：0——错误放心信号）",
       rep["cycles_found"])

    # 3 环去重：同一环只报一次（旧：逐节点重复报告）
    e3 = SpacetimeMemoryEngine(":memory:")
    x, y, z = mk(e3, 3)
    r1 = e3.add_edge(x, y, EdgeType.CAUSAL)
    r2 = e3.add_edge(y, z, EdgeType.CAUSAL)
    r3 = e3.add_edge(z, x, EdgeType.CAUSAL)
    key = frozenset({r1.id, r2.id, r3.id})
    same = [cyc for cyc in e3.store.find_cycles() if edge_ids(cyc) == key]
    ok(len(same) == 1, "C5 3 环 A→B→C→A 恰报 1 次（逐节点重复已去重）",
       len(same))

    # 探测与枚举一致：无环图 has_causal_cycle=False
    e4 = SpacetimeMemoryEngine(":memory:")
    p, q = mk(e4, 2)
    e4.add_edge(p, q, EdgeType.CAUSAL)
    ok(e4.store.has_causal_cycle() is False
       and e4.store.find_cycles() == [],
       "C6 无环图：三色探测 False 且 find_cycles 空")


# ---------- D 组：规模——self_check 不随规模指数增长 ----------

def group_d_scale():
    readings = []
    for w in (2, 3, 4, 5):
        e = SpacetimeMemoryEngine(":memory:")
        layers = [mk(e, w) for _ in range(9)]
        for x, y in zip(layers, layers[1:]):
            for s in x:
                for t in y:
                    e.add_edge(s, t, EdgeType.CAUSAL)
        t0 = time.time()
        e.self_check()
        dt = time.time() - t0
        readings.append((9 * w, dt))
        print(f"    [D] 9 层×{w}（{9 * w} 节点，{8 * w * w} 边）"
              f"self_check {dt:.3f}s")
    ok(all(dt < 1.0 for _, dt in readings),
       "D1 18/27/36/45 节点 self_check 均 < 1s（旧量级 0.02/0.43/4.51/29.3s）",
       readings)


# ---------- E 组：语义区分——空列表=真无因果后果 ----------

def group_e_semantics():
    e = SpacetimeMemoryEngine(":memory:")
    a, b = mk(e, 2)
    ok(e.reason_causal(a) == [], "E1 无出边 ⇒ 空列表（真无因果后果）")
    e.add_edge(a, b, EdgeType.SIMILAR)
    ok(e.reason_causal(a) == [],
       "E2 仅非因果出边 ⇒ 空列表（默认仅 CAUSAL）")
    ok(len(e.reason_causal(a, relation_types=["similar"])) == 1,
       "E3 relation_types 过滤路径照常可查（similar 链 1 条）")

    e2 = SpacetimeMemoryEngine(":memory:")
    ids = mk(e2, 8)
    for s, t in zip(ids, ids[1:]):
        e2.add_edge(s, t, EdgeType.CAUSAL)
    deep = e2.reason_causal(ids[0], max_depth=5)
    ok(len(deep) > 0 and flag(deep[0], "truncated") is True,
       "E4 超深链非空且标截断（空列表不再混同「链太长」）",
       [len(deep), flag(deep[0], "truncated")] if deep else [])


def main():
    print("== A 组：超深链 ⇒ 截断链（不丢弃） ==")
    group_a_truncated()
    print("== B 组：回边/环 ⇒ 存链并标 cyclic（不吞掉可达部分） ==")
    group_b_back_edge()
    print("== C 组：环检测覆盖 2 环/自环 + 去重 ==")
    group_c_cycles()
    print("== D 组：规模——self_check < 1s（45 节点） ==")
    group_d_scale()
    print("== E 组：语义区分——空列表=真无因果后果 ==")
    group_e_semantics()
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（issue #145 守卫：截断链/带环链可区分；2 环与自环可检出；"
          "self_check 不随规模指数增长）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
