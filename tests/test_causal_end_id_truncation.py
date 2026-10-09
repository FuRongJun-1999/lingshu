# -*- coding: utf-8 -*-
"""test_causal_end_id_truncation · 「指定终点」因果推理的静默截断守卫
============================================================================
缺陷（与 issue #145 同族，**#145 只修了不传 end_id 的一半**）：
  `SpacetimeMemoryEngine.reason_causal(start, end_id=..., max_depth=5)` 走的是
  `LayeredStore.infer_causal_paths(start, end, max_depth)` 这条**独立实现**。
  该实现到 `max_depth` 直接 `return`（`core.py` 原 `if depth > max_depth`），
  **既不留截断证据、也不告知调用方**：

      start→…→end 共 7 跳，max_depth=5  ⇒ 返回 []（静默）
      start→end    确无因果路径        ⇒ 返回 []（同样）

  ⇒ 两种语义完全不可区分。「有因果链但超出深度预算」被伪装成「确无因果」——
     在因果推理这一最常用的查询上构成**假阴性**，且无任何可观测痕迹。

  对照：不传 end_id 的那条路径（`reason_causal` 的 chains 分支）在 #145 已修为
  「到深度预算仍存截断链并标 `truncated=True`，空列表=确无因果后果」。本守卫
  把**同一口径**钉到传 end_id 的这条路径上——两半收敛为同一语义。

修法（与 #145 同口径，additive）：
  · 到 `max_depth` 仍有出边 ⇒ 该分支是**截断证据**：保留路径并标 `truncated=True`
    （不丢弃、不与「确无因果」混同）；截断证据条数设上界，超出时在 stats 里显式
    报告被丢弃条数——**不静默**。
  · 新增 `max_depth=None`：不限深度（调用方显式要求时才用），用于「必须找到」场合。
  · 新增可选 `causal_stats` 出参：`{max_depth, truncated, complete, truncated_dropped}`
    ——截断事实外传，调用方可判、可审计。

断言组：
  A 组（核心复现）：7 跳链 start→end，max_depth=5 ⇒ 非空且元素标 truncated=True
     （**旧读数=0 条**）；max_depth=7 ⇒ 1 条完整、truncated=False（预算内不改）
  B 组（语义区分）：确无因果（无出边 / 仅非因果出边）⇒ 空列表且 stats.complete==0、
     stats.truncated==0 —— 空列表不再混同「链太长」
  C 组（外传事实）：causal_stats 三数与实际返回一致；截断证据被丢弃时
     truncated_dropped>0（不静默）
  D 组（opt-in 深搜）：max_depth=None ⇒ 7 跳链可完整找到（且不标截断）
  E 组（兼容）：返回元素仍是 list 子类（下标/迭代/len/json 照旧）；
     importance_weighted 与 include_subgraph 分支对截断链照常工作

定点变异自证（人工做，非本脚本自动执行）：
  ① 把 core.py 里截断分支（`if depth >= max_depth` 处 append CausalChain(...,
     truncated=True)）删除、恢复成 `return` ⇒ A 组必红；
  ② 或把 `CausalChain(..., truncated=True)` 改成 `CausalChain(...)`（丢标记）
     ⇒ A1/A2 必红；
  ③ 或把确无因果分支也 append 截断链 ⇒ B 组必红。
  复原后全绿。

运行（lingshu 仓根）：python -X utf8 tests/test_causal_end_id_truncation.py
退出码：0 = 全过；1 = 有断言失败
"""
from __future__ import annotations

import os
import sys

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
    """节点工厂（独立观察记录，skip_dedup 保独立身份）——同 #145 守卫体例。"""
    return [e.add_perception(f"事件节点{i}：独立观察记录{i * 7919}",
                             skip_dedup=True).id for i in range(n)]


def chain_graph(n=8):
    """n 个节点的纯因果链 ids[0]→ids[1]→…→ids[n-1]。"""
    e = SpacetimeMemoryEngine(":memory:")
    ids = mk(e, n)
    for a, b in zip(ids, ids[1:]):
        e.add_edge(a, b, EdgeType.CAUSAL)
    return e, ids


def flag(chain, name):
    return getattr(chain, name, None)


# ---------- A 组：核心复现——超深链必须留下截断证据 ----------

def group_a_truncation_visible():
    e, ids = chain_graph(8)          # 7 跳

    stats = {}
    deep = e.reason_causal(ids[0], end_id=ids[7], max_depth=5, causal_stats=stats)
    ok(len(deep) > 0,
       "A1 7 跳链 max_depth=5 传 end_id ⇒ 非空（旧读数=0 条，静默丢弃）",
       len(deep))
    if deep:
        c = deep[0]
        ok(flag(c, "truncated") is True,
           "A2 截断证据标 truncated=True（与确无因果可区分）",
           flag(c, "truncated"))
        ok(isinstance(c, list),
           "A3 返回元素仍是 list 子类（兼容下标/迭代/len）",
           type(c).__name__)
    ok(stats.get("truncated", 0) > 0,
       "A4 causal_stats.truncated>0：截断事实外传（可判、可审计）",
       stats)

    # 预算内：照旧完整命中，不得改读数、不得误标截断
    st2 = {}
    exact = e.reason_causal(ids[0], end_id=ids[5], max_depth=5, causal_stats=st2)
    ok(len(exact) == 1 and flag(exact[0], "truncated") is False,
       "A5 预算内（end=第 5 跳）恰 1 条且 truncated=False（原读数保持）",
       [len(exact), flag(exact[0], "truncated") if exact else None])
    ok(st2.get("complete", 0) == 1 and st2.get("truncated", 0) == 0,
       "A6 预算内 stats.complete==1 且 truncated==0",
       st2)

    wider = e.reason_causal(ids[0], end_id=ids[7], max_depth=7)
    ok(len(wider) == 1 and flag(wider[0], "truncated") is False,
       "A7 放宽预算至 7 ⇒ 完整 1 条、不标截断（截断与预算同源）",
       [len(wider), flag(wider[0], "truncated") if wider else None])


# ---------- B 组：语义区分——空列表=确无因果 ----------

def group_b_semantics():
    e = SpacetimeMemoryEngine(":memory:")
    a, b = mk(e, 2)
    st = {}
    ok(e.reason_causal(a, end_id=b, max_depth=5, causal_stats=st) == []
       and st.get("complete", 0) == 0 and st.get("truncated", 0) == 0,
       "B1 无出边 ⇒ 空列表且 complete==truncated==0（真无因果）", st)

    e2 = SpacetimeMemoryEngine(":memory:")
    x, y = mk(e2, 2)
    e2.add_edge(x, y, EdgeType.SIMILAR)
    st2 = {}
    ok(e2.reason_causal(x, end_id=y, max_depth=5, causal_stats=st2) == []
       and st2.get("truncated", 0) == 0,
       "B2 仅非因果出边 ⇒ 空列表且无截断证据（默认仅 CAUSAL）", st2)

    # 同图同预算：到得了=完整、到不了但更深=截断 —— 二者不得同形
    e3, ids = chain_graph(8)
    st_c = {}
    hit = e3.reason_causal(ids[0], end_id=ids[5], max_depth=5, causal_stats=st_c)
    st_t = {}
    cut = e3.reason_causal(ids[0], end_id=ids[7], max_depth=5, causal_stats=st_t)
    ok((len(hit) > 0) and (len(cut) > 0)
       and flag(hit[0], "truncated") is False
       and flag(cut[0], "truncated") is True,
       "B3 同图同预算：命中=完整 / 超深=截断，两种结果形态可区分",
       [len(hit), len(cut), flag(hit[0], "truncated") if hit else None,
        flag(cut[0], "truncated") if cut else None])


# ---------- C 组：外传事实与返回一致、丢弃不静默 ----------

def group_c_stats_fidelity():
    e, ids = chain_graph(8)
    st = {}
    paths = e.reason_causal(ids[0], end_id=ids[7], max_depth=5, causal_stats=st)
    ok(st.get("max_depth") == 5,
       "C1 stats.max_depth 记录实际生效预算", st.get("max_depth"))
    tr = [p for p in paths if flag(p, "truncated") is True]
    ok(st.get("truncated", 0) >= len(tr),
       "C2 stats.truncated 不小于返回的截断证据数（口径一致）",
       [st.get("truncated"), len(tr)])
    ok(isinstance(st.get("truncated_dropped", 0), int)
       and st.get("truncated_dropped", 0) >= 0,
       "C3 stats.truncated_dropped 存在且非负（丢弃量显式，不静默）",
       st.get("truncated_dropped"))

    # 分叉图：多条分支同时超深 ⇒ 截断证据不得静默消失
    e2 = SpacetimeMemoryEngine(":memory:")
    s, t = mk(e2, 2)
    mids = mk(e2, 6)
    e2.add_edge(s, mids[0], EdgeType.CAUSAL)
    e2.add_edge(s, mids[3], EdgeType.CAUSAL)
    for a, b in zip(mids[:3], mids[1:3]):
        e2.add_edge(a, b, EdgeType.CAUSAL)
    e2.add_edge(mids[2], t, EdgeType.CAUSAL)
    st2 = {}
    got = e2.reason_causal(s, end_id=t, max_depth=1, causal_stats=st2)
    ok(st2.get("truncated", 0) > 0,
       "C4 分叉图 max_depth=1 ⇒ 仍有截断证据外传（分支不被静默吞掉）", st2)


# ---------- D 组：opt-in 深搜（max_depth=None） ----------

def group_d_optin_unbounded():
    e, ids = chain_graph(8)
    st = {}
    full = e.reason_causal(ids[0], end_id=ids[7], max_depth=None,
                           causal_stats=st)
    ok(len(full) == 1 and len(full[0]) == 7,
       "D1 max_depth=None ⇒ 7 跳链完整找到（调用方显式要求时不限深度）",
       [len(full), len(full[0]) if full else None])
    ok(full and flag(full[0], "truncated") is False,
       "D2 不限深度时不得标截断（无预算可耗尽）",
       flag(full[0], "truncated") if full else None)


# ---------- E 组：兼容——既有分支与用法照旧 ----------

def group_e_compat():
    e, ids = chain_graph(8)
    weighted = e.reason_causal(ids[0], end_id=ids[7], max_depth=5,
                               importance_weighted=True)
    ok(len(weighted) > 0 and isinstance(weighted[0], list),
       "E1 importance_weighted 分支对截断链照常排序（兼容）", len(weighted))

    sub = e.reason_causal(ids[0], end_id=ids[7], max_depth=5,
                          include_subgraph=True)
    ok(len(sub) > 0 and isinstance(sub[0]["path"], list),
       "E2 include_subgraph 分支照常装饰（兼容）", len(sub))

    # 不传 end_id 的既有路径读数不得被本修影响（#145 口径保持）
    chains = e.reason_causal(ids[0], max_depth=5)
    ok(len(chains) == 1 and flag(chains[0], "truncated") is True,
       "E3 不传 end_id 的 #145 读数保持（1 条截断链）",
       [len(chains), flag(chains[0], "truncated") if chains else None])

    # causal_stats 是可选出参：不传时行为与签名兼容（不抛错）
    plain = e.reason_causal(ids[0], end_id=ids[7], max_depth=5)
    ok(isinstance(plain, list) and len(plain) > 0,
       "E4 causal_stats 不传时照旧可用（可选出参，零侵入）", len(plain))

    # relation_types 过滤在 end_id 路径上照常生效
    e2 = SpacetimeMemoryEngine(":memory:")
    p, q = mk(e2, 2)
    e2.add_edge(p, q, EdgeType.SIMILAR)
    ok(len(e2.reason_causal(p, end_id=q, max_depth=5,
                            relation_types=["similar"])) == 1,
       "E5 relation_types 过滤在 end_id 路径照常生效")


def main():
    print("== A 组：超深链 ⇒ 截断证据（旧：静默 0 条） ==")
    group_a_truncation_visible()
    print("== B 组：语义区分 —— 空列表=确无因果 ==")
    group_b_semantics()
    print("== C 组：截断事实外传、丢弃不静默 ==")
    group_c_stats_fidelity()
    print("== D 组：opt-in 深搜 max_depth=None ==")
    group_d_optin_unbounded()
    print("== E 组：兼容 —— 既有分支与用法照旧 ==")
    group_e_compat()
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（end_id 路径超深 ⇒ 截断证据可见且可审计；"
          "空列表=确无因果；预算内读数不变）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
