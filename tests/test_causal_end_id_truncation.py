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


_SEQ = [0]


def mk(e, n):
    """节点工厂：**每次调用内容唯一**（全局递增序号）。

    注意：不能用「固定模板 + 固定乘数」（如 `i * 7919`）——多次调用会生成重复
    内容，而 `add_perception` 的 M5 去重会把同内容折叠成**同一个节点**
    （实测：同内容两次调用返回同一 id）。那会让「孤立 end」意外等于链上节点，
    使测试构造静默失真而非被测逻辑出错。
    """
    out = []
    for _ in range(n):
        _SEQ[0] += 1
        out.append(e.add_perception(f"节点#{_SEQ[0]} 独立观察记录",
                                    skip_dedup=True).id)
    return out


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


# ---------- F 组：假阳性回归（死胡同不得记截断） ----------
# 缺陷（本 PR 首版引入，对抗性复审 A6 发现）：截断分支未先判「当前节点仍有出边」，
# 于是把**死胡同**（天然终点、路径已被完整走完）也记成截断证据，
# 使「空列表=确无因果」被证伪。
#
# 判据的核心区别：
#   · 到预算**仍有出边**（路径确实没走完）⇒ 记截断 = 合法，
#   · 走到**天然终点**（出边=0）而 end 不在此路径上 ⇒ 不记截断 = 修复点。

def group_f_no_false_truncation():
    # 同一张图（链尾是死胡同），只改预算：md=2 时路径未走完（合法截断），
    # md=3 时走到死胡同（修复点——旧版假阳性）。
    e = SpacetimeMemoryEngine(":memory:")
    c = mk(e, 4)
    for a, b in zip(c, c[1:]):
        e.add_edge(a, b, EdgeType.CAUSAL)
    z = mk(e, 1)[0]                      # 不可达的 end
    ok(len(e.store.get_outgoing_edges(c[3])) == 0,
       "F1 前置：链尾 c[3] 确为天然终点（出边=0）",
       len(e.store.get_outgoing_edges(c[3])))

    st_deep = {}
    r_deep = e.reason_causal(c[0], end_id=z, max_depth=3, causal_stats=st_deep)
    ok(st_deep.get("truncated", 0) == 0 and len(r_deep) == 0,
       "F2 走到死胡同且 end 不可达 ⇒ 无截断证据、空列表（旧版：假阳性 1 条）",
       [len(r_deep), st_deep])

    st_mid = {}
    r_mid = e.reason_causal(c[0], end_id=z, max_depth=2, causal_stats=st_mid)
    if st_mid.get("truncated", 0) > 0:
        ok(all(len(e.store.get_outgoing_edges(x[-1].target_id)) > 0
               for x in r_mid),
           "F3 凡上报的截断证据，链尾必须**仍有出边**（不得是死胡同）",
           [(x[-1].target_id[:8],
             len(e.store.get_outgoing_edges(x[-1].target_id))) for x in r_mid])
    else:
        ok(True, "F3 本预算下无截断证据（同样合法，判据不强制上报）")

    # 正例：链确实未走完 ⇒ 截断证据照常给出（修 A6 不得误伤）
    e2 = SpacetimeMemoryEngine(":memory:")
    d = mk(e2, 4)
    for a, b in zip(d, d[1:]):
        e2.add_edge(a, b, EdgeType.CAUSAL)
    st3 = {}
    e2.reason_causal(d[0], end_id=z, max_depth=1, causal_stats=st3)
    ok(st3.get("truncated", 0) >= 1,
       "F4 到预算仍有出边（路径未走完）⇒ 截断证据照常给出（未误伤正例）", st3)


# ---------- G 组：stats 键必须存在（防「默认值遮蔽」恒真断言） ----------

def group_g_stats_keys():
    e, ids = chain_graph(8)
    st = {}
    e.reason_causal(ids[0], end_id=ids[7], max_depth=5, causal_stats=st)
    ok(all(k in st for k in ("max_depth", "complete", "truncated",
                             "truncated_dropped")),
       "G1 end_id 分支：四个统计键**必须都在**（不允许靠 .get 默认值过）",
       sorted(st.keys()))

    e2 = SpacetimeMemoryEngine(":memory:")
    b = mk(e2, 3)
    e2.add_edge(b[0], b[1], EdgeType.CAUSAL)
    e2.add_edge(b[1], b[2], EdgeType.CAUSAL)
    e2.add_edge(b[2], b[1], EdgeType.CAUSAL)   # A→B→C→B 带环链
    st2 = {}
    r2 = e2.reason_causal(b[0], max_depth=5, causal_stats=st2)
    ok(all(k in st2 for k in ("max_depth", "complete", "truncated",
                              "truncated_dropped", "cyclic")),
       "G2 chains 分支：含 cyclic 键", sorted(st2.keys()))
    ok(st2.get("complete") == 0 and st2.get("cyclic", 0) >= 1,
       "G3 A→B→C→B 不得报 complete（唯一链是 cyclic）——修 cyclic 错算进 complete",
       st2)
    ok(len([c for c in r2 if getattr(c, "cyclic", False)]) == st2.get("cyclic", -1),
       "G4 stats.cyclic 与返回列表里 cyclic 链数一致", st2)


    # importance_weighted=True 时，高 importance 的截断链**不得抢占**首元素
    # （对抗性复审 B2：原排序键忽略了 truncated 位）
    e3 = SpacetimeMemoryEngine(":memory:")
    start = e3.add_perception("起点", skip_dedup=True, importance=0.5).id
    endn = e3.add_perception("终点", skip_dedup=True, importance=0.5).id
    mid = e3.add_perception("中间低重要性", skip_dedup=True, importance=0.01).id
    e3.add_edge(start, mid, EdgeType.CAUSAL)
    e3.add_edge(mid, endn, EdgeType.CAUSAL)          # 完整支：2 跳，低 importance
    hi = [e3.add_perception(f"高重要性{i}", skip_dedup=True,
                            importance=0.99).id for i in range(4)]
    e3.add_edge(start, hi[0], EdgeType.CAUSAL)
    for x, y in zip(hi, hi[1:]):
        e3.add_edge(x, y, EdgeType.CAUSAL)           # 截断支：高 importance
    w = e3.reason_causal(start, end_id=endn, max_depth=2, importance_weighted=True)
    ok(len(w) >= 2 and getattr(w[0], "truncated", False) is False,
       "E6 importance_weighted 下首元素仍是**完整**链（高 importance 截断链不得抢占）",
       [len(w), getattr(w[0], "truncated", None) if w else None])


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
    print("== F 组：假阳性回归 —— 死胡同不得记截断 ==")
    group_f_no_false_truncation()
    print("== G 组：stats 键存在性与 cyclic 归集 ==")
    group_g_stats_keys()
    print()
    print(f"===== SUMMARY {len(_PASS)}/{len(_PASS) + len(_FAIL)} 通过 =====")
    if _FAIL:
        print("FAILS:", _FAIL)
        return 1
    print("VERDICT=PASS（end_id 路径超深 ⇒ 截断证据可见且可审计；"
          "死胡同不误报；空列表=确无因果；预算内读数不变）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
