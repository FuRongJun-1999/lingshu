# -*- coding: utf-8 -*-
"""test_issue252_self_check_cycle_enum_limit_guard · lingshu issue #252 守卫
============================================================================
缺陷（lingshu issue #252 · 基线 2bb8291）：
  `self_check` 里 `has_causal_cycle()` 只是**前置门**——图有环时返回 True，随后
  仍调用 `find_cycles(max_depth=8)` **全量枚举**。`find_cycles` 无上限，稠密 SCC
  的环数随节点数指数增长：issue 实测 n=10 ⇒ 749193 条 / 2.77s / 676MB；
  n=11 ⇒ 3355693 条 / 13.69s / 3041MB（n=12 预计 >1 分钟、>13 GB），
  而 `self_check` 的 `cycle_details` 只用 `cycles[:5]`。

修法（最小 · 按 issue 建议 1/2）：
  `find_cycles` 新增 `limit`（枚举上限，达限提前终止）；`self_check` 传
  `SELF_CHECK_CYCLE_ENUM_LIMIT = 5`。`limit=None`（默认）保持旧语义不变 ⇒
  `tests/test_issue145_causal_depth_cycles.py` 与
  `tests/test_issue215_self_check_cycle_guard.py` 的既有断言不受影响。
  `cycles_found` 语义不变（已枚举环数，现至多 limit 条）；新增 `cycles_capped`
  （已枚举数达到 limit ⇒ 图中环可能更多，报告数不完整）——与 #215 的
  `cycles_truncated`（探测到环但一条都没枚举出，环长超 max_depth）区分。

判据来源：公开 issue #252「建议修复 1」（「`find_cycles` 增加 `limit`（或改成
生成器），`self_check` 取前 5 条明细」）＋ 建议 2（「默认调用（不传 limit）保持
现有语义，不破坏 #145 的守卫」）＋ 其复现实读（n=10/11 的耗时与环数）＋ 分诊表
`triage_lingshu_ALL.md` 行 252（「只有无环图才真走 O(V+E)；稠密 SCC 耗时/内存随
环数指数增长（report 只用前 5）」）。仓内无规定该预算的理论章节，属经验标定。
边界（削掉的判别力，如实标注）：`cycles_found` 仍是 int（未按 issue 建议改成
`"≥N"` 字符串——会波及 #145 C4 的 `>=2` 数值断言）；「报告数不完整」改由新增
`cycles_capped` 承载。

断言组（把修复改回 `find_cycles(max_depth=8)` 无 limit ⇒ Q1/Q2/Q3 红）：
  Q1 枚举有界：11 节点完全有向图 `self_check()["cycles_found"]` 必须 ≤ 5
     （旧形态：3355693 条）。
  Q2 耗时与内存有界：同一图 `self_check()` **CPU 时间 < 1s**，**仅串行环境判**
     （issue 建议 3；旧形态 13.69s。v2：墙钟与 CPU 时间在 `pytest -n auto` 满载下
     实测都会被争抢抬高——0.02–0.03s 串行实测 vs 11.85s 满载墙钟 ⇒ 非受控测量
     不可判，xdist 下 skipif；CI 串行跑，Q2 在 CI 全效力）。
  Q3 截断可观测：`cycles_capped is True`（达到上限）且 `cycle_details` 恰 5 条；
     真无环图 `cycles_capped is False`。
  Q4 非回归：`find_cycles()` 默认（不传 limit）语义不变——A⇄B + 自环仍 ≥2 个环；
     3 环仍恰报 1 次（#145 C 组读数）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue252_self_check_cycle_enum_limit_guard.py -q --no-header
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from lingshu.core.core import EdgeType, SpacetimeMemoryEngine  # noqa: E402

LIMIT = SpacetimeMemoryEngine.SELF_CHECK_CYCLE_ENUM_LIMIT  # 5


def _complete_digraph(n: int):
    """n 节点完全有向图（无自环）——单一大 SCC，环数指数增长（issue 同款构造）。"""
    e = SpacetimeMemoryEngine(":memory:")
    ids = [e.add_perception("因素%d：变量编号%d号" % (i, i), skip_dedup=True).id
           for i in range(n)]
    for a in ids:
        for b in ids:
            if a != b:
                e.add_edge(a, b, EdgeType.CAUSAL, confidence=0.6)
    return e


def test_q1_enumeration_is_bounded():
    """Q1：11 节点完全有向图不得枚举出 335 万条环。"""
    e = _complete_digraph(11)
    rep = e.self_check()
    assert rep["cycles_found"] <= LIMIT, (
        "#252：self_check 仍全量枚举——cycles_found=%r（旧形态 3355693 条）"
        % rep["cycles_found"])
    assert rep["cycles_found"] >= 1, "前置：完全有向图必有环"


@pytest.mark.skipif(
    os.environ.get("PYTEST_XDIST_WORKER") is not None,
    reason="时间量具只在受控（串行）环境有效：`-n auto` 满载下本进程 CPU 时间也会被"
           "缓存/内存带宽争抢抬高（实测串行 0.02–0.03s、满载 >1s）——非受控测量不可判。"
           "CI（gate.yml）串行跑，Q2 在 CI 全效力；本地并行下 Q1/Q3/Q4 仍守语义。")
def test_q2_time_bounded():
    """Q2：同一图 self_check 计算量有界（issue 建议 3；旧形态 13.69s）。

    量具口径（v2）：串行环境下 `time.process_time()`（本进程 CPU），容差 1.0s——
    对当前形态（串行实测 0.02–0.03s）≥30 倍余量，对旧形态（约 13.7s）>10 倍
    判别力。墙钟在 `-n auto` 满载下实测从 0.02s 膨胀到 11.85s（24 逻辑核），
    任何墙钟容差不可判；CPU 时间在同样满载下也被片上停顿抬高过 1s——故并行
    环境整体跳过（见 skipif 理由），只在串行/CI 判定。
    """
    e = _complete_digraph(11)
    c0 = time.process_time()
    rep = e.self_check()
    dt = time.process_time() - c0
    assert dt < 1.0, (
        "#252：稠密 SCC 上 self_check 仍指数耗时——CPU %.2fs（旧形态 13.69s），"
        "cycles_found=%r" % (dt, rep["cycles_found"]))


def test_q3_capped_is_observable():
    """Q3：达到上限须可观测（cycles_capped + cycle_details 恰 limit 条）。"""
    e = _complete_digraph(11)
    rep = e.self_check()
    assert rep["cycles_capped"] is True, (
        "#252：达到枚举上限但 cycles_capped 未置位——report=%r"
        % {k: rep.get(k) for k in ("cycles_found", "cycles_capped")})
    assert len(rep["cycle_details"]) == LIMIT, (
        "#252：cycle_details 应恰为 limit 条——%r" % len(rep["cycle_details"]))
    assert rep["cycles_truncated"] is False, (
        "#252：capped（环太多）与 truncated（环长超 max_depth）不得混同")

    # 真无环图：capped 必须为 False
    e2 = SpacetimeMemoryEngine(":memory:")
    ids = [e2.add_perception("链节点%d：独立记录%d" % (i, i * 7919),
                             skip_dedup=True).id for i in range(5)]
    for a, b in zip(ids, ids[1:]):
        e2.add_edge(a, b, EdgeType.CAUSAL)
    rep2 = e2.self_check()
    assert rep2["cycles_found"] == 0 and rep2["cycles_capped"] is False, (
        "#252：真无环图被误标 capped——report=%r" % rep2)


def test_q4_default_limit_unchanged():
    """Q4：非回归——`find_cycles()` 默认不传 limit 时语义不变（#145 C 组读数）。"""
    e = SpacetimeMemoryEngine(":memory:")
    a, b, c = [e.add_perception("事件节点%d：独立观察记录%d" % (i, i * 7919),
                                skip_dedup=True).id for i in range(3)]
    e_ab = e.add_edge(a, b, EdgeType.CAUSAL)
    e_ba = e.add_edge(b, a, EdgeType.CAUSAL)
    e_cc = e.add_edge(c, c, EdgeType.CAUSAL)
    cycles = e.store.find_cycles()
    sets = [frozenset(x.id for x in cyc) for cyc in cycles]
    assert len(cycles) >= 2, (
        "#252 非回归：find_cycles() 默认调用语义被改——cycles=%r" % len(cycles))
    assert frozenset({e_ab.id, e_ba.id}) in sets and frozenset({e_cc.id}) in sets, (
        "#252 非回归：2 环/自环漏报——%r" % [sorted(s) for s in sets])

    e3 = SpacetimeMemoryEngine(":memory:")
    x, y, z = [e3.add_perception("环节点%d：独立记录%d" % (i, i * 7919),
                                 skip_dedup=True).id for i in range(3)]
    r1 = e3.add_edge(x, y, EdgeType.CAUSAL)
    r2 = e3.add_edge(y, z, EdgeType.CAUSAL)
    r3 = e3.add_edge(z, x, EdgeType.CAUSAL)
    key = frozenset({r1.id, r2.id, r3.id})
    same = [cyc for cyc in e3.store.find_cycles()
            if frozenset(x_.id for x_ in cyc) == key]
    assert len(same) == 1, "#252 非回归：3 环应恰报 1 次——%r" % len(same)
