# -*- coding: utf-8 -*-
"""test_issue787_recall_score_determinism_guard · recall 主排序键确定性守卫（#787）

现象（issue #787）：**同一引擎、同一查询，连续两次 `recall` 产出不同排序**（首元素
不同），候选池相同。#613 的末位确定性键修复**从未对 recall 生效**——`ccae040` 只把
`id` 落在 `search_content` 的三处排序（候选池 SQL ×2 + 终排），recall 自己的终排
（`(-score, -confidence)`）从未补；而更深一层：末位键只救「精确同分」，救不了
「分数自身在漂移」。

根因链（三个符号锚）：
  1. `recall`（core.py）打分 = 0.5·sim + 0.3·importance + 0.2·recency，
     recency = 1/(1+距 `last_access` 的天数)——**读** `last_access`；
  2. recall 的检索腿 `LayeredStore.search_content` 返回前对每个命中节点
     `increment_access_many(..., touch=开关缺省真)`——**写** `last_access` 为
     「现在」（`executemany` 逐行 `time.time()`）⇒ 同一节点的分数在两次 recall
     之间漂移（issue 实测 ~1.76e-08；本机复现 ~1.9e-08）；
  3. 主排序键（score）不稳定 ⇒ sim 并列组的次序按「上次检索的写回时刻」重排
     （Windows 时钟粒度下首调按创建桶序、次调起按写回后的精确并列 ⇒ 结构性翻转）；
     分数不精确相等 ⇒ #613 的 `id` 末位键没有介入时机。

修复（根修 · 打分路径不写回打分输入）：
  · `LayeredStore.search_content`/`_search_content_fts` 增 `touch` 形参
    （缺省 None → `_retrieval_touches_recency()` 开关，既有调用行为逐位不变）；
  · `recall` 检索腿显式 `touch=False`——检索只计数（access_count），不刷新
    `last_access`；近因只由写入、M5 重复观测、置信度复核、巩固复述等**真实强化**
    事件刷新（与 docs/plans/检索不刷新近因_设计_v0.1.md「检索 ≠ 使用 ≠ 强化」
    同则；`LINGSHU_RETRIEVAL_NO_TOUCH` 开关仍只管直接检索路径）；
  · recall 终排补 #613 同款末位键 `id`（精确同分时的唯一确定序）。

修后语义（如实）：分数本身仍随 wall-clock **整体**微降（recency 对固定
`last_access` 单调衰减，两次调用间 ~5e-09 量级）——但组内两两差值的**符号**不变
（输入固定），故**次序**跨调用逐位稳定。本守卫锁次序（issue 的验收口径：连跑
10 次 id 列表逐位相同），**不**锁「分数逐位不变」。

本守卫锁住的性质：
  · 同引擎同查询连跑 **10 次**（非 2 次——防偶然通过）id 列表逐位相同；
  · 机制面：recall 路径**不写回** `last_access`（打分输入只读），`access_count`
    仍递增（统计语义不变）；
  · 边界正对照：直接检索（`search_content` 默认开关）**仍**写回 `last_access`
    ——证明本修是 recall 病灶的定点收口，不是全局翻转 `_retrieval_touches_recency`
    缺省；
  · 精确同分（`last_access` 完全相同 ⇒ recency 完全相同 ⇒ 分数完全相同）时
    recall 序为 `id` 升序（末位键契约）。

**不**锁住的性质（明示边界）：
  · 跨库（两个全新库）首调次序不保证相同——id 含 `uuid4`、`last_access` 是
    wall-clock，存储态本就不同（与 #613 守卫同边界）；
  · 两次 recall 之间若有**其他**真实强化事件（M5 复观测、置信度复核、复述、
    或直接 `search_content` 检索）改写 `last_access`，则下一次 recall 的分数
    变化是**真实状态变化**，不是本 issue 所指非确定性；
  · 分数值不逐位锁（见上「修后语义」）。

变异哨兵：把 recall 检索腿的 `touch=False` 改回开关缺省
（`self.store.search_content(context_content, limit=50)`），
`test_same_query_10_runs_bit_identical` 与
`test_recall_does_not_touch_last_access` 必红（后者确定性红：写回被直接观测）。
"""
import pytest

from lingshu.core.core import SpacetimeMemoryEngine

pytestmark = pytest.mark.filterwarnings("ignore")

Q = "记忆条目"


def _engine(tmp_path):
    return SpacetimeMemoryEngine(str(tmp_path / "e.db"))


def _add(e, n, importance_of, content_of):
    for i in range(n):
        e.add_perception(content_of(i), importance=importance_of(i), skip_dedup=True)


def _ids(e, limit):
    return [n.id for n, _ in e.recall(Q, limit=limit)]


def _last_access(e):
    return {r[0]: r[1] for r in e.store.conn.execute(
        "SELECT id, last_access FROM nodes").fetchall()}


def _access_counts(e):
    return {r[0]: r[1] for r in e.store.conn.execute(
        "SELECT id, access_count FROM nodes").fetchall()}


def test_same_query_10_runs_bit_identical(tmp_path):
    """issue 验收口径：同引擎同查询连跑 10 次 recall，id 列表逐位相同。

    样本取 issue 复现同族（非零填充序号 ⇒ 1/2 位数字两组 sim 并列组，组内次序
    完全交由近因项与末位键裁决——最苛刻面）。10 次而非 2 次：防「前两次恰好
    相同」的偶然通过。"""
    e = _engine(tmp_path)
    _add(e, 20, lambda i: 0.5, lambda i: "%s %d：内容各不相同" % (Q, i))
    seq = [_ids(e, 10) for _ in range(10)]
    assert len(seq) == 10 and all(seq), "10 次调用应各返回非空结果"
    assert all(s == seq[0] for s in seq), (
        "同查询连跑 10 次出现不同顺序：唯一序数 %d" % len({tuple(s) for s in seq}))


def test_recall_does_not_touch_last_access(tmp_path):
    """机制面：recall 路径不写回 `last_access`（打分输入只读），
    `access_count` 仍递增（统计语义不变）。"""
    e = _engine(tmp_path)
    _add(e, 12, lambda i: 0.5, lambda i: "%s %02d" % (Q, i))
    before_la = _last_access(e)
    before_ac = _access_counts(e)
    hits = _ids(e, 20)
    after_la = _last_access(e)
    after_ac = _access_counts(e)
    assert before_la == after_la, (
        "recall 检索路径写回了 last_access——打分输入被自己改写（#787 根因复发）")
    assert any(after_ac[nid] > before_ac.get(nid, 0) for nid in hits), (
        "recall 命中未计数 access_count——检索统计语义不应随本修丢失")


def test_direct_search_still_touches_last_access(tmp_path):
    """边界正对照：直接检索（默认开关）仍写回 `last_access`——本修是 recall
    病灶的定点收口（`touch=False` 只由 recall 传入），不是全局翻转缺省开关。"""
    e = _engine(tmp_path)
    _add(e, 12, lambda i: 0.5, lambda i: "%s %02d" % (Q, i))
    before = _last_access(e)
    e.search_content(Q, limit=20)
    after = _last_access(e)
    assert any(before[k] != after[k] for k in before), (
        "直接检索未观察到 last_access 写回——缺省开关被意外翻转（超出 #787 边界）")


def test_exact_tie_order_falls_to_id(tmp_path):
    """末位键契约：`last_access` 完全相同 ⇒ recency 完全相同 ⇒ 分数精确相等
    ⇒ recall 序必须按 `id` 升序（唯一确定序，#613 同款末位键落到 recall 终排）。"""
    e = _engine(tmp_path)
    _add(e, 10, lambda i: 0.5, lambda i: "%s %02d" % (Q, i))
    same_la = 1700000000.5
    e.store.conn.execute("UPDATE nodes SET last_access=?", (same_la,))
    e.store.conn.commit()
    got = _ids(e, 10)
    all_ids = sorted(r[0] for r in e.store.conn.execute(
        "SELECT id FROM nodes").fetchall())
    assert got == all_ids, (
        "精确同分时应按 id 升序，实得 %r" % [g[:14] for g in got])
