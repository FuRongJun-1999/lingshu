# -*- coding: utf-8 -*-
"""#263 守卫（残余项 A）：induce_concepts 同主题必须**合并**，不得每轮复制新概念。

issue #263 残余项 A：300 轮、3 个主题的输入，旧实现产出 **99 个概念 / 3 主题**
（每主题 33 个互不合并的副本），知识层净膨胀 399 vs 300 条输入 = 1.33×。

判据（三条，都是「缺陷不再存在」而非「能跑」）：
  ① 概念数不随轮数线性增长：300 轮后概念数 ≤ 主题数 × 3（旧实现 99）；
  ② 知识层净膨胀收敛：知识层 ≤ 输入条数 × 1.05（旧实现 1.33×）；
  ③ 已修三条不得回退（回归保护）：维护日志不进记忆图 / 概念不代表日志 / 概念未自抬到 1.0。

★ 判别力：只回滚源码（core.py 的 induce_concepts）而不动本文件 ⇒ 必须变红。
运行: python -X utf8 -m pytest tests/test_issue263_induce_concepts_merge_guard.py -q
"""
import os
import sys
import tempfile
import warnings

_RP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RP)

import pytest  # noqa: E402

warnings.filterwarnings("ignore")

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

ROUNDS = 300
TOPICS = ["用户晨跑路线与配速记录", "用户学日语五十音进度", "用户家里猫咪的饮食习惯"]


@pytest.fixture(scope="module")
def engine():
    """跑 300 轮（与 issue 同口径），返回 (engine, conn)。"""
    e = SpacetimeMemoryEngine(os.path.join(tempfile.mkdtemp(), "m.db"))
    for r in range(1, ROUNDS + 1):
        e.add_perception("%s，第%d次补充：细节编号%d" % (TOPICS[r % 3], r, r * 37),
                         importance=0.5)
        e.run_maintenance_cycle()
    return e, e.store.conn


def _count(conn, where):
    return conn.execute("SELECT count(*) FROM nodes WHERE " + where).fetchone()[0]


def test_concepts_do_not_multiply(engine):
    """① 300 轮后概念数不得随轮数线性增长（旧实现 99）。"""
    _e, c = engine
    concepts = _count(c, "content LIKE '[概念]%'")
    # 每个主题最多留 3 个概念（含合并判据保守误判的余量），3 主题 ⇒ 上界 9
    assert concepts <= len(TOPICS) * 3, (
        "概念数 %d 超过上界 %d：同主题仍在每轮复制新概念（#263 残余项 A 未修）"
        % (concepts, len(TOPICS) * 3))
    # 且必须至少归纳出概念（不能把功能改没了）
    assert concepts >= len(TOPICS), (
        "只产出 %d 个概念，少于主题数 %d —— 归纳功能被改坏"
        % (concepts, len(TOPICS)))


def test_knowledge_layer_no_net_inflation(engine):
    """② 知识层不得净膨胀（旧实现 399 vs 300 = 1.33×）。"""
    _e, c = engine
    kn = _count(c, "layer = 'knowledge'")
    user = _count(c, "content LIKE '用户%'")
    assert user == ROUNDS, "输入条数应恰为 %d，实际 %d" % (ROUNDS, user)
    ratio = kn / float(user)
    assert ratio <= 1.05, (
        "知识层 %d / 输入 %d = %.2f×，仍在净膨胀（#263 残余项 A 未修）"
        % (kn, user, ratio))


def test_concepts_reference_user_input_not_logs(engine):
    """③a 回归保护：概念不得由维护日志归纳（#263 已修项）。"""
    _e, c = engine
    con_log = _count(c, "content LIKE '[概念]%代表：[consolidation]%'")
    assert con_log == 0, "有 %d 个概念代表维护日志（已修项回退）" % con_log
    logs = _count(c, "content LIKE '[consolidation]%'")
    assert logs == 0, "维护日志重新进了记忆图 %d 条（已修项回退）" % logs


def test_concepts_not_self_promoted(engine):
    """③b 回归保护：概念 importance 不得被自抬到 1.0（#263 已修项）。"""
    _e, c = engine
    rows = [x[0] for x in c.execute(
        "SELECT importance FROM nodes WHERE content LIKE '[概念]%'")]
    assert rows, "没有概念节点"
    assert max(rows) < 0.999, (
        "概念 importance 最大值 %.2f —— 自抬到 1.0（已修项回退）" % max(rows))


def test_recall_not_flooded_by_concepts(engine):
    """③c 回归保护：召回不得被概念副本占满（#263 已修项）。"""
    e, _c = engine
    hits = e.recall("用户学日语五十音进度", limit=5)
    assert hits, "召回为空"
    n_con = sum(1 for m, _s in hits if m.content.startswith("[概念]"))
    assert n_con == 0, "召回前 5 里有 %d 条概念副本（已修项回退）" % n_con
