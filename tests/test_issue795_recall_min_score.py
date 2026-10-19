# -*- coding: utf-8 -*-
"""#795 `recall`/`search_content` 无最低分数阈值——返回面 `min_score` 守卫

缺陷（lingshu issue #795，#816 复核判「部分成立」）：`recall` 签名
`(context_content, limit)` 无 `min_score`/`threshold` 参数，完全无关的查询也
返回满额结果（改前读数：5 条无关文章 + 无关查询 ⇒ 5/5 条，score
0.4083/0.4000/0.3976/0.3924/0.3823；相关查询 ⇒ 0.5085/0.5125）——调用方无法
用 `len(results)==0` 表达「库里没有相关记忆」。机制：组合分
`0.5*相似 + 0.3*importance + 0.2*近因` 里后两项与查询无关 ⇒ 无关查询有
~0.35-0.41 地板。#816 勘误口径：空库本就返回 `[]`（非「满库空库不可区分」）。

修法（issue 建议第 1 条，#816 复核认可）：`recall`/`search_content`（引擎面）
加 `min_score: float = 0.0` 形参——低于阈值的结果**不返回**，只截断**返回面**
（检索候选池/终排/加权/记账口径不动）；默认 0.0（及一切 <=0）完全兼容。
推荐阈值不硬编码入库（#816：需用标注语料标定；本守卫的 0.45/0.2 只是
「无关地板与相关带之间」的标定示例，取自上面改前读数）。

断言组（把 core.py 里两处 `if min_score > 0.0:` 过滤抽掉即红）：
  J1 无关查询在阈值（0.45）下返回 0 条——不再满额。
  J2 相关查询同阈值不受影响（0.5085/0.5125 均存活）。
  J3 默认兼容：不传 / 0.0 / 负数 ⇒ 与改前读数同构（满额、分数带一致）。
  J4 阈值边界（带容差，避开跨调用毫秒级近因漂移）：阈值贴着次名分数下沿 ⇒
     恰存前 2 名；抬过次名 ⇒ 只剩第 1 名；返回集自洽（每条 score >= 阈值）。
  J5 `search_content` 同参数面：无关相似带 0.06-0.12 在 0.2 下归零，相关 0.3171 存活。
  J6 空库 + 阈值仍 `[]`（钉 #816 勘误：空库语义本就存在，本补丁不破坏）。

运行（仓根）：python -X utf8 -m pytest tests/test_issue795_recall_min_score.py -q --no-header
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

ARTICLES = ['Python programming', 'machine learning', 'cooking recipes',
            'quantum physics', 'gardening tips']
UNRELATED = 'financial derivatives pricing model'
#: 改前读数标定：无关 recall 地板带 0.38-0.41，相关带 0.51 —— 取中点带
UNRELATED_RECALL_FLOOR = 0.41   # 无关查询分数上界（含裕量）
RELATED_RECALL_MIN = 0.46       # 相关查询分数下界（含裕量）


def _engine_with_articles():
    e = SpacetimeMemoryEngine(":memory:")
    for t in ARTICLES:
        e.add_perception('Article about %s with detailed content' % t)
    return e


def test_j1_unrelated_query_thresholded_to_zero():
    """J1：无关查询 + min_score=0.45 ⇒ 0 条（改前为满额 5 条 ~0.40）。"""
    e = _engine_with_articles()
    r = e.recall(UNRELATED, limit=5, min_score=0.45)
    assert r == [], (
        "无关查询在 min_score=0.45 下应返回 0 条（#795），实得 %d 条：%r"
        % (len(r), [(n.content[:30], round(s, 4)) for n, s in r]))


def test_j2_related_query_survives_same_threshold():
    """J2：相关查询同阈值不受影响——cooking/quantum 照常召回。"""
    e = _engine_with_articles()
    for q in ('cooking recipes', 'quantum physics'):
        r = e.recall(q, limit=5, min_score=0.45)
        assert len(r) >= 1, "相关查询 %r 在 min_score=0.45 下被误杀（#795）" % q
        assert any(q in n.content for n, _ in r), (
            "召回的不是相关文章：%r" % [n.content[:30] for n, _ in r])
        for _, s in r:
            assert s >= 0.45, "返回了低于阈值的结果：%.4f" % s


def test_j3_default_behavior_unchanged():
    """J3：默认兼容——不传 / 0.0 / 负数与改前同构（满额 + 分数带）。"""
    e = _engine_with_articles()
    base = e.recall(UNRELATED, limit=5)
    assert len(base) == 5, "默认（无 min_score）应返回满额 5 条，实得 %d" % len(base)
    scores = [s for _, s in base]
    assert all(UNRELATED_RECALL_FLOOR + 0.005 > s > 0.30 for s in scores), (
        "无关查询默认分数带漂移（改前 0.38-0.41）：%r" % [round(s, 4) for s in scores])
    # 显式 0.0 / 负数 = 关闭过滤（同一引擎连查，跨调用漂移 < 1e-5）
    for ms in (0.0, -1.0):
        r = e.recall(UNRELATED, limit=5, min_score=ms)
        assert [n.id for n, _ in r] == [n.id for n, _ in base], (
            "min_score=%r 应与默认逐节点同构（#795 兼容性）" % ms)
    sc = e.search_content(UNRELATED, limit=5)
    assert len(sc) == 5, "search_content 默认应返回满额 5 条，实得 %d" % len(sc)


def test_j4_threshold_boundary_with_tolerance():
    """J4：阈值贴着次名 ⇒ 恰存前 2；抬过次名 ⇒ 只剩第 1；返回集自洽。

    边界用 ±容差（1e-4 / 1e-3）而非精确相等——同引擎两次 recall 之间近因项有
    毫秒级漂移（<1e-5），而相邻名次分差 ~8e-3，容差取两者之间。
    """
    e = _engine_with_articles()
    base = e.recall(UNRELATED, limit=5)
    assert len(base) >= 3
    second = base[1][1]
    assert base[0][1] - second > 5e-3, "构造前提不成立：前两名分差过小（%r）" % (
        [round(s, 5) for _, s in base[:3]])

    r2 = e.recall(UNRELATED, limit=5, min_score=second - 1e-4)
    assert [n.id for n, _ in r2] == [base[0][0].id, base[1][0].id], (
        "阈值=次名-1e-4 应恰存前两名（#795 边界），实得 %d 条" % len(r2))

    r1 = e.recall(UNRELATED, limit=5, min_score=second + 1e-3)
    assert [n.id for n, _ in r1] == [base[0][0].id], (
        "阈值=次名+1e-3 应只剩第 1 名（#795 边界），实得 %d 条" % len(r1))

    for r, ms in ((r2, second - 1e-4), (r1, second + 1e-3)):
        assert all(s >= ms for _, s in r), "返回集里混入低于阈值的结果（#795）"


def test_j5_search_content_same_parameter_face():
    """J5：search_content 同参数面——无关相似带 0.06-0.12 在 0.2 下归零。"""
    e = _engine_with_articles()
    sc_all = e.search_content(UNRELATED, limit=5)
    assert len(sc_all) == 5
    assert max(s for _, s in sc_all) < 0.2, (
        "构造前提不成立：无关相似带越过 0.2（%r）" % [round(s, 4) for _, s in sc_all])
    r = e.search_content(UNRELATED, limit=5, min_score=0.2)
    assert r == [], (
        "search_content 无关查询在 min_score=0.2 下应 0 条，实得 %d" % len(r))
    hit = e.search_content('cooking recipes', limit=5, min_score=0.2)
    assert len(hit) == 1 and 'cooking recipes' in hit[0][0].content, (
        "search_content 相关查询在 min_score=0.2 下应存活 1 条，实得 %r"
        % [n.content[:30] for n, _ in hit])


def test_j6_empty_library_still_empty_with_threshold():
    """J6：空库 + 阈值仍 []（#816 勘误：空库语义本就有，本补丁不破坏）。"""
    e = SpacetimeMemoryEngine(":memory:")
    assert e.recall('anything', limit=5, min_score=0.45) == []
    assert e.search_content('anything', limit=5, min_score=0.45) == []
