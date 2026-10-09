# -*- coding: utf-8 -*-
"""search_content 相关度区分度回归。

评分分母必须是并集（Jaccard 相似度），而不是查询的 bigram 数（覆盖率）：后者只要文档含
查询的每个 bigram 就恒为 1.0，与文档长度 / 其余内容无关。断言只比序关系，不锁定分数值。

运行: python -X utf8 -m pytest tests/test_core_search_discrimination.py -v
"""
import os as _os, sys as _sys, tempfile

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

_Q = "超时"
# 互不相同的汉字填充：bigram 不重复 => 文档变长必然稀释与查询的重叠率
_FILL = "".join(chr(0x4E00 + i) for i in range(1, 400))


def _search(tag, docs):
    """隔离库中写入 (内容, importance) 列表，按查询词检索一次。

    返回 {内容: 分数}；按名次顺序插入，故首个键即榜首。
    """
    e = SpacetimeMemoryEngine(
        _os.path.join(tempfile.mkdtemp(prefix="lingshu_search_disc_"), tag + ".db"))
    for content, importance in docs:
        e.add_perception(content, importance=importance, skip_dedup=True)
    return {n.content: s for n, s in e.search_content(_Q, limit=50)}


def test_long_document_does_not_tie_with_short():
    """同含查询词、长度递增的文档：分数须随长度下降，不得并列同分。"""
    docs = [_Q + _FILL[:n] for n in (20, 80, 200, 380)]
    s = _search("spread", [(d, 0.5) for d in docs])
    scores = [s[d] for d in docs]
    assert len(set(scores)) > 1, "分数饱和，取值集合仅 %r" % (set(scores),)
    assert scores == sorted(scores, reverse=True), "更长文档分数未下降"


def test_irrelevant_content_lowers_score_and_ranking():
    """查询词 + 大量无关内容：分数低于纯查询词文档，且不得排在它前面。"""
    pure, diluted = _Q, _Q + _FILL * 3
    s = _search("dilute", [(pure, 0.5), (diluted, 0.5)])
    assert s[diluted] < s[pure], "无关内容未稀释相关度"
    assert next(iter(s)) == pure, "纯查询词文档未列首位，排序未体现相关度"


def test_document_without_query_scores_zero():
    """不含查询词的文档得 0，不因预筛 / 回退路径白拿分。"""
    # 两条都不含查询词字面 => LIKE 预筛落空走全表回退；
    # 其中一条折叠空白后与查询共享 bigram，另一条与查询零重叠。
    related, unrelated = "网络超 时间 指标", "天气与食物记录"
    s = _search("zero", [(related, 0.5), (unrelated, 0.5)])
    assert s[related] > 0.0
    assert s[unrelated] == 0.0, "无关文档拿到非零分"
