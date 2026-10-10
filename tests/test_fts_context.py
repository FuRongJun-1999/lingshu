# -*- coding: utf-8 -*-
"""test_fts_context · 检索语境扩展（问句功能词过滤 + 时序邻接平滑）守卫

开关：LINGSHU_FTS=1 且 LINGSHU_FTS_CONTEXT=1；任一未开时行为与 FTS 路径改前一致。

判据：
  · 默认关闭：context_enabled() 为假；只开 LINGSHU_FTS_CONTEXT 不开 LINGSHU_FTS 也为假；
  · 功能词过滤：模板词（「请说明」「这部作品」「哪些」）不再把噪声块顶到证据块前面；
  · 全是功能词的查询不被清空（原样保留）；
  · 时序邻接：命中段落的下一条（与问句零字面重叠）进入 top-k；超过会话窗口的不算；
  · 逐字命中语义不变：存在逐字包含查询的节点时只返回这些，开关开/关结果相同。

变异哨兵：把 NEIGHBOR_GAMMA 改成 0，test_neighbor_enters_topk 必红；
把 QUERY_FUNCTION_GRAMS 改成空集，test_function_grams_do_not_dominate 必红。
"""
import pytest

from lingshu.core import retrieval as R
from lingshu.core.core import SpacetimeMemoryEngine

pytestmark = pytest.mark.filterwarnings("ignore")

NOISE = "请说明这部作品里的人物有哪些立场和作用，以及他的身份是什么样的"


@pytest.fixture
def ctx_on(monkeypatch):
    monkeypatch.setenv("LINGSHU_FTS", "1")
    monkeypatch.setenv("LINGSHU_FTS_CONTEXT", "1")


def _ids(res):
    return [n.id for n, _ in res]


def test_default_off(monkeypatch):
    monkeypatch.delenv("LINGSHU_FTS", raising=False)
    monkeypatch.delenv("LINGSHU_FTS_CONTEXT", raising=False)
    assert not R.context_enabled()
    monkeypatch.setenv("LINGSHU_FTS_CONTEXT", "1")
    assert not R.context_enabled()  # 依赖 FTS 路径
    monkeypatch.setenv("LINGSHU_FTS", "1")
    assert R.context_enabled()


def test_all_function_grams_kept():
    q = "请说明"
    assert R.content_grams([q]) == R.grams(q)


def test_function_grams_do_not_dominate(ctx_on):
    e = SpacetimeMemoryEngine(":memory:")
    for i in range(3):
        e.add_perception(f"{NOISE}（第{i}份阅读题模板）", skip_dedup=True)
    for i in range(50):
        e.add_perception(f"第{i}条天气记录：多云转晴，气温{i}度", skip_dedup=True)
    gold = e.add_perception("老周在码头修了一整夜的渔船，天亮才回家", skip_dedup=True)
    q = "请说明这部作品里老周修渔船的立场和作用"
    top = _ids(e.store.search_content(q, limit=5))
    assert top and top[0] == gold.id


def test_neighbor_enters_topk(ctx_on):
    e = SpacetimeMemoryEngine(":memory:")
    for i in range(40):
        e.add_perception(f"第{i}条北港天气记录：多云转晴，气温{i}度", skip_dedup=True)
    hit = e.add_perception("阿青说她明天要去北港找那台旧机器人", skip_dedup=True)
    nxt = e.add_perception("出发前她把父亲留下的钥匙缝进了外套内衬", skip_dedup=True)
    for i in range(40):
        e.add_perception(f"第{i}条北港交通记录：环线拥堵{i}分钟", skip_dedup=True)
    q = "阿青去北港找旧机器人之前做了什么"
    top = _ids(e.store.search_content(q, limit=5))
    assert hit.id in top and nxt.id in top


def test_neighbor_outside_window_ignored(ctx_on):
    e = SpacetimeMemoryEngine(":memory:")
    hit = e.add_perception("阿青说她明天要去北港找那台旧机器人", skip_dedup=True)
    nxt = e.add_perception("出发前她把父亲留下的钥匙缝进了外套内衬", skip_dedup=True)
    e.store.conn.execute("UPDATE nodes SET temporal_coordinate = temporal_coordinate + ? WHERE id=?",
                         (R.NEIGHBOR_WINDOW_S * 2, nxt.id))
    e.store.conn.commit()
    top = _ids(e.store.search_content("阿青去北港找旧机器人之前做了什么", limit=5))
    assert hit.id in top and nxt.id not in top


def test_verbatim_semantics_unchanged(monkeypatch):
    def run(ctx):
        monkeypatch.setenv("LINGSHU_FTS", "1")
        monkeypatch.setenv("LINGSHU_FTS_CONTEXT", "1" if ctx else "0")
        e = SpacetimeMemoryEngine(":memory:")
        e.add_perception("今天的会议记录：预算通过", skip_dedup=True)
        e.add_perception("会后大家去吃了火锅", skip_dedup=True)
        e.add_perception("预算的细节下周再议", skip_dedup=True)
        return [n.content for n, _ in e.store.search_content("预算", limit=5)]
    assert run(False) == run(True)
    assert all("预算" in c for c in run(True))
