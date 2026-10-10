# -*- coding: utf-8 -*-
"""test_fts_retrieval · 检索候选生成（FTS5 二元组 + BM25）守卫，Refs #35

判据：
  · 默认关闭：不建表、不建触发器、走原 LIKE 路径（与改前一致）；开→关后触发器被拆除；
  · 可达性：库过 500 条后，最后写入的目标对自然语言问句仍排第一（原路径只扫前 500 行）；
  · 同步：add_node / 同 id 覆盖写 / 绕过 add_node 的裸 SQL 改写 / 删除，检索结果即时跟随；
  · 打分契约：逐字重复 ≥0.95；「包含查询的长文」不得 ≥0.95（BM25 分量封顶 0.9）；
  · 层过滤、tags 可检索、超长查询不抛、纯标点查询交回原路径；
  · 文件库重开后索引持久且与节点数一致。

变异哨兵：把 _search_content_fts 改成 `return None`，test_reachability_beyond_500 必红；
把 BM25_CAP 改成 1.0 且 BM25_K 改成 0.01，test_containment_is_not_duplicate 必红。
"""
import os

import pytest

from lingshu.core.core import MemoryLayer, SpacetimeMemoryEngine

pytestmark = pytest.mark.filterwarnings("ignore")

TARGET = "巨子当年下令禁止制造有自我意识的机器人，并把条款写进宪法"
QUESTION = "巨子为什么禁止有自我意识的机器人？"


@pytest.fixture
def fts_on(monkeypatch):
    monkeypatch.setenv("LINGSHU_FTS", "1")


def _fill(e, n=600):
    for i in range(n):
        e.add_perception(f"第{i}条日常记录：今天吃了饭，走了{i}步路", importance=0.5, skip_dedup=True)


def _names(conn):
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE name LIKE 'nodes_fts%'")}


def test_default_off_creates_nothing(monkeypatch):
    monkeypatch.delenv("LINGSHU_FTS", raising=False)
    e = SpacetimeMemoryEngine(":memory:")
    assert e.store._fts is None
    assert _names(e.store.conn) == set()


def test_turning_off_drops_triggers(tmp_path, monkeypatch):
    db = str(tmp_path / "m.db")
    monkeypatch.setenv("LINGSHU_FTS", "1")
    e = SpacetimeMemoryEngine(db)
    e.add_perception(TARGET, skip_dedup=True)
    e.store.close()
    monkeypatch.delenv("LINGSHU_FTS")
    e2 = SpacetimeMemoryEngine(db)
    trig = {r[0] for r in e2.store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'nodes_fts_%'")}
    assert trig == set()
    # 关闭期间的写入不入队；再开启时全量补齐
    e2.add_perception("关闭期间写入的记忆：蜂巢议会改选", skip_dedup=True)
    e2.store.close()
    monkeypatch.setenv("LINGSHU_FTS", "1")
    e3 = SpacetimeMemoryEngine(db)
    hits = e3.store.search_content("蜂巢议会改选", limit=1)
    assert hits and "议会改选" in hits[0][0].content


def test_reachability_beyond_500(fts_on):
    e = SpacetimeMemoryEngine(":memory:")
    _fill(e, 600)
    e.add_perception(TARGET, importance=0.5, skip_dedup=True)
    hits = e.store.search_content(QUESTION, limit=5)
    assert hits and hits[0][0].content == TARGET


def test_sync_insert_replace_rawsql_delete(fts_on):
    e = SpacetimeMemoryEngine(":memory:")
    s = e.store
    n = e.add_perception("陈默站在巨子塔下", skip_dedup=True)
    assert s.search_content("巨子塔", limit=1)[0][0].id == n.id
    # 同 id 覆盖写（INSERT OR REPLACE 换 rowid）
    n.content = "陈默离开了议会大厅"
    s.add_node(n)
    assert not [h for h in s.search_content("巨子塔", limit=3) if h[0].id == n.id]
    assert s.search_content("议会大厅", limit=1)[0][0].id == n.id
    # 绕过 add_node 的裸 SQL
    s.conn.execute("UPDATE nodes SET content=? WHERE id=?", ("备份藏在地下三层", n.id))
    s.conn.commit()
    assert s.search_content("备份藏在哪", limit=1)[0][0].id == n.id
    # 删除
    s.delete_node(n.id)
    assert not [h for h in s.search_content("备份藏在哪", limit=3) if h[0].id == n.id]
    assert s.conn.execute("SELECT COUNT(*) FROM nodes_fts_map").fetchone()[0] == \
        s.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]


def test_exact_duplicate_keeps_threshold(fts_on):
    e = SpacetimeMemoryEngine(":memory:")
    _fill(e, 50)
    e.add_perception(TARGET, skip_dedup=True)
    top = e.store.search_content(TARGET, limit=1)[0]
    assert top[0].content == TARGET and top[1] >= 0.95


def test_containment_is_not_duplicate(fts_on):
    e = SpacetimeMemoryEngine(":memory:")
    _fill(e, 50)  # 需要背景文档，BM25 的 IDF 才有区分度
    long_doc = "会议纪要：" + TARGET + "。随后议会讨论了能源配额、交通改造、教育改革等十余项议题。" * 3
    e.add_perception(long_doc, skip_dedup=True)
    top = e.store.search_content(TARGET, limit=1)[0]
    assert top[0].content == long_doc and top[1] < 0.95


def test_layer_filter_and_tags(fts_on):
    e = SpacetimeMemoryEngine(":memory:")
    a = e.add_perception("巨子塔的地下档案", skip_dedup=True)
    e.store.conn.execute("UPDATE nodes SET layer='context' WHERE id=?", (a.id,))
    e.store.conn.commit()
    b = e.add_perception("巨子塔的顶层观景台", skip_dedup=True)
    hits = e.store.search_content("巨子塔", layers=[MemoryLayer.CONTEXT], limit=5)
    assert [h[0].id for h in hits] == [a.id]
    e.store.tag_node(b.id, "观测站")
    assert e.store.search_content("观测站", limit=1)[0][0].id == b.id


def test_long_and_punct_queries(fts_on):
    e = SpacetimeMemoryEngine(":memory:")
    _fill(e, 20)
    e.store.search_content("巨" * 60000, limit=3)          # 不抛
    assert isinstance(e.store.search_content("？！。", limit=3), list)  # 交回原路径


def test_persist_and_reopen(tmp_path, fts_on):
    db = str(tmp_path / "p.db")
    e = SpacetimeMemoryEngine(db)
    _fill(e, 30)
    e.add_perception(TARGET, skip_dedup=True)
    e.store.close()
    e2 = SpacetimeMemoryEngine(db)
    c = e2.store.conn
    assert c.execute("SELECT COUNT(*) FROM nodes_fts_dirty").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM nodes_fts_map").fetchone()[0] == \
        c.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    assert e2.store.search_content(QUESTION, limit=1)[0][0].content == TARGET


def test_verbatim_hits_rank_first(fts_on):
    """查询原文有逐字命中时，逐字命中排在最前（原 LIKE 主命中语义）；部分重叠的节点只补后位。"""
    e = SpacetimeMemoryEngine(":memory:")
    a = e.add_perception("子线程写入", skip_dedup=True)
    b = e.add_perception("主线程写入日志", skip_dedup=True)
    ids = [h[0].id for h in e.store.search_content("子线程写入", limit=10)]
    assert ids[0] == a.id
    assert e.store.search_content("子线程写入", limit=1)[0][0].id == a.id
    assert set(ids) <= {a.id, b.id}


def test_verbatim_does_not_hide_similar_nodes(fts_on):
    """逐字命中不排他：新快照逐字含查询时，相似旧节点仍须出现在 top-k
    （上游 #44：longterm_snapshot / prefeed 依赖 search_content 建 similar 边）。"""
    e = SpacetimeMemoryEngine(":memory:")
    old = e.add_perception("六边形蜂窝网格等距邻居编码方式的旋转等变上限记录", skip_dedup=True)
    q = "记忆：六边形蜂窝网格等距邻居编码方式的旋转等变上限与对称群阶数"
    new = e.add_perception(q, skip_dedup=True)
    hits = e.store.search_content(q, limit=3)
    assert hits[0][0].id == new.id
    assert old.id in [h[0].id for h in hits]
