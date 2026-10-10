# -*- coding: utf-8 -*-
"""test_retrieval_no_touch · 检索不刷新近因（LINGSHU_RETRIEVAL_NO_TOUCH=1）守卫

判据：
  · 默认（开关关）：search_content 命中刷新 last_access（与改前一致）；
  · 开关开：search_content / recall 命中只计 access_count，last_access 不变；
  · 开关开：真实强化事件（置信度复核 update_node_confidence、M5 重复观测）照常刷新 last_access；
  · 自增强回路被切断：一条被反复检索的旧记忆，开关开时在 recall 中不会因「被搜过」
    压过同样相关、写入更新的记忆。

变异哨兵：把 _retrieval_touches_recency 改成恒 True，test_no_touch_keeps_last_access
与 test_no_self_reinforcement_in_recall 必红。
"""
import time

import pytest

from lingshu.core.core import SpacetimeMemoryEngine

pytestmark = pytest.mark.filterwarnings("ignore")

DAY = 86400.0


def _engine(tmp_path):
    # 需文件库：increment_access 以独立连接写库（:memory: 下该写入落在另一个空库，不生效）
    return SpacetimeMemoryEngine(str(tmp_path / "m.db"))


def _age(e, nid, days):
    t = time.time() - days * DAY
    e.store.conn.execute("UPDATE nodes SET created_at=?, last_access=? WHERE id=?", (t, t, nid))
    e.store.conn.commit()
    return t


def _la(e, nid):
    return e.store.conn.execute("SELECT last_access, access_count FROM nodes WHERE id=?",
                                (nid,)).fetchone()


def test_default_touches(tmp_path, monkeypatch):
    monkeypatch.delenv("LINGSHU_RETRIEVAL_NO_TOUCH", raising=False)
    e = _engine(tmp_path)
    n = e.add_perception("陈默站在巨子塔下", skip_dedup=True)
    t0 = _age(e, n.id, 30)
    e.store.search_content("巨子塔", limit=3)
    la, cnt = _la(e, n.id)
    assert la > t0 + 29 * DAY and cnt == 1


def test_no_touch_keeps_last_access(tmp_path, monkeypatch):
    monkeypatch.setenv("LINGSHU_RETRIEVAL_NO_TOUCH", "1")
    e = _engine(tmp_path)
    n = e.add_perception("陈默站在巨子塔下", skip_dedup=True)
    t0 = _age(e, n.id, 30)
    e.store.search_content("巨子塔", limit=3)
    e.recall("巨子塔", limit=3)
    la, cnt = _la(e, n.id)
    assert la == pytest.approx(t0) and cnt == 2


def test_real_reinforcement_still_touches(tmp_path, monkeypatch):
    monkeypatch.setenv("LINGSHU_RETRIEVAL_NO_TOUCH", "1")
    e = _engine(tmp_path)
    n = e.add_perception("陈默站在巨子塔下", skip_dedup=True)
    t0 = _age(e, n.id, 30)
    e.store.update_node_confidence(n.id, 0.05)
    assert _la(e, n.id)[0] > t0 + 29 * DAY
    t1 = _age(e, n.id, 30)
    e.add_perception("陈默站在巨子塔下")          # M5 重复观测
    assert _la(e, n.id)[0] > t1 + 29 * DAY


def test_no_self_reinforcement_in_recall(tmp_path, monkeypatch):
    monkeypatch.setenv("LINGSHU_RETRIEVAL_NO_TOUCH", "1")
    e = _engine(tmp_path)
    old = e.add_perception("巨子塔的守卫换成了机器人", skip_dedup=True)
    new = e.add_perception("巨子塔的守卫换成了人类", skip_dedup=True)
    _age(e, old.id, 200)
    _age(e, new.id, 20)
    for _ in range(5):                               # 反复检索旧记忆
        e.store.search_content("守卫换成了机器人", limit=1)
    top = e.recall("巨子塔的守卫换成了什么", limit=2)
    assert top[0][0].id == new.id
