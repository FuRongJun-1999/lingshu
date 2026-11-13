# -*- coding: utf-8 -*-
"""#54 守卫：`consume_events` paused 时**不消费也不清空**队列。

缺陷面（公开 issue #54）：旧实现先清空 `_event_queue` 再判 paused ⇒ paused 期间
排队的事件被静默丢弃（清空发生在暂停检查之前，恢复后无从恢复）。

本件锁（暂停检查前置，队列保留）：
  ① paused ⇒ status=paused、队列**保留**（pending 不变）；
  ② 恢复后消费正常（队列里的事件不丢）；
  ③ 非暂停路径行为不变（正常消费后清空）。
`_lifecycle` 为 None（组件未装配）⇒ 无暂停语义，走正常消费（既有口径）。
定点变异：把 paused 早退移回「清空之后」⇒ ① 红。
"""
from lingshu.core.core import SpacetimeMemoryEngine


class _Paused:
    state = "paused"


class _Normal:
    state = "normal"


def _engine_with_events():
    e = SpacetimeMemoryEngine(":memory:")
    e.notify_event("perception")
    e.notify_event("write")
    return e


# ---------------------------------------------------------------- ① paused 保留

def test_paused_preserves_queue():
    e = _engine_with_events()
    e._lifecycle = _Paused()
    r = e.consume_events()
    assert r["status"] == "paused"
    assert r["pending"] == 2, "paused 时队列必须保留：%r" % (r,)
    assert len(e._event_queue) == 2, "paused 时事件不得被清空"
    # 恢复后一次性消费
    e._lifecycle = _Normal()
    r2 = e.consume_events()
    assert r2["status"] == "ok" and len(e._event_queue) == 0


# ---------------------------------------------------------------- ② 恢复后消费

def test_events_survive_pause_then_consumed():
    e = _engine_with_events()
    e._lifecycle = _Paused()
    e.consume_events()
    e._lifecycle.state = "normal"
    r = e.consume_events()
    assert r["status"] == "ok"
    assert len(e._event_queue) == 0


# ---------------------------------------------------------------- ③ 正常路径不变

def test_normal_path_no_lifecycle_unchanged():
    e = _engine_with_events()
    r = e.consume_events()
    assert r["status"] == "ok"
    assert len(e._event_queue) == 0
