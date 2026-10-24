# -*- coding: utf-8 -*-
"""#54 守卫：consume_events 在 paused 期不得丢弃事件。

判据（paused 语义 = 暂停处理，不是丢弃数据）：
  ① paused 期消费后队列长度不变（事件保留）；
  ② 解除 paused 后这些事件可被取走处理（不丢失）；
  ③ paused 回执带 pending 计数（队列状态可观测）。

判别力：只回滚源码（core.py）而不动本文件 ⇒ 必须变红。
运行: python -X utf8 -m pytest tests/test_issue54_consume_events_paused.py -v
"""
import os as _os
import sys as _sys

_RP = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _RP)

from lingshu.core.core import SpacetimeMemoryEngine


class _StubLifecycle:
    """只提供 consume_events 读取的 state 字段。

    为什么用 stub：LifecycleEngine 走裸名导入（from lifecycle_engine import ...），
    本机解析不到 ⇒ engine._lifecycle 恒为 None，真正的 paused 分支不可达
    （已在 50ee03b 实测：_lifecycle is None / _lifecycle_error = "No module named
    'lifecycle_engine'"）。stub 只补 state 字段，不改变被测逻辑。
    """

    def __init__(self, state):
        self.state = state


def _engine(state):
    eng = SpacetimeMemoryEngine(db_path=":memory:")
    eng._lifecycle = _StubLifecycle(state)
    return eng


def test_paused_consume_keeps_pending_events():
    """① paused 期消费不得清空队列。"""
    eng = _engine("paused")
    for t in ("perception", "write", "condition_switch"):
        eng.notify_event(t, {"k": t})
    before = len(eng._event_queue)
    r = eng.consume_events()
    after = len(eng._event_queue)
    assert r["status"] == "paused"
    assert r["consumed"] == 0
    assert after == before == 3, (
        "paused 期消费把队列从 %d 清成 %d ⇒ 事件被静默丢弃" % (before, after))


def test_events_survive_pause_and_are_consumed_after_resume():
    """② 解除 paused 后事件仍可被取走处理（端到端不丢失）。"""
    eng = _engine("paused")
    for t in ("perception", "write", "condition_switch"):
        eng.notify_event(t, {"k": t})
    eng.consume_events()          # paused 期消费：不应吞掉事件
    survived = len(eng._event_queue)
    assert survived == 3, (
        "paused 期消费后只剩 %d 条（投递 3 条）⇒ 恢复后无法再消费" % survived)
    eng._lifecycle = _StubLifecycle("running")
    r2 = eng.consume_events()
    assert r2["status"] == "ok"
    assert len(eng._event_queue) == 0, "恢复后事件仍未被取走处理"


def test_paused_receipt_is_observable():
    """③ paused 回执必须带 pending 计数（否则队列状态不可观测）。"""
    eng = _engine("paused")
    eng.notify_event("write", {})
    r = eng.consume_events()
    assert "pending" in r, "paused 回执缺 pending 字段 ⇒ 队列状态不可观测: %r" % (r,)
    assert r["pending"] == 1


def test_non_paused_path_unchanged():
    """回归：非 paused 路径行为不变（消费后清空队列）。"""
    eng = _engine("running")
    eng.notify_event("write", {})
    r = eng.consume_events()
    assert r["status"] == "ok"
    assert len(eng._event_queue) == 0