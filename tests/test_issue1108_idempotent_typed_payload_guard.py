# -*- coding: utf-8 -*-
"""#1108：幂等比较用 Python 真值相等（`True==1`、`1==1.0`）把**不同 JSON 正文**
吞成重放的收口守卫。

契约（docs/structure-events.md:108）：相同 ID、相同 kind/origin/**payload** 返回
原记录；**正文变化须追加纠正事件**（`Conflict`）。payload 判等按**规范序列化后的
字节**（键序归一、JSON 类型敏感），非 Python 真值相等。
不改重放语义本身（同 ID 同内容返回原记录是设计）。
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lingshu.core.core import LayeredStore, Role, SpacetimeMemoryEngine  # noqa: E402
from lingshu.core.structure_writer import (ActorContext, Conflict,  # noqa: E402
                                           EventInput, ProposalInput,
                                           StructureWriter)


def host(path):
    # 内联构造（依赖闭包守卫：不跨测试文件 import test_structure_writer）
    engine = SpacetimeMemoryEngine.__new__(SpacetimeMemoryEngine)
    engine.role = Role.PRIMARY
    engine.store = LayeredStore(str(path), role=Role.PRIMARY)
    return engine


def actor(subject, *capabilities):
    return ActorContext(subject, "lab", subject, frozenset(capabilities))


@pytest.fixture
def writer(tmp_path):
    engine = host(str(tmp_path / "s1108.sqlite3"))
    w = StructureWriter(engine)
    a = actor("ex", "events.write", "structure.propose", "structure.read")
    try:
        yield w, a
    finally:
        try:
            engine.close()
        except Exception:
            pass    # 测试骨架实例（__new__ 构造）缺完整生命周期属性，close 面不参与判据


def test_record_event_true_vs_one_conflicts(writer):
    w, a = writer
    w.record_event(a, EventInput("e1", "k", {"flag": True}))
    with pytest.raises(Conflict, match="different content"):
        w.record_event(a, EventInput("e1", "k", {"flag": 1}))     # 改前：静默重放


def test_record_event_one_vs_one_point_zero_conflicts(writer):
    w, a = writer
    w.record_event(a, EventInput("e2", "k", {"v": 1}))
    with pytest.raises(Conflict, match="different content"):
        w.record_event(a, EventInput("e2", "k", {"v": 1.0}))


def test_record_event_same_content_replays_regression(writer):
    # 回归锚：真重放（同 ID 同内容）语义保持——键序差异不算正文变化。
    w, a = writer
    w.record_event(a, EventInput("e3", "k", {"a": 1, "b": 2}))
    r = w.record_event(a, EventInput("e3", "k", {"b": 2, "a": 1}))
    rep = r.get("replayed") if isinstance(r, dict) else getattr(r, "replayed")
    assert rep is True


def test_propose_true_vs_one_conflicts(writer):
    w, a = writer
    w.propose(a, ProposalInput("q1", "a", "rule", {"flag": True}, 0))
    with pytest.raises(Conflict, match="another proposal"):
        w.propose(a, ProposalInput("q1", "a", "rule", {"flag": 1}, 0))


def test_propose_same_content_replays_regression(writer):
    w, a = writer
    p1 = w.propose(a, ProposalInput("q2", "a", "rule", {"flag": True}, 0))
    p2 = w.propose(a, ProposalInput("q2", "a", "rule", {"flag": True}, 0))
    pid1 = p1.get("proposal_id") if isinstance(p1, dict) else p1.proposal_id
    pid2 = p2.get("proposal_id") if isinstance(p2, dict) else p2.proposal_id
    assert pid1 == pid2
    rep = p2.get("replayed") if isinstance(p2, dict) else p2.replayed
    assert rep is True
