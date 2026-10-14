# -*- coding: utf-8 -*-
"""D-003 终裁步身份互斥守卫（对应 issue #565，判据来源 C-4 / D-02）。

#118 的修复建议把「终裁人是否允许与前序角色重合」显式标为**由维护者定口径**的
悬置项；同仓 A-2 流水线（`adjudicate_verifier_standard`）已按该口径落实，本件把
D-003 流水线对齐到同一规则。
"""
import os
import shutil
import tempfile

import pytest

from lingshu.core.core import SpacetimeMemoryEngine  # noqa: E402

KEY = "k-demo-for-565"


@pytest.fixture()
def engine():
    os.environ["AEIS_DESIGNER_KEY"] = KEY
    d = tempfile.mkdtemp()
    e = SpacetimeMemoryEngine(os.path.join(d, "g.db"))
    yield e
    shutil.rmtree(d, ignore_errors=True)


def _row(engine, pid):
    return tuple(engine.store.conn.execute(
        "select requester, verified_by, adjudicated_by, status "
        "from promotion_proposals where id=?", (pid,)).fetchone())


def _proposal(engine, requester="A", verifier="B"):
    n = engine.add_perception("待晋升内容 " + requester + verifier)
    pid = engine.propose_promotion(n.id, requester, "r")
    engine.verify_promotion(pid, verifier)
    return n, pid


def test_adjudicate_promotion_rejects_requester(engine):
    """提议人自己终裁 ⇒ 拒（False），节点层不变。

    返回契约：engine 门面的 `adjudicate_promotion` 返回 **bool**
    （`return node_id is not None`），被拒即 False；store 版返回 Optional[str]。
    本守卫锁引擎门面的对外契约。
    """
    n, pid = _proposal(engine, requester="bot", verifier="carol")
    assert engine.adjudicate_promotion(pid, "bot", True, designer_key=KEY) is False
    assert engine.store.get_node(n.id).layer.value != "structure"
    # 提案行不得被改写（不得记上approved）
    assert _row(engine, pid)[3] == "verified", _row(engine, pid)


def test_adjudicate_promotion_rejects_verifier(engine):
    """复核人自己终裁 ⇒ 拒。"""
    n, pid = _proposal(engine, requester="alice", verifier="bob")
    assert engine.adjudicate_promotion(pid, "bob", True, designer_key=KEY) is False
    assert engine.store.get_node(n.id).layer.value != "structure"


def test_adjudicate_promotion_allows_distinct_third_party(engine):
    """防过度收紧：第三方身份（C）终裁必须仍能通过。"""
    n, pid = _proposal(engine, requester="alice", verifier="bob")
    assert engine.adjudicate_promotion(pid, "carol", True, designer_key=KEY) is True
    assert engine.store.get_node(n.id).layer.value == "structure"
    assert _row(engine, pid) == ("alice", "bob", "carol", "approved")


def test_both_pipelines_share_the_same_rule(engine):
    """两条流水线对「终裁人 == 前序角色」的拒绝行为应一致。

    对外契约也一致：engine 门面两者都返回 bool（拒 ⇒ False）。
    """
    # D-003 层晋升
    n, pid = _proposal(engine, requester="bot", verifier="carol")
    d003 = engine.adjudicate_promotion(pid, "bot", True, designer_key=KEY)
    # A-2 验证标准
    vid = engine.propose_verifier_standard("VS", "dedup_static", 0.9, "r", "bot")
    engine.review_verifier_standard(vid, "carol", True)
    engine.cs_review_verifier_standard(vid, "dave", True)
    a2 = engine.adjudicate_verifier_standard(vid, "bot", True, designer_key=KEY)
    assert d003 is False and a2 is False, (d003, a2)
    # 且两者在「合法第三方终裁」时都应放行
    _, pid2 = _proposal(engine, requester="alice", verifier="bob")
    ok_d003 = engine.adjudicate_promotion(pid2, "carol", True, designer_key=KEY)
    vid2 = engine.propose_verifier_standard("VS2", "dedup_static", 0.9, "r", "alice")
    engine.review_verifier_standard(vid2, "bob", True)
    engine.cs_review_verifier_standard(vid2, "dave", True)
    ok_a2 = engine.adjudicate_verifier_standard(vid2, "eve", True, designer_key=KEY)
    assert ok_d003 is True and ok_a2 is True, (ok_d003, ok_a2)


def test_designer_key_still_required_first(engine):
    """密钥闸优先：密钥不对时抛 PermissionError（身份检查不得掩盖它）。"""
    _, pid = _proposal(engine)
    with pytest.raises(PermissionError):
        engine.adjudicate_promotion(pid, "carol", True, designer_key="wrong-key")


def test_rejection_does_not_write_structure_layer(engine):
    """被拒时结构层写入不得发生（不可逆副作用不得漏出）。"""
    n, pid = _proposal(engine, requester="bot", verifier="carol")
    engine.adjudicate_promotion(pid, "bot", True, designer_key=KEY)
    node = engine.store.get_node(n.id)
    assert node.layer.value != "structure"
    assert node.confidence < 1.0, node.confidence


def test_guard_is_discriminating(engine):
    """判别力锚定：修前无身份检查 ⇒ 提议人自终裁能通过。"""
    n, pid = _proposal(engine, requester="bot", verifier="carol")
    # 修前语义：无互斥检查，只看密钥/状态/存续/哈希 ⇒ 通过
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "lingshu", "core", "core.py"), encoding="utf-8").read()
    seg = src[src.find("def adjudicate_promotion"):
              src.find("def adjudicate_promotion") + 3000]
    assert "#565：身份互斥" in seg, "本守卫对应的 #565 身份互斥判据缺失"
    # 现实现：被拒
    assert engine.adjudicate_promotion(pid, "bot", True, designer_key=KEY) is False